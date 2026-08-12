"""Molecule-set (subcellular point-cloud) encoder and hierarchical VQ-VAE.

An alternative input modality for Nicheverse: instead of a per-cell aggregated gene
vector, encode each cell as the *set of its transcript molecules* -- one token per
molecule, carrying the molecule's gene identity and its subcellular offset from the
cell centroid. This keeps the codebook/VQ core of Nicheverse unchanged while letting
the cell branch model spatial molecular structure directly.

Encoder (:class:`MoleculeSetEncoder`) is a permutation-invariant, mask-aware Set
Transformer++ (masked ISAB + PMA with Set Normalization; Zhang, Hare & Prugel-Bennett
2022): each molecule token is ``gene_embedding(gene_id) + posMLP(Fourier(dx, dy))``.
It is deliberately not a mean/max pool and not a serialize-and-patch point transformer,
since a cell holds only a small (<=few hundred) 2D molecule set.

:class:`MoleculeSetVQVAE` mirrors :class:`~nicheverse.models.HierarchicalVQVAE`: a
molecule-set cell branch feeding the cell codebook, an optional aggregated-kNN
neighborhood branch feeding the neighborhood codebook, and cross-attention fusion.
The cell decoder reconstructs the 366-dim log-normalized within-radius gene
composition. It reuses the released :class:`~nicheverse.models.quantizers.VectorQuantizer`
(EMA + k-means++ init + dead-code reset), so codebook behavior matches the rest of the
package.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .encoders import _largest_divisor, _mlp
from .quantizers import VectorQuantizer

__all__ = [
    "FourierPos",
    "SetNorm",
    "MaskedMAB",
    "MaskedISAB",
    "MaskedPMA",
    "MoleculeSetEncoder",
    "MoleculeSetVQVAE",
]


class FourierPos(nn.Module):
    """Fourier features of a 2D micron offset ``(dx, dy)`` with fixed log-spaced
    frequencies (a periodic positional encoding of subcellular position)."""

    def __init__(self, num_freqs: int = 12, max_micron: float = 7.0) -> None:
        super().__init__()
        freqs = 2.0 * math.pi * torch.logspace(0, math.log10(max_micron), num_freqs) / max_micron
        self.register_buffer("freqs", freqs)  # (num_freqs,)
        self.out_dim = 2 * 2 * num_freqs  # sin+cos over x and y

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        proj = xy.unsqueeze(-1) * self.freqs.view(1, 1, 1, -1)  # (B, M, 2, num_freqs)
        emb = torch.cat([proj.sin(), proj.cos()], dim=-1)  # (B, M, 2, 2*num_freqs)
        return emb.reshape(*xy.shape[:2], -1)  # (B, M, 4*num_freqs)


class SetNorm(nn.Module):
    """Set Normalization (Zhang, Hare & Prugel-Bennett 2022): normalize each sample
    jointly over the set (token) and feature axes, ignoring padded slots, then apply a
    per-feature affine. Permutation invariant and mask aware."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, M, D), mask: (B, M) bool (True = real molecule)
        m = mask.unsqueeze(-1).float()  # (B, M, 1)
        cnt = m.sum(dim=(1, 2)).clamp_min(1.0) * x.shape[-1]  # (B,) valid elements
        mean = ((x * m).sum(dim=(1, 2)) / cnt).view(-1, 1, 1)
        var = ((((x - mean) ** 2) * m).sum(dim=(1, 2)) / cnt).view(-1, 1, 1)
        xn = (x - mean) / torch.sqrt(var + self.eps)
        return (xn * self.g + self.b) * m  # zero padded slots


class MaskedMAB(nn.Module):
    """Masked Multihead Attention Block with per-token Layer Normalization (the standard
    Set Transformer choice): query set ``q`` attends over key set ``k`` with key padding
    masked out. Per-token LayerNorm keeps each cell's molecular-composition signal (mean
    and scale) intact; joint set+feature normalization (SetNorm) removes it and collapses
    the pooled cell embedding, so it is not used here. Padded slots are already zeroed by
    the token mask upstream and ignored by ``key_padding_mask``."""

    def __init__(self, dim_q: int, dim_k: int, dim_v: int, num_heads: int) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(dim_v, num_heads, batch_first=True)
        self.fc_q = nn.Linear(dim_q, dim_v)
        self.fc_k = nn.Linear(dim_k, dim_v)
        self.fc_v = nn.Linear(dim_k, dim_v)
        self.n0 = nn.LayerNorm(dim_v)
        self.n1 = nn.LayerNorm(dim_v)
        self.ff = nn.Sequential(nn.Linear(dim_v, dim_v * 2), nn.GELU(), nn.Linear(dim_v * 2, dim_v))

    def forward(self, q, k, q_mask, k_mask):
        qp, kp, vp = self.fc_q(q), self.fc_k(k), self.fc_v(k)
        kpm = ~k_mask
        # guard all-padded key rows (empty cells): open slot 0 so attention stays finite
        empt = kpm.all(dim=1)
        if empt.any():
            kpm = kpm.clone()
            kpm[empt, 0] = False
        attn, _ = self.mha(qp, kp, vp, key_padding_mask=kpm, need_weights=False)
        h = self.n0(qp + attn)
        return self.n1(h + self.ff(h))


class MaskedISAB(nn.Module):
    """Masked Induced Set Attention Block: ``O(n*m)`` set attention via ``m`` learnable
    induced points."""

    def __init__(self, dim_in: int, dim_out: int, num_heads: int, num_inds: int) -> None:
        super().__init__()
        self.inds = nn.Parameter(torch.empty(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.inds)
        self.num_inds = num_inds
        self.mab0 = MaskedMAB(dim_out, dim_in, dim_out, num_heads)  # induced points attend over X
        self.mab1 = MaskedMAB(dim_in, dim_out, dim_out, num_heads)  # X attends over induced points

    def forward(self, x, mask):
        b = x.size(0)
        ind = self.inds.repeat(b, 1, 1)
        ind_mask = torch.ones(b, self.num_inds, dtype=torch.bool, device=x.device)
        h = self.mab0(ind, x, ind_mask, mask)  # (B, m, D)
        return self.mab1(x, h, mask, ind_mask)  # (B, M, D)


class MaskedPMA(nn.Module):
    """Masked Pooling by Multihead Attention: learnable seed queries pool the token set."""

    def __init__(self, dim: int, num_heads: int, num_seeds: int = 1) -> None:
        super().__init__()
        self.seeds = nn.Parameter(torch.empty(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.seeds)
        self.num_seeds = num_seeds
        self.mab = MaskedMAB(dim, dim, dim, num_heads)

    def forward(self, x, mask):
        b = x.size(0)
        s = self.seeds.repeat(b, 1, 1)
        s_mask = torch.ones(b, self.num_seeds, dtype=torch.bool, device=x.device)
        return self.mab(s, x, s_mask, mask)  # (B, num_seeds, D)


class MoleculeSetEncoder(nn.Module):
    """Set Transformer++ over a cell's molecule set -> per-cell latent.

    Each molecule token is ``gene_embedding(gene_id) + posMLP(Fourier(dx, dy))``; masked
    ISAB blocks mix the set; the set is pooled by concatenating masked max, masked mean,
    and a masked PMA seed readout, then projected to ``out_dim``. PMA alone collapses (its
    single seed query converges to a near-constant readout, so the codebook EMA chases it
    into one cluster); the max/mean statistics preserve each cell's molecular-composition
    magnitude and spread, giving the quantizer real structure. For an all-padded (empty)
    cell the masked max and masked mean are already 0, but the PMA seed reads a garbage
    padded token, so it is zeroed for empty rows, giving a fully zero pooled embedding.
    Permutation invariant and mask aware (``True`` = real molecule; ``gene_id == n_genes``
    is the padding token).
    """

    def __init__(
        self,
        out_dim: int = 64,
        width: int = 128,
        n_genes: int = 366,
        num_freqs: int = 12,
        num_heads: int = 4,
        num_inds: int = 16,
        num_isab: int = 2,
        dropout: float = 0.1,
        max_micron: float = 7.0,
    ) -> None:
        super().__init__()
        self.pad_gene = n_genes
        self.gene_embed = nn.Embedding(n_genes + 1, width, padding_idx=n_genes)
        self.pos = FourierPos(num_freqs=num_freqs, max_micron=max_micron)
        self.token_mlp = nn.Sequential(
            nn.Linear(width + self.pos.out_dim, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )
        nh = _largest_divisor(width, num_heads)
        self.isab = nn.ModuleList(MaskedISAB(width, width, nh, num_inds) for _ in range(num_isab))
        self.pma = MaskedPMA(width, nh, num_seeds=1)
        self.out = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(), nn.Linear(width, out_dim))

    def forward(self, gene: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode molecules ``gene:(B,M)`` long, ``coords:(B,M,2)``, ``mask:(B,M)`` bool
        to a latent ``(B, out_dim)``."""
        tok = self.token_mlp(torch.cat([self.gene_embed(gene), self.pos(coords)], dim=-1))
        mfb = mask.unsqueeze(-1).float()
        tok = tok * mfb
        for blk in self.isab:
            tok = blk(tok, mask) * mfb  # re-zero padded rows (per-token LayerNorm is not mask-aware)
        mf = mask.unsqueeze(-1).float()
        mean = (tok * mf).sum(1) / mf.sum(1).clamp_min(1.0)
        neg = tok.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        mx = neg.max(1).values
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))  # empty cells -> 0
        seed = self.pma(tok, mask).squeeze(1)
        # empty (all-padded) cells: PMA reads a garbage padded token, so zero the seed
        seed = torch.where(mask.any(dim=1, keepdim=True), seed, torch.zeros_like(seed))
        return self.out(torch.cat([mx, mean, seed], dim=-1))


class MoleculeSetVQVAE(nn.Module):
    """Hierarchical VQ-VAE with a molecule-set cell branch and an optional aggregated
    kNN neighborhood branch, fused by cross-attention (mirrors
    :class:`~nicheverse.models.HierarchicalVQVAE`). Emits the same cell/neighborhood code
    indices so downstream annotation is unchanged.
    """

    def __init__(
        self,
        n_genes: int = 366,
        cell_embedding_dim: int = 64,
        cell_num_embeddings: int = 256,
        neighborhood_embedding_dim: int = 256,
        neighborhood_num_embeddings: int = 32,
        commitment_cost: float = 0.25,
        hidden=(256, 128),
        use_neighborhood: bool = True,
        use_cross_attention: bool = True,
        cross_attention_weight: float = 0.5,
        enc_width: int = 128,
        enc_isab: int = 2,
        enc_heads: int = 4,
        enc_inds: int = 16,
        enc_freqs: int = 12,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.use_neighborhood = use_neighborhood
        self.use_cross_attention = use_cross_attention and use_neighborhood
        self.cross_attention_weight = cross_attention_weight
        hd = list(hidden)
        self.cell_encoder = MoleculeSetEncoder(
            out_dim=cell_embedding_dim,
            width=enc_width,
            n_genes=n_genes,
            num_freqs=enc_freqs,
            num_heads=enc_heads,
            num_inds=enc_inds,
            num_isab=enc_isab,
        )
        self.cell_vq = VectorQuantizer(cell_num_embeddings, cell_embedding_dim, commitment_cost)
        self.cell_decoder = _mlp(cell_embedding_dim, list(reversed(hd)), n_genes)
        if use_neighborhood:
            self.neighborhood_encoder = _mlp(n_genes * 2, hd, neighborhood_embedding_dim)
            self.neighborhood_vq = VectorQuantizer(
                neighborhood_num_embeddings, neighborhood_embedding_dim, commitment_cost
            )
            self.neighborhood_decoder = _mlp(
                neighborhood_embedding_dim, list(reversed(hd)), n_genes * 2
            )
            if self.use_cross_attention:
                heads = _largest_divisor(cell_embedding_dim, 4)
                self.cross_attention = nn.MultiheadAttention(
                    cell_embedding_dim, heads, dropout=0.1, batch_first=True
                )
                self.neighborhood_projection = nn.Linear(
                    neighborhood_embedding_dim, cell_embedding_dim
                )

    def forward(self, gene, coords, mask, neigh_features=None):
        z_cell = self.cell_encoder(gene, coords, mask).unsqueeze(2)
        cell_vq_loss, q_cell, cell_perp, cell_idx = self.cell_vq(z_cell)
        q_cell = q_cell.squeeze(2)
        neigh_recon = neigh_vq_loss = neigh_idx = neigh_perp = None
        if self.use_neighborhood:
            z_neigh = self.neighborhood_encoder(neigh_features).unsqueeze(2)
            neigh_vq_loss, q_neigh, neigh_perp, neigh_idx = self.neighborhood_vq(z_neigh)
            q_neigh = q_neigh.squeeze(2)
            if self.use_cross_attention:
                proj = self.neighborhood_projection(q_neigh)
                attn, _ = self.cross_attention(
                    q_cell.unsqueeze(1), proj.unsqueeze(1), proj.unsqueeze(1)
                )
                q_cell_final = q_cell + self.cross_attention_weight * attn.squeeze(1)
            else:
                q_cell_final = q_cell
            neigh_recon = self.neighborhood_decoder(q_neigh)
        else:
            q_cell_final = q_cell
        cell_recon = self.cell_decoder(q_cell_final)
        return (
            cell_recon,
            neigh_recon,
            cell_vq_loss,
            neigh_vq_loss,
            cell_idx,
            neigh_idx,
            cell_perp,
            neigh_perp,
        )

    @torch.inference_mode()
    def embed_cell(self, gene, coords, mask):
        return self.cell_encoder(gene, coords, mask)
