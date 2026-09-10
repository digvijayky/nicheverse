"""Multi panel foundation model: one codebook pair shared by every panel and platform.

:class:`~nicheverse.models.HierarchicalVQVAE` assumes a single fixed gene panel, so its
encoders and decoders are dense over ``input_dim``. A model trained jointly on a hundred
panels cannot: each dataset measures a different subset of a shared vocabulary, and no cell
should be asked to explain a gene its panel never assayed.

:class:`FoundationVQVAE` keeps the architecture of the hierarchical model and changes only
what has to change:

* the encoders are :class:`~nicheverse.models.sparse.SparseBagEncoder` instances over the
  shared vocabulary (counts and the transcript context field get separate tables, cells with
  no molecule table get a learned missing context vector);
* the quantizers, the cross attention block and the VQ bookkeeping are the SAME components
  the hierarchical model uses (:func:`~nicheverse.models.build_quantizer`, an
  :class:`torch.nn.MultiheadAttention` residual with the same weight);
* the decoders keep a vocabulary sized weight matrix but are evaluated only at the measured
  gene indices, so the reconstruction target is ``(batch, panel size)`` and never
  ``(batch, vocabulary)``;
* platform and species are conditioned INTO THE DECODERS ONLY (a learned embedding added to
  the decoder input), so the encoder, and therefore the codebook, is pushed to carry biology
  rather than the batch identity of the platform. Switchable with ``condition_decoders``.

Additive module: importing it does not change any existing default.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..losses import masked_bernoulli_detection_bce, masked_nb_nll
from .encoders import _largest_divisor, _mlp, build_encoder
from .quantizers import build_quantizer
from .sparse import SparseBag

__all__ = ["FoundationConfig", "FoundationVQVAE", "load_foundation", "save_foundation"]


@dataclass
class FoundationConfig:
    """Configuration for :class:`FoundationVQVAE`.

    Parameters
    ----------
    vocab_size
        Size of the shared gene vocabulary (see
        :class:`~nicheverse.data.vocabulary.GeneVocabulary`).
    hidden_dims
        Encoder MLP widths; the decoder trunk uses the reversed sequence.
    cell_embedding_dim, cell_num_embeddings
        Cell latent width and codebook size (1024 entries for the foundation run).
    neighborhood_embedding_dim, neighborhood_num_embeddings
        Niche latent width and codebook size (64 entries for the foundation run).
    gene_embed_dim
        Width of the gene embedding tables inside the sparse encoder.
    decoder_hidden
        Width of the decoder trunk output, i.e. of the vocabulary sized head.
    commitment_cost, vq_distance, quantizer_type, quantizer_kwargs
        Passed straight through to :func:`~nicheverse.models.build_quantizer`, exactly as in
        :class:`~nicheverse.models.HierarchicalVQVAE`.
    use_cross_attention, cross_attention_weight, cross_attention_heads
        The same cross attention block as the hierarchical model.
    encoder_type, encoder_kwargs
        Encoder backbone; ``"sparse_bag"`` is the only one that consumes sparse bags.
    use_context
        Allocate the transcript context tables and the missing context vector.
    n_platforms, n_species, n_datasets
        Cardinalities of the conditioning vocabularies.
    condition_decoders
        Add the learned platform and species embeddings to the DECODER input. Set False to
        ablate the conditioning.
    condition_dim
        Width of the platform / species embeddings before they are projected onto the latent.
    detection_weight
        Weight of the masked Bernoulli detection hurdle (0 disables it).
    niche_weight, vq_weight
        Weights of the niche reconstruction and the VQ commitment terms.
    tie_decoder
        Tie the cell decoder head to the count embedding table (halves the head parameters).
    vocabulary
        Optional record of the vocabulary symbols, stored in the checkpoint.
    """

    vocab_size: int
    hidden_dims: tuple[int, ...] = (512, 256)
    cell_embedding_dim: int = 64
    cell_num_embeddings: int = 1024
    neighborhood_embedding_dim: int = 256
    neighborhood_num_embeddings: int = 64
    gene_embed_dim: int = 256
    decoder_hidden: int = 256
    commitment_cost: float = 0.25
    vq_distance: str = "l2"
    quantizer_type: str = "vq"
    quantizer_kwargs: dict[str, Any] = field(default_factory=dict)
    use_cross_attention: bool = True
    cross_attention_weight: float = 0.5
    cross_attention_heads: int = 4
    encoder_type: str = "sparse_bag"
    encoder_kwargs: dict[str, Any] = field(default_factory=dict)
    dropout: float = 0.2
    use_context: bool = True
    n_platforms: int = 1
    n_species: int = 1
    n_datasets: int = 1
    condition_decoders: bool = True
    condition_dim: int = 32
    detection_weight: float = 0.5
    niche_weight: float = 1.0
    vq_weight: float = 1.0
    tie_decoder: bool = False
    vocabulary: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if len(self.hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one layer")
        for name in (
            "cell_embedding_dim",
            "cell_num_embeddings",
            "neighborhood_embedding_dim",
            "neighborhood_num_embeddings",
            "gene_embed_dim",
            "decoder_hidden",
            "condition_dim",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.vq_distance not in ("l2", "cosine"):
            raise ValueError(f"vq_distance must be 'l2' or 'cosine', got {self.vq_distance!r}")
        if self.detection_weight < 0:
            raise ValueError(f"detection_weight must be >= 0, got {self.detection_weight}")
        if self.vocabulary and len(self.vocabulary) != self.vocab_size:
            raise ValueError(
                f"len(vocabulary)={len(self.vocabulary)} != vocab_size={self.vocab_size}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON safe dict."""
        d = asdict(self)
        d["hidden_dims"] = list(self.hidden_dims)
        d["vocabulary"] = list(self.vocabulary)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FoundationConfig:
        """Rebuild from :meth:`to_dict` output, ignoring unknown keys."""
        d = dict(d)
        d["hidden_dims"] = tuple(int(x) for x in d.get("hidden_dims", (512, 256)))
        d["vocabulary"] = tuple(str(x) for x in d.get("vocabulary", ()))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


class _GeneHead(nn.Module):
    """Vocabulary sized linear head evaluated only at the measured gene indices."""

    def __init__(self, hidden: int, vocab_size: int, tied: nn.Parameter | None = None) -> None:
        super().__init__()
        self.tied = tied is not None
        if tied is None:
            self.weight = nn.Parameter(torch.empty(vocab_size, hidden))
            nn.init.normal_(self.weight, std=0.02)
        else:
            self.weight = tied
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, h: torch.Tensor, measured: torch.Tensor) -> torch.Tensor:
        """Return ``(B, len(measured))`` logits."""
        return h @ self.weight.index_select(0, measured).t() + self.bias.index_select(0, measured)


class FoundationVQVAE(nn.Module):
    """Two codebook VQ-VAE over a shared gene vocabulary and many panels.

    Parameters
    ----------
    config
        :class:`FoundationConfig`.
    """

    def __init__(self, config: FoundationConfig) -> None:
        super().__init__()
        self.config = config
        hd = list(config.hidden_dims)
        enc_kw = dict(
            vocab_size=config.vocab_size,
            embed_dim=config.gene_embed_dim,
            use_context=config.use_context,
        )
        enc_kw.update(config.encoder_kwargs)
        self.cell_encoder = build_encoder(
            config.encoder_type,
            in_dim=config.vocab_size,
            out_dim=config.cell_embedding_dim,
            hidden=hd,
            dropout=config.dropout,
            n_bags=1,
            **enc_kw,
        )
        self.neighborhood_encoder = build_encoder(
            config.encoder_type,
            in_dim=config.vocab_size * 2,
            out_dim=config.neighborhood_embedding_dim,
            hidden=hd,
            dropout=config.dropout,
            n_bags=2,
            **enc_kw,
        )
        self.cell_vq = build_quantizer(
            config.quantizer_type,
            num_embeddings=config.cell_num_embeddings,
            embedding_dim=config.cell_embedding_dim,
            commitment_cost=config.commitment_cost,
            distance_metric=config.vq_distance,
            **config.quantizer_kwargs,
        )
        self.neighborhood_vq = build_quantizer(
            config.quantizer_type,
            num_embeddings=config.neighborhood_num_embeddings,
            embedding_dim=config.neighborhood_embedding_dim,
            commitment_cost=config.commitment_cost,
            distance_metric=config.vq_distance,
            **config.quantizer_kwargs,
        )
        self.use_cross_attention = config.use_cross_attention
        self.cross_attention_weight = config.cross_attention_weight
        if self.use_cross_attention:
            heads = _largest_divisor(config.cell_embedding_dim, config.cross_attention_heads)
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=config.cell_embedding_dim, num_heads=heads, dropout=0.1, batch_first=True
            )
            self.neighborhood_projection = nn.Linear(
                config.neighborhood_embedding_dim, config.cell_embedding_dim
            )
        rev = list(reversed(hd))
        self.cell_trunk = _mlp(
            config.cell_embedding_dim, rev, config.decoder_hidden, config.dropout
        )
        self.niche_trunk = _mlp(
            config.neighborhood_embedding_dim, rev, config.decoder_hidden, config.dropout
        )
        tied = None
        if config.tie_decoder and config.gene_embed_dim == config.decoder_hidden:
            tied = self.cell_encoder.count_embed.weight
        self.cell_head = _GeneHead(config.decoder_hidden, config.vocab_size, tied)
        self.niche_self_head = _GeneHead(config.decoder_hidden, config.vocab_size)
        self.niche_nbr_head = _GeneHead(config.decoder_hidden, config.vocab_size)
        self.cell_log_theta = nn.Parameter(torch.zeros(config.vocab_size))
        self.niche_log_theta = nn.Parameter(torch.zeros(config.vocab_size))
        if config.condition_decoders:
            self.platform_embed = nn.Embedding(config.n_platforms, config.condition_dim)
            self.species_embed = nn.Embedding(config.n_species, config.condition_dim)
            nn.init.zeros_(self.platform_embed.weight)
            nn.init.zeros_(self.species_embed.weight)
            self.cond_to_cell = nn.Linear(config.condition_dim, config.cell_embedding_dim)
            self.cond_to_niche = nn.Linear(config.condition_dim, config.neighborhood_embedding_dim)

    # -- helpers -------------------------------------------------------------------------
    def _condition(self, platform_id: torch.Tensor | None, species_id: torch.Tensor | None):
        """Learned platform + species vector added to the decoder inputs (never the encoder)."""
        if not self.config.condition_decoders or platform_id is None:
            return None, None
        c = self.platform_embed(platform_id.reshape(-1)) + self.species_embed(species_id.reshape(-1))
        return self.cond_to_cell(c), self.cond_to_niche(c)

    # -- forward -------------------------------------------------------------------------
    def forward(
        self,
        cell_bag: SparseBag,
        nbr_bag: SparseBag,
        measured: torch.Tensor,
        cell_context: SparseBag | None = None,
        nbr_context: SparseBag | None = None,
        has_context: torch.Tensor | None = None,
        platform_id: torch.Tensor | None = None,
        species_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode, quantize and decode one minibatch (all cells from one dataset).

        Parameters
        ----------
        cell_bag
            RAW counts of each cell, as a :class:`~nicheverse.models.sparse.SparseBag` over
            the shared vocabulary.
        nbr_bag
            The aggregated neighborhood counts of each cell, same form.
        measured
            ``(M,)`` int64 vocabulary indices this panel measures.
        cell_context, nbr_context
            Optional transcript context bags for the cell and its neighborhood.
        has_context
            ``(B,)`` 0/1 flag; rows with 0 get the learned missing context vector.
        platform_id, species_id
            ``(B,)`` int64 conditioning ids (used by the decoders only).

        Returns
        -------
        dict
            ``cell_logits`` and ``niche_self_logits`` / ``niche_nbr_logits`` (all
            ``(B, M)``), ``cell_idx`` / ``niche_idx``, ``cell_vq_loss`` / ``niche_vq_loss``,
            ``cell_perplexity`` / ``niche_perplexity``, and the pre quantization latents
            ``z_cell`` / ``z_niche``.
        """
        z_cell = self.cell_encoder([cell_bag], [cell_context], has_context)
        z_niche = self.neighborhood_encoder(
            [cell_bag, nbr_bag], [cell_context, nbr_context], has_context
        )
        # Quantize in fp32 even under autocast: codebook distances, the EMA buffers and the
        # dead code reseed are all fp32 state, and bf16 distances would blur near ties.
        with torch.autocast(device_type=z_cell.device.type, enabled=False):
            cell_vq_loss, q_cell, cell_perp, cell_idx = self.cell_vq(z_cell.float().unsqueeze(2))
            q_cell = q_cell.squeeze(2)
            niche_vq_loss, q_niche, niche_perp, niche_idx = self.neighborhood_vq(
                z_niche.float().unsqueeze(2)
            )
            q_niche = q_niche.squeeze(2)
        if self.use_cross_attention:
            proj = self.neighborhood_projection(q_niche)
            attn, _ = self.cross_attention(
                q_cell.unsqueeze(1), proj.unsqueeze(1), proj.unsqueeze(1)
            )
            q_cell_final = q_cell + self.cross_attention_weight * attn.squeeze(1)
        else:
            q_cell_final = q_cell
        c_cell, c_niche = self._condition(platform_id, species_id)
        if c_cell is not None:
            q_cell_final = q_cell_final + c_cell
            q_niche = q_niche + c_niche
        h_cell = self.cell_trunk(q_cell_final)
        h_niche = self.niche_trunk(q_niche)
        return dict(
            cell_logits=self.cell_head(h_cell, measured),
            niche_self_logits=self.niche_self_head(h_niche, measured),
            niche_nbr_logits=self.niche_nbr_head(h_niche, measured),
            cell_idx=cell_idx.reshape(-1),
            niche_idx=niche_idx.reshape(-1),
            cell_vq_loss=cell_vq_loss,
            niche_vq_loss=niche_vq_loss,
            cell_perplexity=cell_perp,
            niche_perplexity=niche_perp,
            z_cell=z_cell,
            z_niche=z_niche,
        )

    def compute_loss(
        self,
        out: dict[str, torch.Tensor],
        cell_target: torch.Tensor,
        nbr_target: torch.Tensor,
        measured: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Masked negative binomial reconstruction plus the detection hurdle and VQ terms.

        Parameters
        ----------
        out
            The dict returned by :meth:`forward`.
        cell_target
            ``(B, M)`` RAW counts of the cell, gathered at ``measured``.
        nbr_target
            ``(B, M)`` count scale aggregated neighborhood counts, gathered at ``measured``.
        measured
            ``(M,)`` measured vocabulary indices.

        Returns
        -------
        loss, parts
            The scalar loss and a dict of its terms for logging.
        """
        m = float(measured.numel())
        lib_c = cell_target.sum(1, keepdim=True)
        lib_n = nbr_target.sum(1, keepdim=True)
        cell_nb = masked_nb_nll(
            cell_target, out["cell_logits"], self.cell_log_theta, measured, library=lib_c
        ) / m
        cell = cell_nb
        dw = float(self.config.detection_weight)
        det = torch.zeros((), device=cell.device)
        if dw > 0:
            det = masked_bernoulli_detection_bce(cell_target, out["cell_logits"], measured) / m
            cell = cell + dw * det
        niche_self = masked_nb_nll(
            cell_target, out["niche_self_logits"], self.niche_log_theta, measured, library=lib_c
        ) / m
        niche_nbr = masked_nb_nll(
            nbr_target, out["niche_nbr_logits"], self.niche_log_theta, measured, library=lib_n
        ) / m
        niche = 0.5 * (niche_self + niche_nbr)
        vq = out["cell_vq_loss"] + out["niche_vq_loss"]
        loss = cell + self.config.niche_weight * niche + self.config.vq_weight * vq
        parts = dict(
            loss=float(loss.detach()),
            cell_nb=float(cell_nb.detach()),
            detection=float(det.detach()),
            niche=float(niche.detach()),
            vq=float(vq.detach()),
            cell_perplexity=float(out["cell_perplexity"].detach()),
            niche_perplexity=float(out["niche_perplexity"].detach()),
        )
        return loss, parts

    @torch.inference_mode()
    def encode(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return codes and latents without gradients (same arguments as :meth:`forward`)."""
        was_training = self.training
        self.eval()
        out = self.forward(*args, **kwargs)
        if was_training:
            self.train()
        return {k: out[k] for k in ("cell_idx", "niche_idx", "z_cell", "z_niche")}


def save_foundation(model: FoundationVQVAE, path: str | Path) -> Path:
    """Write ``state_dict`` plus the embedded :class:`FoundationConfig` (and a sibling json)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.config.to_dict()}, path)
    cfg = model.config.to_dict()
    cfg["vocabulary"] = f"<{len(model.config.vocabulary)} tokens>"
    path.with_suffix(".json").write_text(json.dumps(cfg, indent=2))
    return path


def load_foundation(path: str | Path, device: str | torch.device = "cpu") -> FoundationVQVAE:
    """Load a checkpoint written by :func:`save_foundation`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = FoundationVQVAE(FoundationConfig.from_dict(ckpt["config"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
