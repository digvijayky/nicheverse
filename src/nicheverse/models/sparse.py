"""Sparse bag-of-genes encoder over a shared vocabulary (``encoder_type="sparse_bag"``).

Across a hundred panels the input is not a fixed length vector: every dataset measures a
different subset of one shared vocabulary, and most cells detect a few dozen genes. Embedding
the measured genes with an :class:`torch.nn.EmbeddingBag` costs O(number of detected genes)
instead of O(vocabulary), so a 30k token space is as cheap as a 300 gene panel.

Design
------
* per sample weights are the library size normalized ``log1p`` counts, the same feature the
  rest of nicheverse trains on;
* pooling is a MASKED MEAN (weights are normalized per cell), so a 6k panel and a 300 gene
  panel produce embeddings on the same scale and panel size is not a covariate;
* the library size and the number of detected genes, which the masked mean deliberately
  removes, are appended back as two explicit scalars per bag;
* counts and the segmentation free transcript context field get SEPARATE embedding tables,
  and cells with no molecule table use a learned missing context vector;
* several bags can be pooled and concatenated, so the neighborhood branch reuses the same
  module on ``[self, aggregated neighbors]``.

The module also accepts a plain dense tensor, which makes it a drop in member of the encoder
registry (and keeps the registry wide tests meaningful).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
import torch.nn as nn

from .encoders import _mlp, register_encoder

__all__ = ["SparseBag", "SparseBagEncoder"]

_EPS = 1e-8


class SparseBag(NamedTuple):
    """One CSR style bag of genes for a minibatch.

    Attributes
    ----------
    idx
        ``(nnz,)`` int64 vocabulary token ids, concatenated over the cells of the batch.
    val
        ``(nnz,)`` float32 RAW counts aligned with ``idx``.
    off
        ``(B + 1,)`` int64 CSR offsets (``include_last_offset`` convention).
    """

    idx: torch.Tensor
    val: torch.Tensor
    off: torch.Tensor


def _row_index(off: torch.Tensor, nnz: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(row_of_each_entry, per_row_length)`` from CSR offsets."""
    lengths = (off[1:] - off[:-1]).to(torch.long)
    row = torch.repeat_interleave(torch.arange(len(lengths), device=off.device), lengths)
    if row.numel() != nnz:  # pragma: no cover - guards a malformed bag
        raise ValueError(f"offsets describe {row.numel()} entries but the bag holds {nnz}")
    return row, lengths


class SparseBagEncoder(nn.Module):
    """EmbeddingBag encoder over a shared gene vocabulary.

    Parameters
    ----------
    in_dim
        Nominal dense input width, used only to infer ``n_bags`` when it is not given and
        for the dense fallback path.
    out_dim
        Output (latent) width.
    hidden
        Hidden widths of the MLP stack applied to the pooled vector; ``hidden[0]`` is also
        the default embedding width.
    dropout
        Dropout of the MLP stack.
    vocab_size
        Size of the shared vocabulary. Defaults to ``in_dim`` (the dense fallback).
    embed_dim
        Gene embedding width. Defaults to ``hidden[0]``.
    n_bags
        Number of bags concatenated before the MLP (1 for the cell branch, 2 for the
        neighborhood branch). Defaults to ``round(in_dim / vocab_size)``.
    use_context
        Allocate the transcript context table and the learned missing context vector.
    target_sum
        Library size the counts are normalized to before ``log1p`` (10000 by default, the
        standard single cell convention used elsewhere in nicheverse).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int],
        dropout: float = 0.2,
        *,
        vocab_size: int | None = None,
        embed_dim: int | None = None,
        n_bags: int | None = None,
        use_context: bool = True,
        target_sum: float = 1e4,
    ) -> None:
        super().__init__()
        hidden = list(hidden)
        if len(hidden) < 1:
            raise ValueError("hidden must contain at least one layer width")
        self.vocab_size = int(vocab_size or in_dim)
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        self.n_bags = int(n_bags if n_bags is not None else max(1, round(in_dim / self.vocab_size)))
        self.embed_dim = int(embed_dim or hidden[0])
        self.use_context = bool(use_context)
        self.target_sum = float(target_sum)
        self.count_embed = nn.EmbeddingBag(
            self.vocab_size, self.embed_dim, mode="sum", include_last_offset=True
        )
        nn.init.normal_(self.count_embed.weight, std=0.02)
        if self.use_context:
            self.context_embed = nn.EmbeddingBag(
                self.vocab_size, self.embed_dim, mode="sum", include_last_offset=True
            )
            nn.init.normal_(self.context_embed.weight, std=0.02)
            self.missing_context = nn.Parameter(torch.zeros(self.embed_dim))
        pooled = self.n_bags * self.embed_dim * (2 if self.use_context else 1) + 2 * self.n_bags
        self.norm = nn.LayerNorm(pooled)
        self.mlp = _mlp(pooled, hidden, out_dim, dropout)

    # -- pooling -----------------------------------------------------------------------
    def _pool(self, bag: SparseBag, table: nn.EmbeddingBag) -> tuple[torch.Tensor, torch.Tensor]:
        """Masked mean pool one bag; also return ``(log1p library, log1p n detected)``."""
        idx, val, off = bag.idx.to(torch.long), bag.val.to(torch.float32), bag.off.to(torch.long)
        b = off.numel() - 1
        row, _ = _row_index(off, idx.numel())
        lib = torch.zeros(b, device=val.device, dtype=val.dtype).index_add_(0, row, val)
        w = torch.log1p(val * (self.target_sum / lib.clamp_min(_EPS))[row])
        den = torch.zeros(b, device=val.device, dtype=val.dtype).index_add_(0, row, w.abs())
        w = w / den.clamp_min(_EPS)[row]
        emb = table(idx, offsets=off, per_sample_weights=w)
        # Count DETECTED genes, not stored entries, so an explicitly stored zero (a gene
        # the panel measures but this cell did not detect) leaves the encoding untouched.
        nz = torch.zeros(b, device=val.device, dtype=val.dtype).index_add_(
            0, row, (val != 0).to(val.dtype)
        )
        scal = torch.stack([torch.log1p(lib), torch.log1p(nz)], 1)
        return emb, scal

    def _pool_dense(self, x: torch.Tensor, table: nn.EmbeddingBag) -> tuple[torch.Tensor, torch.Tensor]:
        """Dense fallback: the same masked mean written as a matrix product."""
        den = x.abs().sum(1, keepdim=True).clamp_min(_EPS)
        emb = (x / den) @ table.weight
        scal = torch.stack(
            [torch.log1p(den.squeeze(1)), torch.log1p((x != 0).sum(1).to(x.dtype))], 1
        )
        return emb, scal

    # -- forward -----------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor | Sequence[SparseBag],
        context: Sequence[SparseBag | None] | None = None,
        has_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a batch.

        Parameters
        ----------
        x
            Either a dense ``(B, in_dim)`` tensor (fallback path, used by the encoder
            registry tests and by any caller that already has dense features), or a sequence
            of ``n_bags`` :class:`SparseBag` instances of RAW counts.
        context
            Optional sequence of ``n_bags`` transcript context bags (``None`` entries are
            allowed and are replaced by the learned missing context vector).
        has_context
            ``(B,)`` 0/1 tensor marking which cells actually have a transcript context field.
            Rows with 0 get the learned missing context vector.

        Returns
        -------
        torch.Tensor
            ``(B, out_dim)`` latent.
        """
        parts: list[torch.Tensor] = []
        scals: list[torch.Tensor] = []
        if torch.is_tensor(x):
            width = x.shape[1] // self.n_bags
            chunks = [x[:, i * width : (i + 1) * width] for i in range(self.n_bags)]
            if chunks[0].shape[1] != self.count_embed.weight.shape[0]:
                raise ValueError(
                    f"dense input width {x.shape[1]} is not {self.n_bags} x vocab_size "
                    f"{self.vocab_size}"
                )
            for c in chunks:
                e, s = self._pool_dense(c, self.count_embed)
                parts.append(e)
                scals.append(s)
            if self.use_context:
                for c in chunks:
                    ctx = self.missing_context.expand(c.shape[0], -1)
                    parts.append(ctx)
            b = x.shape[0]
            device = x.device
        else:
            bags = list(x)
            if len(bags) != self.n_bags:
                raise ValueError(f"expected {self.n_bags} bags, got {len(bags)}")
            for bag in bags:
                e, s = self._pool(bag, self.count_embed)
                parts.append(e)
                scals.append(s)
            b = bags[0].off.numel() - 1
            device = bags[0].val.device
            if self.use_context:
                ctxs = list(context) if context is not None else [None] * self.n_bags
                if len(ctxs) != self.n_bags:
                    raise ValueError(f"expected {self.n_bags} context bags, got {len(ctxs)}")
                miss = self.missing_context.to(device).expand(b, -1)
                for cb in ctxs:
                    if cb is None:
                        parts.append(miss)
                        continue
                    e, _ = self._pool(cb, self.context_embed)
                    if has_context is not None:
                        keep = has_context.reshape(-1, 1).to(e.dtype)
                        e = keep * e + (1.0 - keep) * miss
                    parts.append(e)
        parts.append(torch.cat(scals, 1).to(parts[0].dtype))
        h = torch.cat(parts, 1)
        if h.shape[1] != self.norm.normalized_shape[0]:  # pragma: no cover - guard
            raise ValueError(f"pooled width {h.shape[1]} != expected {self.norm.normalized_shape[0]}")
        del b, device
        return self.mlp(self.norm(h))


@register_encoder("sparse_bag")
def _sparse_bag_encoder(
    in_dim: int, out_dim: int, hidden: Sequence[int], dropout: float, **kwargs
) -> SparseBagEncoder:
    return SparseBagEncoder(in_dim, out_dim, hidden, dropout, **kwargs)
