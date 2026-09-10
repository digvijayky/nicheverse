"""On disk shards for multi panel training, and on the fly neighborhood aggregation.

A foundation model over tens of millions of cells cannot hold a dense
``cells x k x genes`` neighborhood tensor (the featurizer used by
:class:`~nicheverse.data.SpatialDataset` is fine for one cohort and impossible for a
hundred). A shard stores, per cell, the sparse counts over the shared vocabulary, the
sparse transcript context field when a molecule table exists, the cell metadata, and the
per sample neighbor list (indices and distances) computed once at staging time. The
neighborhood vector is then rebuilt per minibatch by a sparse matrix product, which is
mathematically identical to the dense weighted mean but costs O(nnz) instead of
O(cells x k x genes).

Additive module: nothing in the existing pipeline imports it.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .dataset import MIN_NEIGHBOR_MICRON

__all__ = ["ShardReader", "aggregate_neighbors", "to_bag", "write_shard"]

_ARRAYS = (
    "counts_indptr",
    "counts_idx",
    "counts_val",
    "ctx_indptr",
    "ctx_idx",
    "ctx_val",
    "has_ctx",
    "x",
    "y",
    "sample_local",
    "knn_idx",
    "knn_dist",
)


def write_shard(
    path: str | Path,
    counts: sp.csr_matrix,
    knn_idx: np.ndarray,
    knn_dist: np.ndarray,
    xy: np.ndarray,
    sample_local: np.ndarray,
    context: sp.csr_matrix | None = None,
    has_context: np.ndarray | None = None,
    meta: Mapping[str, object] | None = None,
) -> Path:
    """Write one shard as a compressed ``.npz``.

    Parameters
    ----------
    path
        Destination ``.npz``.
    counts
        ``(n_cells, vocab_size)`` CSR of RAW counts over the shared vocabulary.
    knn_idx, knn_dist
        ``(n_cells, k)`` neighbor row ids LOCAL to this shard (``-1`` padding) and their
        distances in microns (``inf`` padding). Column 0 is the cell itself.
    xy
        ``(n_cells, 2)`` micron coordinates.
    sample_local
        ``(n_cells,)`` integer sample id within the dataset.
    context
        Optional ``(n_cells, vocab_size)`` CSR of the segmentation free transcript context
        field, on the same raw count scale.
    has_context
        ``(n_cells,)`` uint8 flag; defaults to all ones when ``context`` is given.
    meta
        Extra scalars stored alongside (dataset id, platform id, species id, radius, ...).

    Returns
    -------
    Path
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = counts.tocsr()
    n = counts.shape[0]
    if knn_idx.shape[0] != n or xy.shape[0] != n or len(sample_local) != n:
        raise ValueError("shard arrays must all have n_cells rows")
    if context is None:
        ctx_indptr = np.zeros(n + 1, np.int64)
        ctx_idx = np.zeros(0, np.int32)
        ctx_val = np.zeros(0, np.float32)
        hc = np.zeros(n, np.uint8)
    else:
        context = context.tocsr()
        if context.shape != counts.shape:
            raise ValueError("context must have the same shape as counts")
        ctx_indptr = context.indptr.astype(np.int64)
        ctx_idx = context.indices.astype(np.int32)
        ctx_val = context.data.astype(np.float32)
        hc = (np.ones(n, np.uint8) if has_context is None else np.asarray(has_context, np.uint8))
    payload = dict(
        counts_indptr=counts.indptr.astype(np.int64),
        counts_idx=counts.indices.astype(np.int32),
        counts_val=counts.data.astype(np.float32),
        ctx_indptr=ctx_indptr,
        ctx_idx=ctx_idx,
        ctx_val=ctx_val,
        has_ctx=hc,
        x=np.asarray(xy[:, 0], np.float32),
        y=np.asarray(xy[:, 1], np.float32),
        sample_local=np.asarray(sample_local, np.int32),
        knn_idx=np.asarray(knn_idx, np.int32),
        knn_dist=np.asarray(knn_dist, np.float32),
        vocab_size=np.int64(counts.shape[1]),
    )
    for k, v in (meta or {}).items():
        payload["meta_" + k] = np.asarray(v)
    np.savez(path, **payload)
    return path


def aggregate_neighbors(
    counts: sp.csr_matrix,
    knn_idx: np.ndarray,
    knn_dist: np.ndarray,
    rows: np.ndarray | None = None,
    aggregation: str = "weighted_mean",
    min_micron: float = MIN_NEIGHBOR_MICRON,
) -> sp.csr_matrix:
    """Weighted mean of the neighbors of ``rows``, as a sparse matrix.

    Identical in value to the dense aggregation in
    :meth:`nicheverse.data.SpatialDataset._aggregate`: the cell itself (column 0, and any
    slot whose index equals the row) and padded slots (index ``-1``) are excluded, weights
    are ``1 / max(d, min_micron)`` for ``"weighted_mean"`` (``1`` for ``"mean"``) and are
    normalized to sum to one per row.

    Parameters
    ----------
    counts
        ``(n_cells, n_features)`` CSR over the shard.
    knn_idx, knn_dist
        ``(n_cells, k)`` neighbor ids and distances.
    rows
        Rows to aggregate; ``None`` means every row.
    aggregation
        ``"weighted_mean"`` (default) or ``"mean"``.
    min_micron
        Distance floor, so a coincident centroid cannot dominate the mean.

    Returns
    -------
    scipy.sparse.csr_matrix
        ``(len(rows), n_features)``.
    """
    n = counts.shape[0]
    rows = np.arange(n) if rows is None else np.asarray(rows)
    idx = knn_idx[rows]
    dist = knn_dist[rows].astype(np.float64)
    valid = (idx >= 0) & (idx != rows[:, None]) & np.isfinite(dist)
    if aggregation == "weighted_mean":
        w = np.where(valid, 1.0 / np.maximum(dist, min_micron), 0.0)
    elif aggregation == "mean":
        w = valid.astype(np.float64)
    else:
        raise ValueError(f"aggregation must be 'weighted_mean' or 'mean', got {aggregation!r}")
    tot = w.sum(1, keepdims=True)
    w = np.divide(w, tot, out=np.zeros_like(w), where=tot > 0)
    # Restrict the product to the rows actually referenced, so the cost is set by the
    # minibatch and not by the size of the shard.
    uniq, pos = np.unique(np.maximum(idx, 0).ravel(), return_inverse=True)
    r = np.repeat(np.arange(len(rows)), idx.shape[1])
    W = sp.csr_matrix((w.ravel(), (r, pos)), shape=(len(rows), len(uniq)))
    return (W @ counts[uniq]).tocsr()


def to_bag(mat: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a CSR into ``(indices, values, offsets)`` for :class:`torch.nn.EmbeddingBag`."""
    mat = mat.tocsr()
    return (
        mat.indices.astype(np.int64),
        mat.data.astype(np.float32),
        mat.indptr.astype(np.int64),
    )


class ShardReader:
    """Lazy reader for a shard written by :func:`write_shard`."""

    def __init__(self, path: str | Path, mmap: bool = True) -> None:
        self.path = Path(path)
        self._z = np.load(self.path, mmap_mode="r" if mmap else None)
        self.vocab_size = int(self._z["vocab_size"])
        self.n_cells = int(len(self._z["counts_indptr"]) - 1)
        self.meta = {
            k[5:]: self._z[k].item() if self._z[k].ndim == 0 else np.asarray(self._z[k])
            for k in self._z.files
            if k.startswith("meta_")
        }

    def _csr(self, prefix: str) -> sp.csr_matrix:
        return sp.csr_matrix(
            (
                np.asarray(self._z[prefix + "_val"]),
                np.asarray(self._z[prefix + "_idx"]),
                np.asarray(self._z[prefix + "_indptr"]),
            ),
            shape=(self.n_cells, self.vocab_size),
        )

    @property
    def counts(self) -> sp.csr_matrix:
        """Raw counts over the vocabulary, ``(n_cells, vocab_size)`` CSR."""
        return self._csr("counts")

    @property
    def context(self) -> sp.csr_matrix:
        """Transcript context field over the vocabulary, ``(n_cells, vocab_size)`` CSR."""
        return self._csr("ctx")

    @property
    def has_context(self) -> np.ndarray:
        """``(n_cells,)`` uint8 flag: 1 when this cell has a transcript context field."""
        return np.asarray(self._z["has_ctx"])

    @property
    def knn_idx(self) -> np.ndarray:
        """``(n_cells, k)`` local neighbor row ids, ``-1`` padded."""
        return np.asarray(self._z["knn_idx"])

    @property
    def knn_dist(self) -> np.ndarray:
        """``(n_cells, k)`` neighbor distances in microns, ``inf`` padded."""
        return np.asarray(self._z["knn_dist"])

    @property
    def xy(self) -> np.ndarray:
        """``(n_cells, 2)`` micron coordinates."""
        return np.stack([np.asarray(self._z["x"]), np.asarray(self._z["y"])], 1)

    @property
    def sample_local(self) -> np.ndarray:
        """``(n_cells,)`` sample id within the dataset."""
        return np.asarray(self._z["sample_local"])

    def aggregate(self, rows: np.ndarray | None = None, aggregation: str = "weighted_mean") -> sp.csr_matrix:
        """On the fly neighborhood aggregation for ``rows`` (see :func:`aggregate_neighbors`)."""
        return aggregate_neighbors(self.counts, self.knn_idx, self.knn_dist, rows, aggregation)
