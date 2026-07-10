"""Exact nearest-neighbor backend for the spatial graph build.

Every query here is an EXACT nearest-neighbor query -- never an approximate
index (never ivfflat / ivfpq / hnsw / annoy). Two backends are provided and
they return the same neighbor set per cell:

* GPU (OPT-IN, for the kNN query only; the DEFAULT is the CPU ball-tree):
  ``cuml.neighbors.NearestNeighbors(algorithm="brute")``. Brute force is an
  exhaustive pairwise-distance search, so it is exact. RAPIDS cuML docs:
  https://docs.rapids.ai/api/cuml/stable/api/#nearest-neighbors
* CPU fallback: ``sklearn.neighbors.NearestNeighbors(algorithm="ball_tree")``
  (the original exact backend). Also the ONLY backend for the fixed-radius
  query -- cuML 26.x's NearestNeighbors has no ``radius_neighbors`` method.

Backend selection
-----------------
The exact sklearn ball-tree (CPU) is the DEFAULT. The GPU cuML brute-force path
is OPT-IN: enable it with ``NICHEVERSE_KNN_BACKEND=gpu`` or
:func:`set_knn_backend`. It is exact and ~1.7-2x faster, but the kNN graph build
is a one-time cost and the GPU exactness relies on the guards below (a workaround
for a cuML float32 issue), so the robust CPU ball-tree stays the default.

Exactness of the GPU brute force (IMPORTANT)
--------------------------------------------
cuML brute force computes squared distances via the expanded form
``||a-b||^2 = ||a||^2 - 2 a.b + ||b||^2`` in float32. With spatial coordinates
offset far from the origin (Xenium/TMA slide frames put a core anywhere in a
large coordinate system, so ``||a||^2`` can be ~1e7-1e9), float32 catastrophic
cancellation corrupts small distances by up to ~sqrt(2) microns, which can flip
the neighbor chosen right at the k-th boundary (measured against the sklearn
ball-tree ground truth; consistent with cuML issues #4624 / #5569). The
``two_pass_precision`` flag does NOT fix this (it is itself buggy in 26.x and
returns garbage distances). float64 input does not help either -- cuML
downcasts to float32 internally.

Exactness is restored with three guards, each verified on a GPU node against
the sklearn ball-tree ground truth (0 genuine k-th-neighbor errors and 0
neighbor-set mismatches over 5k / 50k / 100k point clouds with coordinate
offsets up to 1e5 and spreads up to 8000 microns):

1. **Mean-center per query.** Subtract the coordinate mean before the search.
   Euclidean distance is translation invariant, so the true neighbors are
   unchanged, but ``||a||^2`` shrinks to O(spread^2), sharply reducing the
   cancellation error. (Centering alone is necessary but not sufficient.)
2. **Over-fetch candidates.** Ask cuML for ``k + _KNN_OVERFETCH`` neighbors,
   not k. The residual float32 error is bounded, so any true k-NN can only be
   displaced from the top-k by a handful of near-tied points; a small pad keeps
   every true k-NN inside the returned candidate window. pad=0 leaves ~0.3% of
   cells wrong at 100k; pad>=8 gives exactly 0 errors, and _KNN_OVERFETCH=16 is
   used for margin (the extra columns are dropped after re-ranking).
3. **Re-rank by exact float64 distance.** Recompute true float64 distances on
   the ORIGINAL coordinates for the candidate indices (cheap: n*(k+pad)) and
   keep the k smallest. This both fixes the candidate ordering and yields exact
   distances for the aggregation weights.

The GPU brute force with these three guards is mathematically identical to the
sklearn ball-tree (same neighbor set per cell, same k-th distance) and ~2x
faster than sklearn at 100k in 2D, with the speedup growing with n. (A pure
torch.cdist path was evaluated and rejected: float32 cdist is still not exact,
and float64 cdist is exact but ~100x SLOWER than sklearn.)

Tie ordering at exactly-equal distances may differ between backends; this does
not change the neighbor SET, and the downstream weighted-mean aggregation is
order independent.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.neighbors import NearestNeighbors as _SkNN

__all__ = ["knn_query", "radius_query", "gpu_knn_available", "set_knn_backend"]

# Module-level override: None = auto-detect, "cpu" = force sklearn, "gpu" = force cuML kNN.
_FORCED_BACKEND: str | None = None

# Extra candidates fetched from cuML brute force beyond k, to guarantee every
# true k-NN survives the float32 distance error before the exact float64
# re-rank. pad>=8 gives 0 errors vs the sklearn ball-tree at 100k across
# coordinate offsets up to 1e5; 16 is used for margin. See module docstring.
_KNN_OVERFETCH = 16


def set_knn_backend(backend: str | None) -> None:
    """Force the kNN backend. ``"cpu"`` forces the sklearn ball-tree, ``"gpu"``
    forces cuML brute force for the kNN query (raises at query time if
    cuML/CUDA are unavailable), ``None`` restores auto-detection. Primarily for
    tests / reproducibility. Note the fixed-radius query is always CPU (cuML has
    no exact radius search)."""
    global _FORCED_BACKEND
    if backend not in (None, "cpu", "gpu"):
        raise ValueError(f"backend must be None, 'cpu', or 'gpu'; got {backend!r}")
    _FORCED_BACKEND = backend


def _env_backend() -> str | None:
    v = os.environ.get("NICHEVERSE_KNN_BACKEND", "").strip().lower()
    return v if v in ("cpu", "gpu") else None


def _cuml_nn():
    """Return the cuML NearestNeighbors class if cuML + a CUDA device are usable, else None.

    Cached so the import + device probe runs once. Any failure (no cuML, no
    driver, no visible GPU, missing shared libs) yields None -> CPU fallback.
    """
    if getattr(_cuml_nn, "_cache", "unset") != "unset":
        return _cuml_nn._cache
    cls = None
    try:  # pragma: no cover - depends on GPU/cuML at runtime
        import cupy

        if cupy.cuda.runtime.getDeviceCount() > 0:
            from cuml.neighbors import NearestNeighbors as _CuNN

            cls = _CuNN
    except Exception:
        cls = None
    _cuml_nn._cache = cls
    return cls


def gpu_knn_available() -> bool:
    """True if the GPU (cuML brute-force) backend would be used for a kNN query now.
    GPU is OPT-IN (default CPU): True only when it is forced on via
    ``NICHEVERSE_KNN_BACKEND=gpu`` / :func:`set_knn_backend` AND cuML + CUDA are usable."""
    forced = _FORCED_BACKEND or _env_backend()
    if forced == "gpu":
        return _cuml_nn() is not None
    return False


def _use_gpu() -> bool:
    # Default (and forced='cpu') -> the exact sklearn ball-tree. GPU is OPT-IN.
    forced = _FORCED_BACKEND or _env_backend()
    if forced == "gpu":
        if _cuml_nn() is None:
            raise RuntimeError(
                "NICHEVERSE_KNN_BACKEND/backend forced to 'gpu' but cuML + a CUDA "
                "device are not available."
            )
        return True
    return False


def _to_numpy(a):
    """Coerce a cuML/cupy/cudf return to a host numpy array."""
    if isinstance(a, np.ndarray):
        return a
    for attr in ("get", "to_numpy"):  # cupy.ndarray.get / cudf .to_numpy
        f = getattr(a, attr, None)
        if callable(f):
            try:
                return np.asarray(f())
            except Exception:
                pass
    return np.asarray(a)


def knn_query(coords: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact k-nearest-neighbor query, self included.

    Returns ``(dist, idx)`` of shape ``(n, k)``: column 0 is the cell itself at
    distance 0, each row sorted by increasing distance -- the same neighbor set
    as ``NearestNeighbors(n_neighbors=k, algorithm="ball_tree")
    .fit(coords).kneighbors(coords)``. GPU (cuML brute force) is used when
    available, else the sklearn ball-tree.
    """
    coords = np.ascontiguousarray(np.asarray(coords, dtype=np.float64))
    n = coords.shape[0]
    k = int(min(k, n))
    if _use_gpu():
        cls = _cuml_nn()
        # Mean-center to defuse float32 catastrophic cancellation in cuML's
        # expanded squared-distance kernel (see module docstring). Translation
        # invariant, so the true neighbors are unchanged.
        c = (coords - coords.mean(axis=0, keepdims=True)).astype(np.float32)
        # Over-fetch k+pad candidates so no true k-NN is lost to the residual
        # float32 error, then re-rank exactly. kf is capped at n.
        kf = int(min(k + _KNN_OVERFETCH, n))
        nn = cls(n_neighbors=kf, algorithm="brute", metric="euclidean", output_type="numpy")
        nn.fit(c)
        _, cand = nn.kneighbors(c, n_neighbors=kf, return_distance=True)
        cand = _to_numpy(cand).astype(np.int64, copy=False)
        # Exact float64 distances on the ORIGINAL coordinates for the candidates
        # (n*kf, cheap), then keep the k nearest. This gives the exact k-NN set
        # AND exact distances for the aggregation weights, independent of cuML's
        # float32 distance output.
        cd = np.linalg.norm(coords[cand] - coords[:, None, :], axis=2)
        order = np.argsort(cd, axis=1, kind="stable")[:, :k]
        idx = np.take_along_axis(cand, order, axis=1)
        dist = np.take_along_axis(cd, order, axis=1)
        _ensure_self_first(idx, dist, n)
        return dist, idx
    nn = _SkNN(n_neighbors=k, algorithm="ball_tree").fit(coords)
    dist, idx = nn.kneighbors(coords)
    return dist, idx


def radius_query(coords: np.ndarray, radius: float) -> tuple[list, list]:
    """Exact fixed-radius neighbor query, self included, distance sorted.

    Always uses the sklearn ball-tree: cuML 26.x's ``NearestNeighbors`` exposes
    no ``radius_neighbors`` method, so there is no exact GPU radius search (and
    an approximate one is not acceptable). Returns ``(dist_list, idx_list)``:
    one variable-length numpy array per cell, sorted by increasing distance,
    self at distance 0 -- same as ``NearestNeighbors(radius=radius,
    algorithm="ball_tree").radius_neighbors(coords, sort_results=True)``.
    """
    coords = np.ascontiguousarray(np.asarray(coords, dtype=np.float64))
    nn = _SkNN(radius=float(radius), algorithm="ball_tree").fit(coords)
    dist_list, idx_list = nn.radius_neighbors(coords, sort_results=True)
    return dist_list, idx_list


def _ensure_self_first(idx: np.ndarray, dist: np.ndarray, n: int) -> None:
    """In place: guarantee row i has index i (distance 0) at column 0, keeping
    ``idx`` and ``dist`` aligned. After the exact-distance re-rank self is
    already first except when a coincident point ties it out of column 0 (all
    tied distances ~0); this rotates self back to the front for those rows so
    the self-at-column-0 contract the callers rely on always holds."""
    rows = np.arange(n)
    for r in np.where(idx[:, 0] != rows)[0]:
        pos = np.where(idx[r] == r)[0]
        if pos.size:
            p = int(pos[0])
            idx[r, 1 : p + 1] = idx[r, 0:p]
            dist[r, 1 : p + 1] = dist[r, 0:p]
            idx[r, 0], dist[r, 0] = r, 0.0
        else:
            idx[r, 0], dist[r, 0] = r, 0.0
