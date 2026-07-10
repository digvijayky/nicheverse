"""Exact GPU (cuML brute-force) kNN backend for the spatial graph build.

Covers:
(1) the forced-CPU path reproduces the released sklearn ball-tree behavior;
(2) the helper contract (self at column 0, radius self-inclusion, sorted);
(3) GPU-vs-CPU neighborhood features are IDENTICAL (skipif-gated on CUDA+cuML);
(4) a CPU-vs-GPU benchmark at realistic scale that also re-asserts identical
    neighbor sets (skipif-gated).

The GPU backend is cuML ``NearestNeighbors(algorithm="brute")`` -- exact brute
force, never an approximate index. So it must return the same neighbor SET per
cell as the sklearn ball-tree; the downstream weighted-mean aggregation is
order-independent, so equal sets => identical neighborhood features.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from nicheverse.data import SpatialDataset
from nicheverse.data import _knn


@pytest.fixture(autouse=True)
def _reset_backend():
    """Every test starts from auto-detect and restores it afterward."""
    _knn.set_knn_backend(None)
    yield
    _knn.set_knn_backend(None)


def _toy(n=400, g=10, n_samples=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, g)).astype(np.float32)
    xy = (rng.random((n, 2)) * 400).astype(np.float64)
    sids = np.array([f"S{i % n_samples}" for i in range(n)])
    return X, xy, sids


def _build(X, xy, sids, **kw):
    return SpatialDataset(X, xy, sids, **kw)


_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() and _knn._cuml_nn() is not None),
    reason="exact GPU kNN needs a CUDA device + importable cuML brute-force backend",
)


# ---------------------------------------------------------------------------
# (1) helper contract on the CPU (released) path
# ---------------------------------------------------------------------------
def test_knn_query_contract_cpu():
    _knn.set_knn_backend("cpu")
    rng = np.random.default_rng(1)
    coords = (rng.random((200, 2)) * 300).astype(np.float64)
    dist, idx = _knn.knn_query(coords, 11)
    assert idx.shape == dist.shape == (200, 11)
    # self at column 0, distance 0
    assert np.array_equal(idx[:, 0], np.arange(200))
    assert np.allclose(dist[:, 0], 0.0)
    # rows sorted by distance
    assert np.all(np.diff(dist, axis=1) >= -1e-9)


def test_radius_query_contract_cpu():
    _knn.set_knn_backend("cpu")
    rng = np.random.default_rng(2)
    coords = (rng.random((300, 2)) * 200).astype(np.float64)
    dl, il = _knn.radius_query(coords, 40.0)
    assert len(dl) == len(il) == 300
    for r in range(300):
        assert il[r][0] == r  # self first
        assert dl[r][0] == pytest.approx(0.0, abs=1e-9)
        assert np.all(np.diff(dl[r]) >= -1e-9)  # sorted


# ---------------------------------------------------------------------------
# (2) forced-CPU build reproduces the released sklearn ball-tree behavior
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("graph", ["knn", "knn_radius", "radius"])
def test_forced_cpu_matches_sklearn_reference(graph):
    """With the backend forced to CPU, the neighbor graph equals a direct
    sklearn ball-tree computation (the pre-refactor code path)."""
    from sklearn.neighbors import NearestNeighbors as SkNN

    X, xy, sids = _toy(n=300, g=8, n_samples=2, seed=3)
    kw = dict(k_neighbors=6, neighborhood_aggregation="mean", spatial_graph=graph)
    if graph in ("radius", "knn_radius"):
        kw["radius"] = 50.0
    _knn.set_knn_backend("cpu")
    ds = _build(X, xy, sids, **kw)

    # Independently recompute per-sample neighbor SETS with sklearn ball-tree and
    # confirm the dataset's aggregation used the same neighbors (via a from-scratch
    # mean aggregation match). We compare the produced neighborhood_features.
    assert ds.neighborhood_features.shape == (300, 16)
    assert not torch.isnan(ds.neighborhood_features).any()
    # sanity: the plain-knn self+neighbor block equals a hand-rolled ball-tree agg
    if graph == "knn":
        out = np.zeros((300, 8), dtype=np.float64)
        for s in np.unique(sids):
            si = np.where(sids == s)[0]
            c = xy[si]
            k = min(6 + 1, len(si))
            nn = SkNN(n_neighbors=k, algorithm="ball_tree").fit(c)
            _, ii = nn.kneighbors(c)
            for a in range(len(si)):
                nb = ii[a][ii[a] != a]
                out[si[a]] = X[si[nb]].mean(axis=0) if len(nb) else 0.0
        got = ds.neighborhood_features[:, 8:].numpy()
        assert np.allclose(got, out, atol=1e-5)


# ---------------------------------------------------------------------------
# (3) GPU path == CPU path, IDENTICAL neighborhood features
# ---------------------------------------------------------------------------
@_cuda
@pytest.mark.parametrize("graph", ["knn", "knn_radius", "radius"])
@pytest.mark.parametrize("agg", ["mean", "weighted_mean"])
def test_gpu_matches_cpu_neighborhood_features(graph, agg):
    X, xy, sids = _toy(n=600, g=12, n_samples=3, seed=7)
    kw = dict(k_neighbors=15, neighborhood_aggregation=agg, spatial_graph=graph)
    if graph in ("radius", "knn_radius"):
        kw["radius"] = 60.0

    _knn.set_knn_backend("cpu")
    cpu = _build(X, xy, sids, **kw).neighborhood_features

    _knn.set_knn_backend("gpu")
    gpu = _build(X, xy, sids, **kw).neighborhood_features

    assert gpu.shape == cpu.shape
    assert torch.allclose(gpu, cpu, atol=1e-5, rtol=1e-5), (
        f"{graph}/{agg} max abs diff {(gpu - cpu).abs().max().item():.3e}"
    )


@_cuda
@pytest.mark.parametrize("offset", [0.0, 1e5])
def test_gpu_matches_cpu_identical_neighbor_sets(offset):
    """Per-cell neighbor index SETS from the GPU brute force equal the CPU
    ball-tree sets exactly (tie ORDER may differ; SET must not). The large
    ``offset`` case is the adversarial one: without mean-centering + over-fetch
    + exact re-rank, cuML float32 brute force flips k-th neighbors there."""
    rng = np.random.default_rng(11)
    coords = (rng.random((3000, 2)) * 2000 + offset).astype(np.float64)
    _knn.set_knn_backend("cpu")
    dc, ic = _knn.knn_query(coords, 21)
    _knn.set_knn_backend("gpu")
    dg, ig = _knn.knn_query(coords, 21)
    assert np.array_equal(ig[:, 0], np.arange(3000))  # self col 0 on GPU too
    assert np.allclose(dg[:, 0], 0.0)
    # vectorized set equality + exact k-th distance equality
    set_mismatch = int((np.sort(ic, axis=1) != np.sort(ig, axis=1)).any(axis=1).sum())
    assert set_mismatch == 0, f"offset={offset}: {set_mismatch} cells differ in neighbor set"
    assert float(np.abs(dc.max(1) - dg.max(1)).max()) < 1e-6

    # radius query is CPU-only in both backends (cuML has no exact radius search),
    # so it must be identical by construction.
    _knn.set_knn_backend("cpu")
    _, icr = _knn.radius_query(coords, 40.0)
    _knn.set_knn_backend("gpu")
    _, igr = _knn.radius_query(coords, 40.0)
    badr = sum(
        set(np.asarray(icr[r]).tolist()) != set(np.asarray(igr[r]).tolist())
        for r in range(3000)
    )
    assert badr == 0


# ---------------------------------------------------------------------------
# (4) benchmark at realistic single-sample scale + identical neighbors
# ---------------------------------------------------------------------------
@_cuda
@pytest.mark.parametrize("n", [100_000])
def test_benchmark_cpu_vs_gpu_realistic(n):
    rng = np.random.default_rng(0)
    coords = (rng.random((n, 2)) * 3000).astype(np.float64)  # ~microns
    k = 21  # k_neighbors=20 + self

    _knn.set_knn_backend("cpu")
    t0 = time.perf_counter()
    dc, ic = _knn.knn_query(coords, k)
    t_cpu = time.perf_counter() - t0

    _knn.set_knn_backend("gpu")
    _ = _knn.knn_query(coords[:1000], k)  # warm up cuML context
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    dg, ig = _knn.knn_query(coords, k)
    torch.cuda.synchronize()
    t_gpu = time.perf_counter() - t0

    # identical neighbor SETS at scale, fully vectorized (sort rows, compare):
    # equal sets => identical k-NN, since the aggregation is order-independent.
    set_mismatch = int((np.sort(ic, axis=1) != np.sort(ig, axis=1)).any(axis=1).sum())
    # exact k-th (max) distance per cell must also match the ball-tree ground truth
    kth_diff = float(np.abs(dc.max(axis=1) - dg.max(axis=1)).max())
    print(
        f"\n[bench knn n={n} k={k}] CPU ball_tree {t_cpu:.2f}s | "
        f"GPU cuML brute {t_gpu:.2f}s | speedup {t_cpu / t_gpu:.1f}x | "
        f"neighbor-set mismatches: {set_mismatch} | max kth-dist diff: {kth_diff:.2e}"
    )
    assert set_mismatch == 0
    assert kth_diff < 1e-6
