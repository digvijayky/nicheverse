"""Regression tests for two dataset.py fixes.

BUG 1: reciprocal neighbor weights (weighted_mean / inverse_square) used a
numeric epsilon inside the reciprocal, so a coincident centroid (d~0) got an
astronomically large weight and dominated the niche aggregate. Fixed with a
physical distance floor MIN_NEIGHBOR_MICRON.

BUG 2: under knn_radius / radius, a cell with zero in-radius neighbors produced
an all-zeros aggregated niche half (indistinguishable from a genuine near-zero
neighborhood), letting the niche codebook allocate a spurious empty-niche code.
Fixed by falling back to the single nearest neighbor for isolated cells.
"""

from __future__ import annotations

import numpy as np
import torch

from nicheverse.data import SpatialDataset
from nicheverse.data.dataset import MIN_NEIGHBOR_MICRON


# ---------------------------------------------------------------------------
# BUG 1: coincident centroid must not dominate the weighted mean
# ---------------------------------------------------------------------------
def test_coincident_centroid_does_not_dominate():
    g = 6
    # cell 0 at origin; cell 1 coincident with it; cells 2,3 nearby but farther.
    xy = np.array(
        [[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [0.0, 4.0]], dtype=np.float64
    )
    X = np.zeros((4, g), dtype=np.float32)
    X[1] = 100.0  # the coincident neighbor carries a huge, distinctive signal
    X[2] = 1.0
    X[3] = 1.0
    sids = np.array(["S"] * 4)
    ds = SpatialDataset(
        X, xy, sids, k_neighbors=3, neighborhood_aggregation="weighted_mean",
        spatial_graph="knn",
    )
    niche0 = ds.neighborhood_features[0, g:]  # aggregated half for cell 0

    # If the coincident neighbor (X[1]=100) dominated, the niche would be ~100.
    # With the 1um floor its weight is capped, so it cannot swamp cells 2 and 3.
    coincident_val = float(niche0[0])
    assert coincident_val < 100.0, coincident_val
    # And the niche must not be a near-copy of that single overlapping cell.
    assert not torch.allclose(niche0, torch.full((g,), 100.0), atol=10.0)

    # Explicit weight-bound check: reproduce the kernel and confirm the
    # coincident neighbor's normalized weight is bounded well below 1.
    d = np.array([3.0, 4.0], dtype=np.float64)  # distances to cells 2 and 3
    d0 = 0.0  # coincident
    w = np.array(
        [1.0 / max(d0, MIN_NEIGHBOR_MICRON), 1.0 / d[0], 1.0 / d[1]]
    )
    w = w / w.sum()
    assert w[0] < 0.7, w  # coincident neighbor does not monopolize the mean

    # Sanity: the aggregated value equals the explicit floored weighted mean.
    vals = np.array([100.0, 1.0, 1.0])
    expected = float((w * vals).sum())
    assert abs(coincident_val - expected) < 1e-3, (coincident_val, expected)


def test_inverse_square_also_floored():
    g = 4
    xy = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 0.0]], dtype=np.float64)
    X = np.zeros((3, g), dtype=np.float32)
    X[1] = 50.0
    X[2] = 1.0
    ds = SpatialDataset(
        X, xy, np.array(["S"] * 3), k_neighbors=2,
        neighborhood_aggregation="inverse_square", spatial_graph="knn",
    )
    niche0 = ds.neighborhood_features[0, g:]
    # floored weights: 1/1^2 for coincident, 1/25 for the far cell
    w = np.array([1.0 / (MIN_NEIGHBOR_MICRON ** 2), 1.0 / 25.0])
    w = w / w.sum()
    expected = float((w * np.array([50.0, 1.0])).sum())
    assert abs(float(niche0[0]) - expected) < 1e-3, (float(niche0[0]), expected)
    assert float(niche0[0]) < 50.0


# ---------------------------------------------------------------------------
# BUG 2: isolated cell (no in-radius neighbor) falls back to nearest neighbor
# ---------------------------------------------------------------------------
def _isolated_setup(graph):
    g = 5
    # A tight cluster of 4 cells, and a 5th cell far away with NO neighbor
    # within a small radius. Its nearest neighbor is cell 3.
    xy = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [100.0, 100.0]],
        dtype=np.float64,
    )
    X = np.arange(5 * g, dtype=np.float32).reshape(5, g) + 1.0
    sids = np.array(["S"] * 5)
    ds = SpatialDataset(
        X, xy, sids, k_neighbors=4, neighborhood_aggregation="mean",
        spatial_graph=graph, radius=5.0,
    )
    return ds, X, g


def test_isolated_cell_knn_radius_uses_nearest_neighbor():
    ds, X, g = _isolated_setup("knn_radius")
    niche4 = ds.neighborhood_features[4, g:]  # aggregated half for the isolated cell
    # NOT all zeros
    assert not torch.allclose(niche4, torch.zeros(g)), niche4
    # nearest neighbor to cell 4 (at 100,100) is cell 3 (at 1,1)
    nn = np.linalg.norm(np.array([100.0, 100.0]) - np.array([1.0, 1.0]))
    d3 = min(
        np.linalg.norm(np.array([100.0, 100.0]) - np.array([px, py]))
        for px, py in [(0, 0), (1, 0), (0, 1), (1, 1)]
    )
    assert abs(nn - d3) < 1e-9  # cell 3 is indeed nearest
    # with a single fallback neighbor, mean aggregation == that neighbor's row
    assert torch.allclose(niche4, torch.as_tensor(X[3])), (niche4, X[3])


def test_isolated_cell_radius_uses_nearest_neighbor():
    ds, X, g = _isolated_setup("radius")
    niche4 = ds.neighborhood_features[4, g:]
    assert not torch.allclose(niche4, torch.zeros(g)), niche4
    assert torch.allclose(niche4, torch.as_tensor(X[3])), (niche4, X[3])


def test_isolated_cell_weighted_mean_not_zeros():
    # Same geometry but with the default weighted_mean kernel: single fallback
    # neighbor still yields exactly that neighbor's row, never zeros.
    g = 5
    xy = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [100.0, 100.0]],
        dtype=np.float64,
    )
    X = np.arange(5 * g, dtype=np.float32).reshape(5, g) + 1.0
    ds = SpatialDataset(
        X, xy, np.array(["S"] * 5), k_neighbors=4,
        neighborhood_aggregation="weighted_mean", spatial_graph="knn_radius",
        radius=5.0,
    )
    niche4 = ds.neighborhood_features[4, g:]
    assert not torch.allclose(niche4, torch.zeros(g))
    assert torch.allclose(niche4, torch.as_tensor(X[3]))


# ---------------------------------------------------------------------------
# Invariance: cells WITH in-radius neighbors are unchanged by the fixes.
# ---------------------------------------------------------------------------
def test_in_radius_cells_match_knn_when_radius_not_binding():
    # With uniform(0,200) and n=120 the typical NN separation is many microns,
    # well above the 1um floor, so knn_radius with a large radius (no edge is
    # pruned) must equal plain knn exactly, and no cell is isolated.
    rng = np.random.default_rng(11)
    n, g = 120, 8
    X = rng.normal(size=(n, g)).astype(np.float32)
    xy = rng.uniform(0, 200, (n, 2))
    sids = np.array(["S"] * n)
    knn = SpatialDataset(
        X, xy, sids, k_neighbors=8, neighborhood_aggregation="weighted_mean",
        spatial_graph="knn",
    ).neighborhood_features
    knn_radius = SpatialDataset(
        X, xy, sids, k_neighbors=8, neighborhood_aggregation="weighted_mean",
        spatial_graph="knn_radius", radius=1e6,
    ).neighborhood_features
    assert torch.allclose(knn, knn_radius, atol=1e-6), (knn - knn_radius).abs().max()


def test_floor_is_inert_on_enhancement_synthetic_geometry():
    # Verifies the note in the task brief: the upstream reference tests
    # (test_enhancements::test_vectorized_matches_reference,
    # test_aggregation) build their point cloud with the exact draw order
    # X = rng.normal(...) BEFORE xy = rng.uniform(0,200,...), seed 3, n=120.
    # Under that ordering the minimum nearest-neighbor separation is ~1.21um,
    # ABOVE MIN_NEIGHBOR_MICRON, so the distance floor never engages there and
    # the floored weighted_mean is identical (to fp32 rounding) to the unfloored
    # 1/(d+eps) reference. Hence the pinned upstream results stay green.
    # (Drawing xy without first drawing X consumes the RNG differently and lands
    # on a different, denser cloud with a sub-1um pair, so the draw order matters
    # and is reproduced faithfully here.)
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(3)  # same seed as the upstream reference tests
    n, g = 120, 10
    X = rng.normal(size=(n, g)).astype(np.float32)  # drawn first, as upstream
    xy = rng.uniform(0, 200, size=(n, 2))
    min_nn = float(
        NearestNeighbors(n_neighbors=2).fit(xy).kneighbors(xy)[0][:, 1].min()
    )
    assert min_nn > MIN_NEIGHBOR_MICRON, min_nn  # floor does not bite here

    got = SpatialDataset(
        X, xy, np.array(["S"] * n), k_neighbors=8,
        neighborhood_aggregation="weighted_mean", spatial_graph="knn",
    ).neighborhood_features[:, g:]
    kk = min(8 + 1, n)
    dist, idx = NearestNeighbors(n_neighbors=kk).fit(xy).kneighbors(xy)
    Xt = torch.as_tensor(X)
    ref = torch.empty((n, g))
    for i in range(n):
        ni, nd = idx[i, 1:], dist[i, 1:]
        w = 1.0 / (torch.as_tensor(nd, dtype=torch.float32) + 1e-8)  # unfloored ref
        w = (w / w.sum()).unsqueeze(1)
        ref[i] = (Xt[ni] * w).sum(0)
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4), (got - ref).abs().max()


def test_floor_does_change_result_when_a_sub_micron_pair_exists():
    # Complementary positive control: construct a cloud that DOES contain a
    # sub-1um neighbor and confirm the floor genuinely alters the weighted_mean
    # (so the floor is not a no-op that happens to never fire). This is the
    # regression guard for BUG 1 in a realistic multi-neighbor setting.
    g = 4
    xy = np.array(
        [[0.0, 0.0], [0.3, 0.0], [5.0, 0.0], [0.0, 6.0]], dtype=np.float64
    )  # cell 1 is 0.3um from cell 0 -> below the floor
    X = np.zeros((4, g), dtype=np.float32)
    X[1] = 20.0
    X[2] = 1.0
    X[3] = 1.0
    got = SpatialDataset(
        X, xy, np.array(["S"] * 4), k_neighbors=3,
        neighborhood_aggregation="weighted_mean", spatial_graph="knn",
    ).neighborhood_features[0, g:]
    # unfloored reference weight for the 0.3um neighbor is 1/0.3 ~= 3.33, which
    # would dominate; the floor caps it at 1/1 = 1.0, so the two aggregates differ.
    d = np.array([0.3, 5.0, np.hypot(0.0, 6.0)])
    w_unfloored = 1.0 / (d + 1e-8)
    w_unfloored /= w_unfloored.sum()
    unfloored_val = float((w_unfloored * np.array([20.0, 1.0, 1.0])).sum())
    assert abs(float(got[0]) - unfloored_val) > 1.0, (float(got[0]), unfloored_val)
    assert float(got[0]) < unfloored_val  # floor pulls the coincident weight down
