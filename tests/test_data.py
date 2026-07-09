import numpy as np
import torch

from nicheverse.data import SpatialDataset


def test_neighborhood_shapes():
    rng = np.random.default_rng(0)
    n, g = 50, 12
    X = rng.normal(size=(n, g)).astype(np.float32)
    xy = rng.normal(size=(n, 2)) * 100
    sids = np.array(["A"] * 30 + ["B"] * 20)
    ds = SpatialDataset(X, xy, sids, k_neighbors=5, neighborhood_aggregation="weighted_mean")
    assert len(ds) == n
    cb, nb, idx = ds[0]
    assert cb.shape == (g,)
    assert nb.shape == (g * 2,)
    assert int(idx) == 0


def test_per_sample_isolation():
    rng = np.random.default_rng(1)
    n, g = 40, 8
    X = rng.normal(size=(n, g)).astype(np.float32)
    xy = np.column_stack([rng.uniform(0, 1, n), rng.uniform(0, 1, n)])
    xy[20:] += np.array([1000.0, 1000.0])
    sids = np.array(["S1"] * 20 + ["S2"] * 20)
    ds = SpatialDataset(X, xy, sids, k_neighbors=3, neighborhood_aggregation="mean")
    nb_self = ds.neighborhood_features[0]
    near = ds.neighborhood_features[1]
    assert nb_self.shape == (g * 2,)
    assert not torch.allclose(nb_self, near)


def test_singleton_sample_fallback():
    X = np.ones((3, 4), dtype=np.float32)
    xy = np.zeros((3, 2))
    sids = np.array(["A", "B", "C"])
    ds = SpatialDataset(X, xy, sids, k_neighbors=5, neighborhood_aggregation="mean")
    for i in range(3):
        cb, nb, _ = ds[i]
        assert torch.allclose(nb[:4], nb[4:])
