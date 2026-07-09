"""Tests for the neighborhood aggregation kernels (default weighted_mean unchanged)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nicheverse.data import SpatialDataset


def _toy(seed=0, n=60, g=8):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(n, g)).astype(np.float32),
        rng.uniform(0, 200, (n, 2)),
        np.array(["S"] * n),
    )


@pytest.mark.parametrize(
    "agg,kw",
    [
        ("weighted_mean", {}),
        ("inverse_square", {}),
        ("gaussian", {"bandwidth": 40.0}),
        ("mean", {}),
        ("max", {}),
    ],
)
def test_kernel_shapes_and_self_half_preserved(agg, kw):
    x, xy, s = _toy()
    ds = SpatialDataset(x, xy, s, k_neighbors=8, neighborhood_aggregation=agg, **kw)
    nf = ds.neighborhood_features
    assert nf.shape == (60, 16)
    assert not torch.isnan(nf).any()
    assert torch.allclose(nf[:, :8], torch.as_tensor(x))  # self half is the raw expression


def test_gaussian_requires_bandwidth():
    x, xy, s = _toy()
    with pytest.raises(ValueError, match="bandwidth"):
        SpatialDataset(x, xy, s, neighborhood_aggregation="gaussian")


def test_gaussian_wide_bandwidth_approaches_mean():
    # As bandwidth -> infinity the Gaussian weights become uniform, i.e. the mean.
    x, xy, s = _toy()
    mean = SpatialDataset(
        x, xy, s, k_neighbors=10, neighborhood_aggregation="mean"
    ).neighborhood_features[:, 8:]
    wide = SpatialDataset(
        x, xy, s, k_neighbors=10, neighborhood_aggregation="gaussian", bandwidth=1e6
    ).neighborhood_features[:, 8:]
    narrow = SpatialDataset(
        x, xy, s, k_neighbors=10, neighborhood_aggregation="gaussian", bandwidth=1.0
    ).neighborhood_features[:, 8:]
    assert torch.allclose(wide, mean, atol=1e-3)
    assert not torch.allclose(narrow, mean, atol=1e-3)
