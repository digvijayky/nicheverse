"""Shard round trip, and on the fly aggregation vs the dense featurizer."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from nicheverse.data.dataset import SpatialDataset
from nicheverse.data.shards import ShardReader, aggregate_neighbors, to_bag, write_shard


def _toy(n=40, v=12, k=6, seed=0):
    rng = np.random.default_rng(seed)
    counts = sp.csr_matrix(rng.poisson(1.2, size=(n, v)).astype(np.float32))
    xy = rng.uniform(0, 100, size=(n, 2)).astype(np.float32)
    from nicheverse.data._knn import knn_query

    dist, idx = knn_query(xy.astype(np.float64), k + 1)
    return counts, xy, idx.astype(np.int32), dist.astype(np.float32)


def test_shard_roundtrip(tmp_path):
    counts, xy, idx, dist = _toy()
    ctx = sp.csr_matrix((counts.toarray() * 2).astype(np.float32))
    has = np.ones(counts.shape[0], np.uint8)
    has[:5] = 0
    p = write_shard(
        tmp_path / "shard_0000.npz",
        counts,
        idx,
        dist,
        xy,
        np.zeros(counts.shape[0], np.int32),
        context=ctx,
        has_context=has,
        meta=dict(dataset_id=3, platform_id=1, species_id=0, radius_um=50.0),
    )
    r = ShardReader(p)
    assert r.n_cells == counts.shape[0] and r.vocab_size == counts.shape[1]
    assert np.allclose(r.counts.toarray(), counts.toarray())
    assert np.allclose(r.context.toarray(), ctx.toarray())
    assert np.array_equal(r.has_context, has)
    assert np.allclose(r.xy, xy)
    assert np.array_equal(r.knn_idx, idx)
    assert r.meta["dataset_id"] == 3 and r.meta["radius_um"] == 50.0


def test_shard_without_context(tmp_path):
    counts, xy, idx, dist = _toy()
    p = write_shard(tmp_path / "s.npz", counts, idx, dist, xy, np.zeros(counts.shape[0], np.int32))
    r = ShardReader(p)
    assert r.context.nnz == 0 and not r.has_context.any()


def test_aggregate_equals_dense_featurizer():
    """The sparse on the fly aggregation must reproduce SpatialDataset's dense neighbor half."""
    counts, xy, idx, dist = _toy(n=60, v=15, k=8, seed=3)
    x = torch.as_tensor(counts.toarray())
    ds = SpatialDataset(
        x, xy, np.zeros(len(xy), dtype=int), k_neighbors=8, spatial_graph="knn",
        neighborhood_aggregation="weighted_mean",
    )
    dense_neighbor_half = ds.neighborhood_features[:, counts.shape[1]:].numpy()
    got = aggregate_neighbors(counts, idx, dist).toarray()
    assert got.shape == dense_neighbor_half.shape
    assert np.allclose(got, dense_neighbor_half, atol=1e-5), np.abs(got - dense_neighbor_half).max()


def test_aggregate_subset_matches_full():
    counts, xy, idx, dist = _toy(n=50, v=9, k=5, seed=7)
    full = aggregate_neighbors(counts, idx, dist).toarray()
    rows = np.array([3, 17, 44, 0])
    assert np.allclose(aggregate_neighbors(counts, idx, dist, rows).toarray(), full[rows])


def test_aggregate_ignores_padding_and_self():
    counts = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 4.0], [2.0, 2.0]], dtype=np.float32))
    idx = np.array([[0, 1, -1], [1, 2, -1], [2, 0, 1]], dtype=np.int32)
    dist = np.array([[0.0, 2.0, np.inf], [0.0, 4.0, np.inf], [0.0, 1.0, 1.0]], dtype=np.float32)
    got = aggregate_neighbors(counts, idx, dist).toarray()
    assert np.allclose(got[0], [0.0, 4.0])          # only cell 1
    assert np.allclose(got[1], [2.0, 2.0])          # only cell 2
    assert np.allclose(got[2], [0.5, 2.0])          # cells 0 and 1 at equal distance


def test_to_bag_shapes():
    counts, *_ = _toy(n=7, v=5)
    i, v, o = to_bag(counts)
    assert o.shape == (8,) and len(i) == len(v) == counts.nnz
    assert o[-1] == counts.nnz
