"""Tests for the sophistication/quality additions: vectorized aggregation,
graph backends, cosine VQ, public namespaces, and training knobs."""

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import torch

import nicheverse as nv
from nicheverse.data import SpatialDataset
from nicheverse.models import HierarchicalVQVAE, ModelConfig, load_checkpoint, save_checkpoint
from nicheverse.training import TrainConfig, train_model


# ----------------------------------------------------------------------------
# Vectorized neighborhood aggregation correctness
# ----------------------------------------------------------------------------
def _reference_knn_agg(X, xy, k, agg):
    """Independent per-cell reference for the kNN aggregation (single sample)."""
    from sklearn.neighbors import NearestNeighbors

    n = X.shape[0]
    kk = min(k + 1, n)
    nbrs = NearestNeighbors(n_neighbors=kk, algorithm="ball_tree").fit(xy)
    dist, idx = nbrs.kneighbors(xy)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    out = torch.empty((n, X.shape[1]), dtype=torch.float32)
    for i in range(n):
        ni, nd = idx[i, 1:], dist[i, 1:]
        nf = Xt[ni]
        if agg == "mean":
            out[i] = nf.mean(0)
        elif agg == "weighted_mean":
            w = 1.0 / (torch.as_tensor(nd, dtype=torch.float32) + 1e-8)
            w = (w / w.sum()).unsqueeze(1)
            out[i] = (nf * w).sum(0)
        else:
            out[i] = nf.max(0).values
    return out


@pytest.mark.parametrize("agg", ["mean", "weighted_mean", "max"])
def test_vectorized_matches_reference(agg):
    rng = np.random.default_rng(3)
    n, g = 120, 10
    X = rng.normal(size=(n, g)).astype(np.float32)
    xy = rng.uniform(0, 200, size=(n, 2))
    sids = np.array(["S"] * n)
    ds = SpatialDataset(X, xy, sids, k_neighbors=8, neighborhood_aggregation=agg, spatial_graph="knn")
    got = ds.neighborhood_features[:, g:]  # aggregated half
    ref = _reference_knn_agg(X, xy, 8, agg)
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4)


def test_chunking_is_invariant():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(300, 6)).astype(np.float32)
    xy = rng.uniform(0, 100, size=(300, 2))
    sids = np.array(["S"] * 300)
    big = SpatialDataset(X, xy, sids, k_neighbors=10, agg_chunk=10_000).neighborhood_features
    small = SpatialDataset(X, xy, sids, k_neighbors=10, agg_chunk=7).neighborhood_features
    assert torch.allclose(big, small, atol=0, rtol=0)


@pytest.mark.parametrize(
    "graph,kw", [("radius", {"radius": 60.0}), ("delaunay", {}), ("alpha_complex", {})]
)
def test_graph_backends_produce_valid_features(graph, kw):
    rng = np.random.default_rng(7)
    n, g = 80, 8
    X = rng.normal(size=(n, g)).astype(np.float32)
    xy = rng.uniform(0, 200, size=(n, 2))
    sids = np.array(["A"] * 40 + ["B"] * 40)
    ds = SpatialDataset(X, xy, sids, k_neighbors=6, spatial_graph=graph, **kw)
    nf = ds.neighborhood_features
    assert nf.shape == (n, g * 2)
    assert not torch.isnan(nf).any()
    assert torch.allclose(nf[:, :g], torch.as_tensor(X))  # self half preserved


def test_radius_requires_radius():
    X = np.ones((5, 3), dtype=np.float32)
    xy = np.random.default_rng(0).uniform(0, 1, (5, 2))
    with pytest.raises(ValueError, match="radius"):
        SpatialDataset(X, xy, np.array(["A"] * 5), spatial_graph="radius", radius=None)


# ----------------------------------------------------------------------------
# Cosine VQ + model convenience
# ----------------------------------------------------------------------------
def test_cosine_vq_forward_and_roundtrip(tmp_path):
    cfg = ModelConfig(
        input_dim=16,
        hidden_dims=(24,),
        cell_embedding_dim=8,
        cell_num_embeddings=12,
        neighborhood_embedding_dim=12,
        neighborhood_num_embeddings=6,
        vq_distance="cosine",
        gene_names=tuple(f"g{i}" for i in range(16)),
    )
    model = HierarchicalVQVAE(cfg).train()
    cb, nb = torch.randn(32, 16), torch.randn(32, 32)
    out = model(cb, nb)
    assert out[0].shape == (32, 16)
    assert model.cell_vq.distance_metric == "cosine"
    ck = tmp_path / "cos.pt"
    save_checkpoint(model, ck)
    assert load_checkpoint(ck).config.vq_distance == "cosine"


def test_invalid_vq_distance():
    with pytest.raises(ValueError, match="vq_distance"):
        ModelConfig(input_dim=4, vq_distance="manhattan")


def test_encode_matches_forward():
    cfg = ModelConfig(
        input_dim=12,
        hidden_dims=(16,),
        cell_embedding_dim=6,
        cell_num_embeddings=8,
        neighborhood_embedding_dim=8,
        neighborhood_num_embeddings=4,
    )
    model = HierarchicalVQVAE(cfg).eval()
    cb, nb = torch.randn(20, 12), torch.randn(20, 24)
    ci, ni = model.encode(cb, nb)
    _, _, _, _, fci, fni, _, _ = model(cb, nb)
    assert torch.equal(ci, fci.reshape(-1))
    assert torch.equal(ni, fni.reshape(-1))
    assert ci.shape == (20,)


def test_from_checkpoint(tmp_path):
    cfg = ModelConfig(
        input_dim=10,
        cell_num_embeddings=8,
        neighborhood_num_embeddings=4,
        gene_names=tuple(f"g{i}" for i in range(10)),
    )
    model = HierarchicalVQVAE(cfg)
    ck = tmp_path / "m.pt"
    save_checkpoint(model, ck)
    loaded = HierarchicalVQVAE.from_checkpoint(ck)
    assert loaded.config.input_dim == 10


# ----------------------------------------------------------------------------
# public namespaces + constants
# ----------------------------------------------------------------------------
def _toy_adata(n=160, g=24, seed=0):
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(rng.poisson(1.0, size=(n, g)).astype(np.float32))
    a = ad.AnnData(X=X)
    a.var_names = [f"g{i}" for i in range(g)]
    a.obs["sample_id"] = np.array(["S1"] * (n // 2) + ["S2"] * (n - n // 2))
    a.obsm["spatial"] = np.column_stack([rng.uniform(0, 500, n), rng.uniform(0, 500, n)])
    return a


def test_namespaces_exposed():
    for mod in ("models", "data", "training", "plotting", "utils"):
        assert hasattr(nv, mod), mod
    assert callable(nv.train_model) and callable(nv.predict_codes) and callable(nv.Trainer)
    assert nv.read_xenium_cohort is nv.data.read_xenium_cohort
    assert nv.load_checkpoint is nv.models.load_checkpoint
    assert nv.Keys.CELL_CODE == "cell_codebook_idx"
    assert nv.anndata_keys()["neighborhood_code"] == "neighborhood_codebook_idx"


def test_spatial_neighbors_writes_obsm():
    a = _toy_adata()
    feats = nv.data.spatial_neighbors(a, k_neighbors=5)
    assert "neighborhood_features" in a.obsm
    assert a.obsm["neighborhood_features"].shape == (a.n_obs, a.n_vars * 2)
    assert feats.shape == (a.n_obs, a.n_vars * 2)


# ----------------------------------------------------------------------------
# Training knobs: validation split + early stopping + resume + grad clip
# ----------------------------------------------------------------------------
def test_val_split_early_stopping_and_best(tmp_path):
    a = _toy_adata()
    mc = ModelConfig(
        input_dim=a.n_vars,
        hidden_dims=(16,),
        cell_embedding_dim=6,
        cell_num_embeddings=8,
        neighborhood_embedding_dim=8,
        neighborhood_num_embeddings=4,
        gene_names=tuple(a.var_names),
    )
    tc = TrainConfig(
        num_epochs=6,
        batch_size=64,
        k_neighbors=5,
        log_every=100,
        val_fraction=0.25,
        early_stopping_patience=2,
        grad_clip=1.0,
        save_best=True,
    )
    model, trained = train_model(a, tmp_path, model_config=mc, train_config=tc)
    import json

    losses = json.loads((tmp_path / "training_losses.json").read_text())
    assert "val_total" in losses[0]
    assert "cell_perplexity" in losses[0]
    assert (tmp_path / "best_checkpoint.pt").exists()


def test_resume_from(tmp_path):
    a = _toy_adata()
    mc = ModelConfig(
        input_dim=a.n_vars,
        hidden_dims=(16,),
        cell_embedding_dim=6,
        cell_num_embeddings=8,
        neighborhood_embedding_dim=8,
        neighborhood_num_embeddings=4,
        gene_names=tuple(a.var_names),
    )
    d1 = tmp_path / "run1"
    train_model(
        a,
        d1,
        model_config=mc,
        train_config=TrainConfig(num_epochs=1, batch_size=64, k_neighbors=5, log_every=100),
    )
    d2 = tmp_path / "run2"
    tc = TrainConfig(
        num_epochs=1,
        batch_size=64,
        k_neighbors=5,
        log_every=100,
        resume_from=str(d1 / "hierarchical_vqvae_checkpoint.pt"),
    )
    model, _ = train_model(a, d2, model_config=mc, train_config=tc)
    assert (d2 / "hierarchical_vqvae_checkpoint.pt").exists()
