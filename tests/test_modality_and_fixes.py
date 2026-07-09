"""Regression tests for modality-agnostic entry points and the review fixes."""

from __future__ import annotations

import logging
import os
import tempfile

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import torch

import nicheverse as nv
from nicheverse.data import SpatialDataset, read_spatial, transcript_context
from nicheverse.models import (
    HierarchicalVQVAE,
    ModelConfig,
    build_quantizer,
    load_checkpoint,
    save_checkpoint,
)
from nicheverse.training import TrainConfig, predict_codes, train_model


def _adata(n=30, g=6, samples=1, sample_col="sample_id"):
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=sp.csr_matrix(rng.poisson(2.0, size=(n, g)).astype("float32")))
    a.var_names = [f"g{i}" for i in range(g)]
    a.obs[sample_col] = np.array([f"S{i % samples}" for i in range(n)])
    a.obsm["spatial"] = rng.uniform(0, 100, (n, 2))
    return a


# --- modality-agnostic constructors ---
def test_from_anndata_obsm():
    ds = SpatialDataset.from_anndata(_adata(40, samples=2), k_neighbors=5)
    assert ds.neighborhood_features.shape == (40, 12)


def test_from_anndata_obs_xy_and_scale():
    rng = np.random.default_rng(1)
    a = ad.AnnData(X=sp.csr_matrix(rng.poisson(1.0, size=(20, 6)).astype("float32")))
    a.obs["fov"] = "F1"
    a.obs["cx"] = rng.uniform(0, 800, 20)
    a.obs["cy"] = rng.uniform(0, 800, 20)
    ds = SpatialDataset.from_anndata(
        a, sample_col="fov", x_col="cx", y_col="cy", coord_scale=0.12028, k_neighbors=4
    )
    assert ds.neighborhood_features.shape == (20, 12)


def test_read_spatial_builds_coords_and_sample():
    rng = np.random.default_rng(2)
    a = ad.AnnData(X=sp.csr_matrix(rng.poisson(1.0, size=(15, 6)).astype("float32")))
    a.obs["center_x"] = rng.uniform(0, 500, 15)
    a.obs["center_y"] = rng.uniform(0, 500, 15)
    out = read_spatial(a, x_col="center_x", y_col="center_y", coord_scale=2.0)
    assert "spatial" in out.obsm and "sample_id" in out.obs
    assert out.obsm["spatial"].shape == (15, 2)
    assert nv.read_spatial is read_spatial


def test_read_spatial_requires_coordinates():
    a = ad.AnnData(X=sp.csr_matrix(np.ones((5, 4), dtype="float32")))
    with pytest.raises(ValueError, match="coordinates"):
        read_spatial(a)


# --- data-layer bug fixes ---
def test_delaunay_small_sample_falls_back():
    coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    feats = np.arange(12.0, dtype="float32").reshape(2, 6)
    ds = SpatialDataset(
        feats, coords, np.array(["s", "s"]), k_neighbors=3, spatial_graph="delaunay"
    )
    assert np.isfinite(ds.neighborhood_features.numpy()).all()


def test_radius_coincident_coordinates():
    coords = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [5.0, 5.0]])
    feats = np.tile(np.arange(6.0, dtype="float32"), (4, 1))
    ds = SpatialDataset(
        feats, coords, np.array(["s"] * 4), k_neighbors=3, spatial_graph="radius", radius=10.0
    )
    assert np.isfinite(ds.neighborhood_features.numpy()).all()


# --- transcript platform presets ---
def test_transcript_cosmx_platform():
    rng = np.random.default_rng(3)
    d = tempfile.mkdtemp()
    import pandas as pd

    tx = pd.DataFrame(
        {
            "x_global_px": rng.uniform(0, 100, 200),
            "y_global_px": rng.uniform(0, 100, 200),
            "target": rng.choice(["g0", "g1", "g2", "NegPrb1"], 200),
        }
    )
    pq = os.path.join(d, "tx.parquet")
    tx.to_parquet(pq)
    a = ad.AnnData(X=sp.csr_matrix(np.zeros((10, 3), dtype="float32")))
    a.var_names = ["g0", "g1", "g2"]
    a.obs["sample_id"] = "S"
    a.obsm["spatial"] = rng.uniform(0, 100, (10, 2))
    feats = transcript_context(a, pq, radius=30.0, platform="cosmx")
    assert feats.shape == (10, 3) and np.isfinite(feats).all()


def test_transcript_unknown_platform_rejected():
    a = ad.AnnData(X=sp.csr_matrix(np.zeros((2, 2), dtype="float32")))
    a.var_names = ["g0", "g1"]
    a.obs["sample_id"] = "S"
    a.obsm["spatial"] = np.zeros((2, 2))
    with pytest.raises(ValueError, match="platform"):
        transcript_context(a, "x.parquet", platform="nope")


# --- quantizer fixes ---
def test_fsq_exposes_embedding():
    f = build_quantizer(
        "fsq", num_embeddings=0, embedding_dim=4, commitment_cost=0.0, levels=(4, 4, 4)
    )
    assert f.embedding.weight.shape == (64, 4)
    assert torch.isfinite(f.embedding.weight).all()


def test_fsq_rejects_level_two():
    from nicheverse.models import FSQ

    with pytest.raises(ValueError, match=">= 3"):
        FSQ(4, levels=(2, 4, 4))


def test_productvq_embedding_dim_matches_weight():
    p = build_quantizer(
        "pq", num_embeddings=16, embedding_dim=8, commitment_cost=0.25, num_subspaces=4
    )
    assert p.embedding_dim == p.embedding.weight.shape[1] == 2


# --- encoder_kwargs ---
def test_encoder_kwargs_roundtrip():
    cfg = ModelConfig(
        input_dim=20,
        hidden_dims=(16,),
        cell_embedding_dim=8,
        cell_num_embeddings=8,
        neighborhood_embedding_dim=8,
        neighborhood_num_embeddings=4,
        encoder_type="transformer",
        encoder_kwargs={"patch_size": 8, "num_layers": 1, "num_heads": 2},
        gene_names=tuple(f"g{i}" for i in range(20)),
    )
    m = HierarchicalVQVAE(cfg).train()
    assert m(torch.randn(6, 20), torch.randn(6, 40))[0].shape == (6, 20)
    p = os.path.join(tempfile.mkdtemp(), "m.pt")
    save_checkpoint(m, p)
    assert load_checkpoint(p).config.encoder_kwargs == {
        "patch_size": 8,
        "num_layers": 1,
        "num_heads": 2,
    }


# --- training guards ---
def test_nb_recon_requires_raw_counts():
    cfg = ModelConfig(input_dim=6, recon="nb", gene_names=tuple(f"g{i}" for i in range(6)))
    with pytest.raises(ValueError, match="raw counts"):
        train_model(
            _adata(20), tempfile.mkdtemp(), model_config=cfg, train_config=TrainConfig(num_epochs=1)
        )


def test_size_one_trailing_batch_does_not_crash():
    train_model(
        _adata(5),
        tempfile.mkdtemp(),
        train_config=TrainConfig(num_epochs=1, batch_size=4, k_neighbors=3, save_best=False),
    )


def test_predict_skips_double_normalization(caplog):
    gn = tuple(f"g{i}" for i in range(6))
    m = HierarchicalVQVAE(
        ModelConfig(
            input_dim=6,
            hidden_dims=(16,),
            cell_embedding_dim=8,
            cell_num_embeddings=8,
            neighborhood_embedding_dim=8,
            neighborhood_num_embeddings=4,
            gene_names=gn,
        )
    ).eval()
    a = _adata(12)
    a.uns["log1p"] = {"base": None}
    with caplog.at_level(logging.WARNING):
        predict_codes(a, m, k_neighbors=3, batch_size=8, device="cpu")
    assert "double normalization" in caplog.text
