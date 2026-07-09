"""Tests for the optional spatial-coherence losses (default training weight is 0)."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from nicheverse import ModelConfig, TrainConfig, Trainer
from nicheverse.losses import SPATIAL_LOSSES

_TYPES = ["laplacian", "contrastive", "codebook_consistency", "graph_tv"]


def test_registry_names():
    assert set(SPATIAL_LOSSES) == set(_TYPES)


@pytest.mark.parametrize("name", _TYPES)
def test_loss_finite_and_differentiable(name):
    torch.manual_seed(0)
    z = torch.randn(30, 8, requires_grad=True)
    coords = torch.rand(30, 2) * 100
    v = SPATIAL_LOSSES[name](z, coords, k=5)
    assert v.ndim == 0 and torch.isfinite(v)
    v.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def _toy():
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=sp.csr_matrix(rng.poisson(1.0, size=(80, 16)).astype("float32")))
    a.var_names = [f"g{i}" for i in range(16)]
    a.obs["sample_id"] = np.array(["S1"] * 40 + ["S2"] * 40)
    a.obsm["spatial"] = np.column_stack([rng.uniform(0, 500, 80), rng.uniform(0, 500, 80)])
    return a


def _mc(a):
    return ModelConfig(
        input_dim=a.n_vars,
        hidden_dims=(16,),
        cell_embedding_dim=6,
        cell_num_embeddings=8,
        neighborhood_embedding_dim=8,
        neighborhood_num_embeddings=4,
        gene_names=tuple(a.var_names),
    )


@pytest.mark.parametrize("lt", _TYPES)
def test_opt_in_training_runs(tmp_path, lt):
    a = _toy()
    tc = TrainConfig(
        num_epochs=2,
        batch_size=32,
        k_neighbors=5,
        log_every=100,
        spatial_loss_type=lt,
        spatial_loss_weight=0.1,
        spatial_loss_k=5,
    )
    Trainer(tc).fit(a, tmp_path, model_config=_mc(a))
    losses = json.loads((Path(tmp_path) / "training_losses.json").read_text())
    assert len(losses) == 2 and all(np.isfinite(x["total"]) for x in losses)


def test_invalid_spatial_loss_rejected(tmp_path):
    a = _toy()
    with pytest.raises(ValueError, match="spatial_loss_type"):
        Trainer(TrainConfig(spatial_loss_type="bad", spatial_loss_weight=1.0)).fit(
            a, tmp_path, model_config=_mc(a)
        )


def test_default_weight_zero_trains(tmp_path):
    a = _toy()
    Trainer(TrainConfig(num_epochs=1, batch_size=32, k_neighbors=5, log_every=100)).fit(
        a, tmp_path, model_config=_mc(a)
    )
    assert (Path(tmp_path) / "hierarchical_vqvae_checkpoint.pt").exists()
