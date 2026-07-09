"""Tests for NB / Poisson reconstruction heads (the mse default stays unchanged)."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from nicheverse import ModelConfig, TrainConfig, Trainer, load_checkpoint
from nicheverse.losses import nb_nll, poisson_nll
from nicheverse.models import HierarchicalVQVAE


def _toy():
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=sp.csr_matrix(rng.poisson(1.5, size=(80, 16)).astype("float32")))
    a.var_names = [f"g{i}" for i in range(16)]
    a.obs["sample_id"] = np.array(["S1"] * 40 + ["S2"] * 40)
    a.obsm["spatial"] = np.column_stack([rng.uniform(0, 500, 80), rng.uniform(0, 500, 80)])
    return a


def _mc(a, recon="mse"):
    return ModelConfig(
        input_dim=a.n_vars,
        hidden_dims=(16,),
        cell_embedding_dim=6,
        cell_num_embeddings=8,
        neighborhood_embedding_dim=8,
        neighborhood_num_embeddings=4,
        recon=recon,
        gene_names=tuple(a.var_names),
    )


def test_nb_nll_finite_and_differentiable():
    x = torch.randint(0, 10, (20, 16)).float()
    cr = torch.randn(20, 16, requires_grad=True)
    lt = torch.zeros(16, requires_grad=True)
    v = nb_nll(x, cr, lt)
    assert v.ndim == 0 and torch.isfinite(v)
    v.backward()
    assert cr.grad is not None and lt.grad is not None


def test_poisson_nll_finite_and_differentiable():
    x = torch.randint(0, 10, (20, 16)).float()
    cr = torch.randn(20, 16, requires_grad=True)
    v = poisson_nll(x, cr)
    assert torch.isfinite(v)
    v.backward()
    assert cr.grad is not None


def test_mse_default_has_no_log_theta():
    assert not hasattr(HierarchicalVQVAE(_mc(_toy(), "mse")), "cell_log_theta")


def test_nb_has_log_theta():
    m = HierarchicalVQVAE(_mc(_toy(), "nb"))
    assert hasattr(m, "cell_log_theta") and m.cell_log_theta.shape == (16,)


def test_invalid_recon_rejected():
    with pytest.raises(ValueError, match="recon"):
        ModelConfig(input_dim=4, recon="bad")


@pytest.mark.parametrize("recon", ["mse", "nb", "poisson"])
def test_train_with_recon(tmp_path, recon):
    a = _toy()
    norm = recon == "mse"
    tc = TrainConfig(
        num_epochs=2, batch_size=32, k_neighbors=5, log_every=100, normalize=norm, log1p=norm
    )
    Trainer(tc).fit(a, tmp_path, model_config=_mc(a, recon))
    losses = json.loads((Path(tmp_path) / "training_losses.json").read_text())
    assert len(losses) == 2 and all(np.isfinite(x["total"]) for x in losses)
    m = load_checkpoint(Path(tmp_path) / "hierarchical_vqvae_checkpoint.pt")
    assert m.config.recon == recon
