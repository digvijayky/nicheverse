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


# ---- opt-in loss options: count cell recon (observed-total library), detection BCE,
#      Dirichlet-multinomial niche recon, graph-TV spatial loss, named gaussian_nll ----

from nicheverse.losses import (  # noqa: E402
    NICHE_SPATIAL_LOSSES,
    SPATIAL_LOSSES,
    bernoulli_detection_bce,
    dirichlet_multinomial_nll,
    gaussian_nll,
    graph_total_variation,
)


def _mc_opt(a, cell_recon="default", niche_recon="mse", detection_weight=0.0):
    return ModelConfig(
        input_dim=a.n_vars,
        hidden_dims=(16,),
        cell_embedding_dim=6,
        cell_num_embeddings=8,
        neighborhood_embedding_dim=8,
        neighborhood_num_embeddings=4,
        recon="mse",
        cell_recon=cell_recon,
        niche_recon=niche_recon,
        detection_weight=detection_weight,
        gene_names=tuple(a.var_names),
    )


def test_gaussian_nll_equals_mse():
    """The named gaussian_nll must be byte-identical to F.mse_loss(pred, target)."""
    import torch.nn.functional as F

    tgt = torch.randn(9, 7)
    pred = torch.randn(9, 7)
    assert torch.equal(gaussian_nll(tgt, pred), F.mse_loss(pred, tgt))


def test_graph_tv_registered_and_niche_scoped():
    """graph_total_variation resolves via SPATIAL_LOSSES['graph_tv'] and is niche-scoped."""
    assert SPATIAL_LOSSES["graph_tv"] is graph_total_variation
    assert "graph_tv" in NICHE_SPATIAL_LOSSES
    z = torch.randn(20, 5, requires_grad=True)
    coords = torch.rand(20, 2) * 100
    v = graph_total_variation(z, coords, k=4)
    assert v.ndim == 0 and torch.isfinite(v)
    v.backward()
    assert z.grad is not None


def test_nb_nll_external_library_matches_manual():
    """Passing an external library must change mu = softmax(cr) * library, not x.sum(1)."""
    x = torch.randint(0, 10, (12, 16)).float()
    cr = torch.randn(12, 16)
    lt = torch.zeros(16)
    lib = torch.rand(12) * 100 + 1.0
    default = nb_nll(x, cr, lt)  # uses x.sum(1)
    ext = nb_nll(x, cr, lt, library=lib)  # uses the explicit library
    same = nb_nll(x, cr, lt, library=x.sum(1))  # equals default when library == x.sum(1)
    assert torch.isfinite(ext) and torch.isfinite(default)
    assert torch.allclose(default, same, atol=1e-5)
    assert not torch.allclose(default, ext)


def test_poisson_nll_external_library():
    x = torch.randint(0, 10, (12, 16)).float()
    cr = torch.randn(12, 16)
    lib = torch.rand(12) * 50 + 1.0
    assert torch.allclose(poisson_nll(x, cr), poisson_nll(x, cr, library=x.sum(1)), atol=1e-5)
    assert torch.isfinite(poisson_nll(x, cr, library=lib))


def test_bernoulli_detection_bce_finite_and_differentiable():
    x = torch.randint(0, 5, (10, 16)).float()
    logits = torch.randn(10, 16, requires_grad=True)
    v = bernoulli_detection_bce(x, logits)
    assert v.ndim == 0 and torch.isfinite(v)
    v.backward()
    assert logits.grad is not None


def test_dirichlet_multinomial_nll_finite_and_differentiable():
    target = torch.rand(10, 12)  # continuous compositional target
    logits = torch.randn(10, 12, requires_grad=True)
    la = torch.zeros(12, requires_grad=True)
    v = dirichlet_multinomial_nll(target, logits, la)
    assert v.ndim == 0 and torch.isfinite(v)
    v.backward()
    assert logits.grad is not None and la.grad is not None


def test_count_modes_have_log_theta_only_for_nb():
    assert hasattr(HierarchicalVQVAE(_mc_opt(_toy(), "nb")), "cell_log_theta")
    assert hasattr(HierarchicalVQVAE(_mc_opt(_toy(), "both")), "cell_log_theta")
    # poisson has no dispersion parameter
    assert not hasattr(HierarchicalVQVAE(_mc_opt(_toy(), "poisson")), "cell_log_theta")


def test_dirmult_niche_has_log_alpha():
    m = HierarchicalVQVAE(_mc_opt(_toy(), niche_recon="dirichlet_multinomial"))
    assert hasattr(m, "niche_log_alpha") and m.niche_log_alpha.shape == (16,)
    assert not hasattr(HierarchicalVQVAE(_mc_opt(_toy())), "niche_log_alpha")


def test_config_validation():
    with pytest.raises(ValueError, match="cell_recon"):
        ModelConfig(input_dim=4, cell_recon="bad")
    with pytest.raises(ValueError, match="niche_recon"):
        ModelConfig(input_dim=4, niche_recon="bad")
    # count cell_recon is incompatible with recon in {nb,poisson}
    with pytest.raises(ValueError, match="cell_recon"):
        ModelConfig(input_dim=4, recon="nb", cell_recon="nb")
    # detection requires a count-mode cell_recon (needs raw counts as target)
    with pytest.raises(ValueError, match="detection_weight"):
        ModelConfig(input_dim=4, cell_recon="mse", detection_weight=1.0)


@pytest.mark.parametrize("mode", ["nb", "poisson", "both"])
def test_train_cell_recon_count(tmp_path, mode):
    """Log1p encoder input + raw-count NLL target scaled by the observed total count."""
    a = _toy()
    tc = TrainConfig(
        num_epochs=2, batch_size=32, k_neighbors=5, log_every=100, normalize=True, log1p=True
    )
    Trainer(tc).fit(a, tmp_path, model_config=_mc_opt(a, mode))
    losses = json.loads((Path(tmp_path) / "training_losses.json").read_text())
    assert len(losses) == 2 and all(np.isfinite(x["total"]) for x in losses)
    m = load_checkpoint(Path(tmp_path) / "hierarchical_vqvae_checkpoint.pt")
    assert m.config.cell_recon == mode
    out = ad.read_h5ad(Path(tmp_path) / "adata_with_hierarchical_embeddings.h5ad")
    assert "_raw_counts" not in out.layers  # internal layer must not leak


def test_train_detection_hurdle(tmp_path):
    a = _toy()
    tc = TrainConfig(num_epochs=2, batch_size=32, k_neighbors=5, normalize=True, log1p=True)
    Trainer(tc).fit(a, tmp_path, model_config=_mc_opt(a, "nb", detection_weight=0.5))
    losses = json.loads((Path(tmp_path) / "training_losses.json").read_text())
    assert all(np.isfinite(x["total"]) for x in losses)


def test_train_dirmult_niche(tmp_path):
    a = _toy()
    tc = TrainConfig(num_epochs=2, batch_size=32, k_neighbors=5)  # mse cell default
    Trainer(tc).fit(a, tmp_path, model_config=_mc_opt(a, niche_recon="dirichlet_multinomial"))
    losses = json.loads((Path(tmp_path) / "training_losses.json").read_text())
    assert all(np.isfinite(x["total"]) for x in losses)


def test_train_graph_tv_spatial(tmp_path):
    a = _toy()
    tc = TrainConfig(
        num_epochs=2, batch_size=32, k_neighbors=5,
        spatial_loss_type="graph_tv", spatial_loss_weight=0.1, spatial_loss_k=4,
    )
    Trainer(tc).fit(a, tmp_path, model_config=_mc_opt(a))
    losses = json.loads((Path(tmp_path) / "training_losses.json").read_text())
    assert all(np.isfinite(x["total"]) for x in losses)


def test_default_config_byte_identical(tmp_path):
    """A default-config run (mse cell, mse niche, no detection/TV/dirmult) must match a
    run built the old way (recon='mse' only), proving the new options are inert when off."""
    a = _toy()
    tc = TrainConfig(num_epochs=3, batch_size=32, k_neighbors=5, normalize=True, log1p=True)
    old = _mc(a, "mse")  # no cell_recon/niche_recon/detection fields set -> all defaults
    new = _mc_opt(a)  # cell_recon="default", niche_recon="mse", detection_weight=0
    Trainer(tc).fit(a.copy(), tmp_path / "old", model_config=old)
    Trainer(tc).fit(a.copy(), tmp_path / "new", model_config=new)
    lo = json.loads((tmp_path / "old" / "training_losses.json").read_text())
    ln = json.loads((tmp_path / "new" / "training_losses.json").read_text())
    for eo, en in zip(lo, ln):
        assert eo["total"] == pytest.approx(en["total"], abs=0, rel=0)
