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
    # Pin the pre-composite defaults so these tests exercise the classic ``recon=``
    # path (cell_recon="default" defers to recon; niche MSE; no detection). The
    # ModelConfig defaults are now the composite loss (cell_recon="nb",
    # niche_recon="mse_dirmult", detection_weight=0.5); those are covered separately.
    return ModelConfig(
        input_dim=a.n_vars,
        hidden_dims=(16,),
        cell_embedding_dim=6,
        cell_num_embeddings=8,
        neighborhood_embedding_dim=8,
        neighborhood_num_embeddings=4,
        recon=recon,
        cell_recon="default",
        niche_recon="mse",
        detection_weight=0.0,
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
    # negative weights rejected
    with pytest.raises(ValueError, match="w_nb"):
        ModelConfig(input_dim=4, w_nb=-1.0)
    with pytest.raises(ValueError, match="detection_weight"):
        ModelConfig(input_dim=4, detection_weight=-0.5)
    # detection on a non-count mode is IGNORED (warned, not an error), so pure-MSE
    # recovery does not have to also zero detection_weight -> must not raise.
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
    # A DM niche mode needs raw counts (count cell mode) so the count-scale neighbor
    # composition can be built; pair it with cell_recon="nb".
    a = _toy()
    tc = TrainConfig(num_epochs=2, batch_size=32, k_neighbors=5, normalize=True, log1p=True)
    Trainer(tc).fit(a, tmp_path, model_config=_mc_opt(a, "nb", niche_recon="dirichlet_multinomial"))
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


# ---- new-default composite loss (cell_recon=nb + niche_recon=mse_dirmult) ----


def _mc_default(a):
    """ModelConfig at the NEW composite defaults (nb + mse_dirmult + detection 0.5),
    only shrinking the codebook/embedding dims for the tiny toy adata."""
    return ModelConfig(
        input_dim=a.n_vars,
        hidden_dims=(16,),
        cell_embedding_dim=6,
        cell_num_embeddings=8,
        neighborhood_embedding_dim=8,
        neighborhood_num_embeddings=4,
        gene_names=tuple(a.var_names),
    )


def test_new_default_is_composite():
    """The shipped ModelConfig default must be the composite loss."""
    mc = ModelConfig(input_dim=8)
    assert mc.cell_recon == "nb"
    assert mc.niche_recon == "mse_dirmult"
    assert mc.detection_weight == 0.5


def test_train_new_default_finite(tmp_path):
    """The NEW default (nb + mse_dirmult + detection) trains on tiny integer counts
    and yields finite cell AND niche losses at every epoch."""
    a = _toy()  # raw integer Poisson counts
    tc = TrainConfig(num_epochs=3, batch_size=32, k_neighbors=5, normalize=True, log1p=True)
    Trainer(tc).fit(a, tmp_path, model_config=_mc_default(a))
    losses = json.loads((Path(tmp_path) / "training_losses.json").read_text())
    assert len(losses) == 3
    for e in losses:
        assert np.isfinite(e["total"]) and np.isfinite(e["cell"]) and np.isfinite(e["neighborhood"])
    m = load_checkpoint(Path(tmp_path) / "hierarchical_vqvae_checkpoint.pt")
    assert m.config.cell_recon == "nb" and m.config.niche_recon == "mse_dirmult"


def test_pure_mse_recovers_old_path(tmp_path):
    """cell_recon='mse' + niche_recon='mse' + detection_weight=0 must reproduce the OLD
    gaussian-only path EXACTLY (torch.equal on the per-epoch losses)."""
    a = _toy()
    tc = TrainConfig(num_epochs=3, batch_size=32, k_neighbors=5, normalize=True, log1p=True)
    # old path built via the classic recon= route (cell_recon="default", recon="mse").
    old = _mc(a, "mse")
    # explicit pure-MSE recovery via the new selectors.
    recov = _mc_opt(a, cell_recon="mse", niche_recon="mse", detection_weight=0.0)
    Trainer(tc).fit(a.copy(), tmp_path / "old", model_config=old)
    Trainer(tc).fit(a.copy(), tmp_path / "recov", model_config=recov)
    lo = json.loads((tmp_path / "old" / "training_losses.json").read_text())
    lr = json.loads((tmp_path / "recov" / "training_losses.json").read_text())
    assert len(lo) == len(lr) == 3
    for eo, er in zip(lo, lr):
        # exact equality: both paths are gaussian_nll(cb,cr) + gaussian_nll(nb,nr) with
        # identical seeds, data, and no count/detection/dirmult terms.
        assert torch.equal(torch.tensor(eo["total"]), torch.tensor(er["total"]))
        assert torch.equal(torch.tensor(eo["cell"]), torch.tensor(er["cell"]))
        assert torch.equal(torch.tensor(eo["neighborhood"]), torch.tensor(er["neighborhood"]))


def test_niche_dm_target_is_count_scale():
    """The DM niche target (niche_count_target) must be on the RAW-count scale, not the
    log1p mean: its per-cell row sum tracks the raw counts, and scaling the raw counts
    by a constant (holding the log1p neighborhood fixed) changes the DM term."""
    from nicheverse.data import SpatialDataset

    rng = np.random.default_rng(3)
    n, g = 60, 10
    counts = rng.poisson(3.0, size=(n, g)).astype("float32")
    log1p = np.log1p(counts)  # a stand-in log1p "cell feature"
    coords = np.column_stack([rng.uniform(0, 200, n), rng.uniform(0, 200, n)])
    samples = np.array(["S1"] * n)
    ds = SpatialDataset(
        log1p, coords, samples, k_neighbors=6, recon_target=counts,
    )
    assert ds.niche_count_target is not None
    nct = ds.niche_count_target
    # count-scale: the weighted mean of raw neighbor counts has a per-row total on the
    # count scale (order ~ mean raw total), NOT a tiny log1p sum. Compare magnitudes:
    raw_total = counts.sum(1).mean()
    nct_total = float(nct.sum(1).mean())
    log1p_total = float(np.log1p(counts).sum(1).mean())
    # nct total should be near the raw per-cell transcript total, far above the log1p sum.
    assert nct_total > 2.0 * log1p_total
    assert abs(nct_total - raw_total) / raw_total < 0.6
    # DM term must respond to a constant rescaling of the RAW counts (10x) even though
    # the log1p neighborhood is unchanged.
    ds10 = SpatialDataset(
        log1p, coords, samples, k_neighbors=6, recon_target=counts * 10.0,
    )
    logits = torch.zeros(n, g)
    la = torch.zeros(g)
    dm1 = dirichlet_multinomial_nll(ds.niche_count_target, logits, la)
    dm10 = dirichlet_multinomial_nll(ds10.niche_count_target, logits, la)
    assert not torch.allclose(dm1, dm10)


def test_dirmult_niche_without_raw_counts_raises(tmp_path):
    """Selecting a DM niche mode without a count cell mode (no raw counts) must fail
    loudly rather than silently feeding a log1p mean to the multinomial."""
    a = _toy()
    tc = TrainConfig(num_epochs=1, batch_size=32, k_neighbors=5, normalize=True, log1p=True)
    # cell_recon="mse" -> no raw-count target -> DM niche cannot build its count target.
    mc = _mc_opt(a, cell_recon="mse", niche_recon="mse_dirmult")
    with pytest.raises(ValueError, match="Dirichlet-multinomial|count"):
        Trainer(tc).fit(a, tmp_path, model_config=mc)


def test_count_losses_fp32_identical_when_amp_off():
    """The fp32-wrapped count losses must be numerically identical to a plain fp32 call
    (AMP off is the default, so the wrapper must be a no-op there)."""
    x = torch.randint(0, 8, (16, 12)).float()
    cr = torch.randn(16, 12)
    lt = torch.randn(12) * 0.1
    la = torch.randn(12) * 0.1
    tgt = torch.rand(16, 12) * 5.0
    # Recompute the NB NLL by hand in fp32 and compare.
    lib = x.sum(1, keepdim=True)
    mu = torch.softmax(cr, 1) * lib
    theta = lt.exp()
    lg = torch.log(theta + mu + 1e-8)
    res = (theta * (torch.log(theta + 1e-8) - lg) + x * (torch.log(mu + 1e-8) - lg)
           + torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0))
    manual_nb = -res.sum(1).mean()
    assert torch.allclose(nb_nll(x, cr, lt), manual_nb, atol=1e-6)
    # Poisson, BCE, DM finite + differentiable through the fp32 wrapper.
    for fn, args in [
        (poisson_nll, (x, cr)),
        (bernoulli_detection_bce, (x, cr.clone().requires_grad_(True))),
    ]:
        v = fn(*args)
        assert torch.isfinite(v)
    dm = dirichlet_multinomial_nll(tgt, cr.clone().requires_grad_(True), la.clone().requires_grad_(True))
    assert torch.isfinite(dm)
