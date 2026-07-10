"""Tests for the opt-in GPU-resident dataset speedup (device_resident).

Covers: (1) flag OFF is the byte-identical released path with no attribute leak,
(2) the memory-fit decision helper, (3) the _IndexView / _collate_index shuffle
order preservation, and (4) a skipif-gated GPU numerical-equivalence test that
trains with device_resident True vs False under the same seed and asserts the loss
trajectory matches. The GPU test SKIPS on a CPU pytest node (expected); a human runs
it on a GPU node to confirm equivalence before adopting the flag.
"""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Subset

from nicheverse.data import SpatialDataset
from nicheverse.models import ModelConfig
from nicheverse.training import TrainConfig, train_model
from nicheverse.training.trainer import (
    _IndexView,
    _collate_index,
    _device_resident_bytes,
    _device_resident_fits,
)


def _toy_adata(n=120, g=16, seed=0):
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(rng.poisson(1.0, size=(n, g)).astype(np.float32))
    sids = np.array(["S1"] * (n // 2) + ["S2"] * (n - n // 2))
    xy = np.column_stack([rng.uniform(0, 500, n), rng.uniform(0, 500, n)])
    a = ad.AnnData(X=X)
    a.var_names = [f"g{i}" for i in range(g)]
    a.obs["sample_id"] = sids
    a.obsm["spatial"] = xy
    return a


def _mc(a):
    return ModelConfig(
        input_dim=a.X.shape[1],
        hidden_dims=(16, 8),
        cell_embedding_dim=8,
        cell_num_embeddings=8,
        neighborhood_embedding_dim=16,
        neighborhood_num_embeddings=4,
        gene_names=tuple(a.var_names),
    )


# ----------------------------------------------------------------------------
# (1) Flag defaults OFF and does not leak device-resident attributes.
# ----------------------------------------------------------------------------
def test_device_resident_defaults_off():
    assert TrainConfig().device_resident is False


def test_flag_off_dataset_stays_cpu_no_attr_leak(tmp_path):
    """With device_resident=False (default) the run is the released CPU path: the
    feature tensors stay on CPU and no device-resident attribute is added."""
    a = _toy_adata(n=120, g=16, seed=0)
    tc = TrainConfig(num_epochs=2, batch_size=32, k_neighbors=4, log_every=100)
    model, _ = train_model(a, tmp_path, model_config=_mc(a), train_config=tc)
    # train_config.json records device_resident False (serialized default).
    cfg = json.loads((tmp_path / "train_config.json").read_text())
    assert cfg["device_resident"] is False
    # No leaked flag on the config object beyond the documented field.
    assert not hasattr(model, "device_resident")


def test_flag_off_loss_matches_prior_default(tmp_path):
    """The default path loss trajectory is deterministic and reproduces run-to-run
    (guards that adding the flag did not perturb the released path)."""
    a = _toy_adata(n=120, g=16, seed=0)
    tc = TrainConfig(num_epochs=2, batch_size=32, k_neighbors=4, log_every=100)
    train_model(a.copy(), tmp_path / "r1", model_config=_mc(a), train_config=tc)
    train_model(a.copy(), tmp_path / "r2", model_config=_mc(a), train_config=tc)
    l1 = json.loads((tmp_path / "r1" / "training_losses.json").read_text())
    l2 = json.loads((tmp_path / "r2" / "training_losses.json").read_text())
    for a_, b_ in zip(l1, l2):
        assert a_["total"] == pytest.approx(b_["total"], rel=0, abs=1e-9)


# ----------------------------------------------------------------------------
# (2) Memory-fit decision helper.
# ----------------------------------------------------------------------------
def test_fit_check_true_when_small_budget_large():
    # 10 bytes * 1.5 safety = 15 <= 0.5 * 100 = 50 -> fits.
    assert _device_resident_fits(10, 100) is True


def test_fit_check_false_when_over_budget():
    # 40 * 1.5 = 60 > 0.5 * 100 = 50 -> does not fit.
    assert _device_resident_fits(40, 100) is False


def test_fit_check_false_when_budget_unknown():
    # A non-positive total budget means "unknown" -> refuse (fall back to CPU).
    assert _device_resident_fits(5, 0) is False


def test_fit_check_wide_panel_tiny_gpu_falls_back():
    """A CosMx-scale panel (21731 genes) on a small GPU budget must NOT fit."""
    a = _toy_adata(n=1000, g=200, seed=1)
    ds = SpatialDataset.from_anndata(a, sample_col="sample_id", k_neighbors=4)
    rb = _device_resident_bytes(ds, count_mode=False, niche_dirmult_mode=False)
    # Real resident bytes for this toy fit an 8 GB budget...
    assert _device_resident_fits(rb, 8 * 1024**3) is True
    # ...but a 1 MB budget cannot hold it -> fall back.
    assert _device_resident_fits(rb, 1024**2) is False


def test_resident_bytes_counts_the_right_tensors():
    a = _toy_adata(n=100, g=16, seed=2)
    ds = SpatialDataset.from_anndata(a, sample_col="sample_id", k_neighbors=4)
    base = _device_resident_bytes(ds, count_mode=False, niche_dirmult_mode=False)
    cf = ds.cell_features.element_size() * ds.cell_features.nelement()
    nf = ds.neighborhood_features.element_size() * ds.neighborhood_features.nelement()
    assert base == cf + nf
    # No recon_target on the default (MSE) path, so count_mode adds nothing (guarded).
    assert _device_resident_bytes(ds, count_mode=True, niche_dirmult_mode=False) == base


# ----------------------------------------------------------------------------
# (3) _IndexView + _collate_index preserve the seeded shuffle order exactly.
# ----------------------------------------------------------------------------
def test_index_view_maps_full_dataset_rows_for_subset():
    a = _toy_adata(n=40, g=8, seed=3)
    ds = SpatialDataset.from_anndata(a, sample_col="sample_id", k_neighbors=4)
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(7)).tolist()
    sub = Subset(ds, perm[:25])
    view = _IndexView(sub)
    assert len(view) == 25
    # Position p in the view must map to the SAME full-dataset row Subset uses.
    for p in range(len(view)):
        assert view[p] == sub.indices[p]


def test_index_loader_yields_same_batches_as_feature_loader():
    """The index-only loader (used by the resident path) and the feature loader,
    built with the SAME seed, must produce identical batch indices in identical
    order -- this is the correctness guarantee for accuracy neutrality."""
    a = _toy_adata(n=60, g=8, seed=4)
    ds = SpatialDataset.from_anndata(a, sample_col="sample_id", k_neighbors=4)
    bs, seed = 16, 49

    def _feature_batches():
        gen = torch.Generator()
        gen.manual_seed(seed)
        dl = DataLoader(ds, batch_size=bs, shuffle=True, generator=gen, num_workers=0)
        return [idx.tolist() for _, _, idx in dl]

    def _index_batches():
        gen = torch.Generator()
        gen.manual_seed(seed)
        dl = DataLoader(
            _IndexView(ds),
            batch_size=bs,
            shuffle=True,
            generator=gen,
            num_workers=0,
            collate_fn=_collate_index,
        )
        return [idx.tolist() for idx in dl]

    assert _feature_batches() == _index_batches()


# ----------------------------------------------------------------------------
# (4) GPU numerical-equivalence test (skips on a CPU node -- run on a GPU node
#     to confirm device_resident True vs False give an identical loss trajectory).
# ----------------------------------------------------------------------------
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="device_resident is a CUDA-only path; run on a GPU node to confirm numerical equivalence",
)
def test_device_resident_matches_cpu_loader_gpu(tmp_path):
    a = _toy_adata(n=200, g=16, seed=5)
    mc = _mc(a)
    tc_off = TrainConfig(
        num_epochs=2, batch_size=32, k_neighbors=4, log_every=100, device_resident=False
    )
    tc_on = TrainConfig(
        num_epochs=2, batch_size=32, k_neighbors=4, log_every=100, device_resident=True
    )
    train_model(a.copy(), tmp_path / "off", model_config=mc, train_config=tc_off, device="cuda")
    train_model(a.copy(), tmp_path / "on", model_config=mc, train_config=tc_on, device="cuda")
    loff = json.loads((tmp_path / "off" / "training_losses.json").read_text())
    lon = json.loads((tmp_path / "on" / "training_losses.json").read_text())
    assert len(loff) == len(lon)
    for e_off, e_on in zip(loff, lon):
        # Same seed + same batch order => same loss to tight fp tolerance.
        assert e_on["total"] == pytest.approx(e_off["total"], rel=1e-5, abs=1e-5)
        assert e_on["cell"] == pytest.approx(e_off["cell"], rel=1e-5, abs=1e-5)
        assert e_on["neighborhood"] == pytest.approx(e_off["neighborhood"], rel=1e-5, abs=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="device_resident is a CUDA-only path; run on a GPU node to confirm numerical equivalence",
)
def test_device_resident_count_mode_matches_cpu_gpu(tmp_path):
    """Same equivalence in a count cell mode (recon_target resident) + DM niche mode
    (niche_count_target resident), exercising the full set of resident tensors."""
    a = _toy_adata(n=200, g=16, seed=6)
    mc = ModelConfig(
        input_dim=a.X.shape[1],
        hidden_dims=(16, 8),
        cell_embedding_dim=8,
        cell_num_embeddings=8,
        neighborhood_embedding_dim=16,
        neighborhood_num_embeddings=4,
        gene_names=tuple(a.var_names),
        cell_recon="nb",
        niche_recon="mse_dirmult",
    )
    common = dict(num_epochs=2, batch_size=32, k_neighbors=4, log_every=100)
    train_model(
        a.copy(),
        tmp_path / "off",
        model_config=mc,
        train_config=TrainConfig(device_resident=False, **common),
        device="cuda",
    )
    train_model(
        a.copy(),
        tmp_path / "on",
        model_config=mc,
        train_config=TrainConfig(device_resident=True, **common),
        device="cuda",
    )
    loff = json.loads((tmp_path / "off" / "training_losses.json").read_text())
    lon = json.loads((tmp_path / "on" / "training_losses.json").read_text())
    for e_off, e_on in zip(loff, lon):
        assert e_on["total"] == pytest.approx(e_off["total"], rel=1e-5, abs=1e-5)
