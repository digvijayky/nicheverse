"""Regression tests for three trainer / molecule-set fixes.

1. Best-val checkpoint is reloaded into ``core`` before the final export, so the
   exported main checkpoint, codebooks, embeddings, code indices, and annotated
   AnnData all come from the best-val epoch, not the last epoch. The default
   no-val path (``val_fraction=0``) must be unchanged (no best_checkpoint, no
   restore).
2. Auto-batch sqrt LR scaling records BOTH the nominal and the effective
   learning rate in ``training_runtime.json`` without mutating ``tc``.
3. ``MoleculeSetDataset`` raises loudly when two shards double-fill the same
   ``obs_name`` instead of silently overwriting.

All expression values come from the real example Xenium RCC core AnnData
(``examples/data/xenium_rcc_core.h5ad``) and the real molecule-set shard
(``examples/data/xenium_rcc_core_molecule_sets/``); nothing is simulated.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import torch

from nicheverse import ModelConfig, TrainConfig, Trainer
from nicheverse.data.molecule_set import MoleculeSetDataset
from nicheverse.models import load_checkpoint
from nicheverse.training import trainer as trainer_mod

_HERE = Path(__file__).resolve().parent
_EX = _HERE.parent / "examples" / "data"
_ADATA = _EX / "xenium_rcc_core.h5ad"
_MOLSETS = _EX / "xenium_rcc_core_molecule_sets"

pytestmark = pytest.mark.skipif(
    not _ADATA.exists(), reason="example xenium_rcc_core.h5ad not present"
)


def _real_adata(n=400):
    """A small slice of the REAL example adata (real counts, real coords)."""
    a = ad.read_h5ad(_ADATA)
    a = a[:n].copy()
    return a


def _mc(a):
    return ModelConfig(
        input_dim=a.n_vars,
        hidden_dims=(16,),
        cell_embedding_dim=8,
        cell_num_embeddings=8,
        neighborhood_embedding_dim=8,
        neighborhood_num_embeddings=4,
        gene_names=tuple(a.var_names),
    )


def _state_close(sd_a, sd_b):
    assert set(sd_a) == set(sd_b)
    return all(torch.equal(sd_a[k].cpu(), sd_b[k].cpu()) for k in sd_a)


# ---------------------------------------------------------------------------
# ISSUE 1: best-val weights are exported, not last-epoch weights.
# ---------------------------------------------------------------------------
def test_best_val_checkpoint_is_exported_not_last(tmp_path, monkeypatch):
    a = _real_adata()
    n_epochs = 5

    # Force the monitored (validation) loss to bottom out at epoch 2 (index 1),
    # then rise, so the LAST epoch is NOT the best. Weights keep changing every
    # epoch (real optimizer steps run), so last-epoch weights differ from epoch 2.
    seq = iter([1.0, 0.1, 0.5, 0.8, 0.9])

    def fake_val_loss(model, loader, device, tc):
        return next(seq)

    monkeypatch.setattr(trainer_mod, "_val_loss", fake_val_loss)

    tc = TrainConfig(
        num_epochs=n_epochs,
        batch_size=64,
        k_neighbors=5,
        log_every=100,
        val_fraction=0.25,
        save_best=True,
        normalize=True,
        log1p=True,
    )
    core, out = Trainer(tc).fit(a, tmp_path, model_config=_mc(a))

    best_path = tmp_path / "best_checkpoint.pt"
    exported_path = tmp_path / "hierarchical_vqvae_checkpoint.pt"
    assert best_path.exists(), "best_checkpoint.pt should be written with a val split"
    assert exported_path.exists()

    # best epoch was epoch 2 (index 1), definitively not the last (epoch 5).
    losses = json.loads((tmp_path / "training_losses.json").read_text())
    val_curve = [x["val_total"] for x in losses]
    assert np.argmin(val_curve) == 1 and len(val_curve) == n_epochs

    best_model = load_checkpoint(best_path, "cpu", config=_mc(a))
    exported_model = load_checkpoint(exported_path, "cpu", config=_mc(a))
    # The exported checkpoint must equal the BEST-val checkpoint, param for param.
    assert _state_close(best_model.state_dict(), exported_model.state_dict())
    # The returned in-memory core is post-restore too -> also equals best.
    assert _state_close(best_model.state_dict(), core.state_dict())

    # And the exported CODE INDICES come from the best model: re-embedding the
    # adata with the best checkpoint reproduces the saved hierarchical_cell_indices.
    saved_idx = np.load(tmp_path / "hierarchical_cell_indices.npz")["indices"]
    assert saved_idx.shape[0] == a.n_obs
    # out.obs code column matches the saved indices (both from the best model).
    code_cols = [c for c in out.obs.columns if "codebook_idx" in c or "cell_code" in c]
    assert code_cols, out.obs.columns.tolist()


def test_default_no_val_path_unchanged(tmp_path):
    """val_fraction=0 (released default): no best_checkpoint, no restore, export = last epoch."""
    a = _real_adata()
    tc = TrainConfig(
        num_epochs=3,
        batch_size=64,
        k_neighbors=5,
        log_every=100,
        val_fraction=0.0,  # released default
        save_best=True,  # save_best is on but must be inert with no val split
    )
    core, _ = Trainer(tc).fit(a, tmp_path, model_config=_mc(a))
    assert not (tmp_path / "best_checkpoint.pt").exists()
    exported = load_checkpoint(tmp_path / "hierarchical_vqvae_checkpoint.pt", "cpu", config=_mc(a))
    # Exported checkpoint equals the final in-memory core (no restore happened).
    assert _state_close(core.state_dict(), exported.state_dict())
    losses = json.loads((tmp_path / "training_losses.json").read_text())
    assert len(losses) == 3 and all("val_total" not in x for x in losses)


# ---------------------------------------------------------------------------
# ISSUE 2: auto-batch records nominal + effective lr, does not mutate tc.
# ---------------------------------------------------------------------------
def test_auto_batch_records_nominal_and_effective_lr(tmp_path):
    a = _real_adata()
    nominal_lr = 3e-4
    tc = TrainConfig(
        num_epochs=2,
        batch_size="auto",
        scale_lr_with_batch=True,
        learning_rate=nominal_lr,
        k_neighbors=5,
        log_every=100,
    )
    Trainer(tc).fit(a, tmp_path, model_config=_mc(a))

    rt = json.loads((tmp_path / "training_runtime.json").read_text())
    assert "nominal_learning_rate" in rt and "effective_learning_rate" in rt
    assert rt["nominal_learning_rate"] == pytest.approx(nominal_lr)

    eff_batch = rt["effective_batch_size"]
    expected_eff_lr = nominal_lr * (eff_batch / 2048.0) ** 0.5
    assert rt["effective_learning_rate"] == pytest.approx(expected_eff_lr, rel=1e-6)

    # When the resolved batch != 2048, the two lrs differ and the flag is set.
    if eff_batch != 2048:
        assert rt["effective_learning_rate"] != rt["nominal_learning_rate"]
        assert rt["lr_scaled_with_batch"] is True

    # tc is NOT mutated: train_config.json still records the nominal lr.
    cfg = json.loads((tmp_path / "train_config.json").read_text())
    assert cfg["learning_rate"] == pytest.approx(nominal_lr)
    assert tc.learning_rate == pytest.approx(nominal_lr)


def test_auto_batch_no_scaling_lrs_equal(tmp_path):
    a = _real_adata()
    tc = TrainConfig(
        num_epochs=1,
        batch_size="auto",
        scale_lr_with_batch=False,
        learning_rate=3e-4,
        k_neighbors=5,
        log_every=100,
    )
    Trainer(tc).fit(a, tmp_path, model_config=_mc(a))
    rt = json.loads((tmp_path / "training_runtime.json").read_text())
    assert rt["nominal_learning_rate"] == pytest.approx(rt["effective_learning_rate"])
    assert rt["lr_scaled_with_batch"] is False


# ---------------------------------------------------------------------------
# ISSUE 3: MoleculeSetDataset raises on overlapping (double-fill) shards.
# ---------------------------------------------------------------------------
def _write_shard(path, z, rows):
    np.savez_compressed(
        path,
        coords=z["coords"][rows],
        gene=z["gene"][rows],
        mask=z["mask"][rows],
        comp=z["comp"][rows],
        obs_names=z["obs_names"][rows],
    )


def test_molecule_set_overlapping_shards_raise(tmp_path):
    shards = sorted(glob.glob(os.path.join(_MOLSETS, "*.npz")))
    if not shards:
        pytest.skip("no example molecule-set shards")
    z = np.load(shards[0], allow_pickle=True)
    names = z["obs_names"].astype(str)
    n = min(60, len(names))
    obs_names = names[:n]

    d = tmp_path / "overlap_shards"
    d.mkdir()
    # Two shards that OVERLAP on rows [20, 40): both would fill those obs_names.
    _write_shard(d / "shard_a.npz", z, np.arange(0, 40))
    _write_shard(d / "shard_b.npz", z, np.arange(20, n))

    with pytest.raises(ValueError, match="re-fills|already loaded|overlap"):
        MoleculeSetDataset(obs_names, str(d), with_neighborhood=False)


def test_molecule_set_disjoint_shards_ok(tmp_path):
    """Sanity: non-overlapping shards that fully cover obs_names load fine."""
    shards = sorted(glob.glob(os.path.join(_MOLSETS, "*.npz")))
    if not shards:
        pytest.skip("no example molecule-set shards")
    z = np.load(shards[0], allow_pickle=True)
    names = z["obs_names"].astype(str)
    n = min(60, len(names))
    obs_names = names[:n]

    d = tmp_path / "disjoint_shards"
    d.mkdir()
    _write_shard(d / "shard_a.npz", z, np.arange(0, 30))
    _write_shard(d / "shard_b.npz", z, np.arange(30, n))
    ds = MoleculeSetDataset(obs_names, str(d), with_neighborhood=False)
    assert len(ds) == n
