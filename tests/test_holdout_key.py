"""TrainConfig.holdout_key: an obs boolean column defines the validation set
(held-out-region protocol). Default None keeps the released random split."""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from nicheverse import ModelConfig, TrainConfig, Trainer

_ADATA = Path(__file__).resolve().parent.parent / "examples" / "data" / "xenium_rcc_core.h5ad"
pytestmark = pytest.mark.skipif(not _ADATA.exists(), reason="example adata not present")


def _adata(n=400):
    return ad.read_h5ad(_ADATA)[:n].copy()


def _mc(a):
    return ModelConfig(input_dim=a.n_vars, hidden_dims=(16,), cell_embedding_dim=8,
                       cell_num_embeddings=8, neighborhood_embedding_dim=8,
                       neighborhood_num_embeddings=4, gene_names=tuple(a.var_names))


def test_holdout_key_defines_val_split(tmp_path):
    a = _adata()
    xy = np.asarray(a.obsm["spatial"])
    a.obs["_held"] = xy[:, 0] > np.quantile(xy[:, 0], 0.8)   # a contiguous block
    n_held = int(a.obs["_held"].sum())
    tc = TrainConfig(num_epochs=2, batch_size=64, k_neighbors=5, log_every=100,
                     holdout_key="_held", save_best=False)
    _, out = Trainer(tc).fit(a, tmp_path, model_config=_mc(a))
    losses = json.loads((tmp_path / "training_losses.json").read_text())
    assert all("val_total" in x for x in losses)
    assert out.n_obs == a.n_obs and "cell_codebook_idx" in out.obs   # held-out cells still coded
    assert 0 < n_held < a.n_obs


def test_holdout_key_missing_or_degenerate_raises(tmp_path):
    a = _adata()
    with pytest.raises(ValueError):
        Trainer(TrainConfig(num_epochs=1, batch_size=64, k_neighbors=5,
                            holdout_key="nope")).fit(a, tmp_path, model_config=_mc(a))
    a.obs["_all"] = True
    with pytest.raises(ValueError):
        Trainer(TrainConfig(num_epochs=1, batch_size=64, k_neighbors=5,
                            holdout_key="_all")).fit(a, tmp_path, model_config=_mc(a))


def test_default_unchanged():
    assert TrainConfig().holdout_key is None
