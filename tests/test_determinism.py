"""Determinism: running predict_codes twice on the same input must produce identical outputs."""

from __future__ import annotations

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from nicheverse.models import ModelConfig
from nicheverse.training import TrainConfig, predict_codes, train_model


def _toy_adata(n=120, g=24, seed=0):
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(rng.poisson(1.0, size=(n, g)).astype(np.float32))
    sids = np.array(["S1"] * (n // 2) + ["S2"] * (n - n // 2))
    xy = np.column_stack([rng.uniform(0, 500, n), rng.uniform(0, 500, n)])
    a = ad.AnnData(X=X)
    a.var_names = [f"g{i}" for i in range(g)]
    a.obs["sample_id"] = sids
    a.obsm["spatial"] = xy
    return a


def test_predict_codes_bit_stable_across_two_runs(tmp_path):
    train_ad = _toy_adata(n=120, g=16, seed=0)
    mc = ModelConfig(
        input_dim=train_ad.X.shape[1],
        hidden_dims=(16, 8),
        cell_embedding_dim=8,
        cell_num_embeddings=8,
        neighborhood_embedding_dim=16,
        neighborhood_num_embeddings=4,
        gene_names=tuple(train_ad.var_names),
    )
    tc = TrainConfig(num_epochs=2, batch_size=32, k_neighbors=4, log_every=100, deterministic=True)
    train_model(train_ad, tmp_path, model_config=mc, train_config=tc)

    pred_ad = _toy_adata(seed=99)
    first = predict_codes(
        pred_ad.copy(),
        tmp_path / "hierarchical_vqvae_checkpoint.pt",
        k_neighbors=4,
        batch_size=32,
        deterministic=True,
    )
    second = predict_codes(
        pred_ad.copy(),
        tmp_path / "hierarchical_vqvae_checkpoint.pt",
        k_neighbors=4,
        batch_size=32,
        deterministic=True,
    )
    np.testing.assert_array_equal(
        first.obs["cell_codebook_idx"].to_numpy(),
        second.obs["cell_codebook_idx"].to_numpy(),
    )
    np.testing.assert_array_equal(
        first.obs["neighborhood_codebook_idx"].to_numpy(),
        second.obs["neighborhood_codebook_idx"].to_numpy(),
    )
    np.testing.assert_allclose(
        first.obsm["X_cell_embedding"],
        second.obsm["X_cell_embedding"],
        rtol=0,
        atol=0,
    )


def test_seed_everything_resets_torch_rng():
    from nicheverse.utils import seed_everything

    seed_everything(7)
    a = torch.randn(8)
    seed_everything(7)
    b = torch.randn(8)
    assert torch.allclose(a, b)
