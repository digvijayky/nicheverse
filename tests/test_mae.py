"""Test MAE masked-gene pretraining and encoder transfer."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import torch

import nicheverse as nv
from nicheverse.models import HierarchicalVQVAE, ModelConfig


def test_mae_pretrain_transfers_into_model():
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=sp.csr_matrix(rng.poisson(1.0, size=(120, 20)).astype("float32")))
    a.var_names = [f"g{i}" for i in range(20)]
    enc = nv.mae_pretrain(
        a, encoder_type="mlp", hidden=(16,), embedding_dim=8, num_epochs=2, batch_size=32
    )
    assert enc(torch.randn(4, 20)).shape == (4, 8)
    m = HierarchicalVQVAE(
        ModelConfig(
            input_dim=20,
            encoder_type="mlp",
            hidden_dims=(16,),
            cell_embedding_dim=8,
            cell_num_embeddings=8,
            neighborhood_embedding_dim=8,
            neighborhood_num_embeddings=4,
            gene_names=tuple(a.var_names),
        )
    )
    missing, unexpected = m.cell_encoder.load_state_dict(enc.state_dict(), strict=False)
    assert not missing and not unexpected


def test_mae_bad_ratio_rejected():
    a = ad.AnnData(X=sp.csr_matrix(np.ones((10, 4), dtype="float32")))
    a.var_names = [f"g{i}" for i in range(4)]
    with pytest.raises(ValueError, match="mask_ratio"):
        nv.mae_pretrain(a, mask_ratio=1.5)
