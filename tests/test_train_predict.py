import anndata as ad
import numpy as np
import scipy.sparse as sp

from nicheverse.models import ModelConfig
from nicheverse.training import TrainConfig, predict_codes, train_model


def _toy_adata(n=200, g=30, seed=0):
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(rng.poisson(1.0, size=(n, g)).astype(np.float32))
    sids = np.array(["S1"] * (n // 2) + ["S2"] * (n - n // 2))
    xy = np.column_stack([rng.uniform(0, 500, n), rng.uniform(0, 500, n)])
    a = ad.AnnData(X=X)
    a.var_names = [f"g{i}" for i in range(g)]
    a.obs["sample_id"] = sids
    a.obsm["spatial"] = xy
    return a


def test_train_then_predict(tmp_path):
    a = _toy_adata()
    mc = ModelConfig(
        input_dim=a.X.shape[1],
        hidden_dims=(32, 16),
        cell_embedding_dim=8,
        cell_num_embeddings=16,
        neighborhood_embedding_dim=16,
        neighborhood_num_embeddings=8,
        gene_names=tuple(a.var_names),
    )
    tc = TrainConfig(num_epochs=2, batch_size=64, k_neighbors=5, log_every=10)
    model, trained = train_model(a, tmp_path, model_config=mc, train_config=tc)
    assert "cell_codebook_idx" in trained.obs
    assert (tmp_path / "hierarchical_vqvae_checkpoint.pt").exists()

    b = _toy_adata(seed=99)
    out = tmp_path / "annotated.h5ad"
    annotated = predict_codes(
        b,
        tmp_path / "hierarchical_vqvae_checkpoint.pt",
        output_path=out,
        k_neighbors=5,
        batch_size=64,
    )
    assert out.exists()
    assert annotated.obs["cell_codebook_idx"].max() < 16
    assert annotated.obs["neighborhood_codebook_idx"].max() < 8


def test_predict_rejects_gene_mismatch(tmp_path):
    a = _toy_adata()
    mc = ModelConfig(
        input_dim=a.X.shape[1],
        hidden_dims=(32, 16),
        cell_embedding_dim=8,
        cell_num_embeddings=8,
        neighborhood_embedding_dim=16,
        neighborhood_num_embeddings=4,
        gene_names=tuple(a.var_names),
    )
    tc = TrainConfig(num_epochs=1, batch_size=64, k_neighbors=5, log_every=100)
    train_model(a, tmp_path, model_config=mc, train_config=tc)
    b = _toy_adata()
    b = b[:, b.var_names[:-1]].copy()
    import pytest

    with pytest.raises(ValueError, match="missing"):
        predict_codes(
            b, tmp_path / "hierarchical_vqvae_checkpoint.pt", k_neighbors=5, batch_size=64
        )
