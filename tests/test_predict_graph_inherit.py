"""predict_codes must inherit the training-time neighborhood graph.

Regression for the predict/train graph-mismatch bug: predict_codes used to
default to spatial_graph="knn" and ignore the graph the checkpoint was trained
with, so neighborhood codes on new data were silently built with a different
graph than training. It now inherits spatial_graph / radius / k_neighbors /
neighborhood_aggregation from the checkpoint's train_config.json when they are
not passed explicitly, while an explicit value still overrides.
"""

import anndata as ad
import numpy as np
import scipy.sparse as sp

from nicheverse.models import ModelConfig
from nicheverse.training import TrainConfig, predict_codes, train_model


def _toy_adata(n=300, g=30, seed=0, spread=1000.0):
    # A wide spatial spread so many k-NN edges exceed the 50 um radius; that is
    # what makes knn and knn_radius build genuinely different neighborhoods.
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(rng.poisson(1.0, size=(n, g)).astype(np.float32))
    sids = np.array(["S1"] * (n // 2) + ["S2"] * (n - n // 2))
    xy = np.column_stack([rng.uniform(0, spread, n), rng.uniform(0, spread, n)])
    a = ad.AnnData(X=X)
    a.var_names = [f"g{i}" for i in range(g)]
    a.obs["sample_id"] = sids
    a.obsm["spatial"] = xy
    return a


def _train_knn_radius(tmp_path):
    a = _toy_adata(seed=0)
    mc = ModelConfig(
        input_dim=a.X.shape[1],
        hidden_dims=(32, 16),
        cell_embedding_dim=8,
        cell_num_embeddings=16,
        neighborhood_embedding_dim=16,
        neighborhood_num_embeddings=8,
        gene_names=tuple(a.var_names),
    )
    tc = TrainConfig(
        num_epochs=2,
        batch_size=64,
        k_neighbors=5,
        spatial_graph="knn_radius",
        radius=50.0,
        neighborhood_aggregation="weighted_mean",
        log_every=100,
    )
    train_model(a, tmp_path, model_config=mc, train_config=tc)
    return tmp_path / "hierarchical_vqvae_checkpoint.pt"


def test_predict_inherits_training_graph(tmp_path):
    ckpt = _train_knn_radius(tmp_path)
    new = _toy_adata(seed=99)

    # No graph args passed: must inherit knn_radius / radius=50 from the checkpoint.
    inherited = predict_codes(new.copy(), ckpt, batch_size=64)
    # Explicitly pass the same graph the model was trained with.
    explicit = predict_codes(
        new.copy(),
        ckpt,
        spatial_graph="knn_radius",
        radius=50.0,
        k_neighbors=5,
        neighborhood_aggregation="weighted_mean",
        batch_size=64,
    )
    # Wrong graph: plain knn (no radius cap) builds different neighborhoods.
    wrong = predict_codes(new.copy(), ckpt, spatial_graph="knn", batch_size=64)

    inh_n = inherited.obs["neighborhood_codebook_idx"].to_numpy()
    exp_n = explicit.obs["neighborhood_codebook_idx"].to_numpy()
    wrong_n = wrong.obs["neighborhood_codebook_idx"].to_numpy()

    # Inheriting reproduces the explicit-correct-graph result exactly.
    assert np.array_equal(inh_n, exp_n), "inherited graph did not match explicit knn_radius,r=50"
    # Cell codes are graph-independent and must match across all three.
    assert np.array_equal(
        inherited.obs["cell_codebook_idx"].to_numpy(),
        explicit.obs["cell_codebook_idx"].to_numpy(),
    )
    # The wrong graph produces different neighborhood codes on at least some cells.
    assert not np.array_equal(inh_n, wrong_n), "knn and knn_radius gave identical neighborhood codes"


def test_predict_explicit_overrides_inheritance(tmp_path):
    # An explicitly-passed graph must win even when it disagrees with training.
    ckpt = _train_knn_radius(tmp_path)
    new = _toy_adata(seed=99)

    default_inherited = predict_codes(new.copy(), ckpt, batch_size=64)
    forced_knn = predict_codes(new.copy(), ckpt, spatial_graph="knn", batch_size=64)

    assert not np.array_equal(
        default_inherited.obs["neighborhood_codebook_idx"].to_numpy(),
        forced_knn.obs["neighborhood_codebook_idx"].to_numpy(),
    ), "explicit spatial_graph=knn did not override the inherited knn_radius"
