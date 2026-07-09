#!/usr/bin/env python3
"""Apply a trained nicheverse checkpoint to held-out data.

Trains on 80 percent of the cells of the bundled Xenium RCC core, then assigns
the learned cell and neighborhood codes to the held-out 20 percent with
predict_codes, without retraining. This is the pattern for annotating a new
sample with an existing codebook.

    python examples/apply_codes.py

The neighborhood-graph arguments passed to predict_codes MUST match the ones
used at training time so the neighborhood codes stay comparable.
"""
from pathlib import Path

import numpy as np

import nicheverse as nv
from nicheverse import ModelConfig, TrainConfig
from nicheverse.training import predict_codes, train_model

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "xenium_rcc_core.h5ad"
CKPT = HERE / "runs" / "apply_demo"

# graph settings shared by train and predict (must match)
GRAPH = dict(k_neighbors=20, neighborhood_aggregation="weighted_mean",
             spatial_graph="knn_radius", radius=50.0)


def main() -> None:
    adata = nv.read_spatial(DATA, sample_col="sample_id", spatial_key="spatial")
    rng = np.random.default_rng(0)
    held = rng.random(adata.n_obs) < 0.2
    train_ad = adata[~held].copy()
    new_ad = adata[held].copy()
    print(f"{train_ad.n_obs} training cells, {new_ad.n_obs} held-out cells")

    mc = ModelConfig(
        input_dim=train_ad.n_vars,
        gene_names=tuple(train_ad.var_names.astype(str)),
        encoder_type="mlp_plr",
        cell_num_embeddings=256,
        neighborhood_num_embeddings=32,
    )
    tc = TrainConfig(num_epochs=20, batch_size=2048, save_best=False, seed=49, **GRAPH)
    train_model(train_ad, CKPT, model_config=mc, train_config=tc, sample_col="sample_id")

    # apply the trained codebook to the held-out cells (no retraining)
    ckpt_pt = CKPT / "hierarchical_vqvae_checkpoint.pt"
    coded = predict_codes(new_ad, ckpt_pt, sample_col="sample_id", **GRAPH)

    c = coded.obs["cell_codebook_idx"].to_numpy()
    n = coded.obs["neighborhood_codebook_idx"].to_numpy()
    assert 0 <= c.min() and c.max() < mc.cell_num_embeddings
    assert "X_cell_embedding" in coded.obsm
    print(f"assigned codes to {coded.n_obs} held-out cells; "
          f"{len(np.unique(c))} distinct cell codes, {len(np.unique(n))} niches")
    print("DONE")


if __name__ == "__main__":
    main()
