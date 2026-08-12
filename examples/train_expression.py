#!/usr/bin/env python3
"""Minimal end-to-end training example for nicheverse.

Trains the hierarchical VQ-VAE (cell codebook + neighborhood codebook) on the
bundled real Xenium RCC TMA core (~7.8k cells, 366-gene panel) and prints a
short verification of the codes written back to the AnnData.

    python examples/train_expression.py

The reference configuration is num_epochs=300, batch_size=32768, seed=9,
spatial_graph='knn_radius' at radius=50 um, k_neighbors=20, lr=3e-4,
encoder_type='mlp_deep', quantizer_type='vq', neighborhood_aggregation=
'weighted_mean' (all ModelConfig/TrainConfig defaults). This demo keeps those
defaults but shrinks num_epochs (300 -> 20) and batch_size (32768 -> 2048) so it
finishes in a couple of minutes on the tiny bundled core (batch 32768 exceeds the
~7.8k cells here); the other fields below are the reference defaults, passed explicitly
only for clarity. On sparse Xenium counts like this panel, per-gene numerical
embeddings ('mlp_plr') degenerate, so the simple MLP encoders are preferred.

Note: this bundled example is a single, fairly homogeneous tumor core, so only a
few cell codes are exercised; how many codes get used grows with the biological
diversity and scale of a real multi-sample cohort. See notebooks/01_quickstart on
the more diverse MERFISH dataset for a fuller codebook. The purpose here is to show
the end-to-end API on data that ships with the repo.
"""
from pathlib import Path

import numpy as np

import nicheverse as nv
from nicheverse import ModelConfig, TrainConfig
from nicheverse.training import train_model

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "xenium_rcc_core.h5ad"
CKPT = HERE / "runs" / "expression_demo"


def main() -> None:
    # read_spatial standardizes the AnnData: it guarantees obsm['spatial'] and
    # obs['sample_id']. adata.X is raw counts; train_model normalizes + log1p.
    adata = nv.read_spatial(DATA, sample_col="sample_id", spatial_key="spatial")
    print(f"loaded {adata.n_obs} cells x {adata.n_vars} genes, "
          f"{adata.obs['sample_id'].nunique()} sample(s)")

    mc = ModelConfig(
        input_dim=adata.n_vars,
        gene_names=tuple(adata.var_names.astype(str)),
        encoder_type="mlp_deep",  # recommended default encoder
        quantizer_type="vq",      # recommended default quantizer
        cell_num_embeddings=256,
        neighborhood_num_embeddings=32,
        cell_embedding_dim=64,
        neighborhood_embedding_dim=256,
    )
    tc = TrainConfig(
        num_epochs=20,            # default is 300; shrunk for a fast demo
        batch_size=2048,          # default is 32768; shrunk for the tiny demo core
        k_neighbors=20,           # reference default
        spatial_graph="knn_radius",  # reference default
        radius=50.0,              # reference default
        neighborhood_aggregation="weighted_mean",  # reference default
        save_best=False,
        seed=9,                   # default seed
    )

    model, adata = train_model(adata, CKPT, model_config=mc, train_config=tc,
                               sample_col="sample_id", device=None)

    c = adata.obs["cell_codebook_idx"].to_numpy()
    n = adata.obs["neighborhood_codebook_idx"].to_numpy()
    assert 0 <= c.min() and c.max() < mc.cell_num_embeddings
    assert 0 <= n.min() and n.max() < mc.neighborhood_num_embeddings
    assert "X_cell_embedding" in adata.obsm
    print(f"cell codes used: {len(np.unique(c))}/{mc.cell_num_embeddings}; "
          f"niches used: {len(np.unique(n))}/{mc.neighborhood_num_embeddings}")
    print(f"checkpoint written to: {CKPT}")
    print("outputs:", sorted(p.name for p in CKPT.glob('*')))
    print("DONE")


if __name__ == "__main__":
    main()
