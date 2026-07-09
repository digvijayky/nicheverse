# Quickstart

A complete pipeline in three commands: preprocess, train, predict.

## 1. Preprocess Xenium output directories

`preprocess` reads native Xenium output directories, joins `cell_feature_matrix.h5` with `cells.parquet`, drops control / blank / unassigned probes by default, and writes a single AnnData. Multi sample cohorts are merged on the intersection of genes.

```bash
nicheverse preprocess \
    --run-dir /path/to/run_A /path/to/run_B /path/to/run_C \
    --sample-id A B C \
    --output ./preprocessed/cohort.h5ad
```

For large cohorts use a manifest CSV with two columns `run_dir,sample_id`:

```bash
nicheverse preprocess --manifest ./manifest.csv --output ./preprocessed/cohort.h5ad
```

The resulting AnnData has `obs['sample_id']`, `obsm['spatial']` in microns, and barcodes namespaced as `cell_id__sample_id`.

## 2. Train

```bash
nicheverse train \
    --input ./preprocessed/cohort.h5ad \
    --checkpoint-dir ./checkpoint \
    --num-epochs 300 \
    --cell-codebook-size 256 \
    --neighborhood-codebook-size 32 \
    --k-neighbors 20
```

The checkpoint directory will receive:

```
hierarchical_vqvae_checkpoint.pt    PyTorch state plus embedded ModelConfig
hierarchical_vqvae_checkpoint.json  Human readable config
cell_codebook.npz                   Cell centroids
neighborhood_codebook.npz           Niche centroids
hierarchical_cell_embeddings.npz    Per cell continuous latent
hierarchical_cell_indices.npz       Per cell hard code assignment
hierarchical_neighborhood_embeddings.npz
hierarchical_neighborhood_indices.npz
training_losses.json
adata_with_hierarchical_embeddings.h5ad
training_losses.pdf
codebook_usage.pdf
per_sample_spatial/                 One PDF per sample
env_snapshot.json                   Python, torch, CUDA, package versions
```

## 3. Predict on new data

```bash
nicheverse preprocess --run-dir /path/to/new_run --output ./new.h5ad
nicheverse predict \
    --input ./new.h5ad \
    --checkpoint ./checkpoint/hierarchical_vqvae_checkpoint.pt \
    --output ./new_annotated.h5ad
```

After predict, `new_annotated.h5ad` has `obs['cell_codebook_idx']` (0..K_c-1), `obs['neighborhood_codebook_idx']` (0..K_n-1), `obsm['X_cell_embedding']`, and `obsm['X_neighborhood_embedding']`.

## Python API

The CLI is a thin wrapper around the Python API:

```python
import scanpy as sc
from nicheverse import (
    load_xenium_cohort, train_model, predict_codes,
    ModelConfig, TrainConfig, seed_everything,
)

seed_everything(49)
adata = load_xenium_cohort(["./run_A", "./run_B", "./run_C"])

mc = ModelConfig(
    input_dim=adata.X.shape[1],
    cell_num_embeddings=256,
    neighborhood_num_embeddings=32,
    gene_names=tuple(adata.var_names),
)
tc = TrainConfig(num_epochs=300, k_neighbors=20)
model, adata = train_model(adata, "./checkpoint", model_config=mc, train_config=tc)

new = load_xenium_cohort(["./run_D"])
annotated = predict_codes(
    new,
    "./checkpoint/hierarchical_vqvae_checkpoint.pt",
    output_path="./run_D_annotated.h5ad",
)
```

## What comes next

Read [Architecture and method](method.md) to understand how the model works, [Choosing hyperparameters](hyperparameters.md) for guidance on codebook sizes and neighborhood parameters, and [Inputs and outputs](io.md) for the loader conventions and the columns written back to your AnnData. The full [API reference](../api.md) documents every public function.
