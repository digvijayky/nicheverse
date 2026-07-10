# Inputs and outputs

## Supported Xenium output

`load_xenium_run` reads one Xenium output directory and returns an AnnData. The directory must contain:

```
cell_feature_matrix.h5      preferred, 10x HDF5 format
cell_feature_matrix/        fallback, 10x mtx format with matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz
cells.parquet               required, with x_centroid, y_centroid in microns
cells.csv.gz                fallback for older Xenium outputs
```

We do not require `transcripts.parquet`, `cell_boundaries.parquet`, or the morphology images for this package. Downstream visualization scripts can read them separately.

## AnnData layout after preprocessing

```
adata.X                     (n_cells, n_genes) raw counts (uint32 sparse)
adata.var_names             gene symbols, controls dropped by default
adata.obs['sample_id']      string label per cell
adata.obs['x_centroid']     micrometers
adata.obs['y_centroid']     micrometers
adata.obs['cell_area']      from Xenium cells.parquet, micrometers squared
adata.obs['nucleus_area']
adata.obs['transcript_counts']
adata.obsm['spatial']       (n_cells, 2) view of (x_centroid, y_centroid)
adata.obs_names             "cell_id__sample_id" so multi sample cohorts are disambiguated
```

## AnnData layout after train or predict

The above plus:

```
adata.obs['cell_codebook_idx']            integer 0..K_c-1
adata.obs['neighborhood_codebook_idx']    integer 0..K_n-1
adata.obsm['X_cell_embedding']            (n_cells, d_c) pre quantization latent
adata.obsm['X_neighborhood_embedding']    (n_cells, d_n)
```

## Multi sample handling

When you pass multiple Xenium run directories to `load_xenium_cohort`, the loader takes the intersection of `var_names` across runs (warns if it is restricting). Cell barcodes are namespaced as `cell_id__sample_id` so that two cells with the same Xenium barcode in different runs do not collide.

The k-NN neighborhood graph in `SpatialDataset` is built per sample. No edges cross slides. This is what you want: a cell next to itself in a different slide is not actually a spatial neighbor.

## Loading data from h5ad you preprocessed yourself

If you already have a preprocessed AnnData and you want to skip the `preprocess` step, ensure it satisfies:

```
adata.obsm['spatial'].shape == (adata.n_obs, 2)
adata.obs['sample_id']    string, used to scope per sample k-NN
adata.X                   raw INTEGER counts (normalized for the encoder input at train / predict time; the raw counts are kept as the reconstruction target)
```

The count-native defaults (`cell_recon="nb"`, `niche_recon="mse_dirmult"`) require raw integer counts: passing an already normalized or log matrix raises. If you must train from a normalized matrix, switch to the pure-MSE path (`cell_recon="mse"`, `niche_recon="mse"`) and pass `normalize=False, log1p=False` to `train_model` or `predict_codes` to skip re normalization.

## Filtering controls

`drop_controls=True` (default) drops any feature whose name matches:

    Control | BLANK | Blank | Codeword | Unassigned

Pass `--keep-controls` on the CLI or `drop_controls=False` in Python to retain them, which is rarely useful.

## Outputs from training

`./checkpoint/` contains everything needed to reproduce assignments:

```
hierarchical_vqvae_checkpoint.pt   PyTorch checkpoint with embedded ModelConfig + gene_names
hierarchical_vqvae_checkpoint.json Plain text config
cell_codebook.npz                  (K_c, d_c) centroid matrix
neighborhood_codebook.npz          (K_n, d_n)
hierarchical_cell_embeddings.npz   (n_cells, d_c) continuous latents
hierarchical_cell_indices.npz      (n_cells,) discrete code 0..K_c-1
hierarchical_neighborhood_embeddings.npz
hierarchical_neighborhood_indices.npz
training_losses.json
training_runtime.json              wall time, mean epoch seconds, cells / second, peak GPU memory
train_config.json                  the TrainConfig used for the run
adata_with_hierarchical_embeddings.h5ad
training_losses.pdf
codebook_usage.pdf
per_sample_spatial/spatial_<sample>.pdf
env_snapshot.json                  Python, torch, CUDA, package versions for the run
```

## Reading codebooks separately

```python
import numpy as np
cell_codebook = np.load("./checkpoint/cell_codebook.npz")["codebook"]   # (K_c, d_c)
neigh_codebook = np.load("./checkpoint/neighborhood_codebook.npz")["codebook"]
```

You can hierarchical cluster `cell_codebook` to derive a coarse-to-fine code grouping for visualization.

## File size expectations

For the 173 sample 5.66M cell cohort:

```
hierarchical_vqvae_checkpoint.pt           3.5 MB
cell_codebook.npz                          60 KB
neighborhood_codebook.npz                  30 KB
hierarchical_cell_embeddings.npz           1.3 GB (compressed float32 64 dim)
hierarchical_neighborhood_embeddings.npz   5.4 GB (256 dim)
hierarchical_cell_indices.npz              8.7 MB
hierarchical_neighborhood_indices.npz      5.2 MB
adata_with_hierarchical_embeddings.h5ad    7.5 GB
```

For a typical single sample run on a Xenium slide (100K cells) divide by roughly 50.
