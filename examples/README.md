# NICHEVERSE examples and quickstart

**NICHEVERSE** = **N**eighborhood-**I**nferred **C**ell type **H**i**E**rarchical annotation +
**VE**ctor-quantized **R**epresentations of **S**patial **E**cotypes (the letters spell
NICHE + VERSE; the second E is the E in VEctor).

NICHEVERSE learns two coupled vector-quantized codebooks from imaging spatial
transcriptomics: a **cell codebook** (recurrent transcriptional states) and a
**neighborhood codebook** (recurrent multicellular niches). A cross-attention block
lets each cell's state assignment be informed by its spatial context.

This folder holds a runnable quickstart, small example scripts, and tested
notebooks that train on a bundled **real** dataset out of the box.

## Install

```bash
# from the repo root (editable install picks up local edits with no reinstall)
pip install -e .
# with the optional dev / docs / test tooling
pip install -e ".[dev,doc,test]"
```

Python 3.10+ is required. A GPU is optional; PyTorch selects CUDA automatically
when available, otherwise everything runs on CPU.

## The three inputs NICHEVERSE needs

Any imaging spatial transcriptomics platform works (Xenium, MERFISH, CosMx,
seqFISH). Load your data into an [AnnData](https://anndata.readthedocs.io/) with:

| requirement | where | notes |
|---|---|---|
| raw counts | `adata.X` | cell-by-gene; `train_model` applies normalize + log1p |
| micron coordinates | `adata.obsm["spatial"]` | `(n_cells, 2)` in microns |
| a sample column | `adata.obs["sample_id"]` | the per-sample neighbor graph is built within each sample only, so cells from different samples never link |

`nicheverse.read_spatial(...)` standardizes an AnnData (or `.h5ad` path): it
builds `obsm["spatial"]` from `obs` x / y columns if needed and ensures the sample
column exists.

## Minimal train (three lines)

```python
import nicheverse as nv
from nicheverse import ModelConfig, TrainConfig

adata = nv.read_spatial("your_data.h5ad", sample_col="sample_id")
mc = ModelConfig(input_dim=adata.n_vars, gene_names=tuple(adata.var_names),
                 encoder_type="mlp_deep")          # recommended default encoder
model, adata = nv.train_model(adata, "checkpoint/", model_config=mc,
                              train_config=TrainConfig(num_epochs=300))
```

After training, `adata.obs["cell_codebook_idx"]` (0..255) and
`adata.obs["neighborhood_codebook_idx"]` (0..31) hold the code assignments, and
`adata.obsm["X_cell_embedding"]` / `adata.obsm["X_neighborhood_embedding"]` hold the
continuous embeddings for downstream UMAP, clustering, or differential analysis.
The canonical key names are available programmatically via `nv.anndata_keys()`.

## Where the codes land (checkpoint directory)

`train_model` writes everything into the checkpoint directory:

```
hierarchical_vqvae_checkpoint.pt        model state_dict + embedded ModelConfig
hierarchical_vqvae_checkpoint.json      human-readable config
cell_codebook.npz                       (n_cell_codes, cell_embedding_dim) centroids
neighborhood_codebook.npz               (n_niches, neighborhood_embedding_dim) centroids
hierarchical_cell_indices.npz           per-cell hard code assignment (key 'indices')
hierarchical_neighborhood_indices.npz   per-cell niche assignment
hierarchical_cell_embeddings.npz        per-cell continuous embedding (key 'embeddings')
hierarchical_neighborhood_embeddings.npz
training_losses.json                    per-epoch total / cell / neighborhood loss
training_runtime.json                   wall-clock + throughput metrics (see below)
train_config.json, env_snapshot.json    reproducibility record
adata_with_hierarchical_embeddings.h5ad AnnData with codes + embeddings attached
```

`training_runtime.json` records `total_seconds` / `total_hms`, `mean_epoch_seconds`,
`cells_per_second`, `iters_per_second`, `effective_batch_size`, `peak_gpu_gb`,
`device`, and the encoder / quantizer used. A `RUNTIME ...` line is also printed at
the end of training.

## Apply a trained model to new data

Assign the learned codes to a held-out sample without retraining. The
neighborhood-graph arguments must match the ones used at training time.

```python
from nicheverse import predict_codes

annotated = predict_codes(
    new_adata, "checkpoint/hierarchical_vqvae_checkpoint.pt",
    sample_col="sample_id",
    k_neighbors=20, neighborhood_aggregation="weighted_mean",
    spatial_graph="knn_radius", radius=50.0,   # match training
    output_path="new_annotated.h5ad",
)
```

## Which encoder / representation should I use?

The recommendations below reflect an internal 26-variant benchmark on RCC
Xenium (cell codebook usage evenness and annotation confidence). The default and
recommended starting encoder is `mlp_deep`.

| input representation | recommended encoder | quantizer | notes |
|---|---|---|---|
| segmented expression (default) | `mlp_deep` (default) | `vq` | `mlp_plr` is an alternative on diverse gene-rich cohorts; plain `mlp` is a baseline |
| transcript context (segmentation-free molecular field concatenated to counts) | `mlp_deep` | `vq` | radius 7 um; input dim doubles |
| molecule set (subcellular transcript point cloud) | `MoleculeSetVQVAE` (Set-Transformer) | `vq` | pool with concat[masked max, masked mean, PMA] |

Benchmark takeaways worth knowing:

- MLP-family encoders (`mlp_deep`, `mlp`, `mlp_plr`) + the default `vq` give the
  healthiest, most even cell codebook. **RVQ underperforms VQ.**
- `mlp_deep` is the default and the safest starting point. `mlp_plr` (per-gene PLE)
  can fill a fuller codebook on diverse, gene-rich cohorts (for example the MERFISH
  retina demo), but it over-parameterizes and degenerates on sparse Xenium panels and
  on tiny datasets (seqFISH 225 cells, CosMx 21k genes), where plain `mlp` or
  `mlp_deep` is correct.
- Plain attention / diffusion encoders (`dit`, `diffusion`) **collapse the cell
  codebook** unless they use permutation-invariant set pooling; `set_transformer`
  and `perceiver_io` stay healthy.
- `lfq` and `bsq` degenerate into per-cell hashing (avoid). `fsq` / `residual_fsq`
  use a larger implicit codebook.
- `ft_transformer` is compute-prohibitive at cohort scale (per-gene attention is
  O(genes^2)).

Available encoders: `mlp`, `mlp_deep`, `mlp_plr`, `residual_mlp`, `cnn`,
`fast_cnn`, `deep_cnn`, `gnn`, `transformer`, `set_transformer`, `perceiver_io`,
`soft_moe`, `ft_transformer`, `dit`, `diffusion`.
Available quantizers: `vq` (default), `rvq`, `grvq`, `pq`, `qinco`, `rot`, `soft`,
`bsq`, `lfq`, `fsq`, `residual_fsq`.

## Adaptive batch size and runtime

`TrainConfig(batch_size="auto")` resolves the batch at train time from the panel
size and available GPU memory, then scales the learning rate by
`sqrt(effective_batch / 2048)` so a larger resolved batch stays well-optimized (a
366 to 732 gene panel at batch 2048 uses only a couple percent of a modern GPU, so
`"auto"` goes much larger). This is a throughput lever. Because the codebook
diversity term and dead-code reset act per batch, a very large batch gives them
fewer updates per epoch, so for the evenest codebook a moderate fixed batch (e.g.
2048) can be preferable; the notebooks use 2048 for that reason. The resolved value
is always recorded as `effective_batch_size` in `training_runtime.json`. Pass an
integer to reproduce a fixed run exactly.

```python
TrainConfig(num_epochs=300, batch_size="auto")
```

## Default hyperparameters

```
cell_num_embeddings         256      neighborhood_num_embeddings   32
cell_embedding_dim          64       neighborhood_embedding_dim    256
spatial_graph               knn_radius   radius                    50.0 um
k_neighbors                 20       neighborhood_aggregation      weighted_mean
encoder_type                mlp_deep quantizer_type                vq
transcript_context radius   7.0 um   molecule-set gather radius    7.0 um
```

Real cohort runs use `num_epochs ~300`. The demos below use far fewer epochs so
they finish in minutes; every notebook says so in a markdown cell.

## Runnable scripts

```bash
python examples/train_expression.py   # train on the bundled Xenium core, verify codes
python examples/apply_codes.py         # train on 80%, assign codes to the held-out 20%
```

## Notebooks (tested end to end)

| notebook | what it shows |
|---|---|
| `notebooks/01_quickstart.ipynb` | load the real MERFISH retina cohort, train `mlp_deep` + `vq`, load the codes, per-code top-marker table, code-usage bar chart |
| `notebooks/02_transcript_context.ipynb` | compute `transcript_context` (radius 7 um) from a transcripts table, concat to counts, train `mlp_deep` |
| `notebooks/03_molecule_set.ipynb` | the subcellular molecule-set (point-cloud) representation with `MoleculeSetVQVAE` |
| `notebooks/04_apply_to_new_data.ipynb` | load a trained checkpoint and assign codes to held-out cells with `predict_codes` |

### Platform-specific notebooks

`notebooks/platforms/` holds one end-to-end example per imaging platform, each on
that platform's **real** data, executed with outputs saved. They show how the input
conventions and the right encoder / codebook size change across platforms.

| notebook | platform | real dataset | encoder / codebook | why |
|---|---|---|---|---|
| `notebooks/platforms/xenium.ipynb` | 10x Xenium | RCC tissue core, 7,824 cells x 366 genes, 1 sample | `mlp` / 64 | a single small homogeneous core; `mlp_plr` over-parameterizes and 256 codes exceed its diversity, so use `mlp` + 64 codes. Also demos `transcript_context` from the bundled molecule table |
| `notebooks/platforms/merfish.ipynb` | Vizgen MERFISH | mouse retina, 113,385 cells x 368 genes, 4 samples | `mlp_plr` / 256 | diverse gene-rich cohort where `mlp_plr` fills a full codebook (about 197/256 in 30 demo epochs); the library default `mlp_deep` also works well here |
| `notebooks/platforms/cosmx.ipynb` | NanoString CosMx WTx | human pancreas, 48,944 cells x 21,731 genes (a real spatial window is cropped for the fast demo) | `mlp` / 256 | on a ~21.7k-gene panel `mlp_plr` collapses the codebook, plain `mlp` stays even (about 255/256); pixel coordinates converted to microns |
| `notebooks/platforms/seqfish.ipynb` | seqFISH+ | NIH/3T3 fibroblasts, 225 cells x 10,000 genes, 17 FOVs | `mlp` / 64 | 225 cells cannot fill 256 codes and `mlp_plr` collapses to 1 code, so lower the codebook to 64 and use `mlp` (about 61/64) |

Platform molecule-table column conventions (`nicheverse.data.transcript`): Xenium =
`x_location` / `y_location` / `feature_name`; CosMx = `x_global_px` / `y_global_px` /
`target`; MERFISH = `global_x` / `global_y` / `gene`. A `transcript_context` input is
only shown where a real molecule table exists for that platform (Xenium here);
otherwise the notebook trains on segmented counts.

## Example datasets (real, bundled)

`examples/data/` ships two real datasets so every notebook runs out of the box.
Nothing is simulated or randomly subsampled.

| file | contents | used by |
|---|---|---|
| `merfish_retina.h5ad` | real MERFISH mouse retina (Vizgen), 113,385 cells x 368 genes, 4 samples, raw counts, `obsm["spatial"]` (microns), `obs["sample_id"]` | `01_quickstart`, `04_apply_to_new_data` |
| `xenium_rcc_core.h5ad` | one human RCC Xenium tissue-microarray core (in-house cohort), 7,824 cells x 366 genes, raw counts, `obsm["spatial"]` (microns), `obs["sample_id"]` | `02_transcript_context`, `03_molecule_set` |
| `xenium_rcc_core_transcripts.parquet` | the matched molecule table (`x_location`, `y_location`, `feature_name`, `cell_id`, `overlaps_nucleus`) | `02_transcript_context` |
| `xenium_rcc_core_molecule_sets/` | per-cell subcellular molecule-set shards (radius 7 um) | `03_molecule_set` |

The MERFISH retina is diverse (many neural cell types across four samples), so the
`mlp_plr` encoder learns a full, even cell codebook there (the default `mlp_deep`
also works well). The single
RCC core is fairly homogeneous, so it exercises fewer cell codes; it is used for the
transcript-context and molecule-set notebooks because it ships with its matched
molecule table. The RCC cell-by-gene counts and centroids were aggregated from the
Xenium transcripts table (centroid = mean position of a cell's nucleus-overlapping
molecules), control / blank / unassigned probes were removed, and cells were kept by
a standard minimum-count QC filter.
