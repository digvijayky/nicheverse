# Nicheverse

[![Docs](https://img.shields.io/badge/docs-nicheverse.org-f59e0b)](https://nicheverse.org)

Interpretable modeling of tissues. Nicheverse is a hierarchical VQ-VAE that tokenizes imaging-based spatial transcriptomics (Xenium, MERFISH, seqFISH, CosMx) into two coupled codebooks: one of recurrent cell states and one of recurrent multicellular niches.

It trains on any cell-by-gene matrix with spatial coordinates. The cell codebook captures transcriptional states; the neighborhood codebook captures the local tissue context around each cell. A gated cross-attention block couples them, so each cell's state assignment is informed by its niche. Nicheverse was built for the renal cell carcinoma and brain metastasis cohort in our Cancer Cell submission, where it resolved 256 cell states and 32 niches across 173 samples (5.66 million cells), and it runs on any Xenium output without changes.

## Install

Install from source with pip:

```bash
git clone https://github.com/digvijayky/nicheverse.git
cd nicheverse
pip install .
```

For development, install editable with the dev, doc, and test extras: `pip install -e ".[dev,doc,test]"`.

GPU is optional. PyTorch selects `cuda` automatically when available. Python 3.10 or newer is required.

## Quickstart

Three stages: preprocess Xenium runs into a single AnnData, train, then predict on new data.

```bash
# 1. Merge Xenium output directories
nicheverse preprocess \
    --run-dir /path/to/run_A /path/to/run_B /path/to/run_C \
    --sample-id sample_A sample_B sample_C \
    --output ./preprocessed/cohort.h5ad

# 2. Train
nicheverse train \
    --input ./preprocessed/cohort.h5ad \
    --checkpoint-dir ./checkpoint \
    --num-epochs 300 \
    --cell-codebook-size 256 \
    --neighborhood-codebook-size 32 \
    --k-neighbors 20

# 3. Annotate a new Xenium dataset
nicheverse preprocess --run-dir /path/to/new_run --output ./new.h5ad
nicheverse predict \
    --input ./new.h5ad \
    --checkpoint ./checkpoint/hierarchical_vqvae_checkpoint.pt \
    --output ./new_annotated.h5ad
```

Manifest mode for large cohorts (CSV with columns `run_dir,sample_id`):

```bash
nicheverse preprocess --manifest ./manifest.csv --output ./cohort.h5ad
```

## Python API

Common entry points are re-exported at the top level (`import nicheverse as nv`); the submodules `models`, `data`, `training`, and `plotting` hold the rest.

```python
import nicheverse as nv
from nicheverse import ModelConfig, TrainConfig

adata = nv.read_xenium_cohort(["./run_A", "./run_B"])
mc = ModelConfig(input_dim=adata.X.shape[1],
                 cell_num_embeddings=256, neighborhood_num_embeddings=32,
                 gene_names=tuple(adata.var_names))
tc = TrainConfig(num_epochs=300, k_neighbors=20)
model, adata = nv.Trainer(tc).fit(adata, "./checkpoint", model_config=mc)

new = nv.read_xenium_cohort(["./run_C"])
annotated = nv.predict_codes(new, "./checkpoint/hierarchical_vqvae_checkpoint.pt",
                          output_path="./run_C_annotated.h5ad")
```

`Trainer` wraps the functional `train_model` / `predict_codes`, which are also importable at the top level.

After training, `adata.obs` carries `cell_codebook_idx` (0 to 255) and `neighborhood_codebook_idx` (0 to 31). `adata.obsm` carries `X_cell_embedding` and `X_neighborhood_embedding` for downstream UMAP, clustering, or differential analysis. The canonical key names are available programmatically via `nv.anndata_keys()`.

## Model and training options

- **Encoders** (`ModelConfig.encoder_type`): `mlp_deep` (default; a SwiGLU pre-norm residual MLP), plus `mlp`, `mlp_plr`, `residual_mlp`, `transformer`, `cnn`, `fast_cnn`, `deep_cnn`, `gnn`, `diffusion`, `dit`, `set_transformer`, `perceiver_io`, `soft_moe`, `ft_transformer`; registry `nicheverse.models.build_encoder`. `mlp_deep` is the default because it gives the healthiest raw codebook on sparse Xenium counts; per-gene numerical embeddings (`mlp_plr` / PLE) degenerate on sparse counts.
- **Quantizers** (`ModelConfig.quantizer_type`): `vq` (default; stabilized EMA codebook with k-means++ init, dead-code reset, and a diversity term, and the EMA codebook is frozen from the optimizer), plus `rvq`, `grvq`, `pq`, `qinco`, `rot`, `soft`, `bsq`, `lfq`, `fsq`, `residual_fsq`; registry `nicheverse.models.build_quantizer`.
- **Cell reconstruction** (`ModelConfig.cell_recon`): default `nb` is a negative-binomial NLL on the RAW counts (scVI-style library from the observed total count) plus a Bernoulli/BCE detection hurdle (`detection_weight=0.5`); no MSE on the cell branch. Set `cell_recon="mse"` (with `detection_weight=0`) to recover the pure MSE-on-log1p path.
- **Niche reconstruction** (`ModelConfig.niche_recon`): default `mse_dirmult` is composition MSE plus a Dirichlet-multinomial on the count-scale aggregated-neighbor composition. Set `niche_recon="mse"` for pure composition MSE.
- **Spatial graphs / kernels** (`TrainConfig`): `knn`, `knn_radius` (default, radius 50 microns), `radius`, `delaunay`, `alpha_complex`, `gabriel`, `rng`; aggregation `weighted_mean` (default, inverse distance), `mean`, `max`, `gaussian`, `inverse_square`.
- **Spatial losses** (opt-in): `laplacian`, `contrastive`, `codebook_consistency`, `graph_tv`.
- **Data utilities**: `nicheverse.data.transcript_context` (default radius 7 microns), the subcellular molecule-set featurizer (default radius 7 microns), and `nicheverse.training.mae_pretrain`.
- **Optimizer**: `AdamW` with decoupled selective weight decay (`weight_decay=0.01`, applied only to Linear / Conv weights; biases, norms, and bare parameters are excluded).
- **Vectorized neighborhood aggregation** in bounded-memory chunks, replacing the per-cell Python loop, for multi-million-cell cohorts.
- **Cosine-distance codebook** option (`ModelConfig.vq_distance="cosine"`) alongside the default squared-Euclidean assignment.
- **Speed knobs (opt-in, accuracy-neutral)**: `TrainConfig.device_resident=True` keeps the feature tensors GPU-resident (roughly 3 to 15x faster, memory-fit-gated with a clean CPU fallback); `batch_size="auto"` adapts the batch to the panel size; per-run timing is written to `training_runtime.json`.
- **Training knobs**: optional validation split with early stopping, best-checkpoint saving, gradient clipping, CUDA automatic mixed precision, checkpoint resume, `tqdm` progress, and per-epoch codebook perplexity logging.
- **Packaging**: hatchling build backend and PyPI-ready metadata.

Pre-loss-refactor checkpoints still load via `ModelConfig.from_dict` backward-compatibility (they fall back to the old MSE-only cell/niche path so their state_dicts load strictly).

## Multi-GPU training (opt-in DDP)

Training scales across GPUs with DistributedDataParallel while holding the global
batch size constant, so the optimization matches a single-GPU run within numerical
tolerance. It is fully opt-in: the single-GPU default path is byte-identical to
before, and DDP activates only when both `TrainConfig(ddp=True)` is set and the
process is launched under `torchrun` with more than one rank.

```bash
# 2 GPUs on one node. batch_size is the GLOBAL batch; each GPU gets batch_size/2.
torchrun --nproc_per_node=2 your_train_script.py
```

```python
from nicheverse import TrainConfig
tc = TrainConfig(
    batch_size=4096,   # global batch, split to 2048/GPU across 2 ranks
    ddp=True,          # activate DDP when launched under torchrun (WORLD_SIZE>1)
    num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4,
)
```

The EMA codebook statistics, k-means++ initialization, diversity term, and (on
CUDA) BatchNorm are all made rank-correct so the trained codebook matches a
single-GPU run on the full global batch. See `DETERMINISM.md` for the exact
equivalence contract and the documented residuals (dead-code reset RNG, dropout
masks, floating-point reduction order). `batch_size` must be divisible by the
number of ranks.

Other accuracy-preserving speed knobs (all opt-in, no effect on the loss curve):
`compile_model=True` wraps the model in `torch.compile`; `num_workers`,
`pin_memory`, `persistent_workers`, and `prefetch_factor` tune the DataLoader.

## Annotating codes

Turn the learned codes into cell types and spatial niches with an LLM (Claude, GPT,
or a local model), grounded in per-code markers, DEGs, and primary literature:

```python
from nicheverse.annotate import annotate_codes, annotate_niches, attach_labels

labels = annotate_codes(adata, "cell_codebook_idx", provider="anthropic",
                        tissue="human clear cell RCC", with_literature=True)
attach_labels(adata, "cell_codebook_idx", labels, key_added="celltype_annot")
niches = annotate_niches(adata, "neighborhood_codebook_idx", "celltype_annot")
```

Install the backends with `pip install ".[llm]"` from the cloned repo. Claude Code and Codex can
run the whole workflow through the bundled MCP server (`nicheverse-mcp`) and the
`nicheverse-annotate` skill.

## Outputs

The checkpoint directory contains:

```
hierarchical_vqvae_checkpoint.pt    PyTorch state_dict plus embedded ModelConfig
hierarchical_vqvae_checkpoint.json  Human readable config
cell_codebook.npz                   (n_codes, embed_dim) centroid matrix
neighborhood_codebook.npz           Niche centroids
hierarchical_cell_embeddings.npz    Per cell latent code (pre-quantization)
hierarchical_neighborhood_embeddings.npz
hierarchical_cell_indices.npz       Per cell hard code assignment
hierarchical_neighborhood_indices.npz
training_losses.json                Per epoch total / cell / neighborhood loss
adata_with_hierarchical_embeddings.h5ad
training_losses.pdf
codebook_usage.pdf
per_sample_spatial/                 One PDF per sample
```

## Inputs supported

Any imaging spatial-transcriptomics data works: bring an AnnData with `obsm['spatial']` (microns) and an `obs` sample column (use `read_spatial` to build these from `obs` x/y columns and to rescale pixel coordinates), then call `train_model` / `predict_codes`. A native Xenium reader is built in. For Xenium, the loader reads `cell_feature_matrix.h5` (or a 10x style mtx folder) plus `cells.parquet` (or `cells.csv.gz`), filters control / blank / unassigned probes by default, and builds a barcode based index of the form `cell_id__sample_id` to keep multi sample cohorts disambiguated.

Each cell carries `obs['sample_id']` and `obsm['spatial']` (x_centroid, y_centroid in microns). For multi sample training, the k nearest neighbor graph used to build the neighborhood feature is computed within sample only, so cross sample edges never form.

## Reproducibility

The exact hyperparameters used in the Cancer Cell submission for the 173 sample RCC plus BrM cohort:

```bash
nicheverse train \
    --input cohort.h5ad \
    --checkpoint-dir checkpoint_rcc_brm_v4_dev \
    --num-epochs 300 \
    --cell-codebook-size 256 \
    --cell-codebook-embdim 64 \
    --neighborhood-codebook-size 32 \
    --neighborhood-codebook-embdim 256 \
    --k-neighbors 20
```

Set `torch.manual_seed(9)` is applied automatically via `TrainConfig.seed`.

## Testing

```bash
pytest -q
```

The test suite uses tiny synthetic data and covers forward pass shapes, every registered encoder and quantizer, checkpoint round trip, per sample k-NN isolation, the count and MSE reconstruction paths, device-resident training, end to end train then predict, and gene panel mismatch detection.

## Citation

If you use Nicheverse in your work, please cite it. A manuscript is in preparation; in the meantime you can cite the software:

```bibtex
@article{nicheverse2026,
  title   = {Nicheverse: hierarchical spatial tokenization of cell states and multicellular niches},
  author  = {Yarlagadda, Digvijay and others},
  year    = {2026},
  note    = {manuscript in preparation}
}
```

Full documentation lives at [nicheverse.org](https://nicheverse.org).
