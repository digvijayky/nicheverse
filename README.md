# Nicheverse

[![Docs](https://img.shields.io/badge/docs-nicheverse.org-f59e0b)](https://nicheverse.org)


Nicheverse is a hierarchical VQ-VAE that tokenizes imaging-based spatial transcriptomics into a discrete vocabulary of cell states and tissue niches. It trains on any cell-by-gene count matrix with spatial coordinates and runs on any imaging platform (Xenium, MERFISH, CosMx, seqFISH, and more). A cell encoder maps each cell to a **cell codebook** of transcriptional states, and a neighborhood encoder maps its surrounding tissue to a **niche codebook**. The two branches are coupled by a one-directional gated cross-attention block: the cell attends to its own niche before decoding, so spatial context can settle a borderline assignment, yet the cell code stays anchored to that cell's own transcripts and is never overridden by its neighbors. Because every cell is quantized against the same fixed codebook, the learned vocabulary is portable, the same code denotes the same state across cohorts, tissues, and platforms.

## Install

```bash
pip install nicheverse
```

Or from source (editable, so edits to the tree take effect with no reinstall):

```bash
git clone https://github.com/digvijayky/nicheverse.git
cd nicheverse
pip install -e .
```



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

- **Encoders** (`ModelConfig.encoder_type`): `mlp_deep` (default; a SwiGLU pre-norm residual MLP), plus `mlp`, `mlp_plr`, `residual_mlp`, `transformer`, `cnn`, `fast_cnn`, `deep_cnn`, `gnn`, `diffusion`, `dit`, `set_transformer`, `perceiver_io`, `soft_moe`, `ft_transformer`; registry `nicheverse.models.build_encoder`. `mlp_deep` is the default because it gives the healthiest raw codebook on the sparse, low-magnitude counts typical of imaging panels; per-gene numerical embeddings (`mlp_plr` / PLE) degenerate on sparse counts.
- **Quantizers** (`ModelConfig.quantizer_type`): `vq` (default; stabilized EMA codebook with k-means++ init, dead-code reset, and a diversity term, and the EMA codebook is frozen from the optimizer), plus `rvq`, `grvq`, `pq`, `qinco`, `rot`, `soft`, `bsq`, `lfq`, `fsq`, `residual_fsq`; registry `nicheverse.models.build_quantizer`.
- **Cell reconstruction** (`ModelConfig.cell_recon`): default `nb` is a negative-binomial NLL on the RAW counts (scVI-style library from the observed total count) plus a Bernoulli/BCE detection hurdle (`detection_weight=0.5`); no MSE on the cell branch. Set `cell_recon="mse"` (with `detection_weight=0`, and `niche_recon="mse"`) to recover the pure MSE-on-log1p path; the two branches switch together, since the default `mse_dirmult` niche term needs the count-scale composition that only a count cell mode provides.
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

## Reproducing the study model

The model in the accompanying manuscript was trained on a 173-sample imaging spatial-transcriptomics cohort (5.66M cells, 366-gene Xenium panel). Its architecture and optimization hyperparameters:

```bash
nicheverse train \
    --input cohort.h5ad \
    --checkpoint-dir checkpoint \
    --num-epochs 300 \
    --cell-codebook-size 256 \
    --cell-codebook-embdim 64 \
    --neighborhood-codebook-size 32 \
    --neighborhood-codebook-embdim 256 \
    --k-neighbors 20
```

The released model additionally uses the transcript-context input: a 7-micron segmentation-free molecular field concatenated onto the segmented counts, giving a 732-dimensional cell input. Build it with `nicheverse.data.transcript_context` and train on the concatenated matrix as shown in `notebooks/02_transcript_context.ipynb`. Every other setting (encoder `mlp_deep`, quantizer `vq`, per-sample `knn_radius` graph at radius 50 microns, inverse-distance aggregation, seed 9) is a library default. `TrainConfig.seed` (default 9) is applied automatically, so a fixed model and seed give bit-reproducible codes.

## Testing

```bash
pytest -q
```

The test suite uses tiny synthetic data and covers forward pass shapes, every registered encoder and quantizer, checkpoint round trip, per sample k-NN isolation, the count and MSE reconstruction paths, device-resident training, end to end train then predict, and gene panel mismatch detection.

## Citation

If you use Nicheverse, please cite the preprint (and the method's original conference paper):

```bibtex
@article{yarlagadda2026developmental,
  title   = {Developmental reversion underlies resistance to immune
             checkpoint blockade in kidney cancer},
  author  = {Yarlagadda, Dig Vijay Kumar and Wang, Zhenghan and
             Jiang, Hui and Vuong, Lynda and Sanmiguel, Andrea L{\'o}pez and
             Yang, Ching-Yeuh and Kotecha, Ritesh R. and Chen, Ying-Bei and
             Hakimi, A. Ari and Leslie, Christina S. and Massagu{\'e}, Joan},
  journal = {bioRxiv},
  year    = {2026},
  month   = {aug},
  doi     = {10.64898/2026.08.05.743137},
  url     = {https://www.biorxiv.org/content/10.64898/2026.08.05.743137v1},
  note    = {Preprint}
}

@INPROCEEDINGS{10350864,
  author={Yarlagadda, Dig Vijay Kumar and Massagué, Joan and Leslie, Christina},
  booktitle={2023 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)}, 
  title={Discrete Representation Learning for Modeling Imaging-based Spatial Transcriptomics Data}, 
  year={2023},
  pages={3848-3857},
  keywords={Representation learning;Annotations;Biological system modeling;Transcriptomics;Predictive models;Data models;Spatial databases},
  doi={10.1109/ICCVW60793.2023.00416}
}
```

Full documentation lives at [nicheverse.org](https://nicheverse.org).
