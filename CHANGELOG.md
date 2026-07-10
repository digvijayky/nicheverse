# Changelog

## Unreleased

Refactored to a PyTorch / deep-learning-library layout; removed the old
`pp` / `tl` / `pl` / `io` namespaces in favor of `nicheverse.models`,
`nicheverse.data`, `nicheverse.training`, and `nicheverse.plotting`, with a
`Trainer` and `build_encoder` / `build_quantizer` registries. Added opt-in
quantizers (FSQ, SoftVQ, RotVQ, QINCo, Product VQ), encoders (residual MLP,
transformer), NB/Poisson reconstruction, distance kernels, spatial losses,
warmup-cosine scheduling, transcript-level context, and MAE pretraining. All defaults reproduce the published production model bit-for-bit.

Added a literature-grounded annotation module (`nicheverse.annotate`): per-code
marker and DEG evidence, LLM labeling of cell states and spatial niches via Claude,
GPT, or a local model (optional PubMed / bioRxiv grounding, coarse-to-fine clustering,
and a reconciliation pass), plus a dotplot review artifact. Shipped an MCP server
(`nicheverse-mcp`) and a Claude Code skill so agents can drive the full workflow.
Validated the modality-agnostic pipeline end-to-end on Xenium, CosMx, MERFISH, and seqFISH.

Count-native defaults and speed. The default encoder is now `mlp_deep`, a SwiGLU
pre-norm residual MLP (no per-gene numerical embedding), which gives the healthiest
raw codebook on sparse Xenium counts; per-gene numerical embeddings (`mlp_plr` / PLE)
degenerate on sparse counts. The default cell reconstruction is now a negative-binomial
NLL on the raw counts (scVI-style library from the observed total count) plus a
Bernoulli detection hurdle (`detection_weight=0.5`), with no MSE on the cell branch
(`cell_recon="nb"`); the default niche reconstruction is composition MSE plus a
Dirichlet-multinomial on the count-scale aggregated composition (`niche_recon="mse_dirmult"`).
These count defaults require raw INTEGER counts in `adata.X` (a normalized or log
matrix now raises); the pure-MSE path is recoverable with `cell_recon="mse"` +
`niche_recon="mse"`. `ModelConfig.from_dict` falls back to the old MSE-only path so
pre-loss-refactor checkpoints still load strictly. The default spatial graph is
`knn_radius` at radius 50 microns (`k_neighbors=20`, `weighted_mean` aggregation),
built per sample. The optimizer is AdamW with decoupled selective weight decay (0.01,
applied only to Linear / Conv weights); the EMA codebook is frozen from the optimizer.
Added opt-in `TrainConfig.device_resident` (GPU-resident dataset, roughly 3 to 15x
faster, accuracy-neutral, memory-fit-gated with a CPU fallback), `batch_size="auto"`
(adapts the batch to the panel size), and per-run `training_runtime.json` metrics.
The transcript-context and molecule-set default radii moved to 7 microns.
Synced the documentation site (README, guides, API reference) to these defaults.

All notable changes to nicheverse are documented in this file. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
starting from 1.0.0.

## [0.2.0] - 2026-07

Sophistication, packaging, and documentation release. All new behavior is
opt-in; the 0.1.x default training path (and released checkpoint) reproduces
bit-for-bit.

### Added
1. A namespaced convenience API, alongside the flat
   backward-compatible entry points.
2. AnnData key registry: `Keys` and `anndata_keys()` (`_constants.py`).
3. `spatial_neighbors(adata, ...)` writes aggregated neighborhood features to `obsm`.
4. Vectorized, bounded-memory (`agg_chunk`) neighborhood aggregation replacing the per-cell
   Python loop; scales to multi-million-cell cohorts.
5. Spatial graph backends `knn` (default), `radius`, and `delaunay` (`SpatialDataset.spatial_graph`,
   `TrainConfig.spatial_graph`, `predict_codes(spatial_graph=...)`).
6. Cosine-distance codebook assignment (`ModelConfig.vq_distance="cosine"`; default `"l2"`).
7. `HierarchicalVQVAE.encode()` (indices-only inference) and `HierarchicalVQVAE.from_checkpoint()`.
8. `TrainConfig` knobs: `val_fraction`, `early_stopping_patience`, `grad_clip`, `amp`,
   `weight_decay`, `resume_from`, `save_best`; per-epoch codebook perplexity in `training_losses.json`;
   `tqdm` progress bars; reproducible seeded DataLoader shuffling.
9. Packaging: migrated to the hatchling build backend; PyPI-ready metadata; `doc` / `test` extras;
   coverage config. Sphinx documentation site (`docs/`, `.readthedocs.yaml`), pre-commit config,
   GitHub Actions CI + PyPI trusted-publishing release workflow, `CODE_OF_CONDUCT.md`, `RELEASING.md`,
   and a Sphinx documentation site.

### Changed
1. `__version__` is now read from installed package metadata (single source of truth).
2. Optimizer is `AdamW` (`weight_decay=0.0` default reduces to the prior Adam behavior).
3. Inference paths use `torch.inference_mode()`.

## [Unreleased]

### Added
1. `python -m nicheverse` runs the CLI (new `__main__.py`).
2. CLI: `--version`, `--quiet`, `--verbose` flags.
3. CLI: `predict --report` writes a JSON SHA256 contract consumable by `verify`.
4. `ModelConfig.cross_attention_weight` exposes the previously hardcoded 0.5 residual weight.
5. `VectorQuantizer.diversity_temperature` exposes the softmax temperature for the diversity entropy term.
6. `VectorQuantizer.dead_code_reset_interval` and `dead_code_usage_fraction` constructor knobs.
7. `train_config.json` written to the checkpoint directory for reproducibility.
8. `TrainConfig.num_workers` for DataLoader workers.
9. Module-level logging via `logging.getLogger(__name__)` throughout.
10. New tests: end-to-end CLI subprocess smoke tests, predict determinism, Xenium loader against a fake-but-realistic output directory.
11. `pyproject.toml`: `ruff`, `mypy`, and `pytest` configuration plus richer trove classifiers.
12. `CHANGELOG.md`, `CONTRIBUTING.md`, `DETERMINISM.md`.

### Changed
1. `VectorQuantizer._kmeans_init`: when the batch is smaller than the codebook, seed every slot from the batch (with small noise) instead of leaving the tail at the uniform random init.
2. `VectorQuantizer` softmax for the diversity term now divides by `diversity_temperature` (default 1.0; previous behavior preserved).
3. `io.load_xenium_run`: control / blank / unassigned / codeword filter is now case-insensitive and also catches `NegControlProbe`, `Deprecated`, `Intergenic`, and `antisense` prefixes.
4. `io.attach_codes_to_adata`: indices are stored as `int16` / `int32` when possible (was always `int64`).
5. `predict.predict_codes`: realizes a fresh AnnData copy before in-place preprocessing to suppress `ImplicitModificationWarning`.
6. `utils.env_snapshot`: now uses `importlib.metadata.version` (no more scanpy `FutureWarning`).
7. `viz`: matplotlib backend is only forced to `Agg` when no display is available; Jupyter users keep their backend.

### Fixed
1. `VectorQuantizer._reset_dead_codes`: no-op when the batch is empty (was an out-of-range `randint`).
2. `ModelConfig.from_dict`: tolerates `bytes` gene names from h5 loading and ignores unknown keys.
3. `SpatialDataset`: input shape validation (n_cells consistency, 2D coords, k_neighbors > 0) with actionable messages.
4. `train.train_model`: skips normalize/log1p when AnnData is already log-normalized (detected via `uns['log1p']` / `uns['normalize_total']`).
5. `cli.verify`: JSON reference contract is documented and accepts both canonical (`predicted_*`) and legacy (`reference_*`) keys.

## [0.1.0] - 2026

Initial release accompanying the Cancer Cell submission. Hierarchical VQ-VAE
with cell codebook (K_c = 256) and neighborhood codebook (K_n = 32) trained on
173 Xenium samples (5.66M cells) of the RCC + BrM cohort.
