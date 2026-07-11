# Determinism guarantees

The package targets bit-identical reproducibility for INFERENCE on the same hardware. Training reproducibility is supported on a best-effort basis.

## What is bit identical

A reviewer running:

```bash
nicheverse predict --input cohort.h5ad --checkpoint rcc_brm_v4dev_173samples.pt --output annotated.h5ad
nicheverse verify  --predicted annotated.h5ad --reference expected_outputs.json
```

on the same hardware (matching CUDA capability, matching cuDNN major version) and with the pinned environment from `requirements-frozen.txt` will get the SAME cell_codebook_idx and neighborhood_codebook_idx for every cell. `verify` confirms this by comparing SHA256 of the two integer arrays.

## What controls determinism

`predict_codes` calls `seed_everything(seed=9, deterministic=True)` before any computation. That helper:

1. Sets `PYTHONHASHSEED`, `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`.
2. Sets `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`.
3. Calls `torch.use_deterministic_algorithms(True, warn_only=True)`.
4. Sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` for deterministic CUBLAS.

The inference path has no shuffled data loaders, no dropout active at eval, no batch normalization train stats. The forward pass is therefore deterministic given the same input.

The two non-obvious sources of cross-machine variance are:

a) `sklearn.neighbors.NearestNeighbors` with `algorithm='ball_tree'` is deterministic for the same input data, but tie breaking on floating point distances can differ between BLAS backends. We have not observed this in practice on the RCC + BrM data, but if a reviewer reports cell-level disagreement under 1%, this is the most likely cause.

b) Different cuDNN versions can choose different deterministic kernels for the same op, producing slightly different floating point outputs that fall on different sides of the VQ argmin boundary. This is rare (we have seen 0.01% rate when stepping cuDNN 8.7 to 9.0 on the 173 sample cohort) but it can occur. The `env_snapshot.json` shipped with the bundle captures the cuDNN version used at training time so the reviewer can match it.

## What is not bit identical

Training from scratch is NOT guaranteed bit identical because:

a) k-means++ initialization of the codebook draws random samples from the first batch. The seed makes this reproducible on the same hardware but the underlying multinomial sampling depends on CUDA RNG state which can diverge across CUDA versions.

b) DataLoader shuffling with `shuffle=True` reads from `torch.utils.data` random state which is seeded but not guaranteed identical across DataLoader workers settings.

c) Batch norm running stats accumulate over the entire training run and depend on the exact order of batches.

If you re-train from scratch with the same seed and the same pinned environment on the SAME machine, you will get an identical checkpoint. Across machines you will get an extremely similar checkpoint (same coarse structure, slight code-index permutation) but not bit identical.

For manuscripts, this is fine: we publish the trained checkpoint as the artifact of record, not the training run. Reviewers verify inference reproducibility, not training reproducibility.

## How to debug a reproducibility failure

Symptom 1: `verify` reports `cell_sha_match: false` with > 5% disagreement.

Likely cause: gene panel mismatch. Check that the input AnnData's var_names exactly match the checkpoint's `gene_names` (in order). Use:

```python
import torch
from nicheverse.models import load_checkpoint
import anndata as ad

ck = load_checkpoint("rcc_brm_v4dev_173samples.pt")
a = ad.read_h5ad("cohort.h5ad", backed="r")
print(set(ck.config.gene_names) - set(a.var_names))   # missing
print(set(a.var_names) - set(ck.config.gene_names))   # extra
```

Symptom 2: `verify` reports `cell_sha_match: false` with < 1% disagreement, mostly on cells near codebook decision boundaries.

Likely cause: cuDNN version mismatch. Compare `env_snapshot.json` between training and the reviewer's environment.

Symptom 3: `verify` reports `cell_sha_match: false` with > 50% disagreement.

Likely cause: the reviewer is running on data that was not normalized the same way. `predict_codes` defaults to `normalize_total + log1p` on the encoder input. Supply raw integer counts (the released checkpoint and count-native defaults expect raw counts). An already normalized or log matrix is detected and, under the count defaults, raises rather than being silently re-normalized.

Symptom 4: `neighborhood_sha_match: false` but `cell_sha_match: true`.

Likely cause: `sample_id` ordering differs. The per-sample k-NN graph depends on `sample_id`. If you renamed sample IDs or reordered, the neighborhood code will shift even though the cell code is unchanged.

## What you can promise reviewers

Phrasing for the manuscript Methods section:

"All cell-level and niche-level code assignments reported in this study are reproducible from the released b32k checkpoint (the 0.2.0 default configuration: batch size 32768, seed 9, spatial graph `knn_radius` at radius 50 microns, k_neighbors 20) and preprocessed AnnData using the `nicheverse predict` command at version 0.2.0. Bit identical output (SHA256 match on the integer code arrays) is guaranteed on the same hardware with the pinned environment in `requirements-frozen.txt`; near-identical output (> 99% per cell agreement) is observed across hardware. Reproducibility can be verified in one command via the `nicheverse verify` subcommand against the released `expected_outputs.json`."
