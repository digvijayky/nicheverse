# Running the released checkpoint as a reviewer

This is the one page recipe distributed with the manuscript. It walks a reviewer from an empty machine to a `verify: PASS` result in roughly 30 minutes on a single CPU node.

## What you need

A workstation or HPC node with:

```
Python 3.10 or later
8 CPU cores (16 preferred)
64 GB RAM
50 GB free disk
optional: an NVIDIA GPU with at least 12 GB memory (cuts wall time from 30 minutes to 2 minutes)
```

The reproducibility bundle, downloadable from Zenodo at the DOI printed in the manuscript:

```
release/
  README.md
  requirements-frozen.txt
  nicheverse-0.1.0.tar.gz
  rcc_brm_v4dev_173samples.pt
  cohort_preprocessed.h5ad
  expected_outputs.json
  env_snapshot_training.json
  verify_reproducibility.sh
  SHA256SUMS
```

## Step 1: verify file integrity

```bash
cd release
sha256sum --check SHA256SUMS
```

All files must report `OK`. If any file fails, redownload from Zenodo.

## Step 2: build a clean environment

```bash
python -m venv ./repro_env
source ./repro_env/bin/activate
pip install --upgrade pip
pip install -r requirements-frozen.txt
pip install nicheverse-0.1.0.tar.gz
```

## Step 3: confirm the install

```bash
nicheverse --help
nicheverse info --checkpoint rcc_brm_v4dev_173samples.pt
```

The `info` command prints the model config, gene panel size, checkpoint SHA256, and a snapshot of YOUR host environment. Compare your `host_env.cuda_version` and `cudnn_version` against `env_snapshot_training.json` in the bundle. Match major versions to maximize the chance of bit identical output.

## Step 4: predict

```bash
nicheverse predict \
    --input cohort_preprocessed.h5ad \
    --checkpoint rcc_brm_v4dev_173samples.pt \
    --output annotated.h5ad \
    --device cpu     # or cuda if available. Determinism (seed 9) is always on for predict.
```

Wall time: under 10 minutes on a single A100, under 30 minutes on a 16 core CPU.

## Step 5: verify

```bash
nicheverse verify \
    --predicted annotated.h5ad \
    --reference expected_outputs.json \
    --report verification_report.json
```

The command prints a JSON report and exits with code 0 on bit identical match, code 2 on mismatch. The report looks like:

```json
{
  "n_cells": 5662265,
  "predicted_cell_sha256":         "a3f2...e91",
  "reference_cell_sha256":         "a3f2...e91",
  "predicted_neighborhood_sha256": "7b41...4cd",
  "reference_neighborhood_sha256": "7b41...4cd",
  "cell_sha_match": true,
  "neigh_sha_match": true
}
```

Both matches must be `true` to confirm bit identical reproduction.

## What to do on mismatch

If `cell_sha_match` or `neigh_sha_match` is `false`, read [determinism.md](determinism.md) for diagnosis. The most common cause is a cuDNN version mismatch. In > 99.9% of cells you will still see the same cell type and niche assignment; the few flips are cells whose embedding sits exactly on a codebook Voronoi boundary.

You can quantify near-match using:

```bash
nicheverse verify \
    --predicted annotated.h5ad \
    --reference adata_with_hierarchical_embeddings.h5ad \
    --report verification_report.json
```

This time the report adds:

```text
{
  ...
  "n_compared": 5662265,
  "cell_exact_match_pct":          99.97,
  "neighborhood_exact_match_pct":  99.94,
  ...
}
```

We consider anything above 99.9% on both metrics to be a successful reproduction for the purposes of validating manuscript findings.

## Re-running with your own Xenium data

The same checkpoint can be applied to ANY Xenium dataset that uses the same 366 gene panel:

```bash
nicheverse preprocess --run-dir /path/to/your/xenium_run --output my.h5ad
nicheverse predict --input my.h5ad --checkpoint rcc_brm_v4dev_173samples.pt --output my_annotated.h5ad
```

This is the recommended way to apply the manuscript's discovered cell states and niches to a new cohort.
