# Producing a frozen reproducibility bundle

To accompany a manuscript submission, you should ship a single artifact that lets a reviewer reproduce every code assignment we report. The reference results come from the b32k checkpoint (the 0.2.0 default configuration: batch size 32768, seed 9, spatial graph `knn_radius` at radius 50 microns, k_neighbors 20).

## What the bundle must contain

```
release/
  README.md                              one-page recipe pointing at the rest
  requirements-frozen.txt                exact dep versions
  nicheverse-0.2.0.tar.gz            source distribution
  rcc_brm_v4dev_173samples.pt            self-describing checkpoint (config + gene_names embedded)
  rcc_brm_v4dev_173samples.json          plain text config snapshot
  cohort_preprocessed.h5ad               input AnnData (raw counts, sample_id, spatial)
  expected_outputs.json                  SHA256 of expected cell + neighborhood code arrays
  env_snapshot_training.json             Python / torch / CUDA versions used at training
  verify_reproducibility.sh              one command reviewer recipe
  SHA256SUMS                             SHA256 of every file above
```

## Step by step

### 1. Re-save the checkpoint in self describing format

The bare `state_dict` that PyTorch saves by default is not enough to rebuild the model: a reviewer would need to know `cell_num_embeddings`, `cell_embedding_dim`, hidden dims, and the gene panel. `save_checkpoint` writes a single `.pt` that contains both `state_dict` and a serialized `ModelConfig` including `gene_names`.

```python
from nicheverse.models import ModelConfig, HierarchicalVQVAE, load_checkpoint, save_checkpoint
import anndata as ad

# Read the gene panel from the source AnnData
adata = ad.read_h5ad("checkpoint/adata_with_hierarchical_embeddings.h5ad", backed="r")
genes = tuple(map(str, adata.var_names))
adata.file.close()

# Rebuild the ModelConfig used to train the checkpoint
cfg = ModelConfig(
    input_dim=len(genes),
    hidden_dims=(256, 128),
    cell_embedding_dim=64,
    cell_num_embeddings=256,
    neighborhood_embedding_dim=256,
    neighborhood_num_embeddings=32,
    commitment_cost=0.25,
    use_cross_attention=True,
    gene_names=genes,
)
model = load_checkpoint("checkpoint/hierarchical_vqvae_checkpoint.pt", config=cfg)
save_checkpoint(model, "release/rcc_brm_v4dev_173samples.pt")
```

### 2. Snapshot expected outputs as SHA256

```python
import anndata as ad
import json
from nicheverse.utils import sha256_array

a = ad.read_h5ad("checkpoint/adata_with_hierarchical_embeddings.h5ad")
expected = {
    "n_cells": int(a.n_obs),
    "n_samples": int(a.obs["sample_id"].nunique()),
    "predicted_cell_sha256": sha256_array(a.obs["cell_codebook_idx"].to_numpy(dtype="int64")),
    "predicted_neighborhood_sha256": sha256_array(a.obs["neighborhood_codebook_idx"].to_numpy(dtype="int64")),
}
with open("release/expected_outputs.json", "w") as f:
    json.dump(expected, f, indent=2)
```

### 3. Freeze the environment

```bash
pip freeze | grep -E '^(torch|numpy|pandas|scipy|scikit-learn|scanpy|anndata|matplotlib|pyarrow|h5py|tqdm)==' \
    > release/requirements-frozen.txt
```

Include also the CUDA driver and toolkit versions in a comment at the top of the file.

### 4. Build the source distribution

```bash
cd nicheverse
python -m build --sdist --outdir release/
```

### 5. Compute SHA256 of every file

```bash
cd release && sha256sum * | tee SHA256SUMS
```

### 6. Write the reviewer recipe

`verify_reproducibility.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
pip install -r requirements-frozen.txt
pip install nicheverse-0.2.0.tar.gz
nicheverse predict \
    --input cohort_preprocessed.h5ad \
    --checkpoint rcc_brm_v4dev_173samples.pt \
    --output annotated.h5ad
nicheverse verify --predicted annotated.h5ad --reference expected_outputs.json
echo "exit code 0 = bit identical to manuscript, 2 = mismatch"
```

### 7. Upload to Zenodo

Zenodo gives you a citable DOI and a permanent URL. The submission flow:

  https://zenodo.org/deposit
  - title: "RCC + BrM spatial atlas reproducibility bundle"
  - upload the contents of `release/`
  - license: CC BY 4.0 for data, MIT for code
  - related identifiers: the Cancer Cell DOI (once assigned), the GitHub repo
  - reserve the DOI before publication so it appears in the manuscript

## What does NOT need to be in the bundle

The raw Xenium output directories are large (each run is 10 to 100 GB). Reviewers need the preprocessed AnnData (a few GB), not the raw bundles. If a reviewer wants to reproduce preprocessing too, they can download the raw Xenium runs from GEO / SRA / your institutional repository and run `nicheverse preprocess` themselves.

## Verifying the bundle yourself

Before publishing, do a clean room test on a different machine:

```bash
python -m venv /tmp/repro_env
source /tmp/repro_env/bin/activate
bash verify_reproducibility.sh
```

If this prints `exit code 0` you are ready to upload. If it prints `exit code 2`, see [determinism](determinism.md) to diagnose the source of the discrepancy.
