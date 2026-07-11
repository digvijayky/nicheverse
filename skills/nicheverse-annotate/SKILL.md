---
name: nicheverse-annotate
description: >-
  Annotate nicheverse codebook codes into cell types / states (and spatial
  niches) from imaging spatial transcriptomics (Xenium, MERFISH, CosMx, seqFISH),
  grounded in per-code marker/DEG evidence and primary literature. Use this when
  a user has (or wants to produce) a nicheverse hierarchical VQ-VAE codebook
  assignment (obs['cell_codebook_idx'] / obs['neighborhood_codebook_idx']) and
  needs each code labeled with a defensible cell type/state, compartment,
  confidence, rationale, and PubMed citations. Also use to run the full
  load -> predict_codes -> annotate pipeline end to end.
---

# nicheverse-annotate

Turn a **nicheverse** codebook (discrete cell-state and spatial-niche codes learned by a
hierarchical VQ-VAE) into biologically meaningful labels. Every code is annotated by an LLM
that is **grounded in the code's own expression evidence** (top markers by z-score, one-vs-rest
DEGs, and metadata distributions) plus **primary literature** (PubMed / bioRxiv). This mirrors a
manual codebook-review pipeline: no label rests on training-data familiarity alone.

Environment python: `/path/to/your/env/bin/python` (an environment that has `nicheverse`
installed). The LLM/network features need the `nicheverse[llm]` extra (`anthropic`, `openai`,
`requests`).

## When to use

- The user has an AnnData with `obs['cell_codebook_idx']` and/or `obs['neighborhood_codebook_idx']`
  (from `nicheverse.predict_codes` or `nicheverse.Trainer`) and wants each code named.
- The user has raw imaging ST and a trained checkpoint and wants the whole thing: load ->
  assign codes -> annotate.
- The user asks to "annotate the codebook", "label the codes / states / niches", "what cell
  type is each code", or to attach a `celltype_annot` column.

Do NOT invent functions. The only annotation entry points are the ones below.

## The workflow (exact API)

Use `import nicheverse as nv`; annotation helpers live in `nicheverse.annotate`.

### 1. Load any imaging ST into AnnData

```python
adata = nv.read_spatial(
    path,                 # an AnnData or an .h5ad path
    sample_col="sample_id",
    spatial_key="spatial",
    x_col=None, y_col=None,   # obs columns to build obsm['spatial'] from, if absent
    coord_scale=1.0,          # pixels -> microns, e.g. 0.12028 for CosMx
)
```

`read_spatial` guarantees `obsm['spatial']` holds micron coordinates and `obs[sample_col]` exists.
For Xenium you can instead use `nv.read_xenium` / `nv.read_xenium_cohort`.

### 2. Assign codes (skip if the code column already exists)

```python
adata = nv.predict_codes(adata, checkpoint_path)   # e.g. "ckpt/hierarchical_vqvae_checkpoint.pt"
# -> obs['cell_codebook_idx'] (0..255) and obs['neighborhood_codebook_idx'] (0..31)
```

`predict_codes` needs `obsm['spatial']` and `obs[sample_col]`. It normalizes/log1p by default
(set `normalize=False, log1p=False` if already log-normalized), and `k_neighbors` /
`neighborhood_aggregation` must match the training-time values. To train first, use
`nv.Trainer(nv.TrainConfig()).fit(adata, "ckpt/", model_config=...)`.

### 3. Generate per-code evidence

```python
from nicheverse.annotate import code_evidence
ev = code_evidence(adata, "cell_codebook_idx", extra_cols=("site_class", "sample_id"))
# {code: {code, n_cells, frac, top_markers[(gene,z)], top_degs[(gene,log2fc,padj)], dist_<col>}}
```

`annotate_codes` (step 4) calls this internally, so you rarely call it directly; use it to
inspect or QC the evidence a code was labeled from.

### 4. Annotate with an LLM grounded in evidence + literature

```python
from nicheverse.annotate import annotate_codes
df = annotate_codes(
    adata, "cell_codebook_idx",
    provider="anthropic",          # "anthropic" | "openai" | "ollama"
    model=None,                    # None -> provider default (Claude Opus / GPT-4o / llama3)
    tissue="human clear cell RCC", # free-text tissue/disease context
    with_literature=True,          # PubMed-search each code's top markers, feed hits to the LLM
    context_cols=("site_class",),  # obs columns summarized per code
    refine=True,                   # second pass reconciles labels across codes
)
```

Returns a DataFrame indexed by code with columns: `n_cells`, `label`, `compartment`
(epithelial/immune/stromal/vascular/neural/other), `confidence` (0-1), `rationale`,
`key_markers`, `citations` (PMIDs), and `label_refined` (the reconciled label). Options beyond
the keywords above live on `nicheverse.annotate.AnnotationConfig` (`top_markers`, `top_degs`,
`marker_context`, `api_key`); pass an `AnnotationConfig` as the 3rd positional arg for full control.

### 5. Attach labels and save

```python
from nicheverse.annotate import attach_labels
adata = attach_labels(adata, "cell_codebook_idx", df, key_added="celltype_annot")
df.to_csv("code_labels.csv")
adata.write_h5ad("annotated.h5ad")
```

`attach_labels` maps `label_refined` (or `label`) onto `obs[key_added]` as a category; unlabeled
codes become `"unknown"`.

### 6. Annotate spatial niches (neighborhood codes)

A spatial niche (neighborhood code) is defined by the community of cell types that co-occur in
it, so its annotation is grounded in **cell-type composition** rather than markers alone. Run this
**after** the cell codes are labeled and attached, so the composition draws on real cell-type
names:

```python
from nicheverse.annotate import niche_evidence, annotate_niches
ev = niche_evidence(
    adata, "neighborhood_codebook_idx", "celltype_annot",
    extra_cols=("site_class",), top_markers=20,
)
# {niche: {code, n_cells, frac, composition [(cell_type, fraction)], top_markers [(gene, z)], dist_<col>}}

ndf = annotate_niches(
    adata, "neighborhood_codebook_idx", "celltype_annot",
    provider="anthropic", model=None, tissue="human clear cell RCC",
    context_cols=("site_class",), top_markers=20,
)
```

`annotate_niches` calls `niche_evidence` internally, then asks the LLM to name each niche from its
composition. It returns a DataFrame indexed by niche code with columns `n_cells`, `label`,
`dominant_types`, `confidence`, `rationale`, and `composition`. `celltype_col` must be a cell-level
label column (typically the `key_added` from step 5). Save it the same way: `ndf.to_csv(...)`.

### 7. Cluster codes for coarse-to-fine review

To group similar codes (cell or niche) before or after labeling, use `cluster_codes`, which
hierarchically clusters codes by mean-expression correlation (correlation distance + average
linkage), mirroring the manual codebook-review pipeline:

```python
from nicheverse.annotate import cluster_codes
clusters = cluster_codes(adata, "cell_codebook_idx", n_clusters=None)  # None -> ~1 cluster per 8 codes
# DataFrame indexed by code with an integer 'cluster' column
```

## Providers and API keys

`provider` is one of `anthropic` (Claude), `openai` (GPT), or `ollama` (a local Ollama /
OpenAI-compatible endpoint). Set the key via environment before running:

- `anthropic` -> `ANTHROPIC_API_KEY`
- `openai` -> `OPENAI_API_KEY`
- `ollama` -> no key; set `OLLAMA_HOST` (default `http://localhost:11434`)

You can override the key per call with `AnnotationConfig(api_key=...)`. The literature step uses
NCBI E-utilities (optionally an NCBI `api_key`) and degrades to empty results offline, so
annotation never hard-crashes without network.

## Grounding requirement (non-negotiable)

Every proposed label MUST be backed by the code's markers/DEGs AND at least one citation. The
system prompt already forces this, and `with_literature=True` supplies PubMed hits, but always
verify before trusting a call:

1. When `confidence` is low or `citations` is empty, inspect `code_evidence(...)` for that code
   and treat the label as tentative in any report.
2. Prefer a specific, defensible label over a vague one; if evidence is ambiguous, keep the
   uncertainty in the rationale rather than overcalling.
3. For a site-restricted label appearing at a non-permissive site (e.g. Microglia in a kidney
   primary), flag it as likely segmentation leakage / contamination rather than a real call.

## Runnable CLI

`scripts/annotate_codebook.py` runs steps 2-6 end to end and writes the label CSV + annotated
h5ad. Always use the env python.

```bash
/path/to/your/env/bin/python \
  skills/nicheverse-annotate/scripts/annotate_codebook.py \
    --adata /path/to/adata.h5ad \
    --code-col cell_codebook_idx \
    --checkpoint /path/to/hierarchical_vqvae_checkpoint.pt \
    --provider anthropic --tissue "human clear cell RCC" \
    --with-literature --context-cols site_class,sample_id \
    --niche-col neighborhood_codebook_idx \
    --out-csv out/code_labels.csv --out-h5ad out/adata_annotated.h5ad
```

Omit `--checkpoint` to annotate a code column that already exists in `obs`. Pass `--niche-col`
(e.g. `neighborhood_codebook_idx`) to also annotate spatial niches after the cell labels are
attached; the niche table is written to `<out-csv stem>_niches.csv`. Niches are composed over
`--celltype-col`, which defaults to the `--key-added` cell-label column. Run
`python scripts/annotate_codebook.py --help` for all flags.

## Codex / plain-instructions fallback

If running outside Claude Code, hand the model these same 5 steps and the CLI above verbatim.
The pipeline is a plain Python API plus one argparse script, so a coding agent with the env
python can execute it without any Claude Code specific tooling.
