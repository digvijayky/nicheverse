# Annotating codes

nicheverse turns tissue into two codebooks: cell states and spatial niches. The
{mod}`nicheverse.annotate` module labels those codes, grounded in expression
evidence and, optionally, primary literature, using an LLM (Claude, GPT, or a
local model).

Install the optional backends: `pip install nicheverse[llm]`.

## Cell states

`adata` already carries `obs['cell_codebook_idx']` (from `predict_codes` or `train_model`).

```python
import nicheverse as nv
from nicheverse.annotate import code_evidence, annotate_codes, attach_labels

evidence = code_evidence(adata, "cell_codebook_idx", extra_cols=("site_class", "sample_id"))
labels = annotate_codes(
    adata,
    "cell_codebook_idx",
    provider="anthropic",        # or "openai", "ollama"
    tissue="human clear cell RCC, brain metastasis",
    with_literature=True,        # PubMed-backed calls
    context_cols=("site_class",),
    refine=True,                 # reconcile labels across codes
)
attach_labels(adata, "cell_codebook_idx", labels, key_added="celltype_annot")
```

Each row carries `label`, `compartment`, `confidence`, `rationale`, `key_markers`,
`citations`, and a reconciled `label_refined`.

## Spatial niches

Niches are named for the community of cell types they contain, so annotate cells
first, then pass those labels in:

```python
from nicheverse.annotate import annotate_niches

niches = annotate_niches(
    adata, "neighborhood_codebook_idx", "celltype_annot",
    provider="anthropic", tissue="human clear cell RCC",
)
```

## Coarse to fine

Group similar codes before annotating, for a hierarchical review that mirrors the
manual codebook pipeline:

```python
from nicheverse.annotate import cluster_codes

clusters = cluster_codes(adata, "cell_codebook_idx", n_clusters=30)
```

## Providers and keys

`provider="anthropic"` reads `ANTHROPIC_API_KEY`, `"openai"` reads `OPENAI_API_KEY`,
and `"ollama"` calls a local server (`OLLAMA_HOST`, default `http://localhost:11434`).
Every label should be defensible from the markers/DEGs shown and, when literature is
on, an accompanying citation.

## From an agent

Claude Code and Codex can drive the whole workflow through the bundled MCP server
(see {doc}`mcp`) and the `nicheverse-annotate` skill.
