# nicheverse annotate: agentic codebook-annotation harness

This subpackage turns the unsupervised codes learned by a nicheverse VQ-VAE (cell
codes and neighborhood / niche codes) into literature-defensible cell-type and niche
labels. It wraps a language model in a set of deterministic guardrails so that a label
is only accepted when the code's own measured evidence supports it.

Scope: imaging-based / in-situ spatial transcriptomics (IMST) only, for example Xenium,
CosMx, MERFISH / MERSCOPE, seqFISH, STARmap, and ISS. Sequencing / array platforms
(Visium, Slide-seq, Stereo-seq, GeoMx) are out of scope and are rejected when a project
context declares them.

## Why a plain linear pipeline (and not a multi-agent platform)

We surveyed the recent spatial-annotation-agent systems (SpatialAgent, STAgent,
STAT-agent, CASSIA, CellTypeAgent, NicheAgent, STAnalyzer). The recurring failure modes
in that line of work are hallucinated markers, fabricated citations, non-reproducible
runs, and opaque control flow that is hard to audit. Our conclusion was that a single
fixed-order pipeline with hard verification gates and one adversarial pass answers those
failure modes more directly than a graph of cooperating agents. So the harness has no
tool registry, no agent-routing graph, and no background services: the stages run in a
known order and every decision is written down.

## Pipeline stages

For each code the harness runs:

1. Evidence. `artifacts.code_evidence` (cells) or `artifacts.niche_evidence` (niches)
   reduces the code to top markers by z-score, one-vs-rest differential expression,
   cell-type composition (niches), and its distribution over metadata columns.
2. Propose. `annotate.propose_label` asks the labeler for a ranked list of up to
   `n_candidates` candidate labels, each with its supporting markers, plus the single
   best label. The project context and the allowed vocabulary are placed next to the
   code's own evidence, not only in a global preamble.
3. Verify gates. `verify.gate` runs the two anti-hallucination checks below and applies
   the lab rules. A rule-adjusted label is adopted, and the labeler's stated confidence
   is reduced by the gate's penalty.
4. Adversarial refuter. `annotate.refute_label` runs a second model whose only job is to
   argue the label is wrong. If it proposes an alternative whose discriminating markers
   are better supported in this code's own evidence, the label flips; otherwise the
   disagreement is recorded and the code is routed to review.
5. Reconcile (cross-code). The existing refine pass disambiguates identical labels and
   merges only true duplicates; then the harness flags any two codes whose marker
   z-profiles are near-identical but whose labels differ, and checks hierarchical-cluster
   consistency so a code that disagrees with its cluster's majority label is flagged.
6. Review split. Codes that pass the gate, are agreed by the refuter, and clear the
   confidence threshold are auto-accepted. Everything else goes to a review table with a
   reason.
7. Provenance and scoring. Every code contributes a record (an evidence content hash, the
   resolved prompt, the raw model output, the gate result, the refuter verdict, the
   citations, and the final label). When a ground-truth column is supplied, each code is
   scored and the run is calibrated. With an output directory, a replayable provenance
   manifest and a scorecard CSV are written.

## The two anti-hallucination gates

Marker presence. A cited marker only counts as support if it is actually enriched in the
code, that is, it appears in the code's top markers at or above the z threshold, or as a
positive one-vs-rest DEG. If too few of the cited markers clear that bar (precision below
`min_marker_precision`), the gate fails the label and applies a confidence penalty. This
catches both invented markers and off-lineage signal bleeding in from adjacent cells
(segmentation leakage), which is a routine artifact in imaging-based data.

Citation resolution. Each citation is parsed for a DOI or PMID; an injectable resolver
looks it up. A parseable identifier that resolves to nothing is treated as fabricated and
fails the gate. An identifier that simply cannot be checked offline is left as unknown and
does not fail the gate, so the check never falsely accuses a real reference. The resolver
is injectable so the whole harness runs offline in tests.

## The adversarial refuter (distinct objective)

The refuter is not a second opinion that tends to agree. Its system instruction gives it
the opposite objective from the labeler: argue that the proposed label is wrong, name the
markers the label needs but lacks, and point to a better-supported alternative from the
allowed set. Crucially it is blind to the labeler's confidence. Showing the refuter how
sure the labeler was would let it anchor on that number and collapse into agreement, so
the confidence is never included in the refuter prompt. A refuter revision only changes
the label when its markers survive the same marker-presence check the labeler had to pass;
an unsupported objection is flagged, not obeyed.

## Project context and lab rules

`context.ProjectContext` carries structured knowledge about the study (species, disease,
tissue, platform, panel, sites, expected cell types with markers, expected niches, and
authoritative references). It supplies the allowed candidate vocabulary and a
`novel / uncertain` escape hatch so the model is never forced to pick a wrong label.
`context.ANNOTATION_RULES` encodes the lab's codebook conventions as text (universal
phenotype, rare-patient caution, site-aware reassignment, composite-label gating,
segmentation leakage); the executable parts of those rules live in `verify.apply_lab_rules`
and fire inside the gate.

## CLI

Run as `python -m nicheverse.annotate <command>`:

    # write a project context template to fill in
    python -m nicheverse.annotate context-template project_context.yaml

    # dump the per-code evidence bundle (CSVs) for manual review
    python -m nicheverse.annotate evidence \
        --adata cohort.h5ad --code-col cell_codebook_idx --out evidence_out

    # run the full harness (cell codes)
    python -m nicheverse.annotate annotate \
        --adata cohort.h5ad --code-col cell_codebook_idx \
        --context project_context.yaml --kind cell --provider anthropic \
        --groundtruth-col leiden_label --out annot_out

    # run the harness on niche codes (needs cell labels for composition)
    python -m nicheverse.annotate annotate \
        --adata cohort.h5ad --code-col neighborhood_codebook_idx \
        --celltype-col celltype_annot --context project_context.yaml \
        --kind niche --out niche_out

    # score an existing per-code label table against a ground-truth column
    python -m nicheverse.annotate evaluate \
        --labels annot_out/labels.csv --code-col cell_codebook_idx \
        --adata cohort.h5ad --groundtruth leiden_label

Use `--no-refuter` to skip the adversarial pass. Both `--kind cell` and `--kind niche`
run through the same code path.

## Python

```python
from nicheverse.annotate import AnnotationConfig, ProjectContext
from nicheverse.annotate.harness import annotate_codebook

ctx = ProjectContext.from_yaml("project_context.yaml")
cfg = AnnotationConfig(provider="anthropic", context=ctx, refuter=True)

res = annotate_codebook(
    adata, "cell_codebook_idx",
    config=cfg, kind="cell",
    groundtruth_col="leiden_label",
    out_dir="annot_out",
)

res.labels_df        # code -> final_label, compartment, confidence, passed, refuter_agree, flags
res.review_df        # codes needing manual sign-off, with a review_reason
res.scorecard_df     # per-code grading (when a ground-truth column was given)
res.manifest_path    # replayable provenance manifest

res.attach(adata, key_added="celltype_annot")  # write final labels onto obs
```

All library code writing / editing this harness is offline-capable: with a monkeypatched
`providers.call_llm` and an injected citation resolver the full pipeline runs without any
network or GPU.
