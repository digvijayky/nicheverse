"""LLM-based iterative annotation of learned codes.

Turns each code's expression evidence (markers, DEGs, distributions) and optional
literature into a cell-type / state label, using Claude, GPT, or a local model.
An optional second pass reconciles labels across codes for consistency.
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import pandas as pd

from .artifacts import code_evidence, niche_evidence
from .literature import literature_for_markers
from .providers import call_llm, parse_json

__all__ = ["AnnotationConfig", "annotate_codes", "annotate_niches", "attach_labels"]

_SYSTEM = (
    "You are a spatial-transcriptomics expert annotating unsupervised codes (clusters) into "
    "specific cell types or cell states. Ground every call in the marker/DEG evidence provided, "
    "and in the literature when given. Prefer a specific, defensible label over a vague one; if "
    "the evidence is ambiguous, say so in the rationale and lower the confidence. Reply with STRICT JSON only."
)


@dataclass
class AnnotationConfig:
    """Settings for :func:`annotate_codes`.

    Parameters
    ----------
    provider, model
        LLM backend (``anthropic`` / ``openai`` / ``ollama``) and model id.
    tissue
        Free-text tissue / disease context (e.g. "human clear cell RCC, brain metastasis").
    with_literature
        If ``True``, search PubMed for the top markers of each code and include the hits.
    context_cols
        ``obs`` columns to summarize per code (e.g. ``("site_class", "sample_id")``).
    refine
        Run a second pass that reconciles labels across codes (resolves duplicates / conflicts).
    cluster_context
        Hierarchically cluster codes first and give the reconciliation pass that structure,
        so similar codes get consistent compartments and distinguishing qualifiers.
    """

    provider: str = "anthropic"
    model: str | None = None
    tissue: str = ""
    with_literature: bool = False
    context_cols: tuple[str, ...] = ()
    marker_context: str = "cell type marker"
    top_markers: int = 25
    top_degs: int = 20
    refine: bool = True
    cluster_context: bool = False
    api_key: str | None = None


def _evidence_prompt(code: str, ev: dict, lit: dict | None, tissue: str) -> str:
    lines = [
        f"Tissue/context: {tissue or 'unspecified'}.",
        f"Code {code}: {ev['n_cells']} cells ({ev['frac'] * 100:.1f}% of the dataset).",
        "Top markers (gene, z-score across codes): "
        + ", ".join(f"{g}({z})" for g, z in ev["top_markers"][:20]),
    ]
    if ev.get("top_degs"):
        lines.append(
            "Top 1-vs-rest DEGs (gene, log2FC): " + ", ".join(f"{g}({lfc})" for g, lfc, _ in ev["top_degs"][:15])
        )
    for k, v in ev.items():
        if k.startswith("dist_"):
            lines.append(f"Distribution over {k[5:]}: {v}")
    if lit:
        lines.append("Literature for the top markers:")
        for g, refs in lit.items():
            for r in refs[:2]:
                lines.append(
                    f"  {g}: {r.get('title', '')[:130]} ({r.get('journal', '')} {r.get('year', '')}, PMID {r.get('pmid', '')})"
                )
    lines.append(
        'Return JSON: {"label": "<specific cell type or state>", "compartment": "<epithelial|immune|stromal|'
        'vascular|neural|other>", "confidence": <0-1>, "rationale": "<=40 words citing the markers", '
        '"key_markers": ["..."], "citations": ["PMID:<id> <first-author> <journal> <year>"]}'
    )
    return "\n".join(lines)


def _refine_prompt(table: dict, tissue: str) -> str:
    has_cluster = any("cluster" in d for d in table.values())
    body = "\n".join(
        f"  {c}: "
        + (f"cluster={d.get('cluster', '-')} " if has_cluster else "")
        + f"label={d.get('label', '')!r}, compartment={d.get('compartment', '')!r}, markers={d.get('key_markers', '')}"
        for c, d in table.items()
    )
    cluster_note = (
        " Codes sharing a cluster are transcriptionally similar: give them a consistent compartment and "
        "distinguish them with qualifiers rather than identical labels."
        if has_cluster
        else ""
    )
    return (
        f"Tissue/context: {tissue or 'unspecified'}. Below are per-code labels proposed independently. "
        "Reconcile them: disambiguate identical labels that are really distinct states (e.g. add a qualifier), "
        f"merge only when truly the same, and fix any label that conflicts with its markers.{cluster_note}\n"
        f"{body}\n"
        'Return JSON: {"revisions": {"<code>": "<final label>", ...}} including ONLY codes you changed.'
    )


def annotate_codes(
    adata: ad.AnnData, code_col: str, config: AnnotationConfig | None = None, **kwargs
) -> pd.DataFrame:
    """Annotate every code in ``adata.obs[code_col]`` with an LLM, grounded in evidence.

    Returns a DataFrame indexed by code with ``label`` / ``compartment`` /
    ``confidence`` / ``rationale`` / ``key_markers`` / ``citations`` (and
    ``label_refined`` when ``config.refine``).
    """
    cfg = config or AnnotationConfig(**kwargs)
    ev = code_evidence(adata, code_col, extra_cols=cfg.context_cols, top_markers=cfg.top_markers, top_degs=cfg.top_degs)
    rows = []
    for code, e in ev.items():
        lit = (
            literature_for_markers([g for g, _ in e["top_markers"][:5]], context=cfg.marker_context, api_key=cfg.api_key)
            if cfg.with_literature
            else None
        )
        out = call_llm(
            _evidence_prompt(code, e, lit, cfg.tissue),
            provider=cfg.provider, model=cfg.model, api_key=cfg.api_key, system=_SYSTEM,
        )
        j = parse_json(out)
        rows.append(
            {
                "code": code, "n_cells": e["n_cells"], "label": j.get("label", ""),
                "compartment": j.get("compartment", ""), "confidence": j.get("confidence"),
                "rationale": j.get("rationale", ""),
                "key_markers": "; ".join(map(str, j.get("key_markers", []))),
                "citations": "; ".join(map(str, j.get("citations", []))),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["n_cells", "label", "compartment", "confidence", "rationale", "key_markers", "citations", "label_refined"]
        )
    df = pd.DataFrame(rows).set_index("code")
    df["label_refined"] = df["label"]
    if cfg.cluster_context and len(df) > 1:
        from .artifacts import cluster_codes

        cl = cluster_codes(adata, code_col)
        df["cluster"] = [int(cl.loc[c, "cluster"]) if c in cl.index else -1 for c in df.index]
    if cfg.refine and len(df) > 1:
        keep = ["label", "compartment", "key_markers"] + (["cluster"] if "cluster" in df.columns else [])
        tbl = df[keep].to_dict("index")
        rj = parse_json(
            call_llm(_refine_prompt(tbl, cfg.tissue), provider=cfg.provider, model=cfg.model, api_key=cfg.api_key, system=_SYSTEM)
        )
        for code, newlab in (rj.get("revisions", {}) or {}).items():
            if code in df.index:
                df.loc[code, "label_refined"] = newlab if isinstance(newlab, str) else str(newlab)
    return df


def attach_labels(adata: ad.AnnData, code_col: str, labels, key_added: str = "celltype_annot") -> ad.AnnData:
    """Map a ``code -> label`` mapping (dict or annotate_codes DataFrame) onto ``obs[key_added]``."""
    if isinstance(labels, pd.DataFrame):
        col = "label_refined" if "label_refined" in labels.columns else "label"
        labels = labels[col].to_dict()
    labels = {str(k): v for k, v in labels.items()}
    adata.obs[key_added] = (
        adata.obs[code_col].astype(str).map(lambda c: labels.get(c, "unknown")).astype("category")
    )
    return adata


_NICHE_SYSTEM = (
    "You are a spatial-transcriptomics expert naming multicellular spatial niches (tissue "
    "microenvironments) by the community of cell types they contain and their marker enrichment. "
    "A niche is named for its composition and tissue context (e.g. 'tumor-immune boundary', "
    "'stroma-rich niche'), not a single cell type. Reply with STRICT JSON only."
)


def _niche_prompt(code: str, ev: dict, tissue: str) -> str:
    lines = [
        f"Tissue/context: {tissue or 'unspecified'}.",
        f"Niche {code}: {ev['n_cells']} cells ({ev['frac'] * 100:.1f}% of the dataset).",
        "Cell-type composition (type, fraction): " + ", ".join(f"{t}({f})" for t, f in ev["composition"]),
        "Enriched genes (z across niches): " + ", ".join(f"{g}({z})" for g, z in ev["top_markers"][:15]),
    ]
    for k, v in ev.items():
        if k.startswith("dist_"):
            lines.append(f"Distribution over {k[5:]}: {v}")
    lines.append(
        'Return JSON: {"label": "<niche / microenvironment name>", "dominant_types": ["..."], '
        '"confidence": <0-1>, "rationale": "<=40 words on the composition"}'
    )
    return "\n".join(lines)


def annotate_niches(
    adata: ad.AnnData,
    niche_col: str,
    celltype_col: str,
    config: AnnotationConfig | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Annotate spatial-niche codes by their cell-type composition with an LLM.

    Requires ``celltype_col`` (cell-level labels, e.g. from :func:`annotate_codes`).
    Returns a DataFrame indexed by niche code with ``label`` / ``dominant_types`` /
    ``confidence`` / ``rationale`` / ``composition``.
    """
    cfg = config or AnnotationConfig(**kwargs)
    ev = niche_evidence(adata, niche_col, celltype_col, extra_cols=cfg.context_cols, top_markers=cfg.top_markers)
    rows = []
    for code, e in ev.items():
        j = parse_json(
            call_llm(_niche_prompt(code, e, cfg.tissue), provider=cfg.provider, model=cfg.model,
                     api_key=cfg.api_key, system=_NICHE_SYSTEM)
        )
        rows.append(
            {
                "code": code, "n_cells": e["n_cells"], "label": j.get("label", ""),
                "dominant_types": "; ".join(map(str, j.get("dominant_types", []))),
                "confidence": j.get("confidence"), "rationale": j.get("rationale", ""),
                "composition": "; ".join(f"{t}:{f}" for t, f in e["composition"][:5]),
            }
        )
    return pd.DataFrame(rows).set_index("code")
