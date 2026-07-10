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
from .context import ANNOTATION_RULES
from .literature import literature_for_markers
from .providers import call_llm, parse_json

__all__ = [
    "AnnotationConfig",
    "annotate_codes",
    "annotate_niches",
    "attach_labels",
    "propose_label",
    "refute_label",
]

_SYSTEM = (
    "You are a spatial-transcriptomics expert annotating unsupervised codes (clusters) into "
    "specific cell types or cell states. Ground every call in the marker/DEG evidence provided, "
    "and in the literature when given. Prefer a specific, defensible label over a vague one; if "
    "the evidence is ambiguous, say so in the rationale and lower the confidence. Reply with STRICT JSON only."
)

# The adversarial refuter is a distinct agent whose only job is to try to knock the proposed
# label down: it argues from the SAME per-code evidence that the label is wrong and names the
# best alternative. Keeping this objective opposed to the labeler's is what makes the pass a real
# check rather than an agreement echo. It is deliberately never shown the labeler's confidence, so
# it cannot anchor on how sure the labeler was.
_REFUTER_SYSTEM = (
    "You are a skeptical spatial-transcriptomics reviewer. Your ONLY objective is to argue that the "
    "proposed cell-type / niche label for this code is WRONG. Scrutinize the marker and DEG evidence: "
    "point out markers the label needs but that are missing, off-lineage signal it ignores, and any "
    "better-supported alternative from the allowed set. Do not agree out of politeness; if the label "
    "is genuinely the best fit say so, but default to challenging it. Reply with STRICT JSON only."
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
    # -- harness / grounding additions -------------------------------------
    # Structured study context; when set it supplies the grounding block + the allowed
    # candidate vocabulary (see :func:`_context_block`). ``None`` falls back to ``tissue``.
    context: "object | None" = None
    # Run the adversarial refuter pass after each proposal.
    refuter: bool = True
    # Ask the labeler for up to this many ranked candidate labels (plus a single best).
    n_candidates: int = 3
    # Marker-precision floor handed to :func:`verify.gate`.
    min_marker_precision: float = 0.5
    # Codes with final confidence below this are routed to manual review rather than auto-accepted.
    confidence_review_threshold: float = 0.6


def _context_block(cfg: "AnnotationConfig", kind: str = "cell") -> str:
    """Grounding text injected next to each code's evidence.

    When ``cfg.context`` is a :class:`~nicheverse.annotate.context.ProjectContext`, this returns
    its ``to_prompt_block()`` followed by the lab's ``ANNOTATION_RULES`` and the explicit allowed
    candidate set (expected cell types for ``kind='cell'``, expected niches for ``kind='niche'``)
    with a 'novel / uncertain' escape hatch. With no context it degrades to the bare ``tissue``
    string so the legacy prompts are unchanged.
    """
    ctx = getattr(cfg, "context", None)
    if ctx is None:
        return f"Tissue/context: {cfg.tissue or 'unspecified'}."
    if kind == "niche":
        names = [getattr(n, "name", "") for n in getattr(ctx, "expected_niches", []) or []]
        heading = "Allowed niche labels (choose one, or 'novel/uncertain' if none fits)"
    else:
        names = [getattr(c, "name", "") for c in getattr(ctx, "expected_cell_types", []) or []]
        heading = "Allowed cell-type labels (choose one, or 'novel/uncertain' if none fits)"
    names = [n for n in names if n]
    allowed = (
        heading + ": " + "; ".join(names) + "; novel/uncertain"
        if names
        else "No fixed label vocabulary was provided; propose the best defensible label, "
        "or 'novel/uncertain' when the evidence is ambiguous."
    )
    return ctx.to_prompt_block() + "\n" + ANNOTATION_RULES + "\n" + allowed


def _format_literature(lit: dict | None) -> list[str]:
    """Short attributed per-claim snippets (marker -> title/journal/year/PMID).

    Never emits raw abstract text: passing full abstracts makes the model over-anchor on a single
    paper, so only the title (trimmed) and the source's attribution fields are surfaced.
    """
    if not lit:
        return []
    lines = ["Literature snippets for the top markers (attribution only, not evidence of enrichment):"]
    for g, refs in lit.items():
        for r in (refs or [])[:2]:
            title = str(r.get("title", "") or "")[:130]
            lines.append(
                f"  {g}: {title} ({r.get('journal', '')} {r.get('year', '')}, PMID {r.get('pmid', '')})"
            )
    return lines


def _candidate_json_spec(n_candidates: int, kind: str) -> str:
    """The strict-JSON reply schema shared by the cell and niche labelers.

    A single JSON object (``providers.parse_json`` keeps only the first ``{...}``) carrying a
    RANKED ``candidates`` array and the single ``best`` label with its supporting evidence.
    """
    comp = (
        '"compartment": "<epithelial|immune|stromal|vascular|neural|other>", '
        if kind == "cell"
        else '"dominant_types": ["..."], '
    )
    return (
        f'Return JSON with a RANKED list of up to {n_candidates} candidate labels (most likely '
        'first), each with the markers that support it, and the single best label:\n'
        '{"candidates": [{"label": "<candidate>", "supporting_markers": ["..."], '
        '"confidence": <0-1>}, ...], '
        f'"label": "<the single best label>", {comp}'
        '"confidence": <0-1>, "rationale": "<=40 words citing the markers", '
        '"key_markers": ["..."], "citations": ["PMID:<id> <first-author> <journal> <year>"]}'
    )


def _evidence_prompt(
    code: str,
    ev: dict,
    lit: dict | None,
    tissue: str,
    *,
    context_block: str | None = None,
    n_candidates: int = 1,
) -> str:
    # The grounding block (project context + rules + allowed labels, or the bare tissue string)
    # sits ADJACENT to this code's own evidence so the model reads them together, not as a
    # detached global preamble.
    head = context_block if context_block is not None else f"Tissue/context: {tissue or 'unspecified'}."
    lines = [
        head,
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
    lines.extend(_format_literature(lit))
    if n_candidates and n_candidates > 1:
        lines.append(_candidate_json_spec(n_candidates, "cell"))
    else:
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


def _niche_prompt(
    code: str,
    ev: dict,
    tissue: str,
    *,
    context_block: str | None = None,
    n_candidates: int = 1,
) -> str:
    head = context_block if context_block is not None else f"Tissue/context: {tissue or 'unspecified'}."
    lines = [
        head,
        f"Niche {code}: {ev['n_cells']} cells ({ev['frac'] * 100:.1f}% of the dataset).",
        "Cell-type composition (type, fraction): " + ", ".join(f"{t}({f})" for t, f in ev["composition"]),
        "Enriched genes (z across niches): " + ", ".join(f"{g}({z})" for g, z in ev["top_markers"][:15]),
    ]
    for k, v in ev.items():
        if k.startswith("dist_"):
            lines.append(f"Distribution over {k[5:]}: {v}")
    if n_candidates and n_candidates > 1:
        lines.append(_candidate_json_spec(n_candidates, "niche"))
    else:
        lines.append(
            'Return JSON: {"label": "<niche / microenvironment name>", "dominant_types": ["..."], '
            '"confidence": <0-1>, "rationale": "<=40 words on the composition"}'
        )
    return "\n".join(lines)


def _refuter_prompt(code: str, ev: dict, proposed_label: str, context_text: str, kind: str = "cell") -> str:
    """Prompt for the adversarial refuter: SAME per-code evidence, allowed set, no confidence.

    The refuter sees exactly the measured evidence and the label under challenge but never the
    labeler's stated confidence (that would let it anchor). It returns keep/revise plus the
    discriminating markers it would use to tell the alternative apart from the proposal.
    """
    lines = [context_text, f"Code {code}: {ev.get('n_cells', 0)} cells."]
    if kind == "niche":
        comp = ev.get("composition", []) or []
        lines.append("Cell-type composition (type, fraction): " + ", ".join(f"{t}({f})" for t, f in comp))
        lines.append(
            "Enriched genes (z across niches): "
            + ", ".join(f"{g}({z})" for g, z in ev.get("top_markers", [])[:15])
        )
    else:
        lines.append(
            "Top markers (gene, z-score across codes): "
            + ", ".join(f"{g}({z})" for g, z in ev.get("top_markers", [])[:20])
        )
        if ev.get("top_degs"):
            lines.append(
                "Top 1-vs-rest DEGs (gene, log2FC): "
                + ", ".join(f"{g}({lfc})" for g, lfc, _ in ev["top_degs"][:15])
            )
    for k, v in ev.items():
        if k.startswith("dist_"):
            lines.append(f"Distribution over {k[5:]}: {v}")
    lines.append(f"PROPOSED LABEL UNDER CHALLENGE: {proposed_label!r}.")
    lines.append(
        "Argue whether this label is wrong given ONLY the evidence above and the allowed set. "
        'Return JSON: {"verdict": "keep"|"revise", "alternative_label": "<a better allowed label or '
        'empty if keep>", "discriminating_markers": ["<markers separating the alternative from the '
        'proposal>"], "reason": "<=40 words"}'
    )
    return "\n".join(lines)


def propose_label(code, ev: dict, cfg: "AnnotationConfig", kind: str = "cell") -> dict:
    """Ask the labeler for a ranked candidate set + the single best label for one code.

    Reuses :func:`providers.call_llm` + :func:`providers.parse_json`. The returned dict carries the
    best-label fields the harness / gate consume (``label``, ``key_markers``, ``citations``,
    ``compartment`` (cell) or ``dominant_types`` (niche), ``confidence``, ``rationale``) plus the
    ranked ``candidates`` list. On an unparseable reply the raw text is kept under ``_raw``.
    """
    block = _context_block(cfg, kind)
    if kind == "niche":
        prompt = _niche_prompt(code, ev, cfg.tissue, context_block=block, n_candidates=cfg.n_candidates)
        system = _NICHE_SYSTEM
    else:
        lit = (
            literature_for_markers(
                [g for g, _ in ev.get("top_markers", [])[:5]], context=cfg.marker_context, api_key=cfg.api_key
            )
            if cfg.with_literature
            else None
        )
        prompt = _evidence_prompt(code, ev, lit, cfg.tissue, context_block=block, n_candidates=cfg.n_candidates)
        system = _SYSTEM
    raw = call_llm(prompt, provider=cfg.provider, model=cfg.model, api_key=cfg.api_key, system=system)
    j = parse_json(raw)
    j.setdefault("candidates", [])
    j["_raw"] = raw
    j["_prompt"] = prompt
    return j


def refute_label(code, ev: dict, proposed: dict, cfg: "AnnotationConfig", kind: str = "cell") -> dict:
    """Run the adversarial refuter on a proposed label (blind to its confidence).

    ``proposed`` is the labeler dict; only its ``label`` is passed on, never its confidence, so the
    refuter cannot anchor. Returns ``{verdict, alternative_label, discriminating_markers, reason,
    _raw, _prompt}`` (verdict defaults to ``keep`` when the reply is unparseable).
    """
    block = _context_block(cfg, kind)
    label = str((proposed or {}).get("label", "") or "")
    prompt = _refuter_prompt(code, ev, label, block, kind=kind)
    raw = call_llm(prompt, provider=cfg.provider, model=cfg.model, api_key=cfg.api_key, system=_REFUTER_SYSTEM)
    j = parse_json(raw)
    return {
        "verdict": str(j.get("verdict", "keep") or "keep").strip().lower(),
        "alternative_label": str(j.get("alternative_label", "") or ""),
        "discriminating_markers": list(j.get("discriminating_markers", []) or []),
        "reason": str(j.get("reason", "") or ""),
        "_raw": raw,
        "_prompt": prompt,
    }


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
