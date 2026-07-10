"""Deterministic verification gates for LLM codebook annotation (anti-hallucination).

The LLM proposes a label; this module's measured evidence disposes. Each function is
pure, deterministic, and offline: a proposed-label dict (as produced by an LLM) and a
per-code evidence dict (as produced by :func:`nicheverse.annotate.artifacts.code_evidence`
or :func:`niche_evidence`) are checked against a :class:`ProjectContext`, and the label is
accepted, penalized, or rejected on the basis of what the evidence actually supports.

The checks:

* :func:`marker_presence` - are the cited markers actually enriched in this code, or are
  they hallucinated / segmentation leakage?
* :func:`check_citations` - do the citation ids parse, and (through an INJECTABLE resolver
  so tests stay offline) do they resolve to a real record that mentions the markers, or is
  the identifier fabricated?
* :func:`apply_lab_rules` - encode the lab's codebook conventions (site-aware reassignment,
  composite-label gating) as code and record violations + an adjusted label.
* :func:`validate_vocabulary` - is the label in the project's expected vocabulary (with a
  novel/uncertain escape hatch)?
* :func:`gate` - run all of the above and synthesize an accept/penalize/reject decision.

Scope: imaging-based / in-situ spatial transcriptomics (IMST) only.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

__all__ = [
    "marker_presence",
    "check_citations",
    "apply_lab_rules",
    "validate_vocabulary",
    "gate",
    "SITE_RESTRICTED",
]

# Site-restricted cell types and the sites that permit them. Keys are lowercase
# substrings tested against a proposed label; values are lowercase substrings that a
# code's dominant-site distribution must contain for the call to be permissible.
# Brain / CNS-restricted lineages are the canonical case in the RCC brain-metastasis
# cohort (glial transcript leakage into kidney-primary codes is the failure this guards).
SITE_RESTRICTED: dict[str, dict] = {
    "microglia": {"permissive": ("brain", "brm", "cns", "metasta"), "general": "Macrophage/Myeloid"},
    "oligodendrocyte": {"permissive": ("brain", "brm", "cns", "metasta"), "general": "Glial-marker-low cell"},
    "astrocyte": {"permissive": ("brain", "brm", "cns", "metasta"), "general": "Stromal/Fibroblast"},
    "neuron": {"permissive": ("brain", "brm", "cns", "metasta"), "general": "Non-neural cell"},
    "neuronal": {"permissive": ("brain", "brm", "cns", "metasta"), "general": "Non-neural cell"},
    "opc": {"permissive": ("brain", "brm", "cns", "metasta"), "general": "Glial-marker-low cell"},
    "ependymal": {"permissive": ("brain", "brm", "cns", "metasta"), "general": "Epithelial cell"},
}


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _evidence_marker_map(code_evidence: dict) -> dict[str, float]:
    """Map lowercase gene -> best z / positive-DEG log2fc available in the evidence.

    A gene present in ``top_markers`` contributes its z-score; a gene present in
    ``top_degs`` with a positive log2 fold change contributes that log2fc. When a gene
    appears in both, the larger value wins, so a strongly enriched gene is never masked
    by a weaker source.
    """
    best: dict[str, float] = {}
    for g, z in code_evidence.get("top_markers", []) or []:
        try:
            v = float(z)
        except (TypeError, ValueError):
            continue
        k = _norm(str(g))
        if k and (k not in best or v > best[k]):
            best[k] = v
    for entry in code_evidence.get("top_degs", []) or []:
        # (gene, log2fc, padj)
        if not entry:
            continue
        g = entry[0]
        lfc = entry[1] if len(entry) > 1 else None
        try:
            v = float(lfc)
        except (TypeError, ValueError):
            continue
        if v <= 0:  # only a POSITIVE DEG counts as support
            continue
        k = _norm(str(g))
        if k and (k not in best or v > best[k]):
            best[k] = v
    return best


def marker_presence(cited_markers, code_evidence: dict, *, z_thresh: float = 1.0) -> dict:
    """Check which cited markers are actually enriched in the code's evidence.

    A cited marker is SUPPORTED if it appears in ``code_evidence['top_markers']`` with
    ``z >= z_thresh`` OR as a positive DEG in ``code_evidence['top_degs']``. Markers that
    clear neither bar are ``absent`` (hallucinated or segmentation leakage). Gene matching
    is case-insensitive.

    Returns ``{"present": [...], "absent": [...], "precision": float, "n_cited": int}``,
    where ``precision = len(present) / n_cited`` (and is ``1.0`` when nothing was cited, so
    an empty citation list is not itself penalized here).
    """
    cited = [str(m).strip() for m in (cited_markers or []) if str(m).strip()]
    ev_map = _evidence_marker_map(code_evidence)
    present, absent = [], []
    seen: set[str] = set()
    for m in cited:
        k = _norm(m)
        if k in seen:  # de-duplicate while preserving first-seen order
            continue
        seen.add(k)
        if ev_map.get(k, float("-inf")) >= z_thresh:
            present.append(m)
        else:
            absent.append(m)
    n_cited = len(seen)
    precision = 1.0 if n_cited == 0 else len(present) / n_cited
    return {"present": present, "absent": absent, "precision": precision, "n_cited": n_cited}


_PMID_RE = re.compile(r"\b(?:pmid[:\s]*)?(\d{5,9})\b", re.IGNORECASE)
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>]+)", re.IGNORECASE)


def _parse_citation(raw: str):
    """Return ``(id, kind)`` for a citation string: a DOI (10.x), then a PMID (digits), else unparsed."""
    s = str(raw or "")
    md = _DOI_RE.search(s)
    if md:
        return md.group(1).rstrip(".,;)"), "doi"
    mp = _PMID_RE.search(s)
    if mp:
        return mp.group(1), "pmid"
    return None, "unparsed"


def _default_resolver(identifier: str):
    """Lazy PubMed resolver used only when no resolver is injected and a network path exists.

    Wrapped so that the absence of ``requests`` / network access yields ``None`` (treated
    upstream as "unresolved") rather than raising. Tests always inject a resolver and never
    reach this code path.
    """
    try:
        from .literature import pubmed_search
    except Exception:
        return None
    try:
        ident = str(identifier)
        # A bare PMID: look it up as an id-scoped query; otherwise pass the string through.
        query = f"{ident}[uid]" if ident.isdigit() else ident
        hits = pubmed_search(query, max_results=1)
        return hits[0] if hits else None
    except Exception:
        return None


def _record_mentions(record: dict, markers) -> bool:
    """True if any marker (case-insensitive) appears in the record's title/abstract text."""
    if not record or not markers:
        return False
    text = " ".join(
        str(record.get(k, "") or "") for k in ("title", "abstract", "journal", "first_author")
    ).lower()
    return any(_norm(m) and _norm(m) in text for m in markers)


def check_citations(citations, *, resolver=None, markers=None) -> list[dict]:
    """Parse and (optionally) resolve each citation, flagging fabricated identifiers.

    Each citation is parsed for a DOI (``10.x``) or PMID (a run of digits). ``resolver`` is
    a callable ``(identifier_or_query) -> record | None``; when ``None`` it lazily falls back
    to PubMed via :mod:`nicheverse.annotate.literature`, but only if a network path is
    available (any failure yields ``resolved=None`` rather than raising, so this stays safe
    offline). Tests inject a dict-backed fake resolver and never touch the network.

    Returns one dict per citation::

        {"citation": raw, "id": parsed_or_None, "kind": "pmid"|"doi"|"unparsed",
         "resolved": bool|None, "supports": bool|None, "record": {...}|None}

    ``resolved`` is ``None`` when no resolver ran (e.g. offline default, or an unparsed id),
    ``True`` when the resolver returned a record, and ``False`` when a parseable id resolved
    to nothing (a likely fabricated identifier). ``supports`` is ``True`` iff a retrieved
    record's text mentions one of ``markers`` (``None`` when no record was retrieved).
    """
    use_default = resolver is None
    resolve_fn = _default_resolver if use_default else resolver
    out: list[dict] = []
    for raw in citations or []:
        cid, kind = _parse_citation(raw)
        row = {
            "citation": raw,
            "id": cid,
            "kind": kind,
            "resolved": None,
            "supports": None,
            "record": None,
        }
        if cid is not None:
            record = None
            try:
                record = resolve_fn(cid)
            except Exception:
                record = None
            if record is not None:
                row["record"] = record
                row["resolved"] = True
                if markers:
                    row["supports"] = _record_mentions(record, markers)
            else:
                # An injected resolver returning nothing means the id is fabricated -> False.
                # The offline default resolver cannot distinguish "no network" from "not found",
                # so it stays None (unknown) rather than falsely accusing a real citation.
                row["resolved"] = None if use_default else False
        out.append(row)
    return out


def _closest(name: str, choices) -> tuple[str | None, float]:
    """Closest allowed name by substring containment first, then difflib ratio."""
    n = _norm(name)
    if not n or not choices:
        return None, 0.0
    best, best_score = None, 0.0
    for c in choices:
        cn = _norm(c)
        if not cn:
            continue
        if n == cn:
            return c, 1.0
        if n in cn or cn in n:
            score = 0.9  # substring containment is a strong signal
        else:
            score = SequenceMatcher(None, n, cn).ratio()
        if score > best_score:
            best, best_score = c, score
    return best, best_score


def validate_vocabulary(label: str, project_context, *, kind: str = "cell", allow_novel: bool = True) -> dict:
    """Check a label against the project's expected vocabulary with a novelty escape hatch.

    For ``kind='cell'`` the allowed set is ``project_context.expected_cell_types`` names; for
    ``kind='niche'`` it is ``project_context.expected_niches`` names. Matching is exact first,
    then substring containment, then fuzzy (difflib) to surface the ``closest`` allowed label.
    When ``allow_novel`` and the label is not in vocab, ``in_vocab`` stays ``False`` but this is
    not treated as an error (novel / uncertain codes are allowed).

    Returns ``{"in_vocab": bool, "closest": str|None, "kind": kind}``.
    """
    if kind == "niche":
        allowed = [getattr(n, "name", "") for n in getattr(project_context, "expected_niches", []) or []]
    else:
        allowed = [getattr(c, "name", "") for c in getattr(project_context, "expected_cell_types", []) or []]
    allowed = [a for a in allowed if a]
    lab = str(label or "")
    n = _norm(lab)
    in_vocab = any(n == _norm(a) for a in allowed)
    closest, score = _closest(lab, allowed)
    # only surface a "closest" suggestion when it is a plausible near-match
    if score < 0.6:
        closest = None
    return {"in_vocab": bool(in_vocab), "closest": closest, "kind": kind}


def _dominant_sites(label_dict, code_evidence, project_context, adata, code):
    """Return a list of lowercase dominant-site strings for the code, or [] if unknown.

    Prefers the evidence dict's ``dist_<site_col>`` (and any other ``dist_*`` key that looks
    like a site distribution); falls back to computing the distribution from ``adata`` when
    ``adata`` + ``code`` + ``project_context.site_col`` are supplied.
    """
    site_col = getattr(project_context, "site_col", "") or ""
    keys = []
    if site_col and f"dist_{site_col}" in code_evidence:
        keys = [f"dist_{site_col}"]
    else:
        keys = [k for k in code_evidence if k.startswith("dist_")]
    for k in keys:
        dist = code_evidence.get(k) or {}
        if isinstance(dist, dict) and dist:
            return [_norm(s) for s in dist.keys()]
    # fall back to computing from adata
    if adata is not None and code is not None and site_col:
        try:
            obs = adata.obs
            # find the code column heuristically (whatever column holds this code value)
            for col in ("cell_codebook_idx", "neighborhood_codebook_idx", "code", "_code"):
                if col in obs.columns and site_col in obs.columns:
                    m = obs[col].astype(str) == str(code)
                    if m.any():
                        vc = obs.loc[m, site_col].astype(str).value_counts(normalize=True)
                        return [_norm(s) for s in vc.head(6).index]
        except Exception:
            return []
    return []


def _canonical_markers(name: str, project_context):
    """Canonical markers for a cell-type name from the project context (case-insensitive)."""
    n = _norm(name)
    for c in getattr(project_context, "expected_cell_types", []) or []:
        if _norm(getattr(c, "name", "")) == n:
            return list(getattr(c, "markers", []) or [])
    # loose containment fallback (e.g. "T cell" component of a composite matches "CD8 T cell")
    for c in getattr(project_context, "expected_cell_types", []) or []:
        cn = _norm(getattr(c, "name", ""))
        if cn and (cn in n or n in cn):
            return list(getattr(c, "markers", []) or [])
    return []


def apply_lab_rules(label_dict: dict, code_evidence: dict, project_context, *, adata=None, code=None) -> dict:
    """Encode the lab's codebook conventions (``context.ANNOTATION_RULES``) as executable checks.

    (a) Site-aware reassignment (rule 4): if the proposed label names a site-restricted
    (brain / CNS-only) type but the code's dominant-site distribution includes a
    non-permissive site (e.g. a kidney primary), record a violation and propose the closest
    general label.

    (b) Composite label ``X/Y`` (rule 5): allowed only if BOTH components have marker support
    clearing a relative bar (``rel >= 1.3x`` cohort mean) and an absolute bar (``abs >= 0.05``)
    in the evidence. When absolute expression cannot be recovered from the evidence dict (the
    common case: only z-scores / DEGs are available), the check APPROXIMATES support by
    requiring each component's canonical markers to be present in the code's markers/DEGs
    (see :func:`marker_presence`); this approximation is recorded in ``notes``. An unsupported
    composite collapses to its dominant (better-supported) component.

    Returns ``{"violations": [...], "adjusted_label": str|None, "notes": [...]}``. ``adjusted_label``
    is ``None`` when no rule fired (i.e. the label is left as-is).
    """
    label = str((label_dict or {}).get("label", "") or "")
    nlab = _norm(label)
    violations: list[str] = []
    notes: list[str] = []
    adjusted_label = None

    # (a) site-aware reassignment
    for key, spec in SITE_RESTRICTED.items():
        if key in nlab:
            sites = _dominant_sites(label_dict, code_evidence, project_context, adata, code)
            if sites:
                permissive = spec["permissive"]
                offending = [s for s in sites if not any(p in s for p in permissive)]
                if offending:
                    violations.append(
                        f"site-restricted label '{label}' but dominant site(s) include "
                        f"non-permissive {sorted(set(offending))}"
                    )
                    adjusted_label = spec["general"]
                    notes.append(
                        f"reassigned '{label}' -> '{spec['general']}' (rule 4: closest general label)"
                    )
            else:
                notes.append(f"site distribution unavailable; could not check site restriction for '{label}'")
            break

    # (b) composite 'X/Y' label gating
    if "/" in label:
        components = [p.strip() for p in label.split("/") if p.strip()]
        if len(components) >= 2:
            comp_support: dict[str, dict] = {}
            approximated = False
            for comp in components:
                mk = _canonical_markers(comp, project_context)
                if not mk:
                    # no priors -> gate on the component name itself as a pseudo-marker
                    mk = [comp]
                approximated = True  # absolute expression not recoverable from the evidence dict
                mp = marker_presence(mk, code_evidence, z_thresh=1.0)
                comp_support[comp] = mp
            if approximated:
                notes.append(
                    "composite gating approximated on marker-presence (rel>=1.3x/abs>=0.05 not "
                    "computable from evidence dict); each component checked via its canonical markers"
                )
            supported = {c: mp for c, mp in comp_support.items() if mp["precision"] > 0 and mp["present"]}
            if len(supported) < 2:
                # collapse to the dominant component: most present markers, then best precision
                dominant = max(
                    components,
                    key=lambda c: (len(comp_support[c]["present"]), comp_support[c]["precision"]),
                )
                violations.append(
                    f"composite label '{label}' not supported for both components "
                    f"(supported: {sorted(supported)}); collapsing to dominant '{dominant}'"
                )
                # a prior site reassignment takes precedence; otherwise use the collapsed label
                if adjusted_label is None:
                    adjusted_label = dominant
                notes.append(f"collapsed composite '{label}' -> '{dominant}' (rule 5)")

    return {"violations": violations, "adjusted_label": adjusted_label, "notes": notes}


def gate(
    label_dict: dict,
    code_evidence: dict,
    project_context,
    *,
    kind: str = "cell",
    resolver=None,
    adata=None,
    code=None,
    z_thresh: float = 1.0,
    min_marker_precision: float = 0.5,
) -> dict:
    """Run every verification gate and synthesize an accept / penalize / reject decision.

    POLICY. ``passed`` is ``False`` (and ``confidence_penalty > 0``) when ANY of:

    * marker precision ``< min_marker_precision`` (the LLM cited markers that are not
      enriched in this code: hallucination or segmentation leakage);
    * any citation with a parseable id has ``resolved is False`` (a fabricated identifier
      the resolver could not retrieve; an offline / unknown ``resolved is None`` does NOT
      fail the gate);
    * ``apply_lab_rules`` recorded any violation (site-restricted mislabel, or an
      unsupported composite).

    Otherwise ``passed`` is ``True``. ``confidence_penalty`` accumulates 0.34 per failing
    category (marker / citation / rule), capped at 1.0, so a caller can downweight rather
    than hard-drop borderline calls. ``adjusted_label`` is the lab-rule-adjusted label when a
    rule fired, else the label's vocabulary-``closest`` when it is out of vocab, else ``None``.

    A marker-only complaint about a NOVEL (out-of-vocab) label still fails the gate: the
    evidence must back whatever the LLM cited regardless of whether the label is expected.

    Returns::

        {"passed": bool, "flags": [str,...], "marker": <marker_presence>,
         "citations": <check_citations>, "rules": <apply_lab_rules>,
         "vocab": <validate_vocabulary>, "adjusted_label": str|None,
         "confidence_penalty": float}
    """
    ld = label_dict or {}
    cited_markers = ld.get("key_markers", []) or []
    citations = ld.get("citations", []) or []
    label = str(ld.get("label", "") or "")

    marker = marker_presence(cited_markers, code_evidence, z_thresh=z_thresh)
    cites = check_citations(citations, resolver=resolver, markers=cited_markers)
    rules = apply_lab_rules(ld, code_evidence, project_context, adata=adata, code=code)
    vocab = validate_vocabulary(label, project_context, kind=kind)

    flags: list[str] = []
    penalty = 0.0

    # marker precision: only penalize when markers were actually cited
    if marker["n_cited"] > 0 and marker["precision"] < min_marker_precision:
        flags.append(
            f"low marker precision {marker['precision']:.2f} < {min_marker_precision} "
            f"(absent: {marker['absent']})"
        )
        penalty += 0.34

    # fabricated citation: a parseable id the resolver returned nothing for
    fabricated = [c for c in cites if c["resolved"] is False]
    if fabricated:
        flags.append(f"unresolved/fabricated citation id(s): {[c['id'] for c in fabricated]}")
        penalty += 0.34

    # lab-rule violations
    if rules["violations"]:
        flags.extend(rules["violations"])
        penalty += 0.34

    penalty = min(1.0, round(penalty, 4))
    passed = penalty == 0.0

    adjusted_label = rules["adjusted_label"]
    if adjusted_label is None and not vocab["in_vocab"] and vocab["closest"]:
        adjusted_label = vocab["closest"]

    return {
        "passed": passed,
        "flags": flags,
        "marker": marker,
        "citations": cites,
        "rules": rules,
        "vocab": vocab,
        "adjusted_label": adjusted_label,
        "confidence_penalty": penalty,
    }
