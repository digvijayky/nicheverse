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

# Site-restricted cell types and the sites that permit them. Keys are lowercase type names
# tested as WHOLE WORDS against a proposed label (so "neuron" does not trip on
# "neuroendocrine"); ``permissive`` values are lowercase site prefixes that a code's site
# distribution must be dominated by for the call to be permissible. Brain / CNS-restricted
# lineages are the canonical case in the RCC brain-metastasis cohort (glial transcript
# leakage into kidney-primary codes is the failure this guards).
SITE_RESTRICTED: dict[str, dict] = {
    "microglia": {"permissive": ("brain", "brm", "cns", "metasta"), "general": "Macrophage/Myeloid"},
    "oligodendrocyte": {"permissive": ("brain", "brm", "cns", "metasta"), "general": "Glial-marker-low cell"},
    "astrocyte": {"permissive": ("brain", "brm", "cns", "metasta"), "general": "Stromal/Fibroblast"},
    "neuron": {"permissive": ("brain", "brm", "cns", "metasta"), "general": "Non-neural cell"},
    "neuronal": {"permissive": ("brain", "brm", "cns", "metasta"), "general": "Non-neural cell"},
    "opc": {"permissive": ("brain", "brm", "cns", "metasta"), "general": "Glial-marker-low cell"},
    "ependymal": {"permissive": ("brain", "brm", "cns", "metasta"), "general": "Epithelial cell"},
}

# A non-permissive site holding at least this fraction of a code's cells is treated as a
# material out-of-site population (not plausibly pure segmentation leakage), and trips the
# site-restriction rule even when it is not the single dominant site.
_SITE_MATERIAL_FRAC = 0.25

# Confidence penalty added per failing gate category (marker / citation / rule).
# Three categories at 0.34 exceed 1.0, so the total is capped at 1.0 in :func:`gate`.
_GATE_CATEGORY_PENALTY = 0.34


def _norm(s: str) -> str:
    return " ".join(str(s or "").strip().lower().split())


# Non-word characters used to test whole-word (token) membership of a site-restricted
# key inside a label, so e.g. "neuron" does not match "neuroendocrine".
_WORD_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _word_tokens(s: str) -> set[str]:
    """Lowercase alphanumeric word tokens of a string (for whole-word matching)."""
    return {t for t in _WORD_SPLIT_RE.split(_norm(s)) if t}


def _phrase_in_words(key: str, text: str) -> bool:
    """True if ``key`` occurs in ``text`` as a whole word / whole contiguous phrase.

    Single-token keys must match a full token (``"neuron"`` matches ``"neuron"`` and
    ``"neuron cell"`` but not ``"neuroendocrine"``). Multi-token keys must match a
    contiguous run of whole tokens.
    """
    ktoks = [t for t in _WORD_SPLIT_RE.split(_norm(key)) if t]
    if not ktoks:
        return False
    ttoks = [t for t in _WORD_SPLIT_RE.split(_norm(text)) if t]
    n = len(ktoks)
    for i in range(len(ttoks) - n + 1):
        if ttoks[i : i + n] == ktoks:
            return True
    return False


def _consider(best: dict[str, float], gene, value) -> None:
    """Record ``gene -> value`` keeping the larger value, skipping unparseable pairs."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return
    if v != v:  # NaN never counts as support
        return
    k = _norm(str(gene))
    if k and (k not in best or v > best[k]):
        best[k] = v


def _evidence_marker_map(code_evidence) -> dict[str, float]:
    """Map lowercase gene -> best z / positive-DEG log2fc available in the evidence.

    A gene present in ``top_markers`` contributes its z-score; a gene present in
    ``top_degs`` with a positive log2 fold change contributes that log2fc. When a gene
    appears in both, the larger value wins, so a strongly enriched gene is never masked by
    a weaker source. Robust to a missing/non-dict evidence object (``top_markers`` /
    ``top_degs`` absent) and to individually malformed entries (wrong arity, non-numeric,
    NaN), which are skipped rather than raising.
    """
    best: dict[str, float] = {}
    if not isinstance(code_evidence, dict):
        return best
    for entry in code_evidence.get("top_markers", []) or []:
        try:
            g, z = entry[0], entry[1]
        except (TypeError, IndexError, KeyError):
            continue
        _consider(best, g, z)
    for entry in code_evidence.get("top_degs", []) or []:
        # (gene, log2fc, padj); only a POSITIVE log2fc counts as support
        try:
            g, lfc = entry[0], entry[1]
        except (TypeError, IndexError, KeyError):
            continue
        try:
            if float(lfc) <= 0:
                continue
        except (TypeError, ValueError):
            continue
        _consider(best, g, lfc)
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


# A DOI is matched first (it embeds digits that must not be mistaken for a PMID).
_DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s\"'<>]+)", re.IGNORECASE)
# An EXPLICITLY prefixed PMID ("PMID:123", "pmid 123", "PubMed ID 123") accepts 1-9
# digits, so short historical PMIDs parse. A BARE run of digits is only accepted when
# it is 5-9 digits (real PMIDs are 1-8 digits, up to ~40M), which excludes 4-digit
# years and other small integers that would otherwise be misread as identifiers.
_PMID_LABELLED_RE = re.compile(r"\bpm(?:id|c)?\b[:#\s]*?(\d{1,9})\b", re.IGNORECASE)
_PMID_LABELLED_RE2 = re.compile(r"\bpubmed(?:\s*id)?\b[:#\s]*?(\d{1,9})\b", re.IGNORECASE)
_PMID_BARE_RE = re.compile(r"(?<!\d)(\d{5,9})(?!\d)")


def _clean_doi(doi: str) -> str:
    """Strip trailing punctuation a DOI regex may have swallowed from surrounding text."""
    return doi.rstrip(".,;:)]}>\"'")


def _parse_citation(raw: str):
    """Return ``(id, kind)`` for one citation string.

    A DOI (``10.xxxx/...``, possibly inside a ``doi:`` or ``https://doi.org/`` URL) is
    matched first, then a PubMed identifier: an explicitly labelled ``PMID``/``PMCID``/
    ``PubMed ID`` accepts 1-9 digits, and an UNlabelled bare run is taken as a PMID only
    when it is 5-9 digits (so a 4-digit year is never mistaken for one). Anything else is
    ``(None, "unparsed")``. Matching is case-insensitive and never raises.
    """
    s = str(raw or "")
    md = _DOI_RE.search(s)
    if md:
        return _clean_doi(md.group(1)), "doi"
    for pat in (_PMID_LABELLED_RE, _PMID_LABELLED_RE2):
        mp = pat.search(s)
        if mp:
            return mp.group(1), "pmid"
    mb = _PMID_BARE_RE.search(s)
    if mb:
        return mb.group(1), "pmid"
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
    """True if any marker appears as a whole word in the record's title/abstract text.

    Case-insensitive and whole-word (so a short gene symbol is not matched as a
    substring of an unrelated word). ``supports`` is advisory only and never gates
    pass/fail, so a missing abstract simply yields ``False``.
    """
    if not isinstance(record, dict) or not markers:
        return False
    text_tokens = _word_tokens(
        " ".join(str(record.get(k, "") or "") for k in ("title", "abstract", "journal", "first_author"))
    )
    return any(_norm(m) in text_tokens for m in markers if _norm(m))


def check_citations(citations, *, resolver=None, markers=None) -> list[dict]:
    """Parse and (optionally) resolve each citation, flagging fabricated identifiers.

    Each citation is parsed (see :func:`_parse_citation`) for a DOI or a PMID. ``resolver`` is
    a callable ``(identifier_or_query) -> record | None``; when ``None`` it lazily falls back
    to PubMed via :mod:`nicheverse.annotate.literature`, but only if a network path is
    available. The resolver call is wrapped, so a resolver that raises (or an offline default
    with no network) yields a ``None`` record rather than propagating -- this function never
    raises. Tests inject a dict-backed fake resolver and never touch the network.

    Returns one dict per citation::

        {"citation": raw, "id": parsed_or_None, "kind": "pmid"|"doi"|"unparsed",
         "resolved": bool|None, "supports": bool|None, "record": {...}|None}

    ``resolved`` semantics (the key anti-hallucination distinction):

    * ``None``  -- no verdict: the id was unparsed, OR the DEFAULT (offline) resolver ran and
      could not reach the network. An unknown id is NEVER accused of being fabricated.
    * ``True``  -- an INJECTED or online resolver returned a record for the id.
    * ``False`` -- an INJECTED resolver ran and returned nothing for a parseable id: the id
      is a likely fabrication. Only the injected path can yield ``False``.

    ``supports`` is ``True`` iff a retrieved record's text mentions one of ``markers`` as a
    whole word, and ``None`` whenever no record was retrieved or ``markers`` is empty.
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
    """Return the closest allowed name and a [0,1] similarity score.

    Scoring, in order: an exact (normalized) match scores 1.0; a whole-word phrase
    containment (one label's tokens are a contiguous run inside the other's, e.g.
    ``"T cell"`` inside ``"CD8 T cell"``) scores 0.9, but only when the shorter side is a
    substantial fraction (>= 0.5) of the longer so a 1-2 character token cannot spuriously
    anchor to a long label; otherwise a difflib ``SequenceMatcher`` ratio. Ties keep the
    first (input-order) choice. The caller applies a cutoff before surfacing the result.
    """
    n = _norm(name)
    if not n or not choices:
        return None, 0.0
    ntoks = _word_tokens(n)
    best, best_score = None, 0.0
    for c in choices:
        cn = _norm(c)
        if not cn:
            continue
        if n == cn:
            return c, 1.0
        ctoks = _word_tokens(cn)
        # whole-word phrase containment, guarded so a tiny fragment cannot anchor
        contained = _phrase_in_words(n, cn) or _phrase_in_words(cn, n)
        substantial = min(len(n), len(cn)) >= 0.5 * max(len(n), len(cn))
        token_overlap = bool(ntoks & ctoks)
        if contained and substantial and token_overlap:
            score = 0.9
        else:
            score = SequenceMatcher(None, n, cn).ratio()
        if score > best_score:
            best, best_score = c, score
    return best, best_score


_VOCAB_FUZZY_CUTOFF = 0.6


def validate_vocabulary(label: str, project_context, *, kind: str = "cell", allow_novel: bool = True) -> dict:
    """Check a label against the project's expected vocabulary with a novelty escape hatch.

    For ``kind='cell'`` the allowed set is ``project_context.expected_cell_types`` names; for
    ``kind='niche'`` it is ``project_context.expected_niches`` names (any other ``kind`` is
    treated as ``'cell'``). ``in_vocab`` is set by an EXACT (case- and whitespace-insensitive)
    name match ONLY, so fuzzy matching can never manufacture a false in-vocab hit.

    ``closest`` surfaces the nearest allowed label (see :func:`_closest`), gated by
    ``allow_novel``:

    * ``allow_novel=True`` (default): only surface ``closest`` when it is a plausible
      near-match (similarity ``>= 0.6``); an out-of-vocab label with no near neighbor
      returns ``closest=None`` and is left untouched by the caller (novel / uncertain codes
      are allowed, not errors).
    * ``allow_novel=False``: novel labels are not tolerated, so ``closest`` is ALWAYS the
      nearest allowed label (when any vocabulary exists), letting the caller coerce an
      out-of-vocab label onto the closest expected type.

    An exact in-vocab match always returns itself as ``closest``. Returns
    ``{"in_vocab": bool, "closest": str|None, "kind": kind}``.
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
    if in_vocab:
        pass  # exact match: keep closest == the matched label
    elif allow_novel and score < _VOCAB_FUZZY_CUTOFF:
        # novel label with no plausible near-match: do not surface a misleading suggestion
        closest = None
    # allow_novel=False keeps the nearest allowed label (if any) so the caller can coerce.
    return {"in_vocab": bool(in_vocab), "closest": closest, "kind": kind}


def _site_permits(site: str, permissive) -> bool:
    """True if ``site`` is compatible with a permissive prefix/token.

    The permissive entries are curated site prefixes (``brain``, ``brm``, ``cns``,
    ``metasta``); a site permits the call if any entry is a whole word of the site name
    OR a leading substring (``"metasta"`` matches ``"metastasis"``). These prefixes are
    specific enough that substring matching does not admit non-permissive sites such as
    ``"primary"`` or ``"kidney"``.
    """
    s = _norm(site)
    return any(_phrase_in_words(p, s) or p in s for p in (permissive or ()))


def _site_distribution(code_evidence, project_context, adata, code) -> list[tuple[str, float]]:
    """Return ``[(lowercase_site, fraction), ...]`` sorted by descending fraction, or ``[]``.

    Prefers the evidence dict's ``dist_<site_col>`` (a normalized value_counts, as written
    by :func:`nicheverse.annotate.artifacts.code_evidence`); if that key is absent, uses any
    single ``dist_*`` key present. Falls back to computing the distribution from ``adata``
    when ``adata`` + ``code`` + ``project_context.site_col`` are supplied. Fractions from the
    evidence dict are used as-is; when a distribution carries no numeric weights (e.g. a bare
    set of site names) every site is given equal weight so the caller can still reason about
    membership. Never raises.
    """
    site_col = getattr(project_context, "site_col", "") or ""
    dist = None
    if isinstance(code_evidence, dict):
        if site_col and f"dist_{site_col}" in code_evidence:
            dist = code_evidence.get(f"dist_{site_col}")
        else:
            for k in code_evidence:
                if isinstance(k, str) and k.startswith("dist_"):
                    cand = code_evidence.get(k)
                    if isinstance(cand, dict) and cand:
                        dist = cand
                        break
    if not (isinstance(dist, dict) and dist):
        # fall back to computing from adata
        dist = None
        if adata is not None and code is not None and site_col:
            try:
                obs = adata.obs
                for col in ("cell_codebook_idx", "neighborhood_codebook_idx", "code", "_code"):
                    if col in obs.columns and site_col in obs.columns:
                        m = (obs[col].astype(str) == str(code)).to_numpy()
                        if m.any():
                            vc = obs.loc[m, site_col].astype(str).value_counts(normalize=True)
                            dist = {str(k): float(v) for k, v in vc.head(10).items()}
                            break
            except Exception:
                dist = None
    if not (isinstance(dist, dict) and dist):
        return []
    pairs: list[tuple[str, float]] = []
    for s, v in dist.items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = float("nan")
        pairs.append((_norm(s), f))
    if all(f != f for _, f in pairs):  # all NaN: no usable weights -> equal weight
        w = 1.0 / len(pairs)
        pairs = [(s, w) for s, _ in pairs]
    else:
        pairs = [(s, (0.0 if f != f else f)) for s, f in pairs]
    pairs.sort(key=lambda t: t[1], reverse=True)
    return [(s, f) for s, f in pairs if s]


def _canonical_markers(name: str, project_context):
    """Canonical markers for a cell-type name from the project context.

    Exact (normalized) match first; then a whole-word phrase-containment fallback so a
    composite component like ``"T cell"`` still recovers the markers of ``"CD8 T cell"``
    without a bare fragment matching an unrelated prior. Case-insensitive.
    """
    n = _norm(name)
    if not n:
        return []
    priors = getattr(project_context, "expected_cell_types", []) or []
    for c in priors:
        if _norm(getattr(c, "name", "")) == n:
            return list(getattr(c, "markers", []) or [])
    for c in priors:
        cn = _norm(getattr(c, "name", ""))
        if cn and (_phrase_in_words(n, cn) or _phrase_in_words(cn, n)):
            return list(getattr(c, "markers", []) or [])
    return []


def apply_lab_rules(label_dict: dict, code_evidence: dict, project_context, *, adata=None, code=None) -> dict:
    """Encode the lab's codebook conventions (``context.ANNOTATION_RULES``) as executable checks.

    (a) Site-aware reassignment (rule 4): if the proposed label names a site-restricted
    (brain / CNS-only) type (matched as a whole word) but the code's site distribution is not
    dominated by a permissive site -- specifically the top-fraction site is non-permissive, OR
    a non-permissive site holds a material fraction (``>= _SITE_MATERIAL_FRAC = 0.25``) of the
    code's cells -- record a violation and propose the closest general label. A permissive site
    holding a small non-permissive tail (plausible segmentation leakage) is NOT flagged. When
    the site distribution is unavailable the check is skipped and a note is recorded.

    (b) Composite label ``X/Y`` (rule 5): allowed only if BOTH components have marker support
    clearing a relative bar (``rel >= 1.3x`` cohort mean) and an absolute bar (``abs >= 0.05``)
    in the evidence. Absolute / relative-to-cohort expression is NOT recoverable from the
    per-code evidence dict (which carries only across-code z-scores and one-vs-rest DEGs, not
    the cohort-mean expression the exact bars need), so the check APPROXIMATES that support by
    requiring each component's canonical markers (or, absent priors, the component name itself)
    to be present via :func:`marker_presence`; this approximation is recorded in ``notes``. A
    composite with fewer than two supported components collapses to its dominant component
    (most present markers, then higher precision). A site reassignment from (a) takes
    precedence over the collapsed label.

    Returns ``{"violations": [...], "adjusted_label": str|None, "notes": [...]}``. ``adjusted_label``
    is ``None`` when no rule fired (i.e. the label is left as-is).
    """
    ld = label_dict if isinstance(label_dict, dict) else {}
    label = str(ld.get("label", "") or "")
    nlab = _norm(label)
    violations: list[str] = []
    notes: list[str] = []
    adjusted_label = None

    # (a) site-aware reassignment. A site-restricted (brain/CNS-only) label is a violation
    #     when the code's DOMINANT (top-fraction) site is non-permissive, or when any
    #     non-permissive site holds a material fraction of the code's cells (>= 0.25) -- a
    #     substantial out-of-site population is not plausibly pure segmentation leakage.
    #     Key matching is whole-word so "neuron" does not trip on "neuroendocrine".
    for key, spec in SITE_RESTRICTED.items():
        if _phrase_in_words(key, nlab):
            sites = _site_distribution(code_evidence, project_context, adata, code)
            if sites:
                permissive = spec["permissive"]
                dominant_site, dominant_frac = sites[0]
                nonpermissive = [(s, f) for s, f in sites if not _site_permits(s, permissive)]
                material = [(s, f) for s, f in nonpermissive if f >= _SITE_MATERIAL_FRAC]
                if (not _site_permits(dominant_site, permissive)) or material:
                    offending = sorted({s for s, _ in nonpermissive})
                    violations.append(
                        f"site-restricted label '{label}' but the code's site distribution "
                        f"includes non-permissive site(s) {offending} "
                        f"(dominant site '{dominant_site}' at {dominant_frac:.2f})"
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

    POLICY. Exactly three failing CATEGORIES are evaluated, each contributing one flag and a
    fixed 0.34 to ``confidence_penalty``:

    1. MARKER. Markers were cited (``n_cited > 0``) AND ``marker['precision'] <
       min_marker_precision`` -- the LLM cited genes not enriched in this code (hallucination
       or segmentation leakage). Citing nothing is not itself a marker failure.
    2. CITATION. At least one citation has ``resolved is False`` -- a parseable id the
       INJECTED resolver could not retrieve, i.e. a likely fabricated identifier. An
       unresolved offline / unknown id (``resolved is None``) does NOT fail the gate, so the
       default (no resolver, offline) path never fails a citation merely for being
       unresolved. An ``unparsed`` citation also does not fail here.
    3. RULE. ``apply_lab_rules`` recorded any violation (a site-restricted mislabel or an
       unsupported composite).

    INVARIANTS. ``confidence_penalty = min(1.0, 0.34 * n_failing_categories)`` and
    ``passed is (confidence_penalty == 0.0)`` exactly; every failing category contributes
    exactly one entry to ``flags`` (rule violations are appended verbatim), so an empty
    ``flags`` list is equivalent to ``passed is True``. The penalty caps at 1.0 so a caller
    can downweight rather than hard-drop borderline calls.

    ``adjusted_label`` is the lab-rule-adjusted label when a rule fired, else the label's
    vocabulary-``closest`` when it is out of vocab and a plausible near-match exists, else
    ``None`` (leave the label as-is).

    A marker complaint about a NOVEL (out-of-vocab) label still fails the gate: the evidence
    must back whatever the LLM cited regardless of whether the label is in the vocabulary.

    Returns::

        {"passed": bool, "flags": [str,...], "marker": <marker_presence>,
         "citations": <check_citations>, "rules": <apply_lab_rules>,
         "vocab": <validate_vocabulary>, "adjusted_label": str|None,
         "confidence_penalty": float}
    """
    ld = label_dict if isinstance(label_dict, dict) else {}
    cited_markers = ld.get("key_markers", []) or []
    citations = ld.get("citations", []) or []
    label = str(ld.get("label", "") or "")

    marker = marker_presence(cited_markers, code_evidence, z_thresh=z_thresh)
    cites = check_citations(citations, resolver=resolver, markers=cited_markers)
    rules = apply_lab_rules(ld, code_evidence, project_context, adata=adata, code=code)
    vocab = validate_vocabulary(label, project_context, kind=kind)

    flags: list[str] = []
    n_failing = 0

    # category 1: marker precision (only when markers were actually cited)
    if marker["n_cited"] > 0 and marker["precision"] < min_marker_precision:
        flags.append(
            f"low marker precision {marker['precision']:.2f} < {min_marker_precision} "
            f"(absent: {marker['absent']})"
        )
        n_failing += 1

    # category 2: fabricated citation (a parseable id the injected resolver could not retrieve)
    fabricated = [c for c in cites if c["resolved"] is False]
    if fabricated:
        flags.append(f"fabricated/unretrievable citation id(s): {[c['id'] for c in fabricated]}")
        n_failing += 1

    # category 3: lab-rule violations (each appended verbatim)
    if rules["violations"]:
        flags.extend(rules["violations"])
        n_failing += 1

    penalty = min(1.0, round(_GATE_CATEGORY_PENALTY * n_failing, 4))
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
