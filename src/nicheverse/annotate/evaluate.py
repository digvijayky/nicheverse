"""Deterministic evaluation and provenance for LLM codebook annotation.

Scores each proposed code label against an independent reference and records an
auditable run manifest, so a codebook annotation harness can be graded and replayed
without any LLM in the loop. All grading here is deterministic (no LLM-as-judge),
following the spatial-agent evaluation literature:

* GPTCelltype (Hou & Ji, Nat Methods 2024) scores agreement on a 1 / 0.5 / 0 scale
  for Full / Partial / Mismatch. :func:`score_code` reuses that convention for
  ``label_match``: exact (or synonym) match = 1.0, same broad lineage = 0.5,
  otherwise 0.0.
* CellTypeAgent and the "Beyond the Hype" benchmark stress that a correct call at
  the wrong granularity (over- or under-calling specificity) is a distinct error;
  :func:`score_code` reports an ``granularity`` verdict from a lineage-depth map.
* SpatialBench-style deterministic grading: achievable-range ground truth, marker
  precision / recall, and, above all, stratified reporting (per platform / per
  compartment) so a single headline number never hides a weak stratum. Aggregation
  in :func:`summarize` is therefore stratified when a ``platform`` or ``compartment``
  key is present on the per-code records.

Cell Ontology approximation. We do NOT load a full Cell Ontology (CL) tree. Instead
the caller passes a small ``lineage_map`` (label -> broad lineage string, optionally
label -> (lineage, depth)) that approximates the ontology tiers: the "same broad
lineage" tier of ``label_match`` and the granularity comparison both read from this
map. This is a deliberate, documented approximation of the CL hierarchy; with no map,
``label_match`` degrades to exact/synonym-vs-mismatch only and ``granularity`` is None.

Dependencies are intentionally light (numpy / pandas, and scipy only for a rank
correlation). No scanpy, no network, no GPU.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

__all__ = [
    "score_code",
    "calibration",
    "scorecard_table",
    "summarize",
    "write_provenance_manifest",
]


# ---------------------------------------------------------------------------
# label / marker normalization helpers
# ---------------------------------------------------------------------------

def _norm_label(s) -> str:
    """Lower-case, collapse whitespace and common separators for label comparison."""
    if s is None:
        return ""
    t = str(s).strip().lower()
    for ch in ("_", "-", "/", "+"):
        t = t.replace(ch, " ")
    return " ".join(t.split())


def _norm_gene(g) -> str:
    return str(g).strip().upper()


def _as_marker_list(x) -> list[str]:
    """Coerce a key_markers field (list, or ';'/','-joined string) to a gene list."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        items = list(x)
    else:
        s = str(x)
        sep = ";" if ";" in s else ","
        items = s.split(sep)
    out, seen = [], set()
    for it in items:
        g = _norm_gene(it)
        if g and g not in seen:
            seen.add(g)
            out.append(g)
    return out


def _lineage_of(label: str, lineage_map: dict | None) -> str | None:
    """Look up the broad lineage for a label in ``lineage_map`` (norm-insensitive).

    A map value may be a plain lineage string or a ``(lineage, depth)`` pair; either
    way this returns the lineage string (or None if unmapped / no map).
    """
    if not lineage_map:
        return None
    nl = _norm_label(label)
    for k, v in lineage_map.items():
        if _norm_label(k) == nl:
            lin = v[0] if isinstance(v, (tuple, list)) else v
            return _norm_label(lin) or None
    return None


def _depth_of(label: str, lineage_map: dict | None):
    """Return the specificity depth of a label if the map provides one, else None.

    Depth comes from a ``(lineage, depth)`` map value; a larger depth = more specific
    (deeper in the ontology). Absent depth info -> None (granularity not computable).
    """
    if not lineage_map:
        return None
    nl = _norm_label(label)
    for k, v in lineage_map.items():
        if _norm_label(k) == nl and isinstance(v, (tuple, list)) and len(v) > 1:
            try:
                return float(v[1])
            except (TypeError, ValueError):
                return None
    return None


def _enriched_genes(code_evidence: dict, z_thresh: float, k: int) -> tuple[set[str], set[str]]:
    """Genes considered enriched for a code, plus its ordered top-k marker genes.

    A gene is enriched if it is among ``top_markers`` with ``z >= z_thresh`` or it
    appears in ``top_degs`` (positive log2FC). ``top_k`` is the code's own leading
    marker set used for marker-recall.
    """
    top_markers = code_evidence.get("top_markers", []) or []
    enriched: set[str] = set()
    topk_genes: list[str] = []
    for entry in top_markers:
        if isinstance(entry, (tuple, list)) and len(entry) >= 2:
            g, z = entry[0], entry[1]
        else:
            g, z = entry, None
        gn = _norm_gene(g)
        if len(topk_genes) < k:
            topk_genes.append(gn)
        try:
            if z is None or float(z) >= z_thresh:
                enriched.add(gn)
        except (TypeError, ValueError):
            enriched.add(gn)
    for entry in code_evidence.get("top_degs", []) or []:
        if isinstance(entry, (tuple, list)) and len(entry) >= 1:
            g = entry[0]
            lfc = entry[1] if len(entry) >= 2 else None
            try:
                if lfc is None or float(lfc) > 0:
                    enriched.add(_norm_gene(g))
            except (TypeError, ValueError):
                enriched.add(_norm_gene(g))
    return enriched, set(topk_genes)


# ---------------------------------------------------------------------------
# per-code scorecard
# ---------------------------------------------------------------------------

def score_code(
    proposed: dict,
    reference_label: str,
    code_evidence: dict,
    *,
    lineage_map: dict | None = None,
    gate_result: dict | None = None,
    z_thresh: float = 1.0,
    k: int = 15,
) -> dict:
    """Score one proposed code label against an independent reference label.

    Parameters
    ----------
    proposed
        The LLM's proposed-label dict, e.g. ``{"label", "compartment", "confidence",
        "key_markers", "citations", ...}``. ``key_markers`` may be a list or a
        ``;`` / ``,``-joined string; missing keys are tolerated.
    reference_label
        Independent ground-truth label for the code (e.g. the majority Leiden group
        from :func:`code_groundtruth_concordance`). May be None/empty (then
        ``label_match`` is None).
    code_evidence
        Per-code evidence dict from :func:`nicheverse.annotate.code_evidence`
        (``top_markers=[(gene, z)]``, optional ``top_degs``, ``dist_<col>`` keys).
    lineage_map
        Optional label -> broad lineage (or label -> ``(lineage, depth)``) map that
        approximates the Cell Ontology tiers. See the module docstring.
    gate_result
        Optional dict from a decoupled gate (verify.gate), carrying marker precision,
        citation resolution, and rule violations. Passed in so this module never
        imports verify. Recognized keys (all optional): ``rules_pass`` (bool) or
        ``rule_violations`` (list); ``n_absent_markers`` / ``absent_markers``;
        ``n_unresolved_citations`` / ``unresolved_citations``; ``marker_precision``.
    z_thresh
        A ``top_markers`` gene counts as enriched when its z-score >= this.
    k
        Marker precision/recall are computed against the code's top-``k`` markers.

    Returns
    -------
    dict
        Flat scorecard with the dimensions (A) ``label_match`` (1.0/0.5/0.0/None),
        (B) ``granularity`` ("over_call"/"under_call"/"match"/None), (C)
        ``marker_precision`` / ``marker_recall``, (D) ``rule_compliance``
        ("pass"/"fail"), (E) hallucination counts ``n_absent_markers`` /
        ``n_unresolved_citations``, plus the raw proposed label, reference, compartment,
        confidence, code, and any ``platform`` carried on the proposed dict.
    """
    proposed = dict(proposed or {})
    code_evidence = dict(code_evidence or {})
    prop_label = proposed.get("label", "")

    # (A) label_match on the GPTCelltype 1 / 0.5 / 0 (Full / Partial / Mismatch) scale.
    if reference_label is None or _norm_label(reference_label) == "":
        label_match = None
    elif _labels_synonymous(prop_label, reference_label, proposed.get("synonyms")):
        label_match = 1.0
    else:
        lp = _lineage_of(prop_label, lineage_map)
        lr = _lineage_of(reference_label, lineage_map)
        label_match = 0.5 if (lp is not None and lr is not None and lp == lr) else 0.0

    # (B) granularity from lineage-map depth (Cell-Ontology approximation).
    dp = _depth_of(prop_label, lineage_map)
    dr = _depth_of(reference_label, lineage_map)
    if dp is None or dr is None:
        granularity = None
    elif dp > dr:
        granularity = "over_call"     # proposed more specific than reference
    elif dp < dr:
        granularity = "under_call"    # proposed less specific than reference
    else:
        granularity = "match"

    # (C) marker precision / recall against the code's own enriched markers.
    enriched, topk = _enriched_genes(code_evidence, z_thresh=z_thresh, k=k)
    prop_markers = _as_marker_list(proposed.get("key_markers"))
    if prop_markers:
        hit = sum(1 for g in prop_markers if g in enriched)
        marker_precision = hit / len(prop_markers)
    else:
        marker_precision = None
    if topk:
        recalled = sum(1 for g in topk if g in prop_markers)
        marker_recall = recalled / len(topk)
    else:
        marker_recall = None
    absent_markers = [g for g in prop_markers if g not in enriched]

    # (D) rule compliance from the gate (pass when no gate is supplied).
    if gate_result is None:
        rule_compliance = "pass"
    else:
        if "rules_pass" in gate_result:
            rule_compliance = "pass" if gate_result.get("rules_pass") else "fail"
        else:
            viol = gate_result.get("rule_violations") or gate_result.get("violations") or []
            rule_compliance = "fail" if len(viol) else "pass"

    # (E) hallucination counts: prefer the gate's numbers, else recompute from evidence.
    if gate_result is not None and (
        "n_absent_markers" in gate_result or "absent_markers" in gate_result
    ):
        n_absent = int(
            gate_result.get("n_absent_markers", len(gate_result.get("absent_markers", []) or []))
        )
    else:
        n_absent = len(absent_markers)
    if gate_result is not None and (
        "n_unresolved_citations" in gate_result or "unresolved_citations" in gate_result
    ):
        n_unresolved = int(
            gate_result.get(
                "n_unresolved_citations", len(gate_result.get("unresolved_citations", []) or [])
            )
        )
    else:
        n_unresolved = None  # cannot judge citation resolution without the gate

    rec = {
        "code": str(proposed.get("code", code_evidence.get("code", ""))),
        "proposed_label": prop_label,
        "reference_label": reference_label,
        "compartment": proposed.get("compartment", ""),
        "confidence": _coerce_conf(proposed.get("confidence")),
        "label_match": label_match,
        "granularity": granularity,
        "marker_precision": marker_precision,
        "marker_recall": marker_recall,
        "rule_compliance": rule_compliance,
        "n_absent_markers": n_absent,
        "n_unresolved_citations": n_unresolved,
        "n_proposed_markers": len(prop_markers),
        "absent_markers": ";".join(absent_markers),
    }
    if gate_result is not None and gate_result.get("marker_precision") is not None:
        rec["gate_marker_precision"] = _coerce_conf(gate_result.get("marker_precision"))
    if "platform" in proposed:
        rec["platform"] = proposed.get("platform")
    return rec


def _coerce_conf(c):
    """Coerce a confidence to float in a tolerant way; None on failure."""
    if c is None:
        return None
    try:
        return float(c)
    except (TypeError, ValueError):
        return None


def _labels_synonymous(a, b, synonyms=None) -> bool:
    """Exact (case/space/separator-insensitive) equality, or a declared synonym."""
    na, nb = _norm_label(a), _norm_label(b)
    if na and na == nb:
        return True
    for s in synonyms or []:
        if _norm_label(s) == nb or _norm_label(s) == na:
            return True
    return False


# ---------------------------------------------------------------------------
# calibration: does stated confidence predict correctness?
# ---------------------------------------------------------------------------

def calibration(scorecards: list[dict], *, n_bins: int = 4) -> dict:
    """Does the LLM's stated confidence predict whether it was right?

    Bins scorecards by ``confidence`` and reports the mean ``label_match`` per bin,
    plus the Spearman rank correlation between confidence and ``label_match``. Records
    with a missing confidence or a None ``label_match`` are dropped from the
    correlation (and binning) but counted in ``n_total``.

    Returns ``{"spearman": float|None, "bins": {bin: {...}}, "n": int, "n_total": int}``.
    A positive Spearman means higher confidence tracks higher correctness.
    """
    conf, corr = [], []
    n_total = len(scorecards)
    for s in scorecards:
        c = _coerce_conf(s.get("confidence"))
        lm = s.get("label_match")
        if c is None or lm is None:
            continue
        conf.append(c)
        corr.append(float(lm))
    n = len(conf)
    out: dict = {"spearman": None, "bins": {}, "n": n, "n_total": n_total}
    if n == 0:
        return out

    conf_a = np.asarray(conf, dtype=float)
    corr_a = np.asarray(corr, dtype=float)

    # Spearman needs variance in both variables to be defined.
    if n >= 2 and np.ptp(conf_a) > 0 and np.ptp(corr_a) > 0:
        from scipy.stats import spearmanr

        rho, _ = spearmanr(conf_a, corr_a)
        out["spearman"] = None if (rho is None or np.isnan(rho)) else float(rho)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf_a, edges[1:-1], right=False), 0, n_bins - 1)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        lo, hi = edges[b], edges[b + 1]
        out["bins"][f"[{lo:.2f},{hi:.2f})"] = {
            "n": int(m.sum()),
            "mean_confidence": float(conf_a[m].mean()),
            "mean_label_match": float(corr_a[m].mean()),
        }
    return out


# ---------------------------------------------------------------------------
# aggregation + table
# ---------------------------------------------------------------------------

def _mean_ignore_none(values) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def summarize(scorecards: list[dict]) -> dict:
    """Aggregate a list of scorecards into headline + stratified metrics.

    Returns the overall means (label_match, marker_precision, marker_recall), the rule
    fail rate, the total hallucinated markers / citations, and, SpatialBench-style, a
    per-stratum breakdown for every ``platform`` and ``compartment`` value present on
    the records, so a single headline number can never hide a weak stratum.
    """
    n = len(scorecards)
    total_absent = sum(int(s.get("n_absent_markers") or 0) for s in scorecards)
    total_unresolved = sum(
        int(s["n_unresolved_citations"])
        for s in scorecards
        if s.get("n_unresolved_citations") is not None
    )
    n_fail = sum(1 for s in scorecards if s.get("rule_compliance") == "fail")

    def block(cards) -> dict:
        return {
            "n": len(cards),
            "mean_label_match": _mean_ignore_none(c.get("label_match") for c in cards),
            "mean_marker_precision": _mean_ignore_none(c.get("marker_precision") for c in cards),
            "mean_marker_recall": _mean_ignore_none(c.get("marker_recall") for c in cards),
            "rule_fail_rate": (
                sum(1 for c in cards if c.get("rule_compliance") == "fail") / len(cards)
                if cards else None
            ),
        }

    out: dict = block(scorecards)
    out["n"] = n
    out["rule_fail_rate"] = (n_fail / n) if n else None
    out["total_hallucinated_markers"] = total_absent
    out["total_hallucinated_citations"] = total_unresolved
    out["by"] = {}
    for strat in ("platform", "compartment"):
        present = [s for s in scorecards if s.get(strat) not in (None, "")]
        if not present:
            continue
        groups: dict = {}
        for s in present:
            groups.setdefault(str(s.get(strat)), []).append(s)
        out["by"][strat] = {g: block(cards) for g, cards in sorted(groups.items())}
    return out


def scorecard_table(scorecards: list[dict]) -> pd.DataFrame:
    """One row per code with every scorecard dimension (a tidy grading table)."""
    cols = [
        "code", "proposed_label", "reference_label", "compartment", "platform",
        "confidence", "label_match", "granularity", "marker_precision", "marker_recall",
        "rule_compliance", "n_absent_markers", "n_unresolved_citations",
        "n_proposed_markers", "absent_markers",
    ]
    if not scorecards:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(scorecards)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    ordered = cols + [c for c in df.columns if c not in cols]
    return df[ordered].reset_index(drop=True)


# ---------------------------------------------------------------------------
# provenance manifest
# ---------------------------------------------------------------------------

def _hash_obj(obj) -> str:
    """Stable short SHA-256 of any JSON-serializable object (sorted keys)."""
    try:
        blob = json.dumps(obj, sort_keys=True, default=str)
    except TypeError:
        blob = str(obj)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def write_provenance_manifest(
    out_dir: str, run_meta: dict, per_code_records: list[dict]
) -> str:
    """Persist an auditable, replayable run manifest as JSON (+ a scorecard CSV).

    The manifest captures ``run_meta`` verbatim (model id, seed, temperature,
    ProjectContext summary, lab-rule version, and a caller-supplied ``timestamp`` --
    this function never calls ``datetime.now``) and, per code, an evidence hash, a
    prompt hash, the raw LLM output, the gate result, the citations, and the final
    label, so any label can be audited or replayed. A ``scorecards`` array is also
    embedded when the records carry scorecard fields, and the scorecard table is
    dumped to ``provenance_scorecards.csv`` next to the manifest.

    Parameters
    ----------
    out_dir
        Directory for the manifest / CSV (created with ``exist_ok=True``).
    run_meta
        Run-level metadata (see above). Copied verbatim into the manifest.
    per_code_records
        One dict per code. Any of ``code``, ``evidence`` (hashed if present, else a
        precomputed ``evidence_hash`` is used), ``prompt`` (-> ``prompt_hash`` or a
        precomputed one), ``raw_output`` / ``output``, ``gate_result``, ``citations``,
        and ``final_label`` / ``label`` are recorded; scorecard fields pass through.

    Returns
    -------
    str
        Absolute path to the written manifest JSON.
    """
    os.makedirs(out_dir, exist_ok=True)

    codes_out = []
    scorecards = []
    for r in per_code_records:
        r = dict(r or {})
        ev = r.get("evidence")
        prompt = r.get("prompt")
        entry = {
            "code": str(r.get("code", "")),
            "evidence_hash": r.get("evidence_hash") or (_hash_obj(ev) if ev is not None else None),
            "prompt_hash": r.get("prompt_hash") or (_hash_obj(prompt) if prompt is not None else None),
            "raw_output": r.get("raw_output", r.get("output")),
            "gate_result": r.get("gate_result"),
            "citations": r.get("citations"),
            "final_label": r.get("final_label", r.get("proposed_label", r.get("label"))),
        }
        codes_out.append(entry)
        # keep scorecard-shaped fields for the embedded array + CSV
        if any(key in r for key in ("label_match", "marker_precision", "rule_compliance")):
            scorecards.append(r)

    manifest = {
        "run_meta": run_meta,
        "n_codes": len(per_code_records),
        "codes": codes_out,
        "scorecards": scorecards,
        "summary": summarize(scorecards) if scorecards else None,
    }
    manifest_path = os.path.abspath(os.path.join(out_dir, "provenance_manifest.json"))
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, default=str)

    csv_path = os.path.join(out_dir, "provenance_scorecards.csv")
    scorecard_table(scorecards if scorecards else per_code_records).to_csv(csv_path, index=False)

    return manifest_path
