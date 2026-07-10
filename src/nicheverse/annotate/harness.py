"""Orchestration core of the nicheverse codebook-annotation harness.

This is a PLAIN LINEAR pipeline, not a multi-agent platform. After reviewing the
spatial-annotation-agent literature (SpatialAgent, STAgent, STAT-agent, CASSIA,
CellTypeAgent, NicheAgent, STAnalyzer) the deliberate conclusion was that a single
deterministic pipeline with hard verification gates and one adversarial pass is
easier to audit, cheaper, and more reproducible than a graph of cooperating agents.
There is no tool registry, no LangGraph, and no external service here: the stages run
in a fixed order and every decision is recorded for provenance.

Per code the stages are:

1. Evidence. :func:`artifacts.code_evidence` (cells) or :func:`artifacts.niche_evidence`
   (niches) turns the code into markers / DEGs / composition / distributions.
2. Propose. :func:`annotate.propose_label` asks the labeler for a ranked candidate set
   plus a single best label, grounded in the project context + allowed vocabulary.
3. Verify gates (anti-hallucination). :func:`verify.gate` checks marker presence and
   citation resolution and applies the lab rules; a rule-adjusted label is adopted and
   the stated confidence is reduced by the gate's penalty.
4. Adversarial refuter. :func:`annotate.refute_label`, whose sole objective is to argue
   the label is wrong, runs BLIND to the labeler's confidence. If it proposes a revision
   whose markers are better supported in this code's own evidence, the label flips;
   otherwise the disagreement is flagged.
5. Reconcile (cross-code). Reuse the existing refine pass to disambiguate identical
   labels and merge only true duplicates, then flag two codes with near-identical
   z-profiles but divergent labels, plus hierarchical-cluster consistency.
6. Confidence-gated review split. Codes that passed the gate AND were refuter-agreed AND
   clear the confidence threshold auto-accept; the rest go to a review table.
7. Provenance / scoring. Per-code records (evidence hash, resolved prompt, raw output,
   gate result, refuter verdict, citations, final label) feed
   :func:`evaluate.write_provenance_manifest`; with a ground-truth column,
   :func:`evaluate.score_code` + :func:`evaluate.calibration` grade the run.

Scope: imaging-based / in-situ spatial transcriptomics (IMST) only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import annotate as _A
from .annotate import AnnotationConfig, attach_labels, propose_label, refute_label
from .artifacts import cluster_codes, code_evidence, niche_evidence
from .context import ProjectContext
from .evaluate import calibration, score_code, scorecard_table, write_provenance_manifest
from .verify import gate, marker_presence

__all__ = ["AnnotationResult", "annotate_codebook"]


@dataclass
class AnnotationResult:
    """Everything the harness produces for one codebook annotation run.

    Attributes
    ----------
    labels_df
        One row per code: ``final_label``, ``compartment``, ``confidence`` (post-penalty),
        ``passed`` (gate), ``refuter_agree``, and ``flags``.
    review_df
        Subset of codes that did not auto-accept (gate failed, refuter disagreed, or low
        confidence), with a ``review_reason`` column, for manual sign-off.
    scorecard_df
        Per-code grading table (from :func:`evaluate.score_code`) when a ground-truth column
        was supplied, else ``None``.
    calibration
        Confidence-vs-correctness calibration dict when scored, else ``None``.
    manifest_path
        Path to the provenance manifest JSON when ``out_dir`` was given, else ``None``.
    records
        The per-code provenance records (kept in memory for inspection / re-use).
    kind
        ``"cell"`` or ``"niche"``.
    """

    labels_df: pd.DataFrame
    review_df: pd.DataFrame
    scorecard_df: "pd.DataFrame | None" = None
    calibration: "dict | None" = None
    manifest_path: "str | None" = None
    records: list = field(default_factory=list)
    kind: str = "cell"

    def label_map(self) -> dict:
        """``code -> final_label`` mapping for attaching onto ``obs``."""
        return {str(c): str(v) for c, v in self.labels_df["final_label"].items()}

    def attach(self, adata, key_added: str = "celltype_annot"):
        """Map final labels onto ``adata.obs[key_added]`` via :func:`annotate.attach_labels`.

        The code column is inferred from the label DataFrame's index name when present.
        """
        code_col = self.labels_df.index.name or (
            "neighborhood_codebook_idx" if self.kind == "niche" else "cell_codebook_idx"
        )
        return attach_labels(adata, code_col, self.label_map(), key_added=key_added)


def _best_from_proposal(proposed: dict, kind: str) -> dict:
    """Normalize a labeler reply into the flat best-label dict the gate + scoring consume.

    Falls back to the top-ranked entry of ``candidates`` when a top-level ``label`` is absent, so
    a model that only filled the ranked list still yields a usable best label.
    """
    p = dict(proposed or {})
    cands = p.get("candidates") or []
    if not p.get("label") and cands:
        top = cands[0] if isinstance(cands[0], dict) else {}
        p["label"] = top.get("label", "")
        if not p.get("key_markers") and top.get("supporting_markers"):
            p["key_markers"] = top.get("supporting_markers")
        if p.get("confidence") is None and top.get("confidence") is not None:
            p["confidence"] = top.get("confidence")
    best = {
        "label": str(p.get("label", "") or ""),
        "key_markers": list(p.get("key_markers", []) or []),
        "citations": list(p.get("citations", []) or []),
        "confidence": p.get("confidence"),
        "rationale": str(p.get("rationale", "") or ""),
        "candidates": cands,
    }
    if kind == "niche":
        best["dominant_types"] = list(p.get("dominant_types", []) or [])
        best["compartment"] = ""
    else:
        best["compartment"] = str(p.get("compartment", "") or "")
    return best


def _coerce_conf(c, default: float = 0.5) -> float:
    try:
        v = float(c)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, v))


def _refuter_alternative_better(alt_label: str, alt_markers, ev: dict, current_markers, z_thresh: float) -> bool:
    """Is the refuter's alternative better supported in THIS code's own evidence?

    The alternative wins only if its discriminating markers clear the marker-presence bar at a
    precision at least as high as the current label's cited markers (ties broken toward the
    alternative). An alternative with no discriminating markers, or no better support, does not
    flip the label; the disagreement is flagged instead. This keeps the flip grounded in measured
    enrichment rather than the refuter's assertion.
    """
    if not str(alt_label or "").strip() or not alt_markers:
        return False
    alt_mp = marker_presence(list(alt_markers), ev, z_thresh=z_thresh)
    if not alt_mp["present"]:
        return False
    cur_mp = marker_presence(list(current_markers or []), ev, z_thresh=z_thresh)
    return alt_mp["precision"] >= cur_mp["precision"]


def _code_mean_profiles(adata, code_col: str) -> tuple:
    """Per-code log1p mean-expression matrix (codes sorted like the evidence dict)."""
    import scipy.sparse as sp

    codes = adata.obs[code_col].astype(str)
    uniq = sorted(codes.unique(), key=lambda c: (len(c), c))
    x = adata.X
    xd = x.toarray() if sp.issparse(x) else np.asarray(x)
    xd = np.nan_to_num(xd, nan=0.0, posinf=0.0, neginf=0.0)
    means = np.vstack([xd[(codes == c).to_numpy()].mean(0) for c in uniq])
    return uniq, np.log1p(means)


def _near_identical_evidence_conflicts(
    adata, code_col: str, labels: dict, *, corr_thresh: float = 0.98
) -> list[tuple]:
    """Flag code PAIRS whose mean-expression profiles are near-identical yet whose labels differ.

    Two codes that describe the same underlying cell population have near-identical mean-expression
    vectors; if they nevertheless carry different labels that is an internal inconsistency the
    reviewer should resolve. Similarity is the Pearson correlation of the codes' log1p
    mean-expression profiles. This deliberately uses mean expression, NOT the z-scored top markers:
    z-scores measure code-SPECIFICITY, so a program shared by two codes drops OUT of both codes' top
    markers and cannot reveal the duplication. Returns ``[(code_a, code_b, corr), ...]`` for pairs at
    or above ``corr_thresh`` with divergent labels.
    """
    uniq, prof = _code_mean_profiles(adata, code_col)
    idx = {c: i for i, c in enumerate(uniq)}
    out: list[tuple] = []
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            a, b = uniq[i], uniq[j]
            if str(labels.get(a, "")) == str(labels.get(b, "")):
                continue
            va, vb = prof[idx[a]], prof[idx[b]]
            if va.std() < 1e-9 or vb.std() < 1e-9:
                continue
            corr = float(np.corrcoef(va, vb)[0, 1])
            if corr >= corr_thresh:
                out.append((a, b, round(corr, 4)))
    return out


def _reconcile(records: list[dict], cfg: AnnotationConfig, ev: dict, adata, code_col: str, kind: str) -> None:
    """Cross-code reconcile pass, mutating each record's ``final_label`` / ``flags`` in place.

    Three checks: (1) reuse the existing refine logic to disambiguate identical labels and merge
    only true duplicates; (2) flag code pairs with near-identical z-profiles but divergent labels;
    (3) hierarchical-cluster consistency, flagging a code whose label disagrees with the majority
    label of its :func:`artifacts.cluster_codes` group.
    """
    if len(records) < 2:
        return
    by_code = {r["code"]: r for r in records}

    # (1) existing refine pass over the current best labels.
    try:
        cluster_map = {}
        if cfg.cluster_context:
            cl = cluster_codes(adata, code_col)
            cluster_map = {str(c): int(cl.loc[c, "cluster"]) for c in cl.index}
        tbl = {}
        for r in records:
            row = {
                "label": r["final_label"],
                "compartment": r.get("compartment", ""),
                "key_markers": "; ".join(map(str, r.get("key_markers", []))),
            }
            if cluster_map:
                row["cluster"] = cluster_map.get(str(r["code"]), -1)
            tbl[str(r["code"])] = row
        # Route through _A.call_llm (not the harness-local import) so the same monkeypatch that
        # stubs the labeler/refuter also drives the reconcile pass in offline tests.
        rj = _A.parse_json(
            _A.call_llm(
                _A._refine_prompt(tbl, cfg.tissue),
                provider=cfg.provider, model=cfg.model, api_key=cfg.api_key, system=_A._SYSTEM,
            )
        )
        for code, newlab in (rj.get("revisions", {}) or {}).items():
            if str(code) in by_code:
                lab = newlab if isinstance(newlab, str) else str(newlab)
                r = by_code[str(code)]
                if lab and lab != r["final_label"]:
                    r["flags"].append(f"reconcile: relabeled '{r['final_label']}' -> '{lab}'")
                    r["final_label"] = lab
    except Exception as exc:  # reconcile is best-effort; never sink the whole run
        for r in records:
            r["flags"].append(f"reconcile-refine skipped ({type(exc).__name__})")

    # (2) near-identical mean-expression profile but divergent labels.
    labels_now = {r["code"]: r["final_label"] for r in records}
    for a, b, corr in _near_identical_evidence_conflicts(adata, code_col, labels_now):
        for c, other in ((a, b), (b, a)):
            by_code[c]["flags"].append(
                f"near-identical expression profile to code {other} (corr={corr}) but different label"
            )

    # (3) hierarchical-cluster consistency.
    try:
        cl = cluster_codes(adata, code_col)
        groups: dict = {}
        for c in cl.index:
            groups.setdefault(int(cl.loc[c, "cluster"]), []).append(str(c))
        for members in groups.values():
            if len(members) < 2:
                continue
            labs = [by_code[m]["final_label"] for m in members if m in by_code]
            if not labs:
                continue
            majority = pd.Series(labs).value_counts().idxmax()
            for m in members:
                if m in by_code and by_code[m]["final_label"] != majority:
                    by_code[m]["flags"].append(
                        f"hier-cluster mismatch: label differs from cluster majority '{majority}'"
                    )
    except Exception:
        pass


def annotate_codebook(
    adata,
    code_col: str,
    *,
    config: AnnotationConfig | None = None,
    kind: str = "cell",
    celltype_col: str | None = None,
    groundtruth_col: str | None = None,
    out_dir: str | None = None,
    run_meta: dict | None = None,
    resolver=None,
    lineage_map: dict | None = None,
) -> AnnotationResult:
    """Run the full labeler -> gate -> refuter -> reconcile -> review -> provenance pipeline.

    Parameters
    ----------
    adata, code_col
        AnnData and the ``obs`` column holding the code index (cell or niche codes).
    config
        :class:`~nicheverse.annotate.annotate.AnnotationConfig`; set ``config.context`` to a
        :class:`~nicheverse.annotate.context.ProjectContext` to ground labels and supply the allowed
        vocabulary. ``config.refuter`` toggles the adversarial pass; ``config.min_marker_precision``
        and ``config.confidence_review_threshold`` tune the gate and the review split.
    kind
        ``"cell"`` (default) or ``"niche"``. For niches, ``celltype_col`` (cell-level labels) is
        required for the composition-primary evidence.
    celltype_col
        Cell-type label column used to build niche composition (``kind='niche'`` only).
    groundtruth_col
        Optional independent label column; when given, each code is scored against its majority
        ground-truth label and the run is calibrated.
    out_dir
        When given, the provenance manifest + scorecard CSV are written here.
    run_meta
        Run-level metadata recorded verbatim in the manifest (model, seed, timestamp, ...).
    resolver
        Injectable citation resolver ``(id) -> record|None`` handed to :func:`verify.gate`
        (keeps citation checks offline in tests).
    lineage_map
        Optional label -> lineage (or ``(lineage, depth)``) map for :func:`evaluate.score_code`.

    Returns
    -------
    AnnotationResult
    """
    cfg = config or AnnotationConfig()
    ctx = getattr(cfg, "context", None)
    if ctx is not None and not isinstance(ctx, ProjectContext):
        ctx = ProjectContext.from_dict(ctx) if isinstance(ctx, dict) else ctx
        cfg.context = ctx
    z_thresh = 1.0

    # -- stage 1: evidence -------------------------------------------------
    if kind == "niche":
        if not celltype_col:
            raise ValueError("kind='niche' requires celltype_col (cell-level labels for composition).")
        ev = niche_evidence(
            adata, code_col, celltype_col, extra_cols=cfg.context_cols, top_markers=cfg.top_markers
        )
    else:
        ev = code_evidence(
            adata, code_col, extra_cols=cfg.context_cols, top_markers=cfg.top_markers, top_degs=cfg.top_degs
        )

    records: list[dict] = []
    for code, e in ev.items():
        flags: list[str] = []

        # -- stage 2: propose (ranked candidates + best) -------------------
        proposed = propose_label(code, e, cfg, kind=kind)
        best = _best_from_proposal(proposed, kind)
        best["code"] = code
        base_conf = _coerce_conf(best.get("confidence"))
        final_label = best["label"]

        # -- stage 3: verify gates ----------------------------------------
        g = gate(
            best, e, cfg.context, kind=kind, resolver=resolver, adata=adata, code=code,
            z_thresh=z_thresh, min_marker_precision=cfg.min_marker_precision,
        )
        if g.get("adjusted_label"):
            if str(g["adjusted_label"]) != str(final_label):
                flags.append(f"gate adjusted '{final_label}' -> '{g['adjusted_label']}'")
            final_label = str(g["adjusted_label"])
        flags.extend(g.get("flags", []))
        conf = max(0.0, base_conf - float(g.get("confidence_penalty", 0.0)))

        # -- stage 4: adversarial refuter (blind to confidence) -----------
        refuter_agree = True
        refutation = None
        if cfg.refuter:
            refutation = refute_label(code, e, {"label": final_label}, cfg, kind=kind)
            if refutation["verdict"] == "revise":
                alt = refutation["alternative_label"]
                if _refuter_alternative_better(
                    alt, refutation["discriminating_markers"], e, best.get("key_markers"), z_thresh
                ):
                    flags.append(f"refuter flip: '{final_label}' -> '{alt}' ({refutation['reason']})")
                    final_label = str(alt)
                    refuter_agree = True  # we adopted the refuter's better-supported alternative
                else:
                    flags.append(
                        f"refuter disagrees (proposed '{alt}': {refutation['reason']}) but its markers "
                        "are not better supported; kept label and flagged"
                    )
                    refuter_agree = False

        records.append(
            {
                "code": str(code),
                "evidence": e,
                "prompt": proposed.get("_prompt", ""),
                "raw_output": proposed.get("_raw", ""),
                "proposed": best,
                "proposed_label": best["label"],
                "final_label": final_label,
                "compartment": best.get("compartment", ""),
                "confidence": round(conf, 4),
                "base_confidence": round(base_conf, 4),
                "passed": bool(g.get("passed")),
                "refuter_agree": bool(refuter_agree),
                "gate_result": g,
                "refuter": refutation,
                "citations": best.get("citations", []),
                "key_markers": best.get("key_markers", []),
                "candidates": best.get("candidates", []),
                "flags": flags,
            }
        )

    # -- stage 5: cross-code reconcile ------------------------------------
    _reconcile(records, cfg, ev, adata, code_col, kind)

    # -- stage 6: confidence-gated review split ---------------------------
    label_rows = []
    review_rows = []
    for r in records:
        auto = r["passed"] and r["refuter_agree"] and r["confidence"] >= cfg.confidence_review_threshold
        row = {
            "code": r["code"],
            "final_label": r["final_label"],
            "compartment": r.get("compartment", ""),
            "confidence": r["confidence"],
            "passed": r["passed"],
            "refuter_agree": r["refuter_agree"],
            "flags": "; ".join(r["flags"]),
        }
        label_rows.append(row)
        if not auto:
            reasons = []
            if not r["passed"]:
                reasons.append("gate failed")
            if not r["refuter_agree"]:
                reasons.append("refuter disagreed")
            if r["confidence"] < cfg.confidence_review_threshold:
                reasons.append(f"confidence {r['confidence']:.2f} < {cfg.confidence_review_threshold}")
            review_rows.append({**row, "review_reason": "; ".join(reasons)})

    labels_df = pd.DataFrame(label_rows).set_index("code") if label_rows else pd.DataFrame(
        columns=["final_label", "compartment", "confidence", "passed", "refuter_agree", "flags"]
    )
    if label_rows:
        labels_df.index.name = code_col
    review_df = pd.DataFrame(review_rows)
    if review_rows:
        review_df = review_df.set_index("code")
        review_df.index.name = code_col

    # -- stage 7: provenance + scoring ------------------------------------
    scorecard_df = None
    calib = None
    gt_majority: dict = {}
    if groundtruth_col is not None and groundtruth_col in adata.obs.columns:
        from .artifacts import code_groundtruth_concordance

        _, maj = code_groundtruth_concordance(adata, code_col, groundtruth_col)
        gt_majority = {str(row["code"]): row["majority_group"] for _, row in maj.iterrows()}
        scorecards = []
        for r in records:
            proposed_for_score = dict(r["proposed"])
            proposed_for_score["label"] = r["final_label"]
            proposed_for_score["confidence"] = r["confidence"]
            proposed_for_score["code"] = r["code"]
            gate_for_score = {
                "violations": r["gate_result"]["rules"]["violations"],
                "absent_markers": r["gate_result"]["marker"]["absent"],
                "marker_precision": r["gate_result"]["marker"]["precision"],
            }
            sc = score_code(
                proposed_for_score, gt_majority.get(r["code"]), r["evidence"],
                lineage_map=lineage_map, gate_result=gate_for_score, z_thresh=z_thresh,
            )
            r["scorecard"] = sc
            scorecards.append(sc)
        scorecard_df = scorecard_table(scorecards)
        calib = calibration(scorecards)

    manifest_path = None
    if out_dir is not None:
        prov_records = []
        for r in records:
            entry = {
                "code": r["code"],
                "evidence": r["evidence"],
                "prompt": r["prompt"],
                "raw_output": r["raw_output"],
                "gate_result": _jsonable(r["gate_result"]),
                "citations": r["citations"],
                "final_label": r["final_label"],
                "refuter_verdict": (r["refuter"] or {}).get("verdict") if r.get("refuter") else None,
                "confidence": r["confidence"],
                "passed": r["passed"],
                "refuter_agree": r["refuter_agree"],
                "flags": r["flags"],
            }
            if "scorecard" in r:
                entry.update(r["scorecard"])
            prov_records.append(entry)
        meta = dict(run_meta or {})
        meta.setdefault("kind", kind)
        meta.setdefault("provider", cfg.provider)
        meta.setdefault("model", cfg.model)
        meta.setdefault("refuter", cfg.refuter)
        meta.setdefault("min_marker_precision", cfg.min_marker_precision)
        meta.setdefault("confidence_review_threshold", cfg.confidence_review_threshold)
        if ctx is not None:
            meta.setdefault("project_context", ctx.to_prompt_block())
        manifest_path = write_provenance_manifest(out_dir, meta, prov_records)

    return AnnotationResult(
        labels_df=labels_df,
        review_df=review_df,
        scorecard_df=scorecard_df,
        calibration=calib,
        manifest_path=manifest_path,
        records=records,
        kind=kind,
    )


def _jsonable(obj):
    """Strip numpy scalars so a gate result serializes cleanly into the manifest."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
