"""Offline tests for the codebook-annotation harness (mock LLM + injected resolver).

Everything here runs without network or GPU: ``providers.call_llm`` is monkeypatched
(as in tests/test_annotate.py) and citation resolution uses an injected dict-backed
resolver, so the labeler / refuter / reconcile passes all resolve to canned JSON.
"""

from __future__ import annotations

import json
import subprocess
import sys

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import nicheverse.annotate.annotate as A
from nicheverse.annotate import AnnotationConfig, CellTypePrior, ProjectContext
from nicheverse.annotate.harness import AnnotationResult, annotate_codebook


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _adata():
    """~90 cells x 12 genes, 3 clean code groups (0/1/2) each with 3 marker genes hot."""
    rng = np.random.default_rng(0)
    x = rng.poisson(0.5, size=(90, 12)).astype("float32")
    x[:30, 0:3] += 8      # code 0 markers: G0 G1 G2
    x[30:60, 4:7] += 8    # code 1 markers: G4 G5 G6
    x[60:, 8:11] += 8     # code 2 markers: G8 G9 G10
    a = ad.AnnData(X=sp.csr_matrix(x))
    a.var_names = [f"G{i}" for i in range(12)]
    a.obs["cell_codebook_idx"] = np.array(["0"] * 30 + ["1"] * 30 + ["2"] * 30)
    a.obs["site_class"] = np.array(["BrM", "Primary", "Metastasis"] * 30)
    a.obs["leiden"] = np.array(["A"] * 30 + ["B"] * 30 + ["C"] * 30)
    return a


def _ctx():
    return ProjectContext(
        name="toy",
        species="human",
        disease="clear cell renal cell carcinoma",
        tissue="kidney",
        platform="xenium",
        site_col="site_class",
        expected_cell_types=[
            CellTypePrior(name="Tumor cell", markers=["G0", "G1", "G2"]),
            CellTypePrior(name="T cell", markers=["G4", "G5", "G6"]),
            CellTypePrior(name="Macrophage", markers=["G8", "G9", "G10"]),
        ],
    )


# Marker sets that are actually enriched in each code group (from _adata above).
_CODE_MARKERS = {"0": ["G0", "G1", "G2"], "1": ["G4", "G5", "G6"], "2": ["G8", "G9", "G10"]}


def _resolver(_id):
    """Everything cited resolves (so no fabricated-citation flags unless a test wants them)."""
    return {"title": "toy record", "journal": "J", "year": "2024", "first_author": "Doe"}


def _which_code(prompt: str) -> str:
    """Recover which code a labeler/refuter prompt is about from its 'Code N:'/'Niche N:' line."""
    import re

    m = re.search(r"(?:Code|Niche)\s+(\S+):", prompt)
    return m.group(1) if m else "0"


# ---------------------------------------------------------------------------
# 1. end-to-end: a label for every code
# ---------------------------------------------------------------------------

def test_annotate_codebook_labels_every_code(monkeypatch):
    def mock(prompt, **kw):
        if "revisions" in prompt:  # reconcile pass
            return '{"revisions": {}}'
        if "PROPOSED LABEL UNDER CHALLENGE" in prompt:  # refuter agrees
            return '{"verdict": "keep", "alternative_label": "", "discriminating_markers": [], "reason": "ok"}'
        code = _which_code(prompt)
        mk = _CODE_MARKERS[code]
        label = {"0": "Tumor cell", "1": "T cell", "2": "Macrophage"}[code]
        return json.dumps(
            {
                "candidates": [{"label": label, "supporting_markers": mk, "confidence": 0.9}],
                "label": label,
                "compartment": "immune",
                "confidence": 0.9,
                "rationale": "markers",
                "key_markers": mk,
                "citations": ["PMID:12345 Doe J 2024"],
            }
        )

    monkeypatch.setattr(A, "call_llm", mock)
    res = annotate_codebook(
        _adata(), "cell_codebook_idx",
        config=AnnotationConfig(context=_ctx(), refuter=True), resolver=_resolver,
    )
    assert isinstance(res, AnnotationResult)
    assert set(res.labels_df.index) == {"0", "1", "2"}
    assert res.labels_df.loc["0", "final_label"] == "Tumor cell"
    assert res.labels_df.loc["1", "final_label"] == "T cell"
    assert res.labels_df.loc["2", "final_label"] == "Macrophage"
    assert res.labels_df["passed"].all()
    assert res.labels_df["refuter_agree"].all()
    # every code auto-accepted -> empty review table
    assert len(res.review_df) == 0
    # attach maps final labels back onto obs
    a = _adata()
    res.attach(a, key_added="celltype_annot")
    assert set(a.obs["celltype_annot"]) == {"Tumor cell", "T cell", "Macrophage"}


# ---------------------------------------------------------------------------
# 2. refuter flips to a BETTER-SUPPORTED alternative
# ---------------------------------------------------------------------------

def test_refuter_flips_to_better_supported_alternative(monkeypatch):
    # Labeler calls code 0 "T cell" (WRONG: T-cell markers G4/G5/G6 are not enriched in code 0).
    # Refuter proposes "Tumor cell" with the actually-enriched G0/G1/G2 -> must flip.
    def mock(prompt, **kw):
        if "revisions" in prompt:
            return '{"revisions": {}}'
        if "PROPOSED LABEL UNDER CHALLENGE" in prompt:
            code = _which_code(prompt)
            if code == "0":
                return json.dumps(
                    {
                        "verdict": "revise",
                        "alternative_label": "Tumor cell",
                        "discriminating_markers": ["G0", "G1", "G2"],
                        "reason": "G0-G2 enriched, not the T-cell program",
                    }
                )
            return '{"verdict": "keep", "alternative_label": "", "discriminating_markers": [], "reason": "ok"}'
        code = _which_code(prompt)
        if code == "0":
            # deliberately wrong label + markers that ARE enriched so the gate still passes
            return json.dumps(
                {
                    "label": "T cell",
                    "compartment": "immune",
                    "confidence": 0.9,
                    "rationale": "x",
                    "key_markers": ["G0", "G1", "G2"],
                    "citations": ["PMID:1 Doe J 2024"],
                }
            )
        lab = {"1": "T cell", "2": "Macrophage"}[code]
        return json.dumps(
            {
                "label": lab, "compartment": "immune", "confidence": 0.9, "rationale": "x",
                "key_markers": _CODE_MARKERS[code], "citations": ["PMID:1 Doe J 2024"],
            }
        )

    monkeypatch.setattr(A, "call_llm", mock)
    res = annotate_codebook(
        _adata(), "cell_codebook_idx",
        config=AnnotationConfig(context=_ctx(), refuter=True), resolver=_resolver,
    )
    assert res.labels_df.loc["0", "final_label"] == "Tumor cell"  # flipped by the refuter
    assert any("refuter flip" in f for f in [res.labels_df.loc["0", "flags"]])


def test_refuter_unsupported_objection_is_flagged_not_obeyed(monkeypatch):
    # Refuter objects to code 0 with an alternative whose markers are NOT enriched -> keep + flag.
    def mock(prompt, **kw):
        if "revisions" in prompt:
            return '{"revisions": {}}'
        if "PROPOSED LABEL UNDER CHALLENGE" in prompt:
            code = _which_code(prompt)
            if code == "0":
                return json.dumps(
                    {
                        "verdict": "revise",
                        "alternative_label": "Macrophage",
                        "discriminating_markers": ["G8", "G9"],  # not enriched in code 0
                        "reason": "claims myeloid",
                    }
                )
            return '{"verdict": "keep", "alternative_label": "", "discriminating_markers": [], "reason": "ok"}'
        code = _which_code(prompt)
        lab = {"0": "Tumor cell", "1": "T cell", "2": "Macrophage"}[code]
        return json.dumps(
            {
                "label": lab, "compartment": "immune", "confidence": 0.9, "rationale": "x",
                "key_markers": _CODE_MARKERS[code], "citations": ["PMID:1 Doe J 2024"],
            }
        )

    monkeypatch.setattr(A, "call_llm", mock)
    res = annotate_codebook(
        _adata(), "cell_codebook_idx",
        config=AnnotationConfig(context=_ctx(), refuter=True), resolver=_resolver,
    )
    assert res.labels_df.loc["0", "final_label"] == "Tumor cell"  # NOT flipped
    assert res.labels_df.loc["0", "refuter_agree"] == False  # noqa: E712
    assert "0" in res.review_df.index  # refuter disagreement routes to review


# ---------------------------------------------------------------------------
# 3. gate catches a hallucinated (absent) marker
# ---------------------------------------------------------------------------

def test_absent_marker_caught_by_gate(monkeypatch):
    # Code 0 label cites G4/G5 (T-cell markers, NOT enriched in code 0) -> low precision -> fail.
    def mock(prompt, **kw):
        if "revisions" in prompt:
            return '{"revisions": {}}'
        if "PROPOSED LABEL UNDER CHALLENGE" in prompt:
            return '{"verdict": "keep", "alternative_label": "", "discriminating_markers": [], "reason": "ok"}'
        code = _which_code(prompt)
        if code == "0":
            return json.dumps(
                {
                    "label": "Tumor cell", "compartment": "epithelial", "confidence": 0.95,
                    "rationale": "x", "key_markers": ["G4", "G5", "G6"],  # all absent in code 0
                    "citations": ["PMID:1 Doe J 2024"],
                }
            )
        lab = {"1": "T cell", "2": "Macrophage"}[code]
        return json.dumps(
            {
                "label": lab, "compartment": "immune", "confidence": 0.9, "rationale": "x",
                "key_markers": _CODE_MARKERS[code], "citations": ["PMID:1 Doe J 2024"],
            }
        )

    monkeypatch.setattr(A, "call_llm", mock)
    res = annotate_codebook(
        _adata(), "cell_codebook_idx",
        config=AnnotationConfig(context=_ctx(), refuter=True), resolver=_resolver,
    )
    assert res.labels_df.loc["0", "passed"] == False  # noqa: E712
    r0 = next(r for r in res.records if r["code"] == "0")
    assert r0["gate_result"]["confidence_penalty"] > 0
    assert r0["confidence"] < r0["base_confidence"]  # penalty subtracted
    assert "0" in res.review_df.index


def test_fabricated_citation_caught_by_gate(monkeypatch):
    # A resolver that returns nothing for id 99999 -> that citation is fabricated -> gate fails.
    def resolver(_id):
        return None if str(_id) == "99999" else {"title": "ok", "journal": "J", "year": "2024"}

    def mock(prompt, **kw):
        if "revisions" in prompt:
            return '{"revisions": {}}'
        if "PROPOSED LABEL UNDER CHALLENGE" in prompt:
            return '{"verdict": "keep", "alternative_label": "", "discriminating_markers": [], "reason": "ok"}'
        code = _which_code(prompt)
        lab = {"0": "Tumor cell", "1": "T cell", "2": "Macrophage"}[code]
        cit = ["PMID:99999 Ghost J 2024"] if code == "0" else ["PMID:1 Doe J 2024"]
        return json.dumps(
            {
                "label": lab, "compartment": "immune", "confidence": 0.9, "rationale": "x",
                "key_markers": _CODE_MARKERS[code], "citations": cit,
            }
        )

    monkeypatch.setattr(A, "call_llm", mock)
    res = annotate_codebook(
        _adata(), "cell_codebook_idx",
        config=AnnotationConfig(context=_ctx(), refuter=True), resolver=resolver,
    )
    assert res.labels_df.loc["0", "passed"] == False  # noqa: E712
    r0 = next(r for r in res.records if r["code"] == "0")
    assert any("fabricated" in f or "unresolved" in f for f in r0["flags"])


# ---------------------------------------------------------------------------
# 4. low-confidence codes land in review_df
# ---------------------------------------------------------------------------

def test_low_confidence_routes_to_review(monkeypatch):
    def mock(prompt, **kw):
        if "revisions" in prompt:
            return '{"revisions": {}}'
        if "PROPOSED LABEL UNDER CHALLENGE" in prompt:
            return '{"verdict": "keep", "alternative_label": "", "discriminating_markers": [], "reason": "ok"}'
        code = _which_code(prompt)
        lab = {"0": "Tumor cell", "1": "T cell", "2": "Macrophage"}[code]
        conf = 0.3 if code == "1" else 0.9  # code 1 is low confidence
        return json.dumps(
            {
                "label": lab, "compartment": "immune", "confidence": conf, "rationale": "x",
                "key_markers": _CODE_MARKERS[code], "citations": ["PMID:1 Doe J 2024"],
            }
        )

    monkeypatch.setattr(A, "call_llm", mock)
    res = annotate_codebook(
        _adata(), "cell_codebook_idx",
        config=AnnotationConfig(context=_ctx(), refuter=True, confidence_review_threshold=0.6),
        resolver=_resolver,
    )
    assert "1" in res.review_df.index
    assert "confidence" in res.review_df.loc["1", "review_reason"]
    # the well-supported high-confidence codes still auto-accept
    assert "0" not in res.review_df.index and "2" not in res.review_df.index


# ---------------------------------------------------------------------------
# 5. provenance manifest is written and reloadable
# ---------------------------------------------------------------------------

def test_provenance_manifest_written(monkeypatch, tmp_path):
    def mock(prompt, **kw):
        if "revisions" in prompt:
            return '{"revisions": {}}'
        if "PROPOSED LABEL UNDER CHALLENGE" in prompt:
            return '{"verdict": "keep", "alternative_label": "", "discriminating_markers": [], "reason": "ok"}'
        code = _which_code(prompt)
        lab = {"0": "Tumor cell", "1": "T cell", "2": "Macrophage"}[code]
        return json.dumps(
            {
                "label": lab, "compartment": "immune", "confidence": 0.9, "rationale": "x",
                "key_markers": _CODE_MARKERS[code], "citations": ["PMID:1 Doe J 2024"],
            }
        )

    monkeypatch.setattr(A, "call_llm", mock)
    out = tmp_path / "prov"
    res = annotate_codebook(
        _adata(), "cell_codebook_idx",
        config=AnnotationConfig(context=_ctx(), refuter=True),
        groundtruth_col="leiden", out_dir=str(out),
        run_meta={"seed": 0, "timestamp": "2026-07-10T00:00:00Z"},
        resolver=_resolver,
    )
    assert res.manifest_path is not None
    manifest = json.loads(open(res.manifest_path).read())
    assert manifest["n_codes"] == 3
    assert {c["code"] for c in manifest["codes"]} == {"0", "1", "2"}
    for c in manifest["codes"]:
        assert c["evidence_hash"] and c["final_label"]  # hashed evidence + a final label
    # ground-truth scoring populated the scorecard
    assert res.scorecard_df is not None and len(res.scorecard_df) == 3
    assert res.calibration is not None


# ---------------------------------------------------------------------------
# 6. propose_label / refute_label thin callables
# ---------------------------------------------------------------------------

def test_propose_and_refute_callables(monkeypatch):
    from nicheverse.annotate import code_evidence
    from nicheverse.annotate.annotate import propose_label, refute_label

    def mock(prompt, **kw):
        if "PROPOSED LABEL UNDER CHALLENGE" in prompt:
            # refuter prompt must NOT mention confidence at all (blind to the labeler's confidence)
            assert "confidence" not in prompt.lower()
            return '{"verdict": "revise", "alternative_label": "Tumor cell", "discriminating_markers": ["G0"], "reason": "r"}'
        return json.dumps(
            {
                "candidates": [{"label": "Tumor cell", "supporting_markers": ["G0", "G1"], "confidence": 0.9}],
                "label": "Tumor cell", "compartment": "epithelial", "confidence": 0.9,
                "rationale": "x", "key_markers": ["G0", "G1"], "citations": ["PMID:1"],
            }
        )

    monkeypatch.setattr(A, "call_llm", mock)
    a = _adata()
    ev = code_evidence(a, "cell_codebook_idx")
    cfg = AnnotationConfig(context=_ctx(), n_candidates=3)
    prop = propose_label("0", ev["0"], cfg, kind="cell")
    assert prop["label"] == "Tumor cell" and prop["candidates"]
    ref = refute_label("0", ev["0"], prop, cfg, kind="cell")
    assert ref["verdict"] == "revise" and ref["alternative_label"] == "Tumor cell"


# ---------------------------------------------------------------------------
# 7. niche path runs through the same function
# ---------------------------------------------------------------------------

def test_niche_kind_runs(monkeypatch):
    a = _adata()
    a.obs["neighborhood_codebook_idx"] = np.array(["0"] * 45 + ["1"] * 45)
    a.obs["celltype_annot"] = np.array(
        ["Tumor cell"] * 30 + ["T cell"] * 15 + ["Macrophage"] * 30 + ["T cell"] * 15
    )

    def mock(prompt, **kw):
        if "revisions" in prompt:
            return '{"revisions": {}}'
        if "PROPOSED LABEL UNDER CHALLENGE" in prompt:
            return '{"verdict": "keep", "alternative_label": "", "discriminating_markers": [], "reason": "ok"}'
        return json.dumps(
            {
                "candidates": [{"label": "tumor niche", "supporting_markers": ["G0"], "confidence": 0.8}],
                "label": "tumor niche", "dominant_types": ["Tumor cell"], "confidence": 0.8,
                "rationale": "composition", "key_markers": ["G0"], "citations": ["PMID:1 Doe J 2024"],
            }
        )

    monkeypatch.setattr(A, "call_llm", mock)
    res = annotate_codebook(
        a, "neighborhood_codebook_idx",
        config=AnnotationConfig(context=_ctx(), refuter=True), kind="niche",
        celltype_col="celltype_annot", resolver=_resolver,
    )
    assert res.kind == "niche"
    assert set(res.labels_df.index) == {"0", "1"}
    assert (res.labels_df["final_label"] == "tumor niche").all()


def test_niche_requires_celltype_col(monkeypatch):
    monkeypatch.setattr(A, "call_llm", lambda p, **k: "{}")
    with pytest.raises(ValueError, match="celltype_col"):
        annotate_codebook(_adata(), "cell_codebook_idx", config=AnnotationConfig(), kind="niche")


# ---------------------------------------------------------------------------
# 8. reconcile flags near-identical evidence with divergent labels
# ---------------------------------------------------------------------------

def test_reconcile_flags_near_identical_divergent(monkeypatch):
    # Codes 0 and 1 share the SAME marker program (near-identical z-profiles); code 2 differs so the
    # z-scores are non-degenerate. The labeler gives codes 0 and 1 divergent labels -> reconcile must
    # flag the near-identical-but-divergent pair.
    rng = np.random.default_rng(1)
    x = rng.poisson(0.5, size=(90, 12)).astype("float32")
    x[:30, 0:3] += 8      # code 0 markers G0 G1 G2
    x[30:60, 0:3] += 8    # code 1 same program as code 0
    x[60:, 8:11] += 8     # code 2 markers G8 G9 G10 (makes z non-zero)
    a = ad.AnnData(X=sp.csr_matrix(x))
    a.var_names = [f"G{i}" for i in range(12)]
    a.obs["cell_codebook_idx"] = np.array(["0"] * 30 + ["1"] * 30 + ["2"] * 30)

    def mock(prompt, **kw):
        if "revisions" in prompt:
            return '{"revisions": {}}'
        if "PROPOSED LABEL UNDER CHALLENGE" in prompt:
            return '{"verdict": "keep", "alternative_label": "", "discriminating_markers": [], "reason": "ok"}'
        code = _which_code(prompt)
        lab = {"0": "Tumor cell A", "1": "Tumor cell B", "2": "Macrophage"}[code]
        mk = ["G8", "G9", "G10"] if code == "2" else ["G0", "G1", "G2"]
        return json.dumps(
            {
                "label": lab, "compartment": "epithelial", "confidence": 0.9, "rationale": "x",
                "key_markers": mk, "citations": ["PMID:1 Doe J 2024"],
            }
        )

    monkeypatch.setattr(A, "call_llm", mock)
    res = annotate_codebook(
        a, "cell_codebook_idx", config=AnnotationConfig(refuter=True), resolver=_resolver,
    )
    flags_all = " ".join(res.labels_df["flags"].tolist())
    assert "near-identical expression profile" in flags_all


# ---------------------------------------------------------------------------
# 9. CLI context-template subcommand writes valid YAML
# ---------------------------------------------------------------------------

def test_cli_context_template(tmp_path):
    import yaml

    out = tmp_path / "ctx.yaml"
    p = subprocess.run(
        [sys.executable, "-m", "nicheverse.annotate", "context-template", str(out)],
        capture_output=True, text=True, check=False,
    )
    assert p.returncode == 0, p.stderr
    assert out.exists()
    loaded = yaml.safe_load(out.read_text())
    assert isinstance(loaded, dict)
    # round-trips into a ProjectContext
    ctx = ProjectContext.from_dict(loaded)
    assert ctx.platform  # template ships a platform
