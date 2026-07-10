"""Tests for the deterministic annotation evaluation + provenance layer.

Covers the GPTCelltype-style 1/0.5/0 label match, marker precision/recall, confidence
calibration, the stratified scorecard summary (SpatialBench-style), and a reloadable
provenance manifest. No network, no GPU, no scanpy.
"""

from __future__ import annotations

import json

import numpy as np

from nicheverse.annotate.evaluate import (
    calibration,
    score_code,
    scorecard_table,
    summarize,
    write_provenance_manifest,
)

# lineage_map approximates the Cell Ontology tiers: label -> (broad lineage, depth).
# Larger depth = more specific. T cell / CD8 T cell share the immune lineage; CD8 is
# deeper (more specific) than the generic "T cell".
LINEAGE_MAP = {
    "CD8 T cell": ("immune", 3),
    "T cell": ("immune", 2),
    "Macrophage": ("immune", 3),
    "ccRCC tumor": ("epithelial", 3),
    "Endothelial": ("vascular", 2),
}


def _evidence(genes_z, degs=None):
    ev = {"code": "0", "top_markers": genes_z}
    if degs is not None:
        ev["top_degs"] = degs
    return ev


def test_label_match_exact():
    ev = _evidence([("CD8A", 3.0), ("CD3D", 2.0)])
    prop = {"code": "0", "label": "CD8 T cell", "key_markers": ["CD8A", "CD3D"], "confidence": 0.9}
    rec = score_code(prop, "CD8 T cell", ev, lineage_map=LINEAGE_MAP)
    assert rec["label_match"] == 1.0
    # case / separator insensitivity
    rec2 = score_code(prop, "cd8_t_cell", ev, lineage_map=LINEAGE_MAP)
    assert rec2["label_match"] == 1.0


def test_label_match_same_lineage_partial():
    ev = _evidence([("CD8A", 3.0), ("CD3D", 2.0)])
    prop = {"code": "0", "label": "CD8 T cell", "key_markers": ["CD8A"], "confidence": 0.7}
    # reference is a different immune type -> same broad lineage -> 0.5
    rec = score_code(prop, "Macrophage", ev, lineage_map=LINEAGE_MAP)
    assert rec["label_match"] == 0.5


def test_label_match_mismatch():
    ev = _evidence([("CD8A", 3.0)])
    prop = {"code": "0", "label": "CD8 T cell", "key_markers": ["CD8A"], "confidence": 0.5}
    rec = score_code(prop, "Endothelial", ev, lineage_map=LINEAGE_MAP)
    assert rec["label_match"] == 0.0  # immune vs vascular


def test_label_match_none_without_reference():
    ev = _evidence([("CD8A", 3.0)])
    prop = {"code": "0", "label": "CD8 T cell"}
    assert score_code(prop, "", ev, lineage_map=LINEAGE_MAP)["label_match"] is None
    assert score_code(prop, None, ev, lineage_map=LINEAGE_MAP)["label_match"] is None


def test_granularity_over_and_under_call():
    ev = _evidence([("CD8A", 3.0)])
    prop_specific = {"code": "0", "label": "CD8 T cell", "key_markers": ["CD8A"]}
    # proposed deeper than the reference -> over-call
    over = score_code(prop_specific, "T cell", ev, lineage_map=LINEAGE_MAP)
    assert over["granularity"] == "over_call"
    prop_broad = {"code": "0", "label": "T cell", "key_markers": ["CD8A"]}
    under = score_code(prop_broad, "CD8 T cell", ev, lineage_map=LINEAGE_MAP)
    assert under["granularity"] == "under_call"
    # no lineage_map -> granularity not computable
    assert score_code(prop_specific, "T cell", ev)["granularity"] is None


def test_marker_precision_drops_on_non_enriched_marker():
    # CD8A/CD3D are enriched (z>=1); ALB is present but has z below threshold -> not enriched.
    ev = _evidence([("CD8A", 3.0), ("CD3D", 2.0), ("ALB", 0.1)])
    good = {"code": "0", "label": "CD8 T cell", "key_markers": ["CD8A", "CD3D"]}
    bad = {"code": "0", "label": "CD8 T cell", "key_markers": ["CD8A", "ALB"]}
    r_good = score_code(good, "CD8 T cell", ev, lineage_map=LINEAGE_MAP, z_thresh=1.0)
    r_bad = score_code(bad, "CD8 T cell", ev, lineage_map=LINEAGE_MAP, z_thresh=1.0)
    assert r_good["marker_precision"] == 1.0
    assert r_bad["marker_precision"] == 0.5  # 1 of 2 enriched
    assert r_bad["n_absent_markers"] == 1
    assert "ALB" in r_bad["absent_markers"]


def test_marker_recall_uses_top_k():
    ev = _evidence([("CD8A", 3.0), ("CD3D", 2.0), ("CD3E", 1.5)])
    prop = {"code": "0", "label": "CD8 T cell", "key_markers": ["CD8A"]}
    rec = score_code(prop, "CD8 T cell", ev, lineage_map=LINEAGE_MAP, k=3)
    # 1 of the code's top-3 markers recalled by the proposal
    assert abs(rec["marker_recall"] - 1.0 / 3.0) < 1e-9


def test_marker_enriched_via_deg():
    # HAVCR1 is not a top marker but is a positive DEG -> counts as enriched.
    ev = _evidence([("CD8A", 3.0)], degs=[("HAVCR1", 2.5, 1e-8)])
    prop = {"code": "0", "label": "ccRCC tumor", "key_markers": ["HAVCR1"]}
    rec = score_code(prop, "ccRCC tumor", ev, lineage_map=LINEAGE_MAP)
    assert rec["marker_precision"] == 1.0
    assert rec["n_absent_markers"] == 0


def test_rule_compliance_and_hallucination_from_gate():
    ev = _evidence([("CD8A", 3.0)])
    prop = {"code": "0", "label": "CD8 T cell", "key_markers": ["CD8A"]}
    gate_fail = {
        "rule_violations": ["site-restricted label in non-permissive site"],
        "n_absent_markers": 2,
        "n_unresolved_citations": 1,
        "marker_precision": 0.5,
    }
    rec = score_code(prop, "CD8 T cell", ev, lineage_map=LINEAGE_MAP, gate_result=gate_fail)
    assert rec["rule_compliance"] == "fail"
    assert rec["n_absent_markers"] == 2  # gate overrides evidence recompute
    assert rec["n_unresolved_citations"] == 1
    assert rec["gate_marker_precision"] == 0.5
    # no gate -> pass, unresolved citations unknown
    rec2 = score_code(prop, "CD8 T cell", ev, lineage_map=LINEAGE_MAP)
    assert rec2["rule_compliance"] == "pass"
    assert rec2["n_unresolved_citations"] is None


def test_gate_rules_pass_flag():
    ev = _evidence([("CD8A", 3.0)])
    prop = {"code": "0", "label": "CD8 T cell", "key_markers": ["CD8A"]}
    rec = score_code(prop, "CD8 T cell", ev, gate_result={"rules_pass": True})
    assert rec["rule_compliance"] == "pass"
    rec2 = score_code(prop, "CD8 T cell", ev, gate_result={"rules_pass": False})
    assert rec2["rule_compliance"] == "fail"


def _cards_confident_correct():
    """High confidence exactly where the label is correct -> positive spearman."""
    ev = _evidence([("CD8A", 3.0)])
    cards = []
    for i in range(6):
        correct = i < 3
        prop = {
            "code": str(i),
            "label": "CD8 T cell" if correct else "Endothelial",
            "key_markers": ["CD8A"],
            "confidence": 0.9 if correct else 0.2,
        }
        ref = "CD8 T cell"  # correct ones match, wrong ones (vascular) mismatch
        cards.append(score_code(prop, ref, ev, lineage_map=LINEAGE_MAP))
    return cards


def test_calibration_positive_spearman_when_confidence_tracks_correctness():
    cards = _cards_confident_correct()
    cal = calibration(cards)
    assert cal["spearman"] is not None
    assert cal["spearman"] > 0.5
    assert cal["n"] == 6
    assert len(cal["bins"]) >= 1


def test_calibration_random_confidence_near_zero_or_none():
    ev = _evidence([("CD8A", 3.0)])
    rng = np.random.default_rng(0)
    cards = []
    for i in range(20):
        correct = i % 2 == 0
        prop = {
            "code": str(i),
            "label": "CD8 T cell" if correct else "Endothelial",
            "key_markers": ["CD8A"],
            "confidence": float(rng.random()),  # random, decoupled from correctness
        }
        cards.append(score_code(prop, "CD8 T cell", ev, lineage_map=LINEAGE_MAP))
    cal = calibration(cards)
    assert cal["spearman"] is None or abs(cal["spearman"]) < 0.5


def test_calibration_handles_missing_confidence():
    ev = _evidence([("CD8A", 3.0)])
    cards = [
        score_code({"code": "0", "label": "CD8 T cell", "key_markers": ["CD8A"]}, "CD8 T cell", ev, lineage_map=LINEAGE_MAP),
        score_code({"code": "1", "label": "CD8 T cell", "key_markers": ["CD8A"], "confidence": 0.8}, "CD8 T cell", ev, lineage_map=LINEAGE_MAP),
    ]
    cal = calibration(cards)
    assert cal["n_total"] == 2
    assert cal["n"] == 1  # only the one with a confidence is usable


def test_scorecard_table_one_row_per_code():
    ev = _evidence([("CD8A", 3.0)])
    cards = [
        score_code({"code": "0", "label": "CD8 T cell", "key_markers": ["CD8A"], "confidence": 0.9}, "CD8 T cell", ev, lineage_map=LINEAGE_MAP),
        score_code({"code": "1", "label": "Macrophage", "key_markers": ["CD8A"], "confidence": 0.4}, "Endothelial", ev, lineage_map=LINEAGE_MAP),
    ]
    df = scorecard_table(cards)
    assert df.shape[0] == 2
    assert list(df["code"]) == ["0", "1"]
    for col in ("label_match", "marker_precision", "rule_compliance"):
        assert col in df.columns


def test_scorecard_table_empty():
    df = scorecard_table([])
    assert df.shape[0] == 0
    assert "label_match" in df.columns


def test_summarize_aggregate_and_stratified_by_platform():
    ev = _evidence([("CD8A", 3.0)])
    cards = []
    # xenium: both correct; cosmx: both wrong -> stratified means must differ
    for i, (plat, correct) in enumerate(
        [("xenium", True), ("xenium", True), ("cosmx", False), ("cosmx", False)]
    ):
        prop = {
            "code": str(i),
            "label": "CD8 T cell" if correct else "Endothelial",
            "key_markers": ["CD8A"],
            "confidence": 0.8,
            "platform": plat,
        }
        cards.append(score_code(prop, "CD8 T cell", ev, lineage_map=LINEAGE_MAP))
    summ = summarize(cards)
    assert summ["n"] == 4
    assert summ["mean_label_match"] == 0.5  # 2 correct, 2 mismatch
    assert "platform" in summ["by"]
    by = summ["by"]["platform"]
    assert by["xenium"]["mean_label_match"] == 1.0
    assert by["cosmx"]["mean_label_match"] == 0.0
    # headline hides the weak stratum; the breakdown surfaces it
    assert by["xenium"]["mean_label_match"] != by["cosmx"]["mean_label_match"]


def test_summarize_stratified_by_compartment_and_totals():
    ev = _evidence([("CD8A", 3.0), ("ALB", 0.0)])
    gate = {"rule_violations": ["x"], "n_absent_markers": 1, "n_unresolved_citations": 2}
    prop = {"code": "0", "label": "CD8 T cell", "compartment": "immune", "key_markers": ["CD8A", "ALB"]}
    rec = score_code(prop, "CD8 T cell", ev, lineage_map=LINEAGE_MAP, gate_result=gate)
    summ = summarize([rec])
    assert "compartment" in summ["by"]
    assert "immune" in summ["by"]["compartment"]
    assert summ["rule_fail_rate"] == 1.0
    assert summ["total_hallucinated_markers"] == 1
    assert summ["total_hallucinated_citations"] == 2


def test_write_provenance_manifest_reloads(tmp_path):
    ev = _evidence([("CD8A", 3.0)])
    rec = score_code(
        {"code": "0", "label": "CD8 T cell", "key_markers": ["CD8A"], "confidence": 0.9,
         "citations": ["PMID:34290408 Krishna Cell 2021"]},
        "CD8 T cell", ev, lineage_map=LINEAGE_MAP,
    )
    rec["evidence"] = ev
    rec["prompt"] = "Code 0: top markers CD8A(3.0). Return JSON..."
    rec["raw_output"] = '{"label": "CD8 T cell", "confidence": 0.9}'
    rec["gate_result"] = {"rules_pass": True}
    rec["citations"] = ["PMID:34290408 Krishna Cell 2021"]
    rec["final_label"] = "CD8 T cell"

    run_meta = {
        "model": "claude-opus",
        "seed": 0,
        "temperature": 0.0,
        "project_context": "RCC BrM Xenium 366-gene",
        "rule_version": "v1",
        "timestamp": "2026-07-09T00:00:00Z",  # passed in, not datetime.now
    }
    path = write_provenance_manifest(str(tmp_path), run_meta, [rec])
    assert path.endswith("provenance_manifest.json")

    with open(path) as fh:
        m = json.load(fh)
    assert m["run_meta"]["timestamp"] == "2026-07-09T00:00:00Z"
    assert m["run_meta"]["model"] == "claude-opus"
    assert m["n_codes"] == 1
    entry = m["codes"][0]
    assert entry["code"] == "0"
    assert entry["evidence_hash"] and len(entry["evidence_hash"]) == 16
    assert entry["prompt_hash"] and len(entry["prompt_hash"]) == 16
    assert entry["final_label"] == "CD8 T cell"
    assert entry["gate_result"] == {"rules_pass": True}
    assert entry["citations"] == ["PMID:34290408 Krishna Cell 2021"]
    assert m["summary"] is not None

    # the scorecard CSV is written next to the manifest and reloads
    import os

    csv_path = os.path.join(str(tmp_path), "provenance_scorecards.csv")
    assert os.path.isfile(csv_path)
    import pandas as pd

    df = pd.read_csv(csv_path)
    assert df.shape[0] == 1
    assert "label_match" in df.columns


def test_write_provenance_manifest_makedirs(tmp_path):
    nested = tmp_path / "a" / "b" / "run1"
    rec = {"code": "0", "label": "T cell", "label_match": 1.0, "marker_precision": 1.0,
           "rule_compliance": "pass", "n_absent_markers": 0}
    path = write_provenance_manifest(str(nested), {"timestamp": "t"}, [rec])
    import os

    assert os.path.isfile(path)


# ---------------------------------------------------------------------------
# label_match: missing lineage_map + synonyms
# ---------------------------------------------------------------------------

def test_label_match_without_lineage_map_degrades_to_exact_vs_mismatch():
    # No lineage_map: exact match still 1.0, but "same broad lineage" is not knowable
    # so a different immune type falls all the way to 0.0 (documented CL approximation).
    ev = _evidence([("CD8A", 3.0)])
    prop = {"code": "0", "label": "CD8 T cell", "key_markers": ["CD8A"]}
    assert score_code(prop, "CD8 T cell", ev)["label_match"] == 1.0
    assert score_code(prop, "Macrophage", ev)["label_match"] == 0.0
    # and granularity is None without a depth map
    assert score_code(prop, "Macrophage", ev)["granularity"] is None


def test_label_match_synonym():
    ev = _evidence([("CD8A", 3.0)])
    # proposal names a synonym of the reference -> full match (1.0)
    prop = {"code": "0", "label": "CTL", "key_markers": ["CD8A"],
            "synonyms": ["CD8 T cell", "cytotoxic T lymphocyte"]}
    rec = score_code(prop, "CD8 T cell", ev, lineage_map=LINEAGE_MAP)
    assert rec["label_match"] == 1.0
    # an unrelated synonym does not manufacture a match
    prop2 = {"code": "0", "label": "CTL", "key_markers": ["CD8A"], "synonyms": ["B cell"]}
    assert score_code(prop2, "Endothelial", ev, lineage_map=LINEAGE_MAP)["label_match"] == 0.0


def test_label_match_none_proposed_label():
    # a missing / empty proposed label against a real reference is a mismatch, not a crash
    ev = _evidence([("CD8A", 3.0)])
    assert score_code({"code": "0"}, "CD8 T cell", ev, lineage_map=LINEAGE_MAP)["label_match"] == 0.0
    assert score_code({"code": "0", "label": None}, "CD8 T cell", ev)["label_match"] == 0.0


# ---------------------------------------------------------------------------
# marker precision / recall degenerate inputs
# ---------------------------------------------------------------------------

def test_marker_precision_none_when_no_markers_cited():
    ev = _evidence([("CD8A", 3.0)])
    prop = {"code": "0", "label": "CD8 T cell"}  # no key_markers
    rec = score_code(prop, "CD8 T cell", ev, lineage_map=LINEAGE_MAP)
    assert rec["marker_precision"] is None
    assert rec["n_proposed_markers"] == 0
    assert rec["n_absent_markers"] == 0


def test_marker_recall_none_when_no_evidence_markers():
    # evidence has neither top_markers nor top_degs -> recall undefined, no crash
    ev = {"code": "0"}
    prop = {"code": "0", "label": "CD8 T cell", "key_markers": ["CD8A"]}
    rec = score_code(prop, "CD8 T cell", ev, lineage_map=LINEAGE_MAP)
    assert rec["marker_recall"] is None
    # every cited marker is absent from an empty enriched set
    assert rec["marker_precision"] == 0.0
    assert rec["n_absent_markers"] == 1


def test_marker_recall_when_fewer_than_k_markers():
    # only 2 markers exist but k=5 -> recall denominator is the 2 that exist, not 5
    ev = _evidence([("CD8A", 3.0), ("CD3D", 2.0)])
    prop = {"code": "0", "label": "CD8 T cell", "key_markers": ["CD8A", "CD3D"]}
    rec = score_code(prop, "CD8 T cell", ev, lineage_map=LINEAGE_MAP, k=5)
    assert rec["marker_recall"] == 1.0


# ---------------------------------------------------------------------------
# calibration edge cases
# ---------------------------------------------------------------------------

def test_calibration_perfect_predictor_high_spearman():
    # confidence rank-orders correctness perfectly: every wrong call (label_match 0.0)
    # sits below every right call (1.0). With a binary label_match the tie structure
    # caps Spearman below 1.0, but it must still be strongly positive. Using the same
    # lineage each time (partial 0.5 vs full 1.0) removes ties and drives it to 1.0.
    ev = _evidence([("CD8A", 3.0)])
    cards = []
    for i, (lab, conf) in enumerate(
        [("Macrophage", 0.1), ("T cell", 0.4), ("CD8 T cell", 0.9)]
    ):
        # Macrophage=same immune lineage (0.5) but not exact; T cell=same lineage (0.5);
        # so make one an outright mismatch to get 3 distinct correctness ranks.
        ref = {"Macrophage": "Endothelial", "T cell": "Macrophage", "CD8 T cell": "CD8 T cell"}[lab]
        prop = {"code": str(i), "label": lab, "key_markers": ["CD8A"], "confidence": conf}
        cards.append(score_code(prop, ref, ev, lineage_map=LINEAGE_MAP))
    # label_match values: 0.0 (immune vs vascular), 0.5 (same immune lineage), 1.0 (exact)
    lms = sorted(c["label_match"] for c in cards)
    assert lms == [0.0, 0.5, 1.0]  # three distinct correctness levels, no ties
    cal = calibration(cards)
    assert cal["spearman"] is not None and cal["spearman"] > 0.99


def test_calibration_constant_confidence_gives_none():
    # no variance in confidence -> spearman is undefined -> None (not NaN, no raise)
    ev = _evidence([("CD8A", 3.0)])
    cards = []
    for i in range(4):
        prop = {"code": str(i), "label": "CD8 T cell", "key_markers": ["CD8A"], "confidence": 0.7}
        cards.append(score_code(prop, "CD8 T cell", ev, lineage_map=LINEAGE_MAP))
    cal = calibration(cards)
    assert cal["spearman"] is None
    assert cal["n"] == 4


def test_calibration_single_bin_and_ties_do_not_raise():
    ev = _evidence([("CD8A", 3.0)])
    cards = [
        score_code({"code": "0", "label": "CD8 T cell", "key_markers": ["CD8A"], "confidence": 0.5},
                   "CD8 T cell", ev, lineage_map=LINEAGE_MAP),
        score_code({"code": "1", "label": "Endothelial", "key_markers": ["CD8A"], "confidence": 0.5},
                   "CD8 T cell", ev, lineage_map=LINEAGE_MAP),
    ]
    cal = calibration(cards, n_bins=1)  # single bin + tied confidences
    assert len(cal["bins"]) == 1
    assert cal["n"] == 2
    # constant confidence -> spearman undefined -> None
    assert cal["spearman"] is None


def test_calibration_empty_and_all_none_label_match():
    assert calibration([])["spearman"] is None
    ev = _evidence([("CD8A", 3.0)])
    # confidence present but reference absent -> label_match None -> dropped from corr
    card = score_code({"code": "0", "label": "CD8 T cell", "confidence": 0.9}, "", ev)
    cal = calibration([card])
    assert cal["n_total"] == 1 and cal["n"] == 0 and cal["spearman"] is None


# ---------------------------------------------------------------------------
# scorecard_table / summarize tolerance
# ---------------------------------------------------------------------------

def test_scorecard_table_tolerates_missing_keys():
    # a bare record missing most scorecard fields still yields one row with all columns
    df = scorecard_table([{"code": "0", "proposed_label": "T cell"}])
    assert df.shape[0] == 1
    for col in ("label_match", "marker_precision", "granularity", "platform", "absent_markers"):
        assert col in df.columns


def test_summarize_without_platform_or_compartment():
    ev = _evidence([("CD8A", 3.0)])
    # records carry no platform/compartment -> "by" is present but empty
    cards = [
        score_code({"code": "0", "label": "CD8 T cell", "key_markers": ["CD8A"]},
                   "CD8 T cell", ev, lineage_map=LINEAGE_MAP),
        score_code({"code": "1", "label": "Endothelial", "key_markers": ["CD8A"]},
                   "CD8 T cell", ev, lineage_map=LINEAGE_MAP),
    ]
    summ = summarize(cards)
    assert summ["n"] == 2
    assert summ["by"] == {}


# ---------------------------------------------------------------------------
# provenance manifest: numpy + non-finite serialization
# ---------------------------------------------------------------------------

def test_manifest_serializes_numpy_and_nonfinite(tmp_path):
    ev = _evidence([("CD8A", 3.0)])
    rec = score_code(
        {"code": "0", "label": "CD8 T cell", "key_markers": ["CD8A"],
         "confidence": np.float64(0.9)},  # numpy scalar confidence
        "CD8 T cell", ev, lineage_map=LINEAGE_MAP,
    )
    rec["evidence"] = ev
    rec["np_array"] = np.array([1, 2, 3])          # numpy array -> JSON list
    rec["np_int"] = np.int64(7)                    # numpy int -> JSON int
    rec["np_bool"] = np.bool_(True)                # numpy bool -> JSON bool
    rec["marker_precision"] = float("nan")         # non-finite -> null
    rec["some_inf"] = float("inf")                 # non-finite -> null
    run_meta = {"timestamp": "t", "seed": np.int64(0), "bad": float("nan")}

    path = write_provenance_manifest(str(tmp_path), run_meta, [rec])

    # strict reload: reject the invalid JSON tokens NaN / Infinity outright
    def _boom(x):
        raise ValueError(f"non-finite JSON constant: {x}")

    with open(path) as fh:
        text = fh.read()
    assert "NaN" not in text and "Infinity" not in text
    m = json.loads(text, parse_constant=_boom)

    sc = m["scorecards"][0]
    assert sc["np_array"] == [1, 2, 3]
    assert sc["np_int"] == 7 and isinstance(sc["np_int"], int)
    assert sc["np_bool"] is True
    assert sc["marker_precision"] is None
    assert sc["some_inf"] is None
    assert m["run_meta"]["seed"] == 0
    assert m["run_meta"]["bad"] is None


def test_manifest_hash_stable_across_numpy_and_native():
    from nicheverse.annotate.evaluate import _hash_obj

    assert _hash_obj({"a": np.int64(5), "b": np.array([1, 2])}) == _hash_obj({"a": 5, "b": [1, 2]})


def test_manifest_writes_csv_for_nonscorecard_records(tmp_path):
    # records with no scorecard fields: manifest.summary is None but the CSV still writes
    import os

    recs = [{"code": "0", "final_label": "T cell", "citations": ["PMID:1"]}]
    path = write_provenance_manifest(str(tmp_path), {"timestamp": "t"}, recs)
    with open(path) as fh:
        m = json.load(fh)
    assert m["summary"] is None
    assert m["scorecards"] == []
    assert os.path.isfile(os.path.join(str(tmp_path), "provenance_scorecards.csv"))
