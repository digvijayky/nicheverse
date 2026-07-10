"""Tests for the deterministic annotation verification gates (annotate/verify.py).

Pure / offline: synthetic evidence dicts, a small ProjectContext, and an INJECTED
dict-backed fake resolver so no test ever touches the network. No GPU.
"""

from __future__ import annotations

from nicheverse.annotate.context import CellTypePrior, NichePrior, ProjectContext
from nicheverse.annotate.verify import (
    apply_lab_rules,
    check_citations,
    gate,
    marker_presence,
    validate_vocabulary,
)


# -- fixtures ---------------------------------------------------------------


def _context():
    return ProjectContext(
        species="human",
        disease="clear cell renal cell carcinoma",
        tissue="kidney and brain metastasis",
        platform="xenium",
        sites=["BrM", "Primary", "Metastasis"],
        site_col="site_class",
        patient_col="mrn",
        expected_cell_types=[
            CellTypePrior(name="ccRCC tumor", markers=["CA9", "NDUFA4L2", "VHL", "PAX8"]),
            CellTypePrior(name="T cell", markers=["CD3D", "CD3E", "CD8A", "IL7R"]),
            CellTypePrior(name="Macrophage", markers=["CD68", "CD163", "LYZ", "C1QA"]),
            CellTypePrior(name="Endothelial", markers=["PECAM1", "VWF", "CLDN5"]),
        ],
        expected_niches=[
            NichePrior(name="tumor core", expected_cell_types=["ccRCC tumor"]),
            NichePrior(name="tumor-immune boundary", expected_cell_types=["ccRCC tumor", "T cell"]),
        ],
    )


def _tumor_evidence(site="Primary"):
    # strong tumor markers, weak/absent everything else
    return {
        "code": "0",
        "n_cells": 5000,
        "frac": 0.2,
        "top_markers": [("CA9", 4.2), ("NDUFA4L2", 3.8), ("VHL", 2.1), ("PAX8", 1.9), ("GAPDH", 0.3)],
        "top_degs": [("CA9", 3.5, 1e-30), ("NDUFA4L2", 3.0, 1e-25), ("SLC17A7", -1.2, 1e-3)],
        "dist_site_class": {site: 0.82, "BrM": 0.1, "Metastasis": 0.08},
    }


# a fake literature record store; keys are ids the "real" resolver knows about
_FAKE_DB = {
    "34290408": {  # real PMID
        "pmid": "34290408",
        "title": "Single-cell atlas of ccRCC: CA9 and NDUFA4L2 mark tumor cells",
        "journal": "Cell",
        "year": "2021",
        "first_author": "Krishna R",
    },
    "10.1038/s41586-021-00000-0": {  # real DOI
        "title": "PAX8 in renal tumor identity",
        "journal": "Nature",
        "year": "2021",
    },
}


def _fake_resolver(identifier):
    """Dict-backed resolver: returns a record for known ids, None otherwise. Never networks."""
    return _FAKE_DB.get(str(identifier))


# -- marker_presence --------------------------------------------------------


def test_marker_presence_flags_invented_gene_and_computes_precision():
    ev = _tumor_evidence()
    res = marker_presence(["CA9", "NDUFA4L2", "MADEUPGENE1"], ev)
    assert set(res["present"]) == {"CA9", "NDUFA4L2"}
    assert res["absent"] == ["MADEUPGENE1"]  # invented gene not in evidence
    assert res["n_cited"] == 3
    assert abs(res["precision"] - 2 / 3) < 1e-9


def test_marker_presence_case_insensitive_and_deg_support():
    ev = _tumor_evidence()
    # lowercase citation still matches; a positive-DEG-only gene counts as supported
    res = marker_presence(["ca9", "vhl"], ev, z_thresh=2.0)
    # CA9 z=4.2 >= 2.0 supported; VHL z=2.1 >= 2.0 supported
    assert set(res["present"]) == {"ca9", "vhl"}
    assert res["absent"] == []


def test_marker_presence_negative_deg_is_not_support():
    ev = _tumor_evidence()
    # SLC17A7 is only a NEGATIVE DEG and not a top marker -> absent
    res = marker_presence(["SLC17A7"], ev)
    assert res["absent"] == ["SLC17A7"]
    assert res["precision"] == 0.0


def test_marker_presence_empty_is_precision_one():
    res = marker_presence([], _tumor_evidence())
    assert res["precision"] == 1.0 and res["n_cited"] == 0


# -- check_citations --------------------------------------------------------


def test_check_citations_real_and_fake_ids_offline():
    citations = [
        "PMID:34290408 Krishna Cell 2021",   # real pmid in fake DB
        "PMID:99999999 fabricated",           # parseable but not in DB -> resolved False
        "doi:10.1038/s41586-021-00000-0",     # real doi in fake DB
        "just some free text with no id",     # unparsed
    ]
    rows = check_citations(citations, resolver=_fake_resolver, markers=["CA9", "PAX8"])
    by_kind = {r["citation"]: r for r in rows}

    real = by_kind["PMID:34290408 Krishna Cell 2021"]
    assert real["kind"] == "pmid" and real["id"] == "34290408"
    assert real["resolved"] is True
    assert real["supports"] is True  # title mentions CA9

    fake = by_kind["PMID:99999999 fabricated"]
    assert fake["kind"] == "pmid" and fake["id"] == "99999999"
    assert fake["resolved"] is False  # injected resolver found nothing => fabricated
    assert fake["supports"] is None

    doi = by_kind["doi:10.1038/s41586-021-00000-0"]
    assert doi["kind"] == "doi" and doi["id"].startswith("10.1038/")
    assert doi["resolved"] is True and doi["supports"] is True  # abstract/title mentions PAX8

    unparsed = by_kind["just some free text with no id"]
    assert unparsed["kind"] == "unparsed" and unparsed["id"] is None
    assert unparsed["resolved"] is None


def test_check_citations_no_marker_leaves_supports_none():
    rows = check_citations(["PMID:34290408"], resolver=_fake_resolver, markers=None)
    assert rows[0]["resolved"] is True
    assert rows[0]["supports"] is None  # no markers to check against


def test_check_citations_default_resolver_no_network_no_crash(monkeypatch):
    # with no injected resolver and no network, parseable ids stay resolved=None, never crash
    import nicheverse.annotate.verify as V

    monkeypatch.setattr(V, "_default_resolver", lambda ident: None)
    rows = check_citations(["PMID:12345678"], resolver=None, markers=["CA9"])
    assert rows[0]["id"] == "12345678"
    assert rows[0]["resolved"] is None  # unknown, not accused of being fabricated


# -- apply_lab_rules --------------------------------------------------------


def test_lab_rules_site_restricted_microglia_on_primary_kidney():
    ev = _tumor_evidence(site="Primary")  # dominant site is a kidney primary
    res = apply_lab_rules({"label": "Microglia"}, ev, _context())
    assert res["violations"], "microglia in a primary-kidney-dominant code must violate"
    assert res["adjusted_label"] is not None
    assert "primary" in " ".join(res["violations"]).lower()


def test_lab_rules_site_restricted_ok_in_brain():
    ev = _tumor_evidence(site="BrM")
    ev["dist_site_class"] = {"BrM": 0.95, "Metastasis": 0.05}  # permissive site only
    res = apply_lab_rules({"label": "Microglia"}, ev, _context())
    assert res["violations"] == []  # brain is permissive for microglia
    assert res["adjusted_label"] is None


def test_lab_rules_valid_composite_passes():
    # both components supported: tumor markers AND macrophage markers present
    ev = {
        "code": "3",
        "n_cells": 800,
        "frac": 0.03,
        "top_markers": [("CA9", 3.0), ("NDUFA4L2", 2.5), ("CD68", 2.2), ("C1QA", 1.8)],
        "top_degs": [("CA9", 2.0, 1e-9), ("CD68", 1.9, 1e-8)],
        "dist_site_class": {"BrM": 0.6, "Primary": 0.4},
    }
    res = apply_lab_rules({"label": "ccRCC tumor/Macrophage"}, ev, _context())
    assert res["violations"] == []  # both components have marker support
    assert res["adjusted_label"] is None


def test_lab_rules_unsupported_composite_collapses_to_dominant():
    # only tumor markers present; T cell markers absent -> collapse to the tumor component
    ev = _tumor_evidence(site="BrM")
    res = apply_lab_rules({"label": "ccRCC tumor/T cell"}, ev, _context())
    assert res["violations"], "an unsupported composite must violate"
    assert res["adjusted_label"] == "ccRCC tumor"  # dominant (supported) component


# -- validate_vocabulary ----------------------------------------------------


def test_validate_vocabulary_matches_expected():
    res = validate_vocabulary("ccRCC tumor", _context(), kind="cell")
    assert res["in_vocab"] is True
    assert res["closest"] == "ccRCC tumor"


def test_validate_vocabulary_fuzzy_closest():
    res = validate_vocabulary("T-cell", _context(), kind="cell")
    # not an exact vocab entry ("T cell"), but closest should surface it
    assert res["closest"] == "T cell"


def test_validate_vocabulary_novel_out_of_vocab():
    res = validate_vocabulary("Plasmacytoid dendritic cell", _context(), kind="cell", allow_novel=True)
    assert res["in_vocab"] is False  # novel escape hatch, not an error


def test_validate_vocabulary_niche_kind():
    res = validate_vocabulary("tumor core", _context(), kind="niche")
    assert res["in_vocab"] is True and res["kind"] == "niche"


# -- gate -------------------------------------------------------------------


def _good_label():
    return {
        "label": "ccRCC tumor",
        "compartment": "epithelial",
        "confidence": 0.9,
        "rationale": "CA9 and NDUFA4L2 strongly enriched",
        "key_markers": ["CA9", "NDUFA4L2", "VHL"],
        "citations": ["PMID:34290408 Krishna Cell 2021"],
    }


def test_gate_passes_clean_label():
    res = gate(_good_label(), _tumor_evidence(site="BrM"), _context(), resolver=_fake_resolver)
    assert res["passed"] is True
    assert res["confidence_penalty"] == 0.0
    assert res["flags"] == []
    assert res["marker"]["precision"] == 1.0
    assert res["vocab"]["in_vocab"] is True


def test_gate_fails_on_low_marker_precision():
    ld = _good_label()
    ld["key_markers"] = ["CA9", "MADEUP1", "MADEUP2", "MADEUP3"]  # 1/4 present
    res = gate(ld, _tumor_evidence(site="BrM"), _context(), resolver=_fake_resolver, min_marker_precision=0.5)
    assert res["passed"] is False
    assert res["confidence_penalty"] > 0
    assert any("marker precision" in f for f in res["flags"])


def test_gate_fails_on_fabricated_citation():
    ld = _good_label()
    ld["citations"] = ["PMID:99999999 fabricated ref"]  # not in the fake DB
    res = gate(ld, _tumor_evidence(site="BrM"), _context(), resolver=_fake_resolver)
    assert res["passed"] is False
    assert any("fabricated" in f or "unresolved" in f for f in res["flags"])


def test_gate_fails_on_site_rule_violation_and_adjusts():
    ld = {
        "label": "Microglia",
        "key_markers": ["CA9"],  # CA9 present so marker precision is fine
        "citations": ["PMID:34290408"],
    }
    res = gate(ld, _tumor_evidence(site="Primary"), _context(), resolver=_fake_resolver)
    assert res["passed"] is False
    assert res["rules"]["violations"]
    assert res["adjusted_label"] is not None  # a general label was proposed


def test_gate_offline_default_resolver_does_not_fail_citation(monkeypatch):
    # unknown (offline) citation resolution must NOT fail the gate
    import nicheverse.annotate.verify as V

    monkeypatch.setattr(V, "_default_resolver", lambda ident: None)
    ld = _good_label()  # good markers, valid vocab, brain site
    res = gate(ld, _tumor_evidence(site="BrM"), _context(), resolver=None)
    # resolved is None (unknown), so no fabrication flag and the gate passes
    assert all(c["resolved"] is None for c in res["citations"])
    assert res["passed"] is True


def test_gate_penalty_caps_at_one():
    # trip all three categories at once: bad markers + fabricated cite + site violation
    ld = {
        "label": "Microglia",
        "key_markers": ["MADEUP1", "MADEUP2"],
        "citations": ["PMID:99999999"],
    }
    res = gate(ld, _tumor_evidence(site="Primary"), _context(), resolver=_fake_resolver)
    assert res["passed"] is False
    assert res["confidence_penalty"] <= 1.0
    assert len(res["flags"]) >= 3
