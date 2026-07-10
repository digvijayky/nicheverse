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


# -- hardening: marker_presence edge cases ----------------------------------


def test_marker_presence_whitespace_and_duplicate_dedup():
    ev = _tumor_evidence()
    # padded, mixed-case duplicates collapse to one cited marker; matching survives padding
    res = marker_presence(["  ca9 ", "CA9", "Ca9", "\tNDUFA4L2\n"], ev)
    assert res["n_cited"] == 2  # CA9 (3 spellings) + NDUFA4L2
    assert set(m.strip().lower() for m in res["present"]) == {"ca9", "ndufa4l2"}
    assert res["precision"] == 1.0


def test_marker_presence_missing_evidence_keys_no_crash():
    # evidence dict with neither top_markers nor top_degs -> everything absent, no crash
    res = marker_presence(["CA9", "CD3D"], {"code": "9", "n_cells": 3})
    assert res["present"] == [] and set(res["absent"]) == {"CA9", "CD3D"}
    assert res["precision"] == 0.0 and res["n_cited"] == 2


def test_marker_presence_none_evidence_precision_one_when_empty():
    # a non-dict / None evidence is treated as empty, never raises
    assert marker_presence([], None)["precision"] == 1.0
    r = marker_presence(["CA9"], None)
    assert r["absent"] == ["CA9"] and r["precision"] == 0.0


def test_marker_presence_malformed_entries_skipped():
    ev = {
        "top_markers": [("CA9", 4.0), ("BAD",), "JUSTASTRING", ("NAN", float("nan")), None],
        "top_degs": [("CD3D", 2.0, 1e-4), ("SHORT",), ("NEG", -3.0, 0.1)],
    }
    res = marker_presence(["CA9", "CD3D", "BAD", "NAN", "NEG"], ev)
    assert set(res["present"]) == {"CA9", "CD3D"}  # only well-formed positive entries support
    assert set(res["absent"]) == {"BAD", "NAN", "NEG"}


# -- hardening: citation parsing --------------------------------------------


def test_parse_short_and_bare_and_url_ids():
    from nicheverse.annotate.verify import _parse_citation

    assert _parse_citation("PMID: 123")[0] == "123"          # short, explicitly labelled
    assert _parse_citation("pmid 34290408") == ("34290408", "pmid")
    assert _parse_citation("34290408") == ("34290408", "pmid")  # bare 8-digit run
    assert _parse_citation("https://doi.org/10.1038/s41586-021-00000-0")[1] == "doi"
    assert _parse_citation("see doi:10.1000/xyz123.") == ("10.1000/xyz123", "doi")  # trailing dot stripped
    # a bare 4-digit year is NOT mistaken for a PMID
    assert _parse_citation("Krishna et al. 2021")[1] == "unparsed"
    # a DOI containing digits is not misread as a PMID (DOI wins)
    assert _parse_citation("10.1038/s41586-021-03569-1")[1] == "doi"


def test_check_citations_offline_default_never_accuses(monkeypatch):
    import nicheverse.annotate.verify as V

    # simulate an offline default resolver that always yields None
    monkeypatch.setattr(V, "_default_resolver", lambda ident: None)
    rows = check_citations(["PMID:12345678", "10.1038/s41586-021-00000-0"], resolver=None)
    assert all(r["resolved"] is None for r in rows)  # unknown, not False


def test_check_citations_never_raises_when_resolver_throws():
    def boom(_ident):
        raise RuntimeError("network down")

    rows = check_citations(["PMID:34290408"], resolver=boom, markers=["CA9"])
    assert rows[0]["resolved"] is False and rows[0]["record"] is None


def test_check_citations_handles_non_string_and_none_entries():
    rows = check_citations([None, 34290408], resolver=_fake_resolver, markers=["CA9"])
    assert rows[0]["kind"] == "unparsed" and rows[0]["resolved"] is None
    assert rows[1]["kind"] == "pmid" and rows[1]["id"] == "34290408" and rows[1]["resolved"] is True


# -- hardening: vocabulary fuzzy guard + allow_novel ------------------------


def test_validate_vocabulary_no_false_closest_for_unrelated_label():
    res = validate_vocabulary("Osteoclast", _context(), kind="cell")
    assert res["in_vocab"] is False
    assert res["closest"] is None  # nothing near "Osteoclast" among the priors


def test_validate_vocabulary_short_fragment_does_not_anchor():
    # a 1-char label must not be surfaced as "closest" to a long allowed name via containment
    res = validate_vocabulary("T", _context(), kind="cell")
    assert res["in_vocab"] is False
    assert res["closest"] is None


def test_validate_vocabulary_allow_novel_false_forces_closest():
    novel = validate_vocabulary("Osteoclast", _context(), kind="cell", allow_novel=True)
    strict = validate_vocabulary("Osteoclast", _context(), kind="cell", allow_novel=False)
    assert novel["closest"] is None            # tolerated novel label, left alone
    assert strict["closest"] in {c.name for c in _context().expected_cell_types}  # coerced


# -- hardening: site-restriction whole-word + dominance ---------------------


def test_lab_rules_site_key_whole_word_no_false_positive():
    # "Neuroendocrine tumor" contains "neuron" as a substring but NOT as a whole word
    ev = _tumor_evidence(site="Primary")
    res = apply_lab_rules({"label": "Neuroendocrine tumor"}, ev, _context())
    assert res["violations"] == []  # must not trip the neuron site-restriction
    assert res["adjusted_label"] is None


def test_lab_rules_brain_dominant_small_primary_tail_not_flagged():
    ev = _tumor_evidence(site="BrM")
    ev["dist_site_class"] = {"BrM": 0.9, "Primary": 0.1}  # tiny non-permissive tail
    res = apply_lab_rules({"label": "Microglia"}, ev, _context())
    assert res["violations"] == []  # 10% primary is plausible leakage, not a violation
    assert res["adjusted_label"] is None


def test_lab_rules_material_nonpermissive_minority_is_flagged():
    ev = _tumor_evidence(site="BrM")
    ev["dist_site_class"] = {"BrM": 0.6, "Primary": 0.4}  # 40% primary is material
    res = apply_lab_rules({"label": "Microglia"}, ev, _context())
    assert res["violations"], "a 40% non-permissive population must trip the rule"
    assert res["adjusted_label"] is not None


def test_lab_rules_site_distribution_unavailable_is_noted_not_violated():
    ev = {"code": "5", "top_markers": [("CA9", 3.0)]}  # no dist_* key, no adata
    res = apply_lab_rules({"label": "Microglia"}, ev, _context())
    assert res["violations"] == []
    assert any("site distribution unavailable" in n for n in res["notes"])


def test_lab_rules_site_distribution_from_adata_fallback():
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace

    obs = pd.DataFrame(
        {
            "cell_codebook_idx": ["7"] * 10,
            "site_class": ["Primary"] * 8 + ["BrM"] * 2,
        }
    )
    adata = SimpleNamespace(obs=obs)
    ev = {"code": "7", "top_markers": [("CA9", 3.0)]}  # no dist_* in evidence
    res = apply_lab_rules({"label": "Microglia"}, ev, _context(), adata=adata, code="7")
    assert res["violations"], "dominant Primary computed from adata must trip the rule"


# -- hardening: gate invariants ---------------------------------------------


def test_gate_invariant_passed_iff_zero_penalty():
    # exhaustively over a few label/evidence combos, passed must equal (penalty == 0)
    ctx = _context()
    cases = [
        (_good_label(), _tumor_evidence(site="BrM")),
        ({"label": "Microglia", "key_markers": ["CA9"], "citations": ["PMID:34290408"]},
         _tumor_evidence(site="Primary")),
        ({"label": "ccRCC tumor", "key_markers": ["MADEUP"], "citations": []},
         _tumor_evidence(site="BrM")),
        ({"label": "ccRCC tumor", "key_markers": [], "citations": []},
         _tumor_evidence(site="BrM")),
    ]
    for ld, ev in cases:
        r = gate(ld, ev, ctx, resolver=_fake_resolver)
        assert r["passed"] is (r["confidence_penalty"] == 0.0)
        assert (r["flags"] == []) is (r["passed"] is True)


def test_gate_empty_markers_and_citations_passes():
    ld = {"label": "ccRCC tumor", "key_markers": [], "citations": []}
    r = gate(ld, _tumor_evidence(site="BrM"), _context(), resolver=_fake_resolver)
    assert r["passed"] is True and r["confidence_penalty"] == 0.0
    assert r["marker"]["precision"] == 1.0  # nothing cited -> not penalized


def test_gate_missing_evidence_dict_does_not_crash():
    ld = _good_label()
    r = gate(ld, {}, _context(), resolver=_fake_resolver)  # empty evidence
    # markers cannot be supported against empty evidence -> marker category fails
    assert r["passed"] is False
    assert any("marker precision" in f for f in r["flags"])


def test_gate_novel_label_marker_failure_still_fails():
    ld = {
        "label": "Some Novel State",  # out of vocab
        "key_markers": ["MADEUP1", "MADEUP2"],
        "citations": [],
    }
    r = gate(ld, _tumor_evidence(site="BrM"), _context(), resolver=_fake_resolver)
    assert r["vocab"]["in_vocab"] is False
    assert r["passed"] is False  # novelty is not an excuse for unsupported markers


def test_gate_penalty_exact_values():
    # one failing category -> 0.34; verify the per-category increment is exact
    ld = {"label": "ccRCC tumor", "key_markers": ["MADEUP"], "citations": []}
    r = gate(ld, _tumor_evidence(site="BrM"), _context(), resolver=_fake_resolver)
    assert r["confidence_penalty"] == 0.34
