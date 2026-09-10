"""Shared gene vocabulary: ortholog mapping, control filtering, per panel masks."""

from __future__ import annotations

import numpy as np
import pytest

from nicheverse.data.vocabulary import (
    GeneVocabulary,
    is_control_probe,
    load_mgi_orthologs,
    normalize_symbol,
)

MGI = "\t".join(["DB Class Key", "Common Organism Name", "NCBI Taxon ID", "Symbol"]) + "\n"
ROWS = [
    ("1", "mouse, laboratory", "10090", "Cd8a"),
    ("1", "human", "9606", "CD8A"),
    ("2", "mouse, laboratory", "10090", "Trp53"),
    ("2", "human", "9606", "TP53"),
    # a many-to-one class must be dropped, never collapsed onto one human token
    ("3", "mouse, laboratory", "10090", "Gm1"),
    ("3", "mouse, laboratory", "10090", "Gm2"),
    ("3", "human", "9606", "GMX"),
    # a mouse-only class has no human partner
    ("4", "mouse, laboratory", "10090", "Xist"),
]


@pytest.fixture
def mgi_file(tmp_path):
    p = tmp_path / "hom.rpt"
    p.write_text(MGI + "".join("\t".join(r) + "\n" for r in ROWS))
    return p


def test_normalize_symbol():
    assert normalize_symbol(" cd8a ") == "CD8A"
    assert normalize_symbol("ENSG00000141510.5") == "ENSG00000141510"
    assert normalize_symbol("HLA-DRA") == "HLA-DRA"


def test_control_probe_detection():
    for bad in ("BLANK_0001", "NegPrb1", "NegControlCodeword_0500", "SystemControl3", "UnassignedCodeword_1"):
        assert is_control_probe(bad), bad
    for good in ("CD8A", "TP53", "PTPRC", "COL1A1"):
        assert not is_control_probe(good), good


def test_load_mgi_one_to_one_only(mgi_file):
    m = load_mgi_orthologs(mgi_file)
    assert m == {"CD8A": "CD8A", "TRP53": "TP53"}
    assert "GM1" not in m and "XIST" not in m


def test_build_union_and_mouse_mapping(mgi_file):
    m = load_mgi_orthologs(mgi_file)
    panels = {
        "human_panel": ["CD8A", "PTPRC", "BLANK_0001"],
        "mouse_panel": ["Cd8a", "Trp53", "Xist", "NegPrb1"],
    }
    species = {"human_panel": "Human", "mouse_panel": "Mouse"}
    v = GeneVocabulary.build(panels, species, m)
    # Cd8a merges onto CD8A; Trp53 becomes TP53; Xist has no ortholog and is kept upper cased.
    assert set(v.symbols) == {"CD8A", "PTPRC", "TP53", "XIST"}
    assert list(v.symbols) == sorted(v.symbols)  # deterministic order


def test_map_panel_masks_and_duplicates(mgi_file):
    m = load_mgi_orthologs(mgi_file)
    v = GeneVocabulary.build(
        {"h": ["CD8A", "PTPRC"], "m": ["Cd8a", "Trp53", "Xist"]},
        {"h": "Human", "m": "Mouse"},
        m,
    )
    col, measured = v.map_panel(["Cd8a", "Trp53", "NegPrb1", "NOT_IN_VOCAB"], "Mouse")
    assert col[0] == v.index["CD8A"] and col[1] == v.index["TP53"]
    assert col[2] == -1 and col[3] == -1
    assert measured.tolist() == sorted([v.index["CD8A"], v.index["TP53"]])
    # duplicated symbols map to the same token, and the mask keeps it once
    col2, measured2 = v.map_panel(["CD8A", "CD8A"], "Human")
    assert col2.tolist() == [v.index["CD8A"], v.index["CD8A"]]
    assert measured2.tolist() == [v.index["CD8A"]]


def test_roundtrip(tmp_path, mgi_file):
    m = load_mgi_orthologs(mgi_file)
    v = GeneVocabulary.build({"h": ["CD8A", "PTPRC"]}, {"h": "Human"}, m)
    p = v.save(tmp_path / "vocab.json")
    w = GeneVocabulary.load(p)
    assert w.symbols == v.symbols and w.ortholog_map == v.ortholog_map


def test_unique_symbols_enforced():
    with pytest.raises(ValueError):
        GeneVocabulary(["A", "A"])


def test_measured_is_int32_sorted(mgi_file):
    v = GeneVocabulary.build({"h": [f"G{i}" for i in range(20)]}, {"h": "Human"})
    _, measured = v.map_panel([f"G{i}" for i in (5, 1, 9)], "Human")
    assert measured.dtype == np.int32
    assert measured.tolist() == sorted(measured.tolist())
