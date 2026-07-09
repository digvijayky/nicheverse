"""Tests for the annotate module (evidence + iterative LLM annotation with a mock provider)."""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import nicheverse.annotate.annotate as A
from nicheverse.annotate import (
    AnnotationConfig,
    annotate_codes,
    attach_labels,
    call_llm,
    code_evidence,
)


def _adata():
    rng = np.random.default_rng(0)
    x = rng.poisson(0.5, size=(90, 12)).astype("float32")
    x[:30, 0:3] += 8
    x[30:60, 4:7] += 8
    x[60:, 8:11] += 8
    a = ad.AnnData(X=sp.csr_matrix(x))
    a.var_names = [f"G{i}" for i in range(12)]
    a.obs["cell_codebook_idx"] = np.array(["0"] * 30 + ["1"] * 30 + ["2"] * 30)
    a.obs["site_class"] = np.array(["BrM", "Primary"] * 45)
    return a


def test_code_evidence():
    ev = code_evidence(_adata(), "cell_codebook_idx", extra_cols=("site_class",))
    assert set(ev) == {"0", "1", "2"}
    assert "top_markers" in ev["0"] and "top_degs" in ev["0"] and "dist_site_class" in ev["0"]
    assert {m for m, _ in ev["0"]["top_markers"][:3]} & {"G0", "G1", "G2"}


def test_annotate_codes_mock(monkeypatch):
    def mock(prompt, **kw):
        if "revisions" in prompt:
            return '{"revisions": {"1": "Refined label"}}'
        return json.dumps(
            {
                "label": "Cell type X",
                "compartment": "immune",
                "confidence": 0.8,
                "rationale": "m",
                "key_markers": ["G0"],
                "citations": ["PMID:1"],
            }
        )

    monkeypatch.setattr(A, "call_llm", mock)
    df = annotate_codes(_adata(), "cell_codebook_idx", AnnotationConfig(refine=True, tissue="t"))
    assert list(df.index) == ["0", "1", "2"] and (df["label"] == "Cell type X").all()
    assert df.loc["1", "label_refined"] == "Refined label"
    a = _adata()
    attach_labels(a, "cell_codebook_idx", df)
    assert "celltype_annot" in a.obs


def test_unknown_provider_rejected():
    with pytest.raises(ValueError, match="unknown provider"):
        call_llm("x", provider="nope")


def test_niche_annotation_mock(monkeypatch):
    import json

    from nicheverse.annotate import AnnotationConfig, annotate_niches, niche_evidence

    a = _adata()
    a.obs["neighborhood_codebook_idx"] = np.array(["0"] * 45 + ["1"] * 45)
    a.obs["celltype_annot"] = np.array(
        ["Tumor"] * 30 + ["T cell"] * 15 + ["Fibroblast"] * 30 + ["Macrophage"] * 15
    )
    ev = niche_evidence(a, "neighborhood_codebook_idx", "celltype_annot")
    assert set(ev) == {"0", "1"} and "composition" in ev["0"]
    monkeypatch.setattr(
        A,
        "call_llm",
        lambda prompt, **kw: json.dumps(
            {"label": "niche X", "dominant_types": ["Tumor"], "confidence": 0.7, "rationale": "c"}
        ),
    )
    df = annotate_niches(a, "neighborhood_codebook_idx", "celltype_annot", AnnotationConfig())
    assert list(df.index) == ["0", "1"] and "dominant_types" in df.columns


def test_cluster_codes():
    from nicheverse.annotate import cluster_codes

    df = cluster_codes(_adata(), "cell_codebook_idx", n_clusters=2)
    assert set(df.index) == {"0", "1", "2"} and "cluster" in df.columns
    assert df["cluster"].nunique() >= 1 and df["cluster"].nunique() <= 3


def test_code_dotplot(tmp_path):
    from nicheverse.annotate import code_dotplot

    out = code_dotplot(_adata(), "cell_codebook_idx", top_n=3, save_path=str(tmp_path / "dot.pdf"))
    assert out.exists() and out.read_bytes()[:4] == b"%PDF"


def test_cluster_context(monkeypatch):
    import json

    seen = {"cluster_in_refine": False}

    def mock(prompt, **kw):
        if "revisions" in prompt:
            seen["cluster_in_refine"] = "cluster=" in prompt
            return '{"revisions": {}}'
        return json.dumps(
            {
                "label": "X",
                "compartment": "immune",
                "confidence": 0.8,
                "rationale": "m",
                "key_markers": ["G0"],
                "citations": [],
            }
        )

    monkeypatch.setattr(A, "call_llm", mock)
    df = annotate_codes(
        _adata(), "cell_codebook_idx", AnnotationConfig(refine=True, cluster_context=True)
    )
    assert "cluster" in df.columns and seen["cluster_in_refine"]


def test_cluster_codes_constant_data():
    from nicheverse.annotate import cluster_codes

    a = ad.AnnData(X=sp.csr_matrix(np.ones((20, 6), dtype="float32")))
    a.var_names = [f"G{i}" for i in range(6)]
    a.obs["cell_codebook_idx"] = np.array(["0"] * 10 + ["1"] * 10)
    df = cluster_codes(a, "cell_codebook_idx")  # zero-variance rows must not crash linkage
    assert set(df.index) == {"0", "1"}


def test_attach_labels_int_keys():
    from nicheverse.annotate import attach_labels

    a = ad.AnnData(X=sp.csr_matrix(np.ones((6, 3), dtype="float32")))
    a.obs["cell_codebook_idx"] = np.array(["0", "1", "2"] * 2)
    attach_labels(
        a, "cell_codebook_idx", {0: "A", 1: "B", 2: "C"}
    )  # int keys must map, not "unknown"
    assert set(a.obs["celltype_annot"]) == {"A", "B", "C"}


def test_code_evidence_handles_nonfinite():
    a = ad.AnnData(
        X=np.array(
            [[1.0, np.nan, 2.0], [3.0, 4.0, np.inf], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            dtype="float32",
        )
    )
    a.var_names = ["A", "B", "C"]
    a.obs["cell_codebook_idx"] = np.array(["0", "0", "1", "1"])
    ev = code_evidence(a, "cell_codebook_idx")
    assert all(np.isfinite(z) for _, z in ev["0"]["top_markers"])
