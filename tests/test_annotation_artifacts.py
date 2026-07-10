"""Tests for the extended per-code evidence bundle (groundtruth concordance, context, bundle dump)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from nicheverse.annotate.artifacts import (
    code_context,
    code_evidence,
    code_groundtruth_concordance,
    write_evidence_bundle,
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
    a.obs["site_class"] = np.array((["BrM"] * 20 + ["Primary"] * 10) * 3)
    a.obs["patient_id"] = np.array(
        [f"P{i % 2}" for i in range(30)]
        + [f"P{2 + i % 3}" for i in range(30)]
        + ["P9"] * 30
    )
    # ground-truth: cleanly tracks codes, with one NaN to exercise robustness
    gt = np.array(["gtA"] * 30 + ["gtB"] * 30 + ["gtC"] * 30, dtype=object)
    gt[0] = np.nan
    a.obs["leiden_gt"] = gt
    return a


def test_code_groundtruth_concordance():
    a = _adata()
    crosstab, maj = code_groundtruth_concordance(a, "cell_codebook_idx", "leiden_gt")
    assert list(crosstab.index) == ["0", "1", "2"]
    # each code row of the normalized crosstab sums to ~1
    assert np.allclose(crosstab.sum(axis=1).to_numpy(), 1.0, atol=1e-6)
    # one majority row per code, well-separated groups map cleanly
    assert list(maj["code"]) == ["0", "1", "2"]
    assert maj.shape[0] == 3
    maj = maj.set_index("code")
    assert maj.loc["0", "majority_group"] == "gtA"
    assert maj.loc["1", "majority_group"] == "gtB"
    assert maj.loc["2", "majority_group"] == "gtC"
    assert (maj["majority_frac"] > 0.9).all()
    assert maj.loc["0", "n_cells"] == 30  # NaN cell still counted in n_cells


def test_code_groundtruth_concordance_missing_col():
    a = _adata()
    crosstab, maj = code_groundtruth_concordance(a, "cell_codebook_idx", "not_a_col")
    assert list(maj["code"]) == ["0", "1", "2"]
    assert maj["majority_group"].isna().all()
    assert crosstab.shape[0] == 3


def test_code_context():
    a = _adata()
    ctx = code_context(
        a, "cell_codebook_idx", patient_col="patient_id", context_cols=("site_class",)
    )
    assert list(ctx.index) == ["0", "1", "2"]
    assert ctx.shape[0] == 3
    assert "n_patients" in ctx.columns
    assert "site_class_dominant" in ctx.columns and "site_class_dominant_frac" in ctx.columns
    assert ctx.loc["0", "n_patients"] == 2
    assert ctx.loc["1", "n_patients"] == 3
    assert ctx.loc["2", "n_patients"] == 1
    assert (ctx["site_class_dominant"] == "BrM").all()  # 20/30 per code
    assert np.allclose(ctx["n_cells"].to_numpy(), 30)


def test_code_context_absent_columns_robust():
    a = _adata()
    ctx = code_context(a, "cell_codebook_idx", patient_col="nope", context_cols=("also_nope",))
    assert list(ctx.index) == ["0", "1", "2"]
    assert "n_patients" not in ctx.columns
    assert "also_nope_dominant" not in ctx.columns
    assert "n_cells" in ctx.columns and "frac" in ctx.columns


def test_write_evidence_bundle(tmp_path):
    a = _adata()
    paths = write_evidence_bundle(
        a,
        "cell_codebook_idx",
        str(tmp_path),
        groundtruth_col="leiden_gt",
        patient_col="patient_id",
        context_cols=("site_class",),
    )
    expected = {
        "per_code_mean_expression",
        "per_code_zscore_across_codes",
        "per_code_top_markers",
        "per_code_DEG_top30_1vsRest",
        "per_code_context",
        "per_code_hier_cluster_assignment",
        "per_code_groundtruth_crosstab",
        "per_code_groundtruth_majority",
    }
    assert expected <= set(paths)
    for name, p in paths.items():
        assert p.endswith(".csv")
        df = pd.read_csv(p)
        assert df.shape[0] > 0, f"{name} is empty"
    # spot-check a couple reload back to expected shape
    mean_df = pd.read_csv(paths["per_code_mean_expression"], index_col=0)
    assert mean_df.shape == (3, 12)
    maj = pd.read_csv(paths["per_code_groundtruth_majority"])
    assert maj.shape[0] == 3


def test_write_evidence_bundle_no_groundtruth(tmp_path):
    a = _adata()
    paths = write_evidence_bundle(a, "cell_codebook_idx", str(tmp_path / "nogt"))
    assert "per_code_groundtruth_crosstab" not in paths
    assert "per_code_groundtruth_majority" not in paths
    assert "per_code_context" in paths and "per_code_top_markers" in paths


def test_write_evidence_bundle_collapsed_code_no_crash(tmp_path):
    # a code with a single cell must not crash the DEG/empty-code path
    a = _adata()
    idx = a.obs["cell_codebook_idx"].to_numpy().copy()
    idx[0] = "9"  # singleton code
    a.obs["cell_codebook_idx"] = idx
    paths = write_evidence_bundle(a, "cell_codebook_idx", str(tmp_path / "collapsed"))
    mean_df = pd.read_csv(paths["per_code_mean_expression"], index_col=0)
    assert "9" in mean_df.index.astype(str).tolist()


def test_existing_code_evidence_unchanged():
    # the existing evidence API must still behave as before
    a = _adata()
    ev = code_evidence(a, "cell_codebook_idx", extra_cols=("site_class",))
    assert set(ev) == {"0", "1", "2"}
    assert "top_markers" in ev["0"] and "dist_site_class" in ev["0"]
    assert {m for m, _ in ev["0"]["top_markers"][:3]} & {"G0", "G1", "G2"}
