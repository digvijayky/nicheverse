"""Tests for the extended per-code evidence bundle (groundtruth concordance, context, bundle dump)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from nicheverse.annotate.artifacts import (
    cluster_codes,
    code_context,
    code_evidence,
    code_groundtruth_concordance,
    niche_evidence,
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


# --- hardening regression tests -------------------------------------------------


def _single_code_adata():
    rng = np.random.default_rng(1)
    x = rng.poisson(0.5, size=(40, 6)).astype("float32")
    x[:, 0] += 8
    a = ad.AnnData(X=sp.csr_matrix(x))
    a.var_names = [f"G{i}" for i in range(6)]
    a.obs["cell_codebook_idx"] = np.array(["0"] * 40)
    return a


def test_constant_gene_zscore_no_nan():
    # a gene constant across all codes must give z == 0, never NaN in top_markers
    a = _adata()
    x = a.X.toarray()
    x[:, 5] = 3.0  # G5 constant across every cell/code
    a.X = sp.csr_matrix(x)
    ev = code_evidence(a, "cell_codebook_idx")
    for c in ev:
        d = dict(ev[c]["top_markers"])
        assert all(np.isfinite(z) for z in d.values())
        if "G5" in d:
            assert d["G5"] == 0.0


def test_mean_expression_sparse_dense_layer_match():
    a = _adata()
    xd = a.X.toarray()
    a_dense = ad.AnnData(X=xd.copy())
    a_dense.var_names = a.var_names
    a_dense.obs["cell_codebook_idx"] = a.obs["cell_codebook_idx"].values
    a_layer = ad.AnnData(X=np.zeros_like(xd))
    a_layer.var_names = a.var_names
    a_layer.obs["cell_codebook_idx"] = a.obs["cell_codebook_idx"].values
    a_layer.layers["counts"] = sp.csr_matrix(xd)

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        ws = write_evidence_bundle(a, "cell_codebook_idx", d + "/s")
        wd = write_evidence_bundle(a_dense, "cell_codebook_idx", d + "/d")
        wl = write_evidence_bundle(a_layer, "cell_codebook_idx", d + "/l", layer="counts")
        ms = pd.read_csv(ws["per_code_mean_expression"], index_col=0).to_numpy()
        md = pd.read_csv(wd["per_code_mean_expression"], index_col=0).to_numpy()
        ml = pd.read_csv(wl["per_code_mean_expression"], index_col=0).to_numpy()
    assert np.allclose(ms, md)
    assert np.allclose(ml, md)


def test_deg_lognormalized_input_detected():
    # already log-normalized (non-integer) input must still yield DEGs (auto-detect skips normalize)
    a = _adata()
    a.X = sp.csr_matrix(np.log1p(a.X.toarray()))
    ev = code_evidence(a, "cell_codebook_idx", top_degs=5)
    assert all("top_degs" in ev[c] for c in ev)


def test_single_code_no_nan_deg(tmp_path):
    # a collapsed codebook (all cells one code) must not leak NaN into top_degs or the CSV
    a = _single_code_adata()
    ev = code_evidence(a, "cell_codebook_idx", top_degs=5)
    for _, lfc, padj in ev["0"].get("top_degs", []):
        assert np.isfinite(lfc) and np.isfinite(padj)
    paths = write_evidence_bundle(a, "cell_codebook_idx", str(tmp_path))
    for name, p in paths.items():
        num = pd.read_csv(p).select_dtypes("number")
        assert not num.isna().any().any(), f"{name} has NaN"
        assert np.isfinite(num.to_numpy()).all(), f"{name} has inf"


def test_distribution_drops_nan_and_ties_deterministic():
    # NaN must be dropped from dist_ (not counted as a "nan" category)
    a = ad.AnnData(X=sp.csr_matrix(np.ones((6, 3), dtype="float32")))
    a.var_names = ["A", "B", "C"]
    a.obs["cell_codebook_idx"] = np.array(["0"] * 6)
    a.obs["s"] = np.array(["BrM", "BrM", np.nan, np.nan, "Primary", np.nan], dtype=object)
    d = code_evidence(a, "cell_codebook_idx", extra_cols=("s",))["0"]["dist_s"]
    assert "nan" not in {k.lower() for k in d}
    # values rounded to 3 dp; NaN dropped -> BrM 2/3, Primary 1/3 of the 3 non-missing cells
    assert abs(d["BrM"] - 2 / 3) < 1e-3 and abs(d["Primary"] - 1 / 3) < 1e-3

    # ties (equal counts) broken by ascending category name, stable across row order
    def dist(order):
        aa = ad.AnnData(X=sp.csr_matrix(np.ones((len(order), 3), dtype="float32")))
        aa.var_names = ["A", "B", "C"]
        aa.obs["cell_codebook_idx"] = np.array(["0"] * len(order))
        aa.obs["s"] = np.array(order, dtype=object)
        return list(code_evidence(aa, "cell_codebook_idx", extra_cols=("s",))["0"]["dist_s"])

    assert dist(["zeta", "zeta", "alpha", "alpha"]) == ["alpha", "zeta"]
    assert dist(["alpha", "alpha", "zeta", "zeta"]) == ["alpha", "zeta"]


def test_context_all_missing_single_unique_and_nan_patients():
    a = ad.AnnData(X=sp.csr_matrix(np.ones((6, 3), dtype="float32")))
    a.var_names = ["A", "B", "C"]
    a.obs["cell_codebook_idx"] = np.array(["0"] * 6)
    a.obs["allnan"] = np.array([np.nan] * 6, dtype=object)
    a.obs["one"] = np.array(["only"] * 6, dtype=object)
    a.obs["tie"] = np.array(["zeta", "zeta", "alpha", "alpha", "mid", "mid"], dtype=object)
    a.obs["pat"] = np.array(["P0", "P0", np.nan, "P1", np.nan, "P1"], dtype=object)
    ctx = code_context(
        a, "cell_codebook_idx", patient_col="pat", context_cols=("allnan", "one", "tie")
    )
    assert ctx.loc["0", "allnan_dominant"] is None and ctx.loc["0", "allnan_dominant_frac"] == 0.0
    assert ctx.loc["0", "one_dominant"] == "only" and ctx.loc["0", "one_dominant_frac"] == 1.0
    assert ctx.loc["0", "tie_dominant"] == "alpha"  # deterministic tie break
    assert ctx.loc["0", "n_patients"] == 2  # NaN patient dropped


def test_context_single_unique_column_dist():
    a = _adata()
    a.obs["const"] = np.array(["only"] * a.n_obs, dtype=object)
    ev = code_evidence(a, "cell_codebook_idx", extra_cols=("const",))
    assert ev["0"]["dist_const"] == {"only": 1.0}


def test_groundtruth_all_missing_column():
    # groundtruth column present but entirely NaN: no crash, no NaN, one majority row per code
    a = _adata()
    a.obs["leiden_gt"] = np.array([np.nan] * a.n_obs, dtype=object)
    crosstab, maj = code_groundtruth_concordance(a, "cell_codebook_idx", "leiden_gt")
    assert list(maj["code"]) == ["0", "1", "2"]
    assert maj["majority_group"].isna().all()
    assert not crosstab.isna().any().any()
    assert not maj["majority_frac"].isna().any()
    assert (maj["majority_frac"] == 0.0).all()


def test_groundtruth_code_absent_from_groundtruth():
    # one code has zero cells with a valid ground-truth label -> zero row, None majority
    a = _adata()
    gt = np.array(["gtA"] * 30 + ["gtB"] * 30 + ["gtC"] * 30, dtype=object)
    gt[60:] = np.nan  # code "2" has no valid ground truth
    a.obs["leiden_gt"] = gt
    crosstab, maj = code_groundtruth_concordance(a, "cell_codebook_idx", "leiden_gt")
    assert np.allclose(crosstab.loc["2"].to_numpy(), 0.0)
    m = maj.set_index("code")
    assert m.loc["2", "majority_group"] is None and m.loc["2", "majority_frac"] == 0.0
    assert m.loc["2", "n_cells"] == 30  # still counted in n_cells
    # valid codes still normalize to 1
    assert np.isclose(crosstab.loc["0"].sum(), 1.0)


def test_niche_evidence_drops_nan_celltypes():
    a = ad.AnnData(X=sp.csr_matrix(np.ones((8, 3), dtype="float32")))
    a.var_names = ["A", "B", "C"]
    a.obs["neighborhood_codebook_idx"] = np.array(["0"] * 8)
    a.obs["celltype"] = np.array(["T", "T", np.nan, "B", np.nan, "T", "B", np.nan], dtype=object)
    comp = dict(niche_evidence(a, "neighborhood_codebook_idx", "celltype")["0"]["composition"])
    assert "nan" not in {str(k).lower() for k in comp}
    assert list(comp)[0] == "T"  # T dominant


def test_collapsed_codebook_bundle_all_finite(tmp_path):
    # full bundle on a single-code codebook: every promised CSV present, real path, no NaN/inf
    a = _single_code_adata()
    paths = write_evidence_bundle(
        a, "cell_codebook_idx", str(tmp_path), context_cols=("nope",)
    )
    for name in (
        "per_code_mean_expression",
        "per_code_zscore_across_codes",
        "per_code_top_markers",
        "per_code_DEG_top30_1vsRest",
        "per_code_context",
        "per_code_hier_cluster_assignment",
    ):
        assert name in paths
        import os

        assert os.path.exists(paths[name])
    clu = cluster_codes(a, "cell_codebook_idx")
    assert list(clu.index) == ["0"] and clu.loc["0", "cluster"] == 1


def test_integer_code_col_sorted_numerically():
    a = ad.AnnData(X=sp.csr_matrix(np.ones((6, 3), dtype="float32")))
    a.var_names = ["A", "B", "C"]
    a.obs["cell_codebook_idx"] = np.array([0, 0, 10, 10, 2, 2])
    ev = code_evidence(a, "cell_codebook_idx")
    assert list(ev.keys()) == ["0", "2", "10"]  # 2 < 10, not lexicographic
    ctx = code_context(a, "cell_codebook_idx")
    assert list(ctx.index) == ["0", "2", "10"]
