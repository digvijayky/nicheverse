"""Xenium loader tests against a fake-but-realistic output directory layout."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from nicheverse.data import (
    attach_codes_to_adata,
    load_xenium_cohort,
    load_xenium_run,
)


def _write_fake_xenium_run(
    out_dir: Path,
    n_cells: int = 30,
    extra_genes: tuple[str, ...] = (),
    include_controls: bool = True,
) -> None:
    """Write a minimal Xenium output bundle.

    The cell_feature_matrix.h5 follows the 10x ``matrix`` group layout that
    ``sc.read_10x_h5`` consumes (CSC format with ``data``, ``indices``,
    ``indptr``, ``shape``, plus a ``features`` subgroup).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    gene_names = ["GENE_A", "GENE_B", "GENE_C", "GENE_D", *extra_genes]
    if include_controls:
        gene_names += ["BLANK_0001", "NegControlProbe_0001", "Codeword_0001"]
    n_genes = len(gene_names)
    rng = np.random.default_rng(0)
    # Simulate a sparse matrix (genes x cells == CSC).
    M = rng.poisson(0.5, size=(n_genes, n_cells)).astype(np.int32)
    sparse_mask = rng.random((n_genes, n_cells)) < 0.5
    M[~sparse_mask] = 0
    import scipy.sparse as sp

    M_csc = sp.csc_matrix(M)
    h5path = out_dir / "cell_feature_matrix.h5"
    with h5py.File(h5path, "w") as f:
        grp = f.create_group("matrix")
        grp.create_dataset("data", data=M_csc.data.astype(np.int32))
        grp.create_dataset("indices", data=M_csc.indices.astype(np.int64))
        grp.create_dataset("indptr", data=M_csc.indptr.astype(np.int64))
        grp.create_dataset("shape", data=np.array([n_genes, n_cells], dtype=np.int32))
        cell_ids = np.array([f"cell_{i:05d}" for i in range(n_cells)], dtype="S")
        grp.create_dataset("barcodes", data=cell_ids)
        feats = grp.create_group("features")
        feats.create_dataset("id", data=np.array(gene_names, dtype="S"))
        feats.create_dataset("name", data=np.array(gene_names, dtype="S"))
        feats.create_dataset(
            "feature_type",
            data=np.array(["Gene Expression"] * n_genes, dtype="S"),
        )
        feats.create_dataset(
            "_all_tag_keys",
            data=np.array(["genome"], dtype="S"),
        )
        feats.create_dataset("genome", data=np.array(["xenium"] * n_genes, dtype="S"))

    cells = pd.DataFrame(
        {
            "cell_id": [f"cell_{i:05d}" for i in range(n_cells)],
            "x_centroid": rng.uniform(0, 1000, n_cells),
            "y_centroid": rng.uniform(0, 1000, n_cells),
            "transcript_counts": rng.integers(10, 200, n_cells),
            "nucleus_area": rng.uniform(20, 80, n_cells),
        }
    )
    cells.to_parquet(out_dir / "cells.parquet")


def test_load_xenium_run_basic(tmp_path):
    run = tmp_path / "run_A"
    _write_fake_xenium_run(run, n_cells=25)
    a = load_xenium_run(run)
    assert "sample_id" in a.obs.columns
    assert (a.obs["sample_id"] == "run_A").all()
    assert "spatial" in a.obsm
    assert a.obsm["spatial"].shape == (a.n_obs, 2)
    # Controls dropped.
    for bad in ("BLANK_0001", "NegControlProbe_0001", "Codeword_0001"):
        assert bad not in a.var_names
    assert a.obs_names[0].endswith("__run_A")


def test_load_xenium_run_keep_controls(tmp_path):
    run = tmp_path / "run_keep"
    _write_fake_xenium_run(run, n_cells=10)
    a = load_xenium_run(run, drop_controls=False)
    assert "BLANK_0001" in a.var_names


def test_load_xenium_run_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="run directory not found"):
        load_xenium_run(tmp_path / "does_not_exist")


def test_load_xenium_cohort_merges_two_runs(tmp_path):
    r1 = tmp_path / "r1"
    r2 = tmp_path / "r2"
    _write_fake_xenium_run(r1, n_cells=15)
    _write_fake_xenium_run(r2, n_cells=20, extra_genes=("GENE_RARE",))
    merged = load_xenium_cohort([r1, r2])
    assert merged.n_obs == 35
    # GENE_RARE present only in r2; merge restricts to intersection.
    assert "GENE_RARE" not in merged.var_names
    assert set(merged.obs["sample_id"].astype(str).unique()) == {"r1", "r2"}


def test_load_xenium_cohort_manifest(tmp_path):
    r1 = tmp_path / "alpha"
    r2 = tmp_path / "beta"
    _write_fake_xenium_run(r1, n_cells=10)
    _write_fake_xenium_run(r2, n_cells=12)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(f"run_dir,sample_id\n{r1},alpha\n{r2},beta\n")
    merged = load_xenium_cohort([], manifest=manifest)
    assert merged.n_obs == 22
    assert set(merged.obs["sample_id"].astype(str).unique()) == {"alpha", "beta"}


def test_load_xenium_cohort_no_common_genes_raises(tmp_path):
    # Two runs whose probes share no gene names.
    import scipy.sparse as sp

    for name, genes in (("r_x", ("X1", "X2", "X3")), ("r_y", ("Y1", "Y2", "Y3"))):
        out = tmp_path / name
        out.mkdir()
        n_g, n_c = 3, 5
        M = sp.csc_matrix(np.ones((n_g, n_c), dtype=np.int32))
        with h5py.File(out / "cell_feature_matrix.h5", "w") as f:
            g = f.create_group("matrix")
            g.create_dataset("data", data=M.data.astype(np.int32))
            g.create_dataset("indices", data=M.indices.astype(np.int64))
            g.create_dataset("indptr", data=M.indptr.astype(np.int64))
            g.create_dataset("shape", data=np.array([n_g, n_c], dtype=np.int32))
            g.create_dataset("barcodes", data=np.array([f"c_{i}".encode() for i in range(n_c)]))
            ft = g.create_group("features")
            ft.create_dataset("id", data=np.array(list(genes), dtype="S"))
            ft.create_dataset("name", data=np.array(list(genes), dtype="S"))
            ft.create_dataset("feature_type", data=np.array(["Gene Expression"] * n_g, dtype="S"))
            ft.create_dataset("_all_tag_keys", data=np.array(["genome"], dtype="S"))
            ft.create_dataset("genome", data=np.array(["xenium"] * n_g, dtype="S"))
        pd.DataFrame(
            {
                "cell_id": [f"c_{i}" for i in range(n_c)],
                "x_centroid": np.zeros(n_c),
                "y_centroid": np.zeros(n_c),
            }
        ).to_parquet(out / "cells.parquet")
    with pytest.raises(ValueError, match="No genes in common"):
        load_xenium_cohort([tmp_path / "r_x", tmp_path / "r_y"])


def test_attach_codes_uses_narrow_dtype():
    import anndata as ad

    a = ad.AnnData(X=np.zeros((10, 3), dtype=np.float32))
    a.var_names = ["g0", "g1", "g2"]
    a = attach_codes_to_adata(
        a,
        cell_idx=np.arange(10, dtype=np.int64),
        neigh_idx=np.full(10, 5, dtype=np.int64),
    )
    assert a.obs["cell_codebook_idx"].dtype == np.int16
    assert a.obs["neighborhood_codebook_idx"].dtype == np.int16


def test_attach_codes_length_mismatch_raises():
    import anndata as ad

    a = ad.AnnData(X=np.zeros((5, 3), dtype=np.float32))
    a.var_names = ["g0", "g1", "g2"]
    with pytest.raises(ValueError, match="cell_idx length"):
        attach_codes_to_adata(
            a, cell_idx=np.zeros(4, dtype=np.int64), neigh_idx=np.zeros(5, dtype=np.int64)
        )
