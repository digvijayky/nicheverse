"""Xenium output loaders.

Reads native 10x Xenium output bundles (``cell_feature_matrix.h5`` or the
companion ``cell_feature_matrix/`` mtx folder, plus ``cells.parquet`` or
``cells.csv.gz``) and merges multi-sample cohorts into a single AnnData with
``obs['sample_id']`` and ``obsm['spatial']`` (microns).
"""

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Iterable
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

logger = logging.getLogger(__name__)

# Xenium control / blank / unassigned probe name patterns. We match
# case-insensitively to handle variation across Xenium output versions
# (e.g. ``BLANK_`` vs ``Blank_``).
_CONTROL_PATTERN = re.compile(
    r"(?:NegControl|Control|BLANK|Blank|Codeword|Unassigned|Deprecated|Intergenic|antisense)",
    re.IGNORECASE,
)


def _read_cell_feature_matrix(path: Path) -> ad.AnnData:
    """Load Xenium ``cell_feature_matrix.h5`` or the equivalent 10x-style mtx folder.

    Raises
    ------
    FileNotFoundError
        If neither layout exists under ``path``.
    """
    h5 = path / "cell_feature_matrix.h5"
    if h5.exists():
        return sc.read_10x_h5(h5)
    mtx_dir = path / "cell_feature_matrix"
    if mtx_dir.exists():
        return sc.read_10x_mtx(mtx_dir)
    raise FileNotFoundError(
        f"Could not locate cell_feature_matrix.h5 or cell_feature_matrix/ under {path}. "
        "Ensure the directory points at a Xenium output bundle."
    )


def _read_cells_parquet(path: Path) -> pd.DataFrame:
    """Load Xenium ``cells.parquet`` or fall back to ``cells.csv.gz``."""
    cells = path / "cells.parquet"
    if cells.exists():
        return pd.read_parquet(cells)
    cells_csv = path / "cells.csv.gz"
    if cells_csv.exists():
        return pd.read_csv(cells_csv)
    raise FileNotFoundError(
        f"Could not locate cells.parquet or cells.csv.gz under {path}. "
        "Ensure the directory points at a Xenium output bundle."
    )


def load_xenium_run(
    run_dir: str | Path,
    sample_id: str | None = None,
    drop_controls: bool = True,
) -> ad.AnnData:
    """Load one Xenium output directory into an :class:`anndata.AnnData`.

    Parameters
    ----------
    run_dir
        Path to a Xenium output directory containing ``cell_feature_matrix.h5``
        and ``cells.parquet`` (or their fallbacks).
    sample_id
        Label written to ``obs['sample_id']`` and appended to every barcode as
        ``cell_id__sample_id``. Defaults to ``Path(run_dir).name``.
    drop_controls
        Drop control / blank / unassigned / negative-control / codeword probes
        from ``var`` (case-insensitive match).

    Returns
    -------
    AnnData
        Cells x genes matrix with ``obs['sample_id']``, ``obsm['spatial']``
        (microns), and ``obs`` joined with the contents of ``cells.parquet``.

    Raises
    ------
    FileNotFoundError
        If ``run_dir`` does not exist or is missing required files.
    ValueError
        If ``cells.parquet`` is missing the required ``x_centroid`` /
        ``y_centroid`` columns.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Xenium run directory not found: {run_dir}. "
            "Pass a directory containing cell_feature_matrix.h5 and cells.parquet."
        )
    sample_id = sample_id or run_dir.name
    adata = _read_cell_feature_matrix(run_dir)
    cells = _read_cells_parquet(run_dir)
    if "cell_id" not in cells.columns:
        raise ValueError(
            f"cells.parquet under {run_dir} is missing the 'cell_id' column; "
            "this file may not be a Xenium output."
        )
    cells["cell_id"] = cells["cell_id"].astype(str)
    cells = cells.set_index("cell_id")
    adata.obs_names = adata.obs_names.astype(str)
    common = adata.obs_names.intersection(cells.index)
    if len(common) == 0:
        raise ValueError(
            f"No cell_ids overlap between cell_feature_matrix and cells.parquet under {run_dir}. "
            "The matrix and cells table may be from different runs."
        )
    adata = adata[common].copy()
    cells = cells.loc[common]
    adata.obs = adata.obs.join(cells, how="left")
    if {"x_centroid", "y_centroid"}.issubset(adata.obs.columns):
        adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float64)
    else:
        raise ValueError(
            f"cells.parquet for {run_dir} is missing 'x_centroid' / 'y_centroid' columns; "
            "cannot build adata.obsm['spatial']."
        )
    adata.obs["sample_id"] = sample_id
    adata.obs_names = adata.obs_names + "__" + sample_id
    if drop_controls:
        var_names = adata.var_names.astype(str)
        is_control = np.array([bool(_CONTROL_PATTERN.search(g)) for g in var_names], dtype=bool)
        keep_mask = ~is_control
        n_dropped = int(is_control.sum())
        if n_dropped > 0:
            logger.info(
                "load_xenium_run(%s): dropped %d control/blank/codeword probes",
                sample_id,
                n_dropped,
            )
        adata = adata[:, keep_mask].copy()
    return adata


def load_xenium_cohort(
    run_dirs: Iterable[str | Path] | str | Path,
    sample_ids: Iterable[str] | None = None,
    drop_controls: bool = True,
    manifest: str | Path | None = None,
) -> ad.AnnData:
    """Merge multiple Xenium runs into a single :class:`anndata.AnnData`.

    Parameters
    ----------
    run_dirs
        One Xenium output directory, an iterable of such directories, or unused
        (pass ``[]``) when ``manifest`` is provided.
    sample_ids
        Optional iterable of sample IDs matching ``run_dirs`` length. Defaults
        to ``Path(p).name`` for each run.
    drop_controls
        Drop control / blank / unassigned / codeword probes (see
        :func:`load_xenium_run`).
    manifest
        CSV with columns ``run_dir`` and ``sample_id``. If provided, takes
        precedence over ``run_dirs`` / ``sample_ids``.

    Returns
    -------
    AnnData
        Cohort AnnData restricted to genes shared across all runs, with a
        categorical ``obs['sample_id']``.

    Raises
    ------
    ValueError
        If ``run_dirs`` is empty and no manifest is given, lengths mismatch,
        or no genes are shared across the runs.
    """
    if manifest is not None:
        manifest = Path(manifest)
        if not manifest.exists():
            raise FileNotFoundError(f"manifest CSV not found: {manifest}")
        m = pd.read_csv(manifest)
        missing_cols = {"run_dir", "sample_id"} - set(m.columns)
        if missing_cols:
            raise ValueError(
                f"manifest {manifest} is missing required columns: {sorted(missing_cols)}"
            )
        run_dirs = m["run_dir"].tolist()
        sample_ids = m["sample_id"].astype(str).tolist()
    if isinstance(run_dirs, (str, Path)):
        run_dirs = [run_dirs]
    run_dirs = list(run_dirs)
    if len(run_dirs) == 0:
        raise ValueError(
            "load_xenium_cohort received no run_dirs. Pass at least one directory "
            "or a manifest CSV."
        )
    if sample_ids is None:
        sample_ids = [Path(p).name for p in run_dirs]
    sample_ids = list(sample_ids)
    if len(sample_ids) != len(run_dirs):
        raise ValueError(
            f"sample_ids length ({len(sample_ids)}) must match run_dirs length ({len(run_dirs)})"
        )
    adatas = []
    for p, sid in zip(run_dirs, sample_ids, strict=True):
        a = load_xenium_run(p, sample_id=sid, drop_controls=drop_controls)
        adatas.append(a)
    if len(adatas) == 1:
        merged = adatas[0]
    else:
        gene_sets = [set(a.var_names) for a in adatas]
        common_genes = sorted(set.intersection(*gene_sets))
        if not common_genes:
            raise ValueError(
                "No genes in common across Xenium runs. Check that all runs use "
                "the same gene panel."
            )
        common_set = set(common_genes)
        n_dropped = sum(len(g - common_set) for g in gene_sets)
        if n_dropped > 0:
            warnings.warn(
                f"Restricting to {len(common_genes)} genes shared across all runs "
                f"(dropped {n_dropped} run-specific probes).",
                stacklevel=2,
            )
        adatas = [a[:, common_genes].copy() for a in adatas]
        merged = ad.concat(adatas, axis=0, join="inner", merge="same")
    merged.obs["sample_id"] = merged.obs["sample_id"].astype("category")
    return merged


def _choose_int_dtype(max_value: int) -> np.dtype:
    """Return the narrowest unsigned-friendly int dtype that fits ``max_value``."""
    if max_value < 2**15:
        return np.dtype(np.int16)
    if max_value < 2**31:
        return np.dtype(np.int32)
    return np.dtype(np.int64)


def attach_codes_to_adata(
    adata: ad.AnnData,
    cell_idx: np.ndarray,
    neigh_idx: np.ndarray,
    cell_emb: np.ndarray | None = None,
    neigh_emb: np.ndarray | None = None,
) -> ad.AnnData:
    """Write codebook indices and (optionally) embeddings back into AnnData.

    Indices are stored as the narrowest integer dtype that fits their range,
    saving disk space for the typical case where the codebook has at most a
    few hundred entries.

    Parameters
    ----------
    adata
        AnnData with ``adata.n_obs`` cells; mutated in place.
    cell_idx, neigh_idx
        ``(n_obs,)`` integer arrays of code assignments.
    cell_emb, neigh_emb
        Optional ``(n_obs, D)`` continuous embeddings written to ``adata.obsm``.

    Returns
    -------
    AnnData
        The same ``adata`` with new columns and (optionally) obsm matrices.

    Raises
    ------
    ValueError
        If any input array length differs from ``adata.n_obs``.
    """
    if len(cell_idx) != adata.n_obs:
        raise ValueError(f"cell_idx length ({len(cell_idx)}) != adata.n_obs ({adata.n_obs})")
    if len(neigh_idx) != adata.n_obs:
        raise ValueError(f"neigh_idx length ({len(neigh_idx)}) != adata.n_obs ({adata.n_obs})")
    cell_arr = np.asarray(cell_idx).reshape(-1)
    neigh_arr = np.asarray(neigh_idx).reshape(-1)
    cell_max = int(cell_arr.max()) if cell_arr.size else 0
    neigh_max = int(neigh_arr.max()) if neigh_arr.size else 0
    adata.obs["cell_codebook_idx"] = cell_arr.astype(_choose_int_dtype(cell_max))
    adata.obs["neighborhood_codebook_idx"] = neigh_arr.astype(_choose_int_dtype(neigh_max))
    if cell_emb is not None:
        emb = np.asarray(cell_emb)
        if emb.shape[0] != adata.n_obs:
            raise ValueError(f"cell_emb has {emb.shape[0]} rows, expected {adata.n_obs}")
        adata.obsm["X_cell_embedding"] = emb
    if neigh_emb is not None:
        emb = np.asarray(neigh_emb)
        if emb.shape[0] != adata.n_obs:
            raise ValueError(f"neigh_emb has {emb.shape[0]} rows, expected {adata.n_obs}")
        adata.obsm["X_neighborhood_embedding"] = emb
    return adata


# Short read aliases.
read_xenium = load_xenium_run
read_xenium_cohort = load_xenium_cohort
attach_codes = attach_codes_to_adata
