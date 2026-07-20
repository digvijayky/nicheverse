"""Transcript-level (subcellular) context.

For each cell, count how many molecules of each panel gene fall within a radius
of its centroid, using the raw per-sample ``transcripts.parquet`` molecule
coordinates. Because it counts every nearby molecule (not just the cell's own
segmented transcripts), the resulting per-cell vector is a segmentation
independent readout of the local molecular field: density plus composition. It
is written to ``obsm`` and can be used as an alternative or additional model
input.
"""

from __future__ import annotations

import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from .xenium import _CONTROL_PATTERN as _CONTROL

__all__ = ["transcript_context"]

# Per-platform molecule-table column names and control-probe patterns.
_PLATFORM_COLS = {
    "xenium": ("x_location", "y_location", "feature_name"),
    "cosmx": ("x_global_px", "y_global_px", "target"),
    "merfish": ("global_x", "global_y", "gene"),
}
_PLATFORM_CONTROL = {
    "xenium": _CONTROL,
    "cosmx": re.compile(r"(?:NegPrb|Negative|SystemControl|FalseCode)", re.IGNORECASE),
    "merfish": re.compile(r"(?:Blank|NegControl)", re.IGNORECASE),
}


def _read_panel_molecules(path, x_col, y_col, feature_col, control, g2c):
    """Read ``(xy, gene_code)`` for panel molecules from a per-sample molecule table.

    Reads row-group batches through :class:`pyarrow.parquet.ParquetFile` and filters each
    batch to panel genes (dropping control/blank probes) before accumulating. This avoids
    the pyarrow dataset-API column-projection path used by ``pandas.read_parquet``, which
    raises ``ArrowInvalid: Invalid number of indices: 0`` on some vendor-written files
    (e.g. 10x Xenium ``transcripts.parquet``), and holds only the kept subset in memory.
    """
    import pyarrow.parquet as _pq

    genes = set(g2c)
    xs, ys, gs = [], [], []
    for batch in _pq.ParquetFile(path).iter_batches(columns=[x_col, y_col, feature_col]):
        d = batch.to_pandas()
        fn = d[feature_col].astype(str)
        keep = (~fn.str.contains(control)) & fn.isin(genes)
        if not keep.any():
            continue
        xs.append(d.loc[keep, x_col].to_numpy(np.float64))
        ys.append(d.loc[keep, y_col].to_numpy(np.float64))
        gs.append(fn[keep].map(g2c).to_numpy())
    if not xs:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.intp)
    return (
        np.column_stack([np.concatenate(xs), np.concatenate(ys)]),
        np.concatenate(gs),
    )


def transcript_context(
    adata: ad.AnnData,
    transcripts: dict | str | Path,
    radius: float = 7.0,
    sample_col: str = "sample_id",
    platform: str = "xenium",
    x_col: str | None = None,
    y_col: str | None = None,
    feature_col: str | None = None,
    control_pattern: re.Pattern | None = None,
    key_added: str = "transcript_context",
    copy: bool = False,
) -> ad.AnnData | np.ndarray:
    """Compute the per-cell local molecular field and store it in ``obsm``.

    Parameters
    ----------
    adata
        AnnData with ``obsm['spatial']`` (microns) and ``obs[sample_col]``.
    transcripts
        Either a mapping ``{sample_id: parquet_path}`` (one molecule table per
        sample) or a single path used for every sample (single-sample runs).
    radius
        Micron radius of the molecular field around each centroid.
    platform
        Molecule-table convention: ``"xenium"`` (``x_location``/``y_location``/
        ``feature_name``), ``"cosmx"`` (``x_global_px``/``y_global_px``/``target``),
        or ``"merfish"`` (``global_x``/``global_y``/``gene``). Sets column and
        control-probe defaults.
    x_col, y_col, feature_col
        Molecule table column names; override the ``platform`` defaults when given.
    control_pattern
        Compiled regex of control/blank probe names to drop; defaults to the
        ``platform`` pattern.
    key_added
        ``obsm`` key for the ``(n_cells, n_genes)`` log1p count matrix.
    copy
        If True, operate on and return a copy; else write in place and return the
        feature matrix.

    Raises
    ------
    ValueError
        If ``obsm['spatial']`` / ``obs[sample_col]`` is missing, or a sample has
        no transcripts path.
    """
    from scipy.spatial import cKDTree

    if platform not in _PLATFORM_COLS:
        raise ValueError(f"unknown platform {platform!r}; choose {sorted(_PLATFORM_COLS)}")
    px, py, pf = _PLATFORM_COLS[platform]
    x_col, y_col, feature_col = x_col or px, y_col or py, feature_col or pf
    control = control_pattern or _PLATFORM_CONTROL[platform]
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] missing; set micron coordinates first.")
    if sample_col not in adata.obs.columns:
        raise ValueError(f"adata.obs['{sample_col}'] missing.")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")

    genes = list(map(str, adata.var_names))
    g2c = {g: i for i, g in enumerate(genes)}
    samples = adata.obs[sample_col].astype(str).to_numpy()
    coords_all = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    if not isinstance(transcripts, dict):
        transcripts = {s: transcripts for s in np.unique(samples)}

    feats = np.zeros((adata.n_obs, len(genes)), dtype=np.float32)
    for sample in np.unique(samples):
        if sample not in transcripts:
            raise ValueError(f"no transcripts path provided for sample {sample!r}")
        cidx = np.where(samples == sample)[0]
        xy, gcol = _read_panel_molecules(
            transcripts[sample], x_col, y_col, feature_col, control, g2c
        )
        if xy.shape[0] == 0:
            continue
        nbrs = cKDTree(xy).query_ball_point(coords_all[cidx], r=radius)
        for j, nb in enumerate(nbrs):
            if nb:
                feats[cidx[j]] = np.bincount(gcol[nb], minlength=len(genes))
    feats = np.log1p(feats)
    target = adata.copy() if copy else adata
    target.obsm[key_added] = feats
    return target if copy else feats
