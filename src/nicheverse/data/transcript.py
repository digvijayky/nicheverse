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


def _iter_molecule_frames(path, x_col, y_col, feature_col):
    """Yield pandas frames of the ``(x, y, feature)`` columns of a molecule table.

    Parquet (Xenium ``transcripts.parquet``) and delimited text, plain or gzipped (CosMx
    ``*_tx_file.csv[.gz]``, MERFISH ``detected_transcripts.csv``) are both accepted; the
    format is taken from the file extension.

    Prefers duckdb when it is importable: it decodes vendor-written molecule tables that
    some pyarrow builds reject outright (10x Xenium ``transcripts.parquet`` fails on
    ``pyarrow`` 22 with ``ArrowInvalid: Invalid number of indices: 0`` on certain row
    groups, whether read whole or column-projected). Falls back to pyarrow row-group
    batches when duckdb is unavailable. Streams in chunks so only a slice is in memory.
    """
    cols = [x_col, y_col, feature_col]
    name = str(path).lower()
    if name.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz", ".txt", ".txt.gz")):
        import pandas as _pd

        sep = "\t" if ".tsv" in name or ".txt" in name else ","
        for chunk in _pd.read_csv(path, sep=sep, usecols=cols, chunksize=2_000_000):
            yield chunk
        return
    try:
        import duckdb  # optional; robust decoder for vendor molecule tables
    except ImportError:
        duckdb = None
    if duckdb is not None:
        esc = str(path).replace("'", "''")
        q = f"SELECT \"{x_col}\", \"{y_col}\", \"{feature_col}\" FROM read_parquet('{esc}')"
        res = duckdb.connect().execute(q)
        reader = (res.to_arrow_reader if hasattr(res, "to_arrow_reader")
                  else res.fetch_record_batch)(1_000_000)
        for batch in reader:
            yield batch.to_pandas()
    else:
        import pyarrow.parquet as _pq

        for batch in _pq.ParquetFile(path).iter_batches(columns=cols):
            yield batch.to_pandas()


def _read_panel_molecules(path, x_col, y_col, feature_col, control, g2c):
    """Read ``(xy, gene_code)`` for panel molecules from a per-sample molecule table.

    Streams frames from :func:`_iter_molecule_frames` and filters each to panel genes
    (dropping control/blank probes) before accumulating, so memory holds only the kept
    subset rather than the full molecule table.
    """
    genes = set(g2c)
    xs, ys, gs = [], [], []
    for d in _iter_molecule_frames(path, x_col, y_col, feature_col):
        fn = d[feature_col].astype(str)
        keep = (~fn.str.contains(control)) & fn.isin(genes)
        if not keep.any():
            continue
        import pandas as _pd

        # Vendor text exports occasionally carry a malformed coordinate; coerce and drop
        # those molecules rather than failing the whole table.
        xv = _pd.to_numeric(d.loc[keep, x_col], errors="coerce").to_numpy(np.float64)
        yv = _pd.to_numeric(d.loc[keep, y_col], errors="coerce").to_numpy(np.float64)
        gv = fn[keep].map(g2c).to_numpy()
        good = np.isfinite(xv) & np.isfinite(yv)
        if not good.any():
            continue
        xs.append(xv[good])
        ys.append(yv[good])
        gs.append(gv[good])
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
    log1p: bool = True,
    sparse: bool = False,
    molecule_scale: float = 1.0,
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
    log1p
        Apply ``log1p`` to the molecule counts (default True, the released behavior).
        Set False to keep the RAW local molecule counts, which is what a count
        likelihood (negative binomial / Dirichlet multinomial) needs as its target.
    sparse
        Return (and store) a :class:`scipy.sparse.csr_matrix` instead of a dense array.
        The field is mostly zeros for a large panel, so this is the memory safe option
        for multi million cell cohorts. Default False (the released behavior).
    molecule_scale
        Multiply the molecule coordinates by this factor before the radius query. Use it
        when the molecule table and ``obsm['spatial']`` share a frame whose unit is not
        microns (CosMx global pixels, for instance): pass the same factor that converts
        the cell coordinates to microns, so ``radius`` stays a real micron distance.
        Default 1.0 (the released behavior).

    Notes
    -----
    Samples that share the same molecule table path are read ONCE and queried together,
    so a TMA slide holding many cores costs one pass over its molecule table rather than
    one pass per core. The result is identical either way, because a cell is only ever
    matched to molecules within ``radius`` of it.

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

    by_path: dict[str, list[int]] = {}
    for sample in np.unique(samples):
        if sample not in transcripts:
            raise ValueError(f"no transcripts path provided for sample {sample!r}")
        by_path.setdefault(str(transcripts[sample]), []).extend(np.where(samples == sample)[0])

    if sparse:
        import scipy.sparse as _sp

        blocks: list = []
    else:
        feats = np.zeros((adata.n_obs, len(genes)), dtype=np.float32)
    rows_all: list[np.ndarray] = []
    for path, cells in by_path.items():
        cidx = np.asarray(sorted(cells))
        xy, gcol = _read_panel_molecules(path, x_col, y_col, feature_col, control, g2c)
        if xy.shape[0] == 0:
            continue
        if molecule_scale != 1.0:
            xy = xy * float(molecule_scale)
        nbrs = cKDTree(xy).query_ball_point(coords_all[cidx], r=radius)
        if sparse:
            ii, jj, vv = [], [], []
            for j, nb in enumerate(nbrs):
                if not nb:
                    continue
                u, c = np.unique(gcol[nb], return_counts=True)
                ii.append(np.full(len(u), j)); jj.append(u); vv.append(c)
            n_local = len(cidx)
            if ii:
                blocks.append(
                    _sp.csr_matrix(
                        (np.concatenate(vv).astype(np.float32), (np.concatenate(ii), np.concatenate(jj))),
                        shape=(n_local, len(genes)),
                    )
                )
            else:
                blocks.append(_sp.csr_matrix((n_local, len(genes)), dtype=np.float32))
            rows_all.append(cidx)
        else:
            for j, nb in enumerate(nbrs):
                if nb:
                    feats[cidx[j]] = np.bincount(gcol[nb], minlength=len(genes))
    if sparse:
        out = _sp.csr_matrix((adata.n_obs, len(genes)), dtype=np.float32)
        if blocks:
            # Blocks are in per-path order; scatter them back to cell order with a
            # permutation matrix (cheap, and it never materializes a dense array).
            stacked = _sp.vstack(blocks).tocsr()
            order = np.concatenate(rows_all)
            perm = _sp.csr_matrix(
                (np.ones(len(order), np.float32), (order, np.arange(len(order)))),
                shape=(adata.n_obs, len(order)),
            )
            out = (perm @ stacked).tocsr()
        if log1p:
            out.data = np.log1p(out.data)
        feats = out
    elif log1p:
        feats = np.log1p(feats)
    target = adata.copy() if copy else adata
    target.obsm[key_added] = feats
    return target if copy else feats
