"""Per-code expression evidence for annotation: markers, DEGs, and distributions.

Assembles, for each learned code, the quantitative evidence a human or an LLM needs
to call a cell type or state: top markers by z-score across codes, one-vs-rest
differential expression, and the code's distribution over metadata columns
(site, sample, diagnosis, ...). This mirrors the manual codebook-review pipeline.
"""

from __future__ import annotations

import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = [
    "code_evidence",
    "niche_evidence",
    "cluster_codes",
    "code_groundtruth_concordance",
    "code_context",
    "write_evidence_bundle",
]


def _sorted_codes(codes: pd.Series) -> list[str]:
    """Codes as strings, sorted by (length, value) so 2 < 10 for numeric-looking codes."""
    return sorted(codes.astype(str).unique(), key=lambda c: (len(c), c))


def code_evidence(
    adata: ad.AnnData,
    code_col: str,
    *,
    extra_cols: tuple[str, ...] = (),
    top_markers: int = 30,
    top_degs: int = 30,
    layer: str | None = None,
) -> dict[str, dict]:
    """Assemble per-code evidence for annotation.

    Parameters
    ----------
    adata
        AnnData carrying the code assignment in ``obs[code_col]``.
    code_col
        Column of ``obs`` with the code index (e.g. ``cell_codebook_idx``).
    extra_cols
        ``obs`` columns to summarize the per-code distribution over (e.g.
        ``("site_class", "sample_id")``).
    top_markers
        Number of top markers per code (ranked by z-score across codes).
    top_degs
        Number of one-vs-rest DEGs per code (``rank_genes_groups``).
    layer
        Optional ``layers`` key to read expression from; defaults to ``X``.

    Returns
    -------
    dict
        ``{code: {code, n_cells, frac, top_markers, top_degs, dist_<col>}}``.
        ``top_markers`` is a list of ``(gene, z)``; ``top_degs`` a list of
        ``(gene, log2fc, padj)``.
    """
    codes = adata.obs[code_col].astype(str)
    genes = list(map(str, adata.var_names))
    x = adata.layers[layer] if layer else adata.X
    xd = x.toarray() if sp.issparse(x) else np.asarray(x)
    xd = np.nan_to_num(xd, nan=0.0, posinf=0.0, neginf=0.0)
    uniq = sorted(codes.unique(), key=lambda c: (len(c), c))
    means = np.vstack([xd[(codes == c).to_numpy()].mean(0) for c in uniq])
    z = (means - means.mean(0)) / (means.std(0) + 1e-9)

    deg = None
    try:
        import scanpy as sc

        a2 = ad.AnnData(
            X=(x.copy() if hasattr(x, "copy") else np.asarray(x)),
            obs=pd.DataFrame({"_code": pd.Categorical(codes.values)}, index=adata.obs_names)
        )
        a2.var_names = adata.var_names
        chk = a2.X.data[:2000] if sp.issparse(a2.X) else np.asarray(a2.X).ravel()[:2000]
        if len(chk) and np.allclose(chk, np.round(chk)):  # raw counts -> log-normalize for the DEG test
            sc.pp.normalize_total(a2)
            sc.pp.log1p(a2)
        sc.tl.rank_genes_groups(a2, "_code", method="t-test_overestim_var", n_genes=top_degs)
        deg = a2.uns["rank_genes_groups"]
    except Exception:
        deg = None

    deg_names = set(deg["names"].dtype.names) if deg is not None else set()
    out: dict[str, dict] = {}
    for i, c in enumerate(uniq):
        m = (codes == c).to_numpy()
        order = np.argsort(z[i])[::-1][:top_markers]
        ev = {
            "code": c,
            "n_cells": int(m.sum()),
            "frac": float(m.mean()),
            "top_markers": [(genes[j], round(float(z[i][j]), 2)) for j in order],
        }
        if deg is not None and c in deg_names:
            names, lfc, pad = deg["names"][c], deg["logfoldchanges"][c], deg["pvals_adj"][c]
            ev["top_degs"] = [
                (str(names[k]), round(float(lfc[k]), 2), float(pad[k])) for k in range(min(top_degs, len(names)))
            ]
        for col in extra_cols:
            if col in adata.obs.columns:
                vc = adata.obs.loc[m, col].astype(str).value_counts(normalize=True)
                ev[f"dist_{col}"] = {str(k): round(float(v), 3) for k, v in vc.head(6).items()}
        out[c] = ev
    return out


def niche_evidence(
    adata: ad.AnnData,
    niche_col: str,
    celltype_col: str,
    *,
    extra_cols: tuple[str, ...] = (),
    top_markers: int = 20,
    top_compositions: int = 8,
) -> dict[str, dict]:
    """Per-niche evidence: the cell-type composition and enriched genes of each spatial niche.

    A niche (neighborhood code) is characterized by the community of cell types that
    co-occur in it, so annotation needs ``celltype_col`` (typically the labels produced
    by annotating the cell codes) alongside marker enrichment.

    Returns ``{niche: {code, n_cells, frac, composition [(cell_type, fraction)],
    top_markers [(gene, z)], dist_<col>}}``.
    """
    niches = adata.obs[niche_col].astype(str)
    cts = adata.obs[celltype_col].astype(str)
    genes = list(map(str, adata.var_names))
    x = adata.X
    xd = x.toarray() if sp.issparse(x) else np.asarray(x)
    xd = np.nan_to_num(xd, nan=0.0, posinf=0.0, neginf=0.0)
    uniq = sorted(niches.unique(), key=lambda c: (len(c), c))
    means = np.vstack([xd[(niches == c).to_numpy()].mean(0) for c in uniq])
    z = (means - means.mean(0)) / (means.std(0) + 1e-9)
    out: dict[str, dict] = {}
    for i, c in enumerate(uniq):
        m = (niches == c).to_numpy()
        comp = cts[m].value_counts(normalize=True).head(top_compositions)
        order = np.argsort(z[i])[::-1][:top_markers]
        ev = {
            "code": c,
            "n_cells": int(m.sum()),
            "frac": float(m.mean()),
            "composition": [(str(k), round(float(v), 3)) for k, v in comp.items()],
            "top_markers": [(genes[j], round(float(z[i][j]), 2)) for j in order],
        }
        for col in extra_cols:
            if col in adata.obs.columns:
                vc = adata.obs.loc[m, col].astype(str).value_counts(normalize=True)
                ev[f"dist_{col}"] = {str(k): round(float(v), 3) for k, v in vc.head(6).items()}
        out[c] = ev
    return out


def cluster_codes(
    adata: ad.AnnData, code_col: str, *, n_clusters: int | None = None, layer: str | None = None
) -> pd.DataFrame:
    """Hierarchically group codes by mean-expression correlation.

    Clusters similar codes together (correlation distance + average linkage) so they
    can be reviewed and annotated coarse-to-fine, as in the manual codebook pipeline.
    ``n_clusters`` defaults to about one cluster per eight codes.

    Returns a DataFrame indexed by code with an integer ``cluster`` column.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    codes = adata.obs[code_col].astype(str)
    x = adata.layers[layer] if layer else adata.X
    xd = x.toarray() if sp.issparse(x) else np.asarray(x)
    uniq = sorted(codes.unique(), key=lambda c: (len(c), c))
    if len(uniq) < 2:
        return pd.DataFrame({"cluster": [1] * len(uniq)}, index=uniq)
    means = np.vstack([xd[(codes == c).to_numpy()].mean(0) for c in uniq])
    z = (means - means.mean(0)) / (means.std(0) + 1e-9)
    dist = pdist(z, metric="correlation")
    dist = np.nan_to_num(dist, nan=2.0, posinf=2.0)  # undefined correlation (constant code) -> max distance
    linkage_z = linkage(dist, method="average")
    k = min(n_clusters or max(2, len(uniq) // 8), len(uniq))
    labels = fcluster(linkage_z, t=k, criterion="maxclust")
    return pd.DataFrame({"cluster": [int(v) for v in labels]}, index=uniq)


def code_groundtruth_concordance(
    adata: ad.AnnData, code_col: str, groundtruth_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-check codes against an independent ground-truth clustering/annotation.

    Compares the learned codes in ``obs[code_col]`` to a separately-derived label
    column (e.g. a Leiden clustering or a manual annotation) in ``obs[groundtruth_col]``.
    This is the Leiden cross-check step of the codebook-review pipeline.

    Parameters
    ----------
    adata
        AnnData carrying both columns in ``obs``.
    code_col
        Column with the learned code index.
    groundtruth_col
        Column with the independent ground-truth label.

    Returns
    -------
    tuple
        ``(crosstab, majority)``. ``crosstab`` is a DataFrame of code (rows) x
        ground-truth group (columns) normalized so each code row sums to ~1
        (rows for codes with no valid ground-truth cells are all zero).
        ``majority`` has one row per code with columns
        ``code, majority_group, majority_frac, n_cells``.
    """
    codes_all = adata.obs[code_col].astype(str)
    uniq = _sorted_codes(codes_all)
    if groundtruth_col not in adata.obs.columns:
        crosstab = pd.DataFrame(0.0, index=uniq, columns=[])
        maj = pd.DataFrame(
            {
                "code": uniq,
                "majority_group": [None] * len(uniq),
                "majority_frac": [0.0] * len(uniq),
                "n_cells": [int((codes_all == c).sum()) for c in uniq],
            }
        )
        return crosstab, maj

    gt = adata.obs[groundtruth_col].astype("object")
    valid = gt.notna().to_numpy() & (gt.astype(str).to_numpy() != "nan")
    gt_str = gt.astype(str)
    counts = pd.crosstab(codes_all[valid], gt_str[valid])
    gt_groups = sorted(map(str, counts.columns)) if counts.shape[1] else []
    counts = counts.reindex(index=uniq, columns=gt_groups, fill_value=0)
    row_tot = counts.sum(1).replace(0, np.nan)
    crosstab = counts.div(row_tot, axis=0).fillna(0.0)
    crosstab.index.name = "code"

    rows = []
    for c in uniq:
        n_cells = int((codes_all == c).sum())
        if len(gt_groups) and float(counts.loc[c].sum()) > 0:
            top = counts.loc[c].idxmax()
            frac = float(crosstab.loc[c, top])
            rows.append((c, str(top), round(frac, 4), n_cells))
        else:
            rows.append((c, None, 0.0, n_cells))
    maj = pd.DataFrame(rows, columns=["code", "majority_group", "majority_frac", "n_cells"])
    return crosstab, maj


def code_context(
    adata: ad.AnnData,
    code_col: str,
    *,
    patient_col: str | None = None,
    context_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Per-code patient / specimen context.

    One row per code with ``n_cells``, ``frac``, the number of distinct patients
    (if ``patient_col`` given), and for each column in ``context_cols`` the
    dominant value and its fraction. This is the patient/specimen-context step of
    the codebook-review pipeline. Robust to absent columns (silently skipped).

    Returns a DataFrame indexed by code with columns ``n_cells, frac``,
    optional ``n_patients``, and per context col ``<col>_dominant`` +
    ``<col>_dominant_frac``.
    """
    codes_all = adata.obs[code_col].astype(str)
    uniq = _sorted_codes(codes_all)
    n_total = len(codes_all)
    rows: dict[str, dict] = {}
    for c in uniq:
        m = (codes_all == c).to_numpy()
        row: dict = {"n_cells": int(m.sum()), "frac": float(m.mean()) if n_total else 0.0}
        if patient_col and patient_col in adata.obs.columns:
            row["n_patients"] = int(adata.obs.loc[m, patient_col].astype(str).nunique())
        for col in context_cols:
            if col in adata.obs.columns:
                vc = adata.obs.loc[m, col].astype(str).value_counts(normalize=True)
                if len(vc):
                    row[f"{col}_dominant"] = str(vc.index[0])
                    row[f"{col}_dominant_frac"] = round(float(vc.iloc[0]), 4)
                else:
                    row[f"{col}_dominant"] = None
                    row[f"{col}_dominant_frac"] = 0.0
        rows[c] = row
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "code"
    return df


def write_evidence_bundle(
    adata: ad.AnnData,
    code_col: str,
    out_dir: str,
    *,
    groundtruth_col: str | None = None,
    patient_col: str | None = None,
    context_cols: tuple[str, ...] = (),
    top_markers: int = 30,
    top_degs: int = 30,
    layer: str | None = None,
) -> dict[str, str]:
    """Compute the full per-code evidence bundle and dump it as CSVs.

    Mirrors the documented 8-part codebook-review pipeline: per-code mean
    expression, z-score across codes, top markers, one-vs-rest DEGs,
    patient/specimen context, hierarchical code clustering, and (when a
    ground-truth column is given) the ground-truth crosstab + per-code majority.

    Parameters
    ----------
    adata
        AnnData carrying the code assignment in ``obs[code_col]``.
    code_col
        Column of ``obs`` with the code index.
    out_dir
        Directory to write CSVs into (created if absent).
    groundtruth_col
        Optional independent label column for the Leiden/ground-truth cross-check.
    patient_col
        Optional column counted for ``n_patients`` in the context table.
    context_cols
        ``obs`` columns to summarize (dominant value) in the context table.
    top_markers, top_degs
        Numbers passed through to :func:`code_evidence`.
    layer
        Optional ``layers`` key for expression; defaults to ``X``.

    Returns
    -------
    dict
        Mapping of artifact name -> written CSV path.
    """
    os.makedirs(out_dir, exist_ok=True)
    written: dict[str, str] = {}

    codes = adata.obs[code_col].astype(str)
    genes = list(map(str, adata.var_names))
    uniq = _sorted_codes(codes)
    x = adata.layers[layer] if layer else adata.X
    xd = x.toarray() if sp.issparse(x) else np.asarray(x)
    xd = np.nan_to_num(xd, nan=0.0, posinf=0.0, neginf=0.0)
    means = np.vstack([xd[(codes == c).to_numpy()].mean(0) for c in uniq])
    z = (means - means.mean(0)) / (means.std(0) + 1e-9)

    mean_df = pd.DataFrame(means, index=uniq, columns=genes)
    mean_df.index.name = "code"
    p = os.path.join(out_dir, "per_code_mean_expression.csv")
    mean_df.to_csv(p)
    written["per_code_mean_expression"] = p

    z_df = pd.DataFrame(z, index=uniq, columns=genes)
    z_df.index.name = "code"
    p = os.path.join(out_dir, "per_code_zscore_across_codes.csv")
    z_df.to_csv(p)
    written["per_code_zscore_across_codes"] = p

    ev = code_evidence(
        adata, code_col, top_markers=top_markers, top_degs=top_degs, layer=layer
    )
    mk_rows = []
    for c in uniq:
        for rank, (g, zz) in enumerate(ev.get(c, {}).get("top_markers", []), start=1):
            mk_rows.append({"code": c, "rank": rank, "gene": g, "zscore": zz})
    mk_df = pd.DataFrame(mk_rows, columns=["code", "rank", "gene", "zscore"])
    p = os.path.join(out_dir, "per_code_top_markers.csv")
    mk_df.to_csv(p, index=False)
    written["per_code_top_markers"] = p

    deg_rows = []
    for c in uniq:
        for rank, (g, lfc, padj) in enumerate(ev.get(c, {}).get("top_degs", []), start=1):
            deg_rows.append(
                {"code": c, "rank": rank, "gene": g, "log2fc": lfc, "pval_adj": padj}
            )
    deg_df = pd.DataFrame(deg_rows, columns=["code", "rank", "gene", "log2fc", "pval_adj"])
    p = os.path.join(out_dir, "per_code_DEG_top30_1vsRest.csv")
    deg_df.to_csv(p, index=False)
    written["per_code_DEG_top30_1vsRest"] = p

    ctx_df = code_context(
        adata, code_col, patient_col=patient_col, context_cols=context_cols
    )
    p = os.path.join(out_dir, "per_code_context.csv")
    ctx_df.to_csv(p)
    written["per_code_context"] = p

    clu = cluster_codes(adata, code_col, layer=layer)
    clu.index.name = "code"
    p = os.path.join(out_dir, "per_code_hier_cluster_assignment.csv")
    clu.to_csv(p)
    written["per_code_hier_cluster_assignment"] = p

    if groundtruth_col is not None:
        crosstab, maj = code_groundtruth_concordance(adata, code_col, groundtruth_col)
        p = os.path.join(out_dir, "per_code_groundtruth_crosstab.csv")
        crosstab.to_csv(p)
        written["per_code_groundtruth_crosstab"] = p
        p = os.path.join(out_dir, "per_code_groundtruth_majority.csv")
        maj.to_csv(p, index=False)
        written["per_code_groundtruth_majority"] = p

    return written
