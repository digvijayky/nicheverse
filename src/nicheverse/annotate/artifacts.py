"""Per-code expression evidence for annotation: markers, DEGs, and distributions.

Assembles, for each learned code, the quantitative evidence a human or an LLM needs
to call a cell type or state: top markers by z-score across codes, one-vs-rest
differential expression, and the code's distribution over metadata columns
(site, sample, diagnosis, ...). This mirrors the manual codebook-review pipeline.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["code_evidence", "niche_evidence", "cluster_codes"]


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
