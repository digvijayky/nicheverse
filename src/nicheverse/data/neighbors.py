"""Spatial neighborhood featurizer: aggregated neighbor expression into ``obsm``."""

from __future__ import annotations

import anndata as ad
import numpy as np

from ..constants import Keys
from .dataset import SpatialDataset

__all__ = ["spatial_neighbors"]


def spatial_neighbors(
    adata: ad.AnnData,
    k_neighbors: int = 20,
    neighborhood_aggregation: str = "weighted_mean",
    spatial_graph: str = "knn_radius",
    radius: float | None = 50.0,
    bandwidth: float | None = None,
    sample_col: str = Keys.SAMPLE,
    key_added: str = "neighborhood_features",
    copy: bool = False,
) -> ad.AnnData | np.ndarray:
    """Compute per-sample aggregated neighborhood features and store them in ``obsm``.

    The neighborhood feature for each cell is the concatenation of its own
    expression and an aggregation over its spatial neighbors, computed within
    each sample so cross-sample edges never form (see
    :class:`~nicheverse.data.SpatialDataset`).

    Parameters
    ----------
    adata
        AnnData with ``obsm['spatial']`` (microns) and ``obs[sample_col]``.
    k_neighbors, neighborhood_aggregation, spatial_graph, radius, bandwidth
        Forwarded to :class:`~nicheverse.data.SpatialDataset`.
    sample_col
        ``obs`` column partitioning cells for the per-sample graph.
    key_added
        ``obsm`` key to write the ``(n_cells, 2 * n_genes)`` feature matrix to.
    copy
        If ``True``, operate on and return a copy; otherwise write in place and
        return the ``(n_cells, 2 * n_genes)`` ndarray.

    Returns
    -------
    AnnData or numpy.ndarray
        The mutated (or copied) AnnData when ``copy=True``, else the feature
        matrix ndarray.
    """
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] missing; set micron coordinates first.")
    if sample_col not in adata.obs.columns:
        raise ValueError(f"adata.obs['{sample_col}'] missing.")
    target = adata.copy() if copy else adata
    ds = SpatialDataset.from_anndata(
        target,
        sample_col=sample_col,
        k_neighbors=k_neighbors,
        neighborhood_aggregation=neighborhood_aggregation,
        spatial_graph=spatial_graph,
        radius=radius,
        bandwidth=bandwidth,
    )
    feats = ds.neighborhood_features.cpu().numpy()
    target.obsm[key_added] = feats
    return target if copy else feats
