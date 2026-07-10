"""Spatial dataset: per-sample neighborhood feature aggregation.

The :class:`SpatialDataset` builds a paired ``(cell_features, neighborhood_features)``
tensor pair per cell, where ``neighborhood_features`` is the concatenation of the
cell's own features and an aggregation over its spatial neighbors restricted to
the same sample (so cross-sample edges never form). Seven spatial graph backends
are supported (``"knn"``, ``"radius"``, ``"knn_radius"``, ``"delaunay"``,
``"alpha_complex"``, ``"gabriel"``, ``"rng"``) and five aggregations (``"mean"``,
``"weighted_mean"``, ``"max"``, ``"gaussian"``, ``"inverse_square"``).

The neighbor gather + aggregation is vectorized and processed in bounded-memory
chunks, so the dataset scales to multi-million-cell cohorts without a per-cell
Python loop.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ._knn import knn_query, radius_query

try:
    from scipy.spatial import QhullError
except ImportError:  # pragma: no cover
    from scipy.spatial.qhull import QhullError
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_VALID_AGGREGATIONS = frozenset({"mean", "weighted_mean", "max", "gaussian", "inverse_square"})
_VALID_GRAPHS = frozenset(
    {"knn", "radius", "knn_radius", "delaunay", "alpha_complex", "gabriel", "rng"}
)

# Cells processed per vectorized aggregation chunk. Bounds peak memory of the
# (chunk, k, n_features) neighbor-gather tensor. 100k * 20 * 400 * 4 bytes ~= 3.2 GiB.
_DEFAULT_AGG_CHUNK = 100_000
_EPS = 1e-8
# Minimum neighbor separation (microns) used as a distance floor in the
# reciprocal weight kernels (weighted_mean, inverse_square). Roughly one cell
# radius: two centroids closer than ~1um are within segmentation noise (Xenium
# nucleus effective radius ~2.8um, cell ~4.3um), and coincident/near-coincident
# centroids are common. Without a floor, 1/(d+eps) with d~0 gives an
# astronomically large weight (up to 1e8) that swamps the niche aggregate,
# making the neighborhood vector a copy of that single overlapping cell. Clamping
# d to this floor caps any single neighbor's weight at 1/1um while leaving normal
# (>1um) separations essentially untouched.
MIN_NEIGHBOR_MICRON = 1.0


class SpatialDataset(Dataset):
    """Cell + neighborhood-aggregated feature pairs, computed per sample.

    Parameters
    ----------
    cell_features
        ``(n_cells, n_features)`` dense ndarray or scipy sparse matrix.
    spatial_coords
        ``(n_cells, 2)`` physical coordinates in microns.
    sample_ids
        ``(n_cells,)`` string label per cell. The spatial graph is computed
        within sample only; cells from different samples never become neighbors.
    k_neighbors
        Neighbors per cell (excluding self) for the ``"knn"`` graph, and the
        cap on neighbors kept for ``"radius"`` / ``"delaunay"``. If a sample has
        fewer than ``k_neighbors + 1`` cells, all available cells are used. A
        single-cell sample falls back to duplicating the cell's own vector.
    neighborhood_aggregation
        One of ``{"mean", "weighted_mean", "max", "gaussian", "inverse_square"}``.
        ``"weighted_mean"`` uses ``1 / (distance + epsilon)`` weights,
        ``"inverse_square"`` uses ``1 / (distance ** 2 + epsilon)``, and
        ``"gaussian"`` uses ``exp(-distance ** 2 / (2 * bandwidth ** 2))``.
    spatial_graph
        One of ``{"knn", "radius", "delaunay", "alpha_complex"}``. ``"knn"``
        (default) keeps the ``k_neighbors`` nearest cells. ``"radius"`` keeps
        every cell within ``radius`` microns (capped at ``k_neighbors``).
        ``"delaunay"`` uses the Delaunay triangulation adjacency;
        ``"alpha_complex"`` additionally prunes long triangulation edges. Both
        fall back to a ``"knn"`` graph for samples with fewer than three cells or
        degenerate (collinear) coordinates.
    radius
        Neighborhood radius in microns; required when ``spatial_graph="radius"``.
    bandwidth
        Gaussian kernel bandwidth in microns; required when
        ``neighborhood_aggregation="gaussian"``.
    agg_chunk
        Cells per vectorized aggregation chunk. Bounds peak memory. Defaults to
        100,000.

    Raises
    ------
    ValueError
        If ``neighborhood_aggregation`` / ``spatial_graph`` is unsupported, if
        input shapes are inconsistent, if ``k_neighbors`` is non-positive, if
        ``spatial_graph="radius"`` without a positive ``radius``, or if
        ``neighborhood_aggregation="gaussian"`` without a positive ``bandwidth``.

    Notes
    -----
    Determinism. The neighbor search produces a deterministic ordering given
    deterministic input: the index into ``sample_ids`` defines the row ordering
    inside each per-sample subset, ``np.unique`` returns samples in sorted
    order, and ties at the boundary distance are broken in input-row order.
    """

    def __init__(
        self,
        cell_features: Any,
        spatial_coords: Any,
        sample_ids: Any,
        k_neighbors: int = 20,
        neighborhood_aggregation: str = "weighted_mean",
        spatial_graph: str = "knn_radius",
        radius: float | None = 50.0,
        bandwidth: float | None = None,
        agg_chunk: int = _DEFAULT_AGG_CHUNK,
        recon_target: Any | None = None,
    ) -> None:
        if hasattr(cell_features, "toarray"):
            cell_features = cell_features.toarray()
        self.cell_features = torch.as_tensor(np.asarray(cell_features), dtype=torch.float32)
        self.spatial_coords = np.asarray(spatial_coords, dtype=np.float64)
        self.sample_ids = np.asarray(sample_ids)
        self.k_neighbors = int(k_neighbors)
        self.neighborhood_aggregation = neighborhood_aggregation
        self.spatial_graph = spatial_graph
        self.radius = radius
        self.bandwidth = bandwidth
        self.agg_chunk = int(agg_chunk)
        if self.k_neighbors <= 0:
            raise ValueError(f"k_neighbors must be positive, got {self.k_neighbors}")
        if self.agg_chunk <= 0:
            raise ValueError(f"agg_chunk must be positive, got {self.agg_chunk}")
        if neighborhood_aggregation not in _VALID_AGGREGATIONS:
            raise ValueError(
                f"Unknown aggregation {neighborhood_aggregation!r}. "
                f"Choose one of {sorted(_VALID_AGGREGATIONS)}."
            )
        if spatial_graph not in _VALID_GRAPHS:
            raise ValueError(
                f"Unknown spatial_graph {spatial_graph!r}. Choose one of {sorted(_VALID_GRAPHS)}."
            )
        if spatial_graph == "radius" and not (radius and radius > 0):
            raise ValueError("spatial_graph='radius' requires a positive `radius` (in microns).")
        if neighborhood_aggregation == "gaussian" and not (bandwidth and bandwidth > 0):
            raise ValueError(
                "neighborhood_aggregation='gaussian' requires a positive `bandwidth` (microns)."
            )
        n_cells = self.cell_features.shape[0]
        if self.spatial_coords.shape[0] != n_cells:
            raise ValueError(
                f"spatial_coords has {self.spatial_coords.shape[0]} rows but "
                f"cell_features has {n_cells}"
            )
        if self.spatial_coords.ndim != 2 or self.spatial_coords.shape[1] != 2:
            raise ValueError(
                f"spatial_coords must be (n_cells, 2), got shape {self.spatial_coords.shape}"
            )
        if self.sample_ids.shape[0] != n_cells:
            raise ValueError(
                f"sample_ids has {self.sample_ids.shape[0]} entries but cell_features has {n_cells}"
            )
        if n_cells == 0:
            raise ValueError("SpatialDataset received zero cells; cannot build a dataset.")
        # Optional per-cell raw-count reconstruction target. Stored, NOT returned
        # by __getitem__ (which keeps its 3-tuple contract); the trainer indexes it
        # by batch_idx so a count likelihood (NB / Poisson) can be evaluated on the
        # raw counts while the encoder input (self.cell_features) stays the log1p
        # expression. Stays None on the released path (byte-identical behavior).
        self.recon_target: torch.Tensor | None = None
        if recon_target is not None:
            if hasattr(recon_target, "toarray"):
                recon_target = recon_target.toarray()
            self.recon_target = torch.as_tensor(np.asarray(recon_target), dtype=torch.float32)
            if self.recon_target.shape != self.cell_features.shape:
                raise ValueError(
                    f"recon_target shape {tuple(self.recon_target.shape)} != cell_features shape "
                    f"{tuple(self.cell_features.shape)}"
                )
        self.neighborhood_features = self._compute_neighborhood_features()

    @classmethod
    def from_anndata(
        cls,
        adata: Any,
        sample_col: str = "sample_id",
        spatial_key: str = "spatial",
        x_col: str | None = None,
        y_col: str | None = None,
        coord_scale: float = 1.0,
        transcript_context_key: str | None = None,
        recon_target_layer: str | None = None,
        **kwargs: Any,
    ) -> SpatialDataset:
        """Build a dataset from an AnnData of any imaging spatial-transcriptomics platform.

        Coordinates are taken from ``obsm[spatial_key]`` when present, otherwise from
        the ``obs`` columns ``x_col`` / ``y_col``, and multiplied by ``coord_scale`` to
        convert to microns (e.g. CosMx pixels use ``coord_scale=0.12028``). If
        ``sample_col`` is absent, all cells are treated as a single sample.

        Parameters
        ----------
        adata
            AnnData with a cell-by-gene ``X`` and spatial coordinates.
        sample_col
            ``obs`` column naming the per-sample (or per-FOV) unit.
        spatial_key
            ``obsm`` key holding an ``(n_cells, 2)`` coordinate array.
        x_col, y_col
            Fallback ``obs`` coordinate columns used when ``obsm[spatial_key]`` is absent.
        coord_scale
            Multiplier converting the stored coordinates to microns.
        transcript_context_key
            If given, concatenate ``obsm[transcript_context_key]`` (segmentation-free
            transcript-context features from :func:`~nicheverse.data.transcript_context`)
            onto ``X`` to form the cell features. The model ``input_dim`` must then be
            ``X.shape[1] + obsm[key].shape[1]``.
        recon_target_layer
            If given, take the per-cell raw-count reconstruction target from
            ``adata.layers[recon_target_layer]`` (shape must match ``X``). Used by
            the count-recon cell modes (``cell_recon`` in ``{"nb","poisson","both"}``)
            so the encoder can see log1p ``X`` while the likelihood models the raw
            counts. ``None`` (default) leaves the dataset with no separate recon
            target (MSE reconstructs the input).
        **kwargs
            Forwarded to :class:`SpatialDataset` (``k_neighbors``, ``spatial_graph``, ...).
        """
        if sample_col in adata.obs:
            samples = adata.obs[sample_col].astype(str).to_numpy()
        else:
            samples = np.full(adata.n_obs, "sample0", dtype=object)
        if spatial_key in adata.obsm:
            coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)[:, :2]
        elif x_col and y_col and x_col in adata.obs and y_col in adata.obs:
            coords = np.column_stack(
                [adata.obs[x_col].to_numpy(), adata.obs[y_col].to_numpy()]
            ).astype(np.float64)
        else:
            raise ValueError(
                f"No coordinates found: provide obsm['{spatial_key}'] or obs x/y columns."
            )
        cell_feats: Any = adata.X
        if transcript_context_key is not None:
            if transcript_context_key not in adata.obsm:
                raise ValueError(
                    f"transcript_context_key={transcript_context_key!r} not in adata.obsm; "
                    "compute it with nicheverse.data.transcript_context first."
                )
            xc = adata.X
            if hasattr(xc, "toarray"):
                xc = xc.toarray()
            tc = np.asarray(adata.obsm[transcript_context_key], dtype=np.float32)
            cell_feats = np.concatenate([np.asarray(xc, dtype=np.float32), tc], axis=1)
        recon_target = None
        if recon_target_layer is not None:
            if recon_target_layer not in adata.layers:
                raise ValueError(
                    f"recon_target_layer={recon_target_layer!r} not in adata.layers "
                    f"(have {list(adata.layers.keys())})."
                )
            recon_target = adata.layers[recon_target_layer]
        return cls(
            cell_feats,
            coords * float(coord_scale),
            samples,
            recon_target=recon_target,
            **kwargs,
        )

    def _knn_graph(self, coords: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Plain k-nearest-neighbor graph (also the fallback for degenerate samples).

        Uses the exact kNN backend (GPU cuML brute force when available, else the
        sklearn ball-tree); both return the same neighbor set, self at column 0.
        """
        k = min(self.k_neighbors + 1, n)
        dist, idx = knn_query(coords, k)
        return idx, dist

    def _neighbor_graph(self, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(idx, dist)`` neighbor arrays of shape ``(n, k)`` for one sample.

        Column 0 is the cell itself (distance 0). Padded slots (variable-degree
        graphs) use index ``-1`` and distance ``inf`` so aggregation can mask them.
        """
        n = coords.shape[0]
        if self.spatial_graph == "knn":
            return self._knn_graph(coords, n)
        if self.spatial_graph == "radius":
            # Exact fixed-radius query (sklearn ball-tree; cuML has no exact
            # radius search, so this path stays on CPU). Returns per-cell
            # neighbor lists sorted by distance, self at distance 0.
            dist_list, idx_list = radius_query(coords, float(self.radius))
            dist_list = list(dist_list)
            idx_list = list(idx_list)
            # Isolated-cell fallback (same rationale as knn_radius below): a cell
            # with no neighbor inside `radius` gets a self-only list, which would
            # aggregate to an all-zeros niche. Inject its single nearest real
            # neighbor so the niche reflects the closest tissue context.
            # Deterministic: nearest neighbor is a function of coordinates. The
            # k=2 kNN is computed once for all cells (exact backend); column 1 is
            # the nearest non-self cell.
            if n >= 2:
                isolated = [i for i in range(n) if not (np.asarray(idx_list[i]) != i).any()]
                if isolated:
                    d2, i2 = knn_query(coords, 2)  # (n, 2): col1 = nearest non-self
                    for i in isolated:
                        j = int(i2[i, 1])
                        idx_list[i] = np.array([i, j])
                        dist_list[i] = np.array([0.0, float(d2[i, 1])])
            return self._pad_variable_degree(idx_list, dist_list, n)
        if self.spatial_graph == "knn_radius":
            # bounded-degree kNN pruned to a distance band: keep up to k neighbors
            # that also lie within `radius` microns. Drops the long edges plain kNN
            # forms across tissue gaps while keeping a bounded, dense degree.
            if self.radius is None:
                raise ValueError("spatial_graph='knn_radius' requires radius (microns).")
            idx, dist = self._knn_graph(coords, n)
            idx, dist = idx.copy(), dist.copy()
            far = dist > float(self.radius)
            far[:, 0] = False  # never drop the self column
            # Isolated-cell fallback: for a cell whose only in-radius entry is
            # itself (every real neighbor is beyond `radius`), keep its single
            # NEAREST real neighbor instead of dropping to an all-zeros niche.
            # Column 1 of the kNN result is the nearest non-self cell (present
            # whenever n>=2, which holds here since single-cell samples short
            # circuit earlier). Nearest-neighbor fallback is preferred over
            # self-duplication for a niche tokenizer: it grounds the niche in the
            # closest real tissue context rather than fabricating a self-only
            # neighborhood, so tissue-edge / low-density cells cannot collapse
            # onto a single spurious empty-niche code. This preserves determinism
            # (column 1 is a deterministic function of the coordinates) and does
            # not touch cells that already have >=1 in-radius neighbor.
            if n >= 2:
                isolated = ~(~far[:, 1:]).any(axis=1)  # no real neighbor within radius
                far[isolated, 1] = False
            idx[far] = -1
            dist[far] = np.inf
            return idx, dist
        # delaunay / alpha_complex / gabriel / rng: start from the Delaunay
        # triangulation adjacency, optionally filter edges by a proximity predicate,
        # then keep the k nearest kept neighbors. Degenerate samples (< 3 cells or
        # collinear points, where Qhull fails) fall back to a knn graph.
        from scipy.spatial import Delaunay

        if n < 3:
            return self._knn_graph(coords, n)
        try:
            tri = Delaunay(coords)
        except QhullError:
            return self._knn_graph(coords, n)
        adj: list[set[int]] = [set() for _ in range(n)]
        for simplex in tri.simplices:
            for a in simplex:
                for b in simplex:
                    if a != b:
                        adj[a].add(int(b))
        if self.spatial_graph in ("gabriel", "rng"):
            adj = self._proximity_filter(adj, coords, self.spatial_graph)
        if self.spatial_graph == "alpha_complex":
            halves = np.array(
                [
                    0.5 * float(np.linalg.norm(coords[i] - coords[j]))
                    for i in range(n)
                    for j in adj[i]
                    if j > i
                ]
            )
            if halves.size:
                alpha = float(np.median(halves) + halves.std())
                for i in range(n):
                    adj[i] = {
                        j for j in adj[i] if 0.5 * np.linalg.norm(coords[i] - coords[j]) <= alpha
                    }
        idx_list, dist_list = [], []
        for i in range(n):
            nb = np.fromiter(adj[i], dtype=np.int64) if adj[i] else np.empty(0, dtype=np.int64)
            d = np.linalg.norm(coords[nb] - coords[i], axis=1) if nb.size else np.empty(0)
            order = np.argsort(d, kind="stable")[: self.k_neighbors]
            idx_list.append(np.concatenate([[i], nb[order]]))
            dist_list.append(np.concatenate([[0.0], d[order]]))
        return self._pad_variable_degree(idx_list, dist_list, n, self_included=True)

    def _proximity_filter(
        self, adj: list[set[int]], coords: np.ndarray, mode: str
    ) -> list[set[int]]:
        """Filter Delaunay edges down to the Gabriel graph or the Relative
        Neighborhood Graph (RNG).

        Gabriel: keep edge ``(i, j)`` iff the open disk with diameter ``ij`` contains
        no other point. RNG: keep iff the lune of ``i`` and ``j`` is empty, i.e. no
        point ``k`` has ``d(i,k) < d(i,j)`` and ``d(j,k) < d(i,j)``. Both are exact
        subgraphs of the Delaunay triangulation, so filtering Delaunay edges with a
        global witness test (via a KD-tree) yields the exact graph. RNG is the
        sparsest, then Gabriel, then Delaunay.
        """
        from scipy.spatial import cKDTree

        tree = cKDTree(coords)
        edges = {(i, j) for i in range(len(adj)) for j in adj[i] if i < j}
        kept: list[set[int]] = [set() for _ in range(len(adj))]
        for i, j in edges:
            pi, pj = coords[i], coords[j]
            dij = float(np.linalg.norm(pi - pj))
            if dij == 0.0:
                keep = True
            elif mode == "gabriel":
                m = 0.5 * (pi + pj)
                w = tree.query_ball_point(m, 0.5 * dij * (1.0 - 1e-9))
                keep = all(x == i or x == j for x in w)
            else:  # rng
                a = set(tree.query_ball_point(pi, dij * (1.0 - 1e-9)))
                b = set(tree.query_ball_point(pj, dij * (1.0 - 1e-9)))
                keep = not ((a & b) - {i, j})
            if keep:
                kept[i].add(j)
                kept[j].add(i)
        return kept

    def _pad_variable_degree(
        self,
        idx_list: list[np.ndarray],
        dist_list: list[np.ndarray],
        n: int,
        self_included: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pad ragged neighbor lists to a dense ``(n, k)`` array with ``-1`` / ``inf`` padding."""
        cap = self.k_neighbors + 1
        widths = [len(a) if self_included else min(len(a), cap) for a in idx_list]
        k = max(2, min(cap, max(widths) if widths else 2))
        idx = np.full((n, k), -1, dtype=np.int64)
        dist = np.full((n, k), np.inf, dtype=np.float64)
        for i in range(n):
            ii, dd = np.asarray(idx_list[i]), np.asarray(dist_list[i])
            # Ensure self is column 0 exactly once. radius_neighbors returns self at
            # distance 0, but coincident cells can displace or duplicate it.
            if not self_included:
                keep = ii != i
                ii = np.concatenate([[i], ii[keep]])
                dd = np.concatenate([[0.0], dd[keep]])
            m = min(len(ii), k)
            idx[i, :m] = ii[:m]
            dist[i, :m] = dd[:m]
        return idx, dist

    def _compute_neighborhood_features(self) -> torch.Tensor:
        n_cells, feat_dim = self.cell_features.shape
        out = torch.full((n_cells, feat_dim * 2), float("nan"), dtype=torch.float32)
        # Count-scale aggregated-neighbor composition, built ONLY when a raw-count
        # recon_target is present (count / Dirichlet-multinomial modes). It applies
        # the SAME neighbor aggregation (same graph, same indices, same weights) as
        # the log1p neighborhood half, but to the RAW counts, so the DM likelihood
        # sees a count-scale composition whose row sum is a meaningful transcript
        # total (not a sum of log1p values). None otherwise.
        make_counts = self.recon_target is not None
        self.niche_count_target: torch.Tensor | None = (
            torch.full((n_cells, feat_dim), float("nan"), dtype=torch.float32)
            if make_counts
            else None
        )
        for sample in np.unique(self.sample_ids):
            sidx = np.where(self.sample_ids == sample)[0]
            n_in_sample = len(sidx)
            if n_in_sample == 0:
                continue
            self_feats = self.cell_features[sidx]
            if n_in_sample == 1:
                out[sidx[0]] = torch.cat([self_feats[0], self_feats[0]])
                if make_counts:
                    # Mirror the log1p single-cell fallback (neighbor half = self):
                    # the count-scale neighbor composition is the cell's own counts.
                    self.niche_count_target[sidx[0]] = self.recon_target[sidx[0]]
                continue
            idx, dist = self._neighbor_graph(self.spatial_coords[sidx])
            agg = self._aggregate(self_feats, idx, dist)
            out[torch.as_tensor(sidx)] = torch.cat([self_feats, agg], dim=1)
            if make_counts:
                # Reuse the identical (idx, dist): re-run only the aggregation on the
                # raw counts. No kNN / graph rebuild.
                agg_counts = self._aggregate(self.recon_target[sidx], idx, dist)
                self.niche_count_target[torch.as_tensor(sidx)] = agg_counts
        if torch.isnan(out).any():
            n_bad = int(torch.isnan(out).any(dim=1).sum().item())
            raise RuntimeError(
                f"Internal error: {n_bad} cells did not receive a neighborhood feature. "
                "This is a bug; please file an issue with input shape information."
            )
        if make_counts and torch.isnan(self.niche_count_target).any():
            raise RuntimeError(
                "Internal error: some cells did not receive a niche_count_target. This is a bug."
            )
        return out

    def _aggregate(self, feats: torch.Tensor, idx: np.ndarray, dist: np.ndarray) -> torch.Tensor:
        """Vectorized, chunked neighbor aggregation (excludes the cell itself and padded slots)."""
        n, feat_dim = feats.shape
        nbr_idx = torch.as_tensor(idx, dtype=torch.long)  # (n, k) including the self column
        nbr_dist = torch.as_tensor(dist, dtype=torch.float32)  # (n, k)
        own = torch.arange(n).unsqueeze(1)
        valid = (nbr_idx >= 0) & (nbr_idx != own)  # drop padded slots and the cell itself (by index)
        safe_idx = nbr_idx.clamp_min(0)
        out = torch.empty((n, feat_dim), dtype=torch.float32)
        for start in range(0, n, self.agg_chunk):
            end = min(start + self.agg_chunk, n)
            gi = safe_idx[start:end]  # (c, k)
            gathered = feats[gi]  # (c, k, F)
            mask = valid[start:end].unsqueeze(-1).float()  # (c, k, 1)
            gathered = gathered * mask
            if self.neighborhood_aggregation == "max":
                neg_inf = float("-inf")
                masked = torch.where(mask.bool(), gathered, torch.full_like(gathered, neg_inf))
                res = masked.max(dim=1).values
                res[torch.isinf(res)] = 0.0
                out[start:end] = res
                continue
            d = nbr_dist[start:end]
            vmask = valid[start:end].float()
            agg = self.neighborhood_aggregation
            if agg == "weighted_mean":
                # Floor the distance (not a numeric epsilon) so a coincident or
                # near-coincident centroid cannot get an astronomically large
                # reciprocal weight and swamp the mean. See MIN_NEIGHBOR_MICRON.
                dd = d.clamp_min(MIN_NEIGHBOR_MICRON)
                w = (1.0 / dd) * vmask
            elif agg == "inverse_square":
                dd = d.clamp_min(MIN_NEIGHBOR_MICRON)
                w = (1.0 / (dd * dd)) * vmask
            elif agg == "gaussian":
                bw = float(self.bandwidth)
                w = torch.exp(-(d * d) / (2.0 * bw * bw)) * vmask
            else:  # mean
                w = vmask
            wsum = w.sum(dim=1, keepdim=True).clamp_min(_EPS)
            w = (w / wsum).unsqueeze(-1)  # (c, k, 1)
            out[start:end] = (gathered * w).sum(dim=1)
        return out

    def __len__(self) -> int:
        return int(self.cell_features.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.cell_features[idx], self.neighborhood_features[idx], idx


def read_spatial(
    source: Any,
    sample_col: str = "sample_id",
    spatial_key: str = "spatial",
    x_col: str | None = None,
    y_col: str | None = None,
    coord_scale: float = 1.0,
    copy: bool = False,
) -> Any:
    """Standardize an AnnData from any imaging spatial-transcriptomics platform.

    Accepts an AnnData or an ``.h5ad`` path and returns an AnnData ready for
    :func:`~nicheverse.train_model`, :class:`~nicheverse.Trainer`, or
    :func:`~nicheverse.predict_codes`. It ensures ``obsm[spatial_key]`` holds
    micron coordinates (building it from ``obs[x_col]`` / ``obs[y_col]`` and
    scaling by ``coord_scale`` when needed) and that ``obs[sample_col]`` exists.

    Parameters
    ----------
    source
        An AnnData or a path to an ``.h5ad`` file.
    sample_col
        ``obs`` column naming the per-sample (or per-FOV) unit; created as a
        single sample if absent.
    spatial_key
        ``obsm`` key for the ``(n_cells, 2)`` micron coordinates.
    x_col, y_col
        ``obs`` coordinate columns used to build ``obsm[spatial_key]`` when absent.
    coord_scale
        Multiplier converting coordinates to microns (e.g. CosMx pixels use ``0.12028``).
    copy
        If ``True`` and ``source`` is already an AnnData, operate on a copy.

    Returns
    -------
    anndata.AnnData
        The standardized AnnData.
    """
    import anndata as ad

    if isinstance(source, (str, Path)):
        adata = ad.read_h5ad(source)
    else:
        adata = source.copy() if copy else source
    if spatial_key not in adata.obsm:
        if x_col and y_col and x_col in adata.obs and y_col in adata.obs:
            adata.obsm[spatial_key] = np.column_stack(
                [adata.obs[x_col].to_numpy(), adata.obs[y_col].to_numpy()]
            ).astype(np.float64)
        else:
            raise ValueError(
                f"No coordinates: provide obsm['{spatial_key}'] or obs x/y columns (x_col, y_col)."
            )
    if coord_scale != 1.0:
        adata.obsm[spatial_key] = (
            np.asarray(adata.obsm[spatial_key], dtype=np.float64) * float(coord_scale)
        )
    if sample_col not in adata.obs:
        adata.obs[sample_col] = "sample0"
    return adata
