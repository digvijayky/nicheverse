"""Molecule-set dataset for the subcellular point-cloud VQ-VAE.

Loads per-cell molecule-set shards (produced offline from the per-sample
``transcripts.parquet`` molecule tables: for each cell, the transcripts within a
radius of its centroid, as gene id + subcellular offset ``(dx, dy)``), aligns them to
an AnnData's ``obs_names``, and optionally attaches aggregated kNN neighborhood
features from :class:`~nicheverse.data.SpatialDataset` so the paired
molecule-set / neighborhood inputs feed :class:`~nicheverse.models.MoleculeSetVQVAE`.

Each shard is a compressed npz with keys ``coords`` ``(n, M, 2)``, ``gene`` ``(n, M)``
int (``n_genes`` is the padding id), ``mask`` ``(n, M)`` bool, ``comp`` ``(n, n_genes)``
log-normalized within-radius composition, and ``obs_names`` ``(n,)``. ``__getitem__``
returns ``(gene, coords, mask, comp, neigh, idx)``.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["MoleculeSetDataset"]


class MoleculeSetDataset(Dataset):
    """Per-cell molecule sets aligned to ``obs_names`` (+ optional kNN neighborhood).

    Parameters
    ----------
    obs_names
        Cell identifiers defining the row order (aligns shards to the model adata).
    shard_dir
        Directory of ``*.npz`` molecule-set shards.
    spatial_coords, sample_ids, expr_lognorm
        Passed to :class:`~nicheverse.data.SpatialDataset` to build the neighborhood
        branch when ``with_neighborhood`` is True.
    k_neighbors, neighborhood_aggregation
        Forwarded to :class:`~nicheverse.data.SpatialDataset`.
    with_neighborhood
        If True, attach ``(n, 2*n_genes)`` aggregated neighborhood features.
    """

    def __init__(
        self,
        obs_names,
        shard_dir,
        spatial_coords=None,
        sample_ids=None,
        expr_lognorm=None,
        k_neighbors: int = 20,
        neighborhood_aggregation: str = "weighted_mean",
        with_neighborhood: bool = True,
    ) -> None:
        obs_names = np.asarray(obs_names).astype(str)
        n = len(obs_names)
        self.n = n
        pos = {nm: i for i, nm in enumerate(obs_names)}
        shards = sorted(glob.glob(os.path.join(shard_dir, "*.npz")))
        if not shards:
            raise FileNotFoundError(f"no molecule-set shards in {shard_dir}")
        z0 = np.load(shards[0], allow_pickle=True)
        M = z0["coords"].shape[1]
        G = z0["comp"].shape[1]
        self.M, self.G = M, G
        self.coords = np.zeros((n, M, 2), dtype=np.float32)
        self.gene = np.full((n, M), G, dtype=np.int16)  # G == padding id; int16 saves RAM
        self.mask = np.zeros((n, M), dtype=bool)
        self.comp = np.zeros((n, G), dtype=np.float32)
        filled = np.zeros(n, dtype=bool)
        for sp in shards:
            z = np.load(sp, allow_pickle=True)
            names = z["obs_names"].astype(str)
            keep = np.fromiter((nm in pos for nm in names), dtype=bool, count=len(names))
            if not keep.any():
                continue
            ridx = np.fromiter(
                (pos[nm] for nm in names[keep]), dtype=np.int64, count=int(keep.sum())
            )
            # Guard against overlapping shards: a row already filled by an earlier
            # shard (cross-shard) OR appearing twice within this shard (intra-shard)
            # means the same obs_name maps to two molecule sets, which would silently
            # overwrite the cell's molecules with a different shard's rows (e.g. the
            # S_0069555 safe-shard overlap). Fail loudly instead of mispairing a cell.
            collided = filled[ridx].copy()
            _, first = np.unique(ridx, return_index=True)
            intra = np.ones(len(ridx), dtype=bool)
            intra[first] = False  # rows that repeat an earlier position in THIS shard
            collided |= intra
            if collided.any():
                dup_names = names[keep][collided]
                raise ValueError(
                    f"molecule-set shard {os.path.basename(sp)} re-fills "
                    f"{int(collided.sum())} obs_name(s) already loaded from a "
                    f"previous shard or duplicated within this shard (overlapping "
                    f"shards would silently pair a cell with the wrong molecules); "
                    f"first collisions: {list(dup_names[:5])}"
                )
            self.coords[ridx] = z["coords"][keep]
            self.gene[ridx] = z["gene"][keep].astype(np.int16)
            self.mask[ridx] = z["mask"][keep]
            self.comp[ridx] = z["comp"][keep]
            filled[ridx] = True
        if not filled.all():
            raise RuntimeError(
                f"molecule-set shards cover {int(filled.sum())}/{n} cells; "
                f"missing {n - int(filled.sum())}"
            )
        self.coords = torch.from_numpy(self.coords)
        self.gene = torch.from_numpy(self.gene)
        self.mask = torch.from_numpy(self.mask)
        self.comp = torch.from_numpy(self.comp)
        self.with_neighborhood = with_neighborhood
        if with_neighborhood:
            from .dataset import SpatialDataset

            sd = SpatialDataset(
                expr_lognorm,
                spatial_coords,
                sample_ids,
                k_neighbors=k_neighbors,
                neighborhood_aggregation=neighborhood_aggregation,
                spatial_graph="knn",
            )
            self.neigh = sd.neighborhood_features
        else:
            self.neigh = torch.zeros((n, 2 * G), dtype=torch.float32)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        return self.gene[i].long(), self.coords[i], self.mask[i], self.comp[i], self.neigh[i], i
