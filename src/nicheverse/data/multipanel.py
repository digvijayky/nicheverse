"""Streaming dataset over many staged panels, for multi panel (foundation) training.

One epoch visits EVERY cell of every staged dataset exactly once. Shards are shuffled, and
cells are shuffled within a shard, but a minibatch never mixes datasets: the reconstruction
target is restricted to the genes a panel measures, so the measured mask has to be constant
within a batch. That also keeps the gathered decoder head small.

The neighborhood vector is rebuilt per minibatch from the stored neighbor lists
(:func:`~nicheverse.data.shards.aggregate_neighbors`), so no dense
``cells x k x genes`` tensor is ever materialized.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .shards import ShardReader, aggregate_neighbors, to_bag

__all__ = ["MultiPanelBatch", "MultiPanelSpatialDataset", "PanelSpec"]


class PanelSpec:
    """Static description of one staged dataset.

    Parameters
    ----------
    name
        Dataset name (also the shard subdirectory).
    shards
        Paths of the dataset's shards.
    measured
        Sorted vocabulary indices this panel measures.
    dataset_id, platform_id, species_id
        Integer ids used for logging and for decoder conditioning.
    """

    def __init__(
        self,
        name: str,
        shards: Sequence[str | Path],
        measured: np.ndarray,
        dataset_id: int,
        platform_id: int,
        species_id: int,
    ) -> None:
        self.name = name
        self.shards = [Path(p) for p in shards]
        self.measured = np.asarray(measured, dtype=np.int64)
        self.dataset_id = int(dataset_id)
        self.platform_id = int(platform_id)
        self.species_id = int(species_id)


class MultiPanelBatch(dict):
    """A minibatch. Keys are documented in :meth:`MultiPanelSpatialDataset.__iter__`."""

    def to(self, device: str | torch.device) -> MultiPanelBatch:
        """Move every tensor to ``device`` (non tensors are passed through)."""
        return MultiPanelBatch(
            {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in self.items()}
        )


def _dense_gathered(mat: sp.csr_matrix, measured: np.ndarray) -> np.ndarray:
    """``(B, len(measured))`` dense view of a CSR restricted to the measured columns."""
    return np.asarray(mat[:, measured].todense(), dtype=np.float32)


def _splits(n: int, batch_size: int, min_size: int) -> list[tuple[int, int]]:
    """Contiguous slices covering ``range(n)`` with no slice smaller than ``min_size``.

    A trailing remainder is merged into the previous slice rather than dropped, so an epoch
    still visits every cell exactly once (the only exception is a shard smaller than
    ``min_size``, which is emitted whole).
    """
    if n <= 0:
        return []
    bounds = [(s, min(s + batch_size, n)) for s in range(0, n, batch_size)]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < min_size:
        s, _ = bounds[-2]
        bounds = bounds[:-2] + [(s, n)]
    return bounds


class MultiPanelSpatialDataset(IterableDataset):
    """Iterable dataset yielding one minibatch per step, streamed from shards.

    Parameters
    ----------
    panels
        The staged datasets to stream.
    batch_size
        Target minibatch size. It is reduced for wide panels so that the dense
        ``(batch, panel size)`` reconstruction target stays under ``max_target_elements``.
    shuffle
        Shuffle the shard order and the cells inside a shard.
    seed
        Base RNG seed; the epoch number is mixed in by :meth:`set_epoch`.
    aggregation
        Neighborhood aggregation, ``"weighted_mean"`` (1/d weights, the package default) or
        ``"mean"``.
    max_target_elements
        Cap on ``batch_size * panel size``, which bounds the dense reconstruction target.
    with_context
        Emit the transcript context bags.
    drop_last_smaller_than
        Skip trailing minibatches with fewer cells than this (BatchNorm needs >= 2).
    """

    def __init__(
        self,
        panels: Sequence[PanelSpec],
        batch_size: int = 4096,
        shuffle: bool = True,
        seed: int = 0,
        aggregation: str = "weighted_mean",
        max_target_elements: int = 8_000_000,
        with_context: bool = True,
        drop_last_smaller_than: int = 2,
    ) -> None:
        self.panels = list(panels)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.aggregation = aggregation
        self.max_target_elements = int(max_target_elements)
        self.with_context = bool(with_context)
        self.drop_last_smaller_than = int(drop_last_smaller_than)
        self.epoch = 0
        self._index: list[tuple[int, int]] = [
            (pi, si) for pi, p in enumerate(self.panels) for si in range(len(p.shards))
        ]

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch so shard and cell order differ between epochs."""
        self.epoch = int(epoch)

    def panel_batch_size(self, panel: PanelSpec) -> int:
        """Minibatch size for one panel, capped so the dense target stays bounded."""
        m = max(1, len(panel.measured))
        return int(max(self.drop_last_smaller_than, min(self.batch_size, self.max_target_elements // m)))

    def __iter__(self) -> Iterator[MultiPanelBatch]:
        """Yield :class:`MultiPanelBatch` dicts.

        Keys
        ----
        ``cell_idx`` / ``cell_val`` / ``cell_off``
            Sparse bag of RAW counts over the shared vocabulary.
        ``ctx_idx`` / ``ctx_val`` / ``ctx_off`` / ``has_ctx``
            Sparse bag of the transcript context field and its per cell availability flag.
        ``nbr_idx`` / ``nbr_val`` / ``nbr_off``
            Sparse bag of the aggregated neighborhood counts, computed on the fly.
        ``nbr_ctx_idx`` / ``nbr_ctx_val`` / ``nbr_ctx_off``
            Aggregated neighborhood transcript context field.
        ``cell_target`` / ``nbr_target``
            Dense ``(B, M)`` reconstruction targets, gathered at the measured genes.
        ``measured``
            ``(M,)`` measured vocabulary indices.
        ``platform_id`` / ``species_id`` / ``dataset_id``
            ``(B,)`` conditioning ids.
        ``row``
            ``(B,)`` cell row ids local to the shard, and ``shard`` / ``dataset`` names.
        """
        order = list(self._index)
        info = get_worker_info()
        rng = np.random.default_rng(self.seed + 9973 * self.epoch)
        if self.shuffle:
            rng.shuffle(order)
        if info is not None:
            order = order[info.id :: info.num_workers]
        for pi, si in order:
            panel = self.panels[pi]
            reader = ShardReader(panel.shards[si])
            counts = reader.counts
            ctx = reader.context if self.with_context else None
            has_ctx = reader.has_context
            knn_idx, knn_dist = reader.knn_idx, reader.knn_dist
            n = reader.n_cells
            rows = rng.permutation(n) if self.shuffle else np.arange(n)
            bs = self.panel_batch_size(panel)
            measured = panel.measured
            for start, stop in _splits(n, bs, self.drop_last_smaller_than):
                sel = np.sort(rows[start:stop])
                c = counts[sel]
                nb = aggregate_neighbors(counts, knn_idx, knn_dist, sel, self.aggregation)
                out = MultiPanelBatch()
                for tag, mat in (("cell", c), ("nbr", nb)):
                    i, v, o = to_bag(mat)
                    out[f"{tag}_idx"] = torch.from_numpy(i)
                    out[f"{tag}_val"] = torch.from_numpy(v)
                    out[f"{tag}_off"] = torch.from_numpy(o)
                if ctx is not None:
                    cc = ctx[sel]
                    nc = aggregate_neighbors(ctx, knn_idx, knn_dist, sel, self.aggregation)
                    for tag, mat in (("ctx", cc), ("nbr_ctx", nc)):
                        i, v, o = to_bag(mat)
                        out[f"{tag}_idx"] = torch.from_numpy(i)
                        out[f"{tag}_val"] = torch.from_numpy(v)
                        out[f"{tag}_off"] = torch.from_numpy(o)
                    out["has_ctx"] = torch.from_numpy(has_ctx[sel].astype(np.float32))
                out["cell_target"] = torch.from_numpy(_dense_gathered(c, measured))
                out["nbr_target"] = torch.from_numpy(_dense_gathered(nb, measured))
                out["measured"] = torch.from_numpy(measured)
                b = len(sel)
                out["platform_id"] = torch.full((b,), panel.platform_id, dtype=torch.long)
                out["species_id"] = torch.full((b,), panel.species_id, dtype=torch.long)
                out["dataset_id"] = torch.full((b,), panel.dataset_id, dtype=torch.long)
                out["row"] = torch.from_numpy(sel.astype(np.int64))
                out["dataset"] = panel.name
                out["shard"] = panel.shards[si].name
                yield out

    # -- construction from a staging directory -------------------------------------------
    @classmethod
    def from_staging(
        cls,
        root: str | Path,
        datasets: Sequence[str] | None = None,
        **kwargs,
    ) -> MultiPanelSpatialDataset:
        """Build from a staging root written by the staging runner.

        The root must hold ``index.json`` with, per dataset, its shard file names, the
        measured vocabulary indices file, and the dataset / platform / species ids.
        """
        root = Path(root)
        index = json.loads((root / "index.json").read_text())
        panels = []
        for name, d in index["datasets"].items():
            if datasets is not None and name not in datasets:
                continue
            measured = np.load(root / name / "measured.npy")
            panels.append(
                PanelSpec(
                    name,
                    [root / name / s for s in d["shards"]],
                    measured,
                    d["dataset_id"],
                    d["platform_id"],
                    d["species_id"],
                )
            )
        return cls(panels, **kwargs)
