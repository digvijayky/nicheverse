"""nicheverse.data: spatial dataset, neighborhood featurizer, and Xenium I/O."""

from __future__ import annotations

from .dataset import SpatialDataset, read_spatial
from .molecule_set import MoleculeSetDataset
from .multipanel import MultiPanelBatch, MultiPanelSpatialDataset, PanelSpec
from .shards import ShardReader, aggregate_neighbors, write_shard
from .vocabulary import GeneVocabulary, load_mgi_orthologs
from .neighbors import spatial_neighbors
from .transcript import transcript_context
from .xenium import (
    attach_codes,
    attach_codes_to_adata,
    load_xenium_cohort,
    load_xenium_run,
    read_xenium,
    read_xenium_cohort,
)

__all__ = [
    "SpatialDataset",
    "read_spatial",
    "MoleculeSetDataset",
    "MultiPanelBatch",
    "MultiPanelSpatialDataset",
    "PanelSpec",
    "ShardReader",
    "aggregate_neighbors",
    "write_shard",
    "GeneVocabulary",
    "load_mgi_orthologs",
    "spatial_neighbors",
    "transcript_context",
    "attach_codes",
    "attach_codes_to_adata",
    "load_xenium_cohort",
    "load_xenium_run",
    "read_xenium",
    "read_xenium_cohort",
]
