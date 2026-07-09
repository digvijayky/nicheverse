"""Canonical AnnData field names written and read by nicheverse.

Centralizing these keys keeps the package, its tests, and downstream user code
in agreement about where codebook assignments and embeddings live.
"""

from __future__ import annotations

from typing import Final


class Keys:
    """Namespaced AnnData keys (see :mod:`anndata`)."""

    #: ``obs`` column with the cell-codebook index (cell state).
    CELL_CODE: Final = "cell_codebook_idx"
    #: ``obs`` column with the neighborhood-codebook index (spatial niche).
    NEIGHBORHOOD_CODE: Final = "neighborhood_codebook_idx"
    #: ``obs`` column with the per-sample identifier used for the spatial graph.
    SAMPLE: Final = "sample_id"
    #: ``obsm`` key with the continuous per-cell embedding (pre-quantization).
    CELL_EMBEDDING: Final = "X_cell_embedding"
    #: ``obsm`` key with the continuous per-neighborhood embedding.
    NEIGHBORHOOD_EMBEDDING: Final = "X_neighborhood_embedding"
    #: ``obsm`` key with spatial coordinates in microns (x, y).
    SPATIAL: Final = "spatial"


def anndata_keys() -> dict[str, str]:
    """Return the AnnData keys nicheverse uses, as a plain ``{name: key}`` dict."""
    return {
        "cell_code": Keys.CELL_CODE,
        "neighborhood_code": Keys.NEIGHBORHOOD_CODE,
        "sample": Keys.SAMPLE,
        "cell_embedding": Keys.CELL_EMBEDDING,
        "neighborhood_embedding": Keys.NEIGHBORHOOD_EMBEDDING,
        "spatial": Keys.SPATIAL,
    }
