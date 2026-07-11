"""Inference: assign codebook indices to new Xenium data using a trained checkpoint."""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader

import json

from ..data import SpatialDataset
from ..data.xenium import attach_codes_to_adata
from ..models import HierarchicalVQVAE, ModelConfig, load_checkpoint
from ..utils import seed_everything
from .trainer import _preprocess

logger = logging.getLogger(__name__)

# Sentinel meaning "not passed" so that an explicit ``radius=None`` (valid for
# graphs like plain knn) can be told apart from an un-passed argument.
_INHERIT = object()

# Fallback defaults used only when a key is absent from a checkpoint's
# train_config.json (e.g. very old checkpoints). Kept in sync with TrainConfig.
_GRAPH_DEFAULTS = {
    "spatial_graph": "knn_radius",
    "radius": 50.0,
    "k_neighbors": 20,
    "neighborhood_aggregation": "weighted_mean",
}


def _load_train_graph_config(checkpoint: str | Path | HierarchicalVQVAE) -> dict:
    """Read the neighborhood-graph settings saved next to a checkpoint.

    The trainer writes ``train_config.json`` into the checkpoint directory with
    ``spatial_graph``, ``radius``, ``k_neighbors`` and ``neighborhood_aggregation``.
    Given a ``.pt`` checkpoint path, look for that JSON in the same directory.

    Returns an empty dict if the checkpoint is an in-memory model or the JSON is
    absent/unreadable; callers then fall back to :data:`_GRAPH_DEFAULTS`.
    """
    if isinstance(checkpoint, HierarchicalVQVAE):
        return {}
    cfg_path = Path(checkpoint).parent / "train_config.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text())
    except (OSError, ValueError) as exc:
        logger.warning("Could not read %s (%s); falling back to graph defaults.", cfg_path, exc)
        return {}


def _resolve_graph_arg(name: str, passed, train_cfg: dict, sentinel=None):
    """Resolve one graph argument, preferring an explicit value, then the
    checkpoint's train_config, then the package default (with a logged warning).

    ``passed`` equal to ``sentinel`` means the caller did not pass the argument
    (so inherit); any other value is honored. For ``radius`` the sentinel is
    :data:`_INHERIT` (not ``None``) so an explicit ``radius=None`` (valid for
    graphs that take no radius) is distinguishable from an un-passed argument.
    """
    if passed is not sentinel:
        return passed
    if name in train_cfg:
        logger.info(
            "predict: inheriting %s=%r from checkpoint train_config.json", name, train_cfg[name]
        )
        return train_cfg[name]
    default = _GRAPH_DEFAULTS[name]
    logger.warning(
        "predict: %s not passed and not found in checkpoint train_config.json; "
        "falling back to default %r. Neighborhood codes may not match training.",
        name,
        default,
    )
    return default


def _align_genes_to_checkpoint(adata: ad.AnnData, gene_names: tuple[str, ...]) -> ad.AnnData:
    """Reorder ``adata`` so its columns match the checkpoint gene panel exactly.

    Returns a freshly-allocated AnnData (never a view) so downstream in-place
    preprocessing does not trigger ``ImplicitModificationWarning``.

    Parameters
    ----------
    adata
        Input AnnData. Must contain every gene in ``gene_names`` as a column.
    gene_names
        Gene panel recorded in the checkpoint config.

    Raises
    ------
    ValueError
        If the checkpoint has no gene_names and ``adata`` is non-empty (we
        cannot verify panel compatibility), or if any checkpoint gene is
        missing from ``adata.var_names``.
    """
    if not gene_names:
        if adata.X.shape[1] != 0:
            raise ValueError(
                "Checkpoint has no gene_names recorded; cannot verify gene panel "
                "compatibility. Re-train with `gene_names=tuple(adata.var_names)` in "
                "ModelConfig, or pass a config explicitly to load_checkpoint."
            )
        return adata.copy()
    dup = adata.var_names[adata.var_names.duplicated()]
    if len(dup):
        raise ValueError(
            f"adata has duplicate var_names (e.g. {list(map(str, dup[:5]))}); gene alignment is "
            "ambiguous. Deduplicate the gene panel before predicting."
        )
    have = set(map(str, adata.var_names))
    missing = [g for g in gene_names if g not in have]
    if missing:
        raise ValueError(
            f"Input is missing {len(missing)} genes recorded in the checkpoint, "
            f"e.g. {missing[:8]}. Predict requires the same gene panel used at training time. "
            "Re-preprocess your input with the same gene filtering / panel selection step."
        )
    # `[:, list(...)].copy()` realizes a fresh AnnData with the correct dtype and layout.
    return adata[:, list(gene_names)].copy()


def _ensure_dense_float32(adata: ad.AnnData) -> None:
    """Convert ``adata.X`` to a dense contiguous float32 array in place.

    The dataset constructor calls ``toarray()`` if needed, but converting once
    here keeps later passes cheap and avoids surprises in user code.
    """
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    if not X.flags["C_CONTIGUOUS"]:
        X = np.ascontiguousarray(X)
    adata.X = X


def predict_codes(
    adata: ad.AnnData,
    checkpoint: str | Path | HierarchicalVQVAE,
    output_path: str | Path | None = None,
    sample_col: str = "sample_id",
    k_neighbors: int | None = None,
    neighborhood_aggregation: str | None = None,
    spatial_graph: str | None = None,
    radius: float | None = _INHERIT,
    bandwidth: float | None = None,
    batch_size: int = 2048,
    normalize: bool = True,
    log1p: bool = True,
    device: str | None = None,
    return_embeddings: bool = True,
    config: ModelConfig | None = None,
    seed: int = 9,
    deterministic: bool = True,
) -> ad.AnnData:
    """Assign cell and neighborhood codebook indices to ``adata`` using a checkpoint.

    Parameters
    ----------
    adata
        Input AnnData with raw counts (or already-normalized data, in which
        case set ``normalize=False`` and ``log1p=False``). Must carry
        ``adata.obsm['spatial']`` and ``adata.obs[sample_col]``.
    checkpoint
        Either a path to a ``.pt`` file written by
        :func:`nicheverse.train_model` or an already-loaded
        :class:`HierarchicalVQVAE`.
    output_path
        Optional ``.h5ad`` path to write the annotated AnnData. Parent
        directories are created if missing.
    sample_col
        Column in ``adata.obs`` used to partition cells for per-sample k-NN.
    k_neighbors
        Neighbors per cell. ``None`` (default) inherits the training-time value
        from the checkpoint's ``train_config.json`` so the neighborhood codebook
        behaves as intended; pass an int to override.
    neighborhood_aggregation
        One of ``{"mean", "weighted_mean", "max", "gaussian", "inverse_square"}``.
        ``None`` (default) inherits the training-time value from the checkpoint.
    spatial_graph
        Spatial-graph backend. ``None`` (default) inherits the training-time
        value from the checkpoint (e.g. ``"knn_radius"``); pass a string to
        override. Building the neighborhood graph with a different backend than
        training silently produces mismatched niche codes, so inheriting is the
        safe default.
    radius
        Radius in microns for radius-based graphs. Left un-passed it inherits
        the training-time value from the checkpoint; pass an explicit value
        (including ``None`` for graphs that take no radius) to override.
    bandwidth
        Gaussian kernel bandwidth in microns; must match training when
        ``neighborhood_aggregation="gaussian"``. ``None`` inherits the
        training-time value from the checkpoint.
    batch_size
        Inference mini-batch size.
    normalize, log1p
        Apply ``sc.pp.normalize_total`` / ``sc.pp.log1p`` before inference.
        Set both to False if your AnnData is already log-normalized.
    device
        Optional explicit device string. Defaults to CUDA if available.
    return_embeddings
        If True, write ``X_cell_embedding`` and ``X_neighborhood_embedding``
        to ``adata.obsm``.
    config
        Optional :class:`ModelConfig` for legacy bare state dict checkpoints.
    seed
        Seed for the random ops within :func:`seed_everything`.
    deterministic
        Forwarded to :func:`seed_everything`.

    Returns
    -------
    AnnData
        Annotated copy of the input with ``obs['cell_codebook_idx']`` and
        ``obs['neighborhood_codebook_idx']`` (plus embeddings in ``obsm`` if
        ``return_embeddings``).

    Raises
    ------
    ValueError
        If required obs / obsm fields are missing, or if the gene panel does
        not match the checkpoint.
    """
    seed_everything(seed, deterministic=deterministic)
    device_t = (
        torch.device(device)
        if device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    # Resolve the neighborhood-graph settings before loading the model so a
    # mismatch is caught early. Explicitly-passed values override; otherwise
    # inherit from the checkpoint's train_config.json (falling back to defaults).
    train_cfg = _load_train_graph_config(checkpoint)
    spatial_graph = _resolve_graph_arg("spatial_graph", spatial_graph, train_cfg, sentinel=None)
    radius = _resolve_graph_arg("radius", radius, train_cfg, sentinel=_INHERIT)
    k_neighbors = _resolve_graph_arg("k_neighbors", k_neighbors, train_cfg, sentinel=None)
    neighborhood_aggregation = _resolve_graph_arg(
        "neighborhood_aggregation", neighborhood_aggregation, train_cfg, sentinel=None
    )
    if bandwidth is None and train_cfg.get("bandwidth") is not None:
        bandwidth = train_cfg["bandwidth"]

    if isinstance(checkpoint, HierarchicalVQVAE):
        model = checkpoint.to(device_t)
    else:
        model = load_checkpoint(checkpoint, device_t, config=config)
    model.eval()

    if "spatial" not in adata.obsm:
        raise ValueError(
            "adata.obsm['spatial'] missing. Use load_xenium_cohort() or set an "
            "(n_cells, 2) micron coordinate array manually."
        )
    if sample_col not in adata.obs.columns:
        raise ValueError(
            f"adata.obs['{sample_col}'] missing. Pass sample_col=<your_column_name> "
            "or add the column to adata.obs."
        )

    adata = _align_genes_to_checkpoint(adata, tuple(model.config.gene_names))
    _ensure_dense_float32(adata)
    _preprocess(adata, normalize, log1p)

    _xvals = adata.X.data if hasattr(adata.X, "indptr") else np.asarray(adata.X)
    if not np.isfinite(_xvals).all():
        raise ValueError("adata.X contains non-finite values (NaN/inf); check the input counts.")

    dataset = SpatialDataset.from_anndata(
        adata,
        sample_col=sample_col,
        k_neighbors=k_neighbors,
        neighborhood_aggregation=neighborhood_aggregation,
        spatial_graph=spatial_graph,
        radius=radius,
        bandwidth=bandwidth,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    ce: list[np.ndarray] = []
    ne: list[np.ndarray] = []
    ci: list[np.ndarray] = []
    ni: list[np.ndarray] = []
    with torch.inference_mode():
        for cb, nb, _ in loader:
            cb = cb.to(device_t)
            nb = nb.to(device_t)
            if return_embeddings:
                ce.append(model.cell_encoder(cb).cpu().numpy())
                ne.append(model.neighborhood_encoder(nb).cpu().numpy())
            _, _, _, _, c_idx, n_idx, _, _ = model(cb, nb)
            ci.append(c_idx.cpu().numpy())
            ni.append(n_idx.cpu().numpy())

    cell_idx = np.concatenate(ci, 0).reshape(-1)
    neigh_idx = np.concatenate(ni, 0).reshape(-1)
    cell_emb = np.concatenate(ce, 0) if return_embeddings and ce else None
    neigh_emb = np.concatenate(ne, 0) if return_embeddings and ne else None
    attach_codes_to_adata(adata, cell_idx, neigh_idx, cell_emb, neigh_emb)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(output_path)
    return adata
