"""Nicheverse: a hierarchical VQ-VAE that tokenizes imaging-based spatial transcriptomics.

Nicheverse learns two coupled discrete codebooks, a cell codebook of transcriptional
states and a neighborhood (niche) codebook of multicellular contexts, joined by a
one-directional gated cross-attention block, and assigns every cell an interpretable
cell-state code and niche code.

Subpackages:

- :mod:`nicheverse.models`   -- the model, encoders, quantizers
- :mod:`nicheverse.data`     -- the spatial dataset, neighbor featurizer, Xenium I/O
- :mod:`nicheverse.training` -- the Trainer, training loop, and inference
- :mod:`nicheverse.plotting` -- vector PDF figures
- :mod:`nicheverse.utils`    -- determinism and reproducibility helpers

Common entry points are re-exported at the top level::

    import nicheverse as nv
    adata = nv.read_xenium_cohort(["run_A", "run_B"])
    model, adata = nv.Trainer(nv.TrainConfig()).fit(adata, "ckpt/")
    annotated = nv.predict_codes(new_adata, "ckpt/hierarchical_vqvae_checkpoint.pt")
"""

from __future__ import annotations

import logging as _logging
from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _version

_logging.getLogger(__name__).addHandler(_logging.NullHandler())

from . import data, losses, models, plotting, training, utils
from .annotate import annotate_codes, code_evidence
from .constants import Keys, anndata_keys
from .data import (
    SpatialDataset,
    attach_codes,
    attach_codes_to_adata,
    load_xenium_cohort,
    load_xenium_run,
    read_spatial,
    read_xenium,
    read_xenium_cohort,
    spatial_neighbors,
)
from .models import (
    HierarchicalVQVAE,
    ModelConfig,
    VectorQuantizer,
    load_checkpoint,
    save_checkpoint,
)
from .training import TrainConfig, Trainer, mae_pretrain, predict_codes, train_model

try:
    __version__ = _version("nicheverse")
except _PkgNotFound:  # pragma: no cover - source tree without install metadata
    __version__ = "0.0.0+unknown"

__all__ = [
    "data",
    "losses",
    "models",
    "plotting",
    "training",
    "utils",
    "HierarchicalVQVAE",
    "ModelConfig",
    "VectorQuantizer",
    "save_checkpoint",
    "load_checkpoint",
    "Trainer",
    "TrainConfig",
    "train_model",
    "predict_codes",
    "mae_pretrain",
    "SpatialDataset",
    "annotate_codes",
    "code_evidence",
    "read_spatial",
    "spatial_neighbors",
    "load_xenium_run",
    "load_xenium_cohort",
    "read_xenium",
    "read_xenium_cohort",
    "attach_codes",
    "attach_codes_to_adata",
    "seed_everything",
    "env_snapshot",
    "write_env_snapshot",
    "sha256_array",
    "sha256_file",
    "Keys",
    "anndata_keys",
    "__version__",
]

from .utils import (
    env_snapshot,
    seed_everything,
    sha256_array,
    sha256_file,
    write_env_snapshot,
)
