"""nicheverse.models: the spatial VQ-VAE tokenizer, encoders, and quantizers."""

from __future__ import annotations

from .encoders import ResidualMLP, TransformerEncoder, build_encoder, register_encoder
from .molecule_set import MoleculeSetEncoder, MoleculeSetVQVAE
from .quantizers import (
    BSQ,
    FSQ,
    LFQ,
    GroupedResidualVQ,
    ProductVQ,
    QINCoVQ,
    ResidualFSQ,
    ResidualVQ,
    RotVQ,
    SoftVQ,
    VectorQuantizer,
    build_quantizer,
    register_quantizer,
)
from .vqvae import HierarchicalVQVAE, ModelConfig, load_checkpoint, save_checkpoint

__all__ = [
    "BSQ",
    "FSQ",
    "LFQ",
    "ResidualMLP",
    "TransformerEncoder",
    "MoleculeSetEncoder",
    "MoleculeSetVQVAE",
    "build_encoder",
    "register_encoder",
    "GroupedResidualVQ",
    "QINCoVQ",
    "ProductVQ",
    "ResidualFSQ",
    "ResidualVQ",
    "HierarchicalVQVAE",
    "ModelConfig",
    "RotVQ",
    "SoftVQ",
    "VectorQuantizer",
    "build_quantizer",
    "load_checkpoint",
    "register_quantizer",
    "save_checkpoint",
]
