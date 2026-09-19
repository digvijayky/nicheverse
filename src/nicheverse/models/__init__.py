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
from .sparse import SparseBag, SparseBagEncoder
from .foundation import FoundationConfig, FoundationVQVAE, load_foundation, save_foundation
from .foundation_moe import MoEConfig, MoEFoundationVQVAE, load_moe_foundation, save_moe_foundation
from .moe import DeterministicRouter, MoEDecoder, MoENicheDecoder, TopKRouter
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
    "SparseBag",
    "SparseBagEncoder",
    "FoundationConfig",
    "FoundationVQVAE",
    "load_foundation",
    "save_foundation",
    "ModelConfig",
    "RotVQ",
    "SoftVQ",
    "VectorQuantizer",
    "build_quantizer",
    "load_checkpoint",
    "register_quantizer",
    "save_checkpoint",
    "MoEConfig",
    "MoEFoundationVQVAE",
    "load_moe_foundation",
    "save_moe_foundation",
    "TopKRouter",
    "DeterministicRouter",
    "MoEDecoder",
    "MoENicheDecoder",
]
