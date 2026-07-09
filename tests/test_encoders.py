"""Tests for the encoder registry (the ``mlp`` default stays a plain Sequential)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from nicheverse.models import HierarchicalVQVAE, ModelConfig
from nicheverse.models.encoders import _ENCODERS, build_encoder


def _cfg(**kw):
    base = dict(
        input_dim=20,
        hidden_dims=(16,),
        cell_embedding_dim=8,
        cell_num_embeddings=8,
        neighborhood_embedding_dim=8,
        neighborhood_num_embeddings=4,
        gene_names=tuple(f"g{i}" for i in range(20)),
    )
    base.update(kw)
    return ModelConfig(**base)


def test_registry_names():
    # Released encoders must always be present.
    assert {"mlp", "residual_mlp", "transformer"} <= set(_ENCODERS)
    # Full registry after porting the real encoder backbones.
    assert set(_ENCODERS) == {
        "mlp",
        "residual_mlp",
        "transformer",
        "cnn",
        "fast_cnn",
        "deep_cnn",
        "gnn",
        "diffusion",
        "dit",
        "set_transformer",
        "perceiver_io",
        "soft_moe",
        "mlp_deep",
        "mlp_plr",
        "ft_transformer",
    }


def test_mlp_default_is_sequential():
    enc = build_encoder("mlp", in_dim=10, out_dim=4, hidden=(8,))
    assert isinstance(enc, nn.Sequential)


def test_residual_mlp_forward_and_grad():
    enc = build_encoder("residual_mlp", in_dim=10, out_dim=4, hidden=(8, 8))
    x = torch.randn(5, 10, requires_grad=True)
    y = enc(x)
    assert y.shape == (5, 4)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_build_encoder_unknown_raises():
    with pytest.raises(ValueError, match="unknown encoder_type"):
        build_encoder("nope", in_dim=4, out_dim=2, hidden=(4,))


def test_default_model_encoder_is_sequential():
    m = HierarchicalVQVAE(
        ModelConfig(
            input_dim=12,
            encoder_type="mlp",
            gene_names=tuple(f"g{i}" for i in range(12)),
        )
    )
    assert isinstance(m.cell_encoder, nn.Sequential)
    assert isinstance(m.neighborhood_encoder, nn.Sequential)


def test_modelconfig_rejects_unknown_encoder():
    with pytest.raises(ValueError, match="encoder_type"):
        _cfg(encoder_type="bad")


@pytest.mark.parametrize("et", ["mlp", "residual_mlp", "transformer"])
def test_model_forward_with_encoder_type(et):
    m = HierarchicalVQVAE(_cfg(encoder_type=et)).train()
    out = m(torch.randn(6, 20), torch.randn(6, 40))
    assert out[0].shape == (6, 20)
    assert out[4].reshape(-1).shape == (6,)
