"""The fused-codebook ablation control: one shared codebook over the
concatenated cell and niche latents, so a single index must carry both."""
import numpy as np
import torch

from nicheverse import ModelConfig
from nicheverse.models.vqvae import HierarchicalVQVAE


def _cfg(**kw):
    return ModelConfig(input_dim=32, hidden_dims=(16, 8), cell_embedding_dim=8,
                       neighborhood_embedding_dim=8, cell_num_embeddings=16,
                       neighborhood_num_embeddings=4, **kw)


def test_fused_defaults_off():
    assert ModelConfig(input_dim=4).fuse_codebooks is False


def test_fused_forward_shapes_and_identity():
    torch.manual_seed(0)
    m = HierarchicalVQVAE(_cfg(fuse_codebooks=True))
    c = torch.randn(20, 32)
    n = torch.randn(20, 64)
    cr, nr, cl, nl, ci, ni, cp, np_ = m(c, n)
    assert cr.shape == (20, 32)
    assert nr.shape == (20, 64)
    # one codebook means the two indices are the same code
    assert torch.equal(ci.reshape(-1), ni.reshape(-1))
    assert int(ci.max()) < 16
    ce, ne = m.encode(c, n)
    assert torch.equal(ce, ne)


def test_unfused_keeps_two_independent_codebooks():
    torch.manual_seed(0)
    m = HierarchicalVQVAE(_cfg(fuse_codebooks=False))
    c = torch.randn(64, 32)
    n = torch.randn(64, 64)
    ci, ni = m.encode(c, n)
    assert int(ni.max()) < 4 and int(ci.max()) < 16
    assert not hasattr(m, "fused_vq")


def test_fused_backward_runs():
    torch.manual_seed(0)
    m = HierarchicalVQVAE(_cfg(fuse_codebooks=True))
    cr, nr, cl, nl, *_ = m(torch.randn(16, 32), torch.randn(16, 64))
    (cr.square().mean() + nr.square().mean() + cl + nl).backward()
    assert any(p.grad is not None for p in m.parameters() if p.requires_grad)
