"""sparse_bag encoder: registry contract, panel invariance, masked mean pooling."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from nicheverse.data.shards import to_bag
from nicheverse.models.encoders import _ENCODERS, build_encoder
from nicheverse.models.sparse import SparseBag, SparseBagEncoder


def _bag(mat: sp.csr_matrix) -> SparseBag:
    i, v, o = to_bag(mat)
    return SparseBag(torch.from_numpy(i), torch.from_numpy(v), torch.from_numpy(o))


def test_registered():
    assert "sparse_bag" in _ENCODERS


def test_dense_fallback_matches_registry_contract():
    enc = build_encoder("sparse_bag", in_dim=20, out_dim=8, hidden=(16,))
    y = enc(torch.randn(6, 20))
    assert y.shape == (6, 8) and torch.isfinite(y).all()


def test_two_bag_neighborhood_branch():
    enc = build_encoder("sparse_bag", in_dim=40, out_dim=8, hidden=(16,), vocab_size=20)
    assert enc.n_bags == 2
    y = enc(torch.randn(5, 40))
    assert y.shape == (5, 8)


def test_sparse_and_dense_pooling_agree():
    """The sparse bag and the equivalent dense weight row must pool to the same embedding."""
    torch.manual_seed(0)
    V = 12
    enc = SparseBagEncoder(V, 6, [10], dropout=0.0, vocab_size=V, use_context=False).eval()
    rng = np.random.default_rng(1)
    X = rng.poisson(1.5, size=(7, V)).astype(np.float32)
    X[0] = 0.0  # an empty cell must not produce NaN
    lib = X.sum(1, keepdims=True)
    W = np.log1p(np.divide(X * 1e4, np.maximum(lib, 1e-8), out=np.zeros_like(X), where=lib > 0))
    with torch.no_grad():
        sparse_emb, _ = enc._pool(_bag(sp.csr_matrix(X)), enc.count_embed)
        dense_emb, _ = enc._pool_dense(torch.as_tensor(W), enc.count_embed)
    assert torch.isfinite(sparse_emb).all()
    assert torch.allclose(sparse_emb, dense_emb, atol=1e-5)


def test_invariant_to_unmeasured_genes():
    """Genes a panel never measures cannot change the embedding, even if stored explicitly."""
    torch.manual_seed(0)
    V = 30
    enc = SparseBagEncoder(V, 6, [10], dropout=0.0, vocab_size=V, use_context=False).eval()
    rng = np.random.default_rng(2)
    measured = np.array([1, 4, 9, 17])
    vals = rng.poisson(2.0, size=(5, len(measured))).astype(np.float32) + 1.0
    rows = np.repeat(np.arange(5), len(measured))
    a = sp.csr_matrix((vals.ravel(), (rows, np.tile(measured, 5))), shape=(5, V))
    # the same cells with EXPLICIT zeros stored at ten unmeasured genes
    extra = np.array([0, 2, 3, 5, 6, 7, 8, 10, 11, 12])
    rows2 = np.concatenate([rows, np.repeat(np.arange(5), len(extra))])
    cols2 = np.concatenate([np.tile(measured, 5), np.tile(extra, 5)])
    vals2 = np.concatenate([vals.ravel(), np.zeros(5 * len(extra), np.float32)])
    b = sp.csr_matrix((vals2, (rows2, cols2)), shape=(5, V))
    b.sort_indices()
    assert b.nnz > a.nnz  # the explicit zeros really are stored
    with torch.no_grad():
        assert torch.allclose(enc([_bag(a)]), enc([_bag(b)]), atol=1e-6)
    # the masked mean means panel size does not set the scale of the embedding
    wide = sp.csr_matrix(rng.poisson(2.0, size=(5, 20)).astype(np.float32) + 1.0)
    wide = sp.hstack([wide, sp.csr_matrix((5, V - 20), dtype=np.float32)]).tocsr()
    with torch.no_grad():
        n_small = enc([_bag(a)]).norm(dim=1).mean().item()
        n_wide = enc([_bag(wide)]).norm(dim=1).mean().item()
    assert 0.2 < n_small / max(n_wide, 1e-8) < 5.0


def test_missing_context_vector_used():
    torch.manual_seed(0)
    V = 10
    enc = SparseBagEncoder(V, 6, [8], dropout=0.0, vocab_size=V, use_context=True).eval()
    rng = np.random.default_rng(3)
    X = sp.csr_matrix(rng.poisson(1.0, size=(4, V)).astype(np.float32))
    C = sp.csr_matrix(rng.poisson(3.0, size=(4, V)).astype(np.float32))
    has = torch.tensor([1.0, 0.0, 1.0, 0.0])
    with torch.no_grad():
        with_ctx = enc([_bag(X)], [_bag(C)], has)
        no_ctx = enc([_bag(X)], [None], None)
    assert with_ctx.shape == no_ctx.shape == (4, 6)
    assert torch.isfinite(with_ctx).all() and torch.isfinite(no_ctx).all()
    # rows whose context is masked off should behave as if no context bag were given
    with torch.no_grad():
        all_missing = enc([_bag(X)], [_bag(C)], torch.zeros(4))
    assert torch.allclose(all_missing, no_ctx, atol=1e-6)


def test_gradients_flow():
    V = 8
    enc = SparseBagEncoder(V, 4, [8], dropout=0.0, vocab_size=V, use_context=True)
    rng = np.random.default_rng(4)
    X = sp.csr_matrix(rng.poisson(1.0, size=(6, V)).astype(np.float32))
    out = enc([_bag(X)], [_bag(X)], torch.ones(6))
    out.sum().backward()
    assert enc.count_embed.weight.grad is not None
    assert torch.isfinite(enc.count_embed.weight.grad).all()
