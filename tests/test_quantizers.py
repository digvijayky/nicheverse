"""Tests for the quantizer registry and the FSQ / SoftVQ / RotVQ quantizers.

The default ``quantizer_type="vq"`` must stay a plain VectorQuantizer so the
published model reproduces bit-for-bit; the alternatives are opt-in.
"""

from __future__ import annotations

import pytest
import torch

from nicheverse.models import (
    BSQ,
    FSQ,
    LFQ,
    GroupedResidualVQ,
    HierarchicalVQVAE,
    ModelConfig,
    ResidualFSQ,
    ResidualVQ,
    RotVQ,
    SoftVQ,
    VectorQuantizer,
    build_quantizer,
    load_checkpoint,
    save_checkpoint,
)

_QZ = [
    ("vq", {}),
    ("soft", {}),
    ("rot", {"num_householders": 3}),
    ("fsq", {"levels": (4, 4, 4)}),
    ("qinco", {"num_levels": 3}),
    ("pq", {"num_subspaces": 4}),
    ("rvq", {"num_stages": 3}),
    ("lfq", {}),
    ("bsq", {}),
    ("residual_fsq", {"levels": (4, 4, 4), "num_stages": 2}),
    ("grvq", {"num_groups": 2, "num_stages": 2}),
]


def _cfg(qt="vq", **kw):
    return ModelConfig(
        input_dim=20,
        hidden_dims=(16,),
        cell_embedding_dim=8,
        cell_num_embeddings=12,
        neighborhood_embedding_dim=8,
        neighborhood_num_embeddings=6,
        quantizer_type=qt,
        quantizer_kwargs=kw,
        gene_names=tuple(f"g{i}" for i in range(20)),
    )


def test_registry_names():
    from nicheverse.models.quantizers import _QUANTIZERS

    assert set(_QUANTIZERS) == {
        "vq",
        "soft",
        "rot",
        "fsq",
        "qinco",
        "pq",
        "rvq",
        "lfq",
        "bsq",
        "residual_fsq",
        "grvq",
    }


def test_build_and_forward_every_registered_quantizer():
    """Every registered quantizer builds and satisfies the (B, D, T) -> 4-tuple contract."""
    from nicheverse.models.quantizers import _QUANTIZERS

    kw = {name: dict(kwv) for name, kwv in _QZ}
    for name in sorted(_QUANTIZERS):
        torch.manual_seed(0)
        qz = build_quantizer(
            name, num_embeddings=256, embedding_dim=64, commitment_cost=0.25, **kw.get(name, {})
        )
        assert isinstance(qz.num_embeddings, int) and qz.num_embeddings > 0
        # embedding_dim is the working dim; delegate quantizers (pq/qinco/rvq/grvq)
        # report the per-subspace/per-group width, so only require it to be positive.
        assert isinstance(qz.embedding_dim, int) and qz.embedding_dim > 0
        assert isinstance(qz.distance_metric, str)
        for train in (True, False):
            qz.train(train)
            z = torch.randn(8, 64, 1)
            loss, zq, perp, idx = qz(z)
            assert zq.shape == (8, 64, 1), name
            assert idx.shape[0] == 8, name
            assert idx.dim() == 2 and idx.shape[1] == 1, name
            assert int(idx.min()) >= 0 and int(idx.max()) < qz.num_embeddings, name
            assert loss.ndim == 0 and torch.isfinite(loss), name
            assert perp.ndim == 0 and torch.isfinite(perp), name


def test_new_quantizer_build_types():
    def _b(n, **k):
        return build_quantizer(n, num_embeddings=16, embedding_dim=8, commitment_cost=0.25, **k)

    assert isinstance(_b("rvq"), ResidualVQ)
    assert isinstance(_b("grvq", num_groups=2), GroupedResidualVQ)
    assert isinstance(_b("lfq"), LFQ)
    assert isinstance(_b("bsq"), BSQ)
    assert isinstance(_b("residual_fsq"), ResidualFSQ)


def test_build_quantizer_types():
    assert type(build_quantizer("vq", num_embeddings=8, embedding_dim=4, commitment_cost=0.25)) is (
        VectorQuantizer
    )
    assert isinstance(
        build_quantizer("soft", num_embeddings=8, embedding_dim=4, commitment_cost=0.25), SoftVQ
    )
    assert isinstance(
        build_quantizer("rot", num_embeddings=8, embedding_dim=4, commitment_cost=0.25), RotVQ
    )
    assert isinstance(
        build_quantizer("fsq", num_embeddings=0, embedding_dim=4, commitment_cost=0.0), FSQ
    )


def test_build_quantizer_unknown_raises():
    with pytest.raises(ValueError, match="unknown quantizer_type"):
        build_quantizer("nope", num_embeddings=8, embedding_dim=4, commitment_cost=0.25)


@pytest.mark.parametrize("name,kw", _QZ)
def test_forward_contract(name, kw):
    torch.manual_seed(0)
    qz = build_quantizer(name, num_embeddings=16, embedding_dim=8, commitment_cost=0.25, **kw)
    x = torch.randn(7, 8, 1, requires_grad=True)
    loss, quant, perp, idx = qz(x)
    assert quant.shape == x.shape
    assert idx.shape == (7, 1)
    assert int(idx.min()) >= 0 and int(idx.max()) < qz.num_embeddings
    assert torch.isfinite(perp) and torch.isfinite(loss)
    (loss + quant.sum()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_fsq_index_is_bijective_on_even_levels():
    # even level 8 must yield all 8 distinct codes on its axis (canonical offset)
    f = FSQ(4, levels=(8, 5, 5, 5))
    z = torch.linspace(-6, 6, 500).view(-1, 1).repeat(1, 4)
    codes0 = (f._round_ste(f._bound(z))[:, 0] + f._half_width[0]).round().long()
    assert set(codes0.tolist()) == set(range(8))
    idx = f(torch.randn(2000, 4, 1))[3].squeeze(1)
    assert int(idx.min()) >= 0 and int(idx.max()) < f.num_embeddings


def test_fsq_batch_one_is_finite():
    f = FSQ(8, levels=(8, 5, 5, 5), diversity_weight=1.0)
    loss = f(torch.randn(1, 8, 1))[0]
    assert torch.isfinite(loss)


def test_fsq_code_count():
    assert FSQ(8, levels=(8, 5, 5, 5)).num_embeddings == 8 * 5 * 5 * 5


def test_proxy_quantizers_satisfy_vq_contract():
    r = build_quantizer("rot", num_embeddings=16, embedding_dim=8, commitment_cost=0.25)
    s = build_quantizer("soft", num_embeddings=16, embedding_dim=8, commitment_cost=0.25)
    for q in (r, s):
        assert q.embedding.weight.shape == (16, 8)
        assert q.num_embeddings == 16 and q.embedding_dim == 8
        assert isinstance(q.distance_metric, str)


def test_rotvq_rotation_is_orthogonal():
    r = RotVQ(16, 8, num_householders=5)
    x = torch.randn(4, 8)
    assert torch.allclose(r._reflect(r._reflect(x), reverse=True), x, atol=1e-5)


def test_default_quantizer_is_vectorquantizer():
    m = HierarchicalVQVAE(_cfg("vq"))
    assert type(m.cell_vq) is VectorQuantizer
    assert type(m.neighborhood_vq) is VectorQuantizer
    assert m.cell_vq.distance_metric == "l2"
    assert m.neighborhood_vq.distance_metric == "l2"


def test_cosine_lookup_still_selectable():
    # _cfg routes **kw into quantizer_kwargs, so build the config directly here
    cfg = ModelConfig(input_dim=20, hidden_dims=(16,), cell_embedding_dim=8,
                      cell_num_embeddings=12, neighborhood_embedding_dim=8,
                      neighborhood_num_embeddings=6, vq_distance="cosine",
                      gene_names=tuple(f"g{i}" for i in range(20)))
    m = HierarchicalVQVAE(cfg)
    assert m.cell_vq.distance_metric == "cosine"


def test_modelconfig_rejects_unknown_quantizer():
    with pytest.raises(ValueError, match="quantizer_type"):
        _cfg("bogus")


@pytest.mark.parametrize("name,kw", _QZ)
def test_model_forward_and_checkpoint_roundtrip(tmp_path, name, kw):
    torch.manual_seed(0)
    model = HierarchicalVQVAE(_cfg(name, **kw)).train()
    cb, nb = torch.randn(10, 20), torch.randn(10, 40)
    out = model(cb, nb)
    assert out[0].shape == (10, 20)
    assert out[4].reshape(-1).shape == (10,)
    ck = tmp_path / f"{name}.pt"
    save_checkpoint(model, ck)
    loaded = load_checkpoint(ck)
    assert loaded.config.quantizer_type == name
    assert type(loaded.cell_vq) is type(model.cell_vq)
