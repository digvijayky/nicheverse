"""Tests for the MoE foundation VQ-VAE."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from nicheverse.models.foundation import FoundationConfig, FoundationVQVAE, save_foundation
from nicheverse.models.foundation_moe import (
    MoEConfig,
    MoEFoundationVQVAE,
    load_moe_foundation,
    save_moe_foundation,
)
from nicheverse.models.moe import DeterministicRouter, TopKRouter, build_platform_to_expert
from nicheverse.models.sparse import SparseBag

V = 100
B = 8
M = 20


def _random_bag(b, v):
    lens = torch.randint(3, 10, (b,))
    total = int(lens.sum())
    idx = torch.randint(0, v, (total,))
    val = torch.rand(total) * 10
    off = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)])
    return SparseBag(idx, val, off)


def _make_batch(b=B, v=V, m=M, n_plat=4):
    measured = torch.arange(m)
    return dict(
        cell_bag=_random_bag(b, v),
        nbr_bag=_random_bag(b, v),
        measured=measured,
        cell_context=_random_bag(b, v),
        nbr_context=_random_bag(b, v),
        has_context=torch.ones(b),
        platform_id=torch.randint(0, n_plat, (b,)),
        species_id=torch.zeros(b, dtype=torch.long),
        cell_target=torch.rand(b, m) * 5,
        nbr_target=torch.rand(b, m) * 5,
    )


def _cfg(**kw):
    defaults = dict(
        vocab_size=V, hidden_dims=(32, 16), cell_embedding_dim=16,
        cell_num_embeddings=32, neighborhood_embedding_dim=16,
        neighborhood_num_embeddings=8, gene_embed_dim=16, decoder_hidden=16,
        n_platforms=4, n_species=2, n_datasets=3, use_cross_attention=True,
        cross_attention_heads=2,
    )
    defaults.update(kw)
    return MoEConfig(**defaults)


class TestTopKRouter:
    def test_output_shape(self):
        r = TopKRouter(16, num_experts=4, top_k=2)
        x = torch.randn(B, 16)
        w, idx, aux = r(x)
        assert w.shape == (B, 2)
        assert idx.shape == (B, 2)
        assert aux.shape == ()
        assert (w.sum(1) - 1.0).abs().max() < 1e-5

    def test_platform_aware(self):
        r = TopKRouter(16, num_experts=4, top_k=2, use_platform=True, n_platforms=4)
        x = torch.randn(B, 16)
        pid = torch.randint(0, 4, (B,))
        w, idx, aux = r(x, pid)
        assert w.shape == (B, 2)

    def test_top_k_clamp(self):
        r = TopKRouter(16, num_experts=3, top_k=5)
        assert r.top_k == 3


class TestDeterministicRouter:
    def test_maps_platforms(self):
        r = DeterministicRouter(4, {0: 0, 1: 1, 2: 2, 3: 3})
        x = torch.randn(B, 16)
        pid = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        w, idx, aux = r(x, pid)
        assert w.shape == (B, 1)
        assert (w == 1.0).all()
        assert (idx.squeeze() == pid).all()
        assert aux.item() == 0.0


class TestBuildPlatformToExpert:
    def test_groups(self):
        platforms = ["Xenium", "CosMx", "MERFISH", "Visium", "osmFISH", "BARISTAseq"]
        mapping = build_platform_to_expert(platforms)
        assert mapping[0] != mapping[1]
        assert mapping[4] != mapping[0]


@pytest.mark.parametrize("moe_mode", ["topk", "deterministic", "hybrid"])
@pytest.mark.parametrize("moe_scope", ["decoder_full", "decoder_head", "encoder", "encoder_decoder"])
class TestMoEFoundationVQVAE:
    def test_forward(self, moe_mode, moe_scope):
        cfg = _cfg(num_experts=4, top_k=2, moe_mode=moe_mode, moe_scope=moe_scope)
        model = MoEFoundationVQVAE(cfg)
        batch = _make_batch()
        out = model(batch['cell_bag'], batch['nbr_bag'], batch['measured'],
                     cell_context=batch['cell_context'], nbr_context=batch['nbr_context'],
                     has_context=batch['has_context'], platform_id=batch['platform_id'],
                     species_id=batch['species_id'])
        assert out['cell_logits'].shape == (B, M)
        assert out['niche_self_logits'].shape == (B, M)
        assert out['niche_nbr_logits'].shape == (B, M)
        assert out['cell_idx'].shape == (B,)
        assert out['niche_idx'].shape == (B,)
        assert 'moe_aux_loss' in out

    def test_loss(self, moe_mode, moe_scope):
        cfg = _cfg(num_experts=4, top_k=2, moe_mode=moe_mode, moe_scope=moe_scope)
        model = MoEFoundationVQVAE(cfg)
        batch = _make_batch()
        out = model(batch['cell_bag'], batch['nbr_bag'], batch['measured'],
                     cell_context=batch['cell_context'], nbr_context=batch['nbr_context'],
                     has_context=batch['has_context'], platform_id=batch['platform_id'],
                     species_id=batch['species_id'])
        loss, parts = model.compute_loss(out, batch['cell_target'], batch['nbr_target'], batch['measured'])
        assert loss.isfinite()
        assert 'moe_aux' in parts
        loss.backward()

    def test_encode(self, moe_mode, moe_scope):
        cfg = _cfg(num_experts=4, top_k=2, moe_mode=moe_mode, moe_scope=moe_scope)
        model = MoEFoundationVQVAE(cfg)
        batch = _make_batch()
        codes = model.encode(batch['cell_bag'], batch['nbr_bag'], batch['measured'],
                              cell_context=batch['cell_context'], nbr_context=batch['nbr_context'],
                              has_context=batch['has_context'], platform_id=batch['platform_id'],
                              species_id=batch['species_id'])
        assert 'cell_idx' in codes
        assert 'z_cell' in codes


class TestMoESaveLoad:
    def test_roundtrip(self):
        cfg = _cfg(num_experts=4, top_k=2)
        model = MoEFoundationVQVAE(cfg)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "model.pt"
            save_moe_foundation(model, p)
            loaded = load_moe_foundation(p)
            batch = _make_batch()
            model.eval(); loaded.eval()
            with torch.no_grad():
                o1 = model(batch['cell_bag'], batch['nbr_bag'], batch['measured'],
                            platform_id=batch['platform_id'], species_id=batch['species_id'])
                o2 = loaded(batch['cell_bag'], batch['nbr_bag'], batch['measured'],
                             platform_id=batch['platform_id'], species_id=batch['species_id'])
            assert torch.allclose(o1['cell_logits'], o2['cell_logits'], atol=1e-5)


class TestFromPretrainedBase:
    @pytest.mark.parametrize("moe_scope", ["decoder_full", "decoder_head"])
    def test_init_from_base(self, moe_scope):
        base_cfg = FoundationConfig(
            vocab_size=V, hidden_dims=(32, 16), cell_embedding_dim=16,
            cell_num_embeddings=32, neighborhood_embedding_dim=16,
            neighborhood_num_embeddings=8, gene_embed_dim=16, decoder_hidden=16,
            n_platforms=4, n_species=2, n_datasets=3, use_cross_attention=True,
            cross_attention_heads=2,
        )
        base = FoundationVQVAE(base_cfg)
        with tempfile.TemporaryDirectory() as td:
            bp = Path(td) / "base.pt"
            save_foundation(base, bp)
            moe_cfg = _cfg(num_experts=4, top_k=2, moe_scope=moe_scope)
            moe_model = MoEFoundationVQVAE.from_pretrained_base(bp, moe_cfg)
            batch = _make_batch()
            out = moe_model(batch['cell_bag'], batch['nbr_bag'], batch['measured'],
                             platform_id=batch['platform_id'], species_id=batch['species_id'])
            assert out['cell_logits'].shape == (B, M)
            # Verify encoder weights match
            for p1, p2 in zip(base.cell_encoder.parameters(), moe_model.cell_encoder.parameters()):
                assert torch.equal(p1, p2)


class TestGradientFlow:
    def test_gradients_reach_router(self):
        cfg = _cfg(num_experts=4, top_k=2, moe_mode='topk')
        model = MoEFoundationVQVAE(cfg)
        batch = _make_batch()
        out = model(batch['cell_bag'], batch['nbr_bag'], batch['measured'],
                     cell_context=batch['cell_context'], nbr_context=batch['nbr_context'],
                     has_context=batch['has_context'], platform_id=batch['platform_id'],
                     species_id=batch['species_id'])
        loss, _ = model.compute_loss(out, batch['cell_target'], batch['nbr_target'], batch['measured'])
        loss.backward()
        router_gate = model.cell_decoder.router.gate.weight
        assert router_gate.grad is not None
        assert router_gate.grad.abs().sum() > 0

    def test_gradients_reach_encoder(self):
        cfg = _cfg(num_experts=4, top_k=2, moe_mode='topk')
        model = MoEFoundationVQVAE(cfg)
        batch = _make_batch()
        out = model(batch['cell_bag'], batch['nbr_bag'], batch['measured'],
                     cell_context=batch['cell_context'], nbr_context=batch['nbr_context'],
                     has_context=batch['has_context'], platform_id=batch['platform_id'],
                     species_id=batch['species_id'])
        loss, _ = model.compute_loss(out, batch['cell_target'], batch['nbr_target'], batch['measured'])
        loss.backward()
        enc_weight = model.cell_encoder.count_embed.weight
        assert enc_weight.grad is not None

    def test_conditioning_gradients(self):
        cfg = _cfg(num_experts=4, top_k=2, moe_mode='topk', condition_decoders=True)
        model = MoEFoundationVQVAE(cfg)
        batch = _make_batch()
        out = model(batch['cell_bag'], batch['nbr_bag'], batch['measured'],
                     cell_context=batch['cell_context'], nbr_context=batch['nbr_context'],
                     has_context=batch['has_context'], platform_id=batch['platform_id'],
                     species_id=batch['species_id'])
        loss, parts = model.compute_loss(out, batch['cell_target'], batch['nbr_target'], batch['measured'])
        loss.backward()
        assert model.platform_embed.weight.grad is not None
        assert model.platform_embed.weight.grad.abs().sum() > 0
        assert model.cond_to_cell.weight.grad is not None

    def test_no_conditioning(self):
        cfg = _cfg(num_experts=4, top_k=2, moe_mode='topk', condition_decoders=False)
        model = MoEFoundationVQVAE(cfg)
        assert not hasattr(model, 'platform_embed')
        batch = _make_batch()
        out = model(batch['cell_bag'], batch['nbr_bag'], batch['measured'],
                     cell_context=batch['cell_context'], nbr_context=batch['nbr_context'],
                     has_context=batch['has_context'], platform_id=batch['platform_id'],
                     species_id=batch['species_id'])
        loss, _ = model.compute_loss(out, batch['cell_target'], batch['nbr_target'], batch['measured'])
        loss.backward()

    def test_conditioning_affects_output(self):
        cfg = _cfg(num_experts=4, top_k=1, moe_mode='topk', condition_decoders=True)
        model = MoEFoundationVQVAE(cfg)
        model.eval()
        nn.init.normal_(model.platform_embed.weight, std=1.0)
        nn.init.normal_(model.cond_to_cell.weight, std=1.0)
        batch = _make_batch()
        with torch.no_grad():
            out1 = model(batch['cell_bag'], batch['nbr_bag'], batch['measured'],
                         cell_context=batch['cell_context'], nbr_context=batch['nbr_context'],
                         has_context=batch['has_context'], platform_id=torch.zeros(B, dtype=torch.long),
                         species_id=batch['species_id'])
            out2 = model(batch['cell_bag'], batch['nbr_bag'], batch['measured'],
                         cell_context=batch['cell_context'], nbr_context=batch['nbr_context'],
                         has_context=batch['has_context'], platform_id=torch.ones(B, dtype=torch.long),
                         species_id=batch['species_id'])
        assert not torch.allclose(out1['cell_logits'], out2['cell_logits'])

    def test_routing_id_overrides_platform(self):
        cfg = _cfg(num_experts=4, top_k=2, moe_mode='topk', moe_scope='decoder_full')
        model = MoEFoundationVQVAE(cfg)
        batch = _make_batch()
        tissue_ids = torch.randint(0, 8, (B,))
        out = model(batch['cell_bag'], batch['nbr_bag'], batch['measured'],
                     cell_context=batch['cell_context'], nbr_context=batch['nbr_context'],
                     has_context=batch['has_context'], platform_id=batch['platform_id'],
                     species_id=batch['species_id'], routing_id=tissue_ids)
        assert out['cell_logits'].shape == (B, M)
        loss, parts = model.compute_loss(out, batch['cell_target'], batch['nbr_target'], batch['measured'])
        assert loss.isfinite()
        loss.backward()
