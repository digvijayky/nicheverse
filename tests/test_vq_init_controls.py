"""Ablation controls on the default VectorQuantizer: k-means++ seeding can be
switched off (uniform random init, van den Oord 2017) and dead code resets can
be disabled with ``dead_code_reset_interval=0``. Both defaults stay unchanged and
both flags travel through ``ModelConfig.quantizer_kwargs`` and the checkpoint.
"""
import pytest
import torch

from nicheverse import ModelConfig
from nicheverse.models import HierarchicalVQVAE, VectorQuantizer
from nicheverse.models.quantizers import DEAD_CODE_RESET_INTERVAL


def _x(n=64, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, 1, generator=g)


def test_defaults_unchanged():
    vq = VectorQuantizer(16, 8, 0.25)
    assert vq.kmeans_init is True
    assert vq.dead_code_reset_interval == DEAD_CODE_RESET_INTERVAL == 50
    assert vq.use_ema is True and vq.ema_decay == 0.99


def test_kmeans_init_seeds_codebook_from_data_by_default():
    torch.manual_seed(0)
    vq = VectorQuantizer(16, 8, 0.25, use_ema=False)  # no EMA refresh, so only init moves weights
    w0 = vq.embedding.weight.detach().clone()
    vq.train(); vq(_x())
    assert bool(vq._initialized)
    assert not torch.allclose(vq.embedding.weight, w0)


def test_kmeans_init_off_keeps_uniform_random_codebook():
    torch.manual_seed(0)
    vq = VectorQuantizer(16, 8, 0.25, use_ema=False, kmeans_init=False)
    w0 = vq.embedding.weight.detach().clone()
    vq.train(); vq(_x())
    assert bool(vq._initialized)
    torch.testing.assert_close(vq.embedding.weight, w0)


@pytest.mark.parametrize("interval,expected_calls", [(50, 2), (0, 0)])
def test_dead_code_reset_interval_zero_disables_resets(interval, expected_calls):
    torch.manual_seed(0)
    vq = VectorQuantizer(16, 8, 0.25, dead_code_reset_interval=interval)
    calls = []
    vq._reset_dead_codes = lambda flat: calls.append(1)  # count scheduling, not effect
    vq.train()
    for i in range(100):
        vq(_x(seed=i))
    assert len(calls) == expected_calls
    assert int(vq.update_count) == 100


def test_negative_reset_interval_rejected():
    with pytest.raises(ValueError, match="dead_code_reset_interval"):
        VectorQuantizer(16, 8, 0.25, dead_code_reset_interval=-1)


def test_flags_reach_both_codebooks_via_model_config_and_survive_roundtrip():
    cfg = ModelConfig(input_dim=20, hidden_dims=(16,), cell_embedding_dim=8,
                      cell_num_embeddings=12, neighborhood_embedding_dim=8,
                      neighborhood_num_embeddings=6,
                      quantizer_kwargs={"kmeans_init": False, "dead_code_reset_interval": 0,
                                        "ema_decay": 0.9})
    m = HierarchicalVQVAE(cfg)
    for q in (m.cell_vq, m.neighborhood_vq):
        assert q.kmeans_init is False and q.dead_code_reset_interval == 0 and q.ema_decay == 0.9
    cfg2 = ModelConfig.from_dict(cfg.to_dict())
    m2 = HierarchicalVQVAE(cfg2)
    assert m2.cell_vq.kmeans_init is False and m2.cell_vq.dead_code_reset_interval == 0
    m2.train()
    c, n = torch.randn(32, 20), torch.randn(32, 40)
    out = m2(c, n)
    assert torch.isfinite(out[2]) and torch.isfinite(out[3])
