"""The two defaults changed on measured evidence: cosine codebook lookup and the
niche-branch spatial loss. These lock the values in and check both remain
overridable, since the previous behaviour must stay reachable."""
import torch

from nicheverse import ModelConfig, TrainConfig
from nicheverse.models.vqvae import HierarchicalVQVAE


def test_cosine_is_the_default_lookup():
    assert ModelConfig(input_dim=8).vq_distance == "cosine"


def test_l2_remains_available():
    assert ModelConfig(input_dim=8, vq_distance="l2").vq_distance == "l2"


def test_niche_spatial_loss_on_by_default():
    tc = TrainConfig()
    assert tc.spatial_loss_type == "graph_tv"
    assert tc.spatial_loss_weight == 0.1


def test_spatial_loss_can_be_disabled():
    tc = TrainConfig(spatial_loss_weight=0.0)
    assert tc.spatial_loss_weight == 0.0


def test_graph_tv_targets_the_niche_branch():
    from nicheverse.losses import NICHE_SPATIAL_LOSSES, SPATIAL_LOSSES
    assert "graph_tv" in SPATIAL_LOSSES
    assert "graph_tv" in NICHE_SPATIAL_LOSSES


def test_both_lookups_train_and_produce_codes():
    for dist in ("cosine", "l2"):
        torch.manual_seed(0)
        m = HierarchicalVQVAE(ModelConfig(
            input_dim=24, hidden_dims=(16, 8), cell_embedding_dim=8,
            neighborhood_embedding_dim=8, cell_num_embeddings=16,
            neighborhood_num_embeddings=4, vq_distance=dist))
        c, n = torch.randn(32, 24), torch.randn(32, 48)
        cr, nr, cl, nl, ci, ni, *_ = m(c, n)
        assert cr.shape == (32, 24) and nr.shape == (32, 48)
        assert int(ci.max()) < 16 and int(ni.max()) < 4
        (cr.square().mean() + nr.square().mean() + cl + nl).backward()
