import torch

from nicheverse.models import HierarchicalVQVAE, ModelConfig, load_checkpoint, save_checkpoint


def test_forward_shapes(tmp_path):
    cfg = ModelConfig(
        input_dim=20,
        hidden_dims=(32, 16),
        cell_embedding_dim=8,
        cell_num_embeddings=16,
        neighborhood_embedding_dim=16,
        neighborhood_num_embeddings=8,
        gene_names=tuple(f"g{i}" for i in range(20)),
    )
    model = HierarchicalVQVAE(cfg).train()
    cb = torch.randn(64, 20)
    nb = torch.randn(64, 40)
    cr, nr, cvq, nvq, ci, ni, cp, np_ = model(cb, nb)
    assert cr.shape == (64, 20) and nr.shape == (64, 40)
    assert ci.shape == (64, 1) and ni.shape == (64, 1)
    assert cvq.dim() == 0 and nvq.dim() == 0


def test_save_load_roundtrip(tmp_path):
    cfg = ModelConfig(
        input_dim=20,
        hidden_dims=(32, 16),
        cell_embedding_dim=8,
        cell_num_embeddings=16,
        neighborhood_embedding_dim=16,
        neighborhood_num_embeddings=8,
        gene_names=tuple(f"g{i}" for i in range(20)),
    )
    model = HierarchicalVQVAE(cfg)
    ck = tmp_path / "ckpt.pt"
    save_checkpoint(model, ck)
    assert ck.exists()
    loaded = load_checkpoint(ck)
    assert loaded.config.cell_num_embeddings == 16
    assert loaded.config.gene_names[:3] == ("g0", "g1", "g2")
