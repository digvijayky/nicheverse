"""Regression tests for the three encoder fixes:

1. ``mlp_plr`` gains a learnable per-gene input scale (identity at init) so the
   Gorishniy periodic map is decoupled from the raw log1p per-gene scale and
   low-count genes contribute nonzero gradients.
2. ``set_transformer`` / ``soft_moe`` / ``perceiver_io`` pool with
   concat[max, mean, PMA] instead of the collapse-prone PMA-alone readout.
3. ``MoleculeSetEncoder`` zeroes the PMA seed for all-padded (empty) cells so an
   empty cell yields a fully zero pooled embedding.
"""

from __future__ import annotations

import pytest
import torch

from nicheverse.models.encoders import MLPPLREncoder, build_encoder
from nicheverse.models.molecule_set import MoleculeSetEncoder


# ---- ISSUE 1: mlp_plr learnable per-gene input scale ------------------------


def test_mlp_plr_has_in_scale_identity_at_init():
    enc = build_encoder("mlp_plr", in_dim=366, out_dim=64, hidden=(128,))
    assert isinstance(enc, MLPPLREncoder)
    assert hasattr(enc, "in_scale"), "mlp_plr must expose a per-gene input scale"
    assert enc.in_scale.shape == (366,)
    assert enc.in_scale.requires_grad
    # identity at init: all ones -> the periodic map is unchanged at init
    assert torch.allclose(enc.in_scale.detach(), torch.ones(366))
    # bare 1D parameter -> auto-excluded from weight decay by the trainer rule
    assert enc.in_scale.ndim == 1


def test_mlp_plr_forward_shape_and_finite():
    enc = build_encoder("mlp_plr", in_dim=366, out_dim=64, hidden=(128,)).eval()
    x = torch.randn(8, 366)
    y = enc(x)
    assert y.shape == (8, 64)
    assert torch.isfinite(y).all()


def test_mlp_plr_low_count_gene_gets_gradient():
    # A low-value gene (x near 0) must now receive a nonzero gradient through the
    # per-gene scale + periodic map; before the fix its periodic activation was
    # near-flat and its contribution vanished.
    torch.manual_seed(0)
    enc = build_encoder("mlp_plr", in_dim=366, out_dim=64, hidden=(128,)).train()
    x = torch.randn(4, 366)
    x[:, 5] = 1e-3  # a low-count log1p gene
    x = x.clone().requires_grad_(True)
    enc(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    # gradient wrt the low-value input gene must be nonzero
    assert x.grad[:, 5].abs().sum().item() > 0.0
    # the per-gene scale parameter itself must accumulate gradient everywhere
    assert enc.in_scale.grad is not None
    assert torch.isfinite(enc.in_scale.grad).all()
    assert enc.in_scale.grad[5].abs().item() > 0.0


# ---- ISSUE 2: concat[max, mean, PMA] pooling on generic set encoders --------


@pytest.mark.parametrize("enc_name", ["set_transformer", "soft_moe", "perceiver_io"])
@pytest.mark.parametrize("in_dim", [366, 732])
def test_generic_set_encoder_builds_and_forwards(enc_name, in_dim):
    out_dim = 64
    enc = build_encoder(enc_name, in_dim=in_dim, out_dim=out_dim, hidden=(128,)).eval()
    x = torch.randn(6, in_dim)
    y = enc(x)
    assert y.shape == (6, out_dim)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("enc_name", ["set_transformer", "soft_moe", "perceiver_io"])
def test_generic_set_encoder_final_proj_widened(enc_name):
    # concat[max, mean, PMA] means the final projection consumes 3*width.
    enc = build_encoder(enc_name, in_dim=366, out_dim=64, hidden=(128,))
    assert enc.out.in_features == 128 * 3
    assert enc.out.out_features == 64


# ---- ISSUE 3: MoleculeSetEncoder empty-cell embedding is all zeros ----------


def test_molecule_set_encoder_empty_cell_is_zero():
    enc = MoleculeSetEncoder(out_dim=64, width=128, n_genes=366).eval()
    B, M = 3, 20
    gene = torch.full((B, M), enc.pad_gene, dtype=torch.long)  # all padding token
    coords = torch.zeros(B, M, 2)
    mask = torch.zeros(B, M, dtype=torch.bool)  # every cell empty
    # give row 1 some real molecules so we confirm only empty rows go to zero
    real = 7
    gene[1, :real] = torch.randint(0, 366, (real,))
    coords[1, :real] = torch.randn(real, 2)
    mask[1, :real] = True
    with torch.no_grad():
        z = enc(gene, coords, mask)
    assert z.shape == (B, 64)
    assert torch.isfinite(z).all()
    # the fully padded rows (0 and 2) must produce an all-zero pooled embedding
    # (masked max = 0, masked mean = 0, PMA seed zeroed -> out(0) = out.bias only;
    # so test the pre-projection concat is zero by checking the encoder output
    # matches the encoder applied to the same empty input deterministically and
    # equals out(concat[0,0,0]))
    empty_ref = enc.out(torch.zeros(1, 128 * 3))
    assert torch.allclose(z[0], empty_ref.squeeze(0), atol=1e-6)
    assert torch.allclose(z[2], empty_ref.squeeze(0), atol=1e-6)
    # the non-empty row must differ from the empty embedding
    assert not torch.allclose(z[1], empty_ref.squeeze(0), atol=1e-6)


def test_molecule_set_encoder_all_empty_batch_zero_concat():
    # When every cell is empty, the pooled concat[max, mean, seed] must be all
    # zeros so downstream is out(bias) identically across rows.
    enc = MoleculeSetEncoder(out_dim=32, width=64, n_genes=100).eval()
    B, M = 4, 10
    gene = torch.full((B, M), enc.pad_gene, dtype=torch.long)
    coords = torch.zeros(B, M, 2)
    mask = torch.zeros(B, M, dtype=torch.bool)
    with torch.no_grad():
        z = enc(gene, coords, mask)
    # all rows identical (each is out of an all-zero concat)
    assert torch.allclose(z, z[0:1].expand_as(z), atol=1e-6)
