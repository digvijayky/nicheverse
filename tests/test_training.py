"""Tests for the optional LR schedule and weight-decay grouping (defaults unchanged)."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from nicheverse import ModelConfig, TrainConfig, Trainer
from nicheverse.models import HierarchicalVQVAE
from nicheverse.models.molecule_set import MoleculeSetVQVAE
from nicheverse.training.trainer import (
    _build_optimizer,
    _build_scheduler,
    _split_decay_params,
    auto_batch_size,
)


def test_auto_batch_size_cpu_power_of_two_and_monotone():
    dims = [366, 732, 5000, 21731]
    bss = [auto_batch_size(d, device=None) for d in dims]
    for bs in bss:
        assert 512 <= bs <= 16384, bs
        assert (bs & (bs - 1)) == 0, f"{bs} is not a power of two"
    # Non-increasing with panel size (equal only at the clamp floor).
    for a_, b_ in zip(bss, bss[1:]):
        assert b_ <= a_, (bss, dims)
    # The two mid-size panels must strictly shrink and both stay above 2048.
    assert bss[0] > bss[1] > 2048


def _toy_adata(n=80, g=16):
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=sp.csr_matrix(rng.poisson(1.0, size=(n, g)).astype("float32")))
    a.var_names = [f"g{i}" for i in range(g)]
    a.obs["sample_id"] = np.array(["S1"] * (n // 2) + ["S2"] * (n - n // 2))
    a.obsm["spatial"] = np.column_stack([rng.uniform(0, 500, n), rng.uniform(0, 500, n)])
    return a


def _mc(a):
    return ModelConfig(
        input_dim=a.n_vars,
        hidden_dims=(16,),
        cell_embedding_dim=6,
        cell_num_embeddings=8,
        neighborhood_embedding_dim=8,
        neighborhood_num_embeddings=4,
        gene_names=tuple(a.var_names),
    )


def test_default_schedule_is_plateau():
    m = HierarchicalVQVAE(_mc(_toy_adata()))
    tc = TrainConfig()
    opt = _build_optimizer(m, tc)
    assert isinstance(opt, torch.optim.AdamW)
    sched, needs = _build_scheduler(opt, tc)
    assert needs is True
    assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_config_defaults_decoupled_wd():
    # New default: selective decoupled AdamW weight decay at 0.01.
    tc = TrainConfig()
    assert tc.decoupled_weight_decay is True
    assert tc.weight_decay == 0.01


def test_default_optimizer_makes_two_groups():
    # With the new defaults, the optimizer has a decay and a no-decay group.
    m = HierarchicalVQVAE(_mc(_toy_adata()))
    opt = _build_optimizer(m, TrainConfig())
    assert len(opt.param_groups) == 2
    assert sorted(g["weight_decay"] for g in opt.param_groups) == [0.0, 0.01]


def test_decoupled_off_single_group():
    # decoupled_weight_decay=False -> one uniform AdamW group (legacy behavior).
    m = HierarchicalVQVAE(_mc(_toy_adata()))
    opt = _build_optimizer(m, TrainConfig(decoupled_weight_decay=False, weight_decay=0.01))
    assert len(opt.param_groups) == 1


def test_decoupled_wd_makes_two_groups():
    m = HierarchicalVQVAE(_mc(_toy_adata()))
    opt = _build_optimizer(m, TrainConfig(decoupled_weight_decay=True, weight_decay=0.01))
    assert len(opt.param_groups) == 2
    assert sorted(g["weight_decay"] for g in opt.param_groups) == [0.0, 0.01]


def _mc_enc(a, enc):
    return ModelConfig(
        input_dim=a.n_vars,
        hidden_dims=(16,),
        cell_embedding_dim=8,
        cell_num_embeddings=8,
        neighborhood_embedding_dim=8,
        neighborhood_num_embeddings=4,
        encoder_type=enc,
        gene_names=tuple(a.var_names),
    )


@pytest.mark.parametrize("enc", ["mlp", "mlp_deep", "mlp_plr", "set_transformer"])
def test_selective_wd_grouping_invariants(enc):
    a = _toy_adata()
    m = HierarchicalVQVAE(_mc_enc(a, enc))
    decay, no_decay, dn, ndn = _split_decay_params(m)
    id_decay = {id(p) for p in decay}
    id_nodecay = {id(p) for p in no_decay}
    all_ids = {id(p) for p in m.parameters() if p.requires_grad}
    nd_names = set(ndn)

    # (e) every trainable param in exactly one group; union == model.parameters()
    assert id_decay.isdisjoint(id_nodecay)
    assert id_decay | id_nodecay == all_ids
    assert len(decay) + len(no_decay) == len(all_ids)

    # (a) both EMA VQ codebooks are frozen, so excluded from the optimizer entirely
    # (never weight-decayed and not in the no-decay group either)
    for cb in (m.cell_vq.embedding.weight, m.neighborhood_vq.embedding.weight):
        assert cb.requires_grad is False
        assert id(cb) not in id_decay and id(cb) not in id_nodecay

    # (c) all biases and all LayerNorm/BatchNorm params in no-decay
    norm_types = (torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.BatchNorm1d,
                  torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
    n_bias = 0
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith("bias"):
            assert n in nd_names
            n_bias += 1
    assert n_bias >= 1
    for mn, mod in m.named_modules():
        pre = f"{mn}." if mn else ""
        if isinstance(mod, norm_types):
            for pn, pp in mod.named_parameters(recurse=False):
                if pp.requires_grad:
                    assert (pre + pn) in nd_names

    # (d) at least one Linear weight IS decayed, and every decayed name is a weight
    assert len(decay) >= 1
    assert all(n.endswith("weight") for n in dn)

    # (b) mlp_plr periodic-embedding parameters excluded from decay
    if enc == "mlp_plr":
        for tail in ("freq", "emb_w", "emb_b"):
            assert any(n.endswith("." + tail) for n in ndn), tail


def test_selective_wd_grouping_molecule_set():
    m = MoleculeSetVQVAE(
        n_genes=16, cell_embedding_dim=8, cell_num_embeddings=8,
        neighborhood_embedding_dim=8, neighborhood_num_embeddings=4,
        hidden=(16,), enc_width=16, enc_inds=4,
    )
    decay, no_decay, dn, ndn = _split_decay_params(m)
    id_decay = {id(p) for p in decay}
    id_nodecay = {id(p) for p in no_decay}
    all_ids = {id(p) for p in m.parameters() if p.requires_grad}
    # Union covers all trainable params; the EMA codebook is frozen (excluded), the
    # gene embedding stays trainable in no-decay.
    assert (id_decay | id_nodecay) == all_ids
    assert m.cell_vq.embedding.weight.requires_grad is False
    assert id(m.cell_vq.embedding.weight) not in id_decay
    assert id(m.cell_vq.embedding.weight) not in id_nodecay
    assert id(m.cell_encoder.gene_embed.weight) in id_nodecay
    assert all(n.endswith("weight") for n in dn)


def test_warmup_cosine_schedule_type():
    m = HierarchicalVQVAE(_mc(_toy_adata()))
    opt = _build_optimizer(m, TrainConfig())
    sched, needs = _build_scheduler(opt, TrainConfig(lr_schedule="warmup_cosine", warmup_steps=2))
    assert needs is False
    assert isinstance(sched, torch.optim.lr_scheduler.LambdaLR)


def test_warmup_cosine_train_runs(tmp_path):
    a = _toy_adata()
    tc = TrainConfig(
        num_epochs=3,
        batch_size=32,
        k_neighbors=5,
        log_every=100,
        lr_schedule="warmup_cosine",
        warmup_steps=1,
        decoupled_weight_decay=True,
        weight_decay=0.01,
    )
    Trainer(tc).fit(a, tmp_path, model_config=_mc(a))
    losses = json.loads((Path(tmp_path) / "training_losses.json").read_text())
    assert len(losses) == 3 and all(np.isfinite(x["total"]) for x in losses)


def test_invalid_schedule_rejected(tmp_path):
    a = _toy_adata()
    with pytest.raises(ValueError, match="lr_schedule"):
        Trainer(TrainConfig(lr_schedule="bad")).fit(a, tmp_path, model_config=_mc(a))


# --------------------------------------------------------------------------- #
# Default-on training metrics + end-of-training training-curves PDF.
# --------------------------------------------------------------------------- #

import csv  # noqa: E402

from nicheverse.training import metrics as _metrics  # noqa: E402


def _short_tc(**kw):
    base = dict(num_epochs=3, batch_size=32, k_neighbors=5, log_every=100)
    base.update(kw)
    return TrainConfig(**base)


def test_codebook_usage_stats_uniform_vs_collapsed():
    # Uniform usage: all K codes active, Gini ~0, perplexity == K.
    uni = _metrics.codebook_usage_stats(np.full(8, 10))
    assert uni["active"] == 8
    assert uni["active_frac"] == 1.0
    assert uni["gini"] < 1e-6
    assert abs(uni["perplexity"] - 8) < 1e-6
    assert abs(uni["entropy_norm"] - 1.0) < 1e-6
    # Collapsed onto one code: 1 active, Gini high, perplexity ~1.
    col = _metrics.codebook_usage_stats(np.array([100, 0, 0, 0, 0, 0, 0, 0]))
    assert col["active"] == 1
    assert col["gini"] > 0.8
    assert col["perplexity"] < 1.01
    # Empty histogram degrades to zeros, no crash.
    z = _metrics.codebook_usage_stats(np.zeros(8))
    assert z["active"] == 0 and z["perplexity"] == 0.0


def test_train_writes_metrics_csv_with_expected_columns(tmp_path):
    a = _toy_adata()
    Trainer(_short_tc()).fit(a, tmp_path, model_config=_mc(a))
    csv_path = Path(tmp_path) / "training_metrics.csv"
    assert csv_path.exists(), "training_metrics.csv was not written"
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3, rows
    expected = {
        "epoch", "total", "cell", "neighborhood", "cell_recon", "niche_recon",
        "cell_vq", "niche_vq", "cell_perplexity", "neighborhood_perplexity",
        "cell_active_codes", "neighborhood_active_codes", "cell_usage_gini",
        "neighborhood_usage_gini", "learning_rate", "grad_norm",
        "epoch_seconds", "cells_per_second",
    }
    assert expected.issubset(set(rows[0].keys())), expected - set(rows[0].keys())
    # Every recorded per-epoch metric is finite.
    for r in rows:
        for k in expected:
            v = r[k]
            if v == "":
                continue
            assert np.isfinite(float(v)), (k, v)
    # Active-code counts stay within [0, K] for both codebooks (K=8 cell, K=4 niche).
    for r in rows:
        assert 0 <= float(r["cell_active_codes"]) <= 8
        assert 0 <= float(r["neighborhood_active_codes"]) <= 4
        assert 0.0 <= float(r["cell_usage_gini"]) <= 1.0


def test_train_writes_training_curves_pdf(tmp_path):
    a = _toy_adata()
    Trainer(_short_tc()).fit(a, tmp_path, model_config=_mc(a))
    pdf_path = Path(tmp_path) / "training_curves.pdf"
    assert pdf_path.exists(), "training_curves.pdf was not written"
    head = pdf_path.read_bytes()[:5]
    assert head.startswith(b"%PDF"), head


def test_training_losses_json_carries_new_metrics(tmp_path):
    a = _toy_adata()
    Trainer(_short_tc()).fit(a, tmp_path, model_config=_mc(a))
    losses = json.loads((Path(tmp_path) / "training_losses.json").read_text())
    assert len(losses) == 3
    for e in losses:
        for k in ("epoch", "cell_recon", "niche_recon", "learning_rate",
                  "grad_norm", "cell_active_codes", "cell_usage_gini"):
            assert k in e, k
            assert np.isfinite(float(e[k]))


def test_plotting_failure_does_not_crash_training(tmp_path, monkeypatch):
    # Force the curve plotter to raise; training must still finish and write all
    # other artifacts (the plot is wrapped in try/except in the trainer).
    import nicheverse.training.trainer as _trainer

    def _boom(*_a, **_k):
        raise RuntimeError("forced plotting failure")

    monkeypatch.setattr(_trainer, "plot_training_curves", _boom)
    a = _toy_adata()
    model, out = Trainer(_short_tc()).fit(a, tmp_path, model_config=_mc(a))
    assert model is not None
    # Training completed: checkpoint, losses json, and metrics csv all present even
    # though plotting failed; the PDF is absent.
    assert (Path(tmp_path) / "hierarchical_vqvae_checkpoint.pt").exists()
    assert (Path(tmp_path) / "training_metrics.csv").exists()
    assert not (Path(tmp_path) / "training_curves.pdf").exists()
    losses = json.loads((Path(tmp_path) / "training_losses.json").read_text())
    assert len(losses) == 3
