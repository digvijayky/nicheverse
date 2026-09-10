"""Foundation (multi panel) model: masked losses, decoder conditioning, streaming dataset."""

from __future__ import annotations

import json

import numpy as np
import scipy.sparse as sp
import torch

from nicheverse.data.multipanel import MultiPanelSpatialDataset, PanelSpec
from nicheverse.data.shards import to_bag, write_shard
from nicheverse.losses import (
    bernoulli_detection_bce,
    dirichlet_multinomial_nll,
    masked_bernoulli_detection_bce,
    masked_dirichlet_multinomial_nll,
    masked_nb_nll,
    nb_nll,
)
from nicheverse.models.foundation import (
    FoundationConfig,
    FoundationVQVAE,
    load_foundation,
    save_foundation,
)
from nicheverse.models.sparse import SparseBag


def _bag(mat):
    i, v, o = to_bag(mat)
    return SparseBag(torch.from_numpy(i), torch.from_numpy(v), torch.from_numpy(o))


# ---- masked losses -----------------------------------------------------------------------
def test_masked_nb_equals_dense_on_measured_genes():
    torch.manual_seed(0)
    B, V = 6, 20
    measured = torch.tensor([1, 3, 4, 11, 19])
    x = torch.zeros(B, V)
    x[:, measured] = torch.randint(0, 8, (B, len(measured))).float()
    cr = torch.randn(B, V)
    log_theta = torch.randn(V)
    got = masked_nb_nll(x, cr, log_theta, measured)
    want = nb_nll(x[:, measured], cr[:, measured], log_theta[measured])
    assert torch.allclose(got, want, atol=1e-5)
    # a pre gathered argument must give the identical value
    got2 = masked_nb_nll(x[:, measured], cr[:, measured], log_theta[measured], measured)
    assert torch.allclose(got, got2, atol=1e-6)


def test_masked_nb_ignores_unmeasured_genes():
    """Changing counts at unmeasured genes cannot change the masked loss."""
    torch.manual_seed(1)
    B, V = 5, 16
    measured = torch.tensor([0, 2, 7])
    x = torch.zeros(B, V)
    x[:, measured] = torch.randint(1, 5, (B, 3)).float()
    cr = torch.randn(B, V)
    lt = torch.zeros(V)
    base = masked_nb_nll(x, cr, lt, measured)
    x2 = x.clone()
    x2[:, 5] = 99.0
    assert torch.allclose(base, masked_nb_nll(x2, cr, lt, measured), atol=1e-6)


def test_masked_detection_and_dirmult_match_dense():
    torch.manual_seed(2)
    B, V = 4, 12
    measured = torch.tensor([2, 5, 6, 9])
    x = torch.zeros(B, V)
    x[:, measured] = torch.randint(0, 4, (B, 4)).float()
    logits = torch.randn(B, V)
    la = torch.randn(V)
    assert torch.allclose(
        masked_bernoulli_detection_bce(x, logits, measured),
        bernoulli_detection_bce(x[:, measured], logits[:, measured]),
        atol=1e-5,
    )
    assert torch.allclose(
        masked_dirichlet_multinomial_nll(x, logits, la, measured),
        dirichlet_multinomial_nll(x[:, measured], logits[:, measured], la[measured]),
        atol=1e-4,
    )


# ---- model -------------------------------------------------------------------------------
def _cfg(**kw):
    base = dict(
        vocab_size=24,
        hidden_dims=(16,),
        cell_embedding_dim=8,
        cell_num_embeddings=8,
        neighborhood_embedding_dim=8,
        neighborhood_num_embeddings=4,
        gene_embed_dim=8,
        decoder_hidden=8,
        n_platforms=3,
        n_species=2,
        n_datasets=2,
    )
    base.update(kw)
    return FoundationConfig(**base)


def _batch(V=24, B=7, seed=0, measured=None):
    rng = np.random.default_rng(seed)
    measured = np.array([1, 2, 5, 8, 13, 21]) if measured is None else measured
    X = np.zeros((B, V), np.float32)
    X[:, measured] = rng.poisson(2.0, size=(B, len(measured)))
    N = np.zeros((B, V), np.float32)
    N[:, measured] = rng.gamma(2.0, 1.0, size=(B, len(measured)))
    return (
        _bag(sp.csr_matrix(X)),
        _bag(sp.csr_matrix(N)),
        torch.as_tensor(measured.astype(np.int64)),
        torch.as_tensor(X[:, measured]),
        torch.as_tensor(N[:, measured]),
    )


def test_forward_and_loss():
    torch.manual_seed(0)
    m = FoundationVQVAE(_cfg()).train()
    cb, nb, measured, ct, nt = _batch()
    out = m(
        cb, nb, measured,
        platform_id=torch.full((7,), 2), species_id=torch.ones(7, dtype=torch.long),
    )
    assert out["cell_logits"].shape == (7, len(measured))
    assert out["niche_nbr_logits"].shape == (7, len(measured))
    assert out["cell_idx"].shape == (7,)
    assert int(out["cell_idx"].max()) < 8 and int(out["niche_idx"].max()) < 4
    loss, parts = m.compute_loss(out, ct, nt, measured)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(m.cell_log_theta.grad).all()
    assert set(parts) >= {"loss", "cell_nb", "detection", "niche", "vq"}


def test_two_panels_share_one_model():
    """Different measured masks must both run through the same decoder heads."""
    torch.manual_seed(0)
    m = FoundationVQVAE(_cfg()).train()
    for seed, meas in ((0, np.array([0, 1, 2])), (1, np.arange(24))):
        cb, nb, measured, ct, nt = _batch(seed=seed, measured=meas)
        out = m(cb, nb, measured)
        loss, _ = m.compute_loss(out, ct, nt, measured)
        assert torch.isfinite(loss)


def test_conditioning_touches_decoder_only():
    """Platform / species must not change the codes, only the reconstruction."""
    torch.manual_seed(0)
    m = FoundationVQVAE(_cfg()).eval()
    with torch.no_grad():
        m.platform_embed.weight.normal_(0, 1.0)
        m.species_embed.weight.normal_(0, 1.0)
    cb, nb, measured, _, _ = _batch()
    with torch.no_grad():
        a = m(cb, nb, measured, platform_id=torch.zeros(7, dtype=torch.long),
              species_id=torch.zeros(7, dtype=torch.long))
        b = m(cb, nb, measured, platform_id=torch.full((7,), 2),
              species_id=torch.ones(7, dtype=torch.long))
    assert torch.equal(a["cell_idx"], b["cell_idx"])
    assert torch.equal(a["niche_idx"], b["niche_idx"])
    assert torch.allclose(a["z_cell"], b["z_cell"], atol=1e-6)
    assert not torch.allclose(a["cell_logits"], b["cell_logits"], atol=1e-4)


def test_conditioning_can_be_disabled():
    m = FoundationVQVAE(_cfg(condition_decoders=False)).eval()
    assert not hasattr(m, "platform_embed")
    cb, nb, measured, ct, nt = _batch()
    with torch.no_grad():
        out = m(cb, nb, measured)
    assert torch.isfinite(out["cell_logits"]).all()


def test_context_missing_flag():
    torch.manual_seed(0)
    m = FoundationVQVAE(_cfg()).eval()
    cb, nb, measured, _, _ = _batch()
    with torch.no_grad():
        no_ctx = m(cb, nb, measured)
        masked_off = m(cb, nb, measured, cell_context=cb, nbr_context=nb,
                       has_context=torch.zeros(7))
    assert torch.allclose(no_ctx["z_cell"], masked_off["z_cell"], atol=1e-6)


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    m = FoundationVQVAE(_cfg(vocabulary=tuple(f"G{i}" for i in range(24)))).eval()
    p = save_foundation(m, tmp_path / "fm.pt")
    m2 = load_foundation(p)
    cb, nb, measured, _, _ = _batch()
    with torch.no_grad():
        assert torch.allclose(m(cb, nb, measured)["cell_logits"], m2(cb, nb, measured)["cell_logits"])
    assert json.loads((tmp_path / "fm.json").read_text())["cell_num_embeddings"] == 8


# ---- streaming dataset --------------------------------------------------------------------
def _stage(tmp_path, name, n=30, V=24, measured=None, ds_id=0, plat=0, spec=0):
    rng = np.random.default_rng(ds_id + 1)
    measured = np.arange(6) if measured is None else measured
    X = np.zeros((n, V), np.float32)
    X[:, measured] = rng.poisson(2.0, size=(n, len(measured)))
    xy = rng.uniform(0, 100, size=(n, 2)).astype(np.float32)
    from nicheverse.data._knn import knn_query

    dist, idx = knn_query(xy.astype(np.float64), 6)
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    write_shard(d / "shard_0000.npz", sp.csr_matrix(X), idx.astype(np.int32),
                dist.astype(np.float32), xy, np.zeros(n, np.int32),
                context=sp.csr_matrix(X * 2), has_context=np.ones(n, np.uint8))
    np.save(d / "measured.npy", measured.astype(np.int64))
    return dict(shards=["shard_0000.npz"], dataset_id=ds_id, platform_id=plat, species_id=spec)


def test_multipanel_stream_covers_every_cell_once(tmp_path):
    idx = {"a": _stage(tmp_path, "a", n=30, ds_id=0, plat=0),
           "b": _stage(tmp_path, "b", n=25, measured=np.arange(10), ds_id=1, plat=2, spec=1)}
    (tmp_path / "index.json").write_text(json.dumps({"datasets": idx}))
    ds = MultiPanelSpatialDataset.from_staging(tmp_path, batch_size=8, shuffle=True, seed=0)
    seen = {"a": [], "b": []}
    for batch in ds:
        seen[batch["dataset"]].extend(batch["row"].tolist())
        assert batch["cell_target"].shape[1] == batch["measured"].numel()
        assert batch["nbr_target"].shape == batch["cell_target"].shape
        assert batch["platform_id"].numel() == batch["cell_target"].shape[0]
    assert sorted(seen["a"]) == list(range(30))
    assert sorted(seen["b"]) == list(range(25))


def test_multipanel_batches_are_single_dataset(tmp_path):
    idx = {"a": _stage(tmp_path, "a", ds_id=0), "b": _stage(tmp_path, "b", ds_id=1, plat=1)}
    (tmp_path / "index.json").write_text(json.dumps({"datasets": idx}))
    ds = MultiPanelSpatialDataset.from_staging(tmp_path, batch_size=7)
    for batch in ds:
        assert len(set(batch["dataset_id"].tolist())) == 1
        assert len(set(batch["platform_id"].tolist())) == 1


def test_multipanel_feeds_the_model(tmp_path):
    idx = {"a": _stage(tmp_path, "a", n=20, ds_id=0)}
    (tmp_path / "index.json").write_text(json.dumps({"datasets": idx}))
    ds = MultiPanelSpatialDataset.from_staging(tmp_path, batch_size=8, shuffle=False)
    m = FoundationVQVAE(_cfg()).train()
    for batch in ds:
        out = m(
            SparseBag(batch["cell_idx"], batch["cell_val"], batch["cell_off"]),
            SparseBag(batch["nbr_idx"], batch["nbr_val"], batch["nbr_off"]),
            batch["measured"],
            cell_context=SparseBag(batch["ctx_idx"], batch["ctx_val"], batch["ctx_off"]),
            nbr_context=SparseBag(batch["nbr_ctx_idx"], batch["nbr_ctx_val"], batch["nbr_ctx_off"]),
            has_context=batch["has_ctx"],
            platform_id=batch["platform_id"],
            species_id=batch["species_id"],
        )
        loss, parts = m.compute_loss(out, batch["cell_target"], batch["nbr_target"], batch["measured"])
        assert torch.isfinite(loss)


def test_batch_size_capped_for_wide_panels(tmp_path):
    idx = {"a": _stage(tmp_path, "a", n=10, ds_id=0)}
    (tmp_path / "index.json").write_text(json.dumps({"datasets": idx}))
    ds = MultiPanelSpatialDataset.from_staging(tmp_path, batch_size=4096, max_target_elements=12)
    assert ds.panel_batch_size(ds.panels[0]) == 2


def test_splits_never_drop_cells():
    from nicheverse.data.multipanel import _splits

    for n in (1, 2, 7, 8, 9, 25, 64, 65):
        for bs in (2, 3, 8, 64):
            b = _splits(n, bs, 2)
            assert b[0][0] == 0 and b[-1][1] == n
            assert all(b[i][1] == b[i + 1][0] for i in range(len(b) - 1))
            if n >= 2:
                assert all(s2 - s1 >= 2 for s1, s2 in b)
