"""Every configuration the docs advertise as supported must at least run.

This is a breadth test: it exercises all 15 registered encoders through a full
model forward, all 7 spatial-graph modes, the default and alternative niche
reconstruction losses, the reconstruction-mode compatibility guard, the bare
default config end to end, and the cross-attention toggle. Codebook *quality*
(occupancy, collapse) is a modelling property covered elsewhere; here we only
assert that each config constructs, runs, and produces valid code indices.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from nicheverse import ModelConfig, TrainConfig, train_model
from nicheverse.data.dataset import _VALID_GRAPHS, SpatialDataset
from nicheverse.models import HierarchicalVQVAE
from nicheverse.models.encoders import _ENCODERS


def _toy_adata(n=64, g=16, seed=0):
    rng = np.random.default_rng(seed)
    a = ad.AnnData(X=sp.csr_matrix(rng.poisson(1.2, size=(n, g)).astype("float32")))
    a.var_names = [f"g{i}" for i in range(g)]
    a.obs["sample_id"] = np.array(["S1"] * (n // 2) + ["S2"] * (n - n // 2))
    a.obsm["spatial"] = np.column_stack([rng.uniform(0, 400, n), rng.uniform(0, 400, n)])
    return a


def _mc(a, **kw):
    base = dict(
        input_dim=a.n_vars,
        hidden_dims=(16,),
        cell_embedding_dim=8,
        cell_num_embeddings=8,
        neighborhood_embedding_dim=8,
        neighborhood_num_embeddings=4,
        gene_names=tuple(a.var_names),
    )
    base.update(kw)
    return ModelConfig(**base)


def _tc(**kw):
    base = dict(num_epochs=1, batch_size=64, k_neighbors=8, spatial_graph="knn",
               save_best=False, seed=0)
    base.update(kw)
    return TrainConfig(**base)


# --- 1. every registered encoder runs through a full model forward ---
@pytest.mark.parametrize("enc", sorted(_ENCODERS))
def test_every_encoder_forward(enc):
    d = 20
    mc = ModelConfig(
        input_dim=d,
        hidden_dims=(16,),
        cell_embedding_dim=8,
        cell_num_embeddings=8,
        neighborhood_embedding_dim=8,
        neighborhood_num_embeddings=4,
        encoder_type=enc,
        gene_names=tuple(f"g{i}" for i in range(d)),
    )
    m = HierarchicalVQVAE(mc).train()
    out = m(torch.randn(6, d), torch.randn(6, 2 * d))
    cell_recon, cell_idx = out[0], out[4].reshape(-1)
    assert cell_recon.shape == (6, d)
    assert torch.isfinite(cell_recon).all(), f"{enc}: non-finite cell reconstruction"
    assert cell_idx.shape == (6,)
    assert int(cell_idx.min()) >= 0 and int(cell_idx.max()) < mc.cell_num_embeddings


# --- 2. every spatial-graph mode builds finite neighborhood features ---
@pytest.mark.parametrize("graph", sorted(_VALID_GRAPHS))
def test_every_spatial_graph_builds(graph):
    a = _toy_adata()
    x = torch.as_tensor(a.X.toarray())
    xy = np.asarray(a.obsm["spatial"])
    s = a.obs["sample_id"].to_numpy()
    kw = {"radius": 80.0} if graph in ("radius", "knn_radius") else {}
    ds = SpatialDataset(x, xy, s, k_neighbors=8, spatial_graph=graph, **kw)
    nf = ds.neighborhood_features
    assert nf.shape == (a.n_obs, 2 * a.n_vars)
    assert torch.isfinite(nf).all(), f"{graph}: non-finite neighborhood features"


# --- 3. niche reconstruction losses, including the default mse_dirmult ---
@pytest.mark.parametrize("niche_recon", ["mse", "mse_dirmult", "dirichlet_multinomial"])
def test_niche_recon_modes_train(niche_recon, tmp_path):
    a = _toy_adata()
    _, out = train_model(a, tmp_path, model_config=_mc(a, niche_recon=niche_recon),
                        train_config=_tc())
    idx = out.obs["neighborhood_codebook_idx"].to_numpy()
    assert idx.min() >= 0 and idx.max() < 4


# --- 4. the reconstruction-mode compatibility guard (regression) ---
def test_mse_cell_recon_requires_mse_niche(tmp_path):
    # cell_recon="mse" with the default niche (mse_dirmult) is an invalid pairing:
    # the Dirichlet-multinomial niche term needs count-scale composition, which is
    # only built for a count cell mode. It must raise a clear, actionable error.
    a = _toy_adata()
    with pytest.raises(ValueError, match="Dirichlet-multinomial"):
        train_model(a, tmp_path, model_config=_mc(a, cell_recon="mse", detection_weight=0.0),
                    train_config=_tc())


def test_pure_mse_path_trains(tmp_path):
    # Setting both branches to MSE is the supported pure-MSE path and must train.
    a = _toy_adata()
    _, out = train_model(
        a, tmp_path,
        model_config=_mc(a, cell_recon="mse", niche_recon="mse", detection_weight=0.0),
        train_config=_tc(),
    )
    idx = out.obs["cell_codebook_idx"].to_numpy()
    assert idx.min() >= 0 and idx.max() < 8


# --- 5. the bare default config (nb + mse_dirmult) end to end ---
def test_default_config_end_to_end(tmp_path):
    a = _toy_adata()
    mc = ModelConfig(input_dim=a.n_vars, gene_names=tuple(a.var_names),
                     cell_num_embeddings=8, neighborhood_num_embeddings=4)
    assert mc.cell_recon == "nb" and mc.niche_recon == "mse_dirmult"
    _, out = train_model(a, tmp_path, model_config=mc, train_config=_tc())
    for key, k in [("cell_codebook_idx", 8), ("neighborhood_codebook_idx", 4)]:
        idx = out.obs[key].to_numpy()
        assert idx.min() >= 0 and idx.max() < k
    assert out.obsm["X_cell_embedding"].shape[0] == a.n_obs


# --- 6. cross-attention toggle (both paths advertised) ---
@pytest.mark.parametrize("use_ca", [True, False])
def test_cross_attention_toggle(use_ca):
    d = 20
    mc = ModelConfig(
        input_dim=d, hidden_dims=(16,), cell_embedding_dim=8, cell_num_embeddings=8,
        neighborhood_embedding_dim=8, neighborhood_num_embeddings=4,
        use_cross_attention=use_ca, gene_names=tuple(f"g{i}" for i in range(d)),
    )
    m = HierarchicalVQVAE(mc).train()
    out = m(torch.randn(6, d), torch.randn(6, 2 * d))
    assert torch.isfinite(out[0]).all()
