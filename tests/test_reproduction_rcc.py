"""Regression anchor: the package reproduces the production RCC v4 codes exactly.

This encodes the reproducibility contract that was proven at refactor time:
running the package on a real sample from the 173-sample RCC/BrM cohort yields
cell-code and neighborhood-code assignments that are byte-for-byte identical
(100% per-cell match, SHA256-equal) to the codes stored by the original
``annotforxenium_model_for_rcc_brm_v4_dev.py`` production run.

It is opt-in so the fast unit suite is unaffected: it only runs when
``NICHEVERSE_RUN_RCC_REPRO=1`` is set AND the production checkpoint + adata are
present (they are large and live outside the repo). Point it at a different
checkpoint dir with ``NICHEVERSE_RCC_CKPT_DIR``. Re-run after every refactor
milestone (on a compute node, it loads a multi-GB backed AnnData)::

    NICHEVERSE_RUN_RCC_REPRO=1 pytest tests/test_reproduction_rcc.py -q
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

_CKPT_DIR = Path(
    os.environ.get(
        "NICHEVERSE_RCC_CKPT_DIR",
        "/data1/lesliec/vijay/spatial_transcriptomicsg/my_work_Gosabopos/checkpointsg/"
        "annotforxenium_model_rcc_brm_v4_2026_06_11_173samples_hp_256cell_32neigh",
    )
)
_PT = _CKPT_DIR / "hierarchical_vqvae_checkpoint.pt"
_ADATA = _CKPT_DIR / "adata_with_hierarchical_embeddings.h5ad"

pytestmark = pytest.mark.skipif(
    os.environ.get("NICHEVERSE_RUN_RCC_REPRO") != "1" or not (_PT.exists() and _ADATA.exists()),
    reason="set NICHEVERSE_RUN_RCC_REPRO=1 with the production RCC checkpoint present to run",
)

# Production architecture (matches the legacy bare-state-dict checkpoint).
_RCC_CFG = dict(
    hidden_dims=(256, 128),
    cell_embedding_dim=64,
    cell_num_embeddings=256,
    neighborhood_embedding_dim=256,
    neighborhood_num_embeddings=32,
    commitment_cost=0.25,
    use_cross_attention=True,
)


def test_reproduces_rcc_v4_codes_sha_exact():
    import anndata as ad
    import scipy.sparse as sp

    from nicheverse.models import ModelConfig, load_checkpoint
    from nicheverse.training import predict_codes
    from nicheverse.utils import seed_everything, sha256_array

    seed_everything(49, deterministic=True)
    backed = ad.read_h5ad(_ADATA, backed="r")
    gene_names = tuple(map(str, backed.var_names))
    col = "sample_id" if "sample_id" in backed.obs.columns else "sample"
    vc = backed.obs[col].value_counts()
    pick = vc[(vc > 5000) & (vc < 80000)].sort_values().index[0]
    idx = np.where(backed.obs[col].astype(str).values == pick)[0]
    x = backed.X[idx, :]
    x = x.toarray() if sp.issparse(x) else np.asarray(x)
    sub = ad.AnnData(
        X=np.asarray(x, np.float32), obs=backed.obs.iloc[idx].copy(), var=backed.var.copy()
    )
    sub.obsm["spatial"] = np.asarray(backed.obsm["spatial"])[idx, :]
    sub.obs[col] = pick
    backed.file.close()

    ref_cell = sub.obs["cell_codebook_idx"].to_numpy().astype(np.int64)
    ref_neigh = sub.obs["neighborhood_codebook_idx"].to_numpy().astype(np.int64)
    del sub.obs["cell_codebook_idx"], sub.obs["neighborhood_codebook_idx"]

    cfg = ModelConfig(input_dim=len(gene_names), gene_names=gene_names, **_RCC_CFG)
    model = load_checkpoint(_PT, device="cpu", config=cfg)
    out = predict_codes(
        sub,
        model,
        sample_col=col,
        k_neighbors=20,
        neighborhood_aggregation="weighted_mean",
        batch_size=2048,
        normalize=False,
        log1p=False,
        device="cpu",
        return_embeddings=False,
        seed=49,
        deterministic=True,
    )
    pred_cell = out.obs["cell_codebook_idx"].to_numpy().astype(np.int64)
    pred_neigh = out.obs["neighborhood_codebook_idx"].to_numpy().astype(np.int64)

    assert (pred_cell == ref_cell).mean() == 1.0, "cell codes drifted from production"
    assert (pred_neigh == ref_neigh).mean() == 1.0, "neighborhood codes drifted from production"
    assert sha256_array(pred_cell) == sha256_array(ref_cell)
    assert sha256_array(pred_neigh) == sha256_array(ref_neigh)
