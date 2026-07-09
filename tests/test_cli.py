"""End-to-end CLI smoke tests via subprocess."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "nicheverse", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_version():
    import re

    p = _run_cli("--version")
    assert p.returncode == 0
    assert "nicheverse" in p.stdout
    assert re.search(r"\d+\.\d+", p.stdout)  # some x.y version, not pinned


def test_cli_help():
    p = _run_cli("--help")
    assert p.returncode == 0
    for sub in ("preprocess", "train", "predict", "verify", "info"):
        assert sub in p.stdout


def _toy_adata(n: int = 80, g: int = 20, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(rng.poisson(1.0, size=(n, g)).astype(np.float32))
    sids = np.array(["S1"] * (n // 2) + ["S2"] * (n - n // 2))
    xy = np.column_stack([rng.uniform(0, 500, n), rng.uniform(0, 500, n)])
    a = ad.AnnData(X=X)
    a.var_names = [f"g{i}" for i in range(g)]
    a.obs["sample_id"] = sids
    a.obsm["spatial"] = xy
    return a


def test_cli_train_then_predict_then_verify(tmp_path: Path):
    inp = tmp_path / "cohort.h5ad"
    _toy_adata().write_h5ad(inp)
    ck = tmp_path / "ck"
    p = _run_cli(
        "train",
        "--input",
        str(inp),
        "--checkpoint-dir",
        str(ck),
        "--num-epochs",
        "1",
        "--cell-codebook-size",
        "8",
        "--cell-codebook-embdim",
        "8",
        "--neighborhood-codebook-size",
        "4",
        "--neighborhood-codebook-embdim",
        "16",
        "--k-neighbors",
        "3",
        "--batch-size",
        "16",
        "--no-figures",
    )
    assert p.returncode == 0, p.stderr
    assert (ck / "hierarchical_vqvae_checkpoint.pt").exists()
    assert (ck / "env_snapshot.json").exists()
    assert (ck / "train_config.json").exists()

    annotated = tmp_path / "annotated.h5ad"
    report = tmp_path / "predict_report.json"
    p2 = _run_cli(
        "predict",
        "--input",
        str(inp),
        "--checkpoint",
        str(ck / "hierarchical_vqvae_checkpoint.pt"),
        "--output",
        str(annotated),
        "--k-neighbors",
        "3",
        "--batch-size",
        "16",
        "--no-figures",
        "--report",
        str(report),
    )
    assert p2.returncode == 0, p2.stderr
    assert annotated.exists()
    assert report.exists()
    rep = json.loads(report.read_text())
    assert "predicted_cell_sha256" in rep
    assert "predicted_neighborhood_sha256" in rep

    verify_report = tmp_path / "verify.json"
    p3 = _run_cli(
        "verify",
        "--predicted",
        str(annotated),
        "--reference",
        str(report),
        "--report",
        str(verify_report),
    )
    assert p3.returncode == 0, p3.stderr
    out = json.loads(verify_report.read_text())
    assert out["cell_sha_match"] is True
    assert out["neigh_sha_match"] is True


def test_cli_info(tmp_path: Path):
    from nicheverse.models import HierarchicalVQVAE, ModelConfig, save_checkpoint

    cfg = ModelConfig(
        input_dim=20,
        hidden_dims=(16, 8),
        cell_embedding_dim=8,
        cell_num_embeddings=8,
        neighborhood_embedding_dim=16,
        neighborhood_num_embeddings=4,
        gene_names=tuple(f"g{i}" for i in range(20)),
    )
    ck = tmp_path / "ck.pt"
    save_checkpoint(HierarchicalVQVAE(cfg), ck)
    p = _run_cli("info", "--checkpoint", str(ck))
    assert p.returncode == 0, p.stderr
    info = json.loads(p.stdout)
    assert info["n_genes"] == 20
    assert info["n_cell_codes"] == 8
    assert "checkpoint_sha256" in info
    assert "host_env" in info


def test_cli_quiet_flag(tmp_path: Path):
    inp = tmp_path / "cohort.h5ad"
    _toy_adata(n=40, g=10).write_h5ad(inp)
    ck = tmp_path / "ck"
    p = _run_cli(
        "--quiet",
        "train",
        "--input",
        str(inp),
        "--checkpoint-dir",
        str(ck),
        "--num-epochs",
        "1",
        "--cell-codebook-size",
        "4",
        "--cell-codebook-embdim",
        "4",
        "--neighborhood-codebook-size",
        "2",
        "--neighborhood-codebook-embdim",
        "8",
        "--k-neighbors",
        "2",
        "--batch-size",
        "8",
        "--no-figures",
    )
    assert p.returncode == 0, p.stderr
    # In quiet mode, training INFO logs should not appear in stderr.
    assert "avg_total=" not in p.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="permission semantics differ on win32")
def test_cli_verify_mismatch_returns_2(tmp_path: Path):
    # Build two distinct toy AnnDatas and label them as if both came from predict.
    a = _toy_adata(seed=0)
    a.obs["cell_codebook_idx"] = np.zeros(a.n_obs, dtype=np.int32)
    a.obs["neighborhood_codebook_idx"] = np.zeros(a.n_obs, dtype=np.int32)
    b = _toy_adata(seed=0)
    b.obs["cell_codebook_idx"] = np.ones(a.n_obs, dtype=np.int32)
    b.obs["neighborhood_codebook_idx"] = np.ones(a.n_obs, dtype=np.int32)
    pa = tmp_path / "a.h5ad"
    pb = tmp_path / "b.h5ad"
    a.write_h5ad(pa)
    b.write_h5ad(pb)
    p = _run_cli("verify", "--predicted", str(pa), "--reference", str(pb))
    assert p.returncode == 2
