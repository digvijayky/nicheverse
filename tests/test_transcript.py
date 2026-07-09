"""Tests for the transcript-level context utility (synthetic molecule table)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from nicheverse.data import transcript_context


def _setup(tmp_path):
    a = ad.AnnData(X=sp.csr_matrix(np.ones((3, 4), dtype="float32")))
    a.var_names = ["g0", "g1", "g2", "g3"]
    a.obs["sample_id"] = ["S", "S", "S"]
    a.obsm["spatial"] = np.array([[0.0, 0.0], [100.0, 100.0], [200.0, 200.0]])
    rows = [(1.0, 1.0, "g0")] * 5 + [(101.0, 101.0, "g1")] * 3
    rows += [(2.0, 2.0, "BLANK_0001"), (0.5, 0.5, "g2")]  # control (filtered) + one g2 near cell0
    df = pd.DataFrame(rows, columns=["x_location", "y_location", "feature_name"])
    p = tmp_path / "transcripts.parquet"
    df.to_parquet(p)
    return a, {"S": str(p)}


def test_transcript_context_counts(tmp_path):
    a, tx = _setup(tmp_path)
    feats = transcript_context(a, tx, radius=10.0)
    assert feats.shape == (3, 4)
    assert "transcript_context" in a.obsm
    np.testing.assert_allclose(feats[0], np.log1p([5, 0, 1, 0]), atol=1e-5)  # control excluded
    np.testing.assert_allclose(feats[1], np.log1p([0, 3, 0, 0]), atol=1e-5)
    np.testing.assert_allclose(feats[2], np.zeros(4), atol=1e-5)


def test_transcript_context_copy(tmp_path):
    a, tx = _setup(tmp_path)
    out = transcript_context(a, tx, radius=10.0, copy=True)
    assert isinstance(out, ad.AnnData) and "transcript_context" in out.obsm
    assert "transcript_context" not in a.obsm  # original untouched


def test_transcript_context_missing_sample(tmp_path):
    a, _ = _setup(tmp_path)
    with pytest.raises(ValueError, match="transcripts path"):
        transcript_context(a, {"OTHER": "x.parquet"}, radius=10.0)


def test_transcript_context_single_path(tmp_path):
    a, tx = _setup(tmp_path)
    feats = transcript_context(a, tx["S"], radius=10.0)  # single path, not a dict
    assert feats.shape == (3, 4)
