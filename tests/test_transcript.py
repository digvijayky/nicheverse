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


def test_transcript_context_multi_row_group(tmp_path):
    """Molecule tables written with many small row groups (e.g. 10x Xenium
    ``transcripts.parquet``) must read via the batched pyarrow path, not the
    dataset-API column projection that raises ArrowInvalid on such files."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    a = ad.AnnData(X=sp.csr_matrix(np.ones((2, 3), dtype="float32")))
    a.var_names = ["g0", "g1", "g2"]
    a.obs["sample_id"] = ["S", "S"]
    a.obsm["spatial"] = np.array([[0.0, 0.0], [100.0, 100.0]])
    rows = [(1.0, 1.0, "g0")] * 6 + [(101.0, 101.0, "g2")] * 4
    rows += [(1.5, 1.5, "NegControlProbe_1")]  # control, must be dropped
    df = pd.DataFrame(rows, columns=["x_location", "y_location", "feature_name"])
    p = tmp_path / "transcripts.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), p, row_group_size=2)
    assert pq.ParquetFile(p).num_row_groups >= 4  # forces multi-batch read

    feats = transcript_context(a, {"S": str(p)}, radius=10.0)
    np.testing.assert_allclose(feats[0], np.log1p([6, 0, 0]), atol=1e-5)
    np.testing.assert_allclose(feats[1], np.log1p([0, 0, 4]), atol=1e-5)
