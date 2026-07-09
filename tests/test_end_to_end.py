"""End-to-end integration test: the whole pipeline composes on tiny synthetic data.

train -> predict codes -> per-code evidence -> cluster -> LLM annotate (mock) ->
attach labels -> annotate niches -> dotplot review. Exercises the model and the
annotation stack together, catching composition bugs the unit tests miss.
"""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import scipy.sparse as sp

import nicheverse.annotate.annotate as A
from nicheverse.annotate import (
    AnnotationConfig,
    annotate_codes,
    annotate_niches,
    attach_labels,
    cluster_codes,
    code_dotplot,
    code_evidence,
)
from nicheverse.models import ModelConfig
from nicheverse.training import TrainConfig, train_model


def test_full_pipeline(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n, g = 120, 12
    x = rng.poisson(1.0, size=(n, g)).astype("float32")
    x[:40, :3] += 6
    x[40:80, 4:7] += 6
    x[80:, 8:11] += 6
    a = ad.AnnData(X=sp.csr_matrix(x))
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs["sample_id"] = np.array(["S0", "S1"] * (n // 2))
    a.obsm["spatial"] = rng.uniform(0, 100, (n, 2))

    cfg = ModelConfig(
        input_dim=g,
        hidden_dims=(16,),
        cell_embedding_dim=8,
        cell_num_embeddings=8,
        neighborhood_embedding_dim=8,
        neighborhood_num_embeddings=4,
        gene_names=tuple(a.var_names),
    )
    _model, coded = train_model(
        a,
        str(tmp_path / "ck"),
        model_config=cfg,
        train_config=TrainConfig(
            num_epochs=2, batch_size=32, k_neighbors=5, save_best=False, deterministic=False
        ),
    )
    assert "cell_codebook_idx" in coded.obs and "neighborhood_codebook_idx" in coded.obs

    assert code_evidence(coded, "cell_codebook_idx")
    assert "cluster" in cluster_codes(coded, "cell_codebook_idx").columns

    def mock_cell(prompt, **kw):
        if "revisions" in prompt:
            return '{"revisions": {}}'
        return json.dumps(
            {
                "label": "T",
                "compartment": "immune",
                "confidence": 0.8,
                "rationale": "m",
                "key_markers": ["G0"],
                "citations": [],
            }
        )

    monkeypatch.setattr(A, "call_llm", mock_cell)
    labels = annotate_codes(coded, "cell_codebook_idx", AnnotationConfig(cluster_context=True))
    attach_labels(coded, "cell_codebook_idx", labels, key_added="celltype_annot")
    assert "celltype_annot" in coded.obs

    monkeypatch.setattr(
        A,
        "call_llm",
        lambda prompt, **kw: json.dumps(
            {"label": "niche", "dominant_types": ["T"], "confidence": 0.7, "rationale": "c"}
        ),
    )
    assert "label" in annotate_niches(coded, "neighborhood_codebook_idx", "celltype_annot").columns

    out = code_dotplot(coded, "cell_codebook_idx", save_path=str(tmp_path / "d.pdf"))
    assert out.exists() and out.read_bytes()[:4] == b"%PDF"
