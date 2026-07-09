"""MCP (Model Context Protocol) server exposing nicheverse as agent tools.

Runs over stdio so Claude Code / Codex can call the nicheverse annotation and
inference API as tools. Every tool takes an ``.h5ad`` path, loads it lazily, and
returns a compact JSON/markdown summary (paths + counts + top rows) rather than
raw matrices.

Register with::

    claude mcp add nicheverse -- \
        /home/yarlagad/conda_envs/annotforimst/bin/python -m nicheverse.mcp_server
"""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("nicheverse")


def _load(adata_path: str):
    import anndata as ad

    return ad.read_h5ad(adata_path)


def _missing_cols(adata, *cols):
    return [c for c in cols if c and c not in adata.obs.columns]


def _col_error(adata, missing):
    return json.dumps(
        {"error": f"obs column(s) not found: {missing}", "available_obs_columns": list(adata.obs.columns)[:40]}
    )


@mcp.tool()
def nicheverse_code_evidence(adata_path: str, code_col: str, extra_cols: str = "") -> str:
    """Summarize per-code marker/DEG evidence for a learned codebook.

    Parameters
    ----------
    adata_path
        Path to an ``.h5ad`` carrying the code assignment in ``obs[code_col]``.
    code_col
        ``obs`` column with the code index (e.g. ``cell_codebook_idx``).
    extra_cols
        Comma-separated ``obs`` columns to report the per-code distribution over
        (e.g. ``"site_class,sample_id"``). Optional.

    Returns a compact JSON string: one entry per code (top 15 codes by cell count)
    with n_cells, fraction, top 10 markers, top 10 DEGs, and any distributions.
    """
    from nicheverse.annotate import code_evidence

    adata = _load(adata_path)
    cols = tuple(c.strip() for c in extra_cols.split(",") if c.strip())
    miss = _missing_cols(adata, code_col, *cols)
    if miss:
        return _col_error(adata, miss)
    ev = code_evidence(adata, code_col, extra_cols=cols)
    ranked = sorted(ev.values(), key=lambda e: e["n_cells"], reverse=True)
    n_total = len(ev)
    out = []
    for e in ranked[:15]:
        row = {
            "code": e["code"],
            "n_cells": e["n_cells"],
            "frac": round(e["frac"], 4),
            "top_markers": [[g, z] for g, z in e["top_markers"][:10]],
        }
        if e.get("top_degs"):
            row["top_degs"] = [[g, lfc] for g, lfc, _ in e["top_degs"][:10]]
        for k, v in e.items():
            if k.startswith("dist_"):
                row[k] = v
        out.append(row)
    return json.dumps(
        {"n_codes": n_total, "shown": len(out), "codes": out}, ensure_ascii=False
    )


@mcp.tool()
def nicheverse_annotate(
    adata_path: str,
    code_col: str,
    provider: str = "anthropic",
    model: str = "",
    tissue: str = "",
    with_literature: bool = False,
    output_csv: str = "",
) -> str:
    """Annotate every code with an LLM, grounded in per-code evidence.

    Runs :func:`nicheverse.annotate.annotate_codes` (optionally with PubMed
    literature grounding), optionally writes the full label table to ``output_csv``,
    and returns the table as a markdown summary.

    Parameters
    ----------
    adata_path
        Path to an ``.h5ad`` with ``obs[code_col]``.
    code_col
        ``obs`` column with the code index.
    provider, model
        LLM backend (``anthropic`` / ``openai`` / ``ollama``) and model id
        (empty = provider default). Needs the relevant API key in the env.
    tissue
        Free-text tissue / disease context (e.g. ``"human clear cell RCC"``).
    with_literature
        If True, search PubMed for each code's top markers and cite hits.
    output_csv
        Optional path to write the full label table as CSV.

    Returns a markdown table of code -> label / compartment / confidence.
    """
    from nicheverse.annotate import annotate_codes

    adata = _load(adata_path)
    miss = _missing_cols(adata, code_col)
    if miss:
        return _col_error(adata, miss)
    df = annotate_codes(
        adata,
        code_col,
        provider=provider,
        model=model or None,
        tissue=tissue,
        with_literature=with_literature,
    )
    written = ""
    if output_csv:
        df.to_csv(output_csv)
        written = f"Wrote {len(df)} rows to {output_csv}\n\n"
    cols = [c for c in ("label", "label_refined", "compartment", "confidence", "n_cells") if c in df.columns]
    view = df[cols].copy()
    header = "| code | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    for code, r in view.head(40).iterrows():
        lines.append("| " + str(code) + " | " + " | ".join(str(r[c]) for c in cols) + " |")
    note = "" if len(df) <= 40 else f"\n({len(df)} codes total; showing 40)"
    return written + "\n".join(lines) + note


@mcp.tool()
def nicheverse_annotate_niches(
    adata_path: str,
    niche_col: str,
    celltype_col: str,
    provider: str = "anthropic",
    model: str = "",
    tissue: str = "",
    output_csv: str = "",
) -> str:
    """Annotate spatial-niche codes by their cell-type composition with an LLM.

    Runs :func:`nicheverse.annotate.annotate_niches`: each niche (neighborhood code)
    is labeled from the community of cell types that co-occur in it (needs
    ``celltype_col``, e.g. the labels produced by annotating the cell codes) plus
    marker enrichment. Optionally writes the full table to ``output_csv``.

    Parameters
    ----------
    adata_path
        Path to an ``.h5ad`` carrying ``obs[niche_col]`` and ``obs[celltype_col]``.
    niche_col
        ``obs`` column with the niche / neighborhood code index (e.g.
        ``neighborhood_codebook_idx``).
    celltype_col
        ``obs`` column with cell-level labels (e.g. ``celltype_annot`` from
        :func:`annotate_codes`).
    provider, model
        LLM backend (``anthropic`` / ``openai`` / ``ollama``) and model id
        (empty = provider default). Needs the relevant API key in the env.
    tissue
        Free-text tissue / disease context (e.g. ``"human clear cell RCC"``).
    output_csv
        Optional path to write the full niche label table as CSV.

    Returns a markdown table of niche -> label / dominant_types / confidence / n_cells.
    """
    from nicheverse.annotate import annotate_niches

    adata = _load(adata_path)
    miss = _missing_cols(adata, niche_col, celltype_col)
    if miss:
        return _col_error(adata, miss)
    df = annotate_niches(
        adata,
        niche_col,
        celltype_col,
        provider=provider,
        model=model or None,
        tissue=tissue,
    )
    written = ""
    if output_csv:
        df.to_csv(output_csv)
        written = f"Wrote {len(df)} rows to {output_csv}\n\n"
    cols = [c for c in ("label", "dominant_types", "confidence", "n_cells") if c in df.columns]
    header = "| niche | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    for code, r in df.head(40).iterrows():
        lines.append("| " + str(code) + " | " + " | ".join(str(r[c]) for c in cols) + " |")
    note = "" if len(df) <= 40 else f"\n({len(df)} niches total; showing 40)"
    return written + "\n".join(lines) + note


@mcp.tool()
def nicheverse_cluster_codes(adata_path: str, code_col: str, n_clusters: int = 0) -> str:
    """Hierarchically group codes by mean-expression correlation for coarse review.

    Runs :func:`nicheverse.annotate.cluster_codes` (correlation distance + average
    linkage) so similar codes can be annotated coarse-to-fine.

    Parameters
    ----------
    adata_path
        Path to an ``.h5ad`` carrying ``obs[code_col]``.
    code_col
        ``obs`` column with the code index (cell or niche).
    n_clusters
        Target number of clusters; ``0`` -> library default (about one cluster per
        eight codes).

    Returns JSON: n_codes, n_clusters, and a compact ``{code: cluster}`` mapping.
    """
    from nicheverse.annotate import cluster_codes

    adata = _load(adata_path)
    miss = _missing_cols(adata, code_col)
    if miss:
        return _col_error(adata, miss)
    df = cluster_codes(adata, code_col, n_clusters=n_clusters or None)
    mapping = {str(code): int(r["cluster"]) for code, r in df.iterrows()}
    return json.dumps(
        {
            "code_col": code_col,
            "n_codes": len(mapping),
            "n_clusters": len(set(mapping.values())),
            "code_to_cluster": mapping,
        },
        ensure_ascii=False,
    )


@mcp.tool()
def nicheverse_predict(
    adata_path: str, checkpoint: str, output_path: str, sample_col: str = "sample_id"
) -> str:
    """Assign cell and neighborhood codebook indices using a trained checkpoint.

    Runs :func:`nicheverse.predict_codes`, writes the annotated AnnData to
    ``output_path``, and returns a compact usage summary.

    Parameters
    ----------
    adata_path
        Path to the input ``.h5ad`` (needs ``obsm['spatial']`` and ``obs[sample_col]``).
    checkpoint
        Path to a ``.pt`` checkpoint written by nicheverse training.
    output_path
        ``.h5ad`` path to write the annotated AnnData.
    sample_col
        ``obs`` column used to partition cells for per-sample k-NN.

    Returns JSON: n_cells, n unique cell/neighborhood codes, and the top-15
    most-used cell codes.
    """
    from nicheverse import predict_codes

    adata = _load(adata_path)
    out = predict_codes(adata, checkpoint, output_path=output_path, sample_col=sample_col)
    cell_vc = out.obs["cell_codebook_idx"].astype(str).value_counts()
    return json.dumps(
        {
            "output_path": output_path,
            "n_cells": int(out.n_obs),
            "n_cell_codes_used": int(cell_vc.shape[0]),
            "n_neighborhood_codes_used": int(
                out.obs["neighborhood_codebook_idx"].astype(str).nunique()
            ),
            "top_cell_codes": [[str(k), int(v)] for k, v in cell_vc.head(15).items()],
        },
        ensure_ascii=False,
    )


@mcp.tool()
def nicheverse_pubmed(query: str, max_results: int = 4) -> str:
    """Search PubMed for literature grounding of a marker or cell-type call.

    Returns JSON: a list of ``{pmid, title, journal, year, first_author}`` hits.
    """
    from nicheverse.annotate import pubmed_search

    return json.dumps(pubmed_search(query, max_results=max_results), ensure_ascii=False)


def main() -> None:
    """Run the nicheverse MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
