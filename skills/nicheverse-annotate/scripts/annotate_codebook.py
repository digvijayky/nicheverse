#!/usr/bin/env python
"""Annotate a nicheverse codebook (cell states / spatial niches) end to end.

Runs steps 2-5 of the nicheverse annotation workflow on one AnnData:
  2. (optional) predict_codes from a trained checkpoint -> obs['cell_codebook_idx'] etc.
  3. per-code marker/DEG/distribution evidence (via code_evidence, inside annotate_codes)
  4. LLM annotation grounded in that evidence + PubMed literature
  5. attach labels onto obs and save the label CSV + annotated .h5ad

Uses only the real public API: nicheverse.{read_spatial, predict_codes} and
nicheverse.annotate.{annotate_codes, attach_labels}. No package logic is modified.

Example
-------
    python annotate_codebook.py --adata adata.h5ad --code-col cell_codebook_idx \
        --checkpoint hierarchical_vqvae_checkpoint.pt --provider anthropic \
        --tissue "human clear cell RCC" --with-literature \
        --context-cols site_class,sample_id --out-csv code_labels.csv
"""
from __future__ import annotations

import argparse
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="annotate_codebook.py",
        description=(
            "Annotate nicheverse codebook codes into cell types/states, grounded in per-code "
            "marker/DEG evidence and literature."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--adata", required=True, help="Input imaging-ST AnnData (.h5ad path).")
    p.add_argument(
        "--code-col",
        default="cell_codebook_idx",
        help="obs column with the code index to annotate (e.g. cell_codebook_idx or neighborhood_codebook_idx).",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Optional trained .pt checkpoint. If given and --code-col is absent from obs, predict_codes runs first.",
    )
    p.add_argument(
        "--sample-col", default="sample_id", help="obs column identifying samples/FOVs (used by predict_codes)."
    )
    p.add_argument(
        "--provider",
        default="anthropic",
        choices=("anthropic", "openai", "ollama"),
        help="LLM backend. Key via ANTHROPIC_API_KEY / OPENAI_API_KEY; ollama uses OLLAMA_HOST.",
    )
    p.add_argument("--model", default=None, help="Model id (None -> provider default: Claude Opus / GPT-4o / llama3).")
    p.add_argument("--tissue", default="", help="Free-text tissue/disease context, e.g. 'human clear cell RCC'.")
    p.add_argument(
        "--with-literature",
        action="store_true",
        help="PubMed-search each code's top markers and feed the hits to the LLM.",
    )
    p.add_argument(
        "--context-cols",
        default="",
        help="Comma-separated obs columns to summarize per code (e.g. 'site_class,sample_id').",
    )
    p.add_argument("--no-refine", action="store_true", help="Skip the second cross-code label reconciliation pass.")
    p.add_argument(
        "--cluster-context",
        action="store_true",
        help="Cluster codes first so the reconciliation pass keeps similar codes consistent (coarse-to-fine).",
    )
    p.add_argument("--key-added", default="celltype_annot", help="obs key to attach the labels under.")
    p.add_argument(
        "--niche-col",
        default=None,
        help="Optional obs column with spatial-niche codes (e.g. neighborhood_codebook_idx). "
        "If given, annotate niches too and write '<out-csv stem>_niches.csv'.",
    )
    p.add_argument(
        "--celltype-col",
        default=None,
        help="obs column with cell-level labels used to compose niches (default: the --key-added value).",
    )
    p.add_argument(
        "--dotplot", default=None, help="Optional path to write a per-code marker dotplot PDF for review."
    )
    p.add_argument("--out-csv", required=True, help="Path to write the per-code label CSV.")
    p.add_argument(
        "--out-h5ad",
        default=None,
        help="Path to write the annotated AnnData (default: <adata stem>_annotated.h5ad).",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    import anndata as ad

    import nicheverse as nv
    from nicheverse.annotate import annotate_codes, annotate_niches, attach_labels

    # Step 1/2: load (standardize coords when a checkpoint is used) and assign codes if needed.
    adata = nv.read_spatial(args.adata, sample_col=args.sample_col) if args.checkpoint else ad.read_h5ad(args.adata)
    if args.checkpoint and args.code_col not in adata.obs:
        adata = nv.predict_codes(adata, args.checkpoint, sample_col=args.sample_col)
    if args.code_col not in adata.obs:
        sys.exit(
            f"[nicheverse] --code-col {args.code_col!r} not found in obs; "
            f"pass --checkpoint to generate codes or choose an existing column."
        )

    context_cols = tuple(c.strip() for c in args.context_cols.split(",") if c.strip())

    # Steps 3-4: per-code evidence (inside annotate_codes) + literature-grounded LLM annotation.
    df = annotate_codes(
        adata,
        args.code_col,
        provider=args.provider,
        model=args.model,
        tissue=args.tissue,
        with_literature=args.with_literature,
        context_cols=context_cols,
        refine=not args.no_refine,
        cluster_context=args.cluster_context,
    )

    out_csv = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv)

    # Step 5: attach labels onto obs and save the annotated AnnData.
    attach_labels(adata, args.code_col, df, key_added=args.key_added)

    if args.dotplot:
        from nicheverse.annotate import code_dotplot

        code_dotplot(adata, args.code_col, save_path=args.dotplot)
        print(f"[nicheverse] review dotplot   -> {os.path.abspath(args.dotplot)}")

    # Step 6 (optional): annotate spatial niches from their cell-type composition.
    niche_csv = None
    if args.niche_col:
        celltype_col = args.celltype_col or args.key_added
        if args.niche_col not in adata.obs:
            sys.exit(
                f"[nicheverse] --niche-col {args.niche_col!r} not found in obs; "
                f"pass --checkpoint to generate niche codes or choose an existing column."
            )
        if celltype_col not in adata.obs:
            sys.exit(f"[nicheverse] --celltype-col {celltype_col!r} not found in obs.")
        ndf = annotate_niches(
            adata,
            args.niche_col,
            celltype_col,
            provider=args.provider,
            model=args.model,
            tissue=args.tissue,
            context_cols=context_cols,
        )
        niche_csv = os.path.splitext(out_csv)[0] + "_niches.csv"
        ndf.to_csv(niche_csv)

    out_h5ad = os.path.abspath(args.out_h5ad or (os.path.splitext(args.adata)[0] + "_annotated.h5ad"))
    os.makedirs(os.path.dirname(out_h5ad) or ".", exist_ok=True)
    adata.write_h5ad(out_h5ad)

    print(f"[nicheverse] annotated {len(df)} codes of {args.code_col!r} using provider={args.provider!r}")
    print(f"[nicheverse] label CSV        -> {out_csv}")
    if niche_csv:
        print(f"[nicheverse] annotated {len(ndf)} niches of {args.niche_col!r} (composed over obs['{celltype_col}'])")
        print(f"[nicheverse] niche CSV        -> {niche_csv}")
    print(f"[nicheverse] annotated AnnData -> {out_h5ad} (obs['{args.key_added}'])")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
