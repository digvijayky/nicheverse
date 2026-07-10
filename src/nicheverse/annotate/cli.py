"""Thin command-line front end for the codebook-annotation harness.

Runnable as ``python -m nicheverse.annotate <command>``. Every subcommand delegates to
the library (``context`` / ``artifacts`` / ``harness`` / ``evaluate``); the CLI only
parses arguments, loads / writes files, and prints a short summary.

Subcommands:

``context-template PATH``
    Write an annotated ``project_context.yaml`` the user can fill in.
``evidence --adata --code-col --out``
    Dump the full per-code evidence bundle (CSVs) for manual review.
``annotate --adata --code-col --context ctx.yaml --kind cell|niche --out``
    Run the full harness and write ``labels.csv`` + ``review.csv`` + the provenance manifest.
``evaluate --labels --code-col --groundtruth ...``
    Score an existing per-code label table against a ground-truth column.

Scope: imaging-based / in-situ spatial transcriptomics (IMST) only.
"""

from __future__ import annotations

import argparse
import os
import sys


def _load_adata(path: str):
    import anndata as ad

    return ad.read_h5ad(path)


def _cmd_context_template(args) -> int:
    from .context import ProjectContext

    ProjectContext.write_template(args.path)
    print(f"wrote project context template to {args.path}")
    return 0


def _cmd_evidence(args) -> int:
    from .artifacts import write_evidence_bundle

    adata = _load_adata(args.adata)
    written = write_evidence_bundle(
        adata,
        args.code_col,
        args.out,
        groundtruth_col=args.groundtruth_col,
        patient_col=args.patient_col,
        context_cols=tuple(args.context_cols or ()),
    )
    print(f"wrote {len(written)} evidence CSV(s) to {args.out}")
    for name, path in written.items():
        print(f"  {name}: {path}")
    return 0


def _cmd_annotate(args) -> int:
    from .annotate import AnnotationConfig
    from .context import ProjectContext
    from .harness import annotate_codebook

    adata = _load_adata(args.adata)
    ctx = ProjectContext.from_yaml(args.context) if args.context else None
    cfg = AnnotationConfig(
        provider=args.provider,
        model=args.model,
        context=ctx,
        tissue=(ctx.tissue if ctx else "") or args.tissue or "",
        context_cols=tuple(ctx.context_cols) if (ctx and ctx.context_cols) else tuple(args.context_cols or ()),
        refuter=not args.no_refuter,
        n_candidates=args.n_candidates,
        min_marker_precision=args.min_marker_precision,
        confidence_review_threshold=args.confidence_review_threshold,
        with_literature=args.with_literature,
    )
    os.makedirs(args.out, exist_ok=True)
    res = annotate_codebook(
        adata,
        args.code_col,
        config=cfg,
        kind=args.kind,
        celltype_col=args.celltype_col,
        groundtruth_col=args.groundtruth_col,
        out_dir=args.out,
        run_meta={"cli": True, "adata": os.path.abspath(args.adata)},
    )
    labels_path = os.path.join(args.out, "labels.csv")
    review_path = os.path.join(args.out, "review.csv")
    res.labels_df.to_csv(labels_path)
    res.review_df.to_csv(review_path)
    n_auto = len(res.labels_df) - len(res.review_df)
    print(f"annotated {len(res.labels_df)} code(s): {n_auto} auto-accepted, {len(res.review_df)} for review")
    print(f"  labels: {labels_path}")
    print(f"  review: {review_path}")
    if res.manifest_path:
        print(f"  manifest: {res.manifest_path}")
    if res.scorecard_df is not None:
        sc_path = os.path.join(args.out, "scorecards.csv")
        res.scorecard_df.to_csv(sc_path, index=False)
        print(f"  scorecards: {sc_path}")
    return 0


def _cmd_evaluate(args) -> int:
    import pandas as pd

    from .evaluate import calibration, score_code, scorecard_table

    adata = _load_adata(args.adata) if args.adata else None
    labels = pd.read_csv(args.labels)
    code_key = args.code_col if args.code_col in labels.columns else labels.columns[0]
    label_key = "final_label" if "final_label" in labels.columns else (
        "label" if "label" in labels.columns else labels.columns[1]
    )

    ev = {}
    gt = {}
    if adata is not None:
        from .artifacts import code_evidence, code_groundtruth_concordance

        ev = code_evidence(adata, args.code_col)
        if args.groundtruth and args.groundtruth in adata.obs.columns:
            _, maj = code_groundtruth_concordance(adata, args.code_col, args.groundtruth)
            gt = {str(r["code"]): r["majority_group"] for _, r in maj.iterrows()}

    scorecards = []
    for _, row in labels.iterrows():
        code = str(row[code_key])
        proposed = {"label": row[label_key], "code": code, "confidence": row.get("confidence")}
        if "key_markers" in labels.columns:
            proposed["key_markers"] = row["key_markers"]
        scorecards.append(score_code(proposed, gt.get(code), ev.get(code, {})))
    table = scorecard_table(scorecards)
    out = args.out or (os.path.splitext(args.labels)[0] + "_scorecards.csv")
    table.to_csv(out, index=False)
    calib = calibration(scorecards)
    print(f"scored {len(scorecards)} code(s) -> {out}")
    if calib.get("spearman") is not None:
        print(f"  confidence/correctness Spearman: {calib['spearman']:.3f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nicheverse.annotate",
        description="Agentic codebook-annotation harness (IMST only).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    ct = sub.add_parser("context-template", help="write an annotated project_context.yaml")
    ct.add_argument("path")
    ct.set_defaults(func=_cmd_context_template)

    ev = sub.add_parser("evidence", help="dump the per-code evidence bundle (CSVs)")
    ev.add_argument("--adata", required=True)
    ev.add_argument("--code-col", required=True)
    ev.add_argument("--out", required=True)
    ev.add_argument("--groundtruth-col", default=None)
    ev.add_argument("--patient-col", default=None)
    ev.add_argument("--context-cols", nargs="*", default=[])
    ev.set_defaults(func=_cmd_evidence)

    an = sub.add_parser("annotate", help="run the full labeler->gate->refuter->reconcile harness")
    an.add_argument("--adata", required=True)
    an.add_argument("--code-col", required=True)
    an.add_argument("--context", default=None, help="project_context.yaml")
    an.add_argument("--kind", choices=("cell", "niche"), default="cell")
    an.add_argument("--celltype-col", default=None, help="required for --kind niche")
    an.add_argument("--groundtruth-col", default=None)
    an.add_argument("--provider", default="anthropic")
    an.add_argument("--model", default=None)
    an.add_argument("--tissue", default="")
    an.add_argument("--context-cols", nargs="*", default=[])
    an.add_argument("--no-refuter", action="store_true", help="skip the adversarial refuter pass")
    an.add_argument("--n-candidates", type=int, default=3)
    an.add_argument("--min-marker-precision", type=float, default=0.5)
    an.add_argument("--confidence-review-threshold", type=float, default=0.6)
    an.add_argument("--with-literature", action="store_true")
    an.add_argument("--out", required=True)
    an.set_defaults(func=_cmd_annotate)

    sc = sub.add_parser("evaluate", help="score an existing per-code label table")
    sc.add_argument("--labels", required=True, help="CSV with a code column and a label column")
    sc.add_argument("--code-col", default="code")
    sc.add_argument("--adata", default=None, help="AnnData for evidence + ground truth")
    sc.add_argument("--groundtruth", default=None, help="obs column with the reference labels")
    sc.add_argument("--out", default=None)
    sc.set_defaults(func=_cmd_evaluate)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
