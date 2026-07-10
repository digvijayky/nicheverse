"""Command line entry points: preprocess, train, predict, verify, info.

Exit codes
----------
0
    Success.
1
    Generic failure (unknown action or unhandled exception).
2
    ``verify`` ran but the predicted codes do not match the reference.

Every path-accepting flag takes either a string or a ``pathlib.Path``-style
argument; argparse passes them as strings and the package functions normalize
internally.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np

from .data import load_xenium_cohort
from .models import ModelConfig, load_checkpoint
from .plotting import codebook_usage_pdf, per_sample_spatial_pdf, training_loss_pdf
from .training import TrainConfig, predict_codes, train_model
from .utils import env_snapshot, sha256_array, sha256_file

logger = logging.getLogger("nicheverse.cli")


def _add_preprocess(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser(
        "preprocess",
        help="Merge one or more Xenium output directories into a single .h5ad",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", nargs="+", help="One or more Xenium output directories")
    g.add_argument("--manifest", help="CSV with columns run_dir,sample_id")
    p.add_argument(
        "--sample-id",
        nargs="+",
        default=None,
        help="Optional matching sample IDs (else dir name)",
    )
    p.add_argument("--output", required=True, help="Output .h5ad path")
    p.add_argument(
        "--keep-controls",
        action="store_true",
        help="Keep control / blank / unassigned / codeword probes (dropped by default)",
    )


def _add_train(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("train", help="Train hierarchical VQ-VAE on a preprocessed .h5ad")
    p.add_argument("--input", required=True, help="Path to preprocessed .h5ad")
    p.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Output directory for model + embeddings + figures",
    )
    p.add_argument("--num-epochs", type=int, default=300)
    p.add_argument("--cell-codebook-size", type=int, default=256)
    p.add_argument("--cell-codebook-embdim", type=int, default=64)
    p.add_argument("--neighborhood-codebook-size", type=int, default=32)
    p.add_argument("--neighborhood-codebook-embdim", type=int, default=256)
    p.add_argument("--k-neighbors", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=9)
    p.add_argument("--sample-col", default="sample_id")
    p.add_argument("--no-cross-attention", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip writing the training loss / usage / per-sample spatial PDFs",
    )


def _add_predict(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("predict", help="Assign codebook indices to a new .h5ad using a checkpoint")
    p.add_argument("--input", required=True, help="Preprocessed .h5ad to annotate")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--output", required=True, help="Output .h5ad with codes attached")
    p.add_argument("--sample-col", default="sample_id")
    p.add_argument("--k-neighbors", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--seed", type=int, default=9)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip writing the usage / per-sample spatial PDFs",
    )
    p.add_argument(
        "--report",
        default=None,
        help="Optional JSON path with predicted code SHA256 sums "
        "(consumable by `nicheverse verify --reference <this>.json`)",
    )


def _add_verify(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser(
        "verify",
        help="Compare predicted codes to a reference (SHA256 + per-cell exact match)",
    )
    p.add_argument(
        "--predicted", required=True, help="Annotated .h5ad written by nicheverse predict"
    )
    p.add_argument(
        "--reference",
        required=True,
        help="Reference .h5ad (with cell_codebook_idx, neighborhood_codebook_idx) "
        "or .json with sha256 sums (the contract written by `nicheverse predict --report`)",
    )
    p.add_argument(
        "--report", default=None, help="Optional JSON path to write the verification report"
    )


def _add_info(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("info", help="Print checkpoint metadata + the host environment snapshot")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nicheverse",
        description="NICHEVERSE: hierarchical VQ-VAE codebooks of cell states and spatial niches for Xenium",
    )
    from . import __version__

    parser.add_argument("--version", action="version", version=f"nicheverse {__version__}")
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase logging verbosity (-v: INFO, -vv: DEBUG)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress INFO-level messages (errors and warnings still shown)",
    )
    sub = parser.add_subparsers(dest="action", required=True)
    _add_preprocess(sub)
    _add_train(sub)
    _add_predict(sub)
    _add_verify(sub)
    _add_info(sub)
    return parser


def _configure_logging(verbose: int, quiet: bool) -> None:
    if quiet:
        level = logging.WARNING
    elif verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _write_predict_report(annotated: ad.AnnData, report_path: Path) -> None:
    cell_arr = annotated.obs["cell_codebook_idx"].to_numpy().astype(np.int64)
    neigh_arr = annotated.obs["neighborhood_codebook_idx"].to_numpy().astype(np.int64)
    payload = {
        "n_cells": int(annotated.n_obs),
        "predicted_cell_sha256": sha256_array(cell_arr),
        "predicted_neighborhood_sha256": sha256_array(neigh_arr),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2))


def _run_preprocess(args: argparse.Namespace) -> int:
    run_dirs = args.run_dir if args.manifest is None else None
    merged = load_xenium_cohort(
        run_dirs=run_dirs or [],
        sample_ids=args.sample_id,
        drop_controls=not args.keep_controls,
        manifest=args.manifest,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.write_h5ad(out)
    logger.info("wrote %s shape=%s", out, merged.shape)
    return 0


def _run_train(args: argparse.Namespace) -> int:
    adata = ad.read_h5ad(args.input)
    mc = ModelConfig(
        input_dim=int(adata.X.shape[1]),
        cell_embedding_dim=args.cell_codebook_embdim,
        cell_num_embeddings=args.cell_codebook_size,
        neighborhood_embedding_dim=args.neighborhood_codebook_embdim,
        neighborhood_num_embeddings=args.neighborhood_codebook_size,
        use_cross_attention=not args.no_cross_attention,
        gene_names=tuple(adata.var_names.astype(str)),
    )
    tc = TrainConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        k_neighbors=args.k_neighbors,
        seed=args.seed,
    )
    _model, adata = train_model(
        adata,
        args.checkpoint_dir,
        model_config=mc,
        train_config=tc,
        sample_col=args.sample_col,
        device=args.device,
    )
    if not args.no_figures:
        ck = Path(args.checkpoint_dir)
        losses = json.loads((ck / "training_losses.json").read_text())
        training_loss_pdf(losses, ck / "training_losses.pdf")
        codebook_usage_pdf(
            adata.obs["cell_codebook_idx"].to_numpy(),
            adata.obs["neighborhood_codebook_idx"].to_numpy(),
            ck / "codebook_usage.pdf",
        )
        per_sample_spatial_pdf(
            adata.obsm["spatial"],
            adata.obs[args.sample_col].astype(str).to_numpy(),
            adata.obs["cell_codebook_idx"].to_numpy(),
            adata.obs["neighborhood_codebook_idx"].to_numpy(),
            ck / "per_sample_spatial",
        )
    return 0


def _run_info(args: argparse.Namespace) -> int:
    ck = load_checkpoint(args.checkpoint, device="cpu")
    out = {
        "config": ck.config.to_dict(),
        "n_cell_codes": ck.config.cell_num_embeddings,
        "n_neighborhood_codes": ck.config.neighborhood_num_embeddings,
        "n_genes": ck.config.input_dim,
        "first_5_genes": list(ck.config.gene_names[:5]),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "host_env": env_snapshot(),
    }
    print(json.dumps(out, indent=2))
    return 0


def _run_verify(args: argparse.Namespace) -> int:
    pred = ad.read_h5ad(args.predicted)
    pred_cell = pred.obs["cell_codebook_idx"].to_numpy().astype(np.int64)
    pred_neigh = pred.obs["neighborhood_codebook_idx"].to_numpy().astype(np.int64)
    report: dict[str, object] = {
        "n_cells": int(pred.n_obs),
        "predicted_cell_sha256": sha256_array(pred_cell),
        "predicted_neighborhood_sha256": sha256_array(pred_neigh),
    }
    ref_path = Path(args.reference)
    if ref_path.suffix == ".json":
        ref = json.loads(ref_path.read_text())
        # JSON contract: `predict --report` writes `predicted_cell_sha256` /
        # `predicted_neighborhood_sha256`. `verify` accepts either that key (the
        # canonical name) or `reference_cell_sha256` / `reference_neighborhood_sha256`
        # for backward compatibility with hand-written references.
        ref_cell_sha = ref.get("predicted_cell_sha256") or ref.get("reference_cell_sha256")
        ref_neigh_sha = ref.get("predicted_neighborhood_sha256") or ref.get(
            "reference_neighborhood_sha256"
        )
        report["reference_cell_sha256"] = ref_cell_sha
        report["reference_neighborhood_sha256"] = ref_neigh_sha
        report["cell_sha_match"] = report["predicted_cell_sha256"] == ref_cell_sha
        report["neigh_sha_match"] = report["predicted_neighborhood_sha256"] == ref_neigh_sha
    else:
        ref = ad.read_h5ad(ref_path)
        common = pred.obs_names.intersection(ref.obs_names)
        if len(common) == 0:
            raise ValueError(
                f"No overlapping cell barcodes between {args.predicted} and {ref_path}; "
                "cannot compare."
            )
        ref_cell = ref.obs.loc[common, "cell_codebook_idx"].to_numpy().astype(np.int64)
        ref_neigh = ref.obs.loc[common, "neighborhood_codebook_idx"].to_numpy().astype(np.int64)
        p_cell = pred.obs.loc[common, "cell_codebook_idx"].to_numpy().astype(np.int64)
        p_neigh = pred.obs.loc[common, "neighborhood_codebook_idx"].to_numpy().astype(np.int64)
        report["n_compared"] = len(common)
        report["cell_exact_match_pct"] = float((p_cell == ref_cell).mean() * 100)
        report["neighborhood_exact_match_pct"] = float((p_neigh == ref_neigh).mean() * 100)
        report["reference_cell_sha256"] = sha256_array(ref_cell)
        report["reference_neighborhood_sha256"] = sha256_array(ref_neigh)
        report["cell_sha_match"] = sha256_array(p_cell) == report["reference_cell_sha256"]
        report["neigh_sha_match"] = sha256_array(p_neigh) == report["reference_neighborhood_sha256"]
    print(json.dumps(report, indent=2))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2))
    return 0 if report.get("cell_sha_match") and report.get("neigh_sha_match") else 2


def _run_predict(args: argparse.Namespace) -> int:
    adata = ad.read_h5ad(args.input)
    annotated = predict_codes(
        adata,
        args.checkpoint,
        output_path=args.output,
        sample_col=args.sample_col,
        k_neighbors=args.k_neighbors,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    if not args.no_figures:
        outdir = Path(args.output).parent
        codebook_usage_pdf(
            annotated.obs["cell_codebook_idx"].to_numpy(),
            annotated.obs["neighborhood_codebook_idx"].to_numpy(),
            outdir / "codebook_usage.pdf",
        )
        per_sample_spatial_pdf(
            annotated.obsm["spatial"],
            annotated.obs[args.sample_col].astype(str).to_numpy(),
            annotated.obs["cell_codebook_idx"].to_numpy(),
            annotated.obs["neighborhood_codebook_idx"].to_numpy(),
            outdir / "per_sample_spatial",
        )
    if args.report:
        _write_predict_report(annotated, Path(args.report))
    return 0


_DISPATCH = {
    "preprocess": _run_preprocess,
    "train": _run_train,
    "predict": _run_predict,
    "verify": _run_verify,
    "info": _run_info,
}


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to the requested subcommand.

    Returns
    -------
    int
        Exit code (see module docstring).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose, args.quiet)
    fn = _DISPATCH.get(args.action)
    if fn is None:
        parser.error(f"Unknown action: {args.action}")
        return 1
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
