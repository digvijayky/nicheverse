"""Lightweight, default-on training metrics + end-of-training curve plotting.

This module holds the small, stdlib-only helpers used by :func:`train_model` to
record the standard vector-quantized / deep-learning training metrics and to draw
a compact multi-panel training-curves PDF at the end of a run. Nothing here is on
the optimization path; the metrics are read from tensors that the loop already
computes (per-code assignment histograms, the current learning rate, the total
gradient norm), so the per-epoch overhead is a handful of cheap reductions.

The metric set follows the conventions reported for vector-quantized models
(van den Oord 2017, VQ-VAE-2 Razavi 2019, VQGAN Esser 2021, MAGVIT-v2) --
total loss + per-component reconstruction / VQ loss, codebook *perplexity*
(``exp(H)``), number of *active* (used) codes, and codebook *usage entropy* --
plus the standard optimization diagnostics logged by mainstream trainers
(nanoGPT-style: learning rate, gradient norm, throughput).
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def codebook_usage_stats(counts: np.ndarray) -> dict[str, float]:
    """Codebook usage summaries from a per-code assignment histogram.

    Parameters
    ----------
    counts
        Length-``K`` array of how many cells were assigned to each of the ``K``
        codebook entries over the epoch (integer counts; may be all zero for a
        never-run codebook).

    Returns
    -------
    dict with keys
        ``active`` (number of codes used at least once),
        ``active_frac`` (``active / K``),
        ``entropy`` (Shannon entropy of the usage distribution, nats),
        ``entropy_norm`` (entropy divided by ``log(K)``, in ``[0, 1]``),
        ``perplexity`` (``exp(entropy)``, the effective number of codes),
        ``gini`` (Gini coefficient of the usage distribution, 0 = uniform usage,
        1 = a single code carries everything).

    All quantities are cheap O(K) NumPy reductions on the histogram, so calling
    this once per epoch adds negligible overhead.
    """
    counts = np.asarray(counts, dtype=np.float64)
    k = int(counts.size)
    total = float(counts.sum())
    if k == 0 or total <= 0:
        return {
            "active": 0.0,
            "active_frac": 0.0,
            "entropy": 0.0,
            "entropy_norm": 0.0,
            "perplexity": 0.0,
            "gini": 0.0,
        }
    p = counts / total
    active = int((counts > 0).sum())
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    entropy_norm = float(entropy / math.log(k)) if k > 1 else 0.0
    perplexity = float(math.exp(entropy))
    # Gini via the sorted-cumulative formula; 0 for a perfectly uniform histogram.
    s = np.sort(counts)
    idx = np.arange(1, k + 1)
    gini = float((2.0 * (idx * s).sum()) / (k * s.sum()) - (k + 1.0) / k)
    return {
        "active": float(active),
        "active_frac": float(active / k),
        "entropy": entropy,
        "entropy_norm": entropy_norm,
        "perplexity": perplexity,
        "gini": max(0.0, gini),
    }


def total_grad_norm(parameters, norm_type: float = 2.0) -> float:
    """Global gradient norm over ``parameters`` (skipping params with no grad).

    Mirrors the norm that ``torch.nn.utils.clip_grad_norm_`` reports, computed
    cheaply from the ``.grad`` tensors already present after ``backward()`` so it
    is available even when gradient clipping is disabled. Returns ``0.0`` when no
    parameter has a gradient (e.g. an all-skipped batch).
    """
    import torch

    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    device = grads[0].device
    norm = torch.norm(
        torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]),
        norm_type,
    )
    val = float(norm.item())
    return val if math.isfinite(val) else 0.0


# Column order for training_metrics.csv. Extra keys present in a metrics dict are
# appended after these in sorted order so the CSV never silently drops a field.
_CSV_PRIMARY = [
    "epoch",
    "total",
    "cell",
    "neighborhood",
    "cell_recon",
    "niche_recon",
    "cell_vq",
    "niche_vq",
    "cell_perplexity",
    "neighborhood_perplexity",
    "cell_active_codes",
    "neighborhood_active_codes",
    "cell_usage_gini",
    "neighborhood_usage_gini",
    "cell_usage_entropy_norm",
    "neighborhood_usage_entropy_norm",
    "learning_rate",
    "grad_norm",
    "val_total",
    "epoch_seconds",
    "cells_per_second",
]


def write_metrics_csv(rows: list[dict[str, float]], path: str | Path) -> None:
    """Write the per-epoch metrics list to a tidy CSV (one row per epoch).

    Columns follow :data:`_CSV_PRIMARY`, then any additional keys seen across the
    rows (sorted) so nothing is dropped. Missing values are written blank.
    """
    if not rows:
        return
    seen: list[str] = list(_CSV_PRIMARY)
    known = set(seen)
    for r in rows:
        for key in r:
            if key not in known:
                known.add(key)
                seen.append(key)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=seen, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in seen})


def plot_training_curves(rows: list[dict[str, float]], save_path: str | Path) -> Path | None:
    """Draw a compact 2x2 training-curves PDF from the per-epoch metrics.

    Panels: (i) total + cell-recon + niche-recon loss vs epoch, (ii) cell &
    neighborhood perplexity vs epoch, (iii) active codes and usage Gini vs epoch,
    (iv) learning rate vs epoch. Uses the repo Arial / PDF (fonttype 42) defaults
    and a single uniform font size across the whole figure. Returns the written
    path, or ``None`` if there is nothing to plot.

    The caller wraps this in try/except; any matplotlib / headless failure logs a
    warning upstream and never aborts training.
    """
    if not rows:
        return None
    # Importing the plotting module sets the Arial + pdf.fonttype=42 rcParams and
    # selects the Agg backend when headless (no DISPLAY). Fall back to a bare
    # matplotlib import if that module is unavailable for any reason.
    try:
        from .. import plotting as _p  # noqa: F401  (import for its rcParams side effects)

        import matplotlib.pyplot as plt

        FS = getattr(_p, "FS", 12)
    except Exception:  # pragma: no cover - defensive
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        FS = 12

    ep = [int(r.get("epoch", i + 1)) for i, r in enumerate(rows)]

    def col(key):
        return [r.get(key, float("nan")) for r in rows]

    save_path = Path(save_path)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    (ax_loss, ax_perp), (ax_code, ax_lr) = axes

    ax_loss.plot(ep, col("total"), label="total", lw=1.8)
    ax_loss.plot(ep, col("cell_recon"), label="cell recon", lw=1.2)
    ax_loss.plot(ep, col("niche_recon"), label="niche recon", lw=1.2)
    ax_loss.set_ylabel("loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend(frameon=False, fontsize=FS - 2)
    ax_loss.set_title("Loss", fontsize=FS)

    ax_perp.plot(ep, col("cell_perplexity"), label="cell", lw=1.5)
    ax_perp.plot(ep, col("neighborhood_perplexity"), label="neighborhood", lw=1.5)
    ax_perp.set_ylabel("perplexity")
    ax_perp.set_xlabel("epoch")
    ax_perp.legend(frameon=False, fontsize=FS - 2)
    ax_perp.set_title("Codebook perplexity", fontsize=FS)

    ax_code.plot(ep, col("cell_active_codes"), label="cell active", lw=1.5)
    ax_code.plot(ep, col("neighborhood_active_codes"), label="niche active", lw=1.5)
    ax_code.set_ylabel("active codes")
    ax_code.set_xlabel("epoch")
    gax = ax_code.twinx()
    gax.plot(ep, col("cell_usage_gini"), color="0.4", ls="--", lw=1.2, label="cell Gini")
    gax.set_ylabel("usage Gini")
    gax.tick_params(labelsize=FS)
    h1, l1 = ax_code.get_legend_handles_labels()
    h2, l2 = gax.get_legend_handles_labels()
    ax_code.legend(h1 + h2, l1 + l2, frameon=False, fontsize=FS - 2)
    ax_code.set_title("Codebook usage", fontsize=FS)

    ax_lr.plot(ep, col("learning_rate"), lw=1.5)
    ax_lr.set_ylabel("learning rate")
    ax_lr.set_xlabel("epoch")
    ax_lr.set_title("Learning rate", fontsize=FS)

    for ax in (ax_loss, ax_perp, ax_code, ax_lr):
        ax.tick_params(labelsize=FS)
        ax.title.set_fontsize(FS)
        ax.xaxis.label.set_fontsize(FS)
        ax.yaxis.label.set_fontsize(FS)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path
