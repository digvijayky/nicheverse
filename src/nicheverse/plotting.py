"""Vector PDF plots: codebook usage, per-sample spatial code maps, training losses.

This module sets a few sensible matplotlib defaults at import time
(``pdf.fonttype = 42`` for editable text, font family Arial when available,
default font size 16). We only switch the backend to ``Agg`` when no display
is available; in Jupyter the user's chosen backend is preserved.
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

import matplotlib
import numpy as np

# Choose backend only if we are not already in a display-bound or notebook
# context. If matplotlib has already loaded a backend (Jupyter, IPython) we
# leave it alone.
_BACKEND_ALREADY_SET = matplotlib.get_backend() != matplotlib.rcParamsDefault["backend"]
if not _BACKEND_ALREADY_SET and not os.environ.get("DISPLAY"):
    with contextlib.suppress(Exception):  # pragma: no cover - extremely rare
        matplotlib.use("Agg", force=False)

import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager

logger = logging.getLogger(__name__)

_ARIAL = Path.home() / ".local/share/fonts/Arial.ttf"
if _ARIAL.exists():
    fontManager.addfont(str(_ARIAL))

FS = 16
plt.rcParams.update(
    {
        "font.size": FS,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelsize": FS,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "legend.fontsize": FS,
    }
)
if _ARIAL.exists():
    plt.rcParams["font.family"] = "Arial"


def _categorical_colors(idx: np.ndarray) -> np.ndarray:
    """Map each distinct integer code to a visually distinct color (HSV wheel)."""
    codes = np.unique(idx)
    palette = plt.cm.hsv(np.linspace(0.0, 1.0, max(len(codes), 1), endpoint=False))
    lut = {int(c): palette[i] for i, c in enumerate(codes)}
    return np.array([lut[int(c)] for c in idx])


def codebook_usage_pdf(cell_idx: np.ndarray, neigh_idx: np.ndarray, save_path: str | Path) -> Path:
    """Save a side-by-side bar chart of cell-code and neighborhood-code usage.

    Parameters
    ----------
    cell_idx, neigh_idx
        Integer code assignments per cell.
    save_path
        Output ``.pdf`` path.

    Returns
    -------
    Path
        The output path written.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    uc, cc = np.unique(cell_idx, return_counts=True)
    axes[0].bar(uc, cc, color="#4C72B0")
    axes[0].set_xlabel("Cell codebook index")
    axes[0].set_ylabel("Cell count")
    axes[0].set_title("Cell codebook usage")
    un, cn = np.unique(neigh_idx, return_counts=True)
    axes[1].bar(un, cn, color="#C44E52")
    axes[1].set_xlabel("Neighborhood codebook index")
    axes[1].set_ylabel("Cell count")
    axes[1].set_title("Neighborhood codebook usage")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    return save_path


def per_sample_spatial_pdf(
    spatial: np.ndarray,
    sample_ids: np.ndarray,
    cell_idx: np.ndarray,
    neigh_idx: np.ndarray,
    save_dir: str | Path,
) -> Path:
    """Write one PDF per sample with cell-code and neighborhood-code spatial maps.

    Parameters
    ----------
    spatial
        ``(n_cells, 2)`` micron coordinates.
    sample_ids
        ``(n_cells,)`` sample label per cell.
    cell_idx, neigh_idx
        Integer code assignments per cell.
    save_dir
        Directory to write the per-sample PDFs into. Created if missing.

    Returns
    -------
    Path
        The output directory.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for sample in np.unique(sample_ids):
        m = sample_ids == sample
        coords = spatial[m]
        cidx = cell_idx[m]
        nidx = neigh_idx[m]
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        for ax, idx_arr, title in (
            (axes[0], cidx, f"{sample}: cell code"),
            (axes[1], nidx, f"{sample}: neighborhood code"),
        ):
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=_categorical_colors(idx_arr),
                s=1,
                alpha=0.7,
                rasterized=True,
            )
            ax.set_title(title)
            ax.set_xlabel("x (um)")
            ax.set_ylabel("y (um)")
            ax.set_aspect("equal")
        plt.tight_layout()
        safe = str(sample).replace("/", "_").replace(" ", "_")
        plt.savefig(save_dir / f"spatial_{safe}.pdf", bbox_inches="tight", format="pdf")
        plt.close(fig)
    return save_dir


def training_loss_pdf(losses: list[dict[str, float]], save_path: str | Path) -> Path:
    """Save the per-epoch total / cell / neighborhood loss curves.

    Parameters
    ----------
    losses
        List of ``{'total', 'cell', 'neighborhood'}`` dicts as written to
        ``training_losses.json``.
    save_path
        Output ``.pdf`` path.

    Returns
    -------
    Path
        The output path written.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ep = range(1, len(losses) + 1)
    total = [d["total"] for d in losses]
    cell = [d["cell"] for d in losses]
    neigh = [d["neighborhood"] for d in losses]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(ep, total, color="#333")
    axes[0].set_title("Total loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].plot(ep, cell, color="#4C72B0")
    axes[1].set_title("Cell loss")
    axes[1].set_xlabel("Epoch")
    axes[2].plot(ep, neigh, color="#C44E52")
    axes[2].set_title("Neighborhood loss")
    axes[2].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    return save_path
