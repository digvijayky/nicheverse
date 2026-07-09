"""Annotation-review dotplot: each code's top markers as a dot grid (PDF).

Dot size encodes the fraction of cells in a code that express a gene; dot color
encodes the gene's z-score across codes. Useful for eyeballing whether a proposed
label is supported before committing it.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

__all__ = ["code_dotplot"]


def code_dotplot(
    adata: ad.AnnData,
    code_col: str,
    *,
    top_n: int = 5,
    genes: list[str] | None = None,
    save_path: str | Path | None = None,
    layer: str | None = None,
    font_size: int = 11,
):
    """Plot each code's top markers as a dot grid and save a vector PDF.

    Parameters
    ----------
    adata, code_col
        AnnData and the ``obs`` column holding the code index.
    top_n
        Top markers per code (by z-score) to include when ``genes`` is not given.
    genes
        Explicit gene list to show; otherwise the union of each code's top markers.
    save_path
        Output ``.pdf`` path. If omitted, the Matplotlib figure is returned.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager

    arial = Path.home() / ".local/share/fonts/Arial.ttf"
    if arial.exists():
        fontManager.addfont(str(arial))
        plt.rcParams["font.family"] = "Arial"
    plt.rcParams.update(
        {"font.size": font_size, "axes.labelsize": font_size, "xtick.labelsize": font_size,
         "ytick.labelsize": font_size, "pdf.fonttype": 42}
    )

    labels = adata.obs[code_col].astype(str)
    codes = sorted(labels.unique(), key=lambda c: (len(c), c))
    x = adata.layers[layer] if layer else adata.X
    xd = x.toarray() if sp.issparse(x) else np.asarray(x)
    xd = np.nan_to_num(xd, nan=0.0, posinf=0.0, neginf=0.0)
    allgenes = list(map(str, adata.var_names))
    masks = {c: (labels == c).to_numpy() for c in codes}
    full_means = np.vstack([xd[masks[c]].mean(0) for c in codes])
    full_z = (full_means - full_means.mean(0)) / (full_means.std(0) + 1e-9)
    if genes is None:
        genes = []
        for i in range(len(codes)):
            for j in np.argsort(full_z[i])[::-1][:top_n]:
                g = allgenes[j]
                if g not in genes:
                    genes.append(g)
    gidx = {g: i for i, g in enumerate(allgenes)}
    genes = [g for g in genes if g in gidx]
    if not genes or not codes:
        raise ValueError("no genes or codes to plot")
    cols = [gidx[g] for g in genes]
    means = full_means[:, cols]
    frac = np.vstack([(xd[masks[c]][:, cols] > 0).mean(0) for c in codes])
    z = full_z[:, cols]

    fig, ax = plt.subplots(figsize=(max(6, 0.34 * len(genes) + 1.5), max(3, 0.32 * len(codes) + 1.2)))
    gx, gy = np.meshgrid(np.arange(len(genes)), np.arange(len(codes)))
    dots = ax.scatter(
        gx.ravel(), gy.ravel(), s=frac.ravel() * 120 + 3,
        c=z.ravel(), cmap="RdBu_r", vmin=-2, vmax=2, edgecolors="none",
    )
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=90)
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels([f"code {c}" for c in codes])
    ax.set_xlim(-0.5, len(genes) - 0.5)
    ax.set_ylim(-0.5, len(codes) - 0.5)
    ax.invert_yaxis()
    plt.colorbar(dots, ax=ax, shrink=0.5, label="z-score")
    plt.tight_layout()
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, format="pdf", bbox_inches="tight")
        plt.close(fig)
        return out
    return fig
