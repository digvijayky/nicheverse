"""Reconstruction likelihoods and optional spatial-coherence losses.

This module holds two families of loss functions used by
:func:`nicheverse.train_model`:

- Count reconstruction likelihoods, :func:`nb_nll` and :func:`poisson_nll`,
  selected by ``ModelConfig.recon`` when the decoder models raw counts instead
  of Gaussian (MSE) residuals.
- Spatial-coherence regularizers, :func:`laplacian_smoothness`,
  :func:`spatial_contrastive` and :func:`codebook_consistency`, collected in the
  :data:`SPATIAL_LOSSES` registry. Each operates on a batch of latent vectors
  ``z`` of shape ``(B, D)`` and the matching cell coordinates ``coords`` of
  shape ``(B, 2)`` (microns), building an intra-batch k-nearest-neighbor graph on
  the fly. They are opt-in: training applies one only when
  ``TrainConfig.spatial_loss_weight > 0`` (default ``0``), so the released
  training path is unaffected.

The spatial losses encourage the cell encoder to produce representations that
vary smoothly across physical space, biasing neighboring cells toward the same
(or nearby) discrete cell-state codes without hard-coding spatial information
into the codebook itself.

Symbols
-------
``B`` batch size, ``D`` latent dimensionality, ``k`` neighbors per cell.

Attributes
----------
SPATIAL_LOSSES : dict[str, Callable]
    Registry mapping a spatial-loss name (``"laplacian"``, ``"contrastive"``,
    ``"codebook_consistency"``) to its function. ``TrainConfig.spatial_loss_type``
    selects the entry used during training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "SPATIAL_LOSSES",
    "nb_nll",
    "poisson_nll",
    "codebook_consistency",
    "laplacian_smoothness",
    "spatial_contrastive",
]


def _knn(coords: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the ``k`` nearest neighbors of each row (excluding itself)."""
    d = torch.cdist(coords, coords)
    d.fill_diagonal_(float("inf"))
    kk = min(k, coords.shape[0] - 1)
    return d.topk(kk, largest=False).indices


def laplacian_smoothness(z: torch.Tensor, coords: torch.Tensor, k: int = 6) -> torch.Tensor:
    """Graph-Laplacian smoothness penalty on latents across spatial neighbors.

    Penalizes the mean squared Euclidean difference between each cell's latent
    and those of its ``k`` spatial nearest neighbors (STAGATE-style Laplacian
    regularization). Minimizing it drives spatially adjacent cells toward similar
    embeddings, which in turn makes them more likely to quantize to the same cell
    state code. It is unbounded below only trivially (constant ``z`` gives 0), so
    it is used with a small weight alongside the reconstruction objective.

    Parameters
    ----------
    z : torch.Tensor
        Latent vectors, shape ``(B, D)``.
    coords : torch.Tensor
        Cell coordinates in microns, shape ``(B, 2)``, used to build the
        intra-batch neighbor graph.
    k : int, default=6
        Number of nearest neighbors per cell (excluding itself). Effectively
        capped at ``B - 1``. Valid range ``>= 1``.

    Returns
    -------
    torch.Tensor
        Scalar smoothness penalty (mean over cells and neighbors).
    """
    nbr = _knn(coords, k)
    return ((z.unsqueeze(1) - z[nbr]) ** 2).sum(-1).mean()


def spatial_contrastive(
    z: torch.Tensor,
    coords: torch.Tensor,
    k: int = 6,
    temperature: float = 0.07,
    num_negatives: int = 64,
) -> torch.Tensor:
    """InfoNCE contrastive loss pulling spatial neighbors together, pushing negatives apart.

    L2-normalizes the latents, then for each cell forms a positive score as the
    mean cosine similarity to its ``k`` spatial neighbors and negative scores
    against ``num_negatives`` randomly sampled in-batch cells, all scaled by
    ``1 / temperature``, and returns the mean InfoNCE loss. Unlike
    :func:`laplacian_smoothness` it is collapse-resistant: the negatives prevent
    the encoder from mapping everything to one point, so it shapes a spatially
    organized yet still discriminative latent space for the cell quantizer.

    Parameters
    ----------
    z : torch.Tensor
        Latent vectors, shape ``(B, D)``.
    coords : torch.Tensor
        Cell coordinates in microns, shape ``(B, 2)``.
    k : int, default=6
        Spatial neighbors treated as positives. Valid range ``>= 1``.
    temperature : float, default=0.07
        Softmax temperature on the similarity logits; lower sharpens the
        contrast. Must be ``> 0``.
    num_negatives : int, default=64
        Number of random in-batch negatives sampled per cell (sampled with
        replacement, so a positive may occasionally recur as a negative).

    Returns
    -------
    torch.Tensor
        Scalar InfoNCE loss averaged over cells.
    """
    z = F.normalize(z, dim=1)
    nbr = _knn(coords, k)
    pos = (z.unsqueeze(1) * z[nbr]).sum(-1) / temperature
    pos_mean = pos.mean(1, keepdim=True)
    neg_idx = torch.randint(0, z.shape[0], (z.shape[0], num_negatives), device=z.device)
    neg = (z.unsqueeze(1) * z[neg_idx]).sum(-1) / temperature
    logits = torch.cat([pos_mean, neg], dim=1)
    return (-pos_mean.squeeze(1) + torch.logsumexp(logits, dim=1)).mean()


def codebook_consistency(
    z: torch.Tensor, coords: torch.Tensor, k: int = 6, margin: float = 0.5
) -> torch.Tensor:
    """Margin hinge penalty on latent distance between spatial neighbors.

    Computes the Euclidean distance between each cell's latent and each of its
    ``k`` spatial neighbors and penalizes only the amount by which that distance
    exceeds ``margin`` (a hinge / ReLU). This is a softer, tolerance-banded
    version of :func:`laplacian_smoothness`: neighbors are allowed to differ up
    to ``margin`` free of charge, so genuine local heterogeneity (e.g. a tumor
    boundary) is not over-smoothed, while grossly divergent neighbors are pulled
    together to stabilize the codebook assignment of adjacent cells.

    Parameters
    ----------
    z : torch.Tensor
        Latent vectors, shape ``(B, D)``.
    coords : torch.Tensor
        Cell coordinates in microns, shape ``(B, 2)``.
    k : int, default=6
        Spatial neighbors per cell. Valid range ``>= 1``.
    margin : float, default=0.5
        Distance tolerance below which neighbor pairs incur no penalty. Larger
        values permit more local variation. Must be ``>= 0``.

    Returns
    -------
    torch.Tensor
        Scalar mean hinge penalty over neighbor pairs.
    """
    nbr = _knn(coords, k)
    diff = (z.unsqueeze(1) - z[nbr]).norm(dim=-1)
    return F.relu(diff - margin).mean()


SPATIAL_LOSSES = {
    "laplacian": laplacian_smoothness,
    "contrastive": spatial_contrastive,
    "codebook_consistency": codebook_consistency,
}


def nb_nll(
    x: torch.Tensor, cr: torch.Tensor, log_theta: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Negative-binomial negative log-likelihood (scVI mean-dispersion parameterization).

    Overdispersed count reconstruction loss selected by ``ModelConfig.recon="nb"``.
    The decoder emits per-gene logits ``cr`` that are softmaxed into a proportion
    over the panel and scaled by the per-cell library size to give the NB mean
    ``mu``; ``exp(log_theta)`` is the per-gene inverse-dispersion. This penalizes
    the model for assigning low probability to the observed integer counts and,
    unlike MSE, respects the mean-variance relationship of molecular counts,
    which better matches sparse Xenium panels. Used in place of the Gaussian
    (MSE) cell reconstruction term when raw counts are modeled directly.

    Parameters
    ----------
    x : torch.Tensor
        Observed raw counts, shape ``(B, N)`` (N genes). Not normalized.
    cr : torch.Tensor
        Per-gene decoder logits, shape ``(B, N)``; softmaxed to proportions and
        scaled by ``x.sum(1)`` (library size).
    log_theta : torch.Tensor
        Per-gene log inverse-dispersion, shape ``(N,)`` (a learned
        ``model.cell_log_theta`` parameter).
    eps : float, default=1e-8
        Numerical floor added inside logs.

    Returns
    -------
    torch.Tensor
        Scalar mean NB negative log-likelihood over the batch.
    """
    library = x.sum(1, keepdim=True)
    mu = torch.softmax(cr, dim=1) * library
    theta = log_theta.exp()
    lg = torch.log(theta + mu + eps)
    res = (
        theta * (torch.log(theta + eps) - lg)
        + x * (torch.log(mu + eps) - lg)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
    )
    return -res.sum(1).mean()


def poisson_nll(x: torch.Tensor, cr: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Poisson negative log-likelihood with a library-size-scaled softmax mean.

    Count reconstruction loss selected by ``ModelConfig.recon="poisson"``. Like
    :func:`nb_nll` the decoder logits ``cr`` are turned into a proportion and
    scaled by the per-cell library size to form the Poisson rate ``mu``, but with
    no separate dispersion parameter (variance equals mean). It is the simpler,
    equidispersed alternative for count reconstruction; prefer :func:`nb_nll`
    when the data are overdispersed.

    Parameters
    ----------
    x : torch.Tensor
        Observed raw counts, shape ``(B, N)``.
    cr : torch.Tensor
        Per-gene decoder logits, shape ``(B, N)``; softmaxed and scaled by the
        library size to form the rate ``mu``.
    eps : float, default=1e-8
        Numerical floor inside the log.

    Returns
    -------
    torch.Tensor
        Scalar mean Poisson negative log-likelihood over the batch.
    """
    library = x.sum(1, keepdim=True)
    mu = torch.softmax(cr, dim=1) * library
    return (mu - x * torch.log(mu + eps)).sum(1).mean()
