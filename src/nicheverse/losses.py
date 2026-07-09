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
    "NICHE_SPATIAL_LOSSES",
    "gaussian_nll",
    "nb_nll",
    "poisson_nll",
    "dirichlet_multinomial_nll",
    "bernoulli_detection_bce",
    "graph_total_variation",
    "codebook_consistency",
    "laplacian_smoothness",
    "spatial_contrastive",
]


def gaussian_nll(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Gaussian reconstruction loss (mean squared error), the default cell/niche head.

    Up to an additive constant and a factor, the negative log-likelihood of a
    Gaussian with fixed unit variance is the mean squared error between the
    prediction and the target, so this is the MSE reconstruction term named as a
    likelihood for uniform, by-name dispatch alongside :func:`nb_nll`,
    :func:`poisson_nll`, and :func:`dirichlet_multinomial_nll`. It is the released
    default for both the cell branch (on the log1p expression) and the niche branch
    (on the aggregated composition), and is numerically identical to
    ``torch.nn.functional.mse_loss(pred, target)`` (mean reduction).

    Parameters
    ----------
    target : torch.Tensor
        Reconstruction target (any shape).
    pred : torch.Tensor
        Predicted reconstruction, same shape as ``target``.

    Returns
    -------
    torch.Tensor
        Scalar mean squared error.
    """
    return F.mse_loss(pred, target)


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


def nb_nll(
    x: torch.Tensor,
    cr: torch.Tensor,
    log_theta: torch.Tensor,
    eps: float = 1e-8,
    library: torch.Tensor | None = None,
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
        scaled by ``library`` to form the NB mean ``mu``.
    log_theta : torch.Tensor
        Per-gene log inverse-dispersion, shape ``(N,)`` (a learned
        ``model.cell_log_theta`` parameter).
    eps : float, default=1e-8
        Numerical floor added inside logs.
    library : torch.Tensor, optional
        Per-cell multiplier for the softmax proportion (the library / GLM offset),
        shape ``(B,)`` or ``(B, 1)``. When ``None`` (default, byte-identical to the
        released path) the observed count sum ``x.sum(1)`` is used, which is the
        standard scVI library. Pass an explicit per-cell size factor to override
        it (the trainer passes the observed total count of the raw-count target so
        the softmax proportion is scaled to the correct count scale even though the
        encoder input was the log1p expression).

    Returns
    -------
    torch.Tensor
        Scalar mean NB negative log-likelihood over the batch.
    """
    # Force fp32 for the lgamma / log / exp / softmax math: torch.lgamma is not
    # fp16-safe, softmax*library can underflow, and log_theta.exp() can overflow in
    # half precision. Disabling autocast + upcasting keeps this correct under AMP and
    # is numerically identical to the fp32 path when AMP is off. See _f32 helper.
    with torch.autocast(device_type=cr.device.type, enabled=False):
        cr = cr.float()
        x = x.float()
        log_theta = log_theta.float()
        library = (
            x.sum(1, keepdim=True) if library is None else library.reshape(-1, 1).float()
        )
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


def poisson_nll(
    x: torch.Tensor, cr: torch.Tensor, eps: float = 1e-8, library: torch.Tensor | None = None
) -> torch.Tensor:
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
    library : torch.Tensor, optional
        Per-cell multiplier for the softmax proportion (the library / GLM offset),
        shape ``(B,)`` or ``(B, 1)``. When ``None`` (default) the observed count
        sum ``x.sum(1)`` is used; pass an explicit per-cell size factor to override
        it. See :func:`nb_nll`.

    Returns
    -------
    torch.Tensor
        Scalar mean Poisson negative log-likelihood over the batch.
    """
    # fp32 for the softmax / log math (fp16-unsafe under AMP); identical when AMP off.
    with torch.autocast(device_type=cr.device.type, enabled=False):
        cr = cr.float()
        x = x.float()
        library = (
            x.sum(1, keepdim=True) if library is None else library.reshape(-1, 1).float()
        )
        mu = torch.softmax(cr, dim=1) * library
        return (mu - x * torch.log(mu + eps)).sum(1).mean()


def bernoulli_detection_bce(x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Binary-cross-entropy detection hurdle on the 0-vs-nonzero mask of counts.

    A per-gene Bernoulli likelihood over whether each gene is DETECTED (count > 0)
    in a cell, using ``logits`` as the detection logits. Summing this with a count
    NLL forms a hurdle model that separates the probability of detection from the
    conditional count magnitude, which better fits the excess zeros of sparse
    imaging panels (Xenium). Opt-in via ``ModelConfig.detection_weight > 0``.

    Parameters
    ----------
    x : torch.Tensor
        Observed raw counts, shape ``(B, N)``. The target mask is ``(x > 0)``.
    logits : torch.Tensor
        Per-gene detection logits, shape ``(B, N)`` (the cell decoder output is
        reused as the logits, so no extra head is needed).

    Returns
    -------
    torch.Tensor
        Scalar mean BCE (summed over genes, averaged over cells).
    """
    # binary_cross_entropy_with_logits is autocast-unsafe in fp16 (PyTorch disallows
    # it under autocast); force fp32. Identical to the fp32 path when AMP is off.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits = logits.float()
        target = (x > 0).float()
        return F.binary_cross_entropy_with_logits(logits, target, reduction="none").sum(1).mean()


def dirichlet_multinomial_nll(
    target: torch.Tensor, logits: torch.Tensor, log_alpha: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Dirichlet-multinomial negative log-likelihood for a compositional target.

    A proper likelihood for a niche composition vector: the decoder ``logits`` are
    softmaxed into a mean composition ``p`` and scaled by a learned per-feature
    concentration ``exp(log_alpha)`` to form the Dirichlet parameters
    ``alpha = p * sum(exp(log_alpha))`` (a mean/precision parameterization). The
    ``target`` is treated as a (possibly fractional) count vector ``c`` with total
    ``N = c.sum(1)``, and the Dirichlet-multinomial log-density is evaluated with
    ``lgamma`` (well-defined for non-integer ``c``, so it handles the continuous
    aggregated-neighbor composition without rounding). Unlike an MSE on the
    composition, it respects that the target is a normalized, overdispersed
    proportion. Opt-in via ``ModelConfig.niche_recon="dirichlet_multinomial"``.

    Parameters
    ----------
    target : torch.Tensor
        COUNT-SCALE compositional target, shape ``(B, N)``, non-negative. Its per-row
        sum is used as the multinomial total ``N``, so this must be on the raw-count
        scale (e.g. the weighted-mean aggregation of the raw neighbor counts), NOT a
        log1p mean whose row sum has no count interpretation. Fractional values are
        allowed (the density is evaluated with ``lgamma``), so a continuous count-scale
        aggregate is fine without rounding.
    logits : torch.Tensor
        Per-feature decoder logits, shape ``(B, N)``; softmaxed to the mean
        composition.
    log_alpha : torch.Tensor
        Per-feature log-concentration, shape ``(N,)`` (a learned
        ``model.niche_log_alpha`` parameter). ``sum(exp(log_alpha))`` sets the
        overall precision.
    eps : float, default=1e-8
        Numerical floor.

    Returns
    -------
    torch.Tensor
        Scalar mean Dirichlet-multinomial NLL over the batch.
    """
    # fp32 for the lgamma / exp / softmax math (fp16-unsafe under AMP); identical when
    # AMP is off.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits = logits.float()
        log_alpha = log_alpha.float()
        c = target.float().clamp_min(0.0)
        N = c.sum(1)  # (B,)
        p = torch.softmax(logits, dim=1)
        conc = log_alpha.exp().sum().clamp_min(eps)  # scalar precision
        alpha = p * conc + eps  # (B, N)
        a0 = alpha.sum(1)  # (B,)
        # log DM(c | alpha) = lgamma(N+1) - sum_i lgamma(c_i+1)
        #                   + lgamma(a0) - lgamma(a0+N)
        #                   + sum_i [ lgamma(c_i + alpha_i) - lgamma(alpha_i) ]
        ll = (
            torch.lgamma(N + 1.0)
            - torch.lgamma(c + 1.0).sum(1)
            + torch.lgamma(a0)
            - torch.lgamma(a0 + N)
            + (torch.lgamma(c + alpha) - torch.lgamma(alpha)).sum(1)
        )
        return -ll.mean()


def graph_total_variation(z: torch.Tensor, coords: torch.Tensor, k: int = 6) -> torch.Tensor:
    """Graph total-variation (L1 fused-lasso) penalty on latents across neighbors.

    The L1 analogue of :func:`laplacian_smoothness`: it penalizes the mean L1
    (not squared-L2) difference between each cell's latent and its ``k`` spatial
    nearest neighbors, ``mean_i mean_j |z_i - z_j|_1``. The L1 form is the fused
    lasso / anisotropic total variation, which promotes piecewise-constant latent
    fields (sharp niche boundaries with flat interiors) rather than the smooth
    gradients an L2 penalty prefers. Registered as spatial loss ``"graph_tv"`` and
    applied to the niche latent over the per-sample in-batch graph; it is opt-in
    (``TrainConfig.spatial_loss_weight > 0``). Note that random mini-batches contain
    few true spatial edges, so a spatial-contiguous sampler would strengthen this
    term (not built here).

    Parameters
    ----------
    z : torch.Tensor
        Latent vectors, shape ``(B, D)``.
    coords : torch.Tensor
        Cell coordinates in microns, shape ``(B, 2)``.
    k : int, default=6
        Nearest neighbors per cell (excluding itself). Capped at ``B - 1``.

    Returns
    -------
    torch.Tensor
        Scalar mean L1 total-variation penalty over neighbor pairs.
    """
    nbr = _knn(coords, k)
    return (z.unsqueeze(1) - z[nbr]).abs().sum(-1).mean()


SPATIAL_LOSSES = {
    "laplacian": laplacian_smoothness,
    "contrastive": spatial_contrastive,
    "codebook_consistency": codebook_consistency,
    "graph_tv": graph_total_variation,
}

#: Spatial-loss names applied to the NICHE (neighborhood) latent instead of the
#: cell latent. ``"graph_tv"`` (L1 fused-lasso total variation) smooths niche
#: assignments across the spatial graph, so the trainer feeds it the neighborhood
#: encoder output; all other entries operate on the cell latent.
NICHE_SPATIAL_LOSSES = frozenset({"graph_tv"})
