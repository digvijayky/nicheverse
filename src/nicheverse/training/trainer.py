"""Train the hierarchical VQ-VAE on a preprocessed AnnData.

After training, the checkpoint directory contains the model state dict, a
human-readable config JSON, NPZ archives of both codebooks and the per-cell
continuous and quantized embeddings, the per-epoch loss curve as JSON, and an
AnnData h5ad with code assignments and embeddings attached to ``obs`` and
``obsm``.

Defaults reproduce the released Cancer Cell training run exactly: full-cohort
training with no validation split, no mixed precision, and no gradient
clipping. The optional knobs (``val_fraction``, ``early_stopping_patience``,
``amp``, ``grad_clip``, ``resume_from``) are additive and off by default.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader, Subset

from ..data import SpatialDataset
from ..data.xenium import attach_codes_to_adata
from ..losses import (
    NICHE_SPATIAL_LOSSES,
    SPATIAL_LOSSES,
    bernoulli_detection_bce,
    dirichlet_multinomial_nll,
    gaussian_nll,
    nb_nll,
    poisson_nll,
)
from ..models import HierarchicalVQVAE, ModelConfig, save_checkpoint
from ..utils import seed_everything, write_env_snapshot

logger = logging.getLogger(__name__)

try:  # tqdm is a declared dependency; degrade gracefully if unavailable.
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover

    def tqdm(x, **_):  # type: ignore
        return x


def _round_pow2(x: float) -> int:
    """Round a positive number to the nearest power of two (>= 1)."""
    if x <= 1:
        return 1
    lo = 2 ** int(math.floor(math.log2(x)))
    hi = lo * 2
    return hi if (x - lo) >= (hi - x) else lo


def auto_batch_size(
    input_dim: int,
    device: object | None = None,
    encoder_type: str = "mlp_deep",
    ref_batch: int = 8192,
    ref_dim: int = 732,
    lo: int = 512,
    hi: int = 16384,
    mem_fraction: float = 0.5,
) -> int:
    """Heuristic mini-batch size that scales inversely with panel size.

    The batch is scaled so wider gene panels use proportionally smaller batches:
    ``bs = ref_batch * ref_dim / input_dim``. It is then halved for
    ``encoder_type == "mlp_plr"`` (per-gene PLR embeddings cost more activation
    memory), rounded to the nearest power of two, and clamped to ``[lo, hi]``.

    If ``device`` is CUDA, the batch is additionally clamped so an estimated
    activation footprint stays under ``mem_fraction`` of total GPU memory. The
    per-sample estimate is ``input_dim * (16 if mlp_plr else 8) * 4 bytes`` times
    a safety factor of ~64 to cover activations, gradients, and the neighborhood
    tensor (which is ``2 * input_dim`` wide).

    The heuristic is calibrated against the empirical finding that a batch of
    2048 at ``input_dim=732`` used only ~3 GB of a 143 GB H200 (about 2 percent),
    so 366 to 732 gene panels have large memory headroom for bigger batches.

    Parameters
    ----------
    input_dim
        Number of genes in the model input (doubled internally is accounted for).
    device
        Optional torch device (or string). When CUDA, applies the memory clamp.
    encoder_type
        Encoder backbone; ``"mlp_plr"`` gets the halving + higher per-sample cost.
    ref_batch, ref_dim
        Reference batch / panel size defining the inverse-scaling anchor.
    lo, hi
        Clamp bounds (both should be powers of two).
    mem_fraction
        Fraction of total GPU memory the activation estimate may occupy.

    Returns
    -------
    int
        A power-of-two batch size in ``[lo, hi]``.
    """
    dim = max(int(input_dim), 1)
    bs = ref_batch * ref_dim / dim
    if encoder_type == "mlp_plr":
        bs = bs / 2.0
    bs = _round_pow2(bs)
    bs = int(min(max(bs, lo), hi))
    try:
        dev = torch.device(device) if device is not None and not isinstance(device, torch.device) else device
    except Exception:  # pragma: no cover - defensive against odd device args
        dev = None
    if dev is not None and getattr(dev, "type", None) == "cuda" and torch.cuda.is_available():
        per_sample_bytes = dim * (16 if encoder_type == "mlp_plr" else 8) * 4
        per_sample_bytes *= 64  # activations + grads + neighborhood (2*input_dim)
        total = torch.cuda.get_device_properties(dev).total_memory
        budget = mem_fraction * total
        max_bs = budget / max(per_sample_bytes, 1)
        bs = min(bs, _round_pow2(max_bs))
    return int(min(max(bs, lo), hi))


@dataclass
class TrainConfig:
    """Hyperparameters for :func:`train_model`.

    Parameters
    ----------
    num_epochs
        Number of full passes over the dataset.
    batch_size
        Cells per mini-batch. An ``int`` (default ``2048``, byte-identical to the
        released run) or the string ``"auto"`` to resolve it at train time from
        the panel size via :func:`auto_batch_size`. Only ``"auto"`` triggers the
        adaptive path; any int keeps the released optimization trajectory.
    scale_lr_with_batch
        When ``batch_size="auto"`` and this is ``True`` (default), scale the
        learning rate by ``sqrt(effective_batch / 2048)`` so a larger resolved
        batch keeps a comparable update magnitude. No effect for an int batch.
    learning_rate
        Adam learning rate. Reduced on plateau by ``factor=0.5`` with
        ``patience=5`` epochs (monitors the validation loss when
        ``val_fraction > 0``, else the training loss).
    weight_decay
        AdamW decoupled weight decay coefficient (Loshchilov and Hutter 2019).
        ``0.01`` (default) is the AdamW reference value and is conservative for a
        VQ tokenizer trained on millions of cells (large data, low overfitting
        risk). Applied ONLY when ``decoupled_weight_decay`` is ``True`` (the
        default), and then only to the 2-D-or-wider weight matrices of
        ``nn.Linear`` / ``nn.Conv*`` and the attention QKV projection; every other
        parameter (both VQ codebook embeddings, the molecule-set gene embedding,
        all ``nn.LayerNorm`` / ``nn.BatchNorm`` scales, all biases, and every bare
        ``nn.Parameter`` such as the ``mlp_plr`` periodic-embedding tensors) is
        excluded. Set to ``0`` to recover plain AdamW with no regularization.
    cell_weight, neighborhood_weight
        Multipliers on the two reconstruction + commitment loss branches.
    k_neighbors
        Neighbors used to build the neighborhood feature.
    neighborhood_aggregation
        One of ``{"mean", "weighted_mean", "max", "gaussian", "inverse_square"}``.
    spatial_graph
        Spatial graph backend: ``{"knn", "radius", "delaunay", "alpha_complex"}``.
    radius
        Radius in microns for ``spatial_graph="radius"``.
    bandwidth
        Gaussian kernel bandwidth in microns for
        ``neighborhood_aggregation="gaussian"``.
    lr_schedule
        ``"plateau"`` (default, released) or ``"warmup_cosine"``.
    warmup_steps, min_lr
        Warmup epochs and learning-rate floor for ``"warmup_cosine"``.
    decoupled_weight_decay
        Use selective (decoupled) AdamW weight decay: decay ONLY the weight
        matrices of ``nn.Linear`` / ``nn.Conv*`` (and the attention QKV
        projection), and exclude everything else (both VQ codebook embeddings,
        the molecule-set gene embedding, all norm scales, all biases, and all
        bare ``nn.Parameter`` tensors including the ``mlp_plr`` ``freq`` /
        ``emb_w`` / ``emb_b`` periodic-embedding parameters). ``True`` (default).
        When ``False``, a single AdamW group applies ``weight_decay`` uniformly
        to every parameter (including the codebooks), which is discouraged for a
        VQ tokenizer because decaying codebook vectors toward zero shrinks code
        norms and distorts nearest-code assignment geometry.
    spatial_loss_type, spatial_loss_weight, spatial_loss_k
        Optional spatial-coherence regularizer on the cell embedding
        (``"laplacian"`` / ``"contrastive"`` / ``"codebook_consistency"``), its
        weight (``0`` disables, the default), and its neighbor count. The graph
        is built within each mini-batch and restricted to same-sample pairs, so
        it is most effective with a large ``batch_size``.
    normalize, log1p
        Apply ``sc.pp.normalize_total`` / ``sc.pp.log1p`` to ``adata.X`` before
        training. Skipped if the AnnData is already log-normalized.
    val_fraction
        Fraction of cells held out for a validation loss. ``0`` (default)
        trains on all cells and monitors the training loss (released behavior).
    early_stopping_patience
        Stop if the monitored loss does not improve for this many epochs.
        ``None`` (default) disables early stopping.
    grad_clip
        Max global gradient norm (``torch.nn.utils.clip_grad_norm_``). ``None``
        (default) disables clipping.
    amp
        Enable CUDA automatic mixed precision. ``False`` (default) keeps full
        fp32 for bit-stable determinism.
    resume_from
        Optional checkpoint ``.pt`` path to resume model weights from.
    save_best
        If ``True`` and a validation split is used, also write
        ``best_checkpoint.pt`` at the epoch with the lowest monitored loss.
    seed
        Seed passed to :func:`nicheverse.utils.seed_everything`.
    log_every
        Emit a batch-level log line every ``log_every`` mini-batches.
    deterministic
        Forwarded to :func:`seed_everything`.
    num_workers
        DataLoader workers.
    ddp
        Opt-in multi-GPU DistributedDataParallel. When ``True`` and the process
        was launched under ``torchrun`` (``WORLD_SIZE > 1``), the model is wrapped
        in DDP and the global batch is split across ranks (per-GPU batch =
        ``batch_size // world_size``), so optimization matches a single-GPU run on
        the full global batch within numerical tolerance. The EMA VQ codebook
        statistics are all-reduced so the codebook is rank-correct. Default
        ``False`` and a no-op when ``WORLD_SIZE == 1`` (single-GPU byte-identical).
        See ``DETERMINISM.md`` for the one documented residual (dead-code reset RNG).
    compile_model
        Opt-in ``torch.compile(model)`` for a faster (accuracy-preserving) forward
        / backward. Default ``False``. The compiled graph is numerically equivalent
        to eager for these ops; only kernel fusion changes.
    pin_memory
        Explicit DataLoader ``pin_memory``. ``None`` (default) auto-selects
        ``True`` on CUDA, preserving the released behavior. Verified accuracy
        neutral (byte-identical loss with ``num_workers=0``).
    persistent_workers
        Explicit DataLoader ``persistent_workers``. ``None`` (default) is
        ``False`` so the epoch-by-epoch shuffle order matches ``num_workers=0``
        exactly; set ``True`` to save worker respawn overhead at the cost of a
        different (still valid) shuffle for epochs after the first.
    prefetch_factor
        DataLoader ``prefetch_factor`` (only applies when ``num_workers > 0``).
        ``None`` (default) uses the PyTorch default (2). Verified accuracy neutral.

    Notes on ``num_workers``. ``num_workers=0`` (the default) is byte-identical
    to the released training run. ``pin_memory`` and ``prefetch_factor`` are
    verified to leave the loss trajectory byte-identical. ``num_workers > 0`` is a
    throughput knob for I/O-bound cohorts (large on-disk data); on small in-memory
    cohorts it can shift the trajectory to a different (equally valid) one, so it is
    not guaranteed bit-identical to ``num_workers=0``. Prefer ``num_workers=0`` when
    exact reproduction of the released run is required.
    """

    num_epochs: int = 300
    batch_size: int | str = 2048
    scale_lr_with_batch: bool = True
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    lr_schedule: str = "plateau"
    warmup_steps: int = 0
    min_lr: float = 0.0
    decoupled_weight_decay: bool = True
    spatial_loss_type: str = "none"
    spatial_loss_weight: float = 0.0
    spatial_loss_k: int = 6
    cell_weight: float = 1.0
    neighborhood_weight: float = 1.0
    k_neighbors: int = 20
    neighborhood_aggregation: str = "weighted_mean"
    spatial_graph: str = "knn_radius"
    radius: float | None = 50.0
    bandwidth: float | None = None
    normalize: bool = True
    log1p: bool = True
    val_fraction: float = 0.0
    early_stopping_patience: int | None = None
    grad_clip: float | None = None
    amp: bool = False
    resume_from: str | None = None
    save_best: bool = True
    seed: int = 49
    log_every: int = 10
    deterministic: bool = True
    num_workers: int = 0
    ddp: bool = False
    compile_model: bool = False
    pin_memory: bool | None = None
    persistent_workers: bool | None = None
    prefetch_factor: int | None = None


def _make_grad_scaler(enabled: bool) -> torch.amp.GradScaler:
    """Construct a GradScaler using the modern ``torch.amp`` API, falling back for old torch."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _already_log_normalized(adata: ad.AnnData) -> bool:
    """Heuristic: scanpy records ``normalize_total`` and ``log1p`` in ``adata.uns``."""
    return "log1p" in adata.uns or "normalize_total" in adata.uns


def _looks_like_raw_counts(adata: ad.AnnData) -> bool:
    """True iff ``adata.X`` holds non-negative integer values (raw counts). Checks a
    bounded sample of the (nonzero) values so it is cheap on multi-million-cell matrices."""
    import scipy.sparse as sp

    X = adata.X
    vals = X.data[:200_000] if sp.issparse(X) else np.asarray(X[:5000]).ravel()
    if vals.size == 0:
        return True  # all-zero matrix: treat as counts
    return bool(np.nanmin(vals) >= 0 and np.allclose(vals, np.round(vals)))


def _preprocess(adata: ad.AnnData, normalize: bool, log1p: bool) -> ad.AnnData:
    if not (normalize or log1p):
        return adata
    # Guard 1: scanpy provenance marker says the matrix is already processed -> skip.
    if _already_log_normalized(adata):
        logger.warning(
            "AnnData appears to be already log-normalized (uns has 'log1p' or "
            "'normalize_total'); skipping requested normalize/log1p to avoid double normalization."
        )
        return adata
    # Guard 2: no marker, but X is not integer-valued -> almost certainly already
    # transformed; refuse to (double-)normalize rather than silently corrupt the input.
    if not _looks_like_raw_counts(adata):
        raise ValueError(
            "normalize/log1p was requested but adata.X does not look like raw counts "
            "(non-integer or negative values) and carries no scanpy normalization marker. "
            "Refusing to transform to avoid double-normalizing an already-processed matrix. "
            "Pass normalize=False, log1p=False if the input is already normalized; "
            "otherwise ensure adata.X holds integer counts."
        )
    if normalize:
        sc.pp.normalize_total(adata)
    if log1p:
        sc.pp.log1p(adata)
    return adata


def _make_loader(
    dataset: SpatialDataset | Subset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
    sampler: object | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
) -> DataLoader:
    """Build a DataLoader with a seeded generator so shuffling is reproducible.

    When ``sampler`` is given (DDP :class:`DistributedSampler`), ``shuffle`` must be
    ``False`` (the sampler owns the shuffle) and the generator is not used.

    ``persistent_workers`` defaults to ``False`` (even when ``num_workers > 0``).
    This is deliberate and accuracy-preserving: with persistent workers the shuffle
    generator is captured per worker at spawn and epochs after the first draw a
    different (still valid) permutation than a ``num_workers=0`` run, whereas with
    ``persistent_workers=False`` the epoch-by-epoch shuffle order is IDENTICAL to
    ``num_workers=0``. Set ``persistent_workers=True`` explicitly to trade that exact
    shuffle reproducibility for slightly lower per-epoch worker spawn overhead.
    ``prefetch_factor`` is pure prefetch depth and never changes accuracy.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    # Accuracy-neutral default: persistent_workers OFF keeps the shuffle order
    # identical to num_workers=0 across all epochs (see docstring).
    pw = False if persistent_workers is None else (persistent_workers and num_workers > 0)
    kwargs: dict = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=pw,
    )
    if prefetch_factor is not None and num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    if sampler is not None:
        kwargs["sampler"] = sampler
    else:
        kwargs["shuffle"] = shuffle
        kwargs["generator"] = gen if shuffle else None
    return DataLoader(dataset, **kwargs)


def _split_decay_params(model: torch.nn.Module) -> tuple[list, list, list, list]:
    """Classify trainable parameters into weight-decay and no-decay groups.

    Selective (decoupled) weight decay following the community convention shared
    by nanoGPT, HuggingFace Transformers, and timm, adapted for a VQ tokenizer.
    Weight decay is applied ONLY to the 2-D-or-wider ``weight`` matrices of
    ``nn.Linear`` and ``nn.Conv*`` modules, plus the attention QKV projection
    (``nn.MultiheadAttention.in_proj_weight`` / ``out_proj.weight``), which are
    genuine matmul weights. Everything else is excluded from decay:

    * both VQ codebook embeddings and the molecule-set gene embedding (all
      ``nn.Embedding`` weights) -- decaying codebook vectors toward zero shrinks
      code norms and distorts nearest-code assignment geometry (VQGAN /
      taming-transformers applies zero weight decay to the codebook);
    * all ``nn.LayerNorm`` / ``nn.GroupNorm`` / ``nn.BatchNorm*`` scales and shifts;
    * all biases (Linear/Conv/attention biases);
    * every bare ``nn.Parameter`` not owned by a Linear/Conv (this catches the
      ``mlp_plr`` periodic-embedding tensors ``freq`` / ``emb_w`` / ``emb_b``,
      Set-Transformer inducing points / seeds, positional and class tokens,
      LayerScale gates, the NB dispersion ``cell_log_theta``, and any future
      standalone parameter).

    Classification walks ``named_modules()`` and assigns each parameter by its
    OWNING module type, deduplicating by parameter ``id`` so a parameter appears
    in exactly one group even though ``named_modules`` visits parent containers.

    Returns
    -------
    (decay, no_decay, decay_names, no_decay_names)
        Two parameter lists and their matching fully qualified name lists.
    """
    _NORM_TYPES = (
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.LocalResponseNorm,
    )
    # Cover BatchNorm / InstanceNorm (1d/2d/3d) via their shared base classes.
    from torch.nn.modules.batchnorm import _NormBase as _BNBase
    from torch.nn.modules.instancenorm import _InstanceNorm as _INBase

    decay, no_decay = [], []
    decay_names, no_decay_names = [], []
    seen: set[int] = set()

    def _add(param, name, to_decay):
        if not param.requires_grad or id(param) in seen:
            return
        seen.add(id(param))
        (decay if to_decay else no_decay).append(param)
        (decay_names if to_decay else no_decay_names).append(name)

    for mod_name, module in model.named_modules():
        prefix = f"{mod_name}." if mod_name else ""
        if isinstance(module, (torch.nn.Linear, torch.nn.modules.conv._ConvNd)):
            # weight is a >=2-D matmul kernel -> decay; bias -> no-decay.
            if module.weight is not None and module.weight.ndim >= 2:
                _add(module.weight, prefix + "weight", True)
            elif module.weight is not None:
                _add(module.weight, prefix + "weight", False)
            if getattr(module, "bias", None) is not None:
                _add(module.bias, prefix + "bias", False)
        elif isinstance(module, torch.nn.MultiheadAttention):
            # QKV / output projection matrices are matmul weights -> decay;
            # their biases -> no-decay. Handle packed and unpacked projections.
            for attr in ("in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"):
                p = getattr(module, attr, None)
                if p is not None:
                    _add(p, prefix + attr, True)
            for attr in ("in_proj_bias", "bias_k", "bias_v"):
                p = getattr(module, attr, None)
                if p is not None:
                    _add(p, prefix + attr, False)
            # out_proj is a NonDynamicallyQuantizableLinear child; classified here
            # explicitly because we skip descending into it via _own_params below.
            op = getattr(module, "out_proj", None)
            if op is not None:
                if getattr(op, "weight", None) is not None:
                    _add(op.weight, prefix + "out_proj.weight", True)
                if getattr(op, "bias", None) is not None:
                    _add(op.bias, prefix + "out_proj.bias", False)
        else:
            # Norms, embeddings, and any container/leaf with bare parameters:
            # decay NOTHING. Use non-recursive params so each bare parameter is
            # attributed to the single module that directly owns it.
            for pname, param in module.named_parameters(recurse=False):
                _add(param, prefix + pname, False)
    return decay, no_decay, decay_names, no_decay_names


def _build_optimizer(
    model: torch.nn.Module, tc: TrainConfig, lr: float | None = None
) -> torch.optim.Optimizer:
    """AdamW with optional selective (decoupled) weight decay.

    When ``tc.decoupled_weight_decay`` (the default), weight decay is applied only
    to Linear/Conv weight matrices and the attention projection, and every other
    parameter (both VQ codebooks, embeddings, norms, biases, and bare parameters
    such as the ``mlp_plr`` periodic embeddings) is excluded. See
    :func:`_split_decay_params`. When ``False``, a single AdamW group applies
    ``tc.weight_decay`` uniformly to all parameters (legacy behavior).

    ``lr`` overrides ``tc.learning_rate`` (used for the auto-batch sqrt LR scaling,
    which must NOT mutate ``tc``). When ``None`` the nominal ``tc.learning_rate``
    is used.
    """
    lr = tc.learning_rate if lr is None else lr
    if tc.decoupled_weight_decay:
        decay, no_decay, _, _ = _split_decay_params(model)
        groups = [
            {"params": decay, "weight_decay": tc.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tc.weight_decay)


def _build_scheduler(
    optim: torch.optim.Optimizer, tc: TrainConfig, lr: float | None = None
) -> tuple[object, bool]:
    """Return ``(scheduler, needs_monitor)``; default is the released ReduceLROnPlateau.

    ``lr`` overrides ``tc.learning_rate`` for the ``warmup_cosine`` floor ratio so
    the schedule is consistent with the effective (possibly batch-scaled) lr; the
    ``ReduceLROnPlateau`` default does not depend on ``lr``.
    """
    lr = tc.learning_rate if lr is None else lr
    if tc.lr_schedule == "warmup_cosine":
        floor = tc.min_lr / lr if lr > 0 else 0.0

        def lr_lambda(epoch):
            if epoch < tc.warmup_steps:
                return (epoch + 1) / max(1, tc.warmup_steps)
            prog = (epoch - tc.warmup_steps) / max(1, tc.num_epochs - tc.warmup_steps)
            cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
            return floor + (1.0 - floor) * cos

        return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda), False
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optim, "min", patience=5, factor=0.5), True


def _cell_recon(
    model: HierarchicalVQVAE,
    cr: torch.Tensor,
    cb: torch.Tensor,
    recon_target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cell reconstruction loss.

    The DEFAULT (``config.cell_recon="nb"``) is a count likelihood on the RAW counts
    plus a detection hurdle, NO MSE on the cell branch:

    ``cell_loss = per_gene_mean(NB_NLL(raw)) + detection_weight * per_gene_mean(BCE)``

    The NB NLL uses a softmax-proportion decoder scaled by the OBSERVED total count
    per cell (``recon_target.sum(1)``, the scVI library). The NB and BCE terms are
    each divided by ``input_dim`` (turned into per-gene means) so they are on a
    comparable O(1) scale and only ``detection_weight`` balances NB vs BCE; there is
    no log-space-vs-count-space mixing because there is no MSE term.

    Other selectable modes: ``"mse"`` = pure MSE-on-log1p (with detection off this
    is the released MSE-only path); ``"poisson"`` = pure Poisson NLL; ``"both"`` =
    ``w_mse * MSE(log1p) + w_nb * per_gene_mean(NB)`` (the NB term reduced to a
    per-gene mean so it is comparable to the per-element MSE at unit weights);
    ``"default"`` = defer to ``config.recon`` on the encoder input directly. The
    detection hurdle is added to any count mode when ``detection_weight > 0``.
    """
    cm = getattr(model.config, "cell_recon", "default")
    if cm in ("nb", "poisson", "both"):
        if recon_target is None:
            raise ValueError(
                f"cell_recon={cm!r} requires a raw-count recon_target; the trainer did not "
                "provide one."
            )
        d = float(recon_target.shape[1])  # n_genes, for per-gene-mean reduction
        library = recon_target.sum(1, keepdim=True)  # observed total count (scVI library)
        if cm == "poisson":
            count_nll = poisson_nll(recon_target, cr, library=library)
        else:
            count_nll = nb_nll(recon_target, cr, model.cell_log_theta, library=library)
        if cm == "both":
            # Balance MSE-on-log1p (per-element mean) against the NB term (reduced to
            # a per-gene mean) so neither dominates; weights default to 1.0.
            w_mse = float(getattr(model.config, "w_mse", 1.0))
            w_nb = float(getattr(model.config, "w_nb", 1.0))
            loss = w_mse * gaussian_nll(cb, cr) + w_nb * (count_nll / d)
        else:
            # nb / poisson default: per-gene mean of the count NLL.
            loss = count_nll / d
        dw = float(getattr(model.config, "detection_weight", 0.0))
        if dw > 0:
            loss = loss + dw * (bernoulli_detection_bce(recon_target, cr) / d)
        return loss
    recon = model.config.recon
    if recon == "mse":
        return gaussian_nll(cb, cr)
    if recon == "nb":
        return nb_nll(cb, cr, model.cell_log_theta)
    return poisson_nll(cb, cr)


def _niche_recon(
    model: HierarchicalVQVAE,
    nr: torch.Tensor,
    nb: torch.Tensor,
    niche_count_target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Neighborhood reconstruction loss.

    The DEFAULT (``config.niche_recon="mse_dirmult"``) is a balanced sum of the
    composition MSE over the full log1p niche vector and a Dirichlet-multinomial NLL
    on the COUNT-SCALE aggregated-neighbor composition, the DirMult reduced to a
    per-feature mean so it is comparable to the MSE:

    ``niche_loss = w_niche_mse * MSE(nb, nr) + w_dirmult * per_feature_mean(DirMult(niche_count_target, comp_logits))``

    The MSE term is on the log1p features (``nb`` / ``nr``) exactly as released. The
    DirMult target is ``niche_count_target`` -- the SAME weighted-mean neighbor
    aggregation applied to the RAW counts (a count-scale composition whose row sum is
    a real transcript total), NOT the log1p mean. ``comp_logits = nr[:, d:]`` (the
    aggregated-neighbor half of the decoder output) are the DM composition logits.

    ``"mse"`` is the released composition MSE (pure, no count target needed).
    ``"dirichlet_multinomial"`` is a pure DirMult on the count-scale composition plus
    an MSE on the self half (log1p). Both DM modes require ``niche_count_target``
    (they need raw integer counts); a clear error is raised when it is missing.
    """
    mode = getattr(model.config, "niche_recon", "mse")
    if mode == "mse":
        return gaussian_nll(nb, nr)
    if niche_count_target is None:
        raise ValueError(
            f"niche_recon={mode!r} needs a count-scale niche_count_target (the raw-count "
            "weighted-mean neighbor composition), but none was provided. The Dirichlet-"
            "multinomial niche modes require raw integer counts; run with a count cell mode "
            "(cell_recon in {'nb','poisson','both'}) so the raw counts are available, or set "
            "niche_recon='mse'."
        )
    d = model.config.input_dim
    comp_logits = nr[:, d:]  # aggregated-neighbor half of the decoder output
    dirmult = dirichlet_multinomial_nll(
        niche_count_target, comp_logits, model.niche_log_alpha
    ) / float(d)
    if mode == "mse_dirmult":
        w_mse = float(getattr(model.config, "w_niche_mse", 1.0))
        w_dm = float(getattr(model.config, "w_dirmult", 1.0))
        return w_mse * gaussian_nll(nb, nr) + w_dm * dirmult
    # pure "dirichlet_multinomial": DirMult on the count-scale composition, MSE on the
    # self (log1p) half so the self-expression part of the target is still fit.
    self_half, self_tgt = nr[:, :d], nb[:, :d]
    return gaussian_nll(self_tgt, self_half) + dirmult


def train_model(
    adata: ad.AnnData,
    checkpoint_dir: str | Path,
    model_config: ModelConfig | None = None,
    train_config: TrainConfig | None = None,
    sample_col: str = "sample_id",
    device: str | None = None,
) -> tuple[HierarchicalVQVAE, ad.AnnData]:
    """Train :class:`HierarchicalVQVAE` on ``adata`` and persist checkpoint + outputs.

    Parameters
    ----------
    adata
        Input AnnData. Required: ``adata.obsm['spatial']`` (microns) and
        ``adata.obs[sample_col]``. ``adata.X`` is expected to be raw counts;
        per-cell normalization and log1p are applied per ``train_config``.
        ``adata`` is copied before mutation; the original is left untouched.
    checkpoint_dir
        Output directory for the checkpoint, embeddings, loss curve, environment
        snapshot, and annotated AnnData. Created if missing.
    model_config
        Optional :class:`ModelConfig`. Defaults to a config with
        ``input_dim = adata.X.shape[1]`` and ``gene_names = adata.var_names``.
    train_config
        Optional :class:`TrainConfig`. Defaults to the constructor defaults.
    sample_col
        Column in ``adata.obs`` used to partition cells for the per-sample graph.
    device
        Optional explicit device string (e.g. ``"cuda:0"``). Defaults to the
        first CUDA device if available, else CPU.

    Returns
    -------
    (HierarchicalVQVAE, AnnData)
        The trained model and the annotated AnnData copy with code indices and
        embeddings attached.

    Raises
    ------
    ValueError
        If ``adata.obsm['spatial']`` or ``adata.obs[sample_col]`` is missing,
        or if ``model_config.input_dim`` disagrees with ``adata.X.shape[1]``.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tc = train_config or TrainConfig()
    if tc.lr_schedule not in ("plateau", "warmup_cosine"):
        raise ValueError(f"lr_schedule must be plateau or warmup_cosine, got {tc.lr_schedule!r}")
    seed_everything(tc.seed, deterministic=tc.deterministic)
    write_env_snapshot(checkpoint_dir / "env_snapshot.json")
    (checkpoint_dir / "train_config.json").write_text(json.dumps(asdict(tc), indent=2))

    if tc.spatial_loss_weight > 0 and tc.spatial_loss_type not in SPATIAL_LOSSES:
        raise ValueError(
            f"spatial_loss_type must be one of {sorted(SPATIAL_LOSSES)}, got {tc.spatial_loss_type!r}"
        )
    if "spatial" not in adata.obsm:
        raise ValueError(
            "adata.obsm['spatial'] missing. Use load_xenium_cohort() to read "
            "Xenium runs, or set adata.obsm['spatial'] to an (n_cells, 2) array of microns."
        )
    if sample_col not in adata.obs.columns:
        raise ValueError(
            f"adata.obs['{sample_col}'] missing. Either rename your sample column "
            f"to '{sample_col}' or pass sample_col=<your_column_name>."
        )
    if not 0.0 <= tc.val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {tc.val_fraction}")
    _cell_recon_mode = (
        getattr(model_config, "cell_recon", "default")
        if model_config is not None
        else ModelConfig.__dataclass_fields__["cell_recon"].default
    )
    # Count-likelihood cell modes: encoder input stays log1p, but the likelihood is
    # evaluated on the RAW counts (captured to a layer below), scaled by the
    # observed total count per cell (the scVI library). No external size factor.
    _count_mode = _cell_recon_mode in ("nb", "poisson", "both")
    _niche_mode = (
        getattr(model_config, "niche_recon", "mse")
        if model_config is not None
        else ModelConfig.__dataclass_fields__["niche_recon"].default
    )
    # Dirichlet-multinomial niche modes need the COUNT-SCALE neighbor composition,
    # which the dataset only builds when a raw-count recon_target is present (i.e. a
    # count cell mode). Fail loudly up front if a DM niche mode is selected without
    # raw counts, mirroring the cell count-mode guard.
    _niche_dirmult_mode = _niche_mode in ("dirichlet_multinomial", "mse_dirmult")
    if _niche_dirmult_mode and not _count_mode:
        raise ValueError(
            f"niche_recon={_niche_mode!r} (Dirichlet-multinomial) needs the count-scale "
            "neighbor composition, which is only built for a count cell mode "
            "(cell_recon in {'nb','poisson','both'}) so raw integer counts are available. "
            f"Got cell_recon={_cell_recon_mode!r}. Set a count cell_recon, or niche_recon='mse'."
        )
    if (
        model_config is not None
        and model_config.recon in ("nb", "poisson")
        and (tc.normalize or tc.log1p)
    ):
        raise ValueError(
            f"recon={model_config.recon!r} expects raw counts; set "
            "TrainConfig(normalize=False, log1p=False)."
        )
    if _count_mode and not (tc.normalize and tc.log1p):
        raise ValueError(
            f"cell_recon={_cell_recon_mode!r} needs log1p encoder input; set "
            "TrainConfig(normalize=True, log1p=True). The raw counts are captured into a "
            "layer before preprocessing and used as the count reconstruction target."
        )

    adata = adata.copy()
    # Capture raw counts BEFORE log1p so the count modes can reconstruct them while
    # the encoder still sees the log1p input. Guarded by _count_mode so the released
    # MSE / nb / poisson paths add no layer and stay byte-identical.
    if _count_mode:
        if not _looks_like_raw_counts(adata):
            raise ValueError(
                f"cell_recon={_cell_recon_mode!r} needs raw integer counts in adata.X to build "
                "the count reconstruction target, but adata.X does not look like raw counts."
            )
        import scipy.sparse as sp

        adata.layers["_raw_counts"] = adata.X.copy() if sp.issparse(adata.X) else np.asarray(
            adata.X
        ).copy()
    _preprocess(adata, tc.normalize, tc.log1p)

    input_dim = int(adata.X.shape[1])
    if model_config is None:
        model_config = ModelConfig(
            input_dim=input_dim, gene_names=tuple(adata.var_names.astype(str))
        )
    if model_config.input_dim != input_dim:
        raise ValueError(
            f"model_config.input_dim={model_config.input_dim} but adata has {input_dim} genes. "
            "Either pass a matching ModelConfig or omit model_config to use defaults."
        )
    if model_config.gene_names and tuple(model_config.gene_names) != tuple(
        adata.var_names.astype(str)
    ):
        raise ValueError(
            "model_config.gene_names does not match adata.var_names in the same order. "
            "Reorder adata to the config panel, or omit model_config/gene_names to use adata order."
        )
    _xvals = adata.X.data if hasattr(adata.X, "indptr") else np.asarray(adata.X)
    if not np.isfinite(_xvals).all():
        raise ValueError(
            "adata.X contains non-finite values (NaN/inf) after preprocessing; check for "
            "all-zero cells (normalize_total can yield NaN) or non-finite input counts."
        )

    # Opt-in DDP: activate only when TrainConfig.ddp is set AND the process was
    # launched under torchrun (WORLD_SIZE > 1). Otherwise every branch below is the
    # released single-process path, byte-identical to before this change.
    from .._distributed import (
        broadcast_module_,
        ddp_env_requested,
        get_rank,
        get_world_size,
        init_distributed,
        is_main_process,
    )

    use_ddp = bool(tc.ddp) and ddp_env_requested()
    if use_ddp:
        rank, world_size, local_rank = init_distributed()
        if world_size <= 1:
            use_ddp = False
    if use_ddp:
        device_t = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device_t = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    # Resolve the effective (global) batch size. An int is used verbatim (released
    # trajectory); "auto" resolves from the panel size and, when scale_lr_with_batch,
    # scales the learning rate by sqrt(effective_batch / 2048). "auto" must never
    # reach the DataLoader as a string, so it is resolved to an int here.
    # ``effective_lr`` is what the optimizer actually uses; it equals the nominal
    # ``tc.learning_rate`` unless auto-batch sqrt scaling is applied. We NEVER mutate
    # tc.learning_rate (train_config.json is written from tc and must record the
    # nominal lr); both values are persisted to training_runtime.json below.
    effective_lr = tc.learning_rate
    if isinstance(tc.batch_size, str):
        if tc.batch_size != "auto":
            raise ValueError(f"batch_size must be an int or 'auto', got {tc.batch_size!r}")
        effective_batch = auto_batch_size(input_dim, device_t, model_config.encoder_type)
        if tc.scale_lr_with_batch:
            effective_lr = tc.learning_rate * math.sqrt(effective_batch / 2048.0)
            logger.info(
                "batch_size='auto' resolved to %d (input_dim=%d encoder=%s); "
                "learning_rate %.3g -> %.3g (sqrt scaling vs 2048)",
                effective_batch,
                input_dim,
                model_config.encoder_type,
                tc.learning_rate,
                effective_lr,
            )
        else:
            logger.info(
                "batch_size='auto' resolved to %d (input_dim=%d encoder=%s); lr unchanged",
                effective_batch,
                input_dim,
                model_config.encoder_type,
            )
    else:
        effective_batch = int(tc.batch_size)

    if use_ddp:
        if effective_batch % world_size != 0:
            raise ValueError(
                f"DDP requires batch_size ({effective_batch}) divisible by world_size "
                f"({world_size}) so the global batch stays constant."
            )
        per_gpu_batch = effective_batch // world_size
        logger.info(
            "DDP enabled: rank %d/%d local_rank %d device %s per_gpu_batch %d global_batch %d",
            rank,
            world_size,
            local_rank,
            device_t,
            per_gpu_batch,
            effective_batch,
        )
    else:
        per_gpu_batch = effective_batch
        logger.info("Training on device: %s", device_t)

    spatial_coords_np = np.asarray(adata.obsm["spatial"])
    dataset = SpatialDataset.from_anndata(
        adata,
        sample_col=sample_col,
        k_neighbors=tc.k_neighbors,
        neighborhood_aggregation=tc.neighborhood_aggregation,
        spatial_graph=tc.spatial_graph,
        radius=tc.radius,
        bandwidth=tc.bandwidth,
        recon_target_layer="_raw_counts" if _count_mode else None,
    )

    pin = (device_t.type == "cuda") if tc.pin_memory is None else bool(tc.pin_memory)
    if tc.val_fraction > 0:
        n = len(dataset)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(tc.seed)).tolist()
        n_val = max(1, round(n * tc.val_fraction))
        val_ds = Subset(dataset, perm[:n_val])
        train_ds: SpatialDataset | Subset = Subset(dataset, perm[n_val:])
        val_loader: DataLoader | None = _make_loader(
            val_ds,
            per_gpu_batch,
            False,
            tc.num_workers,
            tc.seed,
            pin,
            persistent_workers=tc.persistent_workers,
            prefetch_factor=tc.prefetch_factor,
        )
        logger.info("Validation split: %d train / %d val cells", n - n_val, n_val)
    else:
        train_ds = dataset
        val_loader = None
    train_sampler = None
    if use_ddp:
        from .._distributed import BatchContiguousDistributedSampler

        # Batch-contiguous sharding so the union of the per-rank batch b equals the
        # single-GPU batch perm[b*B:(b+1)*B] exactly. This makes the gathered global
        # batch (k-means init, EMA all-reduce, diversity term) match single-GPU
        # cell-for-cell. A stock DistributedSampler would interleave and break that.
        train_sampler = BatchContiguousDistributedSampler(
            len(train_ds),
            num_replicas=world_size,
            rank=rank,
            global_batch_size=effective_batch,
            shuffle=True,
            seed=tc.seed,
        )
    loader = _make_loader(
        train_ds,
        per_gpu_batch,
        True,
        tc.num_workers,
        tc.seed,
        pin,
        sampler=train_sampler,
        persistent_workers=tc.persistent_workers,
        prefetch_factor=tc.prefetch_factor,
    )

    sample_codes_np = coord_span = None
    if tc.spatial_loss_weight > 0:
        sample_codes_np = adata.obs[sample_col].astype("category").cat.codes.to_numpy()
        coord_span = float(np.ptp(spatial_coords_np, axis=0).max()) + 1.0

    core = HierarchicalVQVAE(model_config).to(device_t)
    if tc.resume_from:
        from ..models import load_checkpoint

        prev = load_checkpoint(tc.resume_from, device_t, config=model_config)
        core.load_state_dict(prev.state_dict())
        logger.info("Resumed model weights from %s", tc.resume_from)

    # ``core`` is always the underlying HierarchicalVQVAE (for attribute access and
    # saving). ``model`` is the forward-callable, which under DDP / torch.compile is a
    # wrapper. On the single-process default they are the same object.
    if use_ddp:
        # SyncBatchNorm makes the BatchNorm statistics global (across ranks) so the
        # forward pass matches a single-GPU run on the full batch rather than
        # per-rank statistics. Required for cross-rank accuracy equivalence. It only
        # supports CUDA tensors, so on a CPU (gloo) process group we keep plain
        # BatchNorm (per-rank statistics become a documented residual, like dropout).
        if device_t.type == "cuda":
            core = torch.nn.SyncBatchNorm.convert_sync_batchnorm(core)
        # Broadcast rank-0 initial weights so every rank starts identically.
        broadcast_module_(core)
        from torch.nn.parallel import DistributedDataParallel

        ddp_device_ids = [local_rank] if device_t.type == "cuda" else None
        model = DistributedDataParallel(core, device_ids=ddp_device_ids)
    else:
        model = core
    if tc.compile_model:
        model = torch.compile(model)
    # Optimizer must see the underlying parameters (identical set to core's).
    # Pass the effective (possibly batch-scaled) lr explicitly; tc.learning_rate
    # stays the nominal value so the persisted train_config.json is unchanged.
    optim = _build_optimizer(core, tc, lr=effective_lr)
    sched, _sched_needs_monitor = _build_scheduler(optim, tc, lr=effective_lr)
    scaler = _make_grad_scaler(enabled=tc.amp and device_t.type == "cuda")

    losses: list[dict[str, float]] = []
    n_batches = len(loader)
    best_loss = float("inf")
    best_epoch = -1
    no_improve = 0
    if device_t.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_t)
    t_train_start = time.perf_counter()
    epochs_run = 0
    for ep in range(tc.num_epochs):
        epochs_run += 1
        t_epoch_start = time.perf_counter()
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(ep)  # reshuffle per epoch, consistently across ranks
        tl = tcell = tn = tcp = tnp = 0.0
        processed = 0
        for bi, (cb, nb, batch_idx) in enumerate(
            tqdm(
                loader,
                desc=f"epoch {ep + 1}/{tc.num_epochs}",
                leave=False,
                disable=not is_main_process(),
            )
        ):
            if cb.shape[0] < 2:
                continue  # BatchNorm needs > 1 sample; skip a trailing singleton
            processed += 1
            cb = cb.to(device_t)
            nb = nb.to(device_t)
            optim.zero_grad(set_to_none=True)
            rt = nct = None
            if _count_mode:
                bidx = batch_idx.to(dataset.recon_target.device)
                rt = dataset.recon_target[bidx].to(device_t)
            if _niche_dirmult_mode:
                # Index in the full-dataset row space, exactly like recon_target.
                nct = dataset.niche_count_target[batch_idx.to(dataset.niche_count_target.device)].to(
                    device_t
                )
            with torch.autocast(
                device_type=device_t.type, enabled=tc.amp and device_t.type == "cuda"
            ):
                cr, nr, cvq, nvq, _ci, _ni, cp, np_ = model(cb, nb)
                cell_recon = _cell_recon(core, cr, cb, recon_target=rt)
                neigh_recon = _niche_recon(core, nr, nb, niche_count_target=nct)
                loss = tc.cell_weight * (cell_recon + cvq) + tc.neighborhood_weight * (
                    neigh_recon + nvq
                )
                if tc.spatial_loss_weight > 0:
                    bidx = batch_idx.numpy()
                    # Offset each sample's coordinates so the intra-batch neighbor
                    # graph never links cells from different samples.
                    # float64 so the large per-sample offset does not quantize true
                    # micron separations at high sample counts.
                    coords_np = spatial_coords_np[bidx].astype(np.float64).copy()
                    coords_np += sample_codes_np[bidx][:, None] * (coord_span * 10.0)
                    coords_b = torch.as_tensor(coords_np, dtype=torch.float64, device=device_t)
                    # graph_tv smooths the NICHE latent over the spatial graph; all
                    # other spatial losses regularize the cell latent (released path).
                    if tc.spatial_loss_type in NICHE_SPATIAL_LOSSES:
                        z_reg = core.neighborhood_encoder(nb)
                    else:
                        z_reg = core.cell_encoder(cb)
                    loss = loss + tc.spatial_loss_weight * SPATIAL_LOSSES[tc.spatial_loss_type](
                        z_reg, coords_b, k=tc.spatial_loss_k
                    )
            scaler.scale(loss).backward()
            if tc.grad_clip is not None:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(core.parameters(), tc.grad_clip)
            scaler.step(optim)
            scaler.update()
            tl += float(loss.item())
            tcell += float((cell_recon + cvq).item())
            tn += float((neigh_recon + nvq).item())
            tcp += float(cp.item())
            tnp += float(np_.item())
            if bi % tc.log_every == 0:
                logger.info(
                    "epoch %d/%d batch %d/%d loss=%.4f cell_perp=%.1f neigh_perp=%.1f",
                    ep + 1,
                    tc.num_epochs,
                    bi,
                    n_batches,
                    loss.item(),
                    cp.item(),
                    np_.item(),
                )
        if processed == 0:
            raise ValueError(
                "No mini-batch had more than one cell; increase batch_size or provide more cells."
            )
        nb_ = max(processed, 1)
        avg = {
            "total": tl / nb_,
            "cell": tcell / nb_,
            "neighborhood": tn / nb_,
            "cell_perplexity": tcp / nb_,
            "neighborhood_perplexity": tnp / nb_,
        }
        if use_ddp:
            # Average the per-rank epoch metrics into a single global number so the
            # logged curve, the LR scheduler input, and early-stopping decisions are
            # identical on every rank (otherwise LR schedules could diverge).
            import torch.distributed as _dist

            _m = torch.tensor(
                [avg["total"], avg["cell"], avg["neighborhood"],
                 avg["cell_perplexity"], avg["neighborhood_perplexity"]],
                device=device_t,
                dtype=torch.float64,
            )
            _dist.all_reduce(_m, op=_dist.ReduceOp.SUM)
            _m /= world_size
            avg = {
                "total": float(_m[0]),
                "cell": float(_m[1]),
                "neighborhood": float(_m[2]),
                "cell_perplexity": float(_m[3]),
                "neighborhood_perplexity": float(_m[4]),
            }
        monitor = avg["total"]
        if val_loader is not None:
            # Pass the count-target dataset only in the count-recon modes so the
            # released signature (model, loader, device, tc) is used verbatim
            # otherwise (keeps monkeypatch-based tests and external callers working).
            _val_kw = {"count_dataset": dataset} if _count_mode else {}
            avg["val_total"] = _val_loss(core, val_loader, device_t, tc, **_val_kw)
            monitor = avg["val_total"]
        avg["epoch_seconds"] = round(time.perf_counter() - t_epoch_start, 3)
        losses.append(avg)
        sched.step(monitor) if _sched_needs_monitor else sched.step()
        logger.info(
            "epoch %d avg_total=%.4f cell=%.4f neigh=%.4f%s cell_perp=%.1f",
            ep + 1,
            avg["total"],
            avg["cell"],
            avg["neighborhood"],
            f" val={avg['val_total']:.4f}" if val_loader is not None else "",
            avg["cell_perplexity"],
        )
        if monitor < best_loss - 1e-6:
            best_loss, best_epoch, no_improve = monitor, ep, 0
            if tc.save_best and val_loader is not None and is_main_process():
                save_checkpoint(core, checkpoint_dir / "best_checkpoint.pt")
        else:
            no_improve += 1
            if tc.early_stopping_patience is not None and no_improve >= tc.early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %d (best epoch %d, best loss %.4f)",
                    ep + 1,
                    best_epoch + 1,
                    best_loss,
                )
                break

    # Runtime metrics (always reported). Rank-0 writes training_runtime.json.
    total_seconds = time.perf_counter() - t_train_start
    mean_epoch_seconds = total_seconds / max(epochs_run, 1)
    n_cells = len(dataset)
    cells_per_second = (n_cells * epochs_run) / total_seconds if total_seconds > 0 else None
    iters_per_second = (n_batches * epochs_run) / total_seconds if total_seconds > 0 else None
    peak_gpu_gb = (
        torch.cuda.max_memory_allocated(device_t) / (1024**3)
        if device_t.type == "cuda"
        else None
    )
    _hh = int(total_seconds // 3600)
    _mm = int((total_seconds % 3600) // 60)
    _ss = int(total_seconds % 60)
    total_hms = f"{_hh}:{_mm:02d}:{_ss:02d}"
    runtime = {
        "total_seconds": round(total_seconds, 3),
        "total_hms": total_hms,
        "n_epochs": epochs_run,
        "mean_epoch_seconds": round(mean_epoch_seconds, 3),
        "cells_per_second": round(cells_per_second, 2) if cells_per_second is not None else None,
        "iters_per_second": round(iters_per_second, 3) if iters_per_second is not None else None,
        "n_cells": int(n_cells),
        "effective_batch_size": int(effective_batch),
        "nominal_learning_rate": float(tc.learning_rate),
        "effective_learning_rate": float(effective_lr),
        "lr_scaled_with_batch": bool(effective_lr != tc.learning_rate),
        "n_batches_per_epoch": int(n_batches),
        "peak_gpu_gb": round(peak_gpu_gb, 3) if peak_gpu_gb is not None else None,
        "device": str(device_t),
        "encoder_type": model_config.encoder_type,
        "quantizer_type": model_config.quantizer_type,
        "input_dim": int(input_dim),
    }
    if is_main_process():
        (checkpoint_dir / "training_runtime.json").write_text(json.dumps(runtime, indent=2))
        logger.info(
            "RUNTIME enc=%s input_dim=%d n_cells=%d batch=%d epochs=%d total=%s "
            "mean_epoch=%.1fs cells/s=%s peak_gpu=%s",
            model_config.encoder_type,
            input_dim,
            n_cells,
            effective_batch,
            epochs_run,
            total_hms,
            mean_epoch_seconds,
            f"{cells_per_second:.0f}" if cells_per_second is not None else "NA",
            f"{peak_gpu_gb:.2f}GB" if peak_gpu_gb is not None else "NA",
        )

    # All disk artifacts (checkpoint, codebooks, embeddings, annotated adata) are
    # produced on rank 0 only; non-main ranks return early with the shared core model.
    if use_ddp and not is_main_process():
        from .._distributed import cleanup_distributed

        cleanup_distributed()
        return core, adata

    # Restore best-val weights before exporting artifacts. During training,
    # best_checkpoint.pt is written at the lowest monitored (validation) loss, but
    # the epoch loop leaves ``core`` holding the LAST epoch's weights. Without this
    # reload the exported main checkpoint, both codebook npz files, the embeddings,
    # the code indices, and adata_with_hierarchical_embeddings.h5ad would all come
    # from the last epoch, not the best-val epoch. Reload only when the best path was
    # actually produced (save_best + a val split + an improving epoch); the default
    # no-val path (val_fraction=0 -> val_loader is None -> no best_checkpoint) skips
    # this entirely and its behavior is unchanged.
    best_ckpt_path = checkpoint_dir / "best_checkpoint.pt"
    if (
        tc.save_best
        and val_loader is not None
        and best_epoch >= 0
        and best_ckpt_path.exists()
    ):
        from ..models import load_checkpoint

        best = load_checkpoint(best_ckpt_path, device_t, config=model_config)
        core.load_state_dict(best.state_dict())
        core.to(device_t)
        logger.info(
            "Exporting best-val weights from epoch %d (val_loss=%.4f); last epoch was %d",
            best_epoch + 1,
            best_loss,
            epochs_run,
        )
    else:
        logger.info("Exporting last-epoch weights from epoch %d (no best-val restore)", epochs_run)

    ckpt_path = checkpoint_dir / "hierarchical_vqvae_checkpoint.pt"
    save_checkpoint(core, ckpt_path)
    (checkpoint_dir / "training_losses.json").write_text(json.dumps(losses, indent=2))

    core.eval()
    eval_loader = _make_loader(dataset, effective_batch, False, tc.num_workers, tc.seed, pin)
    cell_emb, neigh_emb, cell_idx, neigh_idx = _embed(core, eval_loader, device_t)
    if hasattr(core.cell_vq, "embedding"):
        np.savez_compressed(
            checkpoint_dir / "cell_codebook.npz",
            codebook=core.cell_vq.embedding.weight.data.cpu().numpy(),
        )
    if hasattr(core.neighborhood_vq, "embedding"):
        np.savez_compressed(
            checkpoint_dir / "neighborhood_codebook.npz",
            codebook=core.neighborhood_vq.embedding.weight.data.cpu().numpy(),
        )
    np.savez_compressed(checkpoint_dir / "hierarchical_cell_embeddings.npz", embeddings=cell_emb)
    np.savez_compressed(
        checkpoint_dir / "hierarchical_neighborhood_embeddings.npz", embeddings=neigh_emb
    )
    np.savez_compressed(checkpoint_dir / "hierarchical_cell_indices.npz", indices=cell_idx)
    np.savez_compressed(checkpoint_dir / "hierarchical_neighborhood_indices.npz", indices=neigh_idx)

    attach_codes_to_adata(adata, cell_idx, neigh_idx, cell_emb, neigh_emb)
    # Drop the internal raw-count target layer so the exported adata stays lean.
    adata.layers.pop("_raw_counts", None)
    adata.write_h5ad(checkpoint_dir / "adata_with_hierarchical_embeddings.h5ad")
    if use_ddp:
        from .._distributed import cleanup_distributed

        cleanup_distributed()
    return core, adata


def _val_loss(
    model: HierarchicalVQVAE,
    loader: DataLoader,
    device: torch.device,
    tc: TrainConfig,
    count_dataset: object | None = None,
) -> float:
    """Mean total loss over a validation loader (no grad).

    ``count_dataset`` (the full :class:`SpatialDataset`) is passed only for the
    count-recon cell modes so the raw-count target can be indexed by the batch
    index; it is ``None`` on all other paths (released signature).
    """
    # The count-scale niche target exists only when a DM niche mode is active (which
    # requires a count cell mode, so count_dataset is not None). Index it in the
    # full-dataset row space exactly like recon_target.
    use_nct = count_dataset is not None and getattr(count_dataset, "niche_count_target", None) is not None
    model.eval()
    total = 0.0
    n = 0
    with torch.inference_mode():
        for cb, nb, bidx in loader:
            cb, nb = cb.to(device), nb.to(device)
            rt = nct = None
            if count_dataset is not None:
                rt = count_dataset.recon_target[bidx].to(device)
            if use_nct:
                nct = count_dataset.niche_count_target[bidx].to(device)
            cr, nr, cvq, nvq, _ci, _ni, _cp, _np = model(cb, nb)
            loss = tc.cell_weight * (
                _cell_recon(model, cr, cb, recon_target=rt) + cvq
            ) + tc.neighborhood_weight * (_niche_recon(model, nr, nb, niche_count_target=nct) + nvq)
            total += float(loss.item())
            n += 1
    model.train()
    return total / max(n, 1)


def _embed(
    model: HierarchicalVQVAE, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pass the dataset through the trained model and return embeddings + code indices."""
    ce: list[np.ndarray] = []
    ne: list[np.ndarray] = []
    ci: list[np.ndarray] = []
    ni: list[np.ndarray] = []
    with torch.inference_mode():
        for cb, nb, _ in loader:
            cb = cb.to(device)
            nb = nb.to(device)
            z_cell = model.cell_encoder(cb)
            z_neigh = model.neighborhood_encoder(nb)
            _, _, _, _, c_idx, n_idx, _, _ = model(cb, nb)
            ce.append(z_cell.cpu().numpy())
            ne.append(z_neigh.cpu().numpy())
            ci.append(c_idx.cpu().numpy())
            ni.append(n_idx.cpu().numpy())
    return (
        np.concatenate(ce, 0),
        np.concatenate(ne, 0),
        np.concatenate(ci, 0).reshape(-1),
        np.concatenate(ni, 0).reshape(-1),
    )


class Trainer:
    """High-level training and inference wrapper (nanoGPT / Lightning style).

    Parameters
    ----------
    config
        A :class:`TrainConfig`; defaults to the reproducible production settings.
    model_config
        Optional :class:`~nicheverse.models.ModelConfig`; if omitted, ``fit``
        builds one from the AnnData gene panel.

    Examples
    --------
    >>> trainer = Trainer(TrainConfig(num_epochs=10))
    >>> model, adata = trainer.fit(adata, "checkpoints/")
    >>> annotated = trainer.predict(new_adata, "checkpoints/hierarchical_vqvae_checkpoint.pt")
    """

    def __init__(self, config=None, model_config=None) -> None:
        self.config = config or TrainConfig()
        self.model_config = model_config
        self.model = None

    def fit(self, adata, checkpoint_dir, model_config=None, sample_col="sample_id", device=None):
        """Train and persist the model; returns ``(model, annotated_adata)``."""
        self.model, out = train_model(
            adata,
            checkpoint_dir,
            model_config=model_config if model_config is not None else self.model_config,
            train_config=self.config,
            sample_col=sample_col,
            device=device,
        )
        return self.model, out

    def predict(self, adata, checkpoint=None, **kwargs):
        """Assign codes to ``adata`` using a checkpoint path or the fitted model."""
        from .predict import predict_codes

        ckpt = checkpoint if checkpoint is not None else self.model
        if ckpt is None:
            raise ValueError("No checkpoint provided and no fitted model; call fit() first.")
        return predict_codes(adata, ckpt, **kwargs)
