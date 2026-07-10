"""Vector quantizers for the NICHEVERSE tokenizer.

The default ``vq`` is an EMA codebook with k-means++ init, dead-code reset, and a
diversity term. All registered quantizers (built by :func:`build_quantizer`):

- ``vq``           : EMA :class:`VectorQuantizer` (released default; do not modify).
- ``soft``         : :class:`SoftVQ`, differentiable softmax assignment.
- ``rot``          : :class:`RotVQ`, VQ in a learned Householder-rotated basis.
- ``fsq``          : :class:`FSQ`, codebook-free finite scalar quantization.
- ``qinco``        : :class:`QINCoVQ`, conditional residual VQ.
- ``pq``           : :class:`ProductVQ`, product quantization over subspaces.
- ``rvq``          : :class:`ResidualVQ`, stacked residual :class:`VectorQuantizer` stages.
- ``lfq``          : :class:`LFQ`, lookup-free (binary) quantization (MAGVIT-v2).
- ``bsq``          : :class:`BSQ`, binary spherical quantization.
- ``residual_fsq`` : :class:`ResidualFSQ`, multi-stage residual FSQ.
- ``grvq``         : :class:`GroupedResidualVQ`, per-group residual VQ.

Every quantizer takes a channels-first ``(B, D, T)`` tensor and returns the 4-tuple
``(loss, quantized (B, D, T), perplexity, encoding_indices (B*T, 1))``.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._distributed import (
    all_reduce_sum_,
    broadcast_module_,
    differentiable_all_reduce_sum,
    get_world_size,
    is_dist_avail_and_initialized,
    is_main_process,
)

logger = logging.getLogger(__name__)

DEAD_CODE_RESET_INTERVAL = 50
DEAD_CODE_USAGE_FRACTION = 0.01
CODE_USAGE_EMA = 0.99


class VectorQuantizer(nn.Module):
    """EMA vector quantizer with k-means++ init, dead code reset, and diversity term.

    This is the default quantizer (``quantizer_type="vq"``) and the discretization
    core of the NICHEVERSE model. Given an encoded cell (or neighborhood)
    embedding it snaps that continuous vector to the nearest of ``K`` learned
    codebook entries, so the integer code index becomes the cell-state (or niche)
    vocabulary symbol that downstream annotation and the neighborhood codebook
    build on. Gradients flow to the encoder through a straight-through estimator,
    while the codebook itself is updated by exponential moving averages rather
    than by the optimizer.

    The forward pass takes a tensor of shape ``(B, D, T)`` (channels-first,
    following the VQ-VAE convention), flattens to ``(B*T, D)``, finds the
    nearest codebook entry for each row by squared Euclidean distance, and
    returns the quantized output with straight-through gradients.

    Parameters
    ----------
    num_embeddings
        Codebook size ``K``.
    embedding_dim
        Codebook entry dimensionality ``D``.
    commitment_cost
        Weight ``beta`` of the commitment loss ``beta * ||sg(e) - z||^2``.
    use_ema
        If True, update codebook entries with EMA rather than gradient descent
        (recommended; matches Razavi et al. 2019).
    ema_decay
        EMA decay factor ``gamma`` for cluster sizes and embedding sums.
    diversity_weight
        Weight of the entropy maximizing diversity term. Set to 0 to disable.
    diversity_temperature
        Softmax temperature applied to negative distances before computing the
        batch average soft assignment used in the diversity term. Lower values
        sharpen the soft assignment (closer to the hard one).
    dead_code_reset_interval
        Number of training steps between dead code resets.
    dead_code_usage_fraction
        A code is considered dead if its EMA usage (an EMA of raw per-code
        assignment counts) falls below this fraction of the per-code fair share,
        i.e. below ``dead_code_usage_fraction * batch_size / num_embeddings``.
        The fair-share normalization matters: without it the threshold would
        exceed the uniform steady state and flag every code dead each interval.
    distance_metric
        Assignment metric, ``"l2"`` (default, squared Euclidean) or ``"cosine"``
        (assign to the maximum-cosine-similarity code). Must be one of those two.

    Attributes
    ----------
    embedding : torch.nn.Embedding
        The codebook, weight shape ``(K, D)`` = ``(num_embeddings, embedding_dim)``.
        Row ``i`` is code ``i``'s prototype vector.
    _initialized : torch.Tensor
        Bool buffer, shape ``()``; flipped to ``True`` after the first-batch
        k-means++ seeding so init runs exactly once.
    ema_cluster_size : torch.Tensor
        EMA of per-code assignment counts, shape ``(K,)`` (only when ``use_ema``).
    ema_embed_sum : torch.Tensor
        EMA of the sum of assigned vectors per code, shape ``(K, D)`` (only when
        ``use_ema``); divided by ``ema_cluster_size`` to refresh ``embedding``.
    code_usage : torch.Tensor
        Slower EMA of per-code usage, shape ``(K,)``, used to detect dead codes.
    update_count : torch.Tensor
        Long buffer, shape ``()``; counts EMA updates to schedule dead-code resets.

    Notes
    -----
    The codebook entries are initialized with k-means++ on the first training
    batch (van den Oord 2017 used uniform init but k-means++ converges faster
    and reduces dead codes early in training). When the batch is smaller than
    ``num_embeddings`` we still seed every slot: the first
    ``min(B*T, num_embeddings)`` slots are filled from the batch, the remainder
    are seeded by sampling with replacement so all slots start from data.

    During training the codebook is refreshed by EMA of the cluster sizes and
    assigned-vector sums (van den Oord 2017; Razavi et al. 2019), so codebook
    entries do not receive optimizer gradients. Encoder gradients bypass the
    non-differentiable argmax through a straight-through estimator
    (``quantized = inputs + (quantized - inputs).detach()``). Every
    ``dead_code_reset_interval`` updates, codes whose usage fell below threshold
    are reseeded from random batch rows to keep the codebook fully utilized. The
    optional diversity term adds ``diversity_weight * (log K - H(p))`` where ``p``
    is the batch-mean soft assignment, pushing the codebook toward balanced use.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        diversity_weight: float = 1.0,
        diversity_temperature: float = 1.0,
        dead_code_reset_interval: int = DEAD_CODE_RESET_INTERVAL,
        dead_code_usage_fraction: float = DEAD_CODE_USAGE_FRACTION,
        distance_metric: str = "l2",
    ) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError(
                f"num_embeddings and embedding_dim must be positive, got "
                f"{num_embeddings}, {embedding_dim}"
            )
        if not 0.0 < ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {ema_decay}")
        if distance_metric not in ("l2", "cosine"):
            raise ValueError(f"distance_metric must be 'l2' or 'cosine', got {distance_metric!r}")
        if diversity_temperature <= 0.0:
            raise ValueError(f"diversity_temperature must be positive, got {diversity_temperature}")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.diversity_weight = diversity_weight
        self.diversity_temperature = diversity_temperature
        self.dead_code_reset_interval = int(dead_code_reset_interval)
        self.dead_code_usage_fraction = float(dead_code_usage_fraction)
        self.distance_metric = distance_metric
        self.epsilon = 1e-5
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0, 1.0)
        if use_ema:
            # The EMA codebook is updated by the EMA rule, not the optimizer; freeze it
            # so the diversity loss (which reads embedding.weight) cannot feed it an
            # AdamW gradient that fights the EMA update. The encoder still receives the
            # diversity gradient through the soft assignment.
            self.embedding.weight.requires_grad_(False)
        self.register_buffer("_initialized", torch.tensor(False))
        if use_ema:
            self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
            self.register_buffer("ema_embed_sum", self.embedding.weight.data.clone())
            self.register_buffer("code_usage", torch.zeros(num_embeddings))
            self.register_buffer("update_count", torch.tensor(0, dtype=torch.long))

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"commitment_cost={self.commitment_cost}, use_ema={self.use_ema}, "
            f"ema_decay={self.ema_decay}, diversity_weight={self.diversity_weight}"
        )

    def _kmeans_init(self, flat_input: torch.Tensor) -> None:
        """k-means++ seeding of the codebook.

        When the available batch is smaller than ``num_embeddings`` we fill the
        first ``B*T`` slots from the batch and seed the remaining slots by
        sampling with replacement from the batch, so no slot is left at the
        uniform random init (a source of immediately-dead codes).
        """
        n_samples = flat_input.size(0)
        if n_samples == 0:
            return
        if n_samples < self.num_embeddings:
            idx = torch.randperm(n_samples, device=flat_input.device)
            self.embedding.weight.data[:n_samples] = flat_input[idx]
            # Seed remaining slots from the batch (with small noise to break ties).
            n_extra = self.num_embeddings - n_samples
            extra_idx = torch.randint(0, n_samples, (n_extra,), device=flat_input.device)
            extra = flat_input[extra_idx] + 0.01 * torch.randn_like(flat_input[extra_idx])
            self.embedding.weight.data[n_samples:] = extra
            if self.use_ema:
                self.ema_embed_sum.copy_(self.embedding.weight.data)
                self.ema_cluster_size.fill_(1.0)
            return
        centroids: list[torch.Tensor] = []
        idx0 = torch.randint(0, n_samples, (1,), device=flat_input.device)
        centroids.append(flat_input[idx0].squeeze(0))
        for _ in range(1, self.num_embeddings):
            stack = torch.stack(centroids)
            d2 = torch.cdist(flat_input, stack).min(dim=1).values ** 2
            total = d2.sum()
            if total <= 0:
                # All points coincide with chosen centroids; fall back to uniform.
                idx = torch.randint(0, n_samples, (1,), device=flat_input.device)
            else:
                probs = d2 / total
                idx = torch.multinomial(probs, 1)
            centroids.append(flat_input[idx].squeeze(0))
        self.embedding.weight.data.copy_(torch.stack(centroids))
        if self.use_ema:
            self.ema_embed_sum.copy_(self.embedding.weight.data)
            self.ema_cluster_size.fill_(1.0)

    def _reset_dead_codes(self, flat_input: torch.Tensor) -> None:
        """Reseed any code whose EMA usage is below threshold from random batch rows.

        Under DDP the reseeding pool is the *global* batch (all ranks' rows gathered)
        and the reset is performed on rank 0 then broadcast, so every rank applies an
        identical reset and the codebook stays consistent across ranks. This is the
        one documented residual vs. a single-GPU run: which random rows seed the dead
        codes differs (the per-step RNG stream diverges once data is sharded), but the
        reset only fires every ``dead_code_reset_interval`` steps and only for codes
        that are genuinely unused, so the effect on the trained codebook is negligible.
        """
        if is_dist_avail_and_initialized():
            flat_input = self._gather_flat(flat_input)
        batch_size = flat_input.size(0)
        if batch_size == 0:
            return
        # ``code_usage`` is an EMA of RAW per-code assignment counts, so under a
        # healthy near-uniform codebook each code sits near its fair share,
        # ``batch_size / num_embeddings`` (e.g. 8 at B=2048, K=256). The dead
        # threshold must therefore be relative to that fair share, not the whole
        # batch: ``batch_size * frac`` (~20 at B=2048, frac=0.01) exceeds the
        # uniform steady state and flags every code dead. Scaling by
        # 1/num_embeddings makes the test "usage below ``frac`` of fair share".
        fair_share = batch_size / self.num_embeddings
        thresh = fair_share * self.dead_code_usage_fraction
        dead = self.code_usage < thresh
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return
        if is_main_process():
            ridx = torch.randint(0, batch_size, (n_dead,), device=flat_input.device)
            new_e = flat_input[ridx] + torch.randn_like(flat_input[ridx]) * 0.01
            self.embedding.weight.data[dead] = new_e
            self.ema_embed_sum[dead] = new_e
            self.ema_cluster_size[dead] = 1.0
            # Grace value for a freshly reseeded code. With the corrected (tiny)
            # ``thresh`` (~0.08), seeding ``code_usage`` at ``thresh`` would leave
            # the new code exactly on the dead line, so if it happens to attract no
            # assignments before the next reset interval it would be re-killed
            # immediately, never getting a real chance. Seeding at the fair share
            # ``batch_size / num_embeddings`` is the value a uniformly-used code
            # converges to, so it (a) is well above ``thresh`` and survives at least
            # one full interval unused, and (b) is not so large that the code looks
            # artificially over-used: the EMA (decay 0.99) relaxes it toward its
            # true usage within tens of steps, letting a genuinely useful reseeded
            # code accumulate real assignments and a genuinely useless one decay
            # back below threshold and be reset again later. This is the natural,
            # scale-correct grace rather than an ad hoc constant.
            self.code_usage[dead] = fair_share
        if is_dist_avail_and_initialized():
            # Broadcast the updated codebook + EMA state so all ranks match rank 0.
            broadcast_module_(self)
        logger.debug(
            "VectorQuantizer: reset %d dead codes (threshold=%.4f, fair_share=%.3f)",
            n_dead,
            thresh,
            fair_share,
        )

    @staticmethod
    def _gather_flat(flat_input: torch.Tensor) -> torch.Tensor:
        """All-gather the per-rank ``flat`` rows into the concatenated global batch.

        Ranks may hold different row counts (last shard), so we use variable-length
        all-gather. No-op-safe: returns ``flat_input`` when not distributed.
        """
        if not is_dist_avail_and_initialized():
            return flat_input
        import torch.distributed as dist

        world = get_world_size()
        local_n = torch.tensor([flat_input.shape[0]], device=flat_input.device)
        sizes = [torch.zeros_like(local_n) for _ in range(world)]
        dist.all_gather(sizes, local_n)
        max_n = int(max(int(s.item()) for s in sizes))
        padded = flat_input.new_zeros((max_n, flat_input.shape[1]))
        padded[: flat_input.shape[0]] = flat_input
        gathered = [torch.zeros_like(padded) for _ in range(world)]
        dist.all_gather(gathered, padded)
        return torch.cat([g[: int(sizes[i].item())] for i, g in enumerate(gathered)], dim=0)

    def _distances(self, flat: torch.Tensor) -> torch.Tensor:
        """Pairwise distance from each row of ``flat`` to every codebook entry.

        ``"l2"`` uses squared Euclidean distance
        ``||x||^2 + ||e||^2 - 2 x . e``. ``"cosine"`` L2-normalizes both sides
        and returns ``1 - cos_sim``, whose argmin matches maximum cosine
        similarity (Yu et al., ViT-VQGAN, 2022, arXiv:2110.04627).
        """
        w = self.embedding.weight
        if self.distance_metric == "cosine":
            fn = F.normalize(flat, dim=1)
            wn = F.normalize(w, dim=1)
            return 1.0 - fn @ wn.t()
        return flat.pow(2).sum(1, keepdim=True) + w.pow(2).sum(1) - 2 * flat @ w.t()

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize ``inputs`` against the learned codebook.

        Parameters
        ----------
        inputs
            Tensor of shape ``(B, D, T)`` (channels-first). The hierarchical
            model uses ``T=1``.

        Returns
        -------
        loss
            Scalar VQ loss (commitment + codebook MSE if not using EMA + diversity).
        quantized
            Tensor of shape ``(B, D, T)``; carries straight-through gradients.
        perplexity
            Scalar perplexity ``exp(H(p))`` where ``p`` is the batch-average
            hard-assignment distribution.
        encoding_indices
            Tensor of shape ``(B*T, 1)`` with the assigned code index per row.

        Shape
        -----
        - Input: ``(B, D, T)`` with ``D = embedding_dim``, ``K = num_embeddings``.
        - Output: ``loss`` ``()``, ``quantized`` ``(B, D, T)``, ``perplexity``
          ``()``, ``encoding_indices`` ``(B*T, 1)``.
        """
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        flat = inputs.view(-1, self.embedding_dim)
        if self.training and not bool(self._initialized):
            if is_dist_avail_and_initialized():
                # DDP equivalence: seed the codebook from the *global* first batch
                # (gathered across ranks) on rank 0, then broadcast so all ranks
                # start from an identical codebook. Single-process path is unchanged.
                global_flat = self._gather_flat(flat)
                if is_main_process():
                    self._kmeans_init(global_flat)
                broadcast_module_(self)
            else:
                self._kmeans_init(flat)
            self._initialized.fill_(True)
        distances = self._distances(flat)
        enc_idx = distances.argmin(1).unsqueeze(1)
        # Build the one-hot and accumulate the EMA statistics in float32. Under
        # fp16 AMP ``flat.dtype`` would be float16, and summing thousands of
        # one-hot rows (enc_sum, whose per-code counts can exceed 2048) or the
        # assigned-vector sums (e_sum) in fp16 rounds badly once the running total
        # passes the fp16 integer-exact limit of 2048. The EMA buffers
        # (``ema_cluster_size`` / ``ema_embed_sum``) are already float32, so this
        # only fixes the pre-buffer accumulation. In the default fp32 path
        # ``flat`` is already float32, so ``enc`` and the casts are no-ops and the
        # computation stays numerically identical to before.
        enc = torch.zeros(
            enc_idx.shape[0], self.num_embeddings, device=inputs.device, dtype=torch.float32
        )
        enc.scatter_(1, enc_idx, 1)
        quantized = self.embedding.weight[enc_idx.squeeze(1)].view(input_shape)
        if self.training and self.use_ema:
            # Disable autocast so the EMA embed-sum matmul accumulates in true fp32
            # (autocast would downcast the matmul to fp16 even inside no_grad).
            with torch.no_grad(), torch.autocast(device_type=flat.device.type, enabled=False):
                self.update_count.add_(1)
                enc_sum = enc.sum(0)
                e_sum = enc.t() @ flat.float()
                # DDP equivalence: sum the per-rank codebook statistics so the EMA
                # update sees exactly the global-batch counts (enc_sum) and
                # embedding sums (e_sum) a single GPU would compute over the
                # concatenated batch. all_reduce_sum_ is a no-op when not
                # distributed, so the single-GPU path stays byte-identical.
                if is_dist_avail_and_initialized():
                    all_reduce_sum_(enc_sum)
                    all_reduce_sum_(e_sum)
                self.ema_cluster_size.mul_(self.ema_decay).add_(enc_sum, alpha=1 - self.ema_decay)
                self.ema_embed_sum.mul_(self.ema_decay).add_(e_sum, alpha=1 - self.ema_decay)
                n = self.ema_cluster_size.sum()
                cluster = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon)
                    * n
                )
                self.embedding.weight.data.copy_(self.ema_embed_sum / cluster.unsqueeze(1))
                self.code_usage.mul_(CODE_USAGE_EMA).add_(enc_sum, alpha=1 - CODE_USAGE_EMA)
                if int(self.update_count.item()) % self.dead_code_reset_interval == 0:
                    self._reset_dead_codes(flat)
        if self.use_ema:
            loss = self.commitment_cost * F.mse_loss(quantized.detach(), inputs)
        else:
            loss = F.mse_loss(quantized, inputs.detach()) + self.commitment_cost * F.mse_loss(
                quantized.detach(), inputs
            )
        if self.diversity_weight > 0:
            soft = F.softmax(-distances / self.diversity_temperature, dim=1)
            # The cross-rank reduction only runs during training. In eval every rank
            # returned before the embedding pass, so a collective here would deadlock;
            # eval also never uses the diversity loss, so the local mean is correct.
            if self.training and is_dist_avail_and_initialized():
                # DDP equivalence: the diversity entropy is a nonlinear function of
                # the *global* mean soft-assignment. Sum the per-rank soft mass and
                # row count across ranks (differentiably) so `avg` equals the
                # single-GPU global-batch mean and the encoder receives the matching
                # gradient. Identity op when not distributed (byte-identical default).
                soft_sum = differentiable_all_reduce_sum(soft.sum(0))
                count = torch.tensor(
                    float(soft.shape[0]), device=inputs.device, dtype=soft.dtype
                )
                all_reduce_sum_(count)
                avg = soft_sum / count
            else:
                avg = soft.mean(0)
            ent = -(avg * (avg + 1e-10).log()).sum()
            max_ent = torch.log(
                torch.tensor(float(self.num_embeddings), device=inputs.device, dtype=ent.dtype)
            )
            loss = loss + self.diversity_weight * (max_ent - ent)
        # Straight-through estimator.
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = enc.mean(0)
        perplexity = (-(avg_probs * (avg_probs + 1e-10).log()).sum()).exp()
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, enc_idx


# ---------------------------------------------------------------------------
# Quantizer registry + alternative quantizers (opt-in; the default remains "vq")
# ---------------------------------------------------------------------------

_QUANTIZERS: dict[str, type] = {}


def register_quantizer(name: str):
    """Class decorator registering a quantizer under ``name`` for :func:`build_quantizer`."""

    def deco(cls: type) -> type:
        _QUANTIZERS[name] = cls
        return cls

    return deco


_QUANTIZERS["vq"] = VectorQuantizer


def _entropy(p: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Shannon entropy (natural log) of a probability vector ``p``."""
    return -(p * (p + eps).log()).sum()


def build_quantizer(
    name: str,
    *,
    num_embeddings: int,
    embedding_dim: int,
    commitment_cost: float,
    distance_metric: str = "l2",
    **kwargs,
):
    """Construct a quantizer by registry name (see :data:`_QUANTIZERS` for the full set).

    ``build_quantizer("vq", ...)`` returns exactly the :class:`VectorQuantizer` the
    model builds by default, so the default path is bit-identical. The codebook-free
    variants (``fsq``, ``lfq``, ``bsq``, ``residual_fsq``) derive their code count
    implicitly and ignore ``num_embeddings`` (a warning is logged if the two disagree).
    """
    if name not in _QUANTIZERS:
        raise ValueError(f"unknown quantizer_type {name!r}; choose from {sorted(_QUANTIZERS)}")
    if name == "vq":
        return VectorQuantizer(
            num_embeddings,
            embedding_dim,
            commitment_cost,
            distance_metric=distance_metric,
            **kwargs,
        )
    if name == "rot":
        return RotVQ(
            num_embeddings,
            embedding_dim,
            commitment_cost=commitment_cost,
            distance_metric=distance_metric,
            **kwargs,
        )
    if name == "soft":
        return SoftVQ(num_embeddings, embedding_dim, commitment_cost=commitment_cost, **kwargs)
    if name == "fsq":
        fsq = FSQ(embedding_dim, commitment_cost=commitment_cost, **kwargs)
        if num_embeddings and fsq.num_embeddings != num_embeddings:
            logger.warning(
                "FSQ ignores num_embeddings=%d; its code count is prod(levels)=%d.",
                num_embeddings,
                fsq.num_embeddings,
            )
        return fsq
    if name == "qinco":
        return QINCoVQ(
            num_embeddings,
            embedding_dim,
            commitment_cost=commitment_cost,
            distance_metric=distance_metric,
            **kwargs,
        )
    if name == "pq":
        return ProductVQ(
            num_embeddings,
            embedding_dim,
            commitment_cost=commitment_cost,
            distance_metric=distance_metric,
            **kwargs,
        )
    if name == "rvq":
        return ResidualVQ(
            num_embeddings,
            embedding_dim,
            commitment_cost=commitment_cost,
            distance_metric=distance_metric,
            **kwargs,
        )
    if name == "grvq":
        return GroupedResidualVQ(
            num_embeddings,
            embedding_dim,
            commitment_cost=commitment_cost,
            distance_metric=distance_metric,
            **kwargs,
        )
    if name == "lfq":
        lfq = LFQ(embedding_dim, commitment_cost=commitment_cost, **kwargs)
        if num_embeddings and lfq.num_embeddings != num_embeddings:
            logger.warning(
                "LFQ ignores num_embeddings=%d; its code count is 2**embedding_dim=%d.",
                num_embeddings,
                lfq.num_embeddings,
            )
        return lfq
    if name == "bsq":
        bsq = BSQ(embedding_dim, commitment_cost=commitment_cost, **kwargs)
        if num_embeddings and bsq.num_embeddings != num_embeddings:
            logger.warning(
                "BSQ ignores num_embeddings=%d; its code count is 2**embedding_dim=%d.",
                num_embeddings,
                bsq.num_embeddings,
            )
        return bsq
    if name == "residual_fsq":
        rfsq = ResidualFSQ(embedding_dim, commitment_cost=commitment_cost, **kwargs)
        if num_embeddings and rfsq.num_embeddings != num_embeddings:
            logger.warning(
                "ResidualFSQ ignores num_embeddings=%d; its per-stage code count is prod(levels)=%d.",
                num_embeddings,
                rfsq.num_embeddings,
            )
        return rfsq
    raise ValueError(  # pragma: no cover - registered name without a constructor branch
        f"quantizer_type {name!r} is registered but build_quantizer has no constructor branch for it"
    )


@register_quantizer("soft")
class SoftVQ(nn.Module):
    """Soft vector quantizer: differentiable softmax assignment over a learnable codebook.

    Parameters
    ----------
    num_embeddings, embedding_dim
        Codebook size ``K`` and entry dimensionality ``D``.
    commitment_cost
        Weight of the soft commitment MSE (``quantized`` toward the input).
    temperature
        Softmax temperature on the cosine-similarity logits (lower is sharper).
    diversity_weight
        Weight of the marginal-entropy load-balancing term (0 disables it).

    Notes
    -----
    ``quantized = softmax(cos_sim / temperature) @ codebook`` is differentiable
    everywhere (no straight-through estimator); the hard ``argmax`` index is
    returned for annotation. Forward takes ``(B, D, T)`` and returns
    ``(loss, quantized (B, D, T), perplexity, index (B*T, 1))``.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        temperature: float = 0.07,
        diversity_weight: float = 0.01,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError(
                f"num_embeddings and embedding_dim must be positive, got {num_embeddings}, {embedding_dim}"
            )
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.temperature = temperature
        self.diversity_weight = diversity_weight
        self.distance_metric = "cosine"
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0, 1.0)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Softly assign the input to the codebook by temperature-scaled cosine similarity.

        Returns ``(loss, quantized, perplexity, index)`` where ``quantized`` is the
        softmax-weighted codebook average (differentiable, no straight-through) and
        ``index`` is the hard argmax code for annotation.

        Shape
        -----
        - Input: ``(B, D, T)``.
        - Output: ``loss`` ``()``, ``quantized`` ``(B, D, T)``, ``perplexity``
          ``()``, ``index`` ``(B*T, 1)``.
        """
        inputs = inputs.permute(0, 2, 1).contiguous()
        shape = inputs.shape
        flat = inputs.view(-1, self.embedding_dim)
        cb = self.embedding.weight
        sim = F.normalize(flat, dim=1) @ F.normalize(cb, dim=1).t()
        p = F.softmax(sim / self.temperature, dim=1)
        quantized = (p @ cb).view(shape)
        enc_idx = sim.argmax(1, keepdim=True)
        loss = self.commitment_cost * F.mse_loss(quantized, inputs)
        if self.diversity_weight > 0:
            max_ent = torch.log(torch.tensor(float(self.num_embeddings), device=flat.device))
            loss = loss + self.diversity_weight * (max_ent - _entropy(p.mean(0)))
        counts = torch.bincount(enc_idx.squeeze(1), minlength=self.num_embeddings).float()
        perplexity = _entropy(counts / counts.sum().clamp_min(1.0)).exp()
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, enc_idx


class _VQDelegate:
    """Expose the codebook surface by delegating to a wrapped :class:`VectorQuantizer`.

    ``embedding`` / ``num_embeddings`` / ``embedding_dim`` / ``distance_metric`` route to
    the quantizer that each subclass names via its ``_delegate`` property.
    """

    @property
    def embedding(self) -> nn.Embedding:
        return self._delegate.embedding

    @property
    def num_embeddings(self) -> int:
        return self._delegate.num_embeddings

    @property
    def embedding_dim(self) -> int:
        return self._delegate.embedding_dim

    @property
    def distance_metric(self) -> str:
        return self._delegate.distance_metric


@register_quantizer("rot")
class RotVQ(_VQDelegate, nn.Module):
    """VQ in a learned rotated basis (product of Householder reflections).

    Applies an exactly-orthogonal learned rotation to the latent, quantizes with
    a standard EMA :class:`VectorQuantizer` in that basis, then applies the exact
    inverse rotation to the quantized vector. Decorrelates axes to raise codebook
    utilization; the per-cell code is a single integer. ``embedding`` /
    ``num_embeddings`` / ``embedding_dim`` / ``distance_metric`` delegate to the
    wrapped quantizer so it satisfies the same contract as :class:`VectorQuantizer`.

    Parameters
    ----------
    num_embeddings, embedding_dim
        Codebook size and entry dimensionality of the wrapped EMA quantizer.
    commitment_cost, distance_metric
        Passed through to the wrapped :class:`VectorQuantizer`.
    num_householders
        Number of Householder reflections composing the rotation.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        num_householders: int = 8,
        distance_metric: str = "l2",
    ) -> None:
        super().__init__()
        if num_householders < 1:
            raise ValueError(f"num_householders must be >= 1, got {num_householders}")
        self.vq = VectorQuantizer(
            num_embeddings, embedding_dim, commitment_cost, distance_metric=distance_metric
        )
        self._embedding_dim = embedding_dim
        self.v = nn.Parameter(torch.randn(num_householders, embedding_dim))

    @property
    def _delegate(self) -> VectorQuantizer:
        return self.vq

    def _reflect(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        order = range(self.v.shape[0] - 1, -1, -1) if reverse else range(self.v.shape[0])
        for i in order:
            vhat = F.normalize(self.v[i], dim=0)
            x = x - 2.0 * (x @ vhat).unsqueeze(1) * vhat.unsqueeze(0)
        return x

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rotate into the learned basis, quantize there, and rotate the result back.

        The input is flattened and passed through the composed Householder
        reflections, the rotated latent is quantized by the wrapped EMA
        :class:`VectorQuantizer` (which owns the loss, perplexity, index, and
        straight-through gradient), and the quantized vector is mapped back with
        the exact inverse rotation. The code index and loss are those of the
        wrapped quantizer, so the contract matches :class:`VectorQuantizer`.

        Parameters
        ----------
        inputs : torch.Tensor
            Encoded latent, shape ``(B, D, T)`` (channels-first; the model uses
            ``T=1``), with ``D = embedding_dim``.

        Returns
        -------
        loss : torch.Tensor
            Scalar VQ loss from the wrapped quantizer.
        quantized : torch.Tensor
            Inverse-rotated quantized latent, shape ``(B, D, T)``; carries
            straight-through gradients through the wrapped quantizer.
        perplexity : torch.Tensor
            Scalar codebook perplexity of the wrapped quantizer.
        encoding_indices : torch.Tensor
            Assigned code index per row, shape ``(B*T, 1)``.

        Shape
        -----
        - Input: ``(B, D, T)``.
        - Output: ``loss`` ``()``, ``quantized`` ``(B, D, T)``, ``perplexity``
          ``()``, ``encoding_indices`` ``(B*T, 1)``.
        """
        inputs_p = inputs.permute(0, 2, 1).contiguous()
        shape = inputs_p.shape
        flat = inputs_p.view(-1, self._embedding_dim)
        rot = self._reflect(flat).view(shape).permute(0, 2, 1).contiguous()
        loss, q_rot, perp, idx = self.vq(rot)
        q_flat = q_rot.permute(0, 2, 1).contiguous().view(-1, self._embedding_dim)
        quantized = self._reflect(q_flat, reverse=True).view(shape).permute(0, 2, 1).contiguous()
        return loss, quantized, perp, idx


@register_quantizer("fsq")
class FSQ(nn.Module):
    """Finite Scalar Quantization (Mentzer et al. 2023): codebook-free discretization.

    Projects the latent to ``len(levels)`` scalar dims, bounds each with a shifted
    ``tanh`` (using the canonical even-level offset so every level yields that many
    distinct integer codes), rounds with a straight-through estimator, and encodes
    a single integer index via mixed-radix. Effective code count is
    ``prod(levels)``; the index is a bijection onto ``[0, prod(levels))``.

    Parameters
    ----------
    embedding_dim
        Latent dimensionality; projected to and from ``len(levels)``.
    levels
        Per-dimension quantization levels (each >= 3; level 2 is degenerate).
    commitment_cost
        Accepted for a uniform interface but unused: FSQ has no codebook to commit
        to; the straight-through round ties the latent to the reconstruction.
    diversity_weight, var_target
        Optional differentiable range-usage penalty
        ``diversity_weight * relu(var_target - var(bounded_z))`` (0 disables it;
        FSQ is collapse-resistant by construction).

    Notes
    -----
    Forward takes ``(B, D, T)`` and returns
    ``(loss, quantized (B, D, T), perplexity, index (B*T, 1))``.
    """

    def __init__(
        self,
        embedding_dim: int,
        levels: tuple[int, ...] = (8, 5, 5, 5),
        commitment_cost: float = 0.0,
        diversity_weight: float = 0.0,
        var_target: float = 1.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.levels = tuple(int(x) for x in levels)
        if any(lvl < 3 for lvl in self.levels):
            raise ValueError(
                f"every FSQ level must be >= 3 (level 2 is degenerate and unstable), got {self.levels}"
            )
        n, basis = 1, []
        for lvl in self.levels:
            basis.append(n)
            n *= lvl
        self.num_embeddings = n
        self.distance_metric = "fsq"
        self.commitment_cost = commitment_cost
        self.diversity_weight = diversity_weight
        self.var_target = var_target
        self.project_in = nn.Linear(embedding_dim, len(self.levels))
        self.project_out = nn.Linear(len(self.levels), embedding_dim)
        self.register_buffer("_basis", torch.tensor(basis, dtype=torch.long))
        self.register_buffer("_levels", torch.tensor(self.levels, dtype=torch.float32))
        self.register_buffer(
            "_half_width", torch.tensor([lvl // 2 for lvl in self.levels], dtype=torch.long)
        )

    @property
    def embedding(self) -> nn.Embedding:
        """Materialize the implicit FSQ codebook (grid points mapped through ``project_out``)."""
        dims = torch.arange(self.num_embeddings, device=self._basis.device).unsqueeze(1)
        idx_dims = (dims // self._basis) % self._levels.long()
        code = (idx_dims.float() - self._half_width) / self._half_width.clamp_min(1)
        with torch.no_grad():
            entries = self.project_out(code)
        return nn.Embedding.from_pretrained(entries, freeze=True)

    def _bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    @staticmethod
    def _round_ste(x: torch.Tensor) -> torch.Tensor:
        return x + (torch.round(x) - x).detach()

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project to scalar dims, bound and round (straight-through), and encode a mixed-radix index.

        The latent is projected to ``len(levels)`` scalar channels, each bounded
        by the shifted ``tanh`` and rounded to its integer level with a
        straight-through estimator, then decoded back to the latent space. The
        per-dimension integers are packed into one index in ``[0, prod(levels))``
        by the mixed-radix ``_basis``. There is no learned codebook, so the loss
        is zero unless the optional range-usage penalty is enabled.

        Parameters
        ----------
        inputs : torch.Tensor
            Encoded latent, shape ``(B, D, T)`` (channels-first; the model uses
            ``T=1``), with ``D = embedding_dim``.

        Returns
        -------
        loss : torch.Tensor
            Scalar loss; ``0`` unless ``diversity_weight > 0`` adds the
            range-usage penalty.
        quantized : torch.Tensor
            Straight-through quantized latent, shape ``(B, D, T)``.
        perplexity : torch.Tensor
            Scalar perplexity of the batch code-usage histogram.
        encoding_indices : torch.Tensor
            Mixed-radix code index per row, shape ``(B*T, 1)``, in
            ``[0, prod(levels))``.

        Shape
        -----
        - Input: ``(B, D, T)``.
        - Output: ``loss`` ``()``, ``quantized`` ``(B, D, T)``, ``perplexity``
          ``()``, ``encoding_indices`` ``(B*T, 1)``.
        """
        inputs = inputs.permute(0, 2, 1).contiguous()
        shape = inputs.shape
        flat = inputs.view(-1, self.embedding_dim)
        bounded = self._bound(self.project_in(flat))
        codes = self._round_ste(bounded)
        idx_dims = (codes + self._half_width).round().long().clamp_min(0)
        idx_dims = torch.minimum(idx_dims, self._levels.long() - 1)
        idx = (idx_dims * self._basis).sum(1, keepdim=True)
        quantized = self.project_out(codes / self._half_width.clamp_min(1)).view(shape)
        loss = torch.zeros((), device=flat.device)
        if self.diversity_weight > 0 and flat.shape[0] > 1:
            var_pen = F.relu(self.var_target - bounded.var(0, unbiased=False).mean())
            loss = loss + self.diversity_weight * var_pen
        counts = torch.bincount(idx.squeeze(1), minlength=self.num_embeddings).float()
        perplexity = _entropy(counts / counts.sum().clamp_min(1.0)).exp()
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, idx


@register_quantizer("qinco")
class QINCoVQ(_VQDelegate, nn.Module):
    """QINCo-style conditional residual VQ (Huijben et al. 2024, arXiv:2401.14732).

    A stack of ``num_levels`` codebooks applied coarse-to-fine on the residual.
    Level 0 is a plain EMA :class:`VectorQuantizer`; each deeper level quantizes a
    query produced by a small MLP conditioned on the residual and the running sum
    of quantized vectors, so later codes refine earlier ones. The returned
    single-integer code is the level-0 (coarse) index, which keeps
    ``cell_codebook_idx`` a stable single column; the quantized output is the sum
    over all levels. ``embedding`` / ``num_embeddings`` / ``embedding_dim`` /
    ``distance_metric`` delegate to level 0.

    Parameters
    ----------
    num_embeddings, embedding_dim
        Per-level codebook size and entry dimensionality.
    commitment_cost, distance_metric
        Passed to each level's :class:`VectorQuantizer`.
    num_levels
        Number of residual levels (>= 1).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        num_levels: int = 4,
        distance_metric: str = "l2",
    ) -> None:
        super().__init__()
        if num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {num_levels}")
        self._embedding_dim = embedding_dim
        self.num_levels = num_levels
        self.levels = nn.ModuleList(
            VectorQuantizer(
                num_embeddings, embedding_dim, commitment_cost, distance_metric=distance_metric
            )
            for _ in range(num_levels)
        )
        self.cond = nn.ModuleList(
            nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Linear(embedding_dim, embedding_dim),
            )
            for _ in range(num_levels - 1)
        )

    @property
    def _delegate(self) -> VectorQuantizer:
        return self.levels[0]

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize coarse-to-fine over the residual and sum the per-level quantized vectors.

        Level 0 quantizes the input with a plain EMA :class:`VectorQuantizer`.
        Each deeper level quantizes a query formed by an MLP conditioned on the
        running residual and the accumulated quantized sum, so later codes refine
        earlier ones (the residual is detached before subtracting each level's
        output). The output is the sum over all levels; the returned single-integer
        code and perplexity are level 0's (coarse) values, keeping
        ``cell_codebook_idx`` a stable single column.

        Parameters
        ----------
        inputs : torch.Tensor
            Encoded latent, shape ``(B, D, T)`` (channels-first; the model uses
            ``T=1``), with ``D = embedding_dim``.

        Returns
        -------
        loss : torch.Tensor
            Scalar sum of the per-level VQ losses.
        quantized : torch.Tensor
            Sum of the per-level quantized vectors, shape ``(B, D, T)``.
        perplexity : torch.Tensor
            Scalar perplexity of the level-0 codebook.
        encoding_indices : torch.Tensor
            Level-0 (coarse) code index per row, shape ``(B*T, 1)``.

        Shape
        -----
        - Input: ``(B, D, T)``.
        - Output: ``loss`` ``()``, ``quantized`` ``(B, D, T)``, ``perplexity``
          ``()``, ``encoding_indices`` ``(B*T, 1)``.
        """
        b, d, tdim = inputs.shape
        residual = inputs
        quantized_sum = torch.zeros_like(inputs)
        total_loss = inputs.new_zeros(())
        l1_idx = l1_perp = None
        for i in range(self.num_levels):
            if i == 0:
                loss_i, q_i, l1_perp, l1_idx = self.levels[0](residual)
            else:
                rf = residual.permute(0, 2, 1).reshape(-1, d)
                qf = quantized_sum.permute(0, 2, 1).reshape(-1, d)
                query = rf + self.cond[i - 1](torch.cat([rf, qf], dim=1))
                query = query.reshape(b, tdim, d).permute(0, 2, 1).contiguous()
                loss_i, q_i, _, _ = self.levels[i](query)
            quantized_sum = quantized_sum + q_i
            residual = residual - q_i.detach()
            total_loss = total_loss + loss_i
        return total_loss, quantized_sum, l1_perp, l1_idx


@register_quantizer("pq")
class ProductVQ(_VQDelegate, nn.Module):
    """Product quantization: split the latent into ``num_subspaces`` chunks, each quantized by its
    own codebook.

    Effective code count is ``num_embeddings ** num_subspaces`` at only
    ``num_embeddings * embedding_dim`` parameters, which resists collapse and
    factorizes a cell into independent programs. The returned single-integer code
    is subspace 0's index (kept for a stable ``cell_codebook_idx``); the quantized
    output is the concatenation over subspaces. ``embedding`` / ``num_embeddings``
    / ``embedding_dim`` / ``distance_metric`` delegate to subspace 0 (so
    ``embedding_dim`` reports the per-subspace width).

    Parameters
    ----------
    num_embeddings, embedding_dim
        Codebook size per subspace and total latent dimensionality (must be
        divisible by ``num_subspaces``).
    commitment_cost, distance_metric
        Passed to each subspace's :class:`VectorQuantizer`.
    num_subspaces
        Number of subspaces the latent is split into.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        num_subspaces: int = 4,
        distance_metric: str = "l2",
    ) -> None:
        super().__init__()
        if num_subspaces < 1:
            raise ValueError(f"num_subspaces must be >= 1, got {num_subspaces}")
        if embedding_dim % num_subspaces != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_subspaces ({num_subspaces})"
            )
        self._embedding_dim = embedding_dim
        self.num_subspaces = num_subspaces
        sub = embedding_dim // num_subspaces
        self.subs = nn.ModuleList(
            VectorQuantizer(num_embeddings, sub, commitment_cost, distance_metric=distance_metric)
            for _ in range(num_subspaces)
        )

    @property
    def _delegate(self) -> VectorQuantizer:
        return self.subs[0]

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the latent into subspaces, quantize each independently, and concatenate.

        The channel dimension is split into ``num_subspaces`` equal chunks, each
        quantized by its own EMA :class:`VectorQuantizer`, and the quantized chunks
        are concatenated back to the full width. The loss is the sum over
        subspaces; the returned single-integer code and perplexity are
        subspace 0's, keeping ``cell_codebook_idx`` a stable single column while the
        effective code count is ``num_embeddings ** num_subspaces``.

        Parameters
        ----------
        inputs : torch.Tensor
            Encoded latent, shape ``(B, D, T)`` (channels-first; the model uses
            ``T=1``), with ``D`` the full ``embedding_dim`` divisible by
            ``num_subspaces``.

        Returns
        -------
        loss : torch.Tensor
            Scalar sum of the per-subspace VQ losses.
        quantized : torch.Tensor
            Concatenation of the per-subspace quantized latents, shape ``(B, D, T)``.
        perplexity : torch.Tensor
            Scalar perplexity of subspace 0's codebook.
        encoding_indices : torch.Tensor
            Subspace-0 code index per row, shape ``(B*T, 1)``.

        Shape
        -----
        - Input: ``(B, D, T)``.
        - Output: ``loss`` ``()``, ``quantized`` ``(B, D, T)``, ``perplexity``
          ``()``, ``encoding_indices`` ``(B*T, 1)``.
        """
        outs = []
        total_loss = inputs.new_zeros(())
        perp0 = idx0 = None
        for s, ch in enumerate(inputs.chunk(self.num_subspaces, dim=1)):
            loss_s, q_s, perp_s, idx_s = self.subs[s](ch)
            outs.append(q_s)
            total_loss = total_loss + loss_s
            if s == 0:
                perp0, idx0 = perp_s, idx_s
        return total_loss, torch.cat(outs, dim=1), perp0, idx0


def _perplexity_capped(idx: torch.Tensor, num_codes: int, cap: int = 4096) -> torch.Tensor:
    """Batch code-usage perplexity ``exp(H)`` with the histogram width capped at ``cap``.

    Implicit-codebook quantizers (LFQ / BSQ / residual FSQ) can have astronomically
    large nominal code counts (``2**embedding_dim``); the perplexity histogram is
    computed over ``min(num_codes, cap)`` bins (indices folded modulo the width, so
    distinct large codes spread across bins rather than collapsing) so the metric
    stays finite and memory-safe.
    """
    k = int(min(int(num_codes), cap))
    counts = torch.bincount(idx.reshape(-1).remainder(k), minlength=k).float()
    return _entropy(counts / counts.sum().clamp_min(1.0)).exp()


@register_quantizer("rvq")
class ResidualVQ(_VQDelegate, nn.Module):
    """Residual vector quantization: a stack of EMA :class:`VectorQuantizer` stages.

    Stage 0 quantizes the input; each later stage quantizes the residual left by the
    running sum of quantized vectors (the residual is detached before subtracting each
    stage's output, exactly as in :class:`QINCoVQ`), so deeper stages refine coarser
    ones. The output is the sum over stages; the returned single-integer code and
    perplexity are stage 0's (coarse) values, keeping ``cell_codebook_idx`` a stable
    single column. ``embedding`` / ``num_embeddings`` / ``embedding_dim`` /
    ``distance_metric`` delegate to stage 0. Unlike :class:`QINCoVQ` there is no
    conditioning MLP: each stage is a plain residual codebook.

    Parameters
    ----------
    num_embeddings, embedding_dim
        Per-stage codebook size and entry dimensionality.
    commitment_cost, distance_metric
        Passed to each stage's :class:`VectorQuantizer`.
    num_stages
        Number of residual stages (>= 1).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        num_stages: int = 4,
        distance_metric: str = "l2",
    ) -> None:
        super().__init__()
        if num_stages < 1:
            raise ValueError(f"num_stages must be >= 1, got {num_stages}")
        self._embedding_dim = embedding_dim
        self.num_stages = num_stages
        self.stages = nn.ModuleList(
            VectorQuantizer(
                num_embeddings, embedding_dim, commitment_cost, distance_metric=distance_metric
            )
            for _ in range(num_stages)
        )

    @property
    def _delegate(self) -> VectorQuantizer:
        return self.stages[0]

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize the residual stage by stage and sum the per-stage quantized vectors.

        Parameters
        ----------
        inputs : torch.Tensor
            Encoded latent, shape ``(B, D, T)`` (channels-first; the model uses
            ``T=1``), with ``D = embedding_dim``.

        Returns
        -------
        loss : torch.Tensor
            Scalar sum of the per-stage VQ losses.
        quantized : torch.Tensor
            Sum of the per-stage quantized vectors, shape ``(B, D, T)``; carries
            straight-through gradients.
        perplexity : torch.Tensor
            Scalar perplexity of stage 0's codebook.
        encoding_indices : torch.Tensor
            Stage-0 (coarse) code index per row, shape ``(B*T, 1)``.

        Shape
        -----
        - Input: ``(B, D, T)``.
        - Output: ``loss`` ``()``, ``quantized`` ``(B, D, T)``, ``perplexity``
          ``()``, ``encoding_indices`` ``(B*T, 1)``.
        """
        residual = inputs
        quantized_sum = torch.zeros_like(inputs)
        total_loss = inputs.new_zeros(())
        perp0 = idx0 = None
        for i in range(self.num_stages):
            loss_i, q_i, perp_i, idx_i = self.stages[i](residual)
            quantized_sum = quantized_sum + q_i
            residual = residual - q_i.detach()
            total_loss = total_loss + loss_i
            if i == 0:
                perp0, idx0 = perp_i, idx_i
        return total_loss, quantized_sum, perp0, idx0


@register_quantizer("grvq")
class GroupedResidualVQ(_VQDelegate, nn.Module):
    """Grouped residual VQ: split the channels into groups, an :class:`ResidualVQ` per group.

    The channel dimension is split into ``num_groups`` equal chunks; each chunk is
    quantized by its own residual VQ, and the quantized chunks are concatenated back
    to the full width (Yang et al., HiFi-Codec, arXiv:2305.02765). The loss is the sum
    over groups; the returned single-integer code and perplexity are group 0's stage-0
    values, keeping ``cell_codebook_idx`` a stable single column. ``embedding`` /
    ``num_embeddings`` / ``embedding_dim`` / ``distance_metric`` delegate to group 0
    (so ``embedding_dim`` reports the per-group width).

    Parameters
    ----------
    num_embeddings, embedding_dim
        Per-stage codebook size and total latent dimensionality (must be divisible by
        ``num_groups``).
    commitment_cost, distance_metric
        Passed to each group's :class:`ResidualVQ`.
    num_groups
        Number of channel groups the latent is split into.
    num_stages
        Number of residual stages within each group's :class:`ResidualVQ`.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        num_groups: int = 2,
        num_stages: int = 2,
        distance_metric: str = "l2",
    ) -> None:
        super().__init__()
        if num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}")
        if embedding_dim % num_groups != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_groups ({num_groups})"
            )
        self._embedding_dim = embedding_dim
        self.num_groups = num_groups
        self.num_stages = num_stages
        group_dim = embedding_dim // num_groups
        self.groups = nn.ModuleList(
            ResidualVQ(
                num_embeddings,
                group_dim,
                commitment_cost=commitment_cost,
                num_stages=num_stages,
                distance_metric=distance_metric,
            )
            for _ in range(num_groups)
        )

    @property
    def _delegate(self) -> VectorQuantizer:
        return self.groups[0]

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split into channel groups, run an independent residual VQ per group, and concatenate.

        Parameters
        ----------
        inputs : torch.Tensor
            Encoded latent, shape ``(B, D, T)`` (channels-first; the model uses
            ``T=1``), with ``D`` divisible by ``num_groups``.

        Returns
        -------
        loss : torch.Tensor
            Scalar sum of the per-group residual VQ losses.
        quantized : torch.Tensor
            Concatenation of the per-group quantized latents, shape ``(B, D, T)``.
        perplexity : torch.Tensor
            Scalar perplexity of group 0's stage-0 codebook.
        encoding_indices : torch.Tensor
            Group-0 stage-0 code index per row, shape ``(B*T, 1)``.

        Shape
        -----
        - Input: ``(B, D, T)``.
        - Output: ``loss`` ``()``, ``quantized`` ``(B, D, T)``, ``perplexity``
          ``()``, ``encoding_indices`` ``(B*T, 1)``.
        """
        outs = []
        total_loss = inputs.new_zeros(())
        perp0 = idx0 = None
        for g, ch in enumerate(inputs.chunk(self.num_groups, dim=1)):
            loss_g, q_g, perp_g, idx_g = self.groups[g](ch)
            outs.append(q_g)
            total_loss = total_loss + loss_g
            if g == 0:
                perp0, idx0 = perp_g, idx_g
        return total_loss, torch.cat(outs, dim=1), perp0, idx0


@register_quantizer("lfq")
class LFQ(nn.Module):
    """Lookup-Free Quantization (Yu et al., MAGVIT-v2, ICLR 2024, arXiv:2310.05737).

    There is no learned codebook: each latent dimension becomes one bit via its sign,
    so the implicit codebook is ``{-1, +1}**embedding_dim`` and the effective code
    count is ``2**embedding_dim``. The integer code is the bit-packed sign pattern.
    A commitment term ties the latent to its binarization and an entropy term
    (marginal minus conditional bit entropy) prevents dead bits.

    Parameters
    ----------
    embedding_dim
        Latent dimensionality; also the number of bits (code count ``2**embedding_dim``).
    commitment_cost
        Default weight of the commitment term (used when ``commitment_weight`` is None).
    entropy_weight
        Weight of the bit-entropy regularizer.
    commitment_weight
        Explicit commitment weight; falls back to ``commitment_cost`` when None.

    Notes
    -----
    ``embedding_dim`` is the working (bit) dimensionality and ``num_embeddings`` is the
    nominal code count ``2**embedding_dim``. To stay within int64, the packed index
    uses at most 62 bits (so the index is exact for ``embedding_dim <= 62`` and always
    in ``[0, num_embeddings)``). Forward takes ``(B, D, T)`` and returns
    ``(loss, quantized (B, D, T), perplexity, index (B*T, 1))``.
    """

    _MAX_PACK_BITS = 62

    def __init__(
        self,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        entropy_weight: float = 0.1,
        commitment_weight: float | None = None,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        self.embedding_dim = embedding_dim
        self.entropy_weight = entropy_weight
        self.commitment_weight = commitment_cost if commitment_weight is None else commitment_weight
        self.distance_metric = "lfq"
        self.num_embeddings = 2 ** min(embedding_dim, self._MAX_PACK_BITS)
        self._pack_bits = min(embedding_dim, self._MAX_PACK_BITS)
        self.register_buffer("_basis", 2 ** torch.arange(self._pack_bits, dtype=torch.long))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Binarize each latent dimension by its sign and pack the bits into one integer code.

        Parameters
        ----------
        inputs : torch.Tensor
            Encoded latent, shape ``(B, D, T)`` (channels-first; the model uses
            ``T=1``), with ``D = embedding_dim``.

        Returns
        -------
        loss : torch.Tensor
            Scalar commitment + entropy loss.
        quantized : torch.Tensor
            Straight-through binarized latent (``+/-1`` per dim), shape ``(B, D, T)``.
        perplexity : torch.Tensor
            Scalar perplexity of the (capped) batch code histogram.
        encoding_indices : torch.Tensor
            Bit-packed code index per row, shape ``(B*T, 1)``, in ``[0, 2**D)``.

        Shape
        -----
        - Input: ``(B, D, T)``.
        - Output: ``loss`` ``()``, ``quantized`` ``(B, D, T)``, ``perplexity``
          ``()``, ``encoding_indices`` ``(B*T, 1)``.
        """
        inputs = inputs.permute(0, 2, 1).contiguous()
        shape = inputs.shape
        flat = inputs.view(-1, self.embedding_dim)
        z_q = torch.sign(flat)
        z_q = torch.where(z_q == 0, torch.ones_like(z_q), z_q)
        z_q_ste = flat + (z_q - flat).detach()
        bits = ((z_q + 1) / 2).long()
        idx = (bits[:, : self._pack_bits] * self._basis).sum(1, keepdim=True)
        p_bit = torch.sigmoid(flat * 2.0)
        p_marg = p_bit.mean(0).clamp(1e-6, 1 - 1e-6)
        h_marg = -(p_marg * p_marg.log() + (1 - p_marg) * (1 - p_marg).log()).sum()
        p_cond = p_bit.clamp(1e-6, 1 - 1e-6)
        h_cond = -(p_cond * p_cond.log() + (1 - p_cond) * (1 - p_cond).log()).sum(-1).mean()
        loss = self.commitment_weight * (flat - z_q.detach()).pow(2).mean()
        loss = loss + self.entropy_weight * (h_cond - h_marg)
        perplexity = _perplexity_capped(idx, self.num_embeddings)
        return loss, z_q_ste.view(shape).permute(0, 2, 1).contiguous(), perplexity, idx


@register_quantizer("bsq")
class BSQ(nn.Module):
    """Binary Spherical Quantization (Zhao et al., ICML 2024, arXiv:2406.07548).

    The latent is L2-normalized onto the unit sphere and then binarized per dimension
    to ``+/- 1/sqrt(D)``, so codes lie on hypercube vertices on the sphere and the
    effective code count is ``2**embedding_dim``. The integer code is the bit-packed
    sign pattern. A marginal bit-entropy term encourages balanced bit usage.

    Parameters
    ----------
    embedding_dim
        Latent dimensionality; also the number of bits (code count ``2**embedding_dim``).
    commitment_cost
        Accepted for a uniform interface; BSQ's loss is the entropy regularizer only.
    gamma
        Temperature of the soft-sign used in the bit-entropy regularizer.

    Notes
    -----
    ``embedding_dim`` is the working (bit) dimensionality and ``num_embeddings`` is the
    nominal code count ``2**embedding_dim``; the packed index uses at most 62 bits so it
    stays in ``[0, num_embeddings)`` within int64. Forward takes ``(B, D, T)`` and
    returns ``(loss, quantized (B, D, T), perplexity, index (B*T, 1))``.
    """

    _MAX_PACK_BITS = 62

    def __init__(
        self,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.gamma = gamma
        self.distance_metric = "bsq"
        self.num_embeddings = 2 ** min(embedding_dim, self._MAX_PACK_BITS)
        self._scale = 1.0 / math.sqrt(embedding_dim)
        self._pack_bits = min(embedding_dim, self._MAX_PACK_BITS)
        self.register_buffer("_basis", 2 ** torch.arange(self._pack_bits, dtype=torch.long))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project onto the unit sphere, binarize each dimension, and pack the bits into a code.

        Parameters
        ----------
        inputs : torch.Tensor
            Encoded latent, shape ``(B, D, T)`` (channels-first; the model uses
            ``T=1``), with ``D = embedding_dim``.

        Returns
        -------
        loss : torch.Tensor
            Scalar marginal bit-entropy regularizer.
        quantized : torch.Tensor
            Straight-through spherical binarized latent, shape ``(B, D, T)``.
        perplexity : torch.Tensor
            Scalar perplexity of the (capped) batch code histogram.
        encoding_indices : torch.Tensor
            Bit-packed code index per row, shape ``(B*T, 1)``, in ``[0, 2**D)``.

        Shape
        -----
        - Input: ``(B, D, T)``.
        - Output: ``loss`` ``()``, ``quantized`` ``(B, D, T)``, ``perplexity``
          ``()``, ``encoding_indices`` ``(B*T, 1)``.
        """
        inputs = inputs.permute(0, 2, 1).contiguous()
        shape = inputs.shape
        flat = inputs.view(-1, self.embedding_dim)
        z_n = F.normalize(flat, dim=-1)
        z_q = torch.sign(z_n) * self._scale
        z_q = torch.where(z_q == 0, self._scale * torch.ones_like(z_q), z_q)
        z_q_ste = z_n + (z_q - z_n).detach()
        bits = (z_n > 0).long()
        idx = (bits[:, : self._pack_bits] * self._basis).sum(1, keepdim=True)
        p_bit = torch.sigmoid(z_n * self.gamma).clamp(1e-6, 1 - 1e-6)
        pm = p_bit.mean(0)
        h_marg = -(pm * pm.log() + (1 - pm) * (1 - pm).log()).sum()
        loss = -0.05 * h_marg
        perplexity = _perplexity_capped(idx, self.num_embeddings)
        return loss, z_q_ste.view(shape).permute(0, 2, 1).contiguous(), perplexity, idx


@register_quantizer("residual_fsq")
class ResidualFSQ(nn.Module):
    """Residual finite scalar quantization (arXiv:2508.15860): multi-stage :class:`FSQ`.

    Stacks ``num_stages`` codebook-free :class:`FSQ` layers, each quantizing the
    residual left by the previous stages (residual detached before subtracting), so
    the stages form a hierarchical, collapse-resistant code. The output is the sum
    over stages; the returned single-integer code and perplexity are stage 0's,
    keeping ``cell_codebook_idx`` a stable single column. ``num_embeddings`` reports the
    per-stage code count ``prod(levels)`` and ``embedding_dim`` the working dimension.

    Parameters
    ----------
    embedding_dim
        Latent dimensionality (each stage projects to and from ``len(levels)``).
    levels
        Per-dimension quantization levels shared by every stage (each >= 3).
    num_stages
        Number of residual FSQ stages (>= 1).
    commitment_cost, diversity_weight
        Passed to each stage's :class:`FSQ` (FSQ is codebook-free; both default off).
    """

    def __init__(
        self,
        embedding_dim: int,
        levels: tuple[int, ...] = (8, 5, 5, 5),
        num_stages: int = 3,
        commitment_cost: float = 0.0,
        diversity_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if num_stages < 1:
            raise ValueError(f"num_stages must be >= 1, got {num_stages}")
        self.embedding_dim = embedding_dim
        self.num_stages = num_stages
        self.stages = nn.ModuleList(
            FSQ(
                embedding_dim,
                levels=levels,
                commitment_cost=commitment_cost,
                diversity_weight=diversity_weight,
            )
            for _ in range(num_stages)
        )
        self.num_embeddings = self.stages[0].num_embeddings
        self.distance_metric = "fsq"

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize the residual with a fresh FSQ at each stage and sum the quantized outputs.

        Parameters
        ----------
        inputs : torch.Tensor
            Encoded latent, shape ``(B, D, T)`` (channels-first; the model uses
            ``T=1``), with ``D = embedding_dim``.

        Returns
        -------
        loss : torch.Tensor
            Scalar sum of the per-stage FSQ losses (``0`` unless a stage penalty is on).
        quantized : torch.Tensor
            Sum of the per-stage quantized vectors, shape ``(B, D, T)``.
        perplexity : torch.Tensor
            Scalar perplexity of stage 0.
        encoding_indices : torch.Tensor
            Stage-0 (coarse) mixed-radix code index per row, shape ``(B*T, 1)``, in
            ``[0, prod(levels))``.

        Shape
        -----
        - Input: ``(B, D, T)``.
        - Output: ``loss`` ``()``, ``quantized`` ``(B, D, T)``, ``perplexity``
          ``()``, ``encoding_indices`` ``(B*T, 1)``.
        """
        residual = inputs
        quantized_sum = torch.zeros_like(inputs)
        total_loss = inputs.new_zeros(())
        perp0 = idx0 = None
        for i in range(self.num_stages):
            loss_i, q_i, perp_i, idx_i = self.stages[i](residual)
            quantized_sum = quantized_sum + q_i
            residual = residual - q_i.detach()
            total_loss = total_loss + loss_i
            if i == 0:
                perp0, idx0 = perp_i, idx_i
        return total_loss, quantized_sum, perp0, idx0
