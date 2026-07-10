"""Distributed (multi-GPU DDP) helpers for accuracy-preserving data-parallel training.

Every function here is a no-op when a distributed process group is not initialized
(the single-process / single-GPU default), so importing and calling them on the
released single-GPU path leaves numerics byte-identical.

The DDP contract implemented on top of these helpers keeps the *global* batch size
constant: with ``world_size`` ranks each rank processes ``batch_size / world_size``
cells, gradients are all-reduced (mean) by :class:`torch.nn.parallel.DistributedDataParallel`,
and the EMA VQ codebook statistics are all-reduced (sum) so the codebook is refreshed
from the full global batch exactly as a single GPU would compute it. See
``DETERMINISM.md`` for the residual differences (dead-code reset RNG, k-means++ init)
that are handled by rank-0 broadcast.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    """True only when torch.distributed is built, available, and a group is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Number of ranks in the active process group (1 when not distributed)."""
    if is_dist_avail_and_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Rank of the current process (0 when not distributed)."""
    if is_dist_avail_and_initialized():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """True on rank 0 (always True when not distributed)."""
    return get_rank() == 0


def ddp_env_requested() -> bool:
    """True when the launcher (torchrun / SLURM) set the DDP rendezvous env vars.

    We treat DDP as requested only when ``WORLD_SIZE`` is set and > 1, so a plain
    ``python train.py`` invocation never enters the distributed path.
    """
    try:
        return int(os.environ.get("WORLD_SIZE", "1")) > 1
    except (TypeError, ValueError):
        return False


def init_distributed(backend: str | None = None, timeout_min: int = 30) -> tuple[int, int, int]:
    """Initialize the default process group from torchrun/SLURM env vars.

    Reads ``RANK``, ``WORLD_SIZE``, and ``LOCAL_RANK`` (set by ``torchrun``). Returns
    ``(rank, world_size, local_rank)``. Idempotent: if a group is already initialized
    it just returns the current ranks. Selects the CUDA device for ``local_rank`` when
    CUDA is available. No-op-safe: raises only if the env vars are inconsistent.
    """
    import datetime

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return rank, world_size, local_rank
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(minutes=timeout_min),
        )
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    """Destroy the default process group if one is initialized."""
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


class BatchContiguousDistributedSampler(torch.utils.data.Sampler):
    """Shard the shuffled epoch so each *global* batch equals the single-GPU batch.

    A stock :class:`torch.utils.data.DistributedSampler` interleaves indices
    (``perm[rank::world]``), so the union of the per-rank batch ``b`` is *not* the
    single-GPU batch ``perm[b*B : (b+1)*B]``. That would make the k-means++ codebook
    init and every EMA update see a different global batch composition than a single
    GPU, breaking equivalence.

    This sampler instead lays the shuffled permutation out as consecutive global
    batches of ``global_batch_size`` and hands rank ``r`` the ``r``-th contiguous
    slice *within* each global batch. Then, for every batch ``b``,

        union_over_ranks(rank_r_batch_b) == perm[b*B : (b+1)*B]

    exactly, so the gathered global batch (used for k-means init, EMA all-reduce, and
    the diversity term) matches a single-GPU run cell-for-cell. The trailing partial
    global batch is padded by wrapping to keep every rank's shard length equal (the
    only residual vs. single-GPU is a few duplicated trailing cells in the very last
    batch, identical to how ``DistributedSampler`` pads).

    Parameters
    ----------
    dataset_len
        Number of samples in the dataset.
    num_replicas, rank
        World size and this process's rank.
    global_batch_size
        The single-GPU batch size ``B`` (must be divisible by ``num_replicas``).
    shuffle, seed
        Per-epoch shuffle (matching a single-GPU seeded ``randperm``) and its seed.
    """

    def __init__(
        self,
        dataset_len: int,
        num_replicas: int,
        rank: int,
        global_batch_size: int,
        shuffle: bool = True,
        seed: int = 9,
    ) -> None:
        if global_batch_size % num_replicas != 0:
            raise ValueError(
                f"global_batch_size ({global_batch_size}) must be divisible by "
                f"num_replicas ({num_replicas})"
            )
        self.dataset_len = int(dataset_len)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.global_batch_size = int(global_batch_size)
        self.per_gpu = self.global_batch_size // self.num_replicas
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        import math as _math

        self.num_global_batches = _math.ceil(self.dataset_len / self.global_batch_size)
        self.num_samples = self.num_global_batches * self.per_gpu

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(self.dataset_len, generator=g).tolist()
        else:
            perm = list(range(self.dataset_len))
        total = self.num_global_batches * self.global_batch_size
        if total > self.dataset_len:  # pad the trailing global batch by wrapping
            perm = perm + perm[: total - self.dataset_len]
        out: list[int] = []
        for b in range(self.num_global_batches):
            base = b * self.global_batch_size + self.rank * self.per_gpu
            out.extend(perm[base : base + self.per_gpu])
        return iter(out)


def all_reduce_sum_(tensor: torch.Tensor) -> torch.Tensor:
    """In-place sum all-reduce across ranks. No-op (returns ``tensor``) when not distributed.

    Used to make the EMA VQ codebook statistics rank-correct: summing the per-rank
    cluster-size and embedding-sum tensors reconstructs exactly the statistics a
    single GPU would compute over the concatenated global batch.
    """
    if is_dist_avail_and_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_reduce_mean_(tensor: torch.Tensor) -> torch.Tensor:
    """In-place mean all-reduce across ranks. No-op when not distributed."""
    if is_dist_avail_and_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(get_world_size())
    return tensor


def differentiable_all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Autograd-aware sum all-reduce (returns a new tensor). Identity when not distributed.

    Uses ``torch.distributed.nn.functional.all_reduce``, which defines the backward
    pass (the gradient is itself summed across ranks). This lets a loss term that is
    a nonlinear function of a *global* batch statistic (e.g. the diversity entropy of
    the global mean soft-assignment) receive exactly the gradient a single GPU would
    compute over the concatenated batch.
    """
    if not is_dist_avail_and_initialized():
        return tensor
    from torch.distributed.nn.functional import all_reduce as _diff_all_reduce

    return _diff_all_reduce(tensor, op=dist.ReduceOp.SUM)


def broadcast_module_(module: torch.nn.Module, src: int = 0) -> None:
    """Broadcast every parameter and buffer of ``module`` from ``src`` in place.

    No-op when not distributed. Used right after k-means++ codebook init so every
    rank starts from the identical (rank-0) codebook, matching the single-GPU init.
    """
    if not is_dist_avail_and_initialized():
        return
    with torch.no_grad():
        for p in module.parameters():
            dist.broadcast(p.data, src=src)
        for b in module.buffers():
            dist.broadcast(b.data, src=src)
