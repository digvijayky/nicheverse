# Determinism contract

This document records exactly what nicheverse does and does not guarantee
in terms of bit-for-bit reproducibility. Reviewers replicating the manuscript
runs should read this first.

## What is bit-for-bit reproducible

Given identical inputs (the same h5ad, the same checkpoint, the same gene
panel) on the same hardware, the same CUDA build, and the same Python
environment, the following are guaranteed bit identical across runs:

1. The per-cell `cell_codebook_idx` integer vector written by `predict_codes`.
2. The per-cell `neighborhood_codebook_idx` integer vector.
3. The SHA256 hashes printed by `nicheverse verify` over those two vectors.
4. The continuous `X_cell_embedding` and `X_neighborhood_embedding` matrices,
   to within `atol=0, rtol=0` (verified by `tests/test_determinism.py`).

The `verify` CLI subcommand is the canonical way to assert this contract.
`nicheverse predict --report report.json` emits the SHA256 sums, and
`nicheverse verify --predicted out.h5ad --reference report.json` returns
exit code 0 only when both hashes match.

## What is reproducible only within the same hardware class

Floating point reductions (matrix multiplies, attention) on CUDA depend on
warp counts and tensor core layout. Switching GPU model (e.g. A100 to H100) or
CUDA version may change the last few bits of the continuous embeddings, which
in rare cases (cells near a code centroid boundary) can change the hard
assignment. The published model was trained on a single GPU class; the
released checkpoint is the source of truth and `predict` against that
checkpoint on a different GPU is expected to reproduce assignments for
>99.9% of cells, with the exact cutoff documented per release.

## What is not reproducible

1. Training from scratch on a different GPU class will produce a different
   final codebook. The codebook is non-convex in the input data; we make no
   bit-for-bit guarantees across hardware for training. We do guarantee that
   training on the same hardware twice with the same seed reproduces the
   final codebook to within numerical noise of accumulated EMA updates.
2. The training loss curve printed to stdout is informational, not part of
   the contract.

## Multi-GPU (DDP) equivalence contract

`train_model` has an opt-in DistributedDataParallel path (`TrainConfig(ddp=True)`,
activated only under `torchrun` with `WORLD_SIZE > 1`). It is designed so that a
2-, 4-, or 8-GPU run reproduces the optimization trajectory of a single-GPU run on
the same **global** batch, to within numerical tolerance. The global batch size is
held constant: with `world_size` ranks each rank processes `batch_size /
world_size` cells and gradients are averaged by DDP.

What is made exactly rank-correct (identical to a single GPU on the full batch):

1. The EMA codebook updates. The per-step cluster-size counts and embedding sums
   are all-reduced (summed) across ranks before the EMA step, so the codebook is
   refreshed from the full global batch, not a per-rank shard.
2. The k-means++ codebook initialization. The first global batch is gathered on
   rank 0, seeded there, and broadcast, so every rank starts from the same
   codebook that a single GPU would build from the first `batch_size` cells.
3. The diversity (entropy) term. The mean soft-assignment is computed over the
   global batch via a differentiable all-reduce, so the encoder receives the same
   gradient a single GPU would.
4. BatchNorm statistics. Encoders are converted to `SyncBatchNorm` under DDP, so
   the normalization uses global batch statistics rather than per-rank ones.
5. The batch composition. A custom `BatchContiguousDistributedSampler` shards the
   shuffled epoch so the union of the per-rank batch `b` equals the single-GPU
   batch `perm[b*B:(b+1)*B]` cell-for-cell (a stock `DistributedSampler`
   interleaves and would break this). Epoch 0 uses the identical permutation to a
   single-GPU seeded run.

Documented residual differences (do not affect the contract above but are not
bit-identical to single-GPU):

1. Dead-code reset RNG. When a code is reseeded (every `dead_code_reset_interval`
   steps, only for genuinely unused codes) the random rows are drawn on rank 0
   from the gathered global batch and broadcast, so all ranks stay consistent; but
   which rows are drawn differs from a single-GPU run because the per-step RNG
   stream diverges once data is sharded. The effect on the final codebook is
   negligible.
2. Per-epoch shuffle after epoch 0. The custom sampler reshuffles with `seed +
   epoch`, an equally valid but different order than a single-GPU `RandomSampler`
   for epochs >= 1.
3. Dropout masks. Any `nn.Dropout` (encoder p=0.2, decoder p=0.2, cross-attention
   p=0.1) samples independent masks per rank; these can never reconstruct the
   single-GPU mask over the full batch. This is inherent to data-parallel training
   with dropout and is the dominant source of the small per-step divergence when
   dropout is on.

Verification (measured). The single-GPU default path (`ddp=False`) is byte
identical to before this feature: the per-step loss over the first 60 steps at a
fixed seed matches a pre-DDP baseline with max abs diff exactly 0. `pin_memory`
and `prefetch_factor` are also verified byte identical to `num_workers=0`.

For DDP, with dropout disabled and both runs resumed from the same fixed
checkpoint (so k-means++ init is removed as a variable), a 2-GPU run reproduces a
1-GPU run's global per-step loss to ~1e-3 at step 1, and the EMA reduction itself
is bit exact (the summed per-rank cluster/embedding statistics equal the
full-batch statistics). Over many steps the residual grows to ~1e-2 on the loss
and ~1e-1 on the codebook, because floating-point reduction-order differences in
the all-reduce and (on GPU) SyncBatchNorm occasionally flip a hard `argmin`
codebook assignment for a cell near a code boundary, and those discrete flips
compound through the EMA. This is the fundamental residual of data-parallel VQ
training and is why DDP is opt-in. The logged per-step loss printed to stdout is
the per-rank local-shard mean and is expected to differ from the single-GPU global
loss; compare the all-reduced global loss or the codebook tensors instead.

`num_workers > 0` is a throughput knob and is NOT guaranteed bit-identical to
`num_workers=0`; use `num_workers=0` for exact reproduction of the released run.

## How we enforce it

1. `seed_everything(seed)` is called at the top of every training and
   inference entry point. It seeds `random`, `numpy`, `torch.manual_seed`,
   and `torch.cuda.manual_seed_all`.
2. `torch.backends.cudnn.deterministic = True` and `cudnn.benchmark = False`.
3. `torch.use_deterministic_algorithms(True, warn_only=True)` is called.
   `warn_only=True` is deliberate: if any op the model uses lacks a
   deterministic CUDA kernel, training still completes but a warning is
   emitted. Reviewers should run with `PYTHONWARNINGS=error` to convert
   such warnings into errors if strict determinism is needed.
4. The DataLoader is constructed with `shuffle=False` at inference time and
   `num_workers=0` by default, so batch order is fixed.
5. `CUBLAS_WORKSPACE_CONFIG=:4096:8` must be set **before** the first `torch`
   import in the process for full cuBLAS determinism. `seed_everything`
   writes this into `os.environ`, but cuBLAS only reads it once at import
   time, so we recommend exporting it in your shell:

   ```bash
   export CUBLAS_WORKSPACE_CONFIG=:4096:8
   python -m nicheverse predict ...
   ```

   If the variable is set after `torch` is imported,
   `nicheverse.utils.seed_everything` logs a warning so the issue is
   visible.

## Verifying determinism on your machine

The test suite ships a determinism check that runs `predict_codes` twice on
the same input and asserts both the integer assignments and the float
embeddings are identical:

```bash
pytest tests/test_determinism.py -q
```

If this test fails on your machine, your environment is the problem; please
report it on the issue tracker with the output of `nicheverse info`.
