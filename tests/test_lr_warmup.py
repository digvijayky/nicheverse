"""Unit tests for the per-optimizer-step warmup + cosine LR schedule.

Fast CPU-only tests of :func:`_warmup_cosine_lr_lambda` and :func:`_build_scheduler`
on a small ``total_steps``; no GPU and no real training. Verifies:

1. with ``warmup_frac > 0`` the per-step LR multiplier increases monotonically over
   the warmup window, peaks at ~1.0 (base LR) at the end of warmup, then decreases
   (cosine) afterward;
2. ``warmup_frac == 0.0`` reproduces the pre-change PURE per-step cosine schedule
   exactly (the LR sequence matches the closed-form cosine element for element);
3. the schedule reaches its floor (``floor_ratio``) exactly at the last step.
"""

import math

import torch

from nicheverse.training.trainer import (
    TrainConfig,
    _build_scheduler,
    _warmup_cosine_lr_lambda,
)


def _pure_cosine_ref(total_steps: int, floor: float):
    """Closed-form pre-change per-step cosine (no warmup): 1.0 at step 0 -> floor at last."""

    def f(step: int) -> float:
        prog = step / max(1, total_steps - 1)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
        return floor + (1.0 - floor) * cos

    return f


def test_warmup_then_cosine_monotone_and_peaks_at_base():
    total, frac, floor = 1000, 0.1, 0.0
    lam = _warmup_cosine_lr_lambda(total, frac, floor)
    warmup_steps = round(frac * total)  # 100
    seq = [lam(s) for s in range(total)]

    # (1a) strictly increasing over the warmup window, up to and including the peak.
    warm = seq[:warmup_steps]
    assert all(warm[i + 1] > warm[i] for i in range(len(warm) - 1))
    # peak (base LR multiplier 1.0) reached at the last warmup step.
    assert abs(seq[warmup_steps - 1] - 1.0) < 1e-12
    assert seq[warmup_steps - 1] >= max(seq)  # nothing exceeds the peak

    # (1b) the cosine holds the peak for exactly the transition step (step
    # warmup_steps has prog=0 -> factor 1.0, same as the last warmup step) then
    # strictly decreases through to the end.
    assert abs(seq[warmup_steps] - 1.0) < 1e-12  # cosine starts at the peak
    post = seq[warmup_steps:]  # start of the cosine (inclusive)
    assert all(post[i + 1] < post[i] for i in range(len(post) - 1))


def test_warmup_frac_zero_matches_pre_change_pure_cosine():
    total, floor = 1000, 0.0
    lam = _warmup_cosine_lr_lambda(total, 0.0, floor)
    ref = _pure_cosine_ref(total, floor)
    for s in range(total):
        assert abs(lam(s) - ref(s)) < 1e-12, f"mismatch at step {s}"
    # non-zero floor case too.
    lam2 = _warmup_cosine_lr_lambda(total, 0.0, 0.05)
    ref2 = _pure_cosine_ref(total, 0.05)
    assert all(abs(lam2(s) - ref2(s)) < 1e-12 for s in range(total))


def test_reaches_floor_only_at_last_step():
    total, floor = 1000, 0.05
    for frac in (0.0, 0.03, 0.2):
        lam = _warmup_cosine_lr_lambda(total, frac, floor)
        # floor reached exactly at the final step.
        assert abs(lam(total - 1) - floor) < 1e-12
        # not bottomed out early: every earlier post-warmup step is strictly above floor.
        warmup_steps = round(frac * total)
        assert all(lam(s) > floor + 1e-9 for s in range(warmup_steps, total - 1))


def test_pure_cosine_starts_at_base_and_ends_at_floor():
    total, floor = 500, 0.0
    lam = _warmup_cosine_lr_lambda(total, 0.0, floor)
    assert abs(lam(0) - 1.0) < 1e-12  # starts at base LR
    assert abs(lam(total - 1) - floor) < 1e-12  # ends at floor at the last step


def test_build_scheduler_warmup_cosine_drives_optimizer_lr():
    # End-to-end through LambdaLR: the real optimizer LR follows the per-step lambda.
    base_lr = 3e-4
    p = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.AdamW([p], lr=base_lr)
    tc = TrainConfig(
        lr_schedule="warmup_cosine",
        learning_rate=base_lr,
        warmup_frac=0.1,
        min_lr=0.0,
        num_epochs=10,
    )
    steps_per_epoch = 100  # total_steps = 1000
    sched, needs, per_step = _build_scheduler(opt, tc, lr=base_lr, steps_per_epoch=steps_per_epoch)
    assert needs is False and per_step is True

    lrs = []
    total = steps_per_epoch * tc.num_epochs
    for _ in range(total):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    warmup_steps = round(tc.warmup_frac * total)  # 100
    # peak base LR at the end of warmup, floor at the last step.
    assert abs(lrs[warmup_steps - 1] - base_lr) < 1e-10
    assert lrs[-1] < 1e-9  # min_lr == 0 -> ~0 at the final step
    # warmup rises, post-warmup falls.
    assert all(lrs[i + 1] > lrs[i] for i in range(warmup_steps - 1))
    assert all(lrs[i + 1] < lrs[i] for i in range(warmup_steps, total - 1))


def test_config_default_batch_size_is_32768():
    assert TrainConfig().batch_size == 32768
    assert TrainConfig().warmup_frac == 0.03
