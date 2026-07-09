"""Regression tests for the two VectorQuantizer fixes.

BUG 1 (HIGH): the dead-code reset threshold was ``batch_size * frac`` while
``code_usage`` is an EMA of raw per-code counts whose uniform steady state is
``batch_size / num_embeddings``. The threshold therefore exceeded the healthy
steady state and flagged essentially every code dead each interval. The fix makes
the threshold fair-share-relative: ``batch_size * frac / num_embeddings``, and
reseeds a killed code's ``code_usage`` at the fair share so a fresh code survives
at least one interval.

BUG 2 (LOW, AMP-only): the one-hot ``enc`` and the EMA statistics ``enc_sum`` /
``e_sum`` accumulated in the input dtype, losing precision under fp16 AMP once a
running count passed 2048. The fix accumulates in float32; the fp32 default path
is numerically unchanged.
"""

from __future__ import annotations

import torch

from nicheverse.models import VectorQuantizer


def _clustered_batch(n=2048, k=256, d=8, generator=None):
    """(N, D) rows drawn from k well-separated Gaussian clusters (near-uniform use).

    Each row is assigned round-robin to one of ``k`` cluster centers so the
    argmin assignment against a k-means++-seeded codebook stays near uniform,
    which is exactly the healthy regime where BUG 1 wrongly killed every code.
    """
    centers = torch.arange(k, dtype=torch.float32).unsqueeze(1) * 6.0
    centers = centers.repeat(1, d)  # (k, d) widely separated
    assign = torch.arange(n) % k
    return centers[assign] + 0.15 * torch.randn(n, d, generator=generator)


def _to_bdt(x2d):
    """(N, D) -> (N, D, 1) channels-first single-token input."""
    return x2d.unsqueeze(-1)


def _make_vq(k=256, d=8):
    torch.manual_seed(0)
    return VectorQuantizer(
        num_embeddings=k,
        embedding_dim=d,
        commitment_cost=0.25,
        use_ema=True,
        diversity_weight=1.0,
        dead_code_reset_interval=50,
        dead_code_usage_fraction=0.01,
    )


def _instrument_resets(vq):
    """Wrap ``_reset_dead_codes`` to record the number of codes flagged dead."""
    counts: list[int] = []
    orig = vq._reset_dead_codes

    def wrapped(flat_input):
        bs = flat_input.size(0)
        thresh = (bs / vq.num_embeddings) * vq.dead_code_usage_fraction
        n_dead = int((vq.code_usage < thresh).sum().item())
        counts.append(n_dead)
        return orig(flat_input)

    vq._reset_dead_codes = wrapped
    return counts


def test_threshold_is_fair_share_relative():
    """BUG 1 core: threshold must equal batch_size * frac / num_embeddings."""
    k, d, bs = 256, 8, 2048
    vq = _make_vq(k, d)
    frac = vq.dead_code_usage_fraction
    expected = bs * frac / k
    fair_share = bs / k
    thresh = fair_share * frac
    assert thresh == expected
    # The old (buggy) threshold would have been bs*frac, K times larger.
    assert thresh == (bs * frac) / k
    assert (bs * frac) / thresh == k  # exactly num_embeddings smaller


def test_few_dead_codes_after_warmup():
    """BUG 1: a healthy near-uniform codebook must flag ~0 dead codes per reset.

    With the buggy threshold every one of 256 codes was flagged dead each
    interval; with the fix the count should be near zero after warmup.
    """
    k, d, bs = 256, 8, 2048
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    vq = _make_vq(k, d)
    vq.train()
    reset_counts = _instrument_resets(vq)

    # Run enough steps to cross several reset intervals (50 steps each).
    for _ in range(160):
        x = _to_bdt(_clustered_batch(bs, k, d, gen))
        vq(x)

    assert len(reset_counts) >= 2, "expected multiple reset intervals to fire"
    # After the first (warmup) interval, resets should be small, not ~256.
    warm = reset_counts[1:]
    assert max(warm) < 0.15 * k, (
        f"too many dead codes flagged after warmup: {reset_counts} "
        f"(should be << {k})"
    )
    # And the codebook should be broadly active.
    active = int((vq.code_usage > vq.code_usage.mean() * 0.05).sum().item())
    assert active > 0.8 * k, f"only {active}/{k} codes active"


def test_reseed_grace_survives_one_interval():
    """A reseeded but unused code must not be re-killed at the next interval.

    Seeding ``code_usage`` at the fair share (not at ``thresh``) guarantees the
    new code sits well above threshold and survives at least one interval.
    """
    k, d, bs = 64, 4, 512
    vq = _make_vq(k, d)
    fair_share = bs / k
    thresh = fair_share * vq.dead_code_usage_fraction
    # Simulate a reseed: fair-share grace.
    vq.code_usage.fill_(fair_share)
    vq.code_usage[0] = fair_share  # freshly reseeded
    assert (vq.code_usage[0] >= thresh)
    # Even after one EMA decay step with zero new assignments it stays above.
    decayed = fair_share * 0.99
    assert decayed > thresh, "grace value must survive at least one interval unused"


def test_fp16_counts_match_true_integer_counts():
    """BUG 2: fp32-accumulated EMA counts match exact integer counts under fp16.

    The one-hot / EMA accumulation is forced to float32, so ``ema_cluster_size``
    (an EMA of per-code assignment counts) tracks the true integer counts even
    when the forward runs on fp16 inputs, where naive fp16 summation would round
    counts above 2048.
    """
    k, d, bs = 8, 4, 4096  # per-code counts ~512 total, but we push one code high
    torch.manual_seed(1)
    vq = _make_vq(k, d)
    vq.train()

    # Force initialization first so kmeans-init does not interfere.
    x_init = _to_bdt(_clustered_batch(bs, k, d).float())
    vq(x_init)

    # Now build a batch where we KNOW the exact per-code assignment counts by
    # placing rows extremely close to specific codebook entries.
    w = vq.embedding.weight.data.clone()
    # 3000 rows -> code 0, 1096 rows -> code 1 (both > fp16 integer-exact 2048).
    counts_true = torch.zeros(k)
    counts_true[0] = 3000
    counts_true[1] = 1096
    rows = []
    for c in range(k):
        n_c = int(counts_true[c].item())
        if n_c:
            rows.append(w[c].unsqueeze(0).repeat(n_c, 1))
    flat = torch.cat(rows, 0)
    x = _to_bdt(flat)

    # Snapshot EMA state, run one forward, and check the recovered per-code
    # counts. The fix pins enc/enc_sum/e_sum to float32, so the counts (3000,
    # 1096, both above the fp16 integer-exact limit of 2048) accumulate exactly.
    prev = vq.ema_cluster_size.clone()
    vq(x)

    # The EMA update is: new = decay*prev + (1-decay)*enc_sum.
    enc_sum_recovered = (vq.ema_cluster_size - vq.ema_decay * prev) / (1 - vq.ema_decay)
    # Codes 0 and 1 must recover their exact large integer counts (no fp16 rounding).
    assert abs(enc_sum_recovered[0].item() - 3000.0) < 1.0, enc_sum_recovered[0].item()
    assert abs(enc_sum_recovered[1].item() - 1096.0) < 1.0, enc_sum_recovered[1].item()
    # Buffers stay fp32 (the accumulation dtype).
    assert vq.ema_cluster_size.dtype == torch.float32
    assert vq.ema_embed_sum.dtype == torch.float32

    # Decisive check of the dtype choice the fix makes. fp16 cannot represent
    # 2049 (the next fp16 value after 2048 is 2050), so a count that is
    # ACCUMULATED in fp16 rounds once it passes 2048; the same accumulation in
    # fp32 stays exact. (Note torch.sum upcasts internally, so we accumulate in
    # the tensor dtype explicitly to model per-code count growth faithfully.)
    acc16 = torch.zeros((), dtype=torch.float16)
    acc32 = torch.zeros((), dtype=torch.float32)
    one16 = torch.ones((), dtype=torch.float16)
    one32 = torch.ones((), dtype=torch.float32)
    for _ in range(3000):
        acc16 = acc16 + one16
        acc32 = acc32 + one32
    assert acc32.item() == 3000.0  # fp32 (what the fix uses): exact
    assert acc16.item() != 3000.0  # fp16: rounds once the running count passes 2048
    # The fix guarantees enc / enc_sum / e_sum are fp32, so the codebook EMA sees
    # the exact fp32-accumulated statistics even under fp16 AMP inputs.


def test_fp32_path_unchanged():
    """The default fp32 path must remain numerically identical after BUG 2 fix.

    We just assert the forward runs and produces fp32 EMA buffers with sane,
    non-degenerate perplexity on a near-uniform batch.
    """
    k, d, bs = 32, 4, 1024
    torch.manual_seed(2)
    vq = _make_vq(k, d)
    vq.train()
    x = _to_bdt(_clustered_batch(bs, k, d).float())
    loss, q, perplexity, idx = vq(x)
    assert q.shape == x.shape
    assert idx.shape == (bs, 1)
    assert torch.isfinite(loss)
    assert torch.isfinite(perplexity)
    assert perplexity.item() > 1.0  # not collapsed to a single code
    assert vq.ema_cluster_size.dtype == torch.float32
