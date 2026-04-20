"""Tests for the pure GRPO maths: advantages, clipped surrogate, KL, aggregation."""

import math

import pytest
import torch

from tiny_grpo.grpo import (
    LossAggregation,
    LossConfig,
    aggregate,
    clipped_surrogate,
    gather_logprobs,
    group_advantages,
    grpo_loss,
    kl_k3,
)

# ---------------------------------------------------------------------------
# Advantages
# ---------------------------------------------------------------------------


def test_advantages_hand_computed() -> None:
    # One group of four: rewards 1, 0, 0, 1 -> mean 0.5.
    # Unbiased std = sqrt(((.5^2) * 4) / 3) = sqrt(1/3).
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])
    adv = group_advantages(rewards, group_size=4, eps=0.0)
    s = math.sqrt(1 / 3)
    torch.testing.assert_close(adv, torch.tensor([0.5 / s, -0.5 / s, -0.5 / s, 0.5 / s]))


def test_advantages_are_per_group() -> None:
    # Two groups with different scales end up on the same scale.
    rewards = torch.tensor([0.0, 1.0, 10.0, 20.0])
    adv = group_advantages(rewards, group_size=2, eps=0.0)
    # Each pair is (-d, +d) / std with std = |d| * sqrt(2): +-1/sqrt(2).
    h = 1 / math.sqrt(2)
    torch.testing.assert_close(adv, torch.tensor([-h, h, -h, h]))


def test_advantages_single_correct_answer() -> None:
    # Group of 4 with rewards 1.1, 0.1, 0.1, 0.1 (one right, three formatted but wrong).
    rewards = torch.tensor([1.1, 0.1, 0.1, 0.1])
    adv = group_advantages(rewards, group_size=4, eps=0.0)
    # mean 0.35; deviations .75, -.25 x3; unbiased var = (.5625 + 3 * .0625) / 3 = .25
    torch.testing.assert_close(adv, torch.tensor([1.5, -0.5, -0.5, -0.5]))


def test_advantages_zero_variance_group() -> None:
    rewards = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    adv = group_advantages(rewards, group_size=3)
    assert torch.equal(adv, torch.zeros(6))
    assert torch.isfinite(adv).all()


def test_advantages_sum_to_zero_per_group() -> None:
    torch.manual_seed(0)
    rewards = torch.rand(5 * 8)
    for scale in (True, False):
        adv = group_advantages(rewards, group_size=8, scale=scale)
        torch.testing.assert_close(adv.view(5, 8).sum(1), torch.zeros(5), atol=1e-5, rtol=0)


def test_advantages_unit_std_when_scaled() -> None:
    torch.manual_seed(0)
    rewards = torch.rand(4 * 16)
    adv = group_advantages(rewards, group_size=16, eps=0.0)
    torch.testing.assert_close(adv.view(4, 16).std(1), torch.ones(4))


def test_dr_grpo_advantages_are_mean_centered_only() -> None:
    rewards = torch.tensor([1.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 1.0])
    adv = group_advantages(rewards, group_size=4, scale=False)
    torch.testing.assert_close(
        adv, torch.tensor([0.75, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, 0.75])
    )


def test_advantages_validation() -> None:
    with pytest.raises(ValueError, match="divisible"):
        group_advantages(torch.zeros(5), group_size=2)
    with pytest.raises(ValueError, match="at least 2"):
        group_advantages(torch.zeros(4), group_size=1)


# ---------------------------------------------------------------------------
# gather_logprobs
# ---------------------------------------------------------------------------


def test_gather_logprobs_matches_log_softmax() -> None:
    torch.manual_seed(0)
    logits = torch.randn(3, 5, 11)
    tokens = torch.randint(0, 11, (3, 5))
    expected = logits.log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(gather_logprobs(logits, tokens), expected)


def test_gather_logprobs_temperature() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 7)
    tokens = torch.randint(0, 7, (2, 4))
    expected = (logits / 0.7).log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(gather_logprobs(logits, tokens, temperature=0.7), expected)


# ---------------------------------------------------------------------------
# Clipped surrogate
# ---------------------------------------------------------------------------


def _surrogate_grad(ratio: float, advantage: float, eps: float = 0.2) -> float:
    logp = torch.tensor([[math.log(ratio)]], requires_grad=True)
    obj, _ = clipped_surrogate(logp, torch.zeros(1, 1), torch.tensor([advantage]), eps, eps)
    obj.sum().backward()
    assert logp.grad is not None
    return float(logp.grad)


@pytest.mark.parametrize(
    ("ratio", "advantage", "zero_grad"),
    [
        # A > 0: the objective wants to push the ratio up, and stops at 1 + eps.
        (1.5, 1.0, True),
        (1.1, 1.0, False),
        (0.5, 1.0, False),  # below the range: unclipped branch is the min, still learns
        # A < 0: wants to push the ratio down, and stops at 1 - eps.
        (0.5, -1.0, True),
        (0.9, -1.0, False),
        (1.5, -1.0, False),  # above the range: the min keeps the (worse) unclipped term
    ],
)
def test_clip_gradient(ratio: float, advantage: float, zero_grad: bool) -> None:
    g = _surrogate_grad(ratio, advantage)
    if zero_grad:
        assert g == 0.0
    else:
        # d/dlogp (ratio * A) = ratio * A
        assert g == pytest.approx(ratio * advantage)


def test_clip_asymmetric_range() -> None:
    logp = torch.log(torch.tensor([[1.25, 1.35]])).requires_grad_(True)
    obj, clipped = clipped_surrogate(logp, torch.zeros(1, 2), torch.tensor([1.0]), 0.2, 0.28)
    obj.sum().backward()
    assert logp.grad is not None
    assert logp.grad[0, 0] > 0  # 1.25 is inside the clip-higher range
    assert logp.grad[0, 1] == 0  # 1.35 is outside it
    assert clipped.tolist() == [[False, True]]


def _reference_surrogate(
    logp: torch.Tensor, old: torch.Tensor, adv: torch.Tensor, eps: float
) -> torch.Tensor:
    """Straight transcription of the PPO-clip formula, per element."""
    out = torch.empty_like(logp)
    for i in range(logp.shape[0]):
        for j in range(logp.shape[1]):
            r = math.exp(float(logp[i, j]) - float(old[i, j]))
            a = float(adv[i])
            out[i, j] = min(r * a, max(min(r, 1 + eps), 1 - eps) * a)
    return out


def test_clip_matches_reference_on_random_tensors() -> None:
    torch.manual_seed(0)
    logp = torch.randn(6, 9) * 0.3
    old = logp + torch.randn(6, 9) * 0.3
    adv = torch.randn(6)
    obj, _ = clipped_surrogate(logp, old, adv, 0.2, 0.2)
    torch.testing.assert_close(obj, _reference_surrogate(logp, old, adv, 0.2))


def test_clip_is_identity_on_policy() -> None:
    # With ratio == 1 the objective is just A and nothing is clipped.
    logp = torch.randn(3, 4)
    adv = torch.tensor([1.0, -2.0, 0.5])
    obj, clipped = clipped_surrogate(logp, logp.clone(), adv)
    torch.testing.assert_close(obj, adv.unsqueeze(-1).expand(3, 4))
    assert not clipped.any()


# ---------------------------------------------------------------------------
# KL k3
# ---------------------------------------------------------------------------


def test_k3_zero_when_equal() -> None:
    logp = torch.randn(4, 7)
    assert torch.equal(kl_k3(logp, logp), torch.zeros(4, 7))


def test_k3_non_negative() -> None:
    torch.manual_seed(0)
    logp = torch.randn(1000) * 3
    ref = torch.randn(1000) * 3
    assert (kl_k3(logp, ref) >= 0).all()


def test_k3_gradient_zero_when_equal() -> None:
    logp = torch.randn(5, requires_grad=True)
    kl_k3(logp, logp.detach()).sum().backward()
    assert logp.grad is not None
    torch.testing.assert_close(logp.grad, torch.zeros(5))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_k3_monte_carlo_matches_exact_kl(seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    p = torch.softmax(torch.randn(6, generator=g), 0)
    q = torch.softmax(torch.randn(6, generator=g), 0)
    exact = float((p * (p.log() - q.log())).sum())

    samples = torch.multinomial(p, 200_000, replacement=True, generator=g)
    estimate = float(kl_k3(p.log()[samples], q.log()[samples]).mean())
    assert estimate == pytest.approx(exact, rel=0.03, abs=2e-3)


def test_k3_has_lower_variance_than_k1() -> None:
    # The reason k3 is used at all: same mean, less noise per sample.
    g = torch.Generator().manual_seed(0)
    p = torch.softmax(torch.randn(8, generator=g) * 0.5, 0)
    q = torch.softmax(torch.randn(8, generator=g) * 0.5, 0)
    x = torch.multinomial(p, 50_000, replacement=True, generator=g)
    k1 = p.log()[x] - q.log()[x]
    k3 = kl_k3(p.log()[x], q.log()[x])
    assert k3.var() < k1.var()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

# Two completions: one of length 1 with loss 4, one of length 3 with losses 1, 1, 1.
PER_TOKEN = torch.tensor([[4.0, 9.0, 9.0], [1.0, 1.0, 1.0]])
MASK = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])


def test_aggregate_sequence() -> None:
    # Mean per sequence (4 and 1), then mean over sequences.
    assert float(aggregate(PER_TOKEN, MASK, "sequence")) == pytest.approx(2.5)


def test_aggregate_token() -> None:
    # (4 + 1 + 1 + 1) / 4 tokens.
    assert float(aggregate(PER_TOKEN, MASK, "token")) == pytest.approx(1.75)


def test_aggregate_constant() -> None:
    # (4 + 1 + 1 + 1) / (2 sequences * 3 max length).
    assert float(aggregate(PER_TOKEN, MASK, "constant")) == pytest.approx(7 / 6)
    assert float(aggregate(PER_TOKEN, MASK, "constant", max_len=7)) == pytest.approx(0.5)


def test_aggregate_padding_is_ignored() -> None:
    # The 9s sit under mask=0 and must not leak into any mode.
    changed = PER_TOKEN.clone()
    changed[0, 1:] = 1e6
    modes: tuple[LossAggregation, ...] = ("sequence", "token", "constant")
    for mode in modes:
        assert float(aggregate(changed, MASK, mode)) == float(aggregate(PER_TOKEN, MASK, mode))


def test_sequence_mode_weights_tokens_by_inverse_length() -> None:
    # The per-token gradient weight is 1 / (B * |o_i|) for "sequence" and
    # 1 / sum|o| for "token".
    x = torch.zeros(2, 3, requires_grad=True)
    aggregate(x, MASK, "sequence").backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.tensor([[0.5, 0, 0], [1 / 6, 1 / 6, 1 / 6]]))

    x = torch.zeros(2, 3, requires_grad=True)
    aggregate(x, MASK, "token").backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad, MASK / 4)


@pytest.mark.parametrize("mode", ["sequence", "token", "constant"])
def test_aggregate_microbatches_sum_to_full_batch(mode: LossAggregation) -> None:
    torch.manual_seed(0)
    per_token = torch.randn(6, 5)
    mask = (torch.rand(6, 5) > 0.3).float()
    mask[:, 0] = 1.0
    full = aggregate(per_token, mask, mode)
    parts = torch.stack(
        [
            aggregate(
                per_token[i : i + 2],
                mask[i : i + 2],
                mode,
                num_sequences=6,
                num_tokens=float(mask.sum()),
                max_len=5,
            )
            for i in range(0, 6, 2)
        ]
    ).sum()
    torch.testing.assert_close(parts, full)


def test_aggregate_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown"):
        aggregate(PER_TOKEN, MASK, "bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Full loss
# ---------------------------------------------------------------------------


def test_grpo_loss_on_policy_equals_negative_mean_advantage() -> None:
    # With ratio 1 and no KL, the token-mean loss is -(sum_i |o_i| A_i) / sum_i |o_i|.
    logp = torch.randn(2, 3)
    adv = torch.tensor([1.0, -1.0])
    out = grpo_loss(logp, logp, None, adv, MASK, LossConfig(kl_coef=0.0, aggregation="token"))
    assert float(out.loss) == pytest.approx(-(1 * 1.0 + 3 * -1.0) / 4)
    assert float(out.kl) == 0.0
    assert float(out.clip_frac) == 0.0
    assert float(out.ratio_mean) == pytest.approx(1.0)


def test_grpo_loss_adds_kl_penalty() -> None:
    logp = torch.zeros(2, 3)
    ref = torch.full((2, 3), -0.5)
    adv = torch.zeros(2)
    cfg = LossConfig(kl_coef=0.1, aggregation="token")
    out = grpo_loss(logp, logp, ref, adv, MASK, cfg)
    k3 = math.exp(-0.5) + 0.5 - 1
    assert float(out.kl) == pytest.approx(k3)
    assert float(out.loss) == pytest.approx(0.1 * k3)


def test_grpo_loss_requires_ref_for_kl() -> None:
    logp = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="ref_logprobs"):
        grpo_loss(logp, logp, None, torch.zeros(2), MASK, LossConfig(kl_coef=0.1))


def test_grpo_loss_ignores_junk_under_the_mask() -> None:
    """Log-probs at padding positions can be anything, including values whose
    difference would overflow exp(). The loss and stats must not notice."""
    torch.manual_seed(0)
    logp = torch.randn(2, 3) * 0.1 - 1.0
    old = logp + 0.05
    ref = logp - 0.05
    adv = torch.tensor([1.0, -1.0])
    cfg = LossConfig(kl_coef=0.1)
    clean = grpo_loss(logp, old, ref, adv, MASK, cfg)
    junk_old, junk_ref = old.clone(), ref.clone()
    junk_old[0, 1:] = -1000.0  # exp(logp - old) would be inf
    junk_ref[0, 1:] = 1000.0
    dirty = grpo_loss(logp, junk_old, junk_ref, adv, MASK, cfg)
    for a, b in [
        (clean.loss, dirty.loss),
        (clean.kl, dirty.kl),
        (clean.ratio_mean, dirty.ratio_mean),
    ]:
        assert torch.isfinite(b)
        torch.testing.assert_close(a, b)
