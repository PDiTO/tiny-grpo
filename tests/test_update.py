"""End-to-end checks of the policy update on a tiny model."""

import pytest
import torch
from torch import nn

from tests.conftest import EOS, PAD, VOCAB
from tiny_grpo.grpo import LossAggregation, LossConfig
from tiny_grpo.lora import apply_lora, lora_parameters
from tiny_grpo.policy import completion_mask, sequence_logprobs
from tiny_grpo.update import Rollouts, optimisation_step, score_rollouts


def _rollouts(advantages: list[float]) -> Rollouts:
    # Two completions of the same prompt (left padded to width 4).
    prompt_ids = torch.tensor([[PAD, 5, 9, 12], [PAD, 5, 9, 12]])
    prompt_mask = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 1]])
    completion_ids = torch.tensor([[4, 6, 7, EOS], [20, 21, EOS, PAD]])
    mask = completion_mask(completion_ids, [EOS])
    return Rollouts(prompt_ids, prompt_mask, completion_ids, mask, torch.tensor(advantages))


def _ragged_rollouts(n: int = 6, seed: int = 0) -> Rollouts:
    """``n`` rollouts with different prompt and completion lengths."""
    g = torch.Generator().manual_seed(seed)
    p_width, t_width = 7, 6
    prompt_ids = torch.full((n, p_width), PAD)
    prompt_mask = torch.zeros(n, p_width, dtype=torch.long)
    completion_ids = torch.full((n, t_width), PAD)
    for i in range(n):
        p_len = 2 + i % 5
        prompt_ids[i, p_width - p_len :] = torch.randint(2, VOCAB, (p_len,), generator=g)
        prompt_mask[i, p_width - p_len :] = 1
        c_len = 1 + (i * 2) % t_width
        completion_ids[i, :c_len] = torch.randint(2, VOCAB, (c_len,), generator=g)
        if c_len < t_width:
            completion_ids[i, c_len] = EOS
    mask = completion_mask(completion_ids, [EOS])
    advantages = torch.randn(n, generator=g)
    return Rollouts(prompt_ids, prompt_mask, completion_ids, mask, advantages)


def _perturb(model: nn.Module, scale: float = 0.3) -> None:
    with torch.no_grad():
        for p in lora_parameters(model):
            p.add_(torch.randn_like(p) * scale)


def _seq_logprob(model: nn.Module, r: Rollouts) -> torch.Tensor:
    with torch.no_grad():
        lp = sequence_logprobs(
            model, r.prompt_ids, r.prompt_mask, r.completion_ids, r.completion_mask
        )
    return (lp * r.completion_mask).sum(-1)


def _step(model: nn.Module, r: Rollouts, cfg: LossConfig, lr: float = 0.5) -> None:
    params = lora_parameters(model)
    # SGD so the effect of a single step is exactly one gradient step.
    optimizer = torch.optim.SGD(params, lr=lr)
    model.train()
    optimisation_step(
        model, params, optimizer, r, torch.arange(len(r)), cfg, 2, 4, max_grad_norm=1e9
    )
    model.eval()


@pytest.mark.parametrize("aggregation", ["sequence", "token", "constant"])
@pytest.mark.parametrize("kl_coef", [0.0, 0.04])
def test_one_step_moves_probability_towards_positive_advantage(
    tiny_model: nn.Module, aggregation: LossAggregation, kl_coef: float
) -> None:
    apply_lora(tiny_model, rank=4, alpha=8.0)
    _perturb(tiny_model)  # so the policy differs from the reference and the KL term is live
    r = _rollouts([1.0, -1.0])
    r.ref_logprobs = score_rollouts(tiny_model, r, 2, reference=True)

    before = _seq_logprob(tiny_model, r)
    _step(tiny_model, r, LossConfig(kl_coef=kl_coef, aggregation=aggregation))
    after = _seq_logprob(tiny_model, r)
    assert after[0] > before[0], "positive-advantage completion should become more likely"
    assert after[1] < before[1], "negative-advantage completion should become less likely"


def test_kl_term_alone_pulls_policy_towards_reference(tiny_model: nn.Module) -> None:
    apply_lora(tiny_model, rank=4, alpha=8.0)
    _perturb(tiny_model, scale=0.5)
    r = _rollouts([0.0, 0.0])  # no policy-gradient signal at all
    r.ref_logprobs = score_rollouts(tiny_model, r, 2, reference=True)

    def gap() -> float:
        current = score_rollouts(tiny_model, r, 2)
        assert r.ref_logprobs is not None
        return float(((current - r.ref_logprobs).abs() * r.completion_mask).sum())

    before = gap()
    _step(tiny_model, r, LossConfig(kl_coef=1.0), lr=0.1)
    assert gap() < before


@pytest.mark.parametrize("aggregation", ["sequence", "token", "constant"])
def test_micro_batching_matches_full_batch_gradient(
    tiny_model: nn.Module, aggregation: LossAggregation
) -> None:
    """A shuffled subset of a ragged batch, split into micro-batches that each
    get trimmed differently, gives the same gradient and stats as one pass."""
    apply_lora(tiny_model, rank=4, alpha=8.0)
    params = lora_parameters(tiny_model)
    _perturb(tiny_model, scale=0.1)
    r = _ragged_rollouts(8)
    current = score_rollouts(tiny_model, r, 8)
    r.old_logprobs = current + 0.3 * torch.randn_like(current)  # some tokens get clipped
    r.ref_logprobs = current - 0.2
    cfg = LossConfig(kl_coef=0.1, aggregation=aggregation)
    idx = torch.randperm(8)[:6]

    grads, stats = [], []
    for mbs in (6, 4, 1):
        opt = torch.optim.SGD(params, lr=0.0)  # lr 0: we only want the gradient
        stats.append(optimisation_step(tiny_model, params, opt, r, idx, cfg, mbs, 6, 1e9))
        grads.append(torch.cat([p.grad.flatten() for p in params if p.grad is not None]))
    for g, s in zip(grads[1:], stats[1:], strict=True):
        torch.testing.assert_close(g, grads[0], atol=1e-6, rtol=1e-4)
        assert s.loss == pytest.approx(stats[0].loss, rel=1e-4, abs=1e-6)
        assert s.kl == pytest.approx(stats[0].kl, rel=1e-4)
        assert s.clip_frac == pytest.approx(stats[0].clip_frac, abs=1e-6)
        assert s.ratio_mean == pytest.approx(stats[0].ratio_mean, rel=1e-5)
    assert 0.0 < stats[0].clip_frac < 1.0


def test_clipped_tokens_get_no_gradient(tiny_model: nn.Module) -> None:
    """If the policy has already moved far past the clip range in the direction
    of the advantage, the update has nothing left to push."""
    apply_lora(tiny_model, rank=4, alpha=8.0)
    params = lora_parameters(tiny_model)
    r = _rollouts([1.0, -1.0])
    current = score_rollouts(tiny_model, r, 2)
    # Pretend the sampling policy was much less likely (for A > 0) or much more
    # likely (for A < 0) to produce these tokens: ratio 2.0 and 0.5.
    r.old_logprobs = current - torch.tensor([[0.693], [-0.693]])
    opt = torch.optim.SGD(params, lr=0.0)
    stats = optimisation_step(
        tiny_model, params, opt, r, torch.arange(2), LossConfig(kl_coef=0.0), 2, 4
    )
    assert stats.clip_frac == pytest.approx(1.0)
    assert stats.grad_norm == 0.0


def test_select_trims_padding_without_changing_logprobs(tiny_model: nn.Module) -> None:
    r = _rollouts([1.0, -1.0])
    full = score_rollouts(tiny_model, r, 2)
    sub = r.select(torch.tensor([1]))
    assert sub.prompt_ids.shape[1] == 3  # the shared left-pad column is gone
    assert sub.completion_ids.shape[1] == 3  # only up to and including EOS
    with torch.no_grad():
        lp = sequence_logprobs(
            tiny_model, sub.prompt_ids, sub.prompt_mask, sub.completion_ids, sub.completion_mask
        )
    torch.testing.assert_close(lp[0], full[1, :3], atol=1e-5, rtol=1e-4)


def test_gradient_checkpointing_matches_plain_backward(tiny_model: nn.Module) -> None:
    """The real run checkpoints every decoder layer; it must not change anything."""
    apply_lora(tiny_model, rank=4, alpha=8.0)
    _perturb(tiny_model, scale=0.1)
    r = _ragged_rollouts(4)
    weights = torch.randn(4, r.completion_ids.shape[1])

    results = []
    for checkpointing in (False, True):
        if checkpointing:
            tiny_model.gradient_checkpointing_enable(  # type: ignore[operator]
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        tiny_model.train()
        tiny_model.zero_grad()
        lp = sequence_logprobs(
            tiny_model, r.prompt_ids, r.prompt_mask, r.completion_ids, r.completion_mask
        )
        (lp * weights * r.completion_mask).sum().backward()
        params = lora_parameters(tiny_model)
        results.append(
            (lp.detach(), torch.cat([p.grad.flatten() for p in params if p.grad is not None]))
        )
    torch.testing.assert_close(results[1][0], results[0][0])
    torch.testing.assert_close(results[1][1], results[0][1])
