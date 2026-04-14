"""End-to-end checks of the policy update on a tiny model."""

import pytest
import torch
from torch import nn

from tests.conftest import EOS, PAD
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


def _seq_logprob(model: nn.Module, r: Rollouts) -> torch.Tensor:
    with torch.no_grad():
        lp = sequence_logprobs(
            model, r.prompt_ids, r.prompt_mask, r.completion_ids, r.completion_mask
        )
    return (lp * r.completion_mask).sum(-1)


@pytest.mark.parametrize("aggregation", ["sequence", "token", "constant"])
@pytest.mark.parametrize("kl_coef", [0.0, 0.04])
def test_one_step_moves_probability_towards_positive_advantage(
    tiny_model: nn.Module, aggregation: LossAggregation, kl_coef: float
) -> None:
    apply_lora(tiny_model, rank=4, alpha=8.0)
    params = lora_parameters(tiny_model)
    # SGD so the effect of a single step is exactly one gradient step.
    optimizer = torch.optim.SGD(params, lr=0.5)
    r = _rollouts([1.0, -1.0])
    r.ref_logprobs = score_rollouts(tiny_model, r, 2, reference=True)

    before = _seq_logprob(tiny_model, r)
    tiny_model.train()
    optimisation_step(
        tiny_model,
        params,
        optimizer,
        r,
        torch.arange(2),
        LossConfig(kl_coef=kl_coef, aggregation=aggregation),
        micro_batch_size=2,
        max_len=4,
        max_grad_norm=1e9,
    )
    tiny_model.eval()
    after = _seq_logprob(tiny_model, r)
    assert after[0] > before[0], "positive-advantage completion should become more likely"
    assert after[1] < before[1], "negative-advantage completion should become less likely"


def test_zero_advantage_and_no_kl_gives_zero_gradient(tiny_model: nn.Module) -> None:
    apply_lora(tiny_model, rank=4, alpha=8.0)
    params = lora_parameters(tiny_model)
    optimizer = torch.optim.SGD(params, lr=0.5)
    r = _rollouts([0.0, 0.0])
    stats = optimisation_step(
        tiny_model, params, optimizer, r, torch.arange(2), LossConfig(kl_coef=0.0), 2, 4
    )
    assert stats.grad_norm == 0.0


@pytest.mark.parametrize("aggregation", ["sequence", "token", "constant"])
def test_micro_batching_matches_full_batch_gradient(
    tiny_model: nn.Module, aggregation: LossAggregation
) -> None:
    apply_lora(tiny_model, rank=4, alpha=8.0)
    params = lora_parameters(tiny_model)
    with torch.no_grad():
        for p in params:
            p.add_(torch.randn_like(p) * 0.1)
    r = _rollouts([0.7, -1.3])
    r.old_logprobs = score_rollouts(tiny_model, r, 2) + 0.05 * torch.randn(2, 4)
    r.ref_logprobs = score_rollouts(tiny_model, r, 2) - 0.1
    cfg = LossConfig(kl_coef=0.1, aggregation=aggregation)

    grads = []
    for mbs in (2, 1):
        opt = torch.optim.SGD(params, lr=0.0)  # lr 0: we only want the gradient
        optimisation_step(tiny_model, params, opt, r, torch.arange(2), cfg, mbs, 4, 1e9)
        grads.append(torch.cat([p.grad.flatten() for p in params if p.grad is not None]))
    torch.testing.assert_close(grads[0], grads[1], atol=1e-6, rtol=1e-4)


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
