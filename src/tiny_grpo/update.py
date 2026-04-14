"""The policy update: turn a batch of scored rollouts into optimiser steps.

This is model-agnostic apart from calling :func:`sequence_logprobs`, so the
tests drive it with a tiny randomly initialised transformer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tiny_grpo.grpo import LossConfig, grpo_loss
from tiny_grpo.lora import adapters_disabled
from tiny_grpo.policy import sequence_logprobs


@dataclass
class Rollouts:
    prompt_ids: Tensor  # (B, P)
    prompt_mask: Tensor  # (B, P)
    completion_ids: Tensor  # (B, T)
    completion_mask: Tensor  # (B, T)
    advantages: Tensor  # (B,)
    old_logprobs: Tensor | None = None  # (B, T); None means "same as current"
    ref_logprobs: Tensor | None = None  # (B, T); None means no KL term

    def __len__(self) -> int:
        return int(self.prompt_ids.shape[0])

    def select(self, idx: Tensor) -> Rollouts:
        """Rows ``idx``, with all-padding columns trimmed off both ends.

        Trimming is purely a speed optimisation: left-pad columns shared by every
        selected prompt, and right-pad columns past the longest selected
        completion, do not change any log-prob.
        """
        idx = idx.to(self.prompt_ids.device)
        pm = self.prompt_mask[idx]
        cm = self.completion_mask[idx]
        p_start = int((pm.sum(0) > 0).float().argmax())
        t_end = int(cm.sum(0).nonzero().max()) + 1 if cm.any() else 1

        def cut(x: Tensor | None) -> Tensor | None:
            return None if x is None else x[idx][:, :t_end]

        return Rollouts(
            prompt_ids=self.prompt_ids[idx][:, p_start:],
            prompt_mask=pm[:, p_start:],
            completion_ids=self.completion_ids[idx][:, :t_end],
            completion_mask=cm[:, :t_end],
            advantages=self.advantages[idx],
            old_logprobs=cut(self.old_logprobs),
            ref_logprobs=cut(self.ref_logprobs),
        )


def _logprobs(model: nn.Module, r: Rollouts, temperature: float) -> Tensor:
    return sequence_logprobs(
        model, r.prompt_ids, r.prompt_mask, r.completion_ids, r.completion_mask, temperature
    )


@torch.no_grad()
def score_rollouts(
    model: nn.Module,
    rollouts: Rollouts,
    micro_batch_size: int,
    temperature: float = 1.0,
    reference: bool = False,
) -> Tensor:
    """Log-probs for every rollout under the current policy, or under the
    reference policy (LoRA switched off) if ``reference`` is set."""
    t = rollouts.completion_ids.shape[1]
    out = torch.zeros(len(rollouts), t, device=rollouts.completion_ids.device)
    for start in range(0, len(rollouts), micro_batch_size):
        end = min(start + micro_batch_size, len(rollouts))
        sub = rollouts.select(torch.arange(start, end))
        if reference:
            with adapters_disabled(model):
                lp = _logprobs(model, sub, temperature)
        else:
            lp = _logprobs(model, sub, temperature)
        out[start:end, : lp.shape[1]] = lp
    return out


@dataclass
class UpdateStats:
    loss: float
    kl: float
    clip_frac: float
    ratio_mean: float
    grad_norm: float


def optimisation_step(
    model: nn.Module,
    params: list[nn.Parameter],
    optimizer: torch.optim.Optimizer,
    rollouts: Rollouts,
    idx: Tensor,
    loss_cfg: LossConfig,
    micro_batch_size: int,
    max_len: int,
    max_grad_norm: float = 1.0,
    temperature: float = 1.0,
    on_step: Callable[[], None] | None = None,
) -> UpdateStats:
    """One optimiser step on the mini-batch ``idx``, accumulated over micro-batches.

    Each micro-batch loss is normalised by the *mini-batch* totals (sequences,
    tokens, or ``max_len`` for the constant normaliser), so the summed gradient
    equals the gradient of the loss over the whole mini-batch. Normalising each
    micro-batch by its own token count and averaging would quietly re-weight
    tokens depending on which micro-batch they landed in.
    """
    mask_all = rollouts.completion_mask[idx]
    num_sequences = int(idx.numel())
    num_tokens = float(mask_all.sum().clamp(min=1.0))

    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "kl": 0.0, "clip": 0.0, "ratio": 0.0}
    for start in range(0, num_sequences, micro_batch_size):
        sub = rollouts.select(idx[start : start + micro_batch_size])
        logprobs = _logprobs(model, sub, temperature)
        old = sub.old_logprobs if sub.old_logprobs is not None else logprobs.detach()
        out = grpo_loss(
            logprobs,
            old,
            sub.ref_logprobs,
            sub.advantages,
            sub.completion_mask,
            loss_cfg,
            num_sequences=num_sequences,
            num_tokens=num_tokens,
            max_len=max_len,
        )
        out.loss.backward()
        weight = float(sub.completion_mask.sum()) / num_tokens
        totals["loss"] += float(out.loss.detach())
        totals["kl"] += float(out.kl) * weight
        totals["clip"] += float(out.clip_frac) * weight
        totals["ratio"] += float(out.ratio_mean) * weight

    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    optimizer.step()
    if on_step is not None:
        on_step()
    return UpdateStats(
        loss=totals["loss"],
        kl=totals["kl"],
        clip_frac=totals["clip"],
        ratio_mean=totals["ratio"],
        grad_norm=float(grad_norm),
    )
