"""The GRPO objective, written out one piece at a time.

Shapes used throughout:

* ``B``: number of completions in the batch (prompts x group size)
* ``T``: completion length (right padded to the longest completion)
* ``mask``: ``(B, T)`` float, 1 for completion tokens that count, 0 for padding

Nothing in this module knows about models or tokenizers. It takes tensors of
per-token log-probabilities and returns a scalar loss, which is what makes it
easy to test against hand-computed values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

LossAggregation = Literal["sequence", "token", "constant"]


# ---------------------------------------------------------------------------
# Advantages
# ---------------------------------------------------------------------------


def group_advantages(
    rewards: Tensor, group_size: int, scale: bool = True, eps: float = 1e-4
) -> Tensor:
    """Group-relative advantages.

    ``rewards`` is flat, ``(num_prompts * group_size,)``, with the completions for
    each prompt stored contiguously. For each group::

        A_i = (r_i - mean(r)) / (std(r) + eps)     if scale (GRPO)
        A_i =  r_i - mean(r)                       if not scale (Dr. GRPO)

    ``std`` is the unbiased sample standard deviation, which is what the DeepSeek
    reference code and TRL use. A group where every completion got the same
    reward has zero advantage everywhere and contributes no policy gradient.
    """
    if rewards.ndim != 1 or rewards.numel() % group_size != 0:
        raise ValueError("rewards must be flat with length divisible by group_size")
    if group_size < 2:
        raise ValueError("group_size must be at least 2 for a group baseline")
    grouped = rewards.view(-1, group_size).float()
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if scale:
        centered = centered / (grouped.std(dim=1, keepdim=True) + eps)
    return centered.view(-1)


# ---------------------------------------------------------------------------
# Log-probabilities
# ---------------------------------------------------------------------------


def gather_logprobs(logits: Tensor, tokens: Tensor, temperature: float = 1.0) -> Tensor:
    """``log p(tokens[b, t] | ...)`` from ``logits[b, t]``.

    ``logits`` is ``(B, T, V)`` and already aligned with ``tokens`` ``(B, T)``,
    i.e. ``logits[:, t]`` is the distribution that produced ``tokens[:, t]``.

    Uses ``logit - logsumexp(logits)`` rather than ``log_softmax(...).gather`` so
    the full ``(B, T, V)`` log-softmax is never materialised. With a 150k token
    vocabulary that is the difference between fitting in memory or not.
    """
    if temperature != 1.0:
        logits = logits / temperature
    picked = logits.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return picked.float() - torch.logsumexp(logits.float(), dim=-1)


# ---------------------------------------------------------------------------
# Objective pieces
# ---------------------------------------------------------------------------


def clipped_surrogate(
    logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
) -> tuple[Tensor, Tensor]:
    """PPO's clipped surrogate, per token.

    ``advantages`` is ``(B,)`` (one value per completion, shared by its tokens) or
    ``(B, T)``. Returns ``(objective, clipped)`` where ``objective`` is the thing
    to maximise and ``clipped`` flags tokens where the clip was the active branch
    of the ``min`` (so they get no gradient).

    ``clip_low`` and ``clip_high`` can differ ("clip-higher" in DAPO), but
    default to the symmetric PPO range.
    """
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(-1)
    ratio = torch.exp(logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
    clipped = clipped_ratio * advantages
    objective = torch.minimum(unclipped, clipped)
    was_clipped = (clipped < unclipped).detach()
    return objective, was_clipped


def kl_k3(logprobs: Tensor, ref_logprobs: Tensor) -> Tensor:
    """Per-token estimate of ``KL(pi_theta || pi_ref)`` using Schulman's k3.

    With ``x ~ pi_theta`` and ``r = pi_ref(x) / pi_theta(x)``::

        k3 = r - log(r) - 1

    It is unbiased (``E[r] = 1``), always non-negative because ``log r <= r - 1``,
    and has lower variance than the naive ``-log r``. This is the estimator in the
    DeepSeekMath GRPO objective.
    """
    log_ratio = ref_logprobs - logprobs
    # Clamp the exponent so a single wild token cannot overflow to inf.
    return torch.exp(log_ratio.clamp(max=20.0)) - log_ratio - 1.0


def aggregate(
    per_token: Tensor,
    mask: Tensor,
    mode: LossAggregation,
    *,
    num_sequences: int | None = None,
    num_tokens: float | None = None,
    max_len: int | None = None,
) -> Tensor:
    """Reduce a ``(B, T)`` per-token loss to a scalar.

    * ``"sequence"``: average over each completion's tokens, then over
      completions. This is the original GRPO formula. Every completion carries
      equal weight, so each token in a long completion counts for less, which
      biases the gradient towards short correct answers and long wrong ones.
    * ``"token"``: sum over all tokens in the batch divided by the number of
      tokens. DAPO's choice. Every token carries equal weight.
    * ``"constant"``: sum over all tokens divided by ``num_sequences * max_len``,
      a constant that does not depend on the sampled lengths. Dr. GRPO's choice;
      it is the token sum up to a fixed scale, so no length bias at all.

    The optional ``num_sequences``, ``num_tokens`` and ``max_len`` let a caller
    doing gradient accumulation normalise each micro-batch by the totals of the
    whole optimisation batch, so that summing micro-batch losses reproduces the
    full-batch loss exactly. Leave them as ``None`` for a single batch.
    """
    masked = per_token * mask
    if mode == "sequence":
        per_seq = masked.sum(-1) / mask.sum(-1).clamp(min=1.0)
        n = num_sequences if num_sequences is not None else per_token.shape[0]
        return per_seq.sum() / n
    if mode == "token":
        n_tok = num_tokens if num_tokens is not None else mask.sum().clamp(min=1.0)
        return masked.sum() / n_tok
    if mode == "constant":
        n = num_sequences if num_sequences is not None else per_token.shape[0]
        length = max_len if max_len is not None else per_token.shape[1]
        return masked.sum() / (n * length)
    raise ValueError(f"unknown loss aggregation {mode!r}")


# ---------------------------------------------------------------------------
# The full loss
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LossConfig:
    clip_low: float = 0.2
    clip_high: float = 0.2
    kl_coef: float = 0.04
    aggregation: LossAggregation = "token"


@dataclass
class LossOutput:
    loss: Tensor
    # Diagnostics, averaged over masked tokens and detached.
    kl: Tensor
    clip_frac: Tensor
    ratio_mean: Tensor


def grpo_loss(
    logprobs: Tensor,
    old_logprobs: Tensor,
    ref_logprobs: Tensor | None,
    advantages: Tensor,
    mask: Tensor,
    cfg: LossConfig,
    *,
    num_sequences: int | None = None,
    num_tokens: float | None = None,
    max_len: int | None = None,
) -> LossOutput:
    """``-J_GRPO``, ready for ``.backward()``.

    Per token::

        L_t = -( min(rho_t A, clip(rho_t, 1-eps_lo, 1+eps_hi) A) - beta * k3_t )

    then reduced with :func:`aggregate`. ``ref_logprobs`` may be ``None`` when
    ``kl_coef == 0``, which skips the reference forward pass entirely.
    """
    surrogate, was_clipped = clipped_surrogate(
        logprobs, old_logprobs, advantages, cfg.clip_low, cfg.clip_high
    )
    per_token = -surrogate
    if cfg.kl_coef != 0.0:
        if ref_logprobs is None:
            raise ValueError("ref_logprobs is required when kl_coef != 0")
        kl = kl_k3(logprobs, ref_logprobs)
        per_token = per_token + cfg.kl_coef * kl
    elif ref_logprobs is not None:
        kl = kl_k3(logprobs, ref_logprobs)
    else:
        kl = torch.zeros_like(logprobs)

    loss = aggregate(
        per_token,
        mask,
        cfg.aggregation,
        num_sequences=num_sequences,
        num_tokens=num_tokens,
        max_len=max_len,
    )
    with torch.no_grad():
        n = mask.sum().clamp(min=1.0)
        stats = LossOutput(
            loss=loss,
            kl=(kl * mask).sum() / n,
            clip_frac=(was_clipped.float() * mask).sum() / n,
            ratio_mean=(torch.exp(logprobs - old_logprobs) * mask).sum() / n,
        )
    return stats
