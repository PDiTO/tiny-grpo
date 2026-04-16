"""Held-out evaluation: greedy pass@1 and mean reward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from tiny_grpo.policy import (
    SamplingParams,
    decode_completions,
    encode_prompts,
    eos_token_ids,
    generate,
    is_truncated,
)
from tiny_grpo.tasks import Problem, Task


@dataclass
class EvalResult:
    accuracy: float  # greedy pass@1
    lenient_accuracy: float  # correct answer anywhere, format ignored
    mean_reward: float
    format_rate: float
    mean_length: float  # completion tokens, including EOS
    truncated_rate: float  # hit max_new_tokens without EOS
    samples: list[dict[str, Any]]

    def metrics(self, prefix: str = "eval/") -> dict[str, float]:
        return {
            f"{prefix}accuracy": self.accuracy,
            f"{prefix}lenient_accuracy": self.lenient_accuracy,
            f"{prefix}reward": self.mean_reward,
            f"{prefix}format_rate": self.format_rate,
            f"{prefix}length": self.mean_length,
            f"{prefix}truncated_rate": self.truncated_rate,
        }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    tokenizer: Any,
    task: Task,
    problems: list[Problem],
    max_new_tokens: int,
    batch_size: int = 64,
) -> EvalResult:
    device = next(model.parameters()).device
    eos_ids = eos_token_ids(model, tokenizer)
    params = SamplingParams(max_new_tokens=max_new_tokens, greedy=True)
    samples: list[dict[str, Any]] = []
    for start in range(0, len(problems), batch_size):
        chunk = problems[start : start + batch_size]
        prompt_ids, prompt_mask = encode_prompts(tokenizer, [p.question for p in chunk], device)
        completion_ids, mask = generate(
            model, prompt_ids, prompt_mask, params, tokenizer.pad_token_id, eos_ids
        )
        texts = decode_completions(tokenizer, completion_ids, mask)
        lengths = mask.sum(-1).tolist()
        truncated = is_truncated(completion_ids, eos_ids).tolist()
        for i, (p, text) in enumerate(zip(chunk, texts, strict=True)):
            r = task.reward(text, p)
            samples.append(
                {
                    "question": p.question,
                    "meta": p.meta,
                    "completion": text,
                    "reward": r.reward,
                    "correct": r.correct,
                    "format_ok": r.format_ok,
                    "lenient_correct": task.lenient_correct(text, p),
                    "length": int(lengths[i]),
                    "truncated": bool(truncated[i]),
                }
            )
    n = max(len(samples), 1)
    return EvalResult(
        accuracy=sum(s["correct"] for s in samples) / n,
        lenient_accuracy=sum(s["lenient_correct"] for s in samples) / n,
        mean_reward=sum(s["reward"] for s in samples) / n,
        format_rate=sum(s["format_ok"] for s in samples) / n,
        mean_length=sum(s["length"] for s in samples) / n,
        truncated_rate=sum(s["truncated"] for s in samples) / n,
        samples=samples,
    )
