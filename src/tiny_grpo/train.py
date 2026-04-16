"""The GRPO training loop.

One step:

1. Draw ``prompts_per_step`` problems and sample ``group_size`` completions for
   each with the current policy.
2. Score every completion with the task's reward function.
3. Turn rewards into group-relative advantages.
4. Record log-probs under the sampling policy (``pi_old``) and, if the KL term
   is on, under the reference policy (``pi_ref``, adapters off).
5. Run ``epochs_per_batch`` passes over the rollouts, each split into
   ``minibatches_per_epoch`` optimiser steps.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from tiny_grpo.config import Config
from tiny_grpo.evaluate import evaluate
from tiny_grpo.grpo import LossConfig, group_advantages
from tiny_grpo.lora import load_adapters, lora_parameters, save_adapters
from tiny_grpo.policy import (
    SamplingParams,
    decode_completions,
    encode_prompts,
    eos_token_ids,
    generate,
    is_truncated,
    load_policy,
    pick_device,
)
from tiny_grpo.tasks import Problem, Task, eval_set, make_task, train_stream
from tiny_grpo.update import Rollouts, UpdateStats, optimisation_step, score_rollouts


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def log(self, record: dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def cosine_schedule(step: int, total: int, warmup: int, min_ratio: float) -> float:
    """LR multiplier: linear warmup to 1, then cosine decay to ``min_ratio``."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _empty_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def collect_rollouts(
    model: nn.Module,
    tokenizer: Any,
    task: Task,
    problems: list[Problem],
    cfg: Config,
    eos_ids: list[int],
) -> tuple[Rollouts, list[str], dict[str, Any]]:
    """Sample, score and compute advantages for one batch of prompts."""
    device = next(model.parameters()).device
    r = cfg.rollout
    questions = [p.question for p in problems for _ in range(r.group_size)]
    prompt_ids, prompt_mask = encode_prompts(tokenizer, questions, device)
    params = SamplingParams(r.max_new_tokens, r.temperature, r.top_p)

    ids_chunks, mask_chunks = [], []
    for start in range(0, len(questions), r.generation_batch_size):
        sl = slice(start, start + r.generation_batch_size)
        ids, mask = generate(
            model, prompt_ids[sl], prompt_mask[sl], params, tokenizer.pad_token_id, eos_ids
        )
        # Chunks can end at different lengths; pad to max_new_tokens.
        pad = r.max_new_tokens - ids.shape[1]
        ids_chunks.append(torch.nn.functional.pad(ids, (0, pad), value=tokenizer.pad_token_id))
        mask_chunks.append(torch.nn.functional.pad(mask, (0, pad), value=0.0))
    completion_ids = torch.cat(ids_chunks)
    completion_mask = torch.cat(mask_chunks)
    # Trim to the longest completion in the batch.
    width = int(completion_mask.sum(-1).max())
    completion_ids = completion_ids[:, :width]
    completion_mask = completion_mask[:, :width]

    texts = decode_completions(tokenizer, completion_ids, completion_mask)
    group_problems = [p for p in problems for _ in range(r.group_size)]
    scored = [task.reward(t, p) for t, p in zip(texts, group_problems, strict=True)]
    rewards = torch.tensor([s.reward for s in scored], dtype=torch.float32)
    advantages = group_advantages(rewards, r.group_size, scale=cfg.loss.scale_advantages)

    grouped = rewards.view(-1, r.group_size)
    lengths = completion_mask.sum(-1).cpu()
    stats = {
        "reward": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "accuracy": float(np.mean([s.correct for s in scored])),
        "format_rate": float(np.mean([s.format_ok for s in scored])),
        "length": float(lengths.mean()),
        "length_max": float(lengths.max()),
        "truncated_rate": float(is_truncated(completion_ids, eos_ids).float().mean()),
        # Groups where every completion scored the same carry no signal.
        "zero_std_groups": float((grouped.std(dim=1) == 0).float().mean()),
    }
    rollouts = Rollouts(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        advantages=advantages.to(device),
    )
    return rollouts, texts, stats


def train(cfg: Config, resume: bool = False) -> None:
    seed_everything(cfg.train.seed)
    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    metrics = JsonlLogger(out_dir / "metrics.jsonl")
    samples_log = JsonlLogger(out_dir / "samples.jsonl")

    device = pick_device(cfg.model.device)
    model, tokenizer = load_policy(
        cfg.model.name,
        device,
        cfg.model.dtype,
        cfg.lora.rank,
        cfg.lora.alpha,
        cfg.lora.targets,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
    )
    eos_ids = eos_token_ids(model, tokenizer)
    params = lora_parameters(model)
    n_trainable = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"device={device} trainable={n_trainable:,} / {n_total:,} params", flush=True)

    o = cfg.optim
    optimizer = torch.optim.AdamW(params, lr=o.lr, betas=o.betas, weight_decay=o.weight_decay)
    total_updates = cfg.train.steps * o.epochs_per_batch * o.minibatches_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: cosine_schedule(s, total_updates, o.warmup_steps, o.min_lr_ratio),
    )
    loss_cfg = LossConfig(
        clip_low=cfg.loss.clip_low,
        clip_high=cfg.loss.clip_high,
        kl_coef=cfg.loss.kl_coef,
        aggregation=cfg.loss.aggregation,
    )

    task = make_task(cfg.task.name, cfg.task.format_weight)
    eval_problems = eval_set(task, cfg.task.eval_size, cfg.task.eval_seed, cfg.task.train_seed)
    stream = train_stream(task, cfg.task.train_seed)

    start_step = 0
    state_path = out_dir / "checkpoints" / "last" / "state.pt"
    if resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        load_adapters(model, state_path.parent / "adapters.pt")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
        # Fast-forward the problem stream so we don't retrain on the same prompts.
        for _ in range(start_step * cfg.rollout.prompts_per_step):
            next(stream)
        print(f"resumed from step {start_step}", flush=True)

    def run_eval(step: int) -> None:
        t0 = time.time()
        result = evaluate(
            model,
            tokenizer,
            task,
            eval_problems,
            cfg.rollout.max_new_tokens,
            cfg.rollout.generation_batch_size,
        )
        record = {"step": step, **result.metrics(), "eval/seconds": time.time() - t0}
        metrics.log(record)
        with (out_dir / f"eval_step{step:04d}.jsonl").open("w") as f:
            for s in result.samples:
                f.write(json.dumps(s) + "\n")
        print(
            f"[eval step {step}] acc={result.accuracy:.3f} reward={result.mean_reward:.3f} "
            f"format={result.format_rate:.3f} len={result.mean_length:.1f}",
            flush=True,
        )
        _empty_cache(device)

    if start_step == 0:
        run_eval(0)

    batch_size = cfg.rollout.prompts_per_step * cfg.rollout.group_size
    mb_size = batch_size // o.minibatches_per_epoch
    for step in range(start_step, cfg.train.steps):
        t_step = time.time()
        problems = [next(stream) for _ in range(cfg.rollout.prompts_per_step)]
        rollouts, texts, stats = collect_rollouts(model, tokenizer, task, problems, cfg, eos_ids)
        t_rollout = time.time() - t_step

        t0 = time.time()
        mbs = o.micro_batch_size
        temp = cfg.rollout.temperature
        # pi_old only differs from the live policy after the first optimiser
        # step, so with a single update per batch we can skip this pass.
        if o.epochs_per_batch * o.minibatches_per_epoch > 1:
            rollouts.old_logprobs = score_rollouts(model, rollouts, mbs, temp)
        if loss_cfg.kl_coef != 0.0:
            rollouts.ref_logprobs = score_rollouts(model, rollouts, mbs, temp, reference=True)
        t_score = time.time() - t0

        t0 = time.time()
        model.train()
        updates: list[UpdateStats] = []
        for _ in range(o.epochs_per_batch):
            perm = torch.randperm(batch_size)
            for m in range(o.minibatches_per_epoch):
                idx = perm[m * mb_size : (m + 1) * mb_size]
                updates.append(
                    optimisation_step(
                        model,
                        params,
                        optimizer,
                        rollouts,
                        idx,
                        loss_cfg,
                        mbs,
                        max_len=cfg.rollout.max_new_tokens,
                        max_grad_norm=o.max_grad_norm,
                        temperature=temp,
                        on_step=scheduler.step,
                    )
                )
        model.eval()
        t_update = time.time() - t0
        _empty_cache(device)

        record: dict[str, Any] = {
            "step": step + 1,
            **{f"train/{k}": v for k, v in stats.items()},
            **{
                f"train/{k}": float(np.mean([asdict(u)[k] for u in updates]))
                for k in asdict(updates[0])
            },
            "train/kl_first": updates[0].kl,
            "train/clip_frac_last": updates[-1].clip_frac,
            "lr": scheduler.get_last_lr()[0],
            "time/rollout": t_rollout,
            "time/score": t_score,
            "time/update": t_update,
            "time/step": time.time() - t_step,
        }
        metrics.log(record)
        for i in range(min(cfg.train.log_samples, len(texts))):
            # One sample from each of the first few groups.
            j = i * cfg.rollout.group_size
            samples_log.log(
                {
                    "step": step + 1,
                    "question": problems[i].question,
                    "completion": texts[j],
                    "advantage": float(rollouts.advantages[j]),
                }
            )
        print(
            f"step {step + 1:4d} reward={stats['reward']:.3f} acc={stats['accuracy']:.3f} "
            f"fmt={stats['format_rate']:.2f} len={stats['length']:.0f} "
            f"kl={record['train/kl']:.4f} clip={record['train/clip_frac']:.3f} "
            f"t={record['time/step']:.0f}s",
            flush=True,
        )

        if (step + 1) % cfg.train.eval_every == 0 or step + 1 == cfg.train.steps:
            run_eval(step + 1)
        if (step + 1) % cfg.train.save_every == 0 or step + 1 == cfg.train.steps:
            ckpt = out_dir / "checkpoints" / "last"
            save_adapters(model, ckpt / "adapters.pt")
            torch.save(
                {
                    "step": step + 1,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                ckpt / "state.pt",
            )
