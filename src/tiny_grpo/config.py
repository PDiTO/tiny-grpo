"""Run configuration, loaded from TOML.

Each TOML table maps to one dataclass below. Unknown keys are an error, so a typo
in a config file fails loudly instead of silently using the default.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

from tiny_grpo.grpo import LossAggregation


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    device: str = "auto"  # "auto" picks mps, then cuda, then cpu
    # Recompute each decoder layer's activations in the backward pass instead of
    # storing them. Roughly halves update memory for about a third more compute.
    gradient_checkpointing: bool = True


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    targets: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class TaskConfig:
    name: str = "arithmetic"
    format_weight: float = 0.1
    train_seed: int = 0
    eval_seed: int = 1
    eval_size: int = 200


@dataclass
class RolloutConfig:
    prompts_per_step: int = 16
    group_size: int = 8
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    generation_batch_size: int = 64


@dataclass
class LossSection:
    clip_low: float = 0.2
    clip_high: float = 0.2
    kl_coef: float = 0.04
    aggregation: LossAggregation = "token"
    scale_advantages: bool = True  # False gives the Dr. GRPO advantage


@dataclass
class OptimConfig:
    lr: float = 5e-5
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.99)
    warmup_steps: int = 5
    min_lr_ratio: float = 0.1
    max_grad_norm: float = 1.0
    epochs_per_batch: int = 2  # mu in the GRPO paper
    minibatches_per_epoch: int = 1
    micro_batch_size: int = 8


@dataclass
class TrainConfig:
    steps: int = 100
    seed: int = 0
    eval_every: int = 10
    save_every: int = 20
    output_dir: str = "runs/arith"
    log_samples: int = 4  # completions written to samples.jsonl each step


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    loss: LossSection = field(default_factory=LossSection)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        r, o = self.rollout, self.optim
        batch = r.prompts_per_step * r.group_size
        if r.group_size < 2:
            raise ValueError("rollout.group_size must be at least 2")
        if batch % o.minibatches_per_epoch != 0:
            raise ValueError("prompts_per_step * group_size must divide into minibatches")
        if self.loss.aggregation not in ("sequence", "token", "constant"):
            raise ValueError(f"unknown loss.aggregation {self.loss.aggregation!r}")
        if not 0.0 <= self.loss.clip_low < 1.0 or self.loss.clip_high < 0.0:
            raise ValueError("clip range must satisfy 0 <= clip_low < 1 and clip_high >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build[T](cls: type[T], table: dict[str, Any], section: str) -> T:
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(table) - known
    if unknown:
        raise ValueError(f"unknown key(s) in [{section}]: {sorted(unknown)}")
    kwargs = dict(table)
    if "betas" in kwargs:
        kwargs["betas"] = tuple(kwargs["betas"])
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    with Path(path).open("rb") as f:
        raw = tomllib.load(f)
    sections = [f.name for f in fields(Config)]
    unknown = set(raw) - set(sections)
    if unknown:
        raise ValueError(f"unknown section(s): {sorted(unknown)}")
    defaults = Config()
    built = {
        name: _build(type(getattr(defaults, name)), raw.get(name, {}), name) for name in sections
    }
    cfg = Config(**built)
    cfg.validate()
    return cfg
