"""A minimal LoRA implementation.

I wrote this instead of pulling in ``peft`` for two reasons. It is about sixty
lines, and the one feature GRPO needs beyond plain LoRA, switching the adapters
off to get the frozen reference policy from the same weights, is a single flag
here rather than a context manager whose behaviour I'd have to go and read.

``W x + (alpha / r) * B A x`` with ``B`` initialised to zero, so the adapted
model starts out identical to the base model and the KL to the reference is
exactly zero at step 0.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import Tensor, nn

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.enabled = True
        # Adapters are always fp32, whatever the base weights are, so small
        # updates are not lost to bf16 rounding.
        device = base.weight.device
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.base(x)
        if not self.enabled:
            return out
        delta = (x.to(self.lora_a.dtype) @ self.lora_a.T) @ self.lora_b.T
        return out + (self.scaling * delta).to(out.dtype)


def apply_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    targets: Iterable[str] = DEFAULT_TARGETS,
) -> list[LoRALinear]:
    """Freeze ``model`` and wrap every ``nn.Linear`` whose attribute name is in
    ``targets`` with a :class:`LoRALinear`. Returns the new modules."""
    targets = set(targets)
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped: list[LoRALinear] = []
    # Materialise the list first; we mutate the module tree while iterating.
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                lora = LoRALinear(child, rank, alpha)
                setattr(parent, name, lora)
                wrapped.append(lora)
    if not wrapped:
        raise ValueError(f"no nn.Linear modules named any of {sorted(targets)}")
    return wrapped


def lora_modules(model: nn.Module) -> Iterator[LoRALinear]:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            yield m


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for m in lora_modules(model) for p in (m.lora_a, m.lora_b)]


@contextmanager
def adapters_disabled(model: nn.Module) -> Iterator[None]:
    """Run ``model`` as the frozen base model, i.e. the GRPO reference policy."""
    modules = list(lora_modules(model))
    previous = [m.enabled for m in modules]
    for m in modules:
        m.enabled = False
    try:
        yield
    finally:
        for m, was in zip(modules, previous, strict=True):
            m.enabled = was


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Only the adapter weights, keyed by their module path."""
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if name.endswith((".lora_a", ".lora_b"))
    }


def save_adapters(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), path)


def load_adapters(model: nn.Module, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    params = dict(model.named_parameters())
    missing = [k for k in params if k.endswith((".lora_a", ".lora_b")) and k not in state]
    unexpected = [k for k in state if k not in params]
    if missing or unexpected:
        raise KeyError(f"adapter mismatch: missing={missing[:3]} unexpected={unexpected[:3]}")
    with torch.no_grad():
        for k, v in state.items():
            params[k].copy_(v)
