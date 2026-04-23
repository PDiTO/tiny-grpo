"""Plot training curves from a metrics JSONL file to a PNG."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
TRAIN = "#2a78d6"  # categorical slot 1
EVAL = "#eb6834"  # categorical slot 2
EVAL_LENIENT = "#1baf7a"  # categorical slot 3

# (title, train key, eval key, y label)
PANELS: list[tuple[str, str | None, str | None, str]] = [
    ("Reward", "train/reward", "eval/reward", "mean reward"),
    ("Accuracy", "train/accuracy", "eval/accuracy", "fraction correct"),
    ("Format", "train/format_rate", "eval/format_rate", "fraction well formatted"),
    ("KL to reference (k3)", "train/kl", None, "per-token KL"),
    ("Response length", "train/length", "eval/length", "tokens"),
    ("Clipped tokens", "train/clip_frac", None, "fraction"),
]


def load_metrics(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def series(records: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    pts = [(r["step"], r[key]) for r in records if key in r]
    if not pts:
        return np.array([]), np.array([])
    steps, values = zip(*pts, strict=True)
    return np.asarray(steps, dtype=float), np.asarray(values, dtype=float)


def ema(values: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """Exponential moving average, for a readable trend over noisy per-step values."""
    out = np.empty_like(values)
    acc = values[0] if len(values) else 0.0
    for i, v in enumerate(values):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def plot_metrics(metrics_path: str | Path, out_path: str | Path, title: str | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = load_metrics(metrics_path)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), facecolor=SURFACE)
    for ax, (name, train_key, eval_key, ylabel) in zip(axes.flat, PANELS, strict=True):
        ax.set_facecolor(SURFACE)
        handles = []
        if train_key is not None:
            x, y = series(records, train_key)
            if len(x):
                ax.plot(x, y, color=TRAIN, alpha=0.25, linewidth=1)
                (h,) = ax.plot(x, ema(y), color=TRAIN, linewidth=2, label="train (EMA)")
                handles.append(h)
        if eval_key is not None:
            x, y = series(records, eval_key)
            if len(x):
                (h,) = ax.plot(
                    x,
                    y,
                    color=EVAL,
                    linewidth=2,
                    marker="o",
                    markersize=5,
                    markeredgecolor=SURFACE,
                    markeredgewidth=1.5,
                    label="held-out eval (greedy)",
                )
                handles.append(h)
        if name == "Accuracy":
            x, y = series(records, "eval/lenient_accuracy")
            if len(x):
                (h,) = ax.plot(
                    x,
                    y,
                    color=EVAL_LENIENT,
                    linewidth=2,
                    linestyle="--",
                    marker="s",
                    markersize=5,
                    markeredgecolor=SURFACE,
                    markeredgewidth=1.5,
                    label="held-out, format ignored",
                )
                handles.append(h)
        ax.set_title(name, color=TEXT, fontsize=11, loc="left")
        ax.set_xlabel("step", color=TEXT_MUTED, fontsize=9)
        ax.set_ylabel(ylabel, color=TEXT_MUTED, fontsize=9)
        ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        if len(handles) > 1:
            ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=TEXT_MUTED)
    if title:
        fig.suptitle(title, color=TEXT, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return out


# Panels for comparing runs: (title, key, y label, smooth).
COMPARE_PANELS: list[tuple[str, str, str, bool]] = [
    ("KL to reference (k3)", "train/kl", "per-token KL", True),
    ("Train reward", "train/reward", "mean reward", True),
    ("Held-out accuracy (greedy)", "eval/accuracy", "fraction correct", False),
]
RUN_COLORS = [TRAIN, EVAL, EVAL_LENIENT]


def plot_comparison(
    metrics_paths: Sequence[str | Path],
    labels: Sequence[str],
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """Overlay a few key metrics from up to three runs."""
    if len(metrics_paths) != len(labels):
        raise ValueError("need one label per metrics file")
    if not 1 <= len(metrics_paths) <= len(RUN_COLORS):
        raise ValueError(f"can compare 1 to {len(RUN_COLORS)} runs")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [load_metrics(p) for p in metrics_paths]
    fig, axes = plt.subplots(1, len(COMPARE_PANELS), figsize=(13, 3.8), facecolor=SURFACE)
    for ax, (name, key, ylabel, smooth) in zip(axes.flat, COMPARE_PANELS, strict=True):
        ax.set_facecolor(SURFACE)
        for records, label, color in zip(runs, labels, RUN_COLORS, strict=False):
            x, y = series(records, key)
            if not len(x):
                continue
            if smooth:
                ax.plot(x, y, color=color, alpha=0.25, linewidth=1)
                ax.plot(x, ema(y), color=color, linewidth=2, label=label)
            else:
                ax.plot(
                    x,
                    y,
                    color=color,
                    linewidth=2,
                    marker="o",
                    markersize=5,
                    markeredgecolor=SURFACE,
                    markeredgewidth=1.5,
                    label=label,
                )
        ax.set_title(name, color=TEXT, fontsize=11, loc="left")
        ax.set_xlabel("step", color=TEXT_MUTED, fontsize=9)
        ax.set_ylabel(ylabel, color=TEXT_MUTED, fontsize=9)
        ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        if len(runs) > 1:
            ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_MUTED)
    if title:
        fig.suptitle(title, color=TEXT, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return out
