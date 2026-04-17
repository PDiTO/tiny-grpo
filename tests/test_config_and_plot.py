import json
from itertools import pairwise
from pathlib import Path

import pytest

from tiny_grpo.cli import build_parser
from tiny_grpo.config import load_config
from tiny_grpo.plot import ema, plot_metrics
from tiny_grpo.train import cosine_schedule

CONFIGS = sorted((Path(__file__).parent.parent / "configs").glob("*.toml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_load(path: Path) -> None:
    cfg = load_config(path)
    assert cfg.rollout.group_size >= 2


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[optim]\nlearning_rate = 1e-4\n")
    with pytest.raises(ValueError, match="learning_rate"):
        load_config(bad)
    bad.write_text("[optimiser]\nlr = 1e-4\n")
    with pytest.raises(ValueError, match="optimiser"):
        load_config(bad)


def test_invalid_values_are_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('[loss]\naggregation = "mean"\n')
    with pytest.raises(ValueError, match="aggregation"):
        load_config(bad)
    bad.write_text("[rollout]\ngroup_size = 1\n")
    with pytest.raises(ValueError, match="group_size"):
        load_config(bad)


def test_cosine_schedule_shape() -> None:
    lrs = [cosine_schedule(s, total=100, warmup=10, min_ratio=0.1) for s in range(100)]
    assert lrs[0] == pytest.approx(0.1)
    assert lrs[9] == pytest.approx(1.0)
    assert lrs[10] == pytest.approx(1.0)
    assert all(a >= b for a, b in pairwise(lrs[10:]))
    assert lrs[-1] == pytest.approx(0.1, abs=1e-3)


def test_ema_smooths_towards_values() -> None:
    import numpy as np

    out = ema(np.array([0.0, 1.0, 1.0, 1.0]), alpha=0.5)
    assert out.tolist() == [0.0, 0.5, 0.75, 0.875]


def test_plot_writes_png(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    rows = [{"step": 0, "eval/accuracy": 0.1, "eval/reward": 0.2}]
    rows += [
        {"step": s, "train/reward": s / 10, "train/kl": 0.001 * s, "train/length": 100 - s}
        for s in range(1, 11)
    ]
    rows.append({"step": 10, "eval/accuracy": 0.3, "eval/reward": 0.5})
    metrics.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = plot_metrics(metrics, tmp_path / "curves.png", title="test")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_parses_subcommands() -> None:
    parser = build_parser()
    args = parser.parse_args(["train", "--config", "configs/arith.toml", "--steps", "3"])
    assert (args.command, args.steps) == ("train", 3)
    args = parser.parse_args(["eval", "--config", "c.toml", "--adapters", "a.pt"])
    assert (args.adapters, args.max_new_tokens) == ("a.pt", None)
    args = parser.parse_args(["plot", "--metrics", "m.jsonl"])
    assert args.out == "results/curves.png"
