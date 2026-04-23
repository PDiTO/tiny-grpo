import json
from itertools import pairwise
from pathlib import Path

import pytest

from tiny_grpo.cli import build_parser
from tiny_grpo.config import load_config
from tiny_grpo.plot import ema, plot_comparison, plot_metrics
from tiny_grpo.train import JsonlLogger, cosine_schedule, get_rng_state, set_rng_state

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
    bad.write_text("[rollout]\ntop_p = 0.9\n")
    with pytest.raises(ValueError, match="top_p"):
        load_config(bad)
    bad.write_text("[train]\neval_every = 0\n")
    with pytest.raises(ValueError, match="eval_every"):
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


def test_comparison_plot_writes_png(tmp_path: Path) -> None:
    paths = []
    for i in range(2):
        path = tmp_path / f"run{i}.jsonl"
        rows = [
            {"step": s, "train/kl": 0.01 * s * (i + 1), "train/reward": s / 10} for s in range(1, 6)
        ]
        rows.append({"step": 5, "eval/accuracy": 0.5 + i / 10})
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        paths.append(path)
    out = plot_comparison(paths, ["a", "b"], tmp_path / "cmp.png")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(ValueError, match="label"):
        plot_comparison(paths, ["a"], tmp_path / "x.png")


def test_cli_parses_subcommands() -> None:
    parser = build_parser()
    args = parser.parse_args(["train", "--config", "configs/arith.toml", "--steps", "3"])
    assert (args.command, args.steps) == ("train", 3)
    args = parser.parse_args(["eval", "--config", "c.toml", "--adapters", "a.pt"])
    assert (args.adapters, args.max_new_tokens) == ("a.pt", None)
    args = parser.parse_args(["plot", "--metrics", "m.jsonl"])
    assert args.out == "results/curves.png"
    args = parser.parse_args(["compare", "--metrics", "a.jsonl", "b.jsonl", "--labels", "a", "b"])
    assert args.metrics == ["a.jsonl", "b.jsonl"]


def test_jsonl_logger_fresh_run_truncates(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 1}\n{"step": 2}\n')
    log = JsonlLogger(path)
    log.log({"step": 1, "x": 1.0})
    assert [json.loads(line) for line in path.read_text().splitlines()] == [{"step": 1, "x": 1.0}]


def test_jsonl_logger_resume_drops_steps_after_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text("".join(json.dumps({"step": s}) + "\n" for s in (0, 1, 2, 3, 4)))
    log = JsonlLogger(path, keep_through_step=2)
    log.log({"step": 3})
    assert [json.loads(line)["step"] for line in path.read_text().splitlines()] == [0, 1, 2, 3]


def test_rng_state_round_trip() -> None:
    import random

    import torch

    device = torch.device("cpu")
    state = get_rng_state(device)
    first = (random.random(), torch.rand(3))
    set_rng_state(state, device)
    second = (random.random(), torch.rand(3))
    assert first[0] == second[0]
    assert torch.equal(first[1], second[1])
