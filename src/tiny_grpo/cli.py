"""``tiny-grpo train | eval | plot | compare``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _train(args: argparse.Namespace) -> None:
    from tiny_grpo.config import load_config
    from tiny_grpo.train import train

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg.train.steps = args.steps
    if args.output_dir is not None:
        cfg.train.output_dir = args.output_dir
    cfg.validate()
    train(cfg, resume=args.resume)


def _eval(args: argparse.Namespace) -> None:
    from tiny_grpo.config import load_config
    from tiny_grpo.evaluate import evaluate
    from tiny_grpo.lora import load_adapters
    from tiny_grpo.policy import load_policy, pick_device
    from tiny_grpo.tasks import eval_set, make_task

    cfg = load_config(args.config)
    device = pick_device(cfg.model.device)
    model, tokenizer = load_policy(
        cfg.model.name, device, cfg.model.dtype, cfg.lora.rank, cfg.lora.alpha, cfg.lora.targets
    )
    if args.adapters is not None:
        load_adapters(model, Path(args.adapters))
    task = make_task(cfg.task.name, cfg.task.format_weight, cfg.task.options)
    size = args.size if args.size is not None else cfg.task.eval_size
    problems = eval_set(task, size, cfg.task.eval_seed, cfg.task.train_seed)
    max_new_tokens = args.max_new_tokens or cfg.rollout.max_new_tokens
    result = evaluate(
        model, tokenizer, task, problems, max_new_tokens, cfg.rollout.generation_batch_size
    )
    print(json.dumps(result.metrics(prefix=""), indent=2))
    if args.samples_out is not None:
        out = Path(args.samples_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for s in result.samples:
                f.write(json.dumps(s) + "\n")


def _plot(args: argparse.Namespace) -> None:
    from tiny_grpo.plot import plot_metrics

    out = plot_metrics(args.metrics, args.out, title=args.title)
    print(f"wrote {out}")


def _compare(args: argparse.Namespace) -> None:
    from tiny_grpo.plot import plot_comparison

    labels = args.labels if args.labels is not None else [str(p) for p in args.metrics]
    out = plot_comparison(args.metrics, labels, args.out, title=args.title)
    print(f"wrote {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tiny-grpo", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="run GRPO training from a TOML config")
    p.add_argument("--config", required=True)
    p.add_argument("--steps", type=int, default=None, help="override train.steps")
    p.add_argument("--output-dir", default=None, help="override train.output_dir")
    p.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    p.set_defaults(func=_train)

    p = sub.add_parser("eval", help="greedy pass@1 and mean reward on the held-out set")
    p.add_argument("--config", required=True)
    p.add_argument("--adapters", default=None, help="adapters.pt; omit for the base model")
    p.add_argument("--size", type=int, default=None, help="override task.eval_size")
    p.add_argument(
        "--max-new-tokens", type=int, default=None, help="override rollout.max_new_tokens"
    )
    p.add_argument("--samples-out", default=None, help="write every completion to this JSONL")
    p.set_defaults(func=_eval)

    p = sub.add_parser("plot", help="plot a metrics.jsonl to PNG")
    p.add_argument("--metrics", required=True)
    p.add_argument("--out", default="results/curves.png")
    p.add_argument("--title", default=None)
    p.set_defaults(func=_plot)

    p = sub.add_parser("compare", help="overlay KL, reward and eval accuracy from several runs")
    p.add_argument("--metrics", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--out", default="results/compare.png")
    p.add_argument("--title", default=None)
    p.set_defaults(func=_compare)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
