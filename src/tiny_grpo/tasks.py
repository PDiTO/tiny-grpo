"""Procedurally generated tasks with programmatic rewards.

Each task produces :class:`Problem` objects from a seeded RNG, so train and eval
sets are reproducible without shipping any data. The eval set is drawn from a
separate seed and any problem that also appears in the train stream's first
``dedupe_window`` items is dropped.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field, fields
from fractions import Fraction
from typing import Any, Protocol

from tiny_grpo.rewards import (
    RewardBreakdown,
    arithmetic_reward,
    countdown_reward,
    lenient_countdown,
    lenient_integer,
)


def format_instruction(answer_example: str) -> str:
    return (
        "Think step by step inside <think> </think> tags, then give only the final answer "
        "inside <answer> </answer> tags, like this: <think> your reasoning </think> "
        f"<answer> {answer_example} </answer>"
    )


def chat_messages(question: str, answer_example: str = "42") -> list[dict[str, str]]:
    """The chat turns for one problem.

    The format instruction goes in the user turn, not a system prompt. On
    Qwen2.5-0.5B-Instruct I measured roughly 12% well-formatted samples with it
    in the user turn and under 2% with it in the system prompt, and GRPO can only
    reinforce a format it occasionally produces.

    ``answer_example`` must look like a real answer for the task. My first
    Countdown run used "42" here, and the model learned to put the target
    number in the answer block instead of an expression.
    """
    instruction = format_instruction(answer_example)
    return [{"role": "user", "content": f"{question}\n\n{instruction}"}]


@dataclass(frozen=True)
class Problem:
    """One prompt plus everything the reward function needs to score it."""

    question: str
    # Task-specific ground truth, e.g. {"answer": 42} or {"numbers": [...], "target": 7}.
    meta: dict[str, int | list[int]] = field(hash=False)
    key: str  # canonical identity used for de-duplication


class Task(Protocol):
    name: str
    answer_example: str  # shown in the format instruction

    def sample(self, rng: random.Random) -> Problem: ...

    def reward(self, completion: str, problem: Problem) -> RewardBreakdown: ...

    def lenient_correct(self, completion: str, problem: Problem) -> bool:
        """Format-agnostic correctness, reported by eval but never rewarded."""
        ...


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


@dataclass
class ArithmeticTask:
    """Integer arithmetic: addition and subtraction of 3-4 digit numbers, and
    multiplication of a 2-3 digit number by a 2 digit number.

    Small instruct models get the additions mostly right and the products mostly
    wrong, so there is room to learn without the reward being all zeros.
    """

    name: str = "arithmetic"
    answer_example: str = "42"
    format_weight: float = 0.1
    add_digits: tuple[int, int] = (3, 4)
    mul_left_digits: tuple[int, int] = (2, 3)
    mul_right_digits: tuple[int, int] = (2, 2)

    @staticmethod
    def _number(rng: random.Random, digits: tuple[int, int]) -> int:
        n = rng.randint(*digits)
        return rng.randint(10 ** (n - 1), 10**n - 1)

    def sample(self, rng: random.Random) -> Problem:
        op = rng.choice(["+", "-", "*"])
        if op == "*":
            a = self._number(rng, self.mul_left_digits)
            b = self._number(rng, self.mul_right_digits)
            answer = a * b
        else:
            a = self._number(rng, self.add_digits)
            b = self._number(rng, self.add_digits)
            answer = a + b if op == "+" else a - b
        question = f"What is {a} {op} {b}? Give the answer as a single integer."
        return Problem(question=question, meta={"answer": answer}, key=f"{a}{op}{b}")

    def reward(self, completion: str, problem: Problem) -> RewardBreakdown:
        answer = problem.meta["answer"]
        assert isinstance(answer, int)
        return arithmetic_reward(completion, answer, self.format_weight)

    def lenient_correct(self, completion: str, problem: Problem) -> bool:
        return lenient_integer(completion) == problem.meta["answer"]


# ---------------------------------------------------------------------------
# Countdown
# ---------------------------------------------------------------------------


@dataclass
class CountdownTask:
    """The Countdown numbers game: combine the given numbers with + - * / to hit
    the target, using each number at most once.

    Problems are built by applying random operations to a random subset of the
    numbers, so every problem has at least one solution. Intermediate results
    must stay positive integers, the same rule as the TV show.
    """

    name: str = "countdown"
    answer_example: str = "(7 - 3) * 5"
    format_weight: float = 0.1
    num_numbers: tuple[int, int] = (3, 4)
    number_range: tuple[int, int] = (1, 25)
    target_range: tuple[int, int] = (10, 200)
    max_tries: int = 1000

    def _target(self, rng: random.Random, numbers: list[int]) -> int | None:
        k = rng.randint(2, len(numbers))
        pool = rng.sample(numbers, k)
        value = Fraction(pool[0])
        for n in pool[1:]:
            op = rng.choice("+-*/")
            if op == "+":
                value += n
            elif op == "-":
                value -= n
            elif op == "*":
                value *= n
            else:
                value /= n
            if value.denominator != 1 or value <= 0:
                return None
        target = int(value)
        lo, hi = self.target_range
        if not lo <= target <= hi or target in numbers:
            return None
        return target

    def sample(self, rng: random.Random) -> Problem:
        for _ in range(self.max_tries):
            count = rng.randint(*self.num_numbers)
            numbers = [rng.randint(*self.number_range) for _ in range(count)]
            target = self._target(rng, numbers)
            if target is not None:
                listed = ", ".join(str(n) for n in numbers)
                question = (
                    f"Using the numbers [{listed}], write an arithmetic expression that equals "
                    f"{target}. You can use + - * / and parentheses, and each number at most "
                    "once. The answer must be the expression itself, not its result."
                )
                key = f"{sorted(numbers)}->{target}"
                return Problem(
                    question=question, meta={"numbers": numbers, "target": target}, key=key
                )
        raise RuntimeError("could not generate a Countdown problem; check the ranges")

    def reward(self, completion: str, problem: Problem) -> RewardBreakdown:
        numbers, target = problem.meta["numbers"], problem.meta["target"]
        assert isinstance(numbers, list)
        assert isinstance(target, int)
        return countdown_reward(completion, numbers, target, self.format_weight)

    def lenient_correct(self, completion: str, problem: Problem) -> bool:
        numbers, target = problem.meta["numbers"], problem.meta["target"]
        assert isinstance(numbers, list)
        assert isinstance(target, int)
        return lenient_countdown(completion, numbers, target)


TASKS: dict[str, type[ArithmeticTask] | type[CountdownTask]] = {
    "arithmetic": ArithmeticTask,
    "countdown": CountdownTask,
}


def make_task(name: str, format_weight: float = 0.1, options: dict[str, Any] | None = None) -> Task:
    """Build a task by name. ``options`` sets task-specific fields, e.g.
    ``{"num_numbers": [3, 3]}`` for Countdown."""
    try:
        cls = TASKS[name]
    except KeyError:
        raise ValueError(f"unknown task {name!r}; choose from {sorted(TASKS)}") from None
    known = {f.name for f in fields(cls)} - {"name", "answer_example", "format_weight"}
    options = dict(options or {})
    unknown = set(options) - known
    if unknown:
        raise ValueError(f"unknown option(s) for {name}: {sorted(unknown)}")
    # TOML has no tuples; the dataclasses use them for (min, max) ranges.
    kwargs: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in options.items()}
    return cls(format_weight=format_weight, **kwargs)


def train_stream(task: Task, seed: int) -> Iterator[Problem]:
    """Endless stream of training problems."""
    rng = random.Random(seed)
    while True:
        yield task.sample(rng)


def eval_set(
    task: Task, size: int, seed: int, train_seed: int, dedupe_window: int = 50_000
) -> list[Problem]:
    """A fixed held-out set that does not overlap the first ``dedupe_window``
    training problems."""
    train_rng = random.Random(train_seed)
    seen = {task.sample(train_rng).key for _ in range(dedupe_window)}
    rng = random.Random(seed)
    problems: list[Problem] = []
    keys: set[str] = set()
    while len(problems) < size:
        p = task.sample(rng)
        if p.key in seen or p.key in keys:
            continue
        keys.add(p.key)
        problems.append(p)
    return problems
