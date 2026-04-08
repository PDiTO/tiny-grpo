import random
import re
from itertools import islice

import pytest

from tiny_grpo.rewards import countdown_is_correct
from tiny_grpo.tasks import (
    ArithmeticTask,
    CountdownTask,
    chat_messages,
    eval_set,
    make_task,
    train_stream,
)


def test_arithmetic_problems_have_correct_answers() -> None:
    task = ArithmeticTask()
    rng = random.Random(0)
    for _ in range(500):
        p = task.sample(rng)
        m = re.match(r"What is (\d+) ([-+*]) (\d+)\?", p.question)
        assert m is not None
        a, op, b = int(m[1]), m[2], int(m[3])
        expected = {"+": a + b, "-": a - b, "*": a * b}[op]
        assert p.meta["answer"] == expected


def test_arithmetic_reward_uses_problem_answer() -> None:
    task = ArithmeticTask()
    p = task.sample(random.Random(3))
    answer = p.meta["answer"]
    assert task.reward(f"<think>x</think><answer>{answer}</answer>", p).correct
    assert not task.reward(f"<think>x</think><answer>{answer}1</answer>", p).correct


def test_countdown_problems_are_solvable_by_construction() -> None:
    task = CountdownTask()
    rng = random.Random(0)
    for _ in range(300):
        p = task.sample(rng)
        numbers, target = p.meta["numbers"], p.meta["target"]
        assert isinstance(numbers, list)
        assert isinstance(target, int)
        assert 3 <= len(numbers) <= 4
        assert all(1 <= n <= 25 for n in numbers)
        assert 10 <= target <= 200
        assert target not in numbers
        assert f"equals {target}" in p.question


def _brute_force_solvable(numbers: list[int], target: int) -> bool:
    """Exhaustive search over left-to-right chains of a subset of the numbers,
    the same family of expressions the generator builds."""
    from itertools import permutations, product

    for k in range(2, len(numbers) + 1):
        for perm in permutations(numbers, k):
            for ops in product("+-*/", repeat=k - 1):
                expr = str(perm[0])
                for op, n in zip(ops, perm[1:], strict=True):
                    expr = f"({expr} {op} {n})"
                if countdown_is_correct(expr, numbers, target):
                    return True
    return False


def test_countdown_problems_solvable_by_brute_force() -> None:
    task = CountdownTask()
    rng = random.Random(1)
    for _ in range(40):
        p = task.sample(rng)
        numbers, target = p.meta["numbers"], p.meta["target"]
        assert isinstance(numbers, list)
        assert isinstance(target, int)
        assert _brute_force_solvable(numbers, target)


def test_train_stream_is_reproducible() -> None:
    task = make_task("arithmetic")
    a = [p.key for p in islice(train_stream(task, 7), 50)]
    b = [p.key for p in islice(train_stream(task, 7), 50)]
    c = [p.key for p in islice(train_stream(task, 8), 50)]
    assert a == b
    assert a != c


@pytest.mark.parametrize("name", ["arithmetic", "countdown"])
def test_eval_set_is_disjoint_from_training_prefix(name: str) -> None:
    task = make_task(name)
    held_out = eval_set(task, 100, seed=1, train_seed=0, dedupe_window=5000)
    train_keys = {p.key for p in islice(train_stream(task, 0), 5000)}
    keys = [p.key for p in held_out]
    assert len(keys) == 100
    assert len(set(keys)) == 100
    assert not set(keys) & train_keys


def test_make_task_unknown() -> None:
    with pytest.raises(ValueError, match="unknown task"):
        make_task("chess")


def test_chat_messages_carry_format_instruction() -> None:
    messages = chat_messages("What is 1 + 1?")
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "<answer>" in messages[0]["content"]
    assert messages[0]["content"].startswith("What is 1 + 1?")


def test_lenient_correct_ignores_format() -> None:
    task = ArithmeticTask()
    p = task.sample(random.Random(5))
    answer = p.meta["answer"]
    assert task.lenient_correct(f"So the result is {answer}.", p)
    assert not task.reward(f"So the result is {answer}.", p).correct
