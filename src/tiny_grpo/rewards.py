"""Pure reward functions.

Every completion is expected to look like::

    <think> ...free-form reasoning... </think>
    <answer> ...final answer... </answer>

and nothing else (surrounding whitespace is fine). The format check is strict on
purpose: a small model will happily emit several answer blocks, or keep talking
after the answer, and I don't want either of those to be rewarded.

Rewards are ``correctness + format_weight * format_ok``. Correctness is only
awarded when the format is valid, because the answer is extracted from the
``<answer>`` block. All functions here are pure: no model, no randomness, no I/O.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

_TAG = re.compile(r"</?(?:think|answer)>")
_FORMAT = re.compile(
    r"\A\s*<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*\Z",
    re.DOTALL,
)
_INTEGER = re.compile(r"\A[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)\Z")


@dataclass(frozen=True)
class ParsedCompletion:
    think: str
    answer: str


def parse_completion(text: str) -> ParsedCompletion | None:
    """Return the think and answer blocks, or ``None`` if the format is wrong.

    Valid means exactly one ``<think>`` block followed by exactly one ``<answer>``
    block, with only whitespace around and between them.
    """
    match = _FORMAT.match(text)
    if match is None:
        return None
    # The regex is non-greedy, so "<think>a</think><answer>1</answer><answer>2</answer>"
    # would not match, but nested or repeated tags inside a block would. Reject them.
    tags = _TAG.findall(text)
    if tags != ["<think>", "</think>", "<answer>", "</answer>"]:
        return None
    return ParsedCompletion(think=match["think"].strip(), answer=match["answer"].strip())


def parse_integer(text: str) -> int | None:
    """Parse a plain integer answer. Allows a sign and thousands separators."""
    text = text.strip()
    if not _INTEGER.match(text):
        return None
    return int(text.replace(",", ""))


# ---------------------------------------------------------------------------
# Countdown: evaluate an arithmetic expression safely and exactly.
# ---------------------------------------------------------------------------

_BINOPS: dict[type[ast.operator], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
}

_MAX_EXPR_CHARS = 200


class InvalidExpressionError(ValueError):
    """Raised when a Countdown answer is not an allowed expression."""


def _eval_node(node: ast.expr, used: list[int]) -> Fraction:
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOPS:
            raise InvalidExpressionError(f"operator {type(node.op).__name__} is not allowed")
        left = _eval_node(node.left, used)
        right = _eval_node(node.right, used)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise InvalidExpressionError("division by zero")
        return left / right
    if isinstance(node, ast.Constant) and type(node.value) is int:
        used.append(node.value)
        return Fraction(node.value)
    raise InvalidExpressionError(f"{type(node).__name__} is not allowed")


def evaluate_countdown_expression(expr: str) -> tuple[Fraction, list[int]]:
    """Evaluate ``expr`` exactly, returning its value and the integer literals used.

    Only integer literals, the four binary operators ``+ - * /`` and parentheses
    are allowed. No unary minus, no exponent, no names, no floats. Division is
    exact (rational), so ``7 / 2 * 2`` is 7, not 6 or 7.000000001.
    """
    expr = expr.strip()
    if not expr or len(expr) > _MAX_EXPR_CHARS:
        raise InvalidExpressionError("empty or overly long expression")
    # Accept the common unicode operator glyphs the model might produce.
    expr = expr.replace("×", "*").replace("÷", "/").replace("−", "-")  # noqa: RUF001
    if "=" in expr:
        raise InvalidExpressionError("the answer must be an expression, not an equation")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise InvalidExpressionError("not a valid expression") from exc
    used: list[int] = []
    value = _eval_node(tree.body, used)
    return value, used


def countdown_is_correct(expr: str, numbers: list[int], target: int) -> bool:
    """True if ``expr`` hits ``target`` using each of ``numbers`` at most once."""
    try:
        value, used = evaluate_countdown_expression(expr)
    except InvalidExpressionError:
        return False
    available = Counter(numbers)
    if any(count > available[n] for n, count in Counter(used).items()):
        return False
    return value == target


# ---------------------------------------------------------------------------
# Task rewards.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardBreakdown:
    reward: float
    format_ok: bool
    correct: bool


def _combine(format_ok: bool, correct: bool, format_weight: float) -> RewardBreakdown:
    reward = float(correct) + format_weight * float(format_ok)
    return RewardBreakdown(reward=reward, format_ok=format_ok, correct=correct)


def arithmetic_reward(completion: str, answer: int, format_weight: float = 0.1) -> RewardBreakdown:
    parsed = parse_completion(completion)
    if parsed is None:
        return _combine(False, False, format_weight)
    return _combine(True, parse_integer(parsed.answer) == answer, format_weight)


def countdown_reward(
    completion: str, numbers: list[int], target: int, format_weight: float = 0.1
) -> RewardBreakdown:
    parsed = parse_completion(completion)
    if parsed is None:
        return _combine(False, False, format_weight)
    return _combine(True, countdown_is_correct(parsed.answer, numbers, target), format_weight)
