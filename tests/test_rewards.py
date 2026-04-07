from fractions import Fraction

import pytest

from tiny_grpo.rewards import (
    InvalidExpressionError,
    arithmetic_reward,
    countdown_is_correct,
    countdown_reward,
    evaluate_countdown_expression,
    parse_completion,
    parse_integer,
)

# ---------------------------------------------------------------------------
# Format parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "think", "answer"),
    [
        ("<think>a</think><answer>1</answer>", "a", "1"),
        ("<think> a </think> <answer> 1 </answer>", "a", "1"),
        (
            "\n  <think>\nline one\nline two\n</think>\n<answer>\n42\n</answer>\n",
            "line one\nline two",
            "42",
        ),
        ("<think></think><answer></answer>", "", ""),
        ("<think>2 < 3 and 5 > 4</think><answer>yes</answer>", "2 < 3 and 5 > 4", "yes"),
    ],
)
def test_parse_valid(text: str, think: str, answer: str) -> None:
    parsed = parse_completion(text)
    assert parsed is not None
    assert parsed.think == think
    assert parsed.answer == answer


@pytest.mark.parametrize(
    "text",
    [
        "",
        "42",
        "<answer>42</answer>",  # no think block
        "<think>x</think>",  # no answer block
        "<think>x</think> 42",
        "<answer>42</answer><think>x</think>",  # wrong order
        "The answer is <think>x</think><answer>42</answer>",  # text before
        "<think>x</think><answer>42</answer> Hope that helps!",  # text after
        "<think>x</think> so <answer>42</answer>",  # text between
        "<think>x</think><answer>42</answer><answer>43</answer>",  # two answers
        "<think>x</think><think>y</think><answer>42</answer>",  # two thinks
        "<think>x <answer>1</answer></think><answer>42</answer>",  # nested tag
        "<think>x</think><answer>42",  # unclosed
        "<think>x<answer>42</answer>",  # unclosed think
        "<THINK>x</THINK><ANSWER>42</ANSWER>",  # tags are case sensitive
    ],
)
def test_parse_malformed(text: str) -> None:
    assert parse_completion(text) is None


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("42", 42),
        (" 42 ", 42),
        ("-7", -7),
        ("+7", 7),
        ("1,234", 1234),
        ("0", 0),
        ("1,234,567", 1234567),
    ],
)
def test_parse_integer_valid(text: str, value: int) -> None:
    assert parse_integer(text) == value


@pytest.mark.parametrize(
    "text", ["", "4 2", "42.0", "1e3", "forty-two", "12,34", "1,2345", "--4", "x = 4", "$42"]
)
def test_parse_integer_invalid(text: str) -> None:
    assert parse_integer(text) is None


# ---------------------------------------------------------------------------
# Arithmetic reward
# ---------------------------------------------------------------------------


def test_arithmetic_reward_levels() -> None:
    good = arithmetic_reward("<think>6*7</think><answer>42</answer>", 42)
    assert good.reward == pytest.approx(1.1)
    assert (good.correct, good.format_ok) == (True, True)

    wrong = arithmetic_reward("<think>6*7</think><answer>41</answer>", 42)
    assert wrong.reward == pytest.approx(0.1)
    assert (wrong.correct, wrong.format_ok) == (False, True)

    # Right number, wrong format: no credit at all.
    bare = arithmetic_reward("The answer is 42.", 42)
    assert (bare.reward, bare.correct, bare.format_ok) == (0.0, False, False)


def test_arithmetic_reward_format_weight() -> None:
    assert arithmetic_reward("<think></think><answer>1</answer>", 2, 0.5).reward == 0.5
    assert arithmetic_reward("<think></think><answer>2</answer>", 2, 0.0).reward == 1.0


def test_arithmetic_reward_unparseable_answer() -> None:
    r = arithmetic_reward("<think>x</think><answer>42 apples</answer>", 42)
    assert r.format_ok
    assert not r.correct


def test_arithmetic_negative_answer() -> None:
    assert arithmetic_reward("<think></think><answer>-15</answer>", -15).correct


# ---------------------------------------------------------------------------
# Countdown expression evaluation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr", "value", "used"),
    [
        ("1 + 2", Fraction(3), [1, 2]),
        ("(1 + 2) * 3", Fraction(9), [1, 2, 3]),
        ("7 / 2 * 2", Fraction(7), [7, 2, 2]),
        ("10 - 3 - 2", Fraction(5), [10, 3, 2]),
        ("10 - (3 - 2)", Fraction(9), [10, 3, 2]),
        ("8 / 3", Fraction(8, 3), [8, 3]),
        ("25", Fraction(25), [25]),
        ("6 × 4 ÷ 2", Fraction(12), [6, 4, 2]),  # noqa: RUF001
    ],
)
def test_evaluate_expression(expr: str, value: Fraction, used: list[int]) -> None:
    got_value, got_used = evaluate_countdown_expression(expr)
    assert got_value == value
    assert got_used == used


@pytest.mark.parametrize(
    "expr",
    [
        "",
        "2 ** 3",  # exponent
        "7 // 2",  # floor division
        "7 % 2",  # modulo
        "-3 + 5",  # unary minus
        "+3",  # unary plus
        "1.5 * 2",  # float literal
        "True + 1",  # bool is not an int here
        "x + 1",  # names
        "abs(3)",  # calls
        "__import__('os')",
        "1 + 2 = 3",  # equations
        "(1 + 2",  # syntax error
        "4 / (2 - 2)",  # division by zero
        "[1, 2]",
        "1 if 1 else 2",
        "1 << 2",
        "9" * 250,  # too long
    ],
)
def test_evaluate_expression_rejects(expr: str) -> None:
    with pytest.raises(InvalidExpressionError):
        evaluate_countdown_expression(expr)


# ---------------------------------------------------------------------------
# Countdown validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr", "numbers", "target", "ok"),
    [
        ("(25 - 5) * 3", [25, 5, 3], 60, True),
        ("25 * 3 - 5 * 3", [25, 5, 3], 60, False),  # 3 used twice
        ("25 * 3 - 15", [25, 5, 3], 60, False),  # 15 not available
        ("25 + 5", [25, 5, 3], 30, True),  # not every number has to be used
        ("5 * 5 + 5", [5, 5, 3], 30, False),  # only two 5s available
        ("5 * 5 + 5", [5, 5, 5], 30, True),  # duplicates are fine if given
        ("(25 - 5) * 3", [25, 5, 3], 61, False),  # wrong target
        ("(3 + 5) / 2", [3, 5, 2], 4, True),  # exact division
        ("7 / 2", [7, 2], 3, False),  # 3.5 is not 3
        ("(7 + 1) / 3 * 3", [7, 1, 3, 3], 8, True),  # fractional intermediate is fine
        ("2 ** 3", [2, 3], 8, False),  # disallowed operator
        ("25 - 5 * 3 =", [25, 5, 3], 10, False),
        ("25 - 5 * 3", [25, 5, 3], 10, True),
        ("nonsense", [1, 2], 3, False),
    ],
)
def test_countdown_is_correct(expr: str, numbers: list[int], target: int, ok: bool) -> None:
    assert countdown_is_correct(expr, numbers, target) is ok


def test_countdown_reward() -> None:
    nums, target = [4, 9, 2], 26
    r = countdown_reward(
        "<think>9*2=18, 18+4... no, 4*9-2... wait 9*2+4*2</think><answer>4 * 9 - 2 * 5</answer>",
        nums,
        target,
    )
    assert r.format_ok
    assert not r.correct  # 5 is not an available number
    r = countdown_reward(
        "<think>4*9=36, 36-2=34? no. 9*2=18, 18+4=22.</think><answer>(9 + 4) * 2</answer>",
        nums,
        target,
    )
    assert r.correct
    assert r.reward == pytest.approx(1.1)
    r = countdown_reward("(9 + 4) * 2", nums, target)
    assert (r.reward, r.format_ok, r.correct) == (0.0, False, False)
