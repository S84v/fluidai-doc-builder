"""Tests for the restricted AST calculator tool."""

import pytest

from backend.app.agent.tools import (
    CalculationError,
    calculate,
    evaluate_expression,
)
from backend.app.agent.tools.calculator import (
    MAX_AST_NODES,
    MAX_EXPRESSION_LENGTH,
)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("10 / 4", 2.5),
        ("-5 + 2", -3),
        ("17 * 23", 391),
        ("1 + 2 + 3 + 4", 10),
    ],
)
def test_evaluate_expression_supports_basic_arithmetic(
    expression: str,
    expected: int | float,
) -> None:
    assert evaluate_expression(expression) == expected


def test_calculate_tool_returns_formatted_result() -> None:
    assert calculate.invoke({"expression": "17 * 23"}) == "391"
    assert calculate.invoke({"expression": "10 / 4"}) == "2.5"


def test_calculate_tool_returns_controlled_errors() -> None:
    assert calculate.invoke({"expression": "1 / 0"}) == (
        "Calculation error: division by zero"
    )
    assert calculate.invoke({"expression": "2 ** 3"}) == (
        "Calculation error: unsupported arithmetic operator"
    )


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "   ",
        "foo + 1",
        "__import__('os')",
        "open('file')",
        "'text'",
        "True + 1",
        "1j + 2",
        "1 if True else 2",
        "[1, 2]",
        "2 ** 3",
        "1 // 2",
        "1 % 2",
        "1 +",
    ],
)
def test_evaluate_expression_rejects_unsafe_or_invalid_input(
    expression: str,
) -> None:
    with pytest.raises(CalculationError):
        evaluate_expression(expression)


def test_evaluate_expression_rejects_division_by_zero() -> None:
    with pytest.raises(CalculationError, match="division by zero"):
        evaluate_expression("1 / (2 - 2)")


def test_evaluate_expression_enforces_length_limit() -> None:
    expression = "1+" * MAX_EXPRESSION_LENGTH
    with pytest.raises(CalculationError, match="too long"):
        evaluate_expression(expression)


def test_evaluate_expression_enforces_node_limit() -> None:
    expression = "1+" * (MAX_AST_NODES // 2) + "1"
    with pytest.raises(CalculationError, match="too complex"):
        evaluate_expression(expression)


def test_evaluate_expression_enforces_literal_limit() -> None:
    with pytest.raises(CalculationError, match="literal is too large"):
        evaluate_expression(str(10**13))


def test_evaluate_expression_enforces_result_limit() -> None:
    with pytest.raises(CalculationError, match="result is too large"):
        evaluate_expression("1000000000000 * 1000000000000")


def test_evaluate_expression_rejects_non_finite_literals() -> None:
    with pytest.raises(CalculationError, match="not finite"):
        evaluate_expression("1e309")
