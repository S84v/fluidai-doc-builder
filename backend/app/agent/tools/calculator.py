"""Safe, deterministic calculator tool based on a restricted AST evaluator."""

import ast
import math

from langchain_core.tools import tool

MAX_EXPRESSION_LENGTH = 200
MAX_AST_NODES = 100
MAX_ABS_LITERAL = 10**12
MAX_ABS_RESULT = 10**15


class CalculationError(ValueError):
    """Raised when an expression is invalid, unsupported, or unsafe."""


def _bounded_result(value: int | float) -> int | float:
    if isinstance(value, float) and not math.isfinite(value):
        raise CalculationError("result is not a finite number")
    if abs(value) > MAX_ABS_RESULT:
        raise CalculationError("result is too large")
    return value


def _evaluate_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CalculationError("only numeric literals are supported")
        if isinstance(value, float) and not math.isfinite(value):
            raise CalculationError("numeric literal is not finite")
        if abs(value) > MAX_ABS_LITERAL:
            raise CalculationError("numeric literal is too large")
        return value

    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return _bounded_result(operand)
        if isinstance(node.op, ast.USub):
            return _bounded_result(-operand)
        raise CalculationError("unsupported unary operator")

    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise CalculationError("division by zero")
            result = left / right
        else:
            raise CalculationError("unsupported arithmetic operator")
        return _bounded_result(result)

    raise CalculationError("unsupported expression")


def evaluate_expression(expression: str) -> int | float:
    """Evaluate basic arithmetic using only a restricted subset of Python AST."""
    if not isinstance(expression, str):
        raise CalculationError("expression must be a string")

    text = expression.strip()
    if not text:
        raise CalculationError("expression must not be empty")
    if len(text) > MAX_EXPRESSION_LENGTH:
        raise CalculationError("expression is too long")

    try:
        parsed = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise CalculationError("invalid arithmetic expression") from exc

    if len(list(ast.walk(parsed))) > MAX_AST_NODES:
        raise CalculationError("expression is too complex")

    return _bounded_result(_evaluate_node(parsed.body))


def _format_result(value: int | float) -> str:
    if isinstance(value, float):
        if value == 0:
            return "0"
        if value.is_integer():
            return str(int(value))
    return str(value)


@tool
def calculate(expression: str) -> str:
    """Calculate a basic arithmetic expression using +, -, *, / and parentheses."""
    try:
        return _format_result(evaluate_expression(expression))
    except CalculationError as exc:
        return f"Calculation error: {exc}"
