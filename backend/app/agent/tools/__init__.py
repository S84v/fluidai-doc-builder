"""Tools available to the document agent."""

from backend.app.agent.tools.calculator import (
    CalculationError,
    calculate,
    evaluate_expression,
)

AGENT_TOOLS = [calculate]


def handle_tool_error(exc: Exception) -> str:
    """Convert unexpected tool failures into a controlled tool message."""
    if isinstance(exc, CalculationError):
        return f"Calculation error: {exc}"
    return "Calculator tool failed: invalid arguments."


__all__ = [
    "AGENT_TOOLS",
    "CalculationError",
    "calculate",
    "evaluate_expression",
    "handle_tool_error",
]
