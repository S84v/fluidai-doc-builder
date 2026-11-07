"""Tests for model-driven tool binding and selection."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from backend.app.agent import (
    DocumentPlan,
    DocumentType,
    PlanSection,
    ToolSelectionError,
    ToolSelector,
    build_tool_selector,
)
from backend.app.agent.tool_selection import build_tool_selection_messages
from backend.tests.plan_factory import make_execution_tasks


def _plan() -> DocumentPlan:
    return DocumentPlan(
        execution_tasks=make_execution_tasks(),
        document_type=DocumentType.BUSINESS_REPORT,
        title="Status Report",
        objective="Plan a report.",
        sections=[
            PlanSection(
                heading="Overview",
                purpose="Introduce the report.",
            )
        ],
    )


class FakeBoundModel:
    def __init__(self, model: "FakeModel") -> None:
        self._model = model

    async def ainvoke(self, messages: list[BaseMessage]) -> Any:
        self._model.calls.append(list(messages))
        if self._model.error is not None:
            raise self._model.error
        return self._model.response


class FakeModel:
    """Minimal model double recording bind_tools configuration."""

    def __init__(
        self,
        response: Any = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response or AIMessage(content="No tool needed.")
        self.error = error
        self.calls: list[list[BaseMessage]] = []
        self.tools: list[Any] = []
        self.tool_choice: Any = None

    def bind_tools(
        self,
        tools: list[Any],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> FakeBoundModel:
        self.tools = list(tools)
        self.tool_choice = tool_choice
        return FakeBoundModel(self)


async def test_tool_selector_binds_calculator_with_auto_choice() -> None:
    model = FakeModel(response=AIMessage(content="No tool needed."))
    selector = ToolSelector(model)

    response = await selector.select([HumanMessage(content="Hello")])

    assert response.content == "No tool needed."
    assert [tool.name for tool in model.tools] == ["calculate"]
    assert model.tool_choice == "auto"
    assert model.tool_choice != "required"
    assert len(model.calls) == 1


async def test_tool_selector_wraps_model_failure() -> None:
    cause = RuntimeError("provider unavailable")
    model = FakeModel(error=cause)
    selector = ToolSelector(model)

    with pytest.raises(ToolSelectionError, match="could not produce") as exc_info:
        await selector.select([HumanMessage(content="Hello")])

    assert exc_info.value.__cause__ is cause


async def test_tool_selector_rejects_unexpected_model_output() -> None:
    model = FakeModel(response={"unexpected": "shape"})
    selector = ToolSelector(model)

    with pytest.raises(ToolSelectionError, match="unexpected response"):
        await selector.select([HumanMessage(content="Hello")])


def test_build_tool_selector_accepts_injected_model() -> None:
    model = FakeModel()

    selector = build_tool_selector(model)

    assert isinstance(selector, ToolSelector)
    assert [tool.name for tool in model.tools] == ["calculate"]
    assert model.tool_choice == "auto"


def test_tool_selection_messages_include_request_and_plan() -> None:
    plan = _plan()

    messages = build_tool_selection_messages("What is 17 * 23?", plan)

    assert len(messages) == 2
    assert isinstance(messages[0], BaseMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "What is 17 * 23?" in messages[1].content
    assert plan.title in messages[1].content
