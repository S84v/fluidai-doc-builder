"""Offline tests for graph tool execution and bounded tool rounds."""

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

from backend.app.agent import (
    MAX_TOOL_ROUNDS,
    DocumentPlan,
    DocumentType,
    PlanningError,
    PlanSection,
    ToolSelectionError,
    run_agent,
)
from backend.app.agent.graph import TOOL_FAILURE_RESPONSE, TOOL_LIMIT_RESPONSE


def _plan() -> DocumentPlan:
    return DocumentPlan(
        document_type=DocumentType.BUSINESS_REPORT,
        title="Status Report",
        objective="Plan a structured business document.",
        sections=[
            PlanSection(
                heading="Overview",
                purpose="Introduce the document.",
            )
        ],
    )


class FakePlanner:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error

    async def create_plan(self, request: str) -> DocumentPlan:
        if self._error is not None:
            raise self._error
        return _plan()


class FakeToolSelector:
    def __init__(
        self,
        responses: list[AIMessage] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._error = error
        self.calls: list[list[BaseMessage]] = []

    async def select(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls.append(list(messages))
        if self._error is not None:
            raise self._error
        if not self._responses:
            raise AssertionError("No fake tool-selection response remains")
        return self._responses.pop(0)


def _tool_call(expression: str, call_id: str) -> dict[str, Any]:
    return {
        "name": "calculate",
        "args": {"expression": expression},
        "id": call_id,
        "type": "tool_call",
    }


def _tool_response(
    expression: str,
    call_id: str,
    *,
    reasoning_content: str | None = None,
) -> AIMessage:
    additional_kwargs = (
        {"reasoning_content": reasoning_content} if reasoning_content else {}
    )
    return AIMessage(
        content="",
        tool_calls=[_tool_call(expression, call_id)],
        additional_kwargs=additional_kwargs,
    )


async def test_tool_request_executes_and_result_returns_to_model(
    fake_writer: Any,
    fake_docx_generator: Any,
) -> None:
    selector = FakeToolSelector(
        [
            _tool_response(
                "17 * 23",
                "call_1",
                reasoning_content="Use the calculator.",
            ),
            AIMessage(content="17 * 23 = 391."),
        ]
    )

    result = await run_agent(
        "What is 17 * 23?",
        planner=FakePlanner(),
        tool_selector=selector,
        writer=fake_writer,
        docx_generator=fake_docx_generator,
    )

    assert result["status"] == "completed"
    assert result["final_response"].startswith("Document generated successfully:")
    assert result["output_filename"] == "test-document-000000000000.docx"
    assert result["tool_rounds"] == 1
    assert len(selector.calls) == 2

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "391"
    assert tool_messages[0].tool_call_id == "call_1"

    second_call = selector.calls[1]
    assert any(
        isinstance(message, ToolMessage) and message.content == "391"
        for message in second_call
    )
    first_ai = next(
        message
        for message in second_call
        if isinstance(message, AIMessage) and message.tool_calls
    )
    assert first_ai.additional_kwargs["reasoning_content"] == "Use the calculator."


async def test_no_tool_needed_skips_tool_execution(
    fake_writer: Any,
    fake_docx_generator: Any,
) -> None:
    selector = FakeToolSelector([AIMessage(content="No tool is needed.")])

    result = await run_agent(
        "Summarise the plan.",
        planner=FakePlanner(),
        tool_selector=selector,
        writer=fake_writer,
        docx_generator=fake_docx_generator,
    )

    assert result["status"] == "completed"
    assert result["final_response"].startswith("Document generated successfully:")
    assert result["tool_rounds"] == 0
    assert not any(isinstance(message, ToolMessage) for message in result["messages"])
    assert len(selector.calls) == 1


async def test_multiple_tool_rounds_are_supported(
    fake_writer: Any,
    fake_docx_generator: Any,
) -> None:
    selector = FakeToolSelector(
        [
            _tool_response("1 + 1", "call_1"),
            _tool_response("2 + 2", "call_2"),
            AIMessage(content="Both calculations are complete."),
        ]
    )

    result = await run_agent(
        "Calculate 1 + 1 and 2 + 2.",
        planner=FakePlanner(),
        tool_selector=selector,
        writer=fake_writer,
        docx_generator=fake_docx_generator,
    )

    assert result["status"] == "completed"
    assert result["tool_rounds"] == 2
    tool_contents = [
        message.content
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]
    assert tool_contents == ["2", "4"]
    assert len(selector.calls) == 3


async def test_tool_round_limit_is_enforced() -> None:
    responses = [
        _tool_response("1 + 1", f"call_{index}") for index in range(MAX_TOOL_ROUNDS + 1)
    ]
    selector = FakeToolSelector(responses)

    result = await run_agent(
        "Keep calculating.",
        planner=FakePlanner(),
        tool_selector=selector,
    )

    assert result["status"] == "failed"
    assert result["final_response"] == TOOL_LIMIT_RESPONSE
    assert result["tool_rounds"] == MAX_TOOL_ROUNDS + 1
    assert len(selector.calls) == MAX_TOOL_ROUNDS + 1
    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == MAX_TOOL_ROUNDS


async def test_model_failure_reaches_safe_failure_state() -> None:
    selector = FakeToolSelector(
        error=ToolSelectionError("model unavailable"),
    )

    result = await run_agent(
        "Calculate 1 + 1.",
        planner=FakePlanner(),
        tool_selector=selector,
    )

    assert result["status"] == "failed"
    assert result["final_response"] == TOOL_FAILURE_RESPONSE
    assert result["tool_error"] == "model unavailable"


async def test_empty_model_response_reaches_safe_failure_state() -> None:
    selector = FakeToolSelector([AIMessage(content="")])

    result = await run_agent(
        "Calculate 1 + 1.",
        planner=FakePlanner(),
        tool_selector=selector,
    )

    assert result["status"] == "failed"
    assert result["final_response"] == TOOL_FAILURE_RESPONSE


async def test_tool_error_is_returned_to_model(
    fake_writer: Any,
    fake_docx_generator: Any,
) -> None:
    selector = FakeToolSelector(
        [
            _tool_response("1 / 0", "call_1"),
            AIMessage(content="The calculation failed because of division by zero."),
        ]
    )

    result = await run_agent(
        "Calculate 1 / 0.",
        planner=FakePlanner(),
        tool_selector=selector,
        writer=fake_writer,
        docx_generator=fake_docx_generator,
    )

    assert result["status"] == "completed"
    tool_message = next(
        message for message in result["messages"] if isinstance(message, ToolMessage)
    )
    assert tool_message.content == "Calculation error: division by zero"
    assert "division by zero" in selector.calls[1][-1].content


async def test_invalid_tool_arguments_are_handled(
    fake_writer: Any,
    fake_docx_generator: Any,
) -> None:
    invalid_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculate",
                "args": {},
                "id": "call_invalid",
                "type": "tool_call",
            }
        ],
    )
    selector = FakeToolSelector(
        [
            invalid_call,
            AIMessage(content="The calculator could not run with those arguments."),
        ]
    )

    result = await run_agent(
        "Calculate something.",
        planner=FakePlanner(),
        tool_selector=selector,
        writer=fake_writer,
        docx_generator=fake_docx_generator,
    )

    assert result["status"] == "completed"
    tool_message = next(
        message for message in result["messages"] if isinstance(message, ToolMessage)
    )
    assert "invalid arguments" in tool_message.content


async def test_tool_messages_are_request_local(
    fake_writer: Any,
    fake_docx_generator: Any,
) -> None:
    first_selector = FakeToolSelector(
        [
            _tool_response("1 + 1", "call_1"),
            AIMessage(content="first result"),
        ]
    )
    second_selector = FakeToolSelector([AIMessage(content="second result")])

    first = await run_agent(
        "Calculate 1 + 1.",
        planner=FakePlanner(),
        tool_selector=first_selector,
        writer=fake_writer,
        docx_generator=fake_docx_generator,
    )
    second = await run_agent(
        "No calculation.",
        planner=FakePlanner(),
        tool_selector=second_selector,
        writer=fake_writer,
        docx_generator=fake_docx_generator,
    )

    assert first["messages"] is not second["messages"]
    assert any(isinstance(message, ToolMessage) for message in first["messages"])
    assert not any(isinstance(message, ToolMessage) for message in second["messages"])

    first["messages"].append(AIMessage(content="mutated"))
    assert len(second["messages"]) == 3


async def test_planning_failure_still_rejects_before_tools() -> None:
    selector = FakeToolSelector([AIMessage(content="should not run")])

    result = await run_agent(
        "Calculate 1 + 1.",
        planner=FakePlanner(error=PlanningError("planning failed")),
        tool_selector=selector,
    )

    assert result["status"] == "failed"
    assert selector.calls == []


async def test_initial_messages_are_system_and_human(
    fake_writer: Any,
    fake_docx_generator: Any,
) -> None:
    selector = FakeToolSelector([AIMessage(content="No tool needed.")])

    await run_agent(
        "What is 17 * 23?",
        planner=FakePlanner(),
        tool_selector=selector,
        writer=fake_writer,
        docx_generator=fake_docx_generator,
    )

    assert len(selector.calls[0]) == 2
    assert isinstance(selector.calls[0][0], SystemMessage)
    assert "What is 17 * 23?" in selector.calls[0][1].content
