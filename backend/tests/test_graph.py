"""Offline tests for the LangGraph foundation and planning routing."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from backend.app.agent import (
    DocumentPlan,
    DocumentType,
    PlanningError,
    PlanSection,
    build_agent_graph,
    compile_agent_graph,
    create_initial_state,
    run_agent,
)
from backend.app.services.llm import MissingDeepSeekApiKeyError


def _plan(
    *,
    document_type: DocumentType = DocumentType.BUSINESS_REPORT,
    title: str = "Status Report",
) -> DocumentPlan:
    return DocumentPlan(
        document_type=document_type,
        title=title,
        objective="Plan a structured business document.",
        sections=[
            PlanSection(
                heading="Overview",
                purpose="Introduce the document.",
                key_points=["Purpose", "Scope"],
            )
        ],
    )


class FakePlanner:
    """Deterministic planner used instead of the DeepSeek-backed planner."""

    def __init__(
        self,
        plan: DocumentPlan | None = None,
        error: Exception | None = None,
    ) -> None:
        self._plan = plan
        self._error = error
        self.requests: list[str] = []

    async def create_plan(self, request: str) -> DocumentPlan:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        assert self._plan is not None
        return self._plan


class EchoPlanner:
    """Return a fresh plan titled with the request, for isolation tests."""

    async def create_plan(self, request: str) -> DocumentPlan:
        return _plan(title=request)


class FakeToolSelector:
    """Return a deterministic no-tool model response."""

    def __init__(
        self,
        response: AIMessage | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response = response or AIMessage(content="No tool was required.")
        self._error = error
        self.calls: list[list[BaseMessage]] = []

    async def select(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls.append(list(messages))
        if self._error is not None:
            raise self._error
        return self._response


def test_graph_builds_and_compiles() -> None:
    graph = build_agent_graph()
    compiled = compile_agent_graph()

    assert isinstance(graph, StateGraph)
    assert isinstance(compiled, CompiledStateGraph)


async def test_valid_request_reaches_completed_state_with_plan(
    fake_writer: Any,
    fake_docx_generator: Any,
    fake_validator: Any,
) -> None:
    plan = _plan()
    planner = FakePlanner(plan=plan)
    selector = FakeToolSelector()

    result = await run_agent(
        "  Create   a status report  ",
        planner=planner,
        tool_selector=selector,
        writer=fake_writer,
        docx_generator=fake_docx_generator,
        validator=fake_validator,
    )

    assert result["status"] == "completed"
    assert result["normalized_request"] == "Create a status report"
    assert result["plan"] == plan
    assert result["content"] is not None
    assert result["planning_error"] == ""
    assert result["document_error"] == ""
    assert result["validation_errors"] == []
    assert result["revision_count"] == 0
    assert result["tool_rounds"] == 0
    assert result["tool_error"] == ""
    assert result["output_filename"] == "test-document-000000000000.docx"
    assert result["final_response"] == (
        "Document generated successfully: test-document-000000000000.docx"
    )
    assert planner.requests == ["Create a status report"]
    assert fake_writer.calls[0][2] == []
    assert len(fake_docx_generator.calls) == 1


async def test_empty_request_is_routed_to_rejection_without_planning() -> None:
    planner = FakePlanner(plan=_plan())

    result = await run_agent("   ", planner=planner)

    assert result["status"] == "rejected"
    assert result["normalized_request"] == ""
    assert result["validation_errors"] == ["request must not be empty"]
    assert result["final_response"] == (
        "Request rejected: a non-empty request string is required."
    )
    assert planner.requests == []


async def test_non_string_request_is_rejected_without_planning() -> None:
    planner = FakePlanner(plan=_plan())
    compiled = compile_agent_graph(planner)

    result = await compiled.ainvoke({"request": 42})

    assert result["status"] == "rejected"
    assert result["validation_errors"] == ["request must be a string"]
    assert planner.requests == []


async def test_missing_request_is_rejected() -> None:
    planner = FakePlanner(plan=_plan())
    compiled = compile_agent_graph(planner)

    result = await compiled.ainvoke({})

    assert result["status"] == "rejected"
    assert result["validation_errors"] == ["request must be a string"]


async def test_planning_failure_reaches_safe_failure_state() -> None:
    planner = FakePlanner(error=PlanningError("planning unavailable"))

    result = await run_agent("Create a report", planner=planner)

    assert result["status"] == "failed"
    assert result["plan"] is None
    assert result["planning_error"] == "planning unavailable"
    assert result["final_response"] == (
        "Unable to create a document plan. Please try again with a clearer request."
    )


async def test_missing_credentials_reach_safe_failure_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_missing_credentials() -> None:
        raise MissingDeepSeekApiKeyError("DEEPSEEK_API_KEY is not configured.")

    monkeypatch.setattr(
        "backend.app.agent.graph.build_document_planner",
        raise_missing_credentials,
    )

    result = await run_agent("Create a report")

    assert result["status"] == "failed"
    assert result["plan"] is None
    assert result["planning_error"] == "DEEPSEEK_API_KEY is not configured."
    assert result["final_response"] == (
        "Unable to create a document plan. Please try again with a clearer request."
    )


async def test_state_is_isolated_between_invocations(
    fake_writer: Any,
    fake_docx_generator: Any,
    fake_validator: Any,
) -> None:
    first = await run_agent(
        "first request",
        planner=EchoPlanner(),
        tool_selector=FakeToolSelector(AIMessage(content="first answer")),
        writer=fake_writer,
        docx_generator=fake_docx_generator,
        validator=fake_validator,
    )
    second = await run_agent(
        "second request",
        planner=EchoPlanner(),
        tool_selector=FakeToolSelector(AIMessage(content="second answer")),
        writer=fake_writer,
        docx_generator=fake_docx_generator,
        validator=fake_validator,
    )

    assert first is not second
    assert first["plan"] is not None
    assert second["plan"] is not None
    assert first["plan"].title == "first request"
    assert second["plan"].title == "second request"
    assert first["content"] is not None
    assert second["content"] is not None
    assert first["content"] is not second["content"]

    first["plan"].title = "mutated"
    first["validation_errors"].append("mutated")
    first["messages"].append(AIMessage(content="mutated"))

    assert second["plan"].title == "second request"
    assert second["validation_errors"] == []
    assert len(second["messages"]) == 3


def test_initial_state_uses_fresh_mutable_containers() -> None:
    first = create_initial_state("first")
    second = create_initial_state("second")

    first["validation_errors"].append("mutated")
    first["messages"].append(AIMessage(content="mutated"))

    assert second["validation_errors"] == []
    assert second["messages"] == []
    assert first["validation_errors"] is not second["validation_errors"]
    assert first["messages"] is not second["messages"]
    assert first["content"] is None
    assert second["content"] is None
    assert first["document_error"] == ""
    assert first["plan"] is None
    assert second["plan"] is None
