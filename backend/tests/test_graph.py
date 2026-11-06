"""Offline tests for the minimal LangGraph foundation."""

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from backend.app.agent import (
    build_agent_graph,
    compile_agent_graph,
    create_initial_state,
    run_agent,
)


def test_graph_builds_and_compiles() -> None:
    graph = build_agent_graph()
    compiled = compile_agent_graph()

    assert isinstance(graph, StateGraph)
    assert isinstance(compiled, CompiledStateGraph)


async def test_valid_request_reaches_completed_terminal_state() -> None:
    result = await run_agent("  Create   a status report  ")

    assert result["status"] == "completed"
    assert result["normalized_request"] == "Create a status report"
    assert result["plan"] == []
    assert result["content"] == ""
    assert result["validation_errors"] == []
    assert result["revision_count"] == 0
    assert result["output_filename"] == ""
    assert result["final_response"] == "Request accepted for document generation."


async def test_empty_request_is_routed_to_rejection() -> None:
    result = await run_agent("   ")

    assert result["status"] == "rejected"
    assert result["normalized_request"] == ""
    assert result["validation_errors"] == ["request must not be empty"]
    assert result["final_response"] == (
        "Request rejected: a non-empty request string is required."
    )


async def test_non_string_request_is_rejected() -> None:
    compiled = compile_agent_graph()

    result = await compiled.ainvoke({"request": 42})

    assert result["status"] == "rejected"
    assert result["validation_errors"] == ["request must be a string"]
    assert result["final_response"] == (
        "Request rejected: a non-empty request string is required."
    )


async def test_missing_request_is_rejected() -> None:
    compiled = compile_agent_graph()

    result = await compiled.ainvoke({})

    assert result["status"] == "rejected"
    assert result["validation_errors"] == ["request must be a string"]


async def test_state_is_isolated_between_invocations() -> None:
    first = await run_agent("first request")
    second = await run_agent("second request")

    assert first is not second
    assert first["normalized_request"] == "first request"
    assert second["normalized_request"] == "second request"

    first["normalized_request"] = "mutated"
    first["validation_errors"].append("mutated")
    first["plan"].append("mutated")

    assert second["normalized_request"] == "second request"
    assert second["validation_errors"] == []
    assert second["plan"] == []


def test_initial_state_uses_fresh_mutable_containers() -> None:
    first = create_initial_state("first")
    second = create_initial_state("second")

    first["validation_errors"].append("mutated")
    first["plan"].append("mutated")

    assert second["validation_errors"] == []
    assert second["plan"] == []
    assert first["validation_errors"] is not second["validation_errors"]
    assert first["plan"] is not second["plan"]
