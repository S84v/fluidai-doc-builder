"""Minimal LangGraph foundation for the document-generation workflow.

The graph currently validates and normalizes a request, then routes to either a
successful terminal node or a rejection terminal node. Future planning, tool,
content-generation, validation, revision and DOCX nodes can be inserted between
``prepare_request`` and ``finalize`` without changing the public invocation
interface.
"""

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from backend.app.agent.state import AgentState, create_initial_state

FINAL_RESPONSE = "Request accepted for document generation."
REJECTION_RESPONSE = "Request rejected: a non-empty request string is required."


def prepare_request(state: AgentState) -> dict[str, Any]:
    """Normalize the request and mark it understood or rejected."""
    request = state.get("request")
    errors: list[str] = []

    if isinstance(request, str):
        normalized_request = " ".join(request.split())
        if not normalized_request:
            errors.append("request must not be empty")
    else:
        normalized_request = ""
        errors.append("request must be a string")

    if errors:
        return {
            "normalized_request": normalized_request,
            "validation_errors": errors,
            "final_response": REJECTION_RESPONSE,
            "status": "rejected",
        }

    return {
        "normalized_request": normalized_request,
        "validation_errors": [],
        "final_response": "",
        "status": "understood",
    }


def route_after_prepare(state: AgentState) -> Literal["finalize", "reject"]:
    """Route rejected requests away from the successful terminal node."""
    return "reject" if state.get("status") == "rejected" else "finalize"


def finalize(state: AgentState) -> dict[str, Any]:
    """Complete a valid request at the current foundation stage."""
    return {
        "final_response": FINAL_RESPONSE,
        "status": "completed",
    }


def reject(state: AgentState) -> dict[str, Any]:
    """Terminate an invalid request without entering the main pipeline."""
    return {
        "final_response": REJECTION_RESPONSE,
        "status": "rejected",
    }


def build_agent_graph() -> StateGraph:
    """Build the explicit graph without compiling it."""
    graph = StateGraph(AgentState)

    graph.add_node("prepare_request", prepare_request)
    graph.add_node("finalize", finalize)
    graph.add_node("reject", reject)

    graph.add_edge(START, "prepare_request")
    graph.add_conditional_edges(
        "prepare_request",
        route_after_prepare,
        {
            "finalize": "finalize",
            "reject": "reject",
        },
    )
    graph.add_edge("finalize", END)
    graph.add_edge("reject", END)

    return graph


def compile_agent_graph() -> CompiledStateGraph:
    """Compile a fresh graph for one invocation."""
    return build_agent_graph().compile()


async def run_agent(request: str) -> AgentState:
    """Run one isolated graph invocation for a request."""
    graph = compile_agent_graph()
    return await graph.ainvoke(create_initial_state(request))
