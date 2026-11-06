"""LangGraph foundation for the document-generation workflow.

The current graph validates and normalizes a request, produces a structured
document plan, then reaches a success or failure terminal node. Future tool,
content-generation, validation, revision and DOCX nodes can be inserted between
``plan_document`` and ``finalize`` without changing the invocation interface.
"""

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from backend.app.agent.planning import (
    DocumentPlanner,
    PlanningError,
    build_document_planner,
)
from backend.app.agent.state import AgentState, create_initial_state
from backend.app.services.llm import MissingDeepSeekApiKeyError

FINAL_RESPONSE = "Document plan created."
REJECTION_RESPONSE = "Request rejected: a non-empty request string is required."
PLANNING_FAILURE_RESPONSE = (
    "Unable to create a document plan. Please try again with a clearer request."
)


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


def route_after_prepare(state: AgentState) -> Literal["plan_document", "reject"]:
    """Send valid requests to planning and invalid requests to rejection."""
    return "reject" if state.get("status") == "rejected" else "plan_document"


def route_after_planning(
    state: AgentState,
) -> Literal["finalize", "planning_failed"]:
    """Route based on whether a validated plan was produced."""
    return "finalize" if state.get("plan") is not None else "planning_failed"


def finalize(state: AgentState) -> dict[str, Any]:
    """Complete a request whose plan was produced successfully."""
    if state.get("plan") is None:
        return {
            "final_response": PLANNING_FAILURE_RESPONSE,
            "status": "failed",
        }
    return {
        "final_response": FINAL_RESPONSE,
        "status": "completed",
    }


def reject(state: AgentState) -> dict[str, Any]:
    """Terminate an invalid request without entering the pipeline."""
    return {
        "final_response": REJECTION_RESPONSE,
        "status": "rejected",
    }


def planning_failed(state: AgentState) -> dict[str, Any]:
    """Terminate a request when planning could not produce a valid plan."""
    return {
        "final_response": PLANNING_FAILURE_RESPONSE,
        "status": "failed",
    }


def build_agent_graph(planner: DocumentPlanner | None = None) -> StateGraph:
    """Build the explicit graph without compiling it.

    ``planner`` may be injected for offline tests. When omitted, the real
    planner and model are constructed lazily inside the planning node.
    """

    async def plan_document(state: AgentState) -> dict[str, Any]:
        try:
            active_planner = planner or build_document_planner()
            plan = await active_planner.create_plan(state.get("normalized_request", ""))
        except (PlanningError, MissingDeepSeekApiKeyError) as exc:
            return {
                "plan": None,
                "planning_error": str(exc),
                "status": "failed",
            }
        return {
            "plan": plan,
            "planning_error": "",
            "status": "planned",
        }

    graph = StateGraph(AgentState)

    graph.add_node("prepare_request", prepare_request)
    graph.add_node("plan_document", plan_document)
    graph.add_node("finalize", finalize)
    graph.add_node("reject", reject)
    graph.add_node("planning_failed", planning_failed)

    graph.add_edge(START, "prepare_request")
    graph.add_conditional_edges(
        "prepare_request",
        route_after_prepare,
        {
            "plan_document": "plan_document",
            "reject": "reject",
        },
    )
    graph.add_conditional_edges(
        "plan_document",
        route_after_planning,
        {
            "finalize": "finalize",
            "planning_failed": "planning_failed",
        },
    )
    graph.add_edge("finalize", END)
    graph.add_edge("reject", END)
    graph.add_edge("planning_failed", END)

    return graph


def compile_agent_graph(planner: DocumentPlanner | None = None) -> CompiledStateGraph:
    """Compile a fresh graph for one invocation."""
    return build_agent_graph(planner).compile()


async def run_agent(
    request: str,
    *,
    planner: DocumentPlanner | None = None,
) -> AgentState:
    """Run one isolated graph invocation for a request."""
    graph = compile_agent_graph(planner)
    return await graph.ainvoke(create_initial_state(request))
