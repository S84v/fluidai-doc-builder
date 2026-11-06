"""LangGraph workflow with planning and bounded tool execution."""

from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from backend.app.agent.planning import (
    DocumentPlanner,
    PlanningError,
    build_document_planner,
)
from backend.app.agent.state import AgentState, create_initial_state
from backend.app.agent.tool_selection import (
    MAX_TOOL_ROUNDS,
    ToolSelectionError,
    ToolSelector,
    build_tool_selection_messages,
    build_tool_selector,
)
from backend.app.agent.tools import AGENT_TOOLS, handle_tool_error
from backend.app.services.llm import MissingDeepSeekApiKeyError

REJECTION_RESPONSE = "Request rejected: a non-empty request string is required."
PLANNING_FAILURE_RESPONSE = (
    "Unable to create a document plan. Please try again with a clearer request."
)
TOOL_FAILURE_RESPONSE = (
    "Unable to complete the tool-assisted response. Please try again."
)
TOOL_LIMIT_RESPONSE = (
    "The tool execution limit was reached before a final answer could be produced."
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
) -> Literal["select_tools", "planning_failed"]:
    """Route based on whether a validated plan was produced."""
    return "select_tools" if state.get("plan") is not None else "planning_failed"


def _last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def route_after_selection(
    state: AgentState,
) -> Literal["tools", "finalize", "tool_failed", "tool_limit_reached"]:
    """Route the latest model response to tools, success, or safe failure."""
    if state.get("status") == "failed":
        return "tool_failed"

    last_message = _last_ai_message(state.get("messages", []))
    if last_message is None:
        return "tool_failed"

    if last_message.tool_calls:
        if state.get("tool_rounds", 0) > MAX_TOOL_ROUNDS:
            return "tool_limit_reached"
        return "tools"

    if not last_message.text.strip():
        return "tool_failed"
    return "finalize"


def finalize(state: AgentState) -> dict[str, Any]:
    """Complete the request with the model's final tool-assisted answer."""
    last_message = _last_ai_message(state.get("messages", []))
    if last_message is None or not last_message.text.strip():
        return {
            "final_response": TOOL_FAILURE_RESPONSE,
            "status": "failed",
        }
    return {
        "final_response": last_message.text,
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


def tool_failed(state: AgentState) -> dict[str, Any]:
    """Terminate a request when tool selection or execution fails safely."""
    return {
        "final_response": TOOL_FAILURE_RESPONSE,
        "status": "failed",
    }


def tool_limit_reached(state: AgentState) -> dict[str, Any]:
    """Terminate a request that exceeded the bounded tool execution limit."""
    return {
        "final_response": TOOL_LIMIT_RESPONSE,
        "status": "failed",
    }


def build_agent_graph(
    planner: DocumentPlanner | None = None,
    tool_selector: ToolSelector | None = None,
) -> StateGraph:
    """Build the explicit graph without compiling it.

    ``planner`` and ``tool_selector`` may be injected for offline tests. When
    omitted, the real implementations and model are constructed lazily inside
    their graph nodes.
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

    async def select_tools(state: AgentState) -> dict[str, Any]:
        plan = state.get("plan")
        if plan is None:
            return {
                "tool_error": "A document plan is required before tool selection.",
                "status": "failed",
            }

        messages = state.get("messages")
        if messages:
            conversation = list(messages)
            initial_messages: list[BaseMessage] = []
        else:
            conversation = build_tool_selection_messages(
                state.get("normalized_request", ""),
                plan,
            )
            initial_messages = list(conversation)

        try:
            active_selector = tool_selector or build_tool_selector()
            response = await active_selector.select(conversation)
        except (ToolSelectionError, MissingDeepSeekApiKeyError) as exc:
            return {
                "tool_error": str(exc),
                "status": "failed",
            }

        update: dict[str, Any] = {
            "messages": [*initial_messages, response],
            "tool_error": "",
            "status": "tool_selection",
        }
        if response.tool_calls:
            update["tool_rounds"] = state.get("tool_rounds", 0) + 1
        return update

    graph = StateGraph(AgentState)

    graph.add_node("prepare_request", prepare_request)
    graph.add_node("plan_document", plan_document)
    graph.add_node("select_tools", select_tools)
    graph.add_node(
        "tools",
        ToolNode(AGENT_TOOLS, handle_tool_errors=handle_tool_error),
    )
    graph.add_node("finalize", finalize)
    graph.add_node("reject", reject)
    graph.add_node("planning_failed", planning_failed)
    graph.add_node("tool_failed", tool_failed)
    graph.add_node("tool_limit_reached", tool_limit_reached)

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
            "select_tools": "select_tools",
            "planning_failed": "planning_failed",
        },
    )
    graph.add_conditional_edges(
        "select_tools",
        route_after_selection,
        {
            "tools": "tools",
            "finalize": "finalize",
            "tool_failed": "tool_failed",
            "tool_limit_reached": "tool_limit_reached",
        },
    )
    graph.add_edge("tools", "select_tools")
    graph.add_edge("finalize", END)
    graph.add_edge("reject", END)
    graph.add_edge("planning_failed", END)
    graph.add_edge("tool_failed", END)
    graph.add_edge("tool_limit_reached", END)

    return graph


def compile_agent_graph(
    planner: DocumentPlanner | None = None,
    tool_selector: ToolSelector | None = None,
) -> CompiledStateGraph:
    """Compile a fresh graph for one invocation."""
    return build_agent_graph(planner, tool_selector).compile()


async def run_agent(
    request: str,
    *,
    planner: DocumentPlanner | None = None,
    tool_selector: ToolSelector | None = None,
) -> AgentState:
    """Run one isolated graph invocation for a request."""
    graph = compile_agent_graph(planner, tool_selector)
    return await graph.ainvoke(create_initial_state(request))
