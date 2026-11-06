"""LangGraph workflow with planning, tool execution, and DOCX generation."""

from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from backend.app.agent.document_writer import (
    ContentGenerationError,
    DocumentWriter,
    build_document_writer,
)
from backend.app.agent.docx_generator import DocxGenerationError, DocxGenerator
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
DOCUMENT_FAILURE_RESPONSE = "Unable to generate the document. Please try again."


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


def _extract_tool_results(messages: list[BaseMessage]) -> list[str]:
    """Collect tool-result text only; never model reasoning or tool calls."""
    return [
        message.content
        for message in messages
        if isinstance(message, ToolMessage) and isinstance(message.content, str)
    ]


def route_after_selection(
    state: AgentState,
) -> Literal["tools", "write_document", "tool_failed", "tool_limit_reached"]:
    """Route the latest model response to tools, writing, or safe failure."""
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
    return "write_document"


def route_after_writing(
    state: AgentState,
) -> Literal["generate_docx", "content_failed"]:
    """Route based on whether validated content was produced."""
    return "generate_docx" if state.get("content") is not None else "content_failed"


def route_after_docx(
    state: AgentState,
) -> Literal["finalize", "docx_failed"]:
    """Route based on whether a verified DOCX file was produced."""
    return "finalize" if state.get("output_filename") else "docx_failed"


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


def content_failed(state: AgentState) -> dict[str, Any]:
    """Terminate a request when structured content generation fails."""
    return {
        "final_response": DOCUMENT_FAILURE_RESPONSE,
        "status": "failed",
    }


def docx_failed(state: AgentState) -> dict[str, Any]:
    """Terminate a request when DOCX generation or verification fails."""
    return {
        "final_response": DOCUMENT_FAILURE_RESPONSE,
        "status": "failed",
    }


def finalize(state: AgentState) -> dict[str, Any]:
    """Complete the request only after a verified DOCX file exists."""
    filename = state.get("output_filename")
    if not filename:
        return {
            "final_response": DOCUMENT_FAILURE_RESPONSE,
            "status": "failed",
        }
    return {
        "final_response": f"Document generated successfully: {filename}",
        "status": "completed",
    }


def build_agent_graph(
    planner: DocumentPlanner | None = None,
    tool_selector: ToolSelector | None = None,
    writer: DocumentWriter | None = None,
    docx_generator: DocxGenerator | None = None,
    output_dir: Path | str | None = None,
) -> StateGraph:
    """Build the explicit graph without compiling it.

    Dependencies may be injected for offline tests and isolated output
    directories. Real implementations and models are constructed lazily.
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

    async def write_document(state: AgentState) -> dict[str, Any]:
        plan = state.get("plan")
        if plan is None:
            return {
                "content": None,
                "document_error": "A document plan is required before writing.",
                "status": "failed",
            }

        try:
            active_writer = writer or build_document_writer()
            content = await active_writer.generate(
                state.get("normalized_request", ""),
                plan,
                _extract_tool_results(state.get("messages", [])),
            )
        except (ContentGenerationError, MissingDeepSeekApiKeyError) as exc:
            return {
                "content": None,
                "document_error": str(exc),
                "status": "failed",
            }

        return {
            "content": content,
            "document_error": "",
            "status": "writing",
        }

    async def generate_docx(state: AgentState) -> dict[str, Any]:
        content = state.get("content")
        if content is None:
            return {
                "output_filename": "",
                "document_error": (
                    "Validated content is required before DOCX generation."
                ),
                "status": "failed",
            }

        try:
            active_generator = docx_generator or (
                DocxGenerator(output_dir) if output_dir is not None else DocxGenerator()
            )
            filename = active_generator.generate(
                content,
                state.get("normalized_request", ""),
            )
        except (DocxGenerationError, OSError) as exc:
            return {
                "output_filename": "",
                "document_error": str(exc),
                "status": "failed",
            }

        return {
            "output_filename": filename,
            "document_error": "",
            "status": "docx_generation",
        }

    graph = StateGraph(AgentState)

    graph.add_node("prepare_request", prepare_request)
    graph.add_node("plan_document", plan_document)
    graph.add_node("select_tools", select_tools)
    graph.add_node(
        "tools",
        ToolNode(AGENT_TOOLS, handle_tool_errors=handle_tool_error),
    )
    graph.add_node("write_document", write_document)
    graph.add_node("generate_docx", generate_docx)
    graph.add_node("finalize", finalize)
    graph.add_node("reject", reject)
    graph.add_node("planning_failed", planning_failed)
    graph.add_node("tool_failed", tool_failed)
    graph.add_node("tool_limit_reached", tool_limit_reached)
    graph.add_node("content_failed", content_failed)
    graph.add_node("docx_failed", docx_failed)

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
            "write_document": "write_document",
            "tool_failed": "tool_failed",
            "tool_limit_reached": "tool_limit_reached",
        },
    )
    graph.add_edge("tools", "select_tools")
    graph.add_conditional_edges(
        "write_document",
        route_after_writing,
        {
            "generate_docx": "generate_docx",
            "content_failed": "content_failed",
        },
    )
    graph.add_conditional_edges(
        "generate_docx",
        route_after_docx,
        {
            "finalize": "finalize",
            "docx_failed": "docx_failed",
        },
    )
    graph.add_edge("finalize", END)
    graph.add_edge("reject", END)
    graph.add_edge("planning_failed", END)
    graph.add_edge("tool_failed", END)
    graph.add_edge("tool_limit_reached", END)
    graph.add_edge("content_failed", END)
    graph.add_edge("docx_failed", END)

    return graph


def compile_agent_graph(
    planner: DocumentPlanner | None = None,
    tool_selector: ToolSelector | None = None,
    writer: DocumentWriter | None = None,
    docx_generator: DocxGenerator | None = None,
    output_dir: Path | str | None = None,
) -> CompiledStateGraph:
    """Compile a fresh graph for one invocation."""
    return build_agent_graph(
        planner,
        tool_selector,
        writer,
        docx_generator,
        output_dir,
    ).compile()


async def run_agent(
    request: str,
    *,
    planner: DocumentPlanner | None = None,
    tool_selector: ToolSelector | None = None,
    writer: DocumentWriter | None = None,
    docx_generator: DocxGenerator | None = None,
    output_dir: Path | str | None = None,
) -> AgentState:
    """Run one isolated graph invocation for a request."""
    graph = compile_agent_graph(
        planner,
        tool_selector,
        writer,
        docx_generator,
        output_dir,
    )
    return await graph.ainvoke(create_initial_state(request))
