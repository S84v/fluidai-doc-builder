"""Typed state for the document-generation agent graph."""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from backend.app.agent.document_content import DocumentContent
from backend.app.agent.models import DocumentPlan
from backend.app.agent.validation import ValidationReport

AgentStatus = Literal[
    "received",
    "understood",
    "planned",
    "tool_selection",
    "writing",
    "validating",
    "revising",
    "docx_generation",
    "completed",
    "rejected",
    "failed",
]

TerminalReason = Literal[
    "completed",
    "rejected",
    "planning_failed",
    "tool_failed",
    "tool_limit_reached",
    "content_failed",
    "validation_failed",
    "review_failed",
    "revision_failed",
    "docx_failed",
]


class AgentState(TypedDict, total=False):
    """State shared by the agent graph.

    ``request`` is required at invocation time. The remaining fields are
    populated by graph nodes and are optional so future stages can add state
    without changing the graph input contract.
    """

    request: str
    normalized_request: str
    plan: DocumentPlan | None
    planning_error: str
    messages: Annotated[list[AnyMessage], add_messages]
    tool_rounds: int
    tool_error: str
    content: DocumentContent | None
    document_error: str
    validation_report: ValidationReport | None
    validation_errors: list[str]
    revision_count: int
    output_filename: str
    final_response: str
    terminal_reason: TerminalReason | None
    status: AgentStatus


def create_initial_state(request: str) -> AgentState:
    """Return a fresh state object for one graph invocation."""
    return {
        "request": request,
        "normalized_request": "",
        "plan": None,
        "planning_error": "",
        "messages": [],
        "tool_rounds": 0,
        "tool_error": "",
        "content": None,
        "document_error": "",
        "validation_report": None,
        "validation_errors": [],
        "revision_count": 0,
        "output_filename": "",
        "final_response": "",
        "terminal_reason": None,
        "status": "received",
    }
