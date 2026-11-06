"""Typed state for the document-generation agent graph."""

from typing import Literal, TypedDict

AgentStatus = Literal["received", "understood", "completed", "rejected"]


class AgentState(TypedDict, total=False):
    """State shared by the agent graph.

    ``request`` is required at invocation time. The remaining fields are
    populated by graph nodes and are optional so future stages can add state
    without changing the graph input contract.
    """

    request: str
    normalized_request: str
    plan: list[str]
    content: str
    validation_errors: list[str]
    revision_count: int
    output_filename: str
    final_response: str
    status: AgentStatus


def create_initial_state(request: str) -> AgentState:
    """Return a fresh state object for one graph invocation."""
    return {
        "request": request,
        "normalized_request": "",
        "plan": [],
        "content": "",
        "validation_errors": [],
        "revision_count": 0,
        "output_filename": "",
        "final_response": "",
        "status": "received",
    }
