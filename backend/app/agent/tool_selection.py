"""Model-driven tool selection for the agent graph."""

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool

from backend.app.agent.models import DocumentPlan
from backend.app.agent.tools import AGENT_TOOLS
from backend.app.services.llm import build_chat_model

MAX_TOOL_ROUNDS = 3

TOOL_SELECTION_PROMPT = """You are selecting tools for a document-generation \
workflow. The available tool schemas are provided by the API.

Rules:
- Use a tool only when it is genuinely needed to satisfy the request.
- Do not force tool use when a direct answer is sufficient.
- Use the calculator for basic arithmetic, not for document planning.
- Never invent tool results or assume a tool succeeded.
- If a tool fails, report the failure honestly.
- Do not write the document itself; this stage only selects and uses tools.
- When no further tool is needed, return a concise final answer.
"""


class ToolSelectionError(RuntimeError):
    """Raised when tool selection cannot produce a usable model response."""


def build_tool_selection_messages(
    request: str,
    plan: DocumentPlan,
) -> list[BaseMessage]:
    """Build the initial messages for the tool-selection model."""
    return [
        SystemMessage(content=TOOL_SELECTION_PROMPT),
        HumanMessage(
            content=(
                f"User request:\n{request}\n\n"
                f"Approved document plan:\n{plan.model_dump_json(indent=2)}"
            )
        ),
    ]


class ToolSelector:
    """Bind the available tools and ask the model whether to use them."""

    def __init__(
        self,
        model: BaseChatModel,
        tools: Sequence[BaseTool] | None = None,
    ):
        self._bound_model = model.bind_tools(
            list(tools) if tools is not None else AGENT_TOOLS,
            tool_choice="auto",
        )

    async def select(self, messages: list[BaseMessage]) -> AIMessage:
        try:
            response = await self._bound_model.ainvoke(messages)
        except Exception as exc:
            raise ToolSelectionError(
                "The tool-selection model could not produce a response."
            ) from exc

        if not isinstance(response, AIMessage):
            raise ToolSelectionError(
                "The tool-selection model returned an unexpected response."
            )
        return response


def build_tool_selector(model: BaseChatModel | None = None) -> ToolSelector:
    """Build a selector, constructing the model only when needed."""
    return ToolSelector(model or build_chat_model())
