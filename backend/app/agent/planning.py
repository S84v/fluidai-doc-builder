"""Request understanding and document planning."""

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from backend.app.agent.models import DocumentPlan, DocumentType
from backend.app.services.llm import build_chat_model

_SUPPORTED_DOCUMENT_TYPES = ", ".join(
    document_type.value for document_type in DocumentType
)
_JSON_SCHEMA = json.dumps(DocumentPlan.model_json_schema(), indent=2)

PLANNING_SYSTEM_PROMPT = f"""You are a document-planning assistant. You plan
structured business documents; you do not write their content.

Classify the request into exactly one document_type from:
{_SUPPORTED_DOCUMENT_TYPES}

Use the following JSON schema for your response:
{_JSON_SCHEMA}

Planning rules:
- Use only facts explicitly stated in the user's request for provided_facts.
- Put reasonable, non-factual choices needed to structure the plan in assumptions.
- Put details that are absent or unclear in missing_information.
- Put only questions that genuinely require user input in clarification_questions.
- Never invent names, dates, figures, people, technical specifications,
  commitments, or other business facts.
- Use bracketed placeholders such as [Client Name] where details are unknown.
- When essential information is missing, still produce a useful outline and ask
  specific clarification questions instead of fabricating details.
- Produce at least one section with a clear heading, purpose and key points.
- Return only a JSON object matching the schema. Do not include markdown.
"""


class PlanningError(RuntimeError):
    """Raised when a valid document plan cannot be produced."""


class DocumentPlanner:
    """Turn a normalized request into a validated document plan."""

    def __init__(self, model: BaseChatModel):
        # JSON mode avoids forced tool_choice, which is not supported in
        # DeepSeek thinking mode. It still uses Chat Completions.
        self._structured_model = model.with_structured_output(
            DocumentPlan,
            method="json_mode",
            include_raw=False,
        )

    async def create_plan(self, request: str) -> DocumentPlan:
        if not request.strip():
            raise PlanningError("Cannot create a plan for an empty request.")

        messages = [
            SystemMessage(content=PLANNING_SYSTEM_PROMPT),
            HumanMessage(content=request),
        ]

        try:
            result = await self._structured_model.ainvoke(messages)
        except Exception as exc:
            raise PlanningError(
                "The planning model could not produce a valid document plan."
            ) from exc

        if isinstance(result, DocumentPlan):
            return result

        try:
            return DocumentPlan.model_validate(result)
        except Exception as exc:
            raise PlanningError(
                "The planning model returned an invalid document plan."
            ) from exc


def build_document_planner(model: BaseChatModel | None = None) -> DocumentPlanner:
    """Build a planner, constructing the model only when needed."""
    return DocumentPlanner(model or build_chat_model())
