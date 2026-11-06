"""Structured document content generation from an approved plan."""

import json
from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from backend.app.agent.document_content import DocumentContent
from backend.app.agent.models import DocumentPlan
from backend.app.agent.validation import ValidationIssue
from backend.app.services.llm import build_chat_model

_CONTENT_SCHEMA = json.dumps(DocumentContent.model_json_schema(), indent=2)

DOCUMENT_WRITING_PROMPT = f"""You write structured business documents.

Return only a JSON object matching this schema:
{_CONTENT_SCHEMA}

Rules:
- Follow the approved plan's document_type, title, section headings and order.
- Use the original request, confirmed facts and tool results as evidence.
- Treat planning assumptions as assumptions, not confirmed facts.
- Use explicit bracketed placeholders such as [Client Name] for unknown details.
- Never invent business facts, names, dates, figures, commitments, or technical
  specifications.
- Include at least one paragraph or bullet in every section.
- Populate audience and metadata only when the request or plan provides them.
- For meeting_minutes, populate meeting details and action_items.
- For project_plan and business_proposal, populate milestones when timeline
  information is available.
- For sop, populate sop_steps as ordered procedure instructions.
- For product_specification, populate requirements.
- For technical_design, populate components.
- Leave optional structured fields empty when they are not supported by known
  facts; do not duplicate structured table data as bullets.
- Do not include internal reasoning, tool calls, or meta commentary.
"""

REVISION_SYSTEM_PROMPT = f"""You revise a structured business document to resolve
blocking validation issues.

Return only a JSON object matching this schema:
{_CONTENT_SCHEMA}

Rules:
- Address every blocking issue and its suggested correction.
- Preserve the approved document type, section headings and section order.
- Preserve supported facts and valid existing content; make targeted edits
  rather than rewriting the whole document unnecessarily.
- Never invent names, dates, figures, commitments, technical specifications, or
  other business facts.
- Use explicit bracketed placeholders for information that is still unknown.
- Keep assumptions distinguishable from confirmed facts.
- Do not introduce internal reasoning, tool calls, or meta commentary.
"""


def _format_items(items: Sequence[str]) -> str:
    if not items:
        return "- None provided."
    return "\n".join(f"- {item}" for item in items)


def build_writing_messages(
    request: str,
    plan: DocumentPlan,
    tool_results: Sequence[str],
) -> list[BaseMessage]:
    """Build writer messages containing only approved inputs, not reasoning."""
    return [
        SystemMessage(content=DOCUMENT_WRITING_PROMPT),
        HumanMessage(
            content=(
                f"Original request:\n{request}\n\n"
                f"Approved plan:\n{plan.model_dump_json(indent=2)}\n\n"
                f"Confirmed facts:\n{_format_items(plan.provided_facts)}\n\n"
                f"Assumptions:\n{_format_items(plan.assumptions)}\n\n"
                f"Missing information:\n{_format_items(plan.missing_information)}\n\n"
                f"Clarification questions:\n"
                f"{_format_items(plan.clarification_questions)}\n\n"
                f"Tool results:\n{_format_items(tool_results)}"
            )
        ),
    ]


class ContentGenerationError(RuntimeError):
    """Raised when valid document content cannot be produced."""


def _format_validation_issues(issues: Sequence[ValidationIssue]) -> str:
    if not issues:
        return "- None provided."
    lines = []
    for issue in issues:
        line = f"- [{issue.code}] {issue.message}"
        if issue.section:
            line += f" (section: {issue.section})"
        if issue.suggestion:
            line += f" Suggested correction: {issue.suggestion}"
        lines.append(line)
    return "\n".join(lines)


def build_revision_messages(
    request: str,
    plan: DocumentPlan,
    content: DocumentContent,
    blocking_issues: Sequence[ValidationIssue],
    tool_results: Sequence[str],
) -> list[BaseMessage]:
    """Build targeted revision messages from blocking validation feedback."""
    return [
        SystemMessage(content=REVISION_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Original request:\n{request}\n\n"
                f"Approved plan:\n{plan.model_dump_json(indent=2)}\n\n"
                f"Current content:\n{content.model_dump_json(indent=2)}\n\n"
                f"Blocking issues:\n"
                f"{_format_validation_issues(blocking_issues)}\n\n"
                f"Confirmed facts:\n{_format_items(plan.provided_facts)}\n\n"
                f"Assumptions:\n{_format_items(plan.assumptions)}\n\n"
                f"Tool results:\n{_format_items(tool_results)}"
            )
        ),
    ]


class DocumentWriter:
    """Turn an approved plan into validated document content."""

    def __init__(self, model: BaseChatModel):
        self._structured_model = model.with_structured_output(
            DocumentContent,
            method="json_mode",
            include_raw=False,
        )

    async def generate(
        self,
        request: str,
        plan: DocumentPlan,
        tool_results: Sequence[str] = (),
    ) -> DocumentContent:
        if not request.strip():
            raise ContentGenerationError("Cannot write content for an empty request.")

        try:
            result = await self._structured_model.ainvoke(
                build_writing_messages(request, plan, tool_results)
            )
        except Exception as exc:
            raise ContentGenerationError(
                "The writing model could not produce valid document content."
            ) from exc

        if isinstance(result, DocumentContent):
            content = result
        else:
            try:
                content = DocumentContent.model_validate(result)
            except Exception as exc:
                raise ContentGenerationError(
                    "The writing model returned invalid document content."
                ) from exc

        _validate_content_against_plan(content, plan)
        return content

    async def revise(
        self,
        request: str,
        plan: DocumentPlan,
        content: DocumentContent,
        blocking_issues: Sequence[ValidationIssue],
        tool_results: Sequence[str] = (),
    ) -> DocumentContent:
        if not blocking_issues:
            raise ContentGenerationError(
                "No blocking issues were supplied for revision."
            )

        try:
            result = await self._structured_model.ainvoke(
                build_revision_messages(
                    request,
                    plan,
                    content,
                    blocking_issues,
                    tool_results,
                )
            )
        except Exception as exc:
            raise ContentGenerationError(
                "The writing model could not revise the document."
            ) from exc

        if isinstance(result, DocumentContent):
            revised = result
        else:
            try:
                revised = DocumentContent.model_validate(result)
            except Exception as exc:
                raise ContentGenerationError(
                    "The writing model returned invalid revised content."
                ) from exc

        _validate_content_against_plan(revised, plan)
        return revised


def _normalize_heading(heading: str) -> str:
    return " ".join(heading.split()).casefold()


def _validate_content_against_plan(
    content: DocumentContent,
    plan: DocumentPlan,
) -> None:
    if content.document_type != plan.document_type:
        raise ContentGenerationError(
            "Generated content does not match the approved document type."
        )

    content_headings = [
        _normalize_heading(section.heading) for section in content.sections
    ]
    plan_headings = [_normalize_heading(section.heading) for section in plan.sections]
    if content_headings != plan_headings:
        raise ContentGenerationError(
            "Generated sections do not match the approved plan order."
        )


def build_document_writer(model: BaseChatModel | None = None) -> DocumentWriter:
    """Build a writer, constructing the model only when needed."""
    return DocumentWriter(model or build_chat_model())
