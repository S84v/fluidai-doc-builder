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
- The original user request is the sole source of factual information.
- You may reorganise, structure, correct grammar, improve readability and
  faithfully rephrase information supplied by the user without changing meaning.
- Never invent names, organisations, dates, financial figures, commitments,
  organisational policies, targets, technical specifications or other factual
  details.
- Do not present planning assumptions or conflict resolutions as confirmed facts.
- Do not include planning assumptions, conflict resolutions or planner suggestions
  in document content.
- Do not invent recommendations, proposals or additional content to fill missing
  information.
- Follow the approved plan's document_type, title, section headings and order;
  every planned section heading must remain in the document.
- Factual accuracy takes precedence over filling every section. Never add filler
  paragraphs, bullets or placeholder text merely to satisfy structure.
- Use explicit bracketed placeholders such as [Client Name] for unknown details.
- Use only placeholder names declared in the approved plan. Do not invent new
  placeholder names or use placeholders for supplied facts.
- When a required section has no user-supported content, use only a justified
  placeholder from the approved plan instead of inventing content.
- Omit optional content when no useful user-supported information exists.
  Optional content means optional paragraphs, bullets and structured fields;
  planned section headings are still required.
- A planned section may have no paragraph or bullet when its information is fully
  represented by populated structured fields, such as meeting details or action
  items; do not duplicate that data as filler.
- Populate audience and metadata only when the request or plan provides them.
- For meeting_minutes, populate meeting details and action_items only from
  explicitly supplied information.
- For project_plan and business_proposal, populate milestones only when timeline
  information is explicitly available.
- For sop, populate sop_steps only from supplied procedure information.
- For product_specification, populate requirements only from supplied requirements.
- For technical_design, populate components only from supplied technical details.
- Leave optional structured fields empty when they are not supported by known
  facts; do not duplicate structured table data as bullets.
- Tool results may be used only as deterministic evidence; never invent tool
  results or infer additional facts from them.
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
- The original user request remains the sole source of factual information.
- Never introduce names, organisations, dates, figures, commitments, policies,
  targets, technical specifications or other unsupported factual details.
- Do not convert planning assumptions or conflict resolutions into established
  facts.
- Do not replace a justified placeholder with an invented value.
- Do not invent recommendations or proposed content to fill missing information.
- Do not add filler paragraphs, bullets or placeholder text merely to satisfy
  structure; keep factual accuracy ahead of completeness.
- An empty section body is acceptable when its information is fully represented
  by populated structured fields; do not add filler to it.
- Use explicit bracketed placeholders only for still-unknown information declared
  in the approved plan.
- Do not introduce internal reasoning, tool calls, or meta commentary.
"""


def _format_items(items: Sequence[str]) -> str:
    if not items:
        return "- None provided."
    return "\n".join(f"- {item}" for item in items)


def _writing_context(plan: DocumentPlan) -> dict[str, object]:
    """Return only plan fields that may inform document content."""
    return {
        "document_type": plan.document_type.value,
        "title": plan.title,
        "objective": plan.objective,
        "audience": plan.audience,
        "sections": [
            {
                "heading": section.heading,
                "placeholders": [
                    {"text": placeholder.text, "reason": placeholder.reason}
                    for placeholder in section.placeholders
                ],
            }
            for section in plan.sections
        ],
        "provided_facts": plan.provided_facts,
        "missing_information": plan.missing_information,
    }


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
                f"Approved writing context:\n"
                f"{json.dumps(_writing_context(plan), indent=2)}\n\n"
                f"Confirmed facts:\n{_format_items(plan.provided_facts)}\n\n"
                f"Missing information:\n{_format_items(plan.missing_information)}\n\n"
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
                f"Approved writing context:\n"
                f"{json.dumps(_writing_context(plan), indent=2)}\n\n"
                f"Current content:\n{content.model_dump_json(indent=2)}\n\n"
                f"Blocking issues:\n"
                f"{_format_validation_issues(blocking_issues)}\n\n"
                f"Confirmed facts:\n{_format_items(plan.provided_facts)}\n\n"
                f"Missing information:\n{_format_items(plan.missing_information)}\n\n"
                f"Tool results:\n{_format_items(tool_results)}"
            )
        ),
    ]


class DocumentWriter:
    """Turn an approved plan into validated document content."""

    MAX_WRITING_ATTEMPTS = 2

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

        messages = build_writing_messages(request, plan, tool_results)
        last_error: Exception | None = None
        for _ in range(self.MAX_WRITING_ATTEMPTS):
            try:
                result = await self._structured_model.ainvoke(messages)
            except Exception as exc:
                last_error = exc
                continue

            try:
                return _coerce_content(
                    result,
                    plan,
                    "The writing model returned invalid document content.",
                )
            except ContentGenerationError as exc:
                last_error = exc

        if isinstance(last_error, ContentGenerationError):
            raise last_error
        raise ContentGenerationError(
            "The writing model could not produce valid document content."
        ) from last_error

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

        messages = build_revision_messages(
            request,
            plan,
            content,
            blocking_issues,
            tool_results,
        )
        last_error: Exception | None = None
        for _ in range(self.MAX_WRITING_ATTEMPTS):
            try:
                result = await self._structured_model.ainvoke(messages)
            except Exception as exc:
                last_error = exc
                continue

            try:
                return _coerce_content(
                    result,
                    plan,
                    "The writing model returned invalid revised content.",
                )
            except ContentGenerationError as exc:
                last_error = exc

        if isinstance(last_error, ContentGenerationError):
            raise last_error
        raise ContentGenerationError(
            "The writing model could not revise the document."
        ) from last_error


def _coerce_content(
    result: object,
    plan: DocumentPlan,
    invalid_message: str,
) -> DocumentContent:
    if isinstance(result, DocumentContent):
        content = result
    else:
        try:
            content = DocumentContent.model_validate(result)
        except Exception as exc:
            raise ContentGenerationError(invalid_message) from exc

    _validate_content_against_plan(content, plan)
    return content


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
