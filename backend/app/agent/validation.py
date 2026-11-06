"""Deterministic validation and LLM review for document content."""

import json
import re
from collections.abc import Iterator, Sequence
from enum import StrEnum
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, computed_field

from backend.app.agent.document_content import DocumentContent
from backend.app.agent.models import DocumentPlan, DocumentType
from backend.app.services.llm import build_chat_model

MAX_REVISION_ATTEMPTS = 2

_PLACEHOLDER_RE = re.compile(r"\[[^\[\]]+\]")
_EMPTY_PLACEHOLDER_RE = re.compile(r"\[\s*\]")


class IssueSeverity(StrEnum):
    BLOCKING = "blocking"
    WARNING = "warning"


class ValidationIssue(BaseModel):
    """One actionable validation issue."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    severity: IssueSeverity
    section: str | None = None
    suggestion: str | None = None


class ValidationReport(BaseModel):
    """Combined deterministic and model-review validation results."""

    model_config = ConfigDict(extra="forbid")

    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def blocking_issues(self) -> list[ValidationIssue]:
        return [
            issue for issue in self.issues if issue.severity is IssueSeverity.BLOCKING
        ]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [
            issue for issue in self.issues if issue.severity is IssueSeverity.WARNING
        ]

    @computed_field
    @property
    def passed(self) -> bool:
        return not self.blocking_issues


class ReviewOutput(BaseModel):
    """Structured output returned by the LLM reviewer."""

    model_config = ConfigDict(extra="forbid")

    issues: list[ValidationIssue] = Field(default_factory=list)


def _issue(
    code: str,
    message: str,
    severity: IssueSeverity,
    *,
    section: str | None = None,
    suggestion: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        message=message,
        severity=severity,
        section=section,
        suggestion=suggestion,
    )


def _normalize_heading(heading: str) -> str:
    return " ".join(heading.split()).casefold()


def _iter_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


def _headings(content: DocumentContent) -> list[str]:
    return [section.heading for section in content.sections]


def _has_heading(content: DocumentContent, keywords: tuple[str, ...]) -> bool:
    headings = " ".join(_normalize_heading(heading) for heading in _headings(content))
    return any(keyword in headings for keyword in keywords)


def validate_document_structure(
    content: DocumentContent,
    plan: DocumentPlan,
) -> list[ValidationIssue]:
    """Validate structure, plan consistency, and type-specific content rules."""
    issues: list[ValidationIssue] = []

    if content.document_type != plan.document_type:
        issues.append(
            _issue(
                "document_type_mismatch",
                "Generated content does not match the approved document type.",
                IssueSeverity.BLOCKING,
                suggestion="Regenerate content using the approved document type.",
            )
        )

    content_headings = [_normalize_heading(h) for h in _headings(content)]
    plan_headings = [_normalize_heading(section.heading) for section in plan.sections]
    if content_headings != plan_headings:
        issues.append(
            _issue(
                "section_structure_mismatch",
                "Generated section headings do not match the approved plan order.",
                IssueSeverity.BLOCKING,
                suggestion="Restore the approved section headings and ordering.",
            )
        )

    for section in content.sections:
        if not section.paragraphs and not section.bullets:
            issues.append(
                _issue(
                    "empty_section",
                    "A generated section contains no paragraphs or bullets.",
                    IssueSeverity.BLOCKING,
                    section=section.heading,
                    suggestion="Add substantive content or an explicit placeholder.",
                )
            )

    document_type = content.document_type
    if (
        content.meeting is not None
        and document_type is not DocumentType.MEETING_MINUTES
    ):
        issues.append(
            _issue(
                "unexpected_meeting_metadata",
                "Meeting metadata is present for a non-meeting document.",
                IssueSeverity.WARNING,
            )
        )
    if content.action_items and document_type is not DocumentType.MEETING_MINUTES:
        issues.append(
            _issue(
                "unexpected_action_items",
                "Meeting action items are present for a non-meeting document.",
                IssueSeverity.WARNING,
            )
        )
    if content.milestones and document_type not in (
        DocumentType.PROJECT_PLAN,
        DocumentType.BUSINESS_PROPOSAL,
    ):
        issues.append(
            _issue(
                "unexpected_milestones",
                "Milestones are present for a document type that does not use them.",
                IssueSeverity.WARNING,
            )
        )
    if content.sop_steps and document_type is not DocumentType.SOP:
        issues.append(
            _issue(
                "unexpected_sop_steps",
                "Procedure steps are present for a non-SOP document.",
                IssueSeverity.WARNING,
            )
        )
    if content.requirements and document_type is not DocumentType.PRODUCT_SPECIFICATION:
        issues.append(
            _issue(
                "unexpected_requirements",
                "Product requirements are present for a non-product document.",
                IssueSeverity.WARNING,
            )
        )
    if content.components and document_type is not DocumentType.TECHNICAL_DESIGN:
        issues.append(
            _issue(
                "unexpected_components",
                "Technical components are present for a non-technical document.",
                IssueSeverity.WARNING,
            )
        )

    if document_type is DocumentType.MEETING_MINUTES:
        if content.meeting is None:
            issues.append(
                _issue(
                    "meeting_metadata_missing",
                    "Meeting date, time, location, or attendees are not represented.",
                    IssueSeverity.WARNING,
                    suggestion="Add meeting metadata or explicit placeholders.",
                )
            )
        elif not content.meeting.attendees:
            issues.append(
                _issue(
                    "meeting_attendees_missing",
                    "No attendees are listed for the meeting.",
                    IssueSeverity.WARNING,
                )
            )
        if not content.action_items:
            issues.append(
                _issue(
                    "meeting_action_items_missing",
                    "No action items are recorded for the meeting.",
                    IssueSeverity.WARNING,
                )
            )
        if not _has_heading(content, ("decision",)):
            issues.append(
                _issue(
                    "meeting_decisions_missing",
                    "No decisions section is present in the meeting minutes.",
                    IssueSeverity.WARNING,
                )
            )

    if document_type is DocumentType.PROJECT_PLAN:
        if not content.milestones:
            issues.append(
                _issue(
                    "project_milestones_missing",
                    "No milestones or timeline entries are present.",
                    IssueSeverity.WARNING,
                    suggestion="Add milestones or explicit timeline placeholders.",
                )
            )
        else:
            for milestone in content.milestones:
                if milestone.target_date is None:
                    issues.append(
                        _issue(
                            "milestone_date_missing",
                            "A milestone has no target date.",
                            IssueSeverity.WARNING,
                            section=milestone.name,
                        )
                    )
        if not _has_heading(content, ("milestone", "timeline", "schedule")):
            issues.append(
                _issue(
                    "project_timeline_section_missing",
                    "No timeline or milestone section is present.",
                    IssueSeverity.WARNING,
                )
            )

    if document_type is DocumentType.BUSINESS_PROPOSAL:
        if not _has_heading(content, ("deliverable", "scope")):
            issues.append(
                _issue(
                    "proposal_deliverables_missing",
                    "No scope or deliverables section is present.",
                    IssueSeverity.WARNING,
                )
            )
        if not _has_heading(content, ("approach", "implementation", "solution")):
            issues.append(
                _issue(
                    "proposal_approach_missing",
                    "No implementation or solution approach is present.",
                    IssueSeverity.WARNING,
                )
            )
        if not content.milestones:
            issues.append(
                _issue(
                    "proposal_timeline_missing",
                    "No timeline information is available for the proposal.",
                    IssueSeverity.WARNING,
                )
            )

    if document_type is DocumentType.SOP and not content.sop_steps:
        issues.append(
            _issue(
                "sop_steps_missing",
                "The SOP contains no numbered procedure steps.",
                IssueSeverity.BLOCKING,
                suggestion="Add ordered procedure steps, using placeholders if needed.",
            )
        )

    if document_type is DocumentType.PRODUCT_SPECIFICATION:
        if not content.requirements:
            issues.append(
                _issue(
                    "product_requirements_missing",
                    "The product specification contains no requirements.",
                    IssueSeverity.BLOCKING,
                    suggestion="Add functional or non-functional requirements.",
                )
            )
        else:
            for requirement in content.requirements:
                if requirement.acceptance_criteria is None:
                    issues.append(
                        _issue(
                            "requirement_acceptance_missing",
                            "A requirement has no acceptance criteria.",
                            IssueSeverity.WARNING,
                            section=requirement.description,
                        )
                    )

    if document_type is DocumentType.TECHNICAL_DESIGN:
        if not content.components:
            issues.append(
                _issue(
                    "technical_components_missing",
                    "The technical design contains no components.",
                    IssueSeverity.BLOCKING,
                    suggestion="Add components, responsibilities and interfaces.",
                )
            )
        else:
            for component in content.components:
                if not component.interfaces:
                    issues.append(
                        _issue(
                            "component_interfaces_missing",
                            "A technical component has no interfaces.",
                            IssueSeverity.WARNING,
                            section=component.name,
                        )
                    )

    text_values = list(_iter_strings(content.model_dump()))
    if any(_EMPTY_PLACEHOLDER_RE.search(value) for value in text_values):
        issues.append(
            _issue(
                "empty_placeholder",
                "An empty placeholder was found.",
                IssueSeverity.WARNING,
                suggestion="Replace empty brackets with a descriptive placeholder.",
            )
        )
    if plan.missing_information and not any(
        _PLACEHOLDER_RE.search(value) for value in text_values
    ):
        issues.append(
            _issue(
                "placeholders_not_used",
                "The plan identifies missing information but no explicit "
                "placeholders appear in the content.",
                IssueSeverity.WARNING,
                suggestion="Use bracketed placeholders for unknown details.",
            )
        )

    return issues


_REVIEW_SCHEMA = json.dumps(ReviewOutput.model_json_schema(), indent=2)

REVIEW_SYSTEM_PROMPT = f"""You review a generated business document before delivery.

Return only a JSON object matching this schema:
{_REVIEW_SCHEMA}

Evaluate:
- Alignment with the original request.
- Completeness against the approved document plan, including planned sections
  and important key points.
- Logical consistency, contradictions between sections, and unsupported factual
  claims.
- Whether assumptions are presented as confirmed facts.
- Whether missing details use appropriate bracketed placeholders.
- Whether the selected document type and structure are appropriate.

Evidence rules:
- Use the original request and tool results as evidence for factual claims.
- Do not blindly trust facts supplied by the planning model.
- Treat planning assumptions as assumptions, not confirmed facts.
- Do not demand external factual verification; no external research tool exists.
- Distinguish factual claims from proposed recommendations and clearly labelled
  assumptions.
- Do not penalize legitimate placeholders for missing information.
- Use blocking severity only for material problems that should prevent delivery.
- Use warning severity for non-blocking improvements.
"""


class ReviewError(RuntimeError):
    """Raised when the LLM review cannot produce valid structured output."""


class DocumentReviewer:
    """LLM review of generated content against the request and plan."""

    def __init__(self, model: BaseChatModel):
        self._structured_model = model.with_structured_output(
            ReviewOutput,
            method="json_mode",
            include_raw=False,
        )

    async def review(
        self,
        request: str,
        plan: DocumentPlan,
        content: DocumentContent,
        tool_results: Sequence[str] = (),
    ) -> ReviewOutput:
        try:
            result = await self._structured_model.ainvoke(
                build_review_messages(request, plan, content, tool_results)
            )
        except Exception as exc:
            raise ReviewError(
                "The review model could not produce a valid review."
            ) from exc

        if isinstance(result, ReviewOutput):
            return result
        try:
            return ReviewOutput.model_validate(result)
        except Exception as exc:
            raise ReviewError(
                "The review model returned invalid structured review output."
            ) from exc


def _format_items(items: Sequence[str]) -> str:
    if not items:
        return "- None provided."
    return "\n".join(f"- {item}" for item in items)


def build_review_messages(
    request: str,
    plan: DocumentPlan,
    content: DocumentContent,
    tool_results: Sequence[str],
) -> list[BaseMessage]:
    return [
        SystemMessage(content=REVIEW_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Original request:\n{request}\n\n"
                f"Approved plan:\n{plan.model_dump_json(indent=2)}\n\n"
                f"Generated content:\n{content.model_dump_json(indent=2)}\n\n"
                f"Tool results:\n{_format_items(tool_results)}"
            )
        ),
    ]


class DocumentValidator:
    """Run deterministic checks and, when needed, LLM review."""

    def __init__(self, reviewer: DocumentReviewer | None = None):
        self._reviewer = reviewer

    async def validate(
        self,
        request: str,
        plan: DocumentPlan,
        content: DocumentContent,
        tool_results: Sequence[str] = (),
    ) -> ValidationReport:
        issues = validate_document_structure(content, plan)
        deterministic_report = ValidationReport(issues=issues)
        if deterministic_report.blocking_issues or self._reviewer is None:
            return deterministic_report

        review = await self._reviewer.review(request, plan, content, tool_results)
        return ValidationReport(issues=[*issues, *review.issues])


def build_document_reviewer(model: BaseChatModel | None = None) -> DocumentReviewer:
    return DocumentReviewer(model or build_chat_model())


def build_document_validator(
    reviewer: DocumentReviewer | None = None,
) -> DocumentValidator:
    return DocumentValidator(reviewer or build_document_reviewer())
