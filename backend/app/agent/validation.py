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
from backend.app.agent.placeholders import TEMPLATE_PLACEHOLDER_REASONS
from backend.app.services.llm import build_chat_model

MAX_REVISION_ATTEMPTS = 2

_PLACEHOLDER_RE = re.compile(r"\[[^\[\]]+\]")
_EMPTY_PLACEHOLDER_RE = re.compile(r"\[\s*\]")

_CRITICAL_REVIEWER_TERMS = (
    "unsupported",
    "invent",
    "fabricat",
    "contradict",
    "assumption_as_confirmed",
    "presented_as_confirmed",
    "presented_as_assigned",
    "materially change",
    "wrong_document_type",
    "empty_document",
    "structurally unusable",
    "malformed",
)
_WARNING_REVIEWER_TERMS = (
    "incomplete",
    "missing",
    "partial",
    "optional",
    "timeline",
    "duration",
    "phase",
    "milestone",
    "responsib",
    "owner",
    "ownership",
    "assign",
    "dependency",
    "risk",
    "scope",
    "success",
    "approach",
    "deliverable",
    "quality",
    "recommendation",
    "placeholder",
    "assumption",
    "govern",
    "raci",
    "change_management",
    "stabilisation",
    "accelerat",
    "budget",
    "misclassified",
    "not_stated",
    "not_defined",
    "not_listed",
    "not_provided",
)


class IssueSeverity(StrEnum):
    BLOCKING = "blocking"
    WARNING = "warning"
    INFO = "info"


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

    @property
    def informational(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity is IssueSeverity.INFO]

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


def _has_structured_content(content: DocumentContent) -> bool:
    """Return True when type-specific structured fields carry real content."""
    meeting = content.meeting
    meeting_has_data = meeting is not None and any(
        (
            meeting.date,
            meeting.time,
            meeting.location,
            meeting.facilitator,
            meeting.attendees,
        )
    )
    return bool(
        meeting_has_data
        or content.action_items
        or content.milestones
        or content.sop_steps
        or content.requirements
        or content.components
    )


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
                IssueSeverity.WARNING,
                suggestion="Keep the useful content and warn about plan divergence.",
            )
        )

    # A planned section may legitimately have no body when its information is
    # represented by structured fields (for example meeting details or action
    # items) or when optional content is omitted. Only a document with no
    # substantive content anywhere is a blocking structural failure.
    has_section_content = any(
        section.paragraphs or section.bullets for section in content.sections
    )
    if not has_section_content and not _has_structured_content(content):
        issues.append(
            _issue(
                "empty_document",
                "The generated document contains no substantive content.",
                IssueSeverity.BLOCKING,
                suggestion=(
                    "Add supported content, justified placeholders, or populated "
                    "structured fields; do not add filler."
                ),
            )
        )

    for section in content.sections:
        if not section.paragraphs and not section.bullets:
            issues.append(
                _issue(
                    "empty_section",
                    "A generated section has no paragraph or bullet.",
                    IssueSeverity.WARNING,
                    section=section.heading,
                    suggestion=(
                        "Keep the heading and omit optional content, or add a "
                        "justified placeholder for required missing information."
                    ),
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
    all_declared_placeholders = {
        placeholder.text
        for plan_section in plan.sections
        for placeholder in plan_section.placeholders
    }
    for section in content.sections:
        for text in [*section.paragraphs, *section.bullets]:
            for placeholder in _PLACEHOLDER_RE.findall(text):
                if placeholder not in all_declared_placeholders:
                    issues.append(
                        _issue(
                            "unexplained_placeholder",
                            f"Placeholder {placeholder} is not declared in the "
                            "approved plan.",
                            IssueSeverity.WARNING,
                            section=section.heading,
                            suggestion=(
                                "Keep it as a transparent placeholder or replace "
                                "it with supported content."
                            ),
                        )
                    )
    structured_values = content.model_dump(
        include={
            "meeting",
            "action_items",
            "milestones",
            "sop_steps",
            "requirements",
            "components",
        }
    )
    seen_unexplained: set[str] = set()
    for value in _iter_strings(structured_values):
        for placeholder in _PLACEHOLDER_RE.findall(value):
            if (
                placeholder in all_declared_placeholders
                or placeholder in TEMPLATE_PLACEHOLDER_REASONS
                or placeholder in seen_unexplained
            ):
                continue
            seen_unexplained.add(placeholder)
            issues.append(
                _issue(
                    "unexplained_placeholder",
                    f"Placeholder {placeholder} is not declared in the approved plan.",
                    IssueSeverity.WARNING,
                    suggestion=(
                        "Keep it as a transparent placeholder or replace it with "
                        "supported content."
                    ),
                )
            )

    metadata_values = {
        "audience": content.audience,
        "metadata": (
            content.metadata.model_dump() if content.metadata is not None else None
        ),
    }
    seen_metadata_placeholders: set[str] = set()
    for value in _iter_strings(metadata_values):
        for placeholder in _PLACEHOLDER_RE.findall(value):
            if (
                placeholder in TEMPLATE_PLACEHOLDER_REASONS
                or placeholder in seen_metadata_placeholders
            ):
                continue
            seen_metadata_placeholders.add(placeholder)
            issues.append(
                _issue(
                    "unexplained_placeholder",
                    f"Placeholder {placeholder} is not valid in document metadata.",
                    IssueSeverity.WARNING,
                    suggestion=(
                        "Use a deterministic metadata placeholder such as "
                        "[Audience], [Date] or [Version], or remove it."
                    ),
                )
            )

    for placeholder in dict.fromkeys(_PLACEHOLDER_RE.findall(content.title)):
        issues.append(
            _issue(
                "unexplained_placeholder",
                f"Placeholder {placeholder} is not valid in the document title.",
                IssueSeverity.WARNING,
                suggestion="Use a literal title or keep the placeholder transparent.",
            )
        )

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
- Invented names, organisations, people, roles, locations, dates, financial
  figures, commitments, policies, targets or technical specifications.
- Assumptions, conflict resolutions or planner suggestions presented as
  confirmed facts.
- Recommendations or proposed content that the user did not request.
- Statements that materially change the meaning of the original request.
- Whether missing details use appropriate bracketed placeholders.
- Whether the selected document type and structure are appropriate.

Evidence rules:
- The original user request is the primary source of factual information.
- Tool results may support only values they deterministically produced.
- Do not blindly trust facts supplied by the planning model.
- Treat planning assumptions and conflict resolutions as non-facts.
- Do not demand external factual verification; no external research tool exists.
- Distinguish factual claims from proposed recommendations and clearly labelled
  assumptions.
- Do not penalize legitimate placeholders for missing information.
- A placeholder declared anywhere in the approved plan may be reused in any
  section. An undeclared bracketed placeholder is a transparent warning unless
  it replaces a required fact with invented content.
- Clearly labelled assumptions and disclosed unknowns are not fabricated facts.

Severity policy:
- Use blocking severity only for material integrity failures: fabricated names,
  organisations, people, dates, figures, commitments, decisions, policies,
  targets or technical specifications presented as confirmed facts; unsupported
  factual claims; material contradictions with explicitly supplied facts;
  serious internal contradictions that materially change meaning; wrong
  document type; empty or structurally unusable output.
- Use warning severity for incomplete timelines, durations, phases or milestones;
  missing owner identities or partially specified responsibilities; unresolved
  but transparently disclosed conflicts; missing non-critical information;
  optional or partially completed sections; missing technical component detail
  when useful architecture content exists; placeholder consistency or
  completeness issues; and quality or completeness recommendations.
- Use info severity for optional omissions and non-essential suggestions.
- When the request asks for a practical phased rollout but calendar dates are
  unavailable, a relative indicative schedule with phases, durations,
  dependencies, milestones and clearly labelled assumptions is meaningful
  timeline coverage. Do not block solely because calendar dates are missing.
- Do not penalize an empty section body when its information is fully represented
  by populated structured fields or when optional content is intentionally omitted.
"""


class ReviewError(RuntimeError):
    """Raised when the LLM review cannot produce valid structured output."""


def review_unavailable_issue() -> ValidationIssue:
    """Return the non-blocking warning used when automated review is unavailable."""
    return _issue(
        "review_unavailable",
        "Automated review was unavailable; deterministic checks passed.",
        IssueSeverity.WARNING,
        suggestion="Review the document manually if factual assurance is required.",
    )


class DocumentReviewer:
    """LLM review of generated content against the request and plan."""

    MAX_REVIEW_ATTEMPTS = 2

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
        messages = build_review_messages(request, plan, content, tool_results)
        last_error: Exception | None = None
        invalid_output = False

        for _ in range(self.MAX_REVIEW_ATTEMPTS):
            try:
                result = await self._structured_model.ainvoke(messages)
            except Exception as exc:
                last_error = exc
                invalid_output = False
                continue

            if isinstance(result, ReviewOutput):
                return result

            try:
                return ReviewOutput.model_validate(result)
            except Exception as exc:
                last_error = exc
                invalid_output = True

        if invalid_output:
            raise ReviewError(
                "The review model returned invalid structured review output."
            ) from last_error
        raise ReviewError(
            "The review model could not produce a valid review."
        ) from last_error


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


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    normalized = text.casefold()
    return any(term in normalized for term in terms)


def normalize_reviewer_severity(
    issues: list[ValidationIssue],
) -> list[ValidationIssue]:
    """Normalize reviewer severity using the critical/warning policy.

    Critical factual integrity terms always block. Clear quality/completeness
    findings are warnings. Unknown blocking codes stay blocking to avoid
    silently weakening an unrecognized safety check.
    """
    normalized: list[ValidationIssue] = []
    for issue in issues:
        text = f"{issue.code} {issue.message}".casefold()
        if _contains_any(text, _CRITICAL_REVIEWER_TERMS):
            adjusted = issue.model_copy(update={"severity": IssueSeverity.BLOCKING})
        elif issue.severity is IssueSeverity.BLOCKING and _contains_any(
            text, _WARNING_REVIEWER_TERMS
        ):
            adjusted = issue.model_copy(update={"severity": IssueSeverity.WARNING})
        else:
            adjusted = issue
        normalized.append(adjusted)
    return normalized


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

        try:
            review = await self._reviewer.review(
                request,
                plan,
                content,
                tool_results,
            )
        except ReviewError:
            issues.append(review_unavailable_issue())
            return ValidationReport(issues=issues)

        review_issues = normalize_reviewer_severity(review.issues)
        return ValidationReport(issues=[*issues, *review_issues])


def build_document_reviewer(model: BaseChatModel | None = None) -> DocumentReviewer:
    return DocumentReviewer(model or build_chat_model())


def build_document_validator(
    reviewer: DocumentReviewer | None = None,
) -> DocumentValidator:
    return DocumentValidator(reviewer or build_document_reviewer())
