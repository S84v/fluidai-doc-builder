"""Tests for deterministic validation and LLM document review."""

from typing import Any

import pytest
from langchain_core.messages import BaseMessage

from backend.app.agent import (
    ActionItem,
    ContentSection,
    DocumentContent,
    DocumentPlan,
    DocumentType,
    MeetingDetails,
    PlanPlaceholder,
    PlanSection,
    ReviewError,
    ReviewOutput,
    ValidationIssue,
    ValidationReport,
)
from backend.app.agent.validation import (
    DocumentReviewer,
    DocumentValidator,
    IssueSeverity,
    build_review_messages,
    validate_document_structure,
)
from backend.tests.plan_factory import make_execution_tasks


class FakeRunnable:
    def __init__(self, handler: Any) -> None:
        self._handler = handler

    async def ainvoke(self, messages: Any) -> Any:
        return await self._handler(messages)


class FakeStructuredModel:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.schema: Any = None
        self.method: str | None = None
        self.calls: list[list[BaseMessage]] = []

    def with_structured_output(
        self,
        schema: Any,
        *,
        method: str = "function_calling",
        include_raw: bool = False,
    ) -> FakeRunnable:
        self.schema = schema
        self.method = method

        async def ainvoke(messages: list[BaseMessage]) -> Any:
            self.calls.append(messages)
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

        return FakeRunnable(ainvoke)


class FakeReviewer:
    def __init__(
        self,
        output: ReviewOutput | None = None,
        error: Exception | None = None,
    ) -> None:
        self.output = output or ReviewOutput()
        self.error = error
        self.calls: list[tuple[str, DocumentPlan, DocumentContent, list[str]]] = []

    async def review(
        self,
        request: str,
        plan: DocumentPlan,
        content: DocumentContent,
        tool_results: list[str] | tuple[str, ...] = (),
    ) -> ReviewOutput:
        self.calls.append((request, plan, content, list(tool_results)))
        if self.error is not None:
            raise self.error
        return self.output


def _plan(
    document_type: DocumentType = DocumentType.BUSINESS_REPORT,
    headings: tuple[str, ...] = ("Overview",),
    *,
    missing_information: list[str] | None = None,
) -> DocumentPlan:
    return DocumentPlan(
        execution_tasks=make_execution_tasks(),
        document_type=document_type,
        title="Test Document",
        objective="Test the document.",
        sections=[
            PlanSection(heading=heading, purpose=f"Purpose of {heading}.")
            for heading in headings
        ],
        missing_information=missing_information or [],
    )


def _content(
    plan: DocumentPlan,
    headings: tuple[str, ...] | None = None,
    **kwargs: Any,
) -> DocumentContent:
    document_type = kwargs.pop("document_type", plan.document_type)
    return DocumentContent(
        document_type=document_type,
        title=plan.title,
        sections=[
            ContentSection(heading=heading, paragraphs=[f"{heading} content."])
            for heading in (headings or tuple(s.heading for s in plan.sections))
        ],
        **kwargs,
    )


def _codes(report: ValidationReport) -> set[str]:
    return {issue.code for issue in report.issues}


def test_validation_report_derives_pass_status_from_blocking_issues() -> None:
    warning = ValidationIssue(
        code="warning",
        message="Minor issue",
        severity=IssueSeverity.WARNING,
    )
    blocking = ValidationIssue(
        code="blocking",
        message="Blocking issue",
        severity=IssueSeverity.BLOCKING,
    )

    assert ValidationReport(issues=[warning]).passed is True
    assert ValidationReport(issues=[warning, blocking]).passed is False


def test_document_type_and_section_mismatches_are_blocking() -> None:
    plan = _plan(headings=("Overview", "Findings"))
    wrong_type = _content(plan, document_type=DocumentType.MEETING_MINUTES)
    wrong_order = _content(plan, headings=("Findings", "Overview"))

    type_report = ValidationReport(issues=validate_document_structure(wrong_type, plan))
    order_report = ValidationReport(
        issues=validate_document_structure(wrong_order, plan)
    )

    assert type_report.passed is False
    assert "document_type_mismatch" in _codes(type_report)
    assert order_report.passed is False
    assert "section_structure_mismatch" in _codes(order_report)


def test_document_with_only_empty_sections_is_blocking() -> None:
    plan = _plan(headings=("Overview",))
    content = DocumentContent(
        document_type=DocumentType.BUSINESS_REPORT,
        title="Test Document",
        sections=[ContentSection(heading="Overview")],
    )

    report = ValidationReport(issues=validate_document_structure(content, plan))

    assert report.passed is False
    assert "empty_document" in _codes(report)
    empty_section = next(
        issue for issue in report.issues if issue.code == "empty_section"
    )
    assert empty_section.severity is IssueSeverity.WARNING


def test_empty_section_is_warning_when_other_sections_have_content() -> None:
    plan = _plan(headings=("Overview", "Appendix"))
    content = DocumentContent(
        document_type=DocumentType.BUSINESS_REPORT,
        title="Test Document",
        sections=[
            ContentSection(heading="Overview", paragraphs=["Supported content."]),
            ContentSection(heading="Appendix"),
        ],
    )

    report = ValidationReport(issues=validate_document_structure(content, plan))

    assert report.passed is True
    assert "empty_section" in _codes(report)
    assert "empty_document" not in _codes(report)


def test_meeting_minutes_allow_empty_sections_for_structured_content() -> None:
    headings = ("Meeting Details", "Agenda", "Decisions", "Action Items")
    plan = _plan(DocumentType.MEETING_MINUTES, headings)
    content = DocumentContent(
        document_type=DocumentType.MEETING_MINUTES,
        title=plan.title,
        meeting=MeetingDetails(date="2026-11-06", attendees=["Anna", "Ben"]),
        action_items=[ActionItem(action="Send notes", owner="Anna")],
        sections=[ContentSection(heading=heading) for heading in headings],
    )

    report = ValidationReport(issues=validate_document_structure(content, plan))

    assert report.passed is True
    assert "empty_document" not in _codes(report)
    assert "empty_section" in _codes(report)


def test_required_section_with_only_a_justified_placeholder_is_accepted() -> None:
    plan = _plan(headings=("Budget",))
    plan.sections[0].placeholders = [
        PlanPlaceholder(
            text="[Budget]",
            reason="The approved budget was not supplied.",
        )
    ]
    content = _content(plan)
    content.sections[0].paragraphs = ["[Budget]"]

    report = ValidationReport(issues=validate_document_structure(content, plan))

    assert report.passed is True
    assert "empty_section" not in _codes(report)


def test_sop_requires_numbered_steps() -> None:
    plan = _plan(DocumentType.SOP, ("Procedure Steps",))
    report = ValidationReport(issues=validate_document_structure(_content(plan), plan))

    assert report.passed is False
    assert "sop_steps_missing" in _codes(report)


def test_product_spec_requires_requirements() -> None:
    plan = _plan(DocumentType.PRODUCT_SPECIFICATION, ("Functional Requirements",))
    report = ValidationReport(issues=validate_document_structure(_content(plan), plan))

    assert report.passed is False
    assert "product_requirements_missing" in _codes(report)


def test_technical_design_requires_components() -> None:
    plan = _plan(DocumentType.TECHNICAL_DESIGN, ("Components and Interfaces",))
    report = ValidationReport(issues=validate_document_structure(_content(plan), plan))

    assert report.passed is False
    assert "technical_components_missing" in _codes(report)


def test_meeting_minutes_warn_but_do_not_block_for_missing_details() -> None:
    plan = _plan(
        DocumentType.MEETING_MINUTES,
        ("Agenda", "Decisions", "Action Items"),
    )
    report = ValidationReport(issues=validate_document_structure(_content(plan), plan))

    assert report.passed is True
    assert "meeting_metadata_missing" in _codes(report)
    assert "meeting_action_items_missing" in _codes(report)
    assert "meeting_decisions_missing" not in _codes(report)


def test_project_plan_warns_for_missing_timeline_information() -> None:
    plan = _plan(DocumentType.PROJECT_PLAN, ("Project Overview", "Milestones"))
    report = ValidationReport(issues=validate_document_structure(_content(plan), plan))

    assert report.passed is True
    assert "project_milestones_missing" in _codes(report)


def test_proposal_warns_for_missing_deliverables_approach_and_timeline() -> None:
    plan = _plan(
        DocumentType.BUSINESS_PROPOSAL,
        ("Executive Summary", "Next Steps"),
    )
    report = ValidationReport(issues=validate_document_structure(_content(plan), plan))

    assert report.passed is True
    assert "proposal_deliverables_missing" in _codes(report)
    assert "proposal_approach_missing" in _codes(report)
    assert "proposal_timeline_missing" in _codes(report)


def test_inappropriate_structured_fields_are_warnings() -> None:
    plan = _plan(headings=("Overview",))
    content = _content(
        plan,
        action_items=[
            {"action": "Review", "owner": None, "due_date": None, "status": None}
        ],
    )
    report = ValidationReport(issues=validate_document_structure(content, plan))

    assert report.passed is True
    assert "unexpected_action_items" in _codes(report)


def test_placeholders_are_accepted_and_missing_placeholders_warn() -> None:
    plan = _plan(headings=("Overview",), missing_information=["Budget"])
    without_placeholder = ValidationReport(
        issues=validate_document_structure(_content(plan), plan)
    )
    with_placeholder = _content(plan)
    with_placeholder.sections[0].paragraphs = ["Budget: [Budget]"]
    with_placeholder_report = ValidationReport(
        issues=validate_document_structure(with_placeholder, plan)
    )

    assert "placeholders_not_used" in _codes(without_placeholder)
    assert "placeholders_not_used" not in _codes(with_placeholder_report)
    assert without_placeholder.passed is True


def test_unexplained_content_placeholder_is_blocking() -> None:
    plan = _plan(headings=("Overview",))
    content = _content(plan)
    content.sections[0].paragraphs = ["Unknown: [Mystery]"]

    report = ValidationReport(issues=validate_document_structure(content, plan))

    assert report.passed is False
    assert "unexplained_placeholder" in _codes(report)


def test_explained_content_placeholder_is_accepted() -> None:
    plan = _plan(headings=("Budget",))
    plan.sections[0].placeholders = [
        PlanPlaceholder(
            text="[Budget]",
            reason="The approved budget was not supplied.",
        )
    ]
    content = _content(plan)
    content.sections[0].paragraphs = ["Budget: [Budget]"]

    report = ValidationReport(issues=validate_document_structure(content, plan))

    assert report.passed is True
    assert "unexplained_placeholder" not in _codes(report)


def test_declared_structured_placeholder_is_accepted() -> None:
    plan = _plan(DocumentType.MEETING_MINUTES, ("Action Items",))
    plan.sections[0].placeholders = [
        PlanPlaceholder(
            text="[Budget]",
            reason="The approved budget was not supplied.",
        )
    ]
    content = _content(
        plan,
        action_items=[ActionItem(action="Review [Budget]")],
    )

    report = ValidationReport(issues=validate_document_structure(content, plan))

    assert report.passed is True
    assert "unexplained_placeholder" not in _codes(report)


def test_known_template_placeholder_in_structured_field_is_accepted() -> None:
    plan = _plan(DocumentType.MEETING_MINUTES, ("Action Items",))
    content = _content(
        plan,
        action_items=[ActionItem(action="Send notes", owner="[Owner]")],
    )

    report = ValidationReport(issues=validate_document_structure(content, plan))

    assert report.passed is True
    assert "unexplained_placeholder" not in _codes(report)


def test_plan_declared_placeholder_in_unrelated_metadata_is_blocking() -> None:
    plan = _plan(headings=("Budget",))
    plan.sections[0].placeholders = [
        PlanPlaceholder(
            text="[Budget]",
            reason="The approved budget was not supplied.",
        )
    ]
    content = _content(plan)
    content.audience = "[Budget]"

    report = ValidationReport(issues=validate_document_structure(content, plan))

    assert report.passed is False
    assert "unexplained_placeholder" in _codes(report)


def test_plan_declared_placeholder_in_title_is_blocking() -> None:
    plan = _plan(headings=("Budget",))
    plan.sections[0].placeholders = [
        PlanPlaceholder(
            text="[Budget]",
            reason="The approved budget was not supplied.",
        )
    ]
    content = _content(plan)
    content.title = "Report for [Budget]"

    report = ValidationReport(issues=validate_document_structure(content, plan))

    assert report.passed is False
    assert "unexplained_placeholder" in _codes(report)


def test_template_metadata_placeholder_is_accepted() -> None:
    plan = _plan(headings=("Overview",))
    content = _content(plan)
    content.audience = "[Audience]"

    report = ValidationReport(issues=validate_document_structure(content, plan))

    assert report.passed is True
    assert "unexplained_placeholder" not in _codes(report)


def test_undeclared_metadata_placeholder_is_blocking() -> None:
    plan = _plan(headings=("Overview",))
    content = _content(plan)
    content.audience = "[Mystery]"

    report = ValidationReport(issues=validate_document_structure(content, plan))

    assert report.passed is False
    assert "unexplained_placeholder" in _codes(report)


def test_placeholder_declared_in_section_a_is_not_allowed_in_section_b() -> None:
    plan = _plan(headings=("Alpha", "Beta"))
    plan.sections[0].placeholders = [
        PlanPlaceholder(text="[Budget]", reason="Alpha budget reason.")
    ]
    content = _content(plan)
    content.sections[0].paragraphs = ["Budget: [Budget]"]
    content.sections[1].paragraphs = ["Budget: [Budget]"]

    report = ValidationReport(issues=validate_document_structure(content, plan))

    assert report.passed is False
    unexplained = [
        issue for issue in report.issues if issue.code == "unexplained_placeholder"
    ]
    assert len(unexplained) == 1
    assert unexplained[0].section == "Beta"


def test_same_placeholder_declared_in_both_sections_is_accepted() -> None:
    plan = _plan(headings=("Alpha", "Beta"))
    plan.sections[0].placeholders = [
        PlanPlaceholder(text="[Budget]", reason="Alpha budget reason.")
    ]
    plan.sections[1].placeholders = [
        PlanPlaceholder(text="[Budget]", reason="Beta budget reason.")
    ]
    content = _content(plan)
    content.sections[0].paragraphs = ["Budget: [Budget]", "Again: [Budget]"]
    content.sections[1].paragraphs = ["Budget: [Budget]"]

    report = ValidationReport(issues=validate_document_structure(content, plan))

    assert report.passed is True
    assert "unexplained_placeholder" not in _codes(report)


def test_validator_skips_reviewer_when_deterministic_issues_block() -> None:
    plan = _plan(DocumentType.SOP, ("Procedure Steps",))
    content = _content(plan)
    reviewer = FakeReviewer()
    validator = DocumentValidator(reviewer)

    report = asyncio_run(validator.validate("request", plan, content))

    assert report.passed is False
    assert reviewer.calls == []


def test_validator_merges_deterministic_and_review_issues() -> None:
    plan = _plan(headings=("Overview",))
    content = _content(plan)
    review_output = ReviewOutput(
        issues=[
            ValidationIssue(
                code="unsupported_claim",
                message="The content claims an unsupported fact.",
                severity=IssueSeverity.BLOCKING,
            )
        ]
    )
    reviewer = FakeReviewer(review_output)
    validator = DocumentValidator(reviewer)

    report = asyncio_run(validator.validate("request", plan, content, ["tool result"]))

    assert report.passed is False
    assert "unsupported_claim" in _codes(report)
    assert reviewer.calls[0][3] == ["tool result"]


def test_reviewer_uses_json_mode_and_receives_evidence() -> None:
    plan = _plan(headings=("Overview",))
    content = _content(plan)
    output = ReviewOutput()
    model = FakeStructuredModel(output)
    reviewer = DocumentReviewer(model)

    result = asyncio_run(
        reviewer.review("Original request", plan, content, ["17 * 23 = 391"])
    )

    assert result == output
    assert model.schema is ReviewOutput
    assert model.method == "json_mode"
    prompt = model.calls[0][1].content
    assert "Original request" in prompt
    assert "17 * 23 = 391" in prompt
    system_prompt = model.calls[0][0].content
    assert "Do not blindly trust" in system_prompt
    assert "Do not penalize legitimate placeholders" in system_prompt
    assert "Do not penalize an empty section body" in system_prompt
    assert "original user request is the primary source" in system_prompt
    assert "Invented names, organisations" in system_prompt
    assert "Recommendations or proposed content" in system_prompt
    assert "materially change the meaning" in system_prompt
    assert "Treat planning assumptions and conflict resolutions as non-facts" in (
        system_prompt
    )


def test_reviewer_wraps_model_failure() -> None:
    cause = RuntimeError("provider unavailable")
    reviewer = DocumentReviewer(FakeStructuredModel(cause))
    plan = _plan(headings=("Overview",))

    with pytest.raises(ReviewError, match="could not produce") as exc_info:
        asyncio_run(reviewer.review("request", plan, _content(plan)))

    assert exc_info.value.__cause__ is cause


def test_reviewer_rejects_invalid_structured_output() -> None:
    reviewer = DocumentReviewer(FakeStructuredModel({"passed": True}))
    plan = _plan(headings=("Overview",))

    with pytest.raises(ReviewError, match="invalid structured review"):
        asyncio_run(reviewer.review("request", plan, _content(plan)))


def test_review_messages_exclude_internal_reasoning() -> None:
    plan = _plan(headings=("Overview",))
    content = _content(plan)

    messages = build_review_messages(
        "request",
        plan,
        content,
        ["tool result"],
    )

    assert "tool result" in messages[1].content
    assert "reasoning_content" not in messages[1].content


def asyncio_run(coroutine: Any) -> Any:
    import asyncio

    return asyncio.run(coroutine)
