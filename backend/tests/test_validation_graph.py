"""Graph tests for validation, review failures, and bounded revision."""

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from backend.app.agent import (
    ContentGenerationError,
    ContentSection,
    DocumentContent,
    DocumentPlan,
    DocumentType,
    GeneratedDocument,
    PlanSection,
    ReviewError,
    ValidationIssue,
    ValidationReport,
    run_agent,
)
from backend.app.agent.graph import (
    REVIEW_FAILURE_RESPONSE,
    REVISION_FAILURE_RESPONSE,
    VALIDATION_FAILURE_RESPONSE,
)
from backend.app.agent.validation import MAX_REVISION_ATTEMPTS, IssueSeverity
from backend.tests.plan_factory import make_execution_tasks


def _plan() -> DocumentPlan:
    return DocumentPlan(
        execution_tasks=make_execution_tasks(),
        document_type=DocumentType.BUSINESS_REPORT,
        title="Status Report",
        objective="Report progress.",
        sections=[
            PlanSection(heading="Overview", purpose="Introduce the report."),
            PlanSection(heading="Findings", purpose="State findings."),
        ],
    )


def _content(title: str = "Status Report") -> DocumentContent:
    plan = _plan()
    return DocumentContent(
        document_type=plan.document_type,
        title=title,
        sections=[
            ContentSection(heading=section.heading, paragraphs=["Content."])
            for section in plan.sections
        ],
    )


def _blocking_report() -> ValidationReport:
    return ValidationReport(
        issues=[
            ValidationIssue(
                code="missing_evidence",
                message="Findings lack evidence.",
                severity=IssueSeverity.BLOCKING,
                suggestion="Add the confirmed result.",
            )
        ]
    )


class FakePlanner:
    async def create_plan(self, request: str) -> DocumentPlan:
        return _plan()


class NoToolSelector:
    async def select(self, messages: list[BaseMessage]) -> AIMessage:
        return AIMessage(content="No tool needed.")


class SequenceValidator:
    def __init__(self, reports: list[ValidationReport]) -> None:
        self._reports = list(reports)
        self.calls: list[tuple[DocumentContent, DocumentPlan]] = []

    async def validate(
        self,
        request: str,
        plan: DocumentPlan,
        content: DocumentContent,
        tool_results: list[str] | tuple[str, ...] = (),
    ) -> ValidationReport:
        self.calls.append((content, plan))
        return self._reports.pop(0)


class RaisingValidator:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def validate(
        self,
        request: str,
        plan: DocumentPlan,
        content: DocumentContent,
        tool_results: list[str] | tuple[str, ...] = (),
    ) -> ValidationReport:
        raise self.error


class RevisionWriter:
    def __init__(
        self,
        revised: DocumentContent | None = None,
        error: Exception | None = None,
    ) -> None:
        self.revised = revised or _content()
        self.error = error
        self.revise_calls: list[tuple[DocumentContent, list[ValidationIssue]]] = []

    async def generate(
        self,
        request: str,
        plan: DocumentPlan,
        tool_results: list[str] | tuple[str, ...] = (),
    ) -> DocumentContent:
        return _content()

    async def revise(
        self,
        request: str,
        plan: DocumentPlan,
        content: DocumentContent,
        blocking_issues: list[ValidationIssue],
        tool_results: list[str] | tuple[str, ...] = (),
    ) -> DocumentContent:
        self.revise_calls.append((content, list(blocking_issues)))
        if self.error is not None:
            raise self.error
        return self.revised


class RecordingDocx:
    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        content: DocumentContent,
        request: str,
        plan: DocumentPlan | None = None,
        today: object | None = None,
    ) -> GeneratedDocument:
        self.calls += 1
        return GeneratedDocument(
            filename="validated-000000000000.docx",
            document_date="2026-11-07",
            document_version="1.0",
        )


def _run(
    *,
    validator: Any,
    writer: Any = None,
    docx_generator: Any = None,
) -> Any:
    return run_agent(
        "Create a status report",
        planner=FakePlanner(),
        tool_selector=NoToolSelector(),
        writer=writer or RevisionWriter(),
        docx_generator=docx_generator or RecordingDocx(),
        validator=validator,
    )


async def test_passed_validation_generates_docx() -> None:
    validator = SequenceValidator([ValidationReport()])
    docx = RecordingDocx()

    result = await _run(validator=validator, docx_generator=docx)

    assert result["status"] == "completed"
    assert result["validation_report"] is not None
    assert result["validation_report"].passed is True
    assert result["revision_count"] == 0
    assert docx.calls == 1
    assert result["output_filename"] == "validated-000000000000.docx"


async def test_warnings_alone_do_not_trigger_revision() -> None:
    warning_report = ValidationReport(
        issues=[
            ValidationIssue(
                code="style_warning",
                message="Minor style issue.",
                severity=IssueSeverity.WARNING,
            )
        ]
    )
    validator = SequenceValidator([warning_report])
    writer = RevisionWriter()

    result = await _run(validator=validator, writer=writer)

    assert result["status"] == "completed"
    assert result["revision_count"] == 0
    assert writer.revise_calls == []


async def test_blocking_issue_revises_then_generates_docx() -> None:
    validator = SequenceValidator([_blocking_report(), ValidationReport()])
    writer = RevisionWriter()
    docx = RecordingDocx()

    result = await _run(validator=validator, writer=writer, docx_generator=docx)

    assert result["status"] == "completed"
    assert result["revision_count"] == 1
    assert len(writer.revise_calls) == 1
    assert writer.revise_calls[0][1][0].code == "missing_evidence"
    assert len(validator.calls) == 2
    assert docx.calls == 1


async def test_revision_budget_is_bounded_and_blocks_docx() -> None:
    validator = SequenceValidator(
        [_blocking_report() for _ in range(MAX_REVISION_ATTEMPTS + 1)]
    )
    writer = RevisionWriter()
    docx = RecordingDocx()

    result = await _run(validator=validator, writer=writer, docx_generator=docx)

    assert result["status"] == "failed"
    assert result["final_response"] == VALIDATION_FAILURE_RESPONSE
    assert result["revision_count"] == MAX_REVISION_ATTEMPTS
    assert len(writer.revise_calls) == MAX_REVISION_ATTEMPTS
    assert len(validator.calls) == MAX_REVISION_ATTEMPTS + 1
    assert docx.calls == 0
    assert result["output_filename"] == ""


async def test_review_failure_routes_to_safe_failure() -> None:
    validator = RaisingValidator(ReviewError("review unavailable"))
    docx = RecordingDocx()

    result = await _run(validator=validator, docx_generator=docx)

    assert result["status"] == "failed"
    assert result["final_response"] == REVIEW_FAILURE_RESPONSE
    assert result["validation_report"] is None
    assert docx.calls == 0


async def test_revision_failure_routes_to_safe_failure() -> None:
    validator = SequenceValidator([_blocking_report()])
    writer = RevisionWriter(error=ContentGenerationError("revision failed"))
    docx = RecordingDocx()

    result = await _run(validator=validator, writer=writer, docx_generator=docx)

    assert result["status"] == "failed"
    assert result["final_response"] == REVISION_FAILURE_RESPONSE
    assert result["revision_count"] == 1
    assert docx.calls == 0


async def test_validation_state_is_request_local() -> None:
    first = await _run(validator=SequenceValidator([ValidationReport()]))
    second = await _run(validator=SequenceValidator([ValidationReport()]))

    assert first["validation_report"] is not second["validation_report"]
    assert first["content"] is not second["content"]
    assert first["messages"] is not second["messages"]

    first["messages"].append(AIMessage(content="mutated"))
    assert len(second["messages"]) == 3
