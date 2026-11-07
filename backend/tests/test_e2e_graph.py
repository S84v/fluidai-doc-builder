"""End-to-end offline graph verification for all document types."""

from pathlib import Path

import pytest
from docx import Document
from langchain_core.messages import AIMessage

from backend.app.agent import (
    ActionItem,
    ContentGenerationError,
    ContentSection,
    DocumentContent,
    DocumentMetadata,
    DocumentPlan,
    DocumentType,
    DocxGenerationError,
    DocxGenerator,
    GeneratedDocument,
    MeetingDetails,
    Milestone,
    PlanningError,
    PlanSection,
    Requirement,
    ReviewError,
    SopStep,
    TechnicalComponent,
    ToolSelectionError,
    ValidationIssue,
    ValidationReport,
    run_agent,
)
from backend.app.agent.graph import (
    DOCUMENT_FAILURE_RESPONSE,
    PLANNING_FAILURE_RESPONSE,
    REJECTION_RESPONSE,
    REVIEW_FAILURE_RESPONSE,
    REVISION_FAILURE_RESPONSE,
    TOOL_FAILURE_RESPONSE,
    VALIDATION_FAILURE_RESPONSE,
)
from backend.app.agent.validation import (
    MAX_REVISION_ATTEMPTS,
    DocumentValidator,
    IssueSeverity,
)
from backend.tests.plan_factory import make_execution_tasks


class StaticPlanner:
    def __init__(self, plan: DocumentPlan, error: Exception | None = None) -> None:
        self.plan = plan
        self.error = error

    async def create_plan(self, request: str) -> DocumentPlan:
        if self.error is not None:
            raise self.error
        return self.plan


class NoToolSelector:
    async def select(self, messages: object) -> AIMessage:
        return AIMessage(content="No tool was required.")


class StaticWriter:
    def __init__(
        self,
        content: DocumentContent,
        generate_error: Exception | None = None,
        revise_error: Exception | None = None,
        revised: DocumentContent | None = None,
    ) -> None:
        self.content = content
        self.generate_error = generate_error
        self.revise_error = revise_error
        self.revised = revised or content

    async def generate(
        self,
        request: str,
        plan: DocumentPlan,
        tool_results: list[str] | tuple[str, ...] = (),
    ) -> DocumentContent:
        if self.generate_error is not None:
            raise self.generate_error
        return self.content

    async def revise(
        self,
        request: str,
        plan: DocumentPlan,
        content: DocumentContent,
        blocking_issues: list[ValidationIssue],
        tool_results: list[str] | tuple[str, ...] = (),
    ) -> DocumentContent:
        if self.revise_error is not None:
            raise self.revise_error
        return self.revised


class NoIssueReviewer:
    async def review(
        self,
        request: str,
        plan: DocumentPlan,
        content: DocumentContent,
        tool_results: list[str] | tuple[str, ...] = (),
    ) -> object:
        from backend.app.agent import ReviewOutput

        return ReviewOutput()


class SequenceValidator:
    def __init__(self, reports: list[ValidationReport]) -> None:
        self.reports = list(reports)
        self.calls = 0

    async def validate(
        self,
        request: str,
        plan: DocumentPlan,
        content: DocumentContent,
        tool_results: list[str] | tuple[str, ...] = (),
    ) -> ValidationReport:
        self.calls += 1
        return self.reports.pop(0)


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


class RecordingDocx:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    def generate(
        self,
        content: DocumentContent,
        request: str,
        plan: DocumentPlan | None = None,
        today: object | None = None,
    ) -> GeneratedDocument:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return GeneratedDocument(
            filename="unused-000000000000.docx",
            document_date="2026-11-07",
            document_version="1.0",
        )


def _plan(document_type: DocumentType, headings: tuple[str, ...]) -> DocumentPlan:
    return DocumentPlan(
        execution_tasks=make_execution_tasks(),
        document_type=document_type,
        title=f"{document_type.value.replace('_', ' ').title()} Test",
        objective="Exercise the complete offline graph.",
        sections=[
            PlanSection(heading=heading, purpose=f"Purpose of {heading}.")
            for heading in headings
        ],
    )


def _sections(headings: tuple[str, ...]) -> list[ContentSection]:
    return [
        ContentSection(
            heading=heading,
            paragraphs=[f"Content for {heading}."],
            bullets=["A supporting point."] if heading == "Next Steps" else [],
        )
        for heading in headings
    ]


def _case(document_type: DocumentType) -> tuple[DocumentPlan, DocumentContent]:
    if document_type is DocumentType.BUSINESS_PROPOSAL:
        headings = (
            "Executive Summary",
            "Scope and Deliverables",
            "Implementation Approach",
            "Timeline",
        )
        plan = _plan(document_type, headings)
        content = DocumentContent(
            document_type=document_type,
            title=plan.title,
            audience="Atlas Leadership",
            metadata=DocumentMetadata(date="2026-11-07", version="1.0"),
            sections=_sections(headings),
            milestones=[
                Milestone(
                    name="Phase one",
                    target_date="2026-12-01",
                    owner="[Owner]",
                    status="Planned",
                )
            ],
        )
        return plan, content

    if document_type is DocumentType.MEETING_MINUTES:
        headings = ("Agenda", "Decisions", "Action Items")
        plan = _plan(document_type, headings)
        content = DocumentContent(
            document_type=document_type,
            title=plan.title,
            meeting=MeetingDetails(
                date="2026-11-06",
                time="10:00",
                location="Room A",
                facilitator="Anna",
                attendees=["Anna", "Ben"],
            ),
            action_items=[
                ActionItem(
                    action="Send notes",
                    owner="[Owner]",
                    due_date="[Due date]",
                    status="Open",
                )
            ],
            sections=_sections(headings),
        )
        return plan, content

    if document_type is DocumentType.PROJECT_PLAN:
        headings = ("Project Overview", "Milestones and Timeline")
        plan = _plan(document_type, headings)
        content = DocumentContent(
            document_type=document_type,
            title=plan.title,
            sections=_sections(headings),
            milestones=[
                Milestone(
                    name="Phase one",
                    target_date="2026-12-01",
                    owner="Ben",
                    status="Planned",
                )
            ],
        )
        return plan, content

    if document_type is DocumentType.BUSINESS_REPORT:
        headings = ("Executive Summary", "Findings")
        plan = _plan(document_type, headings)
        content = DocumentContent(
            document_type=document_type,
            title=plan.title,
            audience="Operations Team",
            metadata=DocumentMetadata(date="2026-11-07"),
            sections=_sections(headings),
        )
        return plan, content

    if document_type is DocumentType.TECHNICAL_DESIGN:
        headings = ("Context and Objectives", "Components and Interfaces")
        plan = _plan(document_type, headings)
        content = DocumentContent(
            document_type=document_type,
            title=plan.title,
            sections=_sections(headings),
            components=[
                TechnicalComponent(
                    name="API",
                    responsibility="Serve document requests.",
                    interfaces=["REST"],
                )
            ],
        )
        return plan, content

    if document_type is DocumentType.SOP:
        headings = ("Purpose and Scope", "Procedure Steps")
        plan = _plan(document_type, headings)
        content = DocumentContent(
            document_type=document_type,
            title=plan.title,
            sections=_sections(headings),
            sop_steps=[
                SopStep(instruction="Open the system."),
                SopStep(instruction="Record the result.", note="Retain the log."),
            ],
        )
        return plan, content

    if document_type is DocumentType.PRODUCT_SPECIFICATION:
        headings = ("Functional Requirements",)
        plan = _plan(document_type, headings)
        content = DocumentContent(
            document_type=document_type,
            title=plan.title,
            sections=_sections(headings),
            requirements=[
                Requirement(
                    identifier="FR-1",
                    description="Generate a document.",
                    priority="High",
                    acceptance_criteria="A DOCX file is created.",
                )
            ],
        )
        return plan, content

    headings = ("Executive Overview", "Main Content")
    plan = _plan(document_type, headings)
    content = DocumentContent(
        document_type=document_type,
        title=plan.title,
        sections=_sections(headings),
    )
    return plan, content


def _template_checks(document_type: DocumentType, document: Document) -> None:
    tables = document.tables
    paragraphs = document.paragraphs
    table_values = {
        cell.text for table in tables for row in table.rows for cell in row.cells
    }

    assert "Document Date" in table_values
    assert "Document Version" in table_values

    if document_type is DocumentType.BUSINESS_PROPOSAL:
        assert tables
        assert any('w:type="page"' in p._p.xml for p in paragraphs)
        assert "Prepared For" in table_values
    elif document_type is DocumentType.MEETING_MINUTES:
        assert len(tables) >= 2
        assert "Meeting Date" in table_values
        assert "Action" in table_values
    elif document_type is DocumentType.PROJECT_PLAN:
        assert "Milestone" in table_values
    elif document_type is DocumentType.BUSINESS_REPORT:
        assert "Document Type" in table_values
    elif document_type is DocumentType.TECHNICAL_DESIGN:
        assert "Component" in table_values
    elif document_type is DocumentType.SOP:
        numbered = [p.text for p in paragraphs if p.style.name == "List Number"]
        assert numbered == ["Open the system.", "Record the result."]
    elif document_type is DocumentType.PRODUCT_SPECIFICATION:
        assert "Requirement" in table_values
    else:
        assert len(tables) == 1


@pytest.mark.parametrize("document_type", list(DocumentType))
async def test_complete_graph_for_every_document_type(
    tmp_path: Path,
    document_type: DocumentType,
) -> None:
    plan, content = _case(document_type)
    validator = DocumentValidator(NoIssueReviewer())

    result = await run_agent(
        f"Create a {document_type.value.replace('_', ' ')} document.",
        planner=StaticPlanner(plan),
        tool_selector=NoToolSelector(),
        writer=StaticWriter(content),
        docx_generator=DocxGenerator(tmp_path),
        validator=validator,
    )

    assert result["status"] == "completed"
    assert result["plan"] is not None
    assert result["plan"].document_type is document_type
    assert result["content"] is not None
    assert result["content"].document_type is document_type
    assert result["validation_report"] is not None
    assert result["validation_report"].passed is True
    assert result["output_filename"]
    assert result["final_response"] == (
        f"Document generated successfully: {result['output_filename']}"
    )

    destination = tmp_path / result["output_filename"]
    assert destination.exists()
    document = Document(destination)
    assert document.paragraphs[0].text == content.title
    headings = [p.text for p in document.paragraphs if p.style.name == "Heading 1"]
    for section in content.sections:
        assert section.heading in headings
    _template_checks(document_type, document)


async def test_meeting_minutes_allow_sections_backed_only_by_structured_fields(
    tmp_path: Path,
) -> None:
    headings = ("Meeting Details", "Agenda", "Decisions", "Action Items")
    plan = _plan(DocumentType.MEETING_MINUTES, headings)
    content = DocumentContent(
        document_type=DocumentType.MEETING_MINUTES,
        title=plan.title,
        meeting=MeetingDetails(
            date="2026-11-06",
            time="10:00",
            location="Room A",
            facilitator="Anna",
            attendees=["Anna", "Ben"],
        ),
        action_items=[
            ActionItem(
                action="Send notes",
                owner="Anna",
                due_date="2026-11-08",
                status="Open",
            )
        ],
        sections=[ContentSection(heading=heading) for heading in headings],
    )

    result = await run_agent(
        "Create short meeting minutes for the Atlas kickoff.",
        planner=StaticPlanner(plan),
        tool_selector=NoToolSelector(),
        writer=StaticWriter(content),
        docx_generator=DocxGenerator(tmp_path),
        validator=DocumentValidator(NoIssueReviewer()),
    )

    assert result["status"] == "completed"
    assert result["validation_report"] is not None
    assert result["validation_report"].passed is True
    assert result["output_filename"]
    assert (tmp_path / result["output_filename"]).exists()

    document = Document(tmp_path / result["output_filename"])
    headings_in_document = [
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.style.name == "Heading 1"
    ]
    assert headings_in_document == list(headings)
    table_values = {
        cell.text
        for table in document.tables
        for row in table.rows
        for cell in row.cells
    }
    assert "Meeting Date" in table_values
    assert "Send notes" in table_values


async def test_invalid_request_never_generates_document(tmp_path: Path) -> None:
    docx = RecordingDocx()
    result = await run_agent(
        "   ",
        planner=StaticPlanner(_plan(DocumentType.BUSINESS_REPORT, ("Overview",))),
        tool_selector=NoToolSelector(),
        writer=StaticWriter(_case(DocumentType.BUSINESS_REPORT)[1]),
        docx_generator=docx,
        validator=DocumentValidator(NoIssueReviewer()),
    )

    assert result["status"] == "rejected"
    assert result["final_response"] == REJECTION_RESPONSE
    assert result["output_filename"] == ""
    assert docx.calls == 0


async def test_planning_failure_never_generates_document() -> None:
    docx = RecordingDocx()
    plan, content = _case(DocumentType.BUSINESS_REPORT)
    result = await run_agent(
        "Create a report",
        planner=StaticPlanner(plan, error=PlanningError("planning failed")),
        tool_selector=NoToolSelector(),
        writer=StaticWriter(content),
        docx_generator=docx,
        validator=DocumentValidator(NoIssueReviewer()),
    )

    assert result["status"] == "failed"
    assert result["final_response"] == PLANNING_FAILURE_RESPONSE
    assert result["output_filename"] == ""
    assert docx.calls == 0


async def test_tool_selection_failure_never_generates_document() -> None:
    class FailingSelector:
        async def select(self, messages: object) -> AIMessage:
            raise ToolSelectionError("tool selection failed")

    plan, content = _case(DocumentType.BUSINESS_REPORT)
    docx = RecordingDocx()
    result = await run_agent(
        "Create a report",
        planner=StaticPlanner(plan),
        tool_selector=FailingSelector(),
        writer=StaticWriter(content),
        docx_generator=docx,
        validator=DocumentValidator(NoIssueReviewer()),
    )

    assert result["status"] == "failed"
    assert result["final_response"] == TOOL_FAILURE_RESPONSE
    assert docx.calls == 0


async def test_tool_execution_failure_never_generates_document() -> None:
    class TwoStepSelector:
        def __init__(self) -> None:
            self.calls = 0

        async def select(self, messages: object) -> AIMessage:
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "unknown_tool",
                            "args": {},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(content="")

    plan, content = _case(DocumentType.BUSINESS_REPORT)
    docx = RecordingDocx()
    result = await run_agent(
        "Create a report",
        planner=StaticPlanner(plan),
        tool_selector=TwoStepSelector(),
        writer=StaticWriter(content),
        docx_generator=docx,
        validator=DocumentValidator(NoIssueReviewer()),
    )

    assert result["status"] == "failed"
    assert result["final_response"] == TOOL_FAILURE_RESPONSE
    assert docx.calls == 0


async def test_content_generation_failure_never_generates_document() -> None:
    plan, content = _case(DocumentType.BUSINESS_REPORT)
    docx = RecordingDocx()
    result = await run_agent(
        "Create a report",
        planner=StaticPlanner(plan),
        tool_selector=NoToolSelector(),
        writer=StaticWriter(
            content,
            generate_error=ContentGenerationError("content failed"),
        ),
        docx_generator=docx,
        validator=DocumentValidator(NoIssueReviewer()),
    )

    assert result["status"] == "failed"
    assert result["final_response"] == DOCUMENT_FAILURE_RESPONSE
    assert docx.calls == 0


async def test_review_failure_never_generates_document() -> None:
    plan, content = _case(DocumentType.BUSINESS_REPORT)
    docx = RecordingDocx()
    result = await run_agent(
        "Create a report",
        planner=StaticPlanner(plan),
        tool_selector=NoToolSelector(),
        writer=StaticWriter(content),
        docx_generator=docx,
        validator=RaisingValidator(ReviewError("review failed")),
    )

    assert result["status"] == "failed"
    assert result["final_response"] == REVIEW_FAILURE_RESPONSE
    assert docx.calls == 0


def _blocking_report() -> ValidationReport:
    return ValidationReport(
        issues=[
            ValidationIssue(
                code="blocking",
                message="Blocking issue.",
                severity=IssueSeverity.BLOCKING,
            )
        ]
    )


async def test_revision_failure_never_generates_document() -> None:
    plan, content = _case(DocumentType.BUSINESS_REPORT)
    docx = RecordingDocx()
    result = await run_agent(
        "Create a report",
        planner=StaticPlanner(plan),
        tool_selector=NoToolSelector(),
        writer=StaticWriter(
            content,
            revise_error=ContentGenerationError("revision failed"),
        ),
        docx_generator=docx,
        validator=SequenceValidator([_blocking_report()]),
    )

    assert result["status"] == "failed"
    assert result["final_response"] == REVISION_FAILURE_RESPONSE
    assert result["revision_count"] == 1
    assert docx.calls == 0


async def test_revision_exhaustion_never_generates_document() -> None:
    plan, content = _case(DocumentType.BUSINESS_REPORT)
    docx = RecordingDocx()
    result = await run_agent(
        "Create a report",
        planner=StaticPlanner(plan),
        tool_selector=NoToolSelector(),
        writer=StaticWriter(content),
        docx_generator=docx,
        validator=SequenceValidator(
            [_blocking_report() for _ in range(MAX_REVISION_ATTEMPTS + 1)]
        ),
    )

    assert result["status"] == "failed"
    assert result["final_response"] == VALIDATION_FAILURE_RESPONSE
    assert result["revision_count"] == MAX_REVISION_ATTEMPTS
    assert docx.calls == 0


async def test_docx_failure_never_reports_success() -> None:
    plan, content = _case(DocumentType.BUSINESS_REPORT)
    docx = RecordingDocx(error=DocxGenerationError("docx failed"))
    result = await run_agent(
        "Create a report",
        planner=StaticPlanner(plan),
        tool_selector=NoToolSelector(),
        writer=StaticWriter(content),
        docx_generator=docx,
        validator=DocumentValidator(NoIssueReviewer()),
    )

    assert result["status"] == "failed"
    assert result["final_response"] == DOCUMENT_FAILURE_RESPONSE
    assert result["output_filename"] == ""
    assert docx.calls == 1
