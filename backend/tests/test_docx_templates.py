"""Tests for document-type-specific DOCX templates."""

from pathlib import Path

import pytest
from docx import Document

from backend.app.agent import (
    ActionItem,
    ContentSection,
    DocumentContent,
    DocumentMetadata,
    DocumentType,
    DocxGenerationError,
    DocxGenerator,
    MeetingDetails,
    Milestone,
    Requirement,
    SopStep,
    TechnicalComponent,
)
from backend.app.agent.docx_templates import TEMPLATE_RENDERERS, render_generic


def _section(heading: str) -> ContentSection:
    return ContentSection(heading=heading, paragraphs=[f"{heading} content."])


def _content(
    document_type: DocumentType,
    sections: tuple[str, ...],
    **kwargs: object,
) -> DocumentContent:
    return DocumentContent(
        document_type=document_type,
        title=f"{document_type.value.replace('_', ' ').title()} Document",
        sections=[_section(heading) for heading in sections],
        **kwargs,
    )


def _generate(tmp_path: Path, content: DocumentContent) -> Document:
    filename = DocxGenerator(tmp_path).generate(content, "Create the document")
    return Document(tmp_path / filename)


def _table_rows(document: Document, table_index: int) -> list[tuple[str, ...]]:
    table = document.tables[table_index]
    return [tuple(cell.text for cell in row.cells) for row in table.rows]


def _paragraph_texts(document: Document, style: str | None = None) -> list[str]:
    return [
        paragraph.text
        for paragraph in document.paragraphs
        if style is None or paragraph.style.name == style
    ]


def test_every_document_type_has_a_renderer() -> None:
    assert set(TEMPLATE_RENDERERS) == set(DocumentType)
    assert TEMPLATE_RENDERERS[DocumentType.OTHER] is render_generic


def test_business_proposal_has_cover_details_and_timeline(tmp_path: Path) -> None:
    content = _content(
        DocumentType.BUSINESS_PROPOSAL,
        ("Executive Summary", "Timeline"),
        audience="Atlas Leadership",
        metadata=DocumentMetadata(date="2026-11-07", version="1.0"),
        milestones=[
            Milestone(
                name="Discovery",
                target_date="2026-11-21",
                owner="[Owner]",
                status="Planned",
            )
        ],
    )

    document = _generate(tmp_path, content)

    assert document.paragraphs[0].text == content.title
    assert any('w:type="page"' in p._p.xml for p in document.paragraphs)
    assert ("Prepared For", "Atlas Leadership") in _table_rows(document, 0)
    timeline_rows = _table_rows(document, 1)
    assert timeline_rows[0] == ("Milestone", "Target Date", "Owner", "Status")
    assert timeline_rows[1][0] == "Discovery"


def test_business_proposal_uses_placeholders_for_missing_metadata(
    tmp_path: Path,
) -> None:
    content = _content(
        DocumentType.BUSINESS_PROPOSAL,
        ("Executive Summary",),
    )

    document = _generate(tmp_path, content)

    rows = _table_rows(document, 0)
    assert ("Prepared For", "[Audience]") in rows
    assert ("Date", "[Date]") in rows
    assert ("Version", "[Version]") in rows


def test_meeting_minutes_have_metadata_and_action_items(tmp_path: Path) -> None:
    content = _content(
        DocumentType.MEETING_MINUTES,
        ("Agenda", "Action Items"),
        meeting=MeetingDetails(
            date="2026-11-06",
            time="10:00",
            location="Room A",
            facilitator="Anna",
            attendees=["Anna", "Ben"],
        ),
        action_items=[
            ActionItem(action="Send notes"),
            ActionItem(
                action="Review budget",
                owner="Ben",
                due_date="2026-11-10",
                status="Open",
            ),
        ],
    )

    document = _generate(tmp_path, content)

    meeting_rows = _table_rows(document, 0)
    assert ("Date", "2026-11-06") in meeting_rows
    assert ("Attendees", "Anna, Ben") in meeting_rows

    action_rows = _table_rows(document, 1)
    assert action_rows[0] == ("Action", "Owner", "Due Date", "Status")
    assert action_rows[1] == ("Send notes", "[Owner]", "[Due date]", "[Status]")
    assert action_rows[2] == ("Review budget", "Ben", "2026-11-10", "Open")


def test_project_plan_has_milestone_table(tmp_path: Path) -> None:
    content = _content(
        DocumentType.PROJECT_PLAN,
        ("Project Overview", "Milestones and Timeline"),
        milestones=[
            Milestone(
                name="Phase one",
                target_date="2026-12-01",
                owner="Ben",
                status="Planned",
            )
        ],
    )

    document = _generate(tmp_path, content)

    rows = _table_rows(document, 1)
    assert rows[0] == ("Milestone", "Target Date", "Owner", "Status")
    assert rows[1] == ("Phase one", "2026-12-01", "Ben", "Planned")


def test_business_report_has_report_metadata(tmp_path: Path) -> None:
    content = _content(
        DocumentType.BUSINESS_REPORT,
        ("Executive Summary", "Findings"),
        audience="Operations Team",
        metadata=DocumentMetadata(date="2026-11-07"),
    )

    document = _generate(tmp_path, content)

    rows = _table_rows(document, 0)
    assert ("Document Type", "Business Report") in rows
    assert ("Prepared For", "Operations Team") in rows
    assert ("Date", "2026-11-07") in rows


def test_technical_design_has_component_table(tmp_path: Path) -> None:
    content = _content(
        DocumentType.TECHNICAL_DESIGN,
        ("Architecture Overview", "Components and Interfaces"),
        components=[
            TechnicalComponent(
                name="API",
                responsibility="Serve document requests.",
                interfaces=["REST", "JSON"],
            )
        ],
    )

    document = _generate(tmp_path, content)

    rows = _table_rows(document, 1)
    assert rows[0] == ("Component", "Responsibility", "Interfaces")
    assert rows[1] == ("API", "Serve document requests.", "REST, JSON")


def test_sop_has_numbered_procedure_steps(tmp_path: Path) -> None:
    content = _content(
        DocumentType.SOP,
        ("Purpose and Scope", "Procedure Steps"),
        sop_steps=[
            SopStep(instruction="Open the system."),
            SopStep(instruction="Record the result.", note="Retain the audit log."),
        ],
    )

    document = _generate(tmp_path, content)

    numbered = _paragraph_texts(document, "List Number")
    assert numbered == ["Open the system.", "Record the result."]
    assert "Retain the audit log." in _paragraph_texts(document)


def test_product_specification_has_requirements_table(tmp_path: Path) -> None:
    content = _content(
        DocumentType.PRODUCT_SPECIFICATION,
        ("Functional Requirements",),
        requirements=[
            Requirement(
                identifier="FR-1",
                description="Users can generate a document.",
                priority="High",
                acceptance_criteria="A DOCX file is created.",
            )
        ],
    )

    document = _generate(tmp_path, content)

    rows = _table_rows(document, 1)
    assert rows[0] == ("ID", "Requirement", "Priority", "Acceptance Criteria")
    assert rows[1] == (
        "FR-1",
        "Users can generate a document.",
        "High",
        "A DOCX file is created.",
    )


def test_generic_fallback_is_clean_and_has_no_specialized_tables(
    tmp_path: Path,
) -> None:
    content = _content(
        DocumentType.OTHER,
        ("Executive Overview", "Main Content"),
    )

    document = _generate(tmp_path, content)

    assert document.tables == []
    assert _paragraph_texts(document, "Heading 1") == [
        "Executive Overview",
        "Main Content",
    ]


def test_unknown_document_type_fails_safely(tmp_path: Path) -> None:
    content = _content(DocumentType.OTHER, ("Overview",))
    invalid_content = content.model_copy(update={"document_type": "unsupported"})

    with pytest.raises(DocxGenerationError, match="Unsupported document type"):
        DocxGenerator(tmp_path).generate(invalid_content, "Create a document")


def test_header_footer_and_page_number_are_present(tmp_path: Path) -> None:
    content = _content(DocumentType.BUSINESS_REPORT, ("Findings",))

    filename = DocxGenerator(tmp_path).generate(content, "Create a report")
    document = Document(tmp_path / filename)

    assert document.sections[0].header.paragraphs[0].text == content.title
    footer_xml = document.sections[0].footer.paragraphs[0]._p.xml
    assert "FluidAI Doc Builder" in footer_xml
    assert "PAGE" in footer_xml
