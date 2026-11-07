"""Tests for deterministic DOCX generation and filename safety."""

import re
from datetime import date
from pathlib import Path

import pytest
from docx import Document

from backend.app.agent import (
    ContentSection,
    DocumentContent,
    DocumentPlan,
    DocumentType,
    DocxGenerationError,
    DocxGenerator,
    PlanSection,
)
from backend.tests.plan_factory import make_execution_tasks


def _plan(
    *,
    document_date: str | None = None,
    document_version: str | None = None,
) -> DocumentPlan:
    return DocumentPlan(
        document_type=DocumentType.BUSINESS_REPORT,
        title="Quarterly Business Report",
        objective="Report the quarter.",
        document_date=document_date,
        document_version=document_version,
        sections=[
            PlanSection(heading="Overview", purpose="Overview."),
            PlanSection(heading="Next Steps", purpose="Next steps."),
        ],
        execution_tasks=make_execution_tasks(),
    )


def _table_value(document: Document, label: str) -> str | None:
    for table in document.tables:
        for row in table.rows:
            if len(row.cells) >= 2 and row.cells[0].text == label:
                return row.cells[1].text
    return None


def _content(
    document_type: DocumentType = DocumentType.BUSINESS_REPORT,
    *,
    title: str = "Quarterly Business Report",
) -> DocumentContent:
    return DocumentContent(
        document_type=document_type,
        title=title,
        sections=[
            ContentSection(
                heading="Overview",
                paragraphs=["This report summarizes the quarter."],
                bullets=["Revenue increased.", "Costs remained stable."],
            ),
            ContentSection(
                heading="Next Steps",
                paragraphs=["The team will review the findings."],
            ),
        ],
    )


@pytest.mark.parametrize(
    "document_type",
    [
        DocumentType.BUSINESS_REPORT,
        DocumentType.MEETING_MINUTES,
        DocumentType.TECHNICAL_DESIGN,
    ],
)
def test_generated_docx_reopens_with_expected_structure(
    tmp_path: Path,
    document_type: DocumentType,
) -> None:
    content = _content(document_type)
    generator = DocxGenerator(tmp_path)

    filename = generator.generate(
        content, "Create a quarterly business report"
    ).filename
    path = tmp_path / filename

    assert path.exists()
    assert path.parent == tmp_path.resolve()

    reopened = Document(path)
    assert reopened.core_properties.title == content.title
    assert reopened.paragraphs[0].text == content.title
    headings = [
        paragraph.text
        for paragraph in reopened.paragraphs
        if paragraph.style.name == "Heading 1"
    ]
    assert headings == ["Overview", "Next Steps"]
    assert "Revenue increased." in [p.text for p in reopened.paragraphs]
    bullet_paragraphs = [
        paragraph
        for paragraph in reopened.paragraphs
        if paragraph.style.name == "List Bullet"
    ]
    assert [p.text for p in bullet_paragraphs] == [
        "Revenue increased.",
        "Costs remained stable.",
    ]


def test_docx_uses_consistent_formatting(tmp_path: Path) -> None:
    generator = DocxGenerator(tmp_path)

    filename = generator.generate(_content(), "Create a business report").filename
    reopened = Document(tmp_path / filename)

    normal = reopened.styles["Normal"]
    assert normal.font.name == "Calibri"
    assert normal.font.size.pt == 11
    section = reopened.sections[0]
    assert section.top_margin.inches == 1.0
    assert section.bottom_margin.inches == 1.0
    assert section.left_margin.inches == 1.0
    assert section.right_margin.inches == 1.0


def test_filename_has_slug_identifier_and_single_extension(tmp_path: Path) -> None:
    generator = DocxGenerator(tmp_path)

    filename = generator.generate(
        _content(), "Create a quarterly business report"
    ).filename

    assert filename.startswith("quarterly-business-report-")
    assert re.fullmatch(r"[a-z0-9-]+-[0-9a-f]{12}\.docx", filename)
    assert filename.count(".docx") == 1


def test_filenames_are_unique(tmp_path: Path) -> None:
    generator = DocxGenerator(tmp_path)

    first = generator.generate(_content(), "Create a report").filename
    second = generator.generate(_content(), "Create a report").filename

    assert first != second
    assert (tmp_path / first).exists()
    assert (tmp_path / second).exists()


def test_request_cannot_escape_output_directory(tmp_path: Path) -> None:
    generator = DocxGenerator(tmp_path)

    filename = generator.generate(_content(), "../../etc/passwd").filename

    destination = (tmp_path / filename).resolve()
    assert destination.parent == tmp_path.resolve()
    assert destination.exists()


def test_nested_output_directory_is_created(tmp_path: Path) -> None:
    output_dir = tmp_path / "nested" / "documents"
    generator = DocxGenerator(output_dir)

    filename = generator.generate(_content(), "Create a report").filename

    assert (output_dir / filename).exists()


def test_filesystem_error_is_wrapped(tmp_path: Path) -> None:
    blocked_path = tmp_path / "not-a-directory"
    blocked_path.write_text("blocked", encoding="utf-8")
    generator = DocxGenerator(blocked_path)

    with pytest.raises(DocxGenerationError, match="output directory"):
        generator.generate(_content(), "Create a report")


def test_missing_document_metadata_uses_effective_defaults(tmp_path: Path) -> None:
    content = _content()
    generated = DocxGenerator(tmp_path).generate(
        content,
        "Create a report",
        _plan(),
        today=date(2026, 11, 7),
    )
    document = Document(tmp_path / generated.filename)

    assert generated.document_date == "2026-11-07"
    assert generated.document_version == "1.0"
    assert _table_value(document, "Document Date") == "2026-11-07"
    assert _table_value(document, "Document Version") == "1.0"


def test_explicit_document_metadata_is_returned_unchanged(tmp_path: Path) -> None:
    content = _content()
    generated = DocxGenerator(tmp_path).generate(
        content,
        "Create a report",
        _plan(document_date="2026-10-01", document_version="2.5"),
        today=date(2026, 11, 7),
    )
    document = Document(tmp_path / generated.filename)

    assert generated.document_date == "2026-10-01"
    assert generated.document_version == "2.5"
    assert _table_value(document, "Document Date") == "2026-10-01"
    assert _table_value(document, "Document Version") == "2.5"
