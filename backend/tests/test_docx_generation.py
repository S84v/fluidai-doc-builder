"""Tests for deterministic DOCX generation and filename safety."""

import re
from pathlib import Path

import pytest
from docx import Document

from backend.app.agent import (
    ContentSection,
    DocumentContent,
    DocumentType,
    DocxGenerationError,
    DocxGenerator,
)


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

    filename = generator.generate(content, "Create a quarterly business report")
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

    filename = generator.generate(_content(), "Create a business report")
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

    filename = generator.generate(_content(), "Create a quarterly business report")

    assert filename.startswith("quarterly-business-report-")
    assert re.fullmatch(r"[a-z0-9-]+-[0-9a-f]{12}\.docx", filename)
    assert filename.count(".docx") == 1


def test_filenames_are_unique(tmp_path: Path) -> None:
    generator = DocxGenerator(tmp_path)

    first = generator.generate(_content(), "Create a report")
    second = generator.generate(_content(), "Create a report")

    assert first != second
    assert (tmp_path / first).exists()
    assert (tmp_path / second).exists()


def test_request_cannot_escape_output_directory(tmp_path: Path) -> None:
    generator = DocxGenerator(tmp_path)

    filename = generator.generate(_content(), "../../etc/passwd")

    destination = (tmp_path / filename).resolve()
    assert destination.parent == tmp_path.resolve()
    assert destination.exists()


def test_nested_output_directory_is_created(tmp_path: Path) -> None:
    output_dir = tmp_path / "nested" / "documents"
    generator = DocxGenerator(output_dir)

    filename = generator.generate(_content(), "Create a report")

    assert (output_dir / filename).exists()


def test_filesystem_error_is_wrapped(tmp_path: Path) -> None:
    blocked_path = tmp_path / "not-a-directory"
    blocked_path.write_text("blocked", encoding="utf-8")
    generator = DocxGenerator(blocked_path)

    with pytest.raises(DocxGenerationError, match="output directory"):
        generator.generate(_content(), "Create a report")
