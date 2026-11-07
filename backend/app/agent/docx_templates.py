"""Deterministic DOCX templates for each supported document type."""

from collections.abc import Callable

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.text.paragraph import Paragraph

from backend.app.agent.document_content import ContentSection, DocumentContent
from backend.app.agent.models import DocumentType

TemplateRenderer = Callable[
    [Document, DocumentContent, str, str],
    None,
]


class TemplateNotFoundError(RuntimeError):
    """Raised when no renderer exists for a document type."""


def _value_or_placeholder(value: str | None, label: str) -> str:
    if value and value.strip():
        return value.strip()
    return f"[{label}]"


def _add_title(
    document: Document,
    content: DocumentContent,
    subtitle: str | None,
) -> None:
    title = document.add_heading(content.title, level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    if subtitle is None:
        return
    subtitle_paragraph = document.add_paragraph()
    subtitle_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle_paragraph.add_run(subtitle)
    run.italic = True
    run.font.size = Pt(14)


def _add_header_footer(document: Document, content: DocumentContent) -> None:
    section = document.sections[0]

    header_paragraph = section.header.paragraphs[0]
    header_paragraph.text = content.title
    header_paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in header_paragraph.runs:
        run.font.size = Pt(8)

    footer_paragraph = section.footer.paragraphs[0]
    footer_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer_paragraph.add_run("FluidAI Doc Builder | Page ")
    _add_page_field(footer_paragraph)


def _add_page_field(paragraph: Paragraph) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = "PAGE"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instruction)
    run._r.append(end)


def _add_table(
    document: Document,
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
) -> None:
    if not rows:
        return
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = True

    for cell, header in zip(table.rows[0].cells, headers, strict=True):
        cell.text = header
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True

    for row_values in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row_values, strict=True):
            cell.text = value


def _add_metadata_table(
    document: Document,
    title: str,
    rows: list[tuple[str, str]],
) -> None:
    document.add_heading(title, level=2)
    table = document.add_table(rows=0, cols=2)
    table.style = "Table Grid"
    table.autofit = True
    for label, value in rows:
        cells = table.add_row().cells
        cells[0].text = label
        cells[1].text = value
        for paragraph in cells[0].paragraphs:
            for run in paragraph.runs:
                run.bold = True


def _document_metadata_rows(
    content: DocumentContent,
    document_date: str,
    document_version: str,
) -> list[tuple[str, str]]:
    return [
        ("Document Type", content.document_type.value.replace("_", " ").title()),
        (
            "Prepared For",
            _value_or_placeholder(content.audience, "Audience"),
        ),
        ("Document Date", document_date),
        ("Document Version", document_version),
    ]


def _render_section(document: Document, section: ContentSection) -> None:
    document.add_heading(section.heading, level=1)
    for paragraph in section.paragraphs:
        document.add_paragraph(paragraph)
    for bullet in section.bullets:
        document.add_paragraph(bullet, style="List Bullet")


def _matches(heading: str, keywords: tuple[str, ...]) -> bool:
    normalized = " ".join(heading.casefold().split())
    return any(keyword in normalized for keyword in keywords)


def _render_sections_with_insert(
    document: Document,
    content: DocumentContent,
    *,
    keywords: tuple[str, ...],
    fallback_heading: str,
    insert: Callable[[Document, DocumentContent], None],
    has_data: bool,
) -> None:
    inserted = False
    for section in content.sections:
        _render_section(document, section)
        if not inserted and has_data and _matches(section.heading, keywords):
            insert(document, content)
            inserted = True

    if has_data and not inserted:
        document.add_heading(fallback_heading, level=1)
        insert(document, content)


def _insert_milestones(document: Document, content: DocumentContent) -> None:
    _add_table(
        document,
        ("Milestone", "Target Date", "Owner", "Status"),
        [
            (
                milestone.name,
                _value_or_placeholder(milestone.target_date, "Target date"),
                _value_or_placeholder(milestone.owner, "Owner"),
                _value_or_placeholder(milestone.status, "Status"),
            )
            for milestone in content.milestones
        ],
    )


def _insert_action_items(document: Document, content: DocumentContent) -> None:
    _add_table(
        document,
        ("Action", "Owner", "Due Date", "Status"),
        [
            (
                item.action,
                _value_or_placeholder(item.owner, "Owner"),
                _value_or_placeholder(item.due_date, "Due date"),
                _value_or_placeholder(item.status, "Status"),
            )
            for item in content.action_items
        ],
    )


def _insert_requirements(document: Document, content: DocumentContent) -> None:
    _add_table(
        document,
        ("ID", "Requirement", "Priority", "Acceptance Criteria"),
        [
            (
                _value_or_placeholder(requirement.identifier, "ID"),
                requirement.description,
                _value_or_placeholder(requirement.priority, "Priority"),
                _value_or_placeholder(
                    requirement.acceptance_criteria, "Acceptance criteria"
                ),
            )
            for requirement in content.requirements
        ],
    )


def _insert_components(document: Document, content: DocumentContent) -> None:
    _add_table(
        document,
        ("Component", "Responsibility", "Interfaces"),
        [
            (
                component.name,
                component.responsibility,
                ", ".join(component.interfaces)
                if component.interfaces
                else "[Interfaces]",
            )
            for component in content.components
        ],
    )


def _insert_sop_steps(document: Document, content: DocumentContent) -> None:
    for step in content.sop_steps:
        document.add_paragraph(step.instruction, style="List Number")
        if step.note:
            note = document.add_paragraph(step.note)
            note.paragraph_format.left_indent = Pt(24)
            for run in note.runs:
                run.italic = True


def render_business_proposal(
    document: Document,
    content: DocumentContent,
    document_date: str,
    document_version: str,
) -> None:
    _add_title(document, content, "Business Proposal")
    _add_metadata_table(
        document,
        "Document Details",
        _document_metadata_rows(content, document_date, document_version),
    )
    document.add_page_break()
    _render_sections_with_insert(
        document,
        content,
        keywords=("timeline", "milestone", "schedule"),
        fallback_heading="Timeline",
        insert=_insert_milestones,
        has_data=bool(content.milestones),
    )


def render_meeting_minutes(
    document: Document,
    content: DocumentContent,
    document_date: str,
    document_version: str,
) -> None:
    _add_title(document, content, "Meeting Minutes")
    meeting = content.meeting
    meeting_rows = [
        (
            "Meeting Date",
            _value_or_placeholder(
                meeting.date if meeting else None,
                "Meeting date",
            ),
        ),
        ("Time", _value_or_placeholder(meeting.time if meeting else None, "Time")),
        (
            "Location",
            _value_or_placeholder(meeting.location if meeting else None, "Location"),
        ),
        (
            "Facilitator",
            _value_or_placeholder(
                meeting.facilitator if meeting else None, "Facilitator"
            ),
        ),
        (
            "Attendees",
            ", ".join(meeting.attendees)
            if meeting and meeting.attendees
            else "[Attendees]",
        ),
    ]
    _add_metadata_table(document, "Meeting Details", meeting_rows)
    _add_metadata_table(
        document,
        "Document Details",
        [
            ("Document Date", document_date),
            ("Document Version", document_version),
        ],
    )
    _render_sections_with_insert(
        document,
        content,
        keywords=("action", "next step"),
        fallback_heading="Action Items",
        insert=_insert_action_items,
        has_data=bool(content.action_items),
    )


def render_business_report(
    document: Document,
    content: DocumentContent,
    document_date: str,
    document_version: str,
) -> None:
    _add_title(document, content, "Business Report")
    _add_metadata_table(
        document,
        "Report Details",
        _document_metadata_rows(content, document_date, document_version),
    )
    for section in content.sections:
        _render_section(document, section)


def render_project_plan(
    document: Document,
    content: DocumentContent,
    document_date: str,
    document_version: str,
) -> None:
    _add_title(document, content, "Project Plan")
    _add_metadata_table(
        document,
        "Project Details",
        _document_metadata_rows(content, document_date, document_version),
    )
    _render_sections_with_insert(
        document,
        content,
        keywords=("milestone", "timeline", "schedule"),
        fallback_heading="Milestones and Timeline",
        insert=_insert_milestones,
        has_data=bool(content.milestones),
    )


def render_technical_design(
    document: Document,
    content: DocumentContent,
    document_date: str,
    document_version: str,
) -> None:
    _add_title(document, content, "Technical Design")
    _add_metadata_table(
        document,
        "Document Details",
        _document_metadata_rows(content, document_date, document_version),
    )
    _render_sections_with_insert(
        document,
        content,
        keywords=("component", "architecture", "interface"),
        fallback_heading="Components and Interfaces",
        insert=_insert_components,
        has_data=bool(content.components),
    )


def render_sop(
    document: Document,
    content: DocumentContent,
    document_date: str,
    document_version: str,
) -> None:
    _add_title(document, content, "Standard Operating Procedure")
    _add_metadata_table(
        document,
        "Document Details",
        _document_metadata_rows(content, document_date, document_version),
    )
    _render_sections_with_insert(
        document,
        content,
        keywords=("procedure", "step"),
        fallback_heading="Procedure Steps",
        insert=_insert_sop_steps,
        has_data=bool(content.sop_steps),
    )


def render_product_specification(
    document: Document,
    content: DocumentContent,
    document_date: str,
    document_version: str,
) -> None:
    _add_title(document, content, "Product Specification")
    _add_metadata_table(
        document,
        "Document Details",
        _document_metadata_rows(content, document_date, document_version),
    )
    _render_sections_with_insert(
        document,
        content,
        keywords=("requirement",),
        fallback_heading="Requirements",
        insert=_insert_requirements,
        has_data=bool(content.requirements),
    )


def render_generic(
    document: Document,
    content: DocumentContent,
    document_date: str,
    document_version: str,
) -> None:
    _add_title(document, content, None)
    _add_metadata_table(
        document,
        "Document Details",
        _document_metadata_rows(content, document_date, document_version),
    )
    for section in content.sections:
        _render_section(document, section)


TEMPLATE_RENDERERS: dict[DocumentType, TemplateRenderer] = {
    DocumentType.BUSINESS_PROPOSAL: render_business_proposal,
    DocumentType.MEETING_MINUTES: render_meeting_minutes,
    DocumentType.PROJECT_PLAN: render_project_plan,
    DocumentType.BUSINESS_REPORT: render_business_report,
    DocumentType.TECHNICAL_DESIGN: render_technical_design,
    DocumentType.SOP: render_sop,
    DocumentType.PRODUCT_SPECIFICATION: render_product_specification,
    DocumentType.OTHER: render_generic,
}


def render_document(
    document: Document,
    content: DocumentContent,
    document_date: str,
    document_version: str,
) -> None:
    renderer = TEMPLATE_RENDERERS.get(content.document_type)
    if renderer is None:
        raise TemplateNotFoundError("Unsupported document type.")
    _add_header_footer(document, content)
    renderer(document, content, document_date, document_version)
