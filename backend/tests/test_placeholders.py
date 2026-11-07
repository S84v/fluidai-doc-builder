"""Tests for rendered placeholder inventory and document metadata resolution."""

from datetime import date
from pathlib import Path

import pytest
from docx import Document
from docx.enum.section import WD_SECTION

from backend.app.agent import (
    ActionItem,
    ContentSection,
    DocumentContent,
    DocumentMetadata,
    DocumentPlan,
    DocumentType,
    DocxGenerator,
    MeetingDetails,
    Milestone,
    PlanPlaceholder,
    PlanSection,
    Requirement,
    TechnicalComponent,
)
from backend.app.agent.placeholders import (
    AMBIGUOUS_PLACEHOLDER_REASON,
    PLACEHOLDER_RE,
    TEMPLATE_PLACEHOLDER_REASONS,
    UNKNOWN_PLACEHOLDER_REASON,
    collect_rendered_placeholders,
)
from backend.tests.plan_factory import make_execution_tasks

FIXED_TODAY = date(2026, 11, 7)


def _plan(
    document_type: DocumentType,
    headings: tuple[str, ...],
    *,
    placeholders: list[PlanPlaceholder] | None = None,
    document_date: str | None = None,
    document_version: str | None = None,
) -> DocumentPlan:
    return DocumentPlan(
        document_type=document_type,
        title="Test Document",
        objective="Test the document.",
        document_date=document_date,
        document_version=document_version,
        sections=[
            PlanSection(
                heading=heading,
                purpose=f"Purpose of {heading}.",
                placeholders=placeholders or [],
            )
            for heading in headings
        ],
        execution_tasks=make_execution_tasks(),
    )


def _sections(headings: tuple[str, ...]) -> list[ContentSection]:
    return [
        ContentSection(heading=heading, paragraphs=["Text."]) for heading in headings
    ]


def _missing_value_content(
    document_type: DocumentType,
) -> tuple[DocumentPlan, DocumentContent]:
    if document_type is DocumentType.MEETING_MINUTES:
        headings = ("Agenda", "Action Items")
        plan = _plan(document_type, headings)
        content = DocumentContent(
            document_type=document_type,
            title=plan.title,
            meeting=MeetingDetails(),
            action_items=[ActionItem(action="Do it")],
            sections=_sections(headings),
        )
        return plan, content

    if document_type is DocumentType.PROJECT_PLAN:
        headings = ("Milestones",)
        plan = _plan(document_type, headings)
        content = DocumentContent(
            document_type=document_type,
            title=plan.title,
            sections=_sections(headings),
            milestones=[Milestone(name="Phase one")],
        )
        return plan, content

    if document_type is DocumentType.BUSINESS_PROPOSAL:
        headings = ("Timeline",)
        plan = _plan(document_type, headings)
        content = DocumentContent(
            document_type=document_type,
            title=plan.title,
            sections=_sections(headings),
            milestones=[Milestone(name="Phase one")],
        )
        return plan, content

    if document_type is DocumentType.PRODUCT_SPECIFICATION:
        headings = ("Requirements",)
        plan = _plan(document_type, headings)
        content = DocumentContent(
            document_type=document_type,
            title=plan.title,
            sections=_sections(headings),
            requirements=[Requirement(description="Needed")],
        )
        return plan, content

    if document_type is DocumentType.TECHNICAL_DESIGN:
        headings = ("Components",)
        plan = _plan(document_type, headings)
        content = DocumentContent(
            document_type=document_type,
            title=plan.title,
            sections=_sections(headings),
            components=[TechnicalComponent(name="API", responsibility="Serve")],
        )
        return plan, content

    headings = ("Overview",)
    plan = _plan(document_type, headings)
    content = DocumentContent(
        document_type=document_type,
        title=plan.title,
        sections=_sections(headings),
    )
    return plan, content


def _all_placeholders(document: Document) -> list[str]:
    texts = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            texts.extend(cell.text for cell in row.cells)
    for section in document.sections:
        texts.extend(paragraph.text for paragraph in section.header.paragraphs)
        texts.extend(paragraph.text for paragraph in section.footer.paragraphs)
    return [
        placeholder for text in texts for placeholder in PLACEHOLDER_RE.findall(text)
    ]


def _table_value(document: Document, label: str) -> str | None:
    for table in document.tables:
        for row in table.rows:
            if len(row.cells) >= 2 and row.cells[0].text == label:
                return row.cells[1].text
    return None


@pytest.mark.parametrize("document_type", list(DocumentType))
def test_every_rendered_placeholder_has_an_inventory_entry(
    tmp_path: Path,
    document_type: DocumentType,
) -> None:
    plan, content = _missing_value_content(document_type)
    generated = DocxGenerator(tmp_path).generate(
        content,
        "Create the document",
        plan,
        today=FIXED_TODAY,
    )
    document = Document(tmp_path / generated.filename)

    actual = _all_placeholders(document)
    recorded = [placeholder.text for placeholder in generated.placeholders]
    assert sorted(actual) == sorted(recorded)
    assert all(placeholder.reason for placeholder in generated.placeholders)
    assert all(placeholder.location for placeholder in generated.placeholders)


def test_duplicate_placeholders_keep_separate_locations(tmp_path: Path) -> None:
    headings = ("Action Items",)
    plan = _plan(DocumentType.MEETING_MINUTES, headings)
    content = DocumentContent(
        document_type=DocumentType.MEETING_MINUTES,
        title=plan.title,
        meeting=MeetingDetails(),
        action_items=[
            {"action": "First", "owner": None, "due_date": None, "status": None},
            {"action": "Second", "owner": None, "due_date": None, "status": None},
        ],
        sections=[ContentSection(heading="Action Items", paragraphs=["Text."])],
    )

    generated = DocxGenerator(tmp_path).generate(
        content, "Create minutes", plan, today=FIXED_TODAY
    )
    owners = [item for item in generated.placeholders if item.text == "[Owner]"]

    assert len(owners) == 2
    assert owners[0].location != owners[1].location


def test_explained_content_placeholder_is_recorded(tmp_path: Path) -> None:
    placeholder = PlanPlaceholder(
        text="[Budget]",
        reason="The approved budget was not supplied.",
    )
    plan = _plan(DocumentType.BUSINESS_REPORT, ("Budget",), placeholders=[placeholder])
    content = DocumentContent(
        document_type=DocumentType.BUSINESS_REPORT,
        title=plan.title,
        sections=[
            ContentSection(
                heading="Budget",
                paragraphs=["Budget: [Budget]"],
            )
        ],
    )

    generated = DocxGenerator(tmp_path).generate(
        content, "Create report", plan, today=FIXED_TODAY
    )

    budget = next(item for item in generated.placeholders if item.text == "[Budget]")
    assert budget.reason == placeholder.reason
    assert "Budget" in budget.location


def test_plan_placeholder_in_structured_table_uses_plan_reason(tmp_path: Path) -> None:
    placeholder = PlanPlaceholder(
        text="[Budget]",
        reason="The approved budget was not supplied.",
    )
    plan = _plan(
        DocumentType.PRODUCT_SPECIFICATION,
        ("Functional Requirements",),
        placeholders=[placeholder],
    )
    content = DocumentContent(
        document_type=DocumentType.PRODUCT_SPECIFICATION,
        title=plan.title,
        sections=_sections(("Functional Requirements",)),
        requirements=[Requirement(description="Stay within [Budget].")],
    )

    generated = DocxGenerator(tmp_path).generate(
        content, "Create specification", plan, today=FIXED_TODAY
    )
    budget = next(item for item in generated.placeholders if item.text == "[Budget]")

    assert budget.reason == placeholder.reason
    assert "Table" in budget.location


def test_ambiguous_structured_placeholder_reasons_get_a_generic_reason() -> None:
    plan = _plan(DocumentType.BUSINESS_REPORT, ("Alpha", "Beta"))
    plan.sections[0].placeholders = [
        PlanPlaceholder(text="[Budget]", reason="Alpha budget reason.")
    ]
    plan.sections[1].placeholders = [
        PlanPlaceholder(text="[Budget]", reason="Beta budget reason.")
    ]
    document = Document()
    table = document.add_table(rows=1, cols=1)
    table.cell(0, 0).text = "Budget: [Budget]"

    records = collect_rendered_placeholders(document, plan)

    assert len(records) == 1
    assert records[0].reason == AMBIGUOUS_PLACEHOLDER_REASON


def test_template_placeholder_reason_takes_precedence_in_structured_table() -> None:
    plan = _plan(DocumentType.MEETING_MINUTES, ("Action Items",))
    plan.sections[0].placeholders = [
        PlanPlaceholder(text="[Owner]", reason="Custom plan owner reason.")
    ]
    document = Document()
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Action"
    table.cell(0, 1).text = "Owner: [Owner]"

    records = collect_rendered_placeholders(document, plan)
    owner = next(record for record in records if record.text == "[Owner]")

    assert owner.reason == TEMPLATE_PLACEHOLDER_REASONS["[Owner]"]
    assert owner.reason != "Custom plan owner reason."


def test_undeclared_structured_table_placeholder_is_transparent(
    tmp_path: Path,
) -> None:
    plan = _plan(
        DocumentType.PRODUCT_SPECIFICATION,
        ("Functional Requirements",),
    )
    content = DocumentContent(
        document_type=DocumentType.PRODUCT_SPECIFICATION,
        title=plan.title,
        sections=_sections(("Functional Requirements",)),
        requirements=[Requirement(description="Stay within [Mystery].")],
    )

    generated = DocxGenerator(tmp_path).generate(
        content, "Create specification", plan, today=FIXED_TODAY
    )

    mystery = next(item for item in generated.placeholders if item.text == "[Mystery]")
    assert mystery.reason == UNKNOWN_PLACEHOLDER_REASON


def test_unexplained_content_placeholder_is_transparent(tmp_path: Path) -> None:
    plan = _plan(DocumentType.BUSINESS_REPORT, ("Overview",))
    content = DocumentContent(
        document_type=DocumentType.BUSINESS_REPORT,
        title=plan.title,
        sections=[
            ContentSection(
                heading="Overview",
                paragraphs=["Unknown: [Mystery]"],
            )
        ],
    )

    generated = DocxGenerator(tmp_path).generate(
        content, "Create report", plan, today=FIXED_TODAY
    )

    mystery = next(item for item in generated.placeholders if item.text == "[Mystery]")
    assert mystery.reason == UNKNOWN_PLACEHOLDER_REASON


def test_document_date_and_version_defaults(tmp_path: Path) -> None:
    plan, content = _missing_value_content(DocumentType.BUSINESS_REPORT)

    generated = DocxGenerator(tmp_path).generate(
        content, "Create report", plan, today=FIXED_TODAY
    )
    document = Document(tmp_path / generated.filename)

    assert _table_value(document, "Document Date") == "2026-11-07"
    assert _table_value(document, "Document Version") == "1.0"
    assert "[Date]" not in _all_placeholders(document)
    assert "[Version]" not in _all_placeholders(document)


def test_plan_date_and_version_override_writer_metadata(tmp_path: Path) -> None:
    plan = _plan(
        DocumentType.BUSINESS_REPORT,
        ("Overview",),
        document_date="2026-10-01",
        document_version="2.5",
    )
    content = DocumentContent(
        document_type=DocumentType.BUSINESS_REPORT,
        title=plan.title,
        metadata=DocumentMetadata(date="2099-01-01", version="9.9"),
        sections=[ContentSection(heading="Overview", paragraphs=["Text."])],
    )

    generated = DocxGenerator(tmp_path).generate(
        content, "Create report", plan, today=FIXED_TODAY
    )
    document = Document(tmp_path / generated.filename)

    assert _table_value(document, "Document Date") == "2026-10-01"
    assert _table_value(document, "Document Version") == "2.5"


def test_meeting_date_is_not_replaced_by_document_date(tmp_path: Path) -> None:
    plan = _plan(
        DocumentType.MEETING_MINUTES,
        ("Action Items",),
        document_date="2026-11-07",
    )
    content = DocumentContent(
        document_type=DocumentType.MEETING_MINUTES,
        title=plan.title,
        meeting=MeetingDetails(),
        action_items=[ActionItem(action="Do it")],
        sections=[ContentSection(heading="Action Items", paragraphs=["Text."])],
    )

    generated = DocxGenerator(tmp_path).generate(
        content, "Create minutes", plan, today=FIXED_TODAY
    )
    document = Document(tmp_path / generated.filename)

    assert _table_value(document, "Meeting Date") == "[Meeting date]"
    assert _table_value(document, "Document Date") == "2026-11-07"
    assert any(item.text == "[Meeting date]" for item in generated.placeholders)


def test_same_placeholder_uses_section_specific_reasons(tmp_path: Path) -> None:
    plan = _plan(DocumentType.BUSINESS_REPORT, ("Alpha", "Beta"))
    plan.sections[0].placeholders = [
        PlanPlaceholder(text="[Owner]", reason="Alpha owner reason.")
    ]
    plan.sections[1].placeholders = [
        PlanPlaceholder(text="[Owner]", reason="Beta owner reason.")
    ]
    content = DocumentContent(
        document_type=DocumentType.BUSINESS_REPORT,
        title=plan.title,
        sections=[
            ContentSection(heading="Alpha", paragraphs=["Owner: [Owner]"]),
            ContentSection(heading="Beta", paragraphs=["Owner: [Owner]"]),
        ],
    )

    generated = DocxGenerator(tmp_path).generate(
        content, "Create report", plan, today=FIXED_TODAY
    )
    owners = [item for item in generated.placeholders if item.text == "[Owner]"]

    assert len(owners) == 2
    assert owners[0].location.startswith("Section 'Alpha'")
    assert owners[1].location.startswith("Section 'Beta'")
    assert owners[0].reason == "Alpha owner reason."
    assert owners[1].reason == "Beta owner reason."


def test_multiple_occurrences_in_one_section_remain_separate(tmp_path: Path) -> None:
    plan = _plan(DocumentType.BUSINESS_REPORT, ("Budget",))
    plan.sections[0].placeholders = [
        PlanPlaceholder(text="[Budget]", reason="Budget was not supplied.")
    ]
    content = DocumentContent(
        document_type=DocumentType.BUSINESS_REPORT,
        title=plan.title,
        sections=[
            ContentSection(
                heading="Budget",
                paragraphs=["First [Budget].", "Second [Budget]."],
            )
        ],
    )

    generated = DocxGenerator(tmp_path).generate(
        content, "Create report", plan, today=FIXED_TODAY
    )
    budgets = [item for item in generated.placeholders if item.text == "[Budget]"]

    assert len(budgets) == 2
    assert budgets[0].location != budgets[1].location
    assert all(item.reason == "Budget was not supplied." for item in budgets)


def test_rendered_placeholder_in_unmatched_section_uses_global_reason(
    tmp_path: Path,
) -> None:
    plan = _plan(DocumentType.BUSINESS_REPORT, ("Alpha",))
    plan.sections[0].placeholders = [
        PlanPlaceholder(text="[Budget]", reason="Alpha budget reason.")
    ]
    content = DocumentContent(
        document_type=DocumentType.BUSINESS_REPORT,
        title=plan.title,
        sections=[ContentSection(heading="Gamma", paragraphs=["Budget: [Budget]"])],
    )

    generated = DocxGenerator(tmp_path).generate(
        content, "Create report", plan, today=FIXED_TODAY
    )

    budget = next(item for item in generated.placeholders if item.text == "[Budget]")
    assert budget.reason == "Alpha budget reason."


def test_header_and_footer_placeholders_get_generic_reasons() -> None:
    placeholder = PlanPlaceholder(
        text="[Budget]",
        reason="The approved budget was not supplied.",
    )
    plan = _plan(
        DocumentType.BUSINESS_REPORT, ("Overview",), placeholders=[placeholder]
    )
    document = Document()
    document.sections[0].header.paragraphs[0].text = "Budget: [Budget]"
    document.sections[0].footer.paragraphs[0].text = "Budget: [Budget]"

    records = collect_rendered_placeholders(document, plan)

    assert len(records) == 2
    assert all(record.reason == UNKNOWN_PLACEHOLDER_REASON for record in records)


def test_template_placeholders_in_header_receive_deterministic_reason() -> None:
    plan = _plan(DocumentType.BUSINESS_REPORT, ("Overview",))
    document = Document()
    document.sections[0].header.paragraphs[0].text = "Owner: [Owner]"

    records = collect_rendered_placeholders(document, plan)

    assert len(records) == 1
    assert records[0].text == "[Owner]"
    assert records[0].location.startswith("Primary header")
    assert records[0].reason == "No responsible owner was provided."


def _save_and_reopen(document: Document, tmp_path: Path) -> Document:
    path = tmp_path / "audit.docx"
    document.save(path)
    return Document(path)


def test_first_and_even_page_header_footer_variants_are_scanned(
    tmp_path: Path,
) -> None:
    document = Document()
    section = document.sections[0]
    section.different_first_page_header_footer = True
    document.settings.odd_and_even_pages_header_footer = True
    section.header.paragraphs[0].text = "Owner: [Owner]"
    section.footer.paragraphs[0].text = "Status: [Status]"
    section.first_page_header.paragraphs[0].text = "Priority: [Priority]"
    section.first_page_footer.paragraphs[0].text = "ID: [ID]"
    section.even_page_header.paragraphs[0].text = "Interfaces: [Interfaces]"
    section.even_page_footer.paragraphs[0].text = "Due: [Due date]"

    reopened = _save_and_reopen(document, tmp_path)
    records = collect_rendered_placeholders(reopened, None)
    locations = [record.location for record in records]

    assert len(records) == 6
    assert any(location.startswith("Primary header") for location in locations)
    assert any(location.startswith("Primary footer") for location in locations)
    assert any(location.startswith("First-page header") for location in locations)
    assert any(location.startswith("First-page footer") for location in locations)
    assert any(location.startswith("Even-page header") for location in locations)
    assert any(location.startswith("Even-page footer") for location in locations)


def test_defined_first_and_even_parts_are_scanned_when_display_disabled(
    tmp_path: Path,
) -> None:
    document = Document()
    section = document.sections[0]
    section.first_page_header.paragraphs[0].text = "Owner: [Owner]"
    section.even_page_footer.paragraphs[0].text = "Status: [Status]"

    reopened = _save_and_reopen(document, tmp_path)
    records = collect_rendered_placeholders(reopened, None)
    locations = [record.location for record in records]

    assert any(location.startswith("First-page header") for location in locations)
    assert any(location.startswith("Even-page footer") for location in locations)


def test_shared_header_part_is_scanned_once(tmp_path: Path) -> None:
    document = Document()
    document.sections[0].header.paragraphs[0].text = "Owner: [Owner]"
    document.add_section(WD_SECTION.NEW_PAGE)

    reopened = _save_and_reopen(document, tmp_path)
    records = collect_rendered_placeholders(reopened, None)
    owners = [record for record in records if record.text == "[Owner]"]

    assert len(owners) == 1
    assert owners[0].location.startswith("Primary header (sections 1, 2)")


def test_separate_header_parts_with_same_placeholder_are_separate(
    tmp_path: Path,
) -> None:
    document = Document()
    section = document.sections[0]
    section.different_first_page_header_footer = True
    section.header.paragraphs[0].text = "Owner: [Owner]"
    section.first_page_header.paragraphs[0].text = "Owner: [Owner]"

    reopened = _save_and_reopen(document, tmp_path)
    records = collect_rendered_placeholders(reopened, None)
    owners = [record for record in records if record.text == "[Owner]"]

    assert len(owners) == 2
    assert {record.location.split(" (")[0] for record in owners} == {
        "Primary header",
        "First-page header",
    }


def test_nested_table_cells_are_scanned_recursively(tmp_path: Path) -> None:
    document = Document()
    table = document.add_table(rows=1, cols=1)
    cell = table.cell(0, 0)
    nested = cell.add_table(rows=1, cols=1)
    nested_cell = nested.cell(0, 0)
    nested_cell.paragraphs[0].text = "Owner: [Owner]"
    nested_two = nested_cell.add_table(rows=1, cols=1)
    nested_two.cell(0, 0).paragraphs[0].text = "Status: [Status]"

    reopened = _save_and_reopen(document, tmp_path)
    records = collect_rendered_placeholders(reopened, None)

    owner = next(record for record in records if record.text == "[Owner]")
    status = next(record for record in records if record.text == "[Status]")
    assert "Table 1" in owner.location
    assert owner.location.count("nested table") == 1
    assert status.location.count("nested table") == 2


def test_merged_table_cells_are_not_scanned_twice(tmp_path: Path) -> None:
    document = Document()
    table = document.add_table(rows=1, cols=2)
    merged = table.cell(0, 0).merge(table.cell(0, 1))
    merged.paragraphs[0].text = "Owner: [Owner]"

    reopened = _save_and_reopen(document, tmp_path)
    records = collect_rendered_placeholders(reopened, None)
    owners = [record for record in records if record.text == "[Owner]"]

    assert len(owners) == 1
