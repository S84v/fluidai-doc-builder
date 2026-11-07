"""Placeholder inventory and verification for generated DOCX files."""

import re
from typing import Any

from docx import Document
from pydantic import BaseModel, ConfigDict, Field

from backend.app.agent.models import DocumentPlan

PLACEHOLDER_RE = re.compile(r"\[[^\[\]]+\]")

UNKNOWN_PLACEHOLDER_REASON = "This placeholder was not declared in the approved plan."
AMBIGUOUS_PLACEHOLDER_REASON = (
    "This placeholder is declared in the approved plan; multiple section-specific "
    "explanations exist."
)

TEMPLATE_PLACEHOLDER_REASONS: dict[str, str] = {
    "[Audience]": "The intended audience was not provided in the request.",
    "[Date]": (
        "No document date was supplied; the document date is resolved separately."
    ),
    "[Version]": (
        "No document version was supplied; the document version is resolved separately."
    ),
    "[Meeting date]": "The meeting date was not provided and is not inferred.",
    "[Time]": "The meeting time was not provided.",
    "[Location]": "The meeting location was not provided.",
    "[Facilitator]": "The facilitator was not provided.",
    "[Attendees]": "No attendees were provided.",
    "[Owner]": "No responsible owner was provided.",
    "[Due date]": "No due date was provided.",
    "[Status]": "No status was provided.",
    "[Target date]": "No target date was provided.",
    "[ID]": "No requirement identifier was provided.",
    "[Priority]": "No priority was provided.",
    "[Acceptance criteria]": "No acceptance criteria were provided.",
    "[Interfaces]": "No component interfaces were specified.",
}


class PlaceholderVerificationError(RuntimeError):
    """Raised when a rendered placeholder has no explanation."""


class RenderedPlaceholder(BaseModel):
    """One placeholder that remains in the rendered document."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(pattern=r"^\[[^\[\]]+\]$")
    location: str = Field(min_length=1)
    reason: str = Field(min_length=1)


def _normalize_heading(heading: str) -> str:
    return " ".join(heading.split()).casefold()


def _plan_placeholder_reasons(
    plan: DocumentPlan | None,
) -> dict[str, dict[str, str]]:
    reasons: dict[str, dict[str, str]] = {}
    if plan is None:
        return reasons
    for section in plan.sections:
        heading = _normalize_heading(section.heading)
        section_reasons = reasons.setdefault(heading, {})
        for placeholder in section.placeholders:
            section_reasons[placeholder.text] = placeholder.reason
    return reasons


def _all_plan_placeholder_reasons(plan: DocumentPlan | None) -> dict[str, set[str]]:
    """Return every declared plan reason for each placeholder text.

    A placeholder declared in any section may be reused in prose or structured
    table fields. When one distinct reason exists it is reused; multiple
    section-specific reasons fall back to a generic explanation.
    """
    reasons: dict[str, set[str]] = {}
    if plan is None:
        return reasons
    for section in plan.sections:
        for placeholder in section.placeholders:
            reasons.setdefault(placeholder.text, set()).add(placeholder.reason)
    return reasons


def _reason_for_prose(
    text: str,
    section_heading: str,
    location: str,
    plan_reasons: dict[str, dict[str, str]],
    global_plan_reasons: dict[str, set[str]],
) -> str:
    section_reasons = plan_reasons.get(_normalize_heading(section_heading), {})
    reason = section_reasons.get(text)
    if reason is not None:
        return reason
    declared_reasons = global_plan_reasons.get(text)
    if declared_reasons:
        if len(declared_reasons) == 1:
            return next(iter(declared_reasons))
        return AMBIGUOUS_PLACEHOLDER_REASON
    return UNKNOWN_PLACEHOLDER_REASON


def _reason_for_template(
    text: str,
    location: str,
    plan_reasons: dict[str, set[str]],
) -> str:
    reason = TEMPLATE_PLACEHOLDER_REASONS.get(text)
    if reason is not None:
        return reason
    declared_reasons = plan_reasons.get(text)
    if not declared_reasons:
        return UNKNOWN_PLACEHOLDER_REASON
    if len(declared_reasons) > 1:
        return AMBIGUOUS_PLACEHOLDER_REASON
    return next(iter(declared_reasons))


def _collect_prose(
    text: str,
    section_heading: str,
    location: str,
    plan_reasons: dict[str, dict[str, str]],
    global_plan_reasons: dict[str, set[str]],
    records: list[RenderedPlaceholder],
) -> None:
    for placeholder in PLACEHOLDER_RE.findall(text):
        records.append(
            RenderedPlaceholder(
                text=placeholder,
                location=location,
                reason=_reason_for_prose(
                    placeholder,
                    section_heading,
                    location,
                    plan_reasons,
                    global_plan_reasons,
                ),
            )
        )


def _collect_template(
    text: str,
    location: str,
    records: list[RenderedPlaceholder],
    plan_reasons: dict[str, set[str]],
) -> None:
    for placeholder in PLACEHOLDER_RE.findall(text):
        records.append(
            RenderedPlaceholder(
                text=placeholder,
                location=location,
                reason=_reason_for_template(placeholder, location, plan_reasons),
            )
        )


_HEADER_FOOTER_VARIANTS: tuple[tuple[str, str], ...] = (
    ("header", "primary header"),
    ("footer", "primary footer"),
    ("first_page_header", "first-page header"),
    ("first_page_footer", "first-page footer"),
    ("even_page_header", "even-page header"),
    ("even_page_footer", "even-page footer"),
)


def _scan_table(
    table: Any,
    location_prefix: str,
    records: list[RenderedPlaceholder],
    seen_cells: set[Any],
    plan_reasons: dict[str, set[str]],
) -> None:
    for row_index, row in enumerate(table.rows, start=1):
        for column_index, cell in enumerate(row.cells, start=1):
            cell_key = cell._tc
            if cell_key in seen_cells:
                continue
            seen_cells.add(cell_key)
            cell_location = f"{location_prefix}, row {row_index}, column {column_index}"
            for paragraph_index, paragraph in enumerate(cell.paragraphs, start=1):
                _collect_template(
                    paragraph.text,
                    f"{cell_location}, paragraph {paragraph_index}",
                    records,
                    plan_reasons,
                )
            for nested_index, nested_table in enumerate(cell.tables, start=1):
                _scan_table(
                    nested_table,
                    f"{cell_location} → nested table {nested_index}",
                    records,
                    seen_cells,
                    plan_reasons,
                )


def _scan_container(
    container: Any,
    location_prefix: str,
    records: list[RenderedPlaceholder],
    plan_reasons: dict[str, set[str]],
) -> None:
    for index, paragraph in enumerate(container.paragraphs, start=1):
        _collect_template(
            paragraph.text,
            f"{location_prefix}, paragraph {index}",
            records,
            plan_reasons,
        )

    seen_cells: set[Any] = set()
    for table_index, table in enumerate(container.tables, start=1):
        _scan_table(
            table,
            f"{location_prefix}, table {table_index}",
            records,
            seen_cells,
            plan_reasons,
        )


def _header_footer_entries(document: Document) -> list[dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for section_index, section in enumerate(document.sections, start=1):
        for attribute, label in _HEADER_FOOTER_VARIANTS:
            proxy = getattr(section, attribute)
            part = proxy.part
            key = str(part.partname)
            entry = entries.setdefault(
                key,
                {
                    "proxy": proxy,
                    "labels": set(),
                    "sections": set(),
                },
            )
            entry["labels"].add(label)
            entry["sections"].add(section_index)
    return list(entries.values())


def _header_footer_location(entry: dict[str, Any]) -> str:
    labels = " / ".join(label.capitalize() for label in sorted(entry["labels"]))
    sections = sorted(entry["sections"])
    section_text = (
        f"section {sections[0]}"
        if len(sections) == 1
        else f"sections {', '.join(str(section) for section in sections)}"
    )
    return f"{labels} ({section_text})"


def collect_rendered_placeholders(
    document: Document,
    plan: DocumentPlan | None,
) -> list[RenderedPlaceholder]:
    """Scan the actual DOCX and explain every rendered placeholder.

    Declared plan placeholders reuse their reason, ambiguous declarations get a
    generic explanation, and undeclared placeholders remain transparent instead
    of blocking document generation.
    """
    records: list[RenderedPlaceholder] = []
    plan_reasons = _plan_placeholder_reasons(plan)
    structured_plan_reasons = _all_plan_placeholder_reasons(plan)

    current_heading = "Document body"
    for index, paragraph in enumerate(document.paragraphs, start=1):
        if paragraph.style.name in ("Title", "Heading 1", "Heading 2"):
            current_heading = paragraph.text.strip() or current_heading
        _collect_prose(
            paragraph.text,
            current_heading,
            f"Section '{current_heading}', paragraph {index}",
            plan_reasons,
            structured_plan_reasons,
            records,
        )

    seen_body_cells: set[Any] = set()
    for table_index, table in enumerate(document.tables, start=1):
        _scan_table(
            table,
            f"Table {table_index}",
            records,
            seen_body_cells,
            structured_plan_reasons,
        )

    for entry in _header_footer_entries(document):
        # Header/footer placeholders must come from deterministic template
        # context, not arbitrary plan prose placeholders.
        _scan_container(
            entry["proxy"],
            _header_footer_location(entry),
            records,
            {},
        )

    return records
