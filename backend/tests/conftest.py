"""Shared deterministic fakes for graph tests."""

import pytest

from backend.app.agent import (
    ContentSection,
    DocumentContent,
    DocumentPlan,
    GeneratedDocument,
    ValidationReport,
)


class StaticWriter:
    """Writer double that mirrors the approved plan structure."""

    def __init__(
        self,
        content: DocumentContent | None = None,
        error: Exception | None = None,
    ) -> None:
        self._content = content
        self._error = error
        self.calls: list[tuple[str, DocumentPlan, list[str]]] = []

    async def generate(
        self,
        request: str,
        plan: DocumentPlan,
        tool_results: list[str] | tuple[str, ...] = (),
    ) -> DocumentContent:
        self.calls.append((request, plan, list(tool_results)))
        if self._error is not None:
            raise self._error
        if self._content is not None:
            return self._content
        return DocumentContent(
            document_type=plan.document_type,
            title=plan.title,
            sections=[
                ContentSection(
                    heading=section.heading,
                    paragraphs=["Generated placeholder content."],
                )
                for section in plan.sections
            ],
        )


class StaticDocxGenerator:
    """DOCX double that records calls without touching the filesystem."""

    def __init__(
        self,
        filename: str = "test-document-000000000000.docx",
        error: Exception | None = None,
    ) -> None:
        self._filename = filename
        self._error = error
        self.calls: list[tuple[DocumentContent, str]] = []

    def generate(
        self,
        content: DocumentContent,
        request: str,
        plan: DocumentPlan | None = None,
        today: object | None = None,
    ) -> GeneratedDocument:
        self.calls.append((content, request))
        if self._error is not None:
            raise self._error
        return GeneratedDocument(
            filename=self._filename,
            document_date="2026-11-07",
            document_version="1.0",
        )


class StaticValidator:
    """Validator double that returns preconfigured reports."""

    def __init__(self, reports: list[ValidationReport] | None = None) -> None:
        self._reports = list(reports or [])
        self.calls: list[tuple[DocumentContent, DocumentPlan]] = []

    async def validate(
        self,
        request: str,
        plan: DocumentPlan,
        content: DocumentContent,
        tool_results: list[str] | tuple[str, ...] = (),
    ) -> ValidationReport:
        self.calls.append((content, plan))
        if self._reports:
            return self._reports.pop(0)
        return ValidationReport()


@pytest.fixture
def fake_writer() -> StaticWriter:
    return StaticWriter()


@pytest.fixture
def fake_docx_generator() -> StaticDocxGenerator:
    return StaticDocxGenerator()


@pytest.fixture
def fake_validator() -> StaticValidator:
    return StaticValidator()
