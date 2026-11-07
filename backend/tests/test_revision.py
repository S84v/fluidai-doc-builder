"""Tests for controlled document revision."""

from typing import Any

import pytest
from langchain_core.messages import BaseMessage

from backend.app.agent import (
    ContentGenerationError,
    ContentSection,
    DocumentContent,
    DocumentPlan,
    DocumentType,
    DocumentWriter,
    PlanSection,
    RequirementConflict,
    ValidationIssue,
)
from backend.app.agent.document_writer import build_revision_messages
from backend.app.agent.validation import IssueSeverity
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
        provided_facts=["Revenue was confirmed at 10 units."],
        assumptions=["A one-page summary is appropriate."],
    )


def _content(plan: DocumentPlan) -> DocumentContent:
    return DocumentContent(
        document_type=plan.document_type,
        title=plan.title,
        sections=[
            ContentSection(heading=section.heading, paragraphs=["Original content."])
            for section in plan.sections
        ],
    )


def _blocking_issue() -> ValidationIssue:
    return ValidationIssue(
        code="missing_evidence",
        message="The findings lack supporting evidence.",
        severity=IssueSeverity.BLOCKING,
        section="Findings",
        suggestion="Add the confirmed revenue figure.",
    )


def test_revision_uses_json_mode_and_feedback() -> None:
    plan = _plan()
    original = _content(plan)
    revised = _content(plan)
    model = FakeStructuredModel(revised)
    writer = DocumentWriter(model)

    result = asyncio_run(
        writer.revise(
            "Create the report",
            plan,
            original,
            [_blocking_issue()],
            ["Revenue: 10"],
        )
    )

    assert result == revised
    assert model.schema is DocumentContent
    assert model.method == "json_mode"
    prompt = model.calls[0][1].content
    assert "missing_evidence" in prompt
    assert "Add the confirmed revenue figure." in prompt
    assert "Original content." in prompt
    assert "Revenue: 10" in prompt


def test_revision_preserves_plan_section_order() -> None:
    plan = _plan()
    revised = _content(plan)
    writer = DocumentWriter(FakeStructuredModel(revised))

    result = asyncio_run(
        writer.revise("Create the report", plan, _content(plan), [_blocking_issue()])
    )

    assert [section.heading for section in result.sections] == ["Overview", "Findings"]


def test_revision_rejects_wrong_document_type() -> None:
    plan = _plan()
    wrong_type = DocumentContent(
        document_type=DocumentType.MEETING_MINUTES,
        title=plan.title,
        sections=[
            ContentSection(heading=section.heading, paragraphs=["Content."])
            for section in plan.sections
        ],
    )
    writer = DocumentWriter(FakeStructuredModel(wrong_type))

    with pytest.raises(ContentGenerationError, match="document type"):
        asyncio_run(
            writer.revise(
                "Create the report",
                plan,
                _content(plan),
                [_blocking_issue()],
            )
        )


def test_revision_allows_changed_section_order() -> None:
    plan = _plan()
    reversed_content = DocumentContent(
        document_type=plan.document_type,
        title=plan.title,
        sections=[
            ContentSection(heading="Findings", paragraphs=["Content."]),
            ContentSection(heading="Overview", paragraphs=["Content."]),
        ],
    )
    writer = DocumentWriter(FakeStructuredModel(reversed_content))

    revised = asyncio_run(
        writer.revise(
            "Create the report",
            plan,
            _content(plan),
            [_blocking_issue()],
        )
    )

    assert [section.heading for section in revised.sections] == [
        "Findings",
        "Overview",
    ]


def test_revision_requires_blocking_issues_without_calling_model() -> None:
    model = FakeStructuredModel(_content(_plan()))
    writer = DocumentWriter(model)

    with pytest.raises(ContentGenerationError, match="No blocking issues"):
        asyncio_run(writer.revise("Create the report", _plan(), _content(_plan()), []))

    assert model.calls == []


def test_revision_wraps_model_failure() -> None:
    cause = RuntimeError("provider unavailable")
    writer = DocumentWriter(FakeStructuredModel(cause))

    with pytest.raises(ContentGenerationError, match="could not revise") as exc_info:
        asyncio_run(
            writer.revise(
                "Create the report",
                _plan(),
                _content(_plan()),
                [_blocking_issue()],
            )
        )

    assert exc_info.value.__cause__ is cause


def test_revision_messages_exclude_internal_reasoning() -> None:
    plan = _plan()
    messages = build_revision_messages(
        "request",
        plan,
        _content(plan),
        [_blocking_issue()],
        ["tool result"],
    )

    assert "tool result" in messages[1].content
    assert "reasoning_content" not in messages[1].content


def test_revision_messages_exclude_agent_only_planning() -> None:
    plan = _plan().model_copy(
        update={
            "conflicts": [
                RequirementConflict(
                    requirements=["Four-week rollout", "Three-week cleanup"],
                    impact="The dates conflict.",
                    resolution="Use a phased pilot rollout.",
                )
            ]
        }
    )
    messages = build_revision_messages(
        "Create the report",
        plan,
        _content(plan),
        [_blocking_issue()],
        ["tool result"],
    )

    prompt = messages[1].content
    assert "A one-page summary is appropriate." not in prompt
    assert "The dates conflict." not in prompt
    assert "Use a phased pilot rollout." not in prompt
    assert "Introduce the report." not in prompt


def test_revision_prompt_preserves_factuality_and_placeholders() -> None:
    messages = build_revision_messages(
        "request",
        _plan(),
        _content(_plan()),
        [_blocking_issue()],
        [],
    )
    system_prompt = messages[0].content

    assert "original user request remains the sole source" in system_prompt
    assert "Never introduce names, organisations, dates, figures" in system_prompt
    assert "Do not convert planning assumptions" in system_prompt
    assert "Do not replace a justified placeholder with an invented value" in (
        system_prompt
    )
    assert "Do not add filler paragraphs" in system_prompt
    assert "keep factual accuracy ahead of completeness" in system_prompt
    assert "An empty section body is acceptable" in system_prompt
    normalized = " ".join(system_prompt.split())
    assert "Use each placeholder only in the section where it is declared" in normalized
    assert "do not import a placeholder from another section" in normalized
    assert "preserve or add relative phases" in normalized


def asyncio_run(coroutine: Any) -> Any:
    import asyncio

    return asyncio.run(coroutine)
