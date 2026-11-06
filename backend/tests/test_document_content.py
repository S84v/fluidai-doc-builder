"""Tests for document-content models and the document writer."""

from typing import Any

import pytest
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import ValidationError

from backend.app.agent import (
    ActionItem,
    ContentGenerationError,
    ContentSection,
    DocumentContent,
    DocumentMetadata,
    DocumentPlan,
    DocumentType,
    DocumentWriter,
    MeetingDetails,
    Milestone,
    PlanSection,
    Requirement,
    SopStep,
    TechnicalComponent,
)
from backend.app.agent.document_writer import (
    DOCUMENT_WRITING_PROMPT,
    build_writing_messages,
)


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
        document_type=DocumentType.BUSINESS_REPORT,
        title="Quarterly Business Report",
        objective="Report quarterly progress.",
        sections=[
            PlanSection(heading="Overview", purpose="Introduce the report."),
            PlanSection(heading="Risks", purpose="Describe known risks."),
        ],
        provided_facts=["Revenue was confirmed at 10 units."],
        assumptions=["A one-page summary is appropriate."],
        missing_information=["Budget"],
        clarification_questions=["What is the approved budget?"],
    )


def _content(
    plan: DocumentPlan,
    *,
    sections: list[ContentSection] | None = None,
    document_type: DocumentType | None = None,
) -> DocumentContent:
    return DocumentContent(
        document_type=document_type or plan.document_type,
        title=plan.title,
        sections=sections
        or [
            ContentSection(
                heading=section.heading,
                paragraphs=[f"Content for {section.heading}."],
                bullets=["[Budget]"] if section.heading == "Risks" else [],
            )
            for section in plan.sections
        ],
    )


def test_document_content_requires_at_least_one_section() -> None:
    with pytest.raises(ValidationError):
        DocumentContent(
            document_type=DocumentType.BUSINESS_REPORT,
            title="Report",
            sections=[],
        )


def test_content_section_requires_a_paragraph_or_bullet() -> None:
    with pytest.raises(ValidationError):
        ContentSection(heading="Empty")


def test_content_section_rejects_empty_items() -> None:
    with pytest.raises(ValidationError):
        ContentSection(heading="Overview", paragraphs=["   "])


def test_document_content_accepts_type_specific_structures() -> None:
    content = DocumentContent(
        document_type=DocumentType.MEETING_MINUTES,
        title="Kickoff Minutes",
        audience="Project Team",
        metadata=DocumentMetadata(date="2026-11-06", version="1.0"),
        meeting=MeetingDetails(
            date="2026-11-06",
            attendees=["Anna", "Ben"],
        ),
        action_items=[ActionItem(action="Send notes")],
        milestones=[Milestone(name="Phase one")],
        sop_steps=[SopStep(instruction="Open the system.")],
        requirements=[Requirement(description="Generate a document.")],
        components=[
            TechnicalComponent(
                name="API",
                responsibility="Serve requests.",
            )
        ],
        sections=[ContentSection(heading="Overview", paragraphs=["Content."])],
    )

    assert content.meeting is not None
    assert content.meeting.attendees == ["Anna", "Ben"]
    assert content.action_items[0].owner is None
    assert content.milestones[0].name == "Phase one"
    assert content.sop_steps[0].instruction == "Open the system."
    assert content.requirements[0].description == "Generate a document."
    assert content.components[0].name == "API"


def test_action_item_requires_an_action() -> None:
    with pytest.raises(ValidationError):
        ActionItem(action="")


def test_writing_prompt_includes_structured_content_guidance() -> None:
    assert "meeting_minutes" in DOCUMENT_WRITING_PROMPT
    assert "action_items" in DOCUMENT_WRITING_PROMPT
    assert "milestones" in DOCUMENT_WRITING_PROMPT
    assert "sop_steps" in DOCUMENT_WRITING_PROMPT
    assert "requirements" in DOCUMENT_WRITING_PROMPT
    assert "components" in DOCUMENT_WRITING_PROMPT


async def test_writer_uses_json_mode_structured_output() -> None:
    plan = _plan()
    model = FakeStructuredModel(_content(plan))
    writer = DocumentWriter(model)

    result = await writer.generate("Create the report", plan)

    assert result.document_type == plan.document_type
    assert model.schema is DocumentContent
    assert model.method == "json_mode"
    assert len(model.calls) == 1


async def test_writer_preserves_plan_section_order() -> None:
    plan = _plan()
    content = _content(plan)
    writer = DocumentWriter(FakeStructuredModel(content))

    result = await writer.generate("Create the report", plan)

    assert [section.heading for section in result.sections] == [
        "Overview",
        "Risks",
    ]


async def test_writer_rejects_section_order_mismatch() -> None:
    plan = _plan()
    reversed_content = DocumentContent(
        document_type=plan.document_type,
        title=plan.title,
        sections=[
            ContentSection(heading="Risks", paragraphs=["Risks content."]),
            ContentSection(heading="Overview", paragraphs=["Overview content."]),
        ],
    )
    writer = DocumentWriter(FakeStructuredModel(reversed_content))

    with pytest.raises(ContentGenerationError, match="approved plan order"):
        await writer.generate("Create the report", plan)


async def test_writer_rejects_document_type_mismatch() -> None:
    plan = _plan()
    content = _content(plan, document_type=DocumentType.MEETING_MINUTES)
    writer = DocumentWriter(FakeStructuredModel(content))

    with pytest.raises(ContentGenerationError, match="document type"):
        await writer.generate("Create the report", plan)


def test_writing_messages_include_facts_assumptions_and_tool_results() -> None:
    plan = _plan()

    messages = build_writing_messages(
        "Create the quarterly report",
        plan,
        ["Calculated total: 42"],
    )

    assert isinstance(messages[1], HumanMessage)
    prompt = messages[1].content
    assert "Create the quarterly report" in prompt
    assert "Revenue was confirmed at 10 units." in prompt
    assert "A one-page summary is appropriate." in prompt
    assert "What is the approved budget?" in prompt
    assert "Calculated total: 42" in prompt
    assert "[Budget]" not in prompt


async def test_writer_keeps_placeholders_in_generated_content() -> None:
    plan = _plan()
    content = _content(plan)
    writer = DocumentWriter(FakeStructuredModel(content))

    result = await writer.generate("Create the report", plan)

    assert result.sections[1].bullets == ["[Budget]"]


async def test_writer_wraps_model_failure() -> None:
    cause = RuntimeError("provider unavailable")
    writer = DocumentWriter(FakeStructuredModel(cause))

    with pytest.raises(ContentGenerationError, match="could not produce") as exc_info:
        await writer.generate("Create the report", _plan())

    assert exc_info.value.__cause__ is cause


async def test_writer_rejects_invalid_structured_output() -> None:
    writer = DocumentWriter(
        FakeStructuredModel({"document_type": "business_report", "title": "Incomplete"})
    )

    with pytest.raises(ContentGenerationError, match="invalid document content"):
        await writer.generate("Create the report", _plan())


async def test_writer_rejects_empty_request_without_calling_model() -> None:
    plan = _plan()
    model = FakeStructuredModel(_content(plan))
    writer = DocumentWriter(model)

    with pytest.raises(ContentGenerationError, match="empty request"):
        await writer.generate("   ", plan)

    assert model.calls == []
