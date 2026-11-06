"""Graph integration tests for writing and DOCX generation."""

from pathlib import Path
from typing import Any

from docx import Document
from langchain_core.messages import AIMessage, BaseMessage

from backend.app.agent import (
    ContentGenerationError,
    ContentSection,
    DocumentContent,
    DocumentPlan,
    DocumentType,
    DocxGenerationError,
    DocxGenerator,
    PlanSection,
    run_agent,
)

REASONING_SENTINEL = "INTERNAL_REASONING_MUST_NOT_REACH_WRITER"


def _plan() -> DocumentPlan:
    return DocumentPlan(
        document_type=DocumentType.BUSINESS_REPORT,
        title="Status Report",
        objective="Plan a structured business document.",
        sections=[
            PlanSection(heading="Overview", purpose="Introduce the document."),
            PlanSection(heading="Outcome", purpose="Describe the outcome."),
        ],
    )


class FakePlanner:
    async def create_plan(self, request: str) -> DocumentPlan:
        return _plan()


class FakeToolSelector:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)
        self.calls: list[list[BaseMessage]] = []

    async def select(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls.append(list(messages))
        return self._responses.pop(0)


class NoToolSelector:
    async def select(self, messages: list[BaseMessage]) -> AIMessage:
        return AIMessage(content="No tool needed.")


class FailingWriter:
    async def generate(
        self,
        request: str,
        plan: DocumentPlan,
        tool_results: list[str] | tuple[str, ...] = (),
    ) -> Any:
        raise ContentGenerationError("content generation failed")


class FailingDocxGenerator:
    def generate(self, content: Any, request: str) -> str:
        raise DocxGenerationError("docx generation failed")


async def test_tool_results_reach_writer_without_reasoning(
    fake_writer: Any,
    fake_docx_generator: Any,
) -> None:
    selector = FakeToolSelector(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate",
                        "args": {"expression": "17 * 23"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
                additional_kwargs={"reasoning_content": REASONING_SENTINEL},
            ),
            AIMessage(content="The result is 391."),
        ]
    )

    result = await run_agent(
        "What is 17 * 23?",
        planner=FakePlanner(),
        tool_selector=selector,
        writer=fake_writer,
        docx_generator=fake_docx_generator,
    )

    assert result["status"] == "completed"
    assert len(fake_writer.calls) == 1
    writer_tool_results = fake_writer.calls[0][2]
    assert writer_tool_results == ["391"]
    assert REASONING_SENTINEL not in writer_tool_results


async def test_real_docx_is_created_after_writing(
    tmp_path: Path,
    fake_writer: Any,
) -> None:
    result = await run_agent(
        "Create a status report",
        planner=FakePlanner(),
        tool_selector=NoToolSelector(),
        writer=fake_writer,
        docx_generator=DocxGenerator(tmp_path),
    )

    assert result["status"] == "completed"
    assert result["content"] is not None
    assert result["output_filename"]
    destination = tmp_path / result["output_filename"]
    assert destination.exists()
    reopened = Document(destination)
    assert reopened.paragraphs[0].text == _plan().title


async def test_content_generation_failure_routes_to_safe_failure() -> None:
    result = await run_agent(
        "Create a status report",
        planner=FakePlanner(),
        tool_selector=NoToolSelector(),
        writer=FailingWriter(),
    )

    assert result["status"] == "failed"
    assert result["content"] is None
    assert result["output_filename"] == ""
    assert result["document_error"] == "content generation failed"
    assert result["final_response"] == (
        "Unable to generate the document. Please try again."
    )


async def test_docx_generation_failure_routes_to_safe_failure(
    fake_writer: Any,
) -> None:
    result = await run_agent(
        "Create a status report",
        planner=FakePlanner(),
        tool_selector=NoToolSelector(),
        writer=fake_writer,
        docx_generator=FailingDocxGenerator(),
    )

    assert result["status"] == "failed"
    assert result["content"] is not None
    assert result["output_filename"] == ""
    assert result["document_error"] == "docx generation failed"
    assert result["final_response"] == (
        "Unable to generate the document. Please try again."
    )


async def test_filesystem_failure_routes_to_safe_failure(
    tmp_path: Path,
    fake_writer: Any,
) -> None:
    blocked_path = tmp_path / "not-a-directory"
    blocked_path.write_text("blocked", encoding="utf-8")

    result = await run_agent(
        "Create a status report",
        planner=FakePlanner(),
        tool_selector=NoToolSelector(),
        writer=fake_writer,
        output_dir=blocked_path,
    )

    assert result["status"] == "failed"
    assert result["output_filename"] == ""
    assert "output directory" in result["document_error"]


async def test_document_state_is_request_local(
    fake_writer: Any,
    fake_docx_generator: Any,
) -> None:
    first = await run_agent(
        "Create the first report",
        planner=FakePlanner(),
        tool_selector=NoToolSelector(),
        writer=fake_writer,
        docx_generator=fake_docx_generator,
    )
    second = await run_agent(
        "Create the second report",
        planner=FakePlanner(),
        tool_selector=NoToolSelector(),
        writer=fake_writer,
        docx_generator=fake_docx_generator,
    )

    assert first["content"] is not second["content"]
    assert first["messages"] is not second["messages"]
    assert first["document_error"] == ""
    assert second["document_error"] == ""

    first["messages"].append(AIMessage(content="mutated"))
    assert len(second["messages"]) == 3


async def test_tool_message_content_only_is_extracted(
    fake_docx_generator: Any,
) -> None:
    selector = FakeToolSelector(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate",
                        "args": {"expression": "1 + 1"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
                additional_kwargs={"reasoning_content": REASONING_SENTINEL},
            ),
            AIMessage(content="2"),
        ]
    )

    class RecordingWriter:
        def __init__(self) -> None:
            self.tool_results: list[str] | None = None

        async def generate(
            self,
            request: str,
            plan: DocumentPlan,
            tool_results: list[str] | tuple[str, ...] = (),
        ) -> Any:
            self.tool_results = list(tool_results)
            return DocumentContent(
                document_type=plan.document_type,
                title=plan.title,
                sections=[
                    ContentSection(
                        heading=section.heading,
                        paragraphs=["Content."],
                    )
                    for section in plan.sections
                ],
            )

    writer = RecordingWriter()
    result = await run_agent(
        "Calculate 1 + 1.",
        planner=FakePlanner(),
        tool_selector=selector,
        writer=writer,
        docx_generator=fake_docx_generator,
    )

    assert result["status"] == "completed"
    assert writer.tool_results == ["2"]
    assert REASONING_SENTINEL not in writer.tool_results
