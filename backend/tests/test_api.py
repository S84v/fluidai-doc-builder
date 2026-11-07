"""Tests for the SSE agent endpoint and secure document download."""

import asyncio
import io
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from docx import Document
from langchain_core.messages import AIMessage

from backend.app.agent import (
    ActionItem,
    ContentSection,
    DocumentContent,
    DocumentPlan,
    DocumentType,
    DocumentValidator,
    DocxGenerator,
    ExecutionStage,
    ExecutionTask,
    MeetingDetails,
    PlanningError,
    PlanPlaceholder,
    PlanSection,
    RenderedPlaceholder,
    ReviewOutput,
)
from backend.app.agent.graph import (
    DOCUMENT_FAILURE_RESPONSE,
    PLANNING_FAILURE_RESPONSE,
    REJECTION_RESPONSE,
    REVIEW_FAILURE_RESPONSE,
    REVISION_FAILURE_RESPONSE,
    TOOL_FAILURE_RESPONSE,
    TOOL_LIMIT_RESPONSE,
    VALIDATION_FAILURE_RESPONSE,
    compile_agent_graph,
)
from backend.app.api.agent import (
    DOCX_MEDIA_TYPE,
    _sse_event,
    stream_agent_events,
)
from backend.app.config import Settings
from backend.app.main import create_app
from backend.tests.plan_factory import make_execution_tasks

SENSITIVE_SENTINEL = "SECRET_INTERNAL_STATE"


class FakeGraph:
    def __init__(
        self,
        events: list[dict[str, Any]],
        error: Exception | None = None,
    ) -> None:
        self.events = events
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def astream_events(
        self,
        state: dict[str, Any],
        *,
        version: str = "v2",
    ):
        self.calls.append(state)
        for event in self.events:
            yield event
        if self.error is not None:
            raise self.error


def _settings() -> Settings:
    return Settings(_env_file=None, app_name="Test FluidAI")


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


def _parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in text.strip().split("\n\n"):
        if not block:
            continue
        event = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = line.removeprefix("data: ")
        events.append((event, json.loads(data)))
    return events


def _event(
    phase: str,
    node: str,
    output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "event": f"on_chain_{phase}",
        "name": node,
        "metadata": {"langgraph_node": node},
        "data": {"output": output} if phase == "end" else {"input": {}},
    }


def _plan(
    *,
    title: str = "Status Report",
    document_type: DocumentType = DocumentType.BUSINESS_REPORT,
) -> DocumentPlan:
    tasks = [
        *make_execution_tasks(),
        ExecutionTask(
            id="select_tools",
            title="Select required tools",
            description="Decide whether tools are required.",
            stage=ExecutionStage.TOOL_SELECTION,
        ),
        ExecutionTask(
            id="run_calculation",
            title="Run required calculation",
            description="Use the calculator when arithmetic is needed.",
            stage=ExecutionStage.TOOL_EXECUTION,
        ),
        ExecutionTask(
            id="revise_content",
            title="Revise if validation finds issues",
            description="Correct blocking validation issues.",
            stage=ExecutionStage.REVISION,
        ),
    ]
    return DocumentPlan(
        document_type=document_type,
        title=title,
        objective="Exercise the streaming contract.",
        document_date=None,
        document_version=None,
        sections=[
            PlanSection(
                heading="Overview",
                purpose="Introduce the document.",
                placeholders=[
                    PlanPlaceholder(
                        text="[Budget]",
                        reason="No approved budget was supplied.",
                    )
                ],
            )
        ],
        missing_information=["Budget"],
        assumptions=["A phased approach is acceptable."],
        clarification_questions=["What is the approved budget?"],
        conflicts=[],
        execution_tasks=tasks,
    )


def _success_events(
    plan: DocumentPlan,
    filename: str = "report-123456789abc.docx",
) -> list[dict[str, Any]]:
    final_text = f"Document generated successfully: {filename}"
    placeholder = RenderedPlaceholder(
        text="[Budget]",
        location="Section 'Overview', paragraph 3",
        reason="No approved budget was supplied.",
    )
    return [
        _event("start", "prepare_request"),
        _event(
            "end",
            "prepare_request",
            {"status": "understood", "normalized_request": "Create a report"},
        ),
        _event("start", "plan_document"),
        _event(
            "end",
            "plan_document",
            {"status": "planned", "planning_error": "", "plan": plan},
        ),
        _event("start", "select_tools"),
        _event("end", "select_tools", {"status": "tool_selection"}),
        _event("start", "write_document"),
        _event(
            "end",
            "write_document",
            {"status": "writing", "content": SENSITIVE_SENTINEL},
        ),
        _event("start", "validate_document"),
        _event(
            "end",
            "validate_document",
            {
                "status": "validating",
                "validation_report": SENSITIVE_SENTINEL,
            },
        ),
        _event("start", "generate_docx"),
        _event(
            "end",
            "generate_docx",
            {
                "status": "docx_generation",
                "output_filename": filename,
                "document_date": "2026-11-07",
                "document_version": "1.0",
                "rendered_placeholders": [placeholder],
            },
        ),
        _event("start", "finalize"),
        _event(
            "end",
            "finalize",
            {"status": "completed", "terminal_reason": "completed"},
        ),
        _event("start", "final_response"),
        _event("end", "final_response", {"final_response": final_text}),
    ]


async def test_successful_sse_stream_has_plan_tasks_placeholders_and_complete() -> None:
    plan = _plan()
    app = create_app(
        _settings(),
        graph_factory=lambda: FakeGraph(_success_events(plan)),
    )

    async with _client(app) as client:
        response = await client.post("/agent", json={"request": "Create a report"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"

    events = _parse_sse(response.text)
    event_names = [event for event, _ in events]
    assert event_names[0] == "plan"
    assert event_names[-2:] == ["token", "complete"]
    assert event_names.count("metadata") == 1
    assert event_names.index("metadata") < event_names.index("placeholders")
    assert event_names.index("placeholders") < event_names.index("token")
    assert "error" not in event_names

    metadata = next(data for event, data in events if event == "metadata")
    assert metadata == {
        "document_date": "2026-11-07",
        "document_version": "1.0",
    }

    plan_event = events[0][1]
    tasks = {task["id"]: task for task in plan_event["tasks"]}
    assert tasks["understand_request"]["status"] == "completed"
    assert tasks["plan_document"]["status"] == "completed"
    assert tasks["write_document"]["status"] == "pending"
    assert tasks["run_calculation"]["optional"] is True
    assert tasks["revise_content"]["optional"] is True
    assert tasks["run_calculation"]["stage"] == "tool_execution"
    assert plan_event["missing_information"] == ["Budget"]
    assert plan_event["assumptions"] == ["A phased approach is acceptable."]
    assert plan_event["placeholders"] == [
        {
            "text": "[Budget]",
            "section": "Overview",
            "reason": "No approved budget was supplied.",
        }
    ]

    progress = [data for event, data in events if event == "progress"]
    progress_by_stage = {}
    for item in progress:
        progress_by_stage.setdefault(item["stage"], []).append(item)
    assert progress_by_stage["tool_execution"][0]["status"] == "skipped"
    assert progress_by_stage["revision"][0]["status"] == "skipped"
    assert progress_by_stage["document_writing"][0]["status"] == "active"
    assert progress_by_stage["document_writing"][-1]["status"] == "completed"
    assert progress_by_stage["docx_generation"][-1]["status"] == "completed"
    assert progress_by_stage["finalization"][-1]["status"] == "completed"
    assert all("task_ids" in item for item in progress)

    placeholders = next(data for event, data in events if event == "placeholders")
    assert placeholders["placeholders"][0]["text"] == "[Budget]"
    assert placeholders["placeholders"][0]["location"].startswith("Section")

    token = next(data for event, data in events if event == "token")
    complete = next(data for event, data in events if event == "complete")
    assert token["text"] == (
        "Document generated successfully: report-123456789abc.docx"
    )
    assert complete == {
        "filename": "report-123456789abc.docx",
        "url": "/documents/report-123456789abc.docx",
    }
    assert SENSITIVE_SENTINEL not in response.text


async def test_rejected_request_emits_error_not_token() -> None:
    events = [
        _event("start", "prepare_request"),
        _event("end", "prepare_request", {"status": "rejected"}),
        _event("start", "reject"),
        _event(
            "end",
            "reject",
            {"status": "rejected", "terminal_reason": "rejected"},
        ),
        _event("start", "final_response"),
        _event("end", "final_response", {"final_response": REJECTION_RESPONSE}),
    ]
    app = create_app(_settings(), graph_factory=lambda: FakeGraph(events))

    async with _client(app) as client:
        response = await client.post("/agent", json={"request": "   "})

    parsed = _parse_sse(response.text)
    names = [event for event, _ in parsed]
    assert "plan" not in names
    assert "token" not in names
    assert names[-1] == "error"
    assert parsed[-1][1] == {"message": REJECTION_RESPONSE}
    assert SENSITIVE_SENTINEL not in response.text


@pytest.mark.parametrize(
    ("node", "reason", "message"),
    [
        ("planning_failed", "planning_failed", PLANNING_FAILURE_RESPONSE),
        ("tool_failed", "tool_failed", TOOL_FAILURE_RESPONSE),
        ("tool_limit_reached", "tool_limit_reached", TOOL_LIMIT_RESPONSE),
        ("content_failed", "content_failed", DOCUMENT_FAILURE_RESPONSE),
        ("validation_failed", "validation_failed", VALIDATION_FAILURE_RESPONSE),
        ("review_failed", "review_failed", REVIEW_FAILURE_RESPONSE),
        ("revision_failed", "revision_failed", REVISION_FAILURE_RESPONSE),
        ("docx_failed", "docx_failed", DOCUMENT_FAILURE_RESPONSE),
    ],
)
async def test_terminal_failures_emit_safe_error(
    node: str,
    reason: str,
    message: str,
) -> None:
    events = [
        _event("start", "prepare_request"),
        _event("end", "prepare_request", {"status": "understood"}),
        _event("start", node),
        _event("end", node, {"status": "failed", "terminal_reason": reason}),
        _event("start", "final_response"),
        _event("end", "final_response", {"final_response": message}),
    ]
    app = create_app(_settings(), graph_factory=lambda: FakeGraph(events))

    async with _client(app) as client:
        response = await client.post("/agent", json={"request": "Create a report"})

    parsed = _parse_sse(response.text)
    assert [event for event, _ in parsed][-1] == "error"
    assert parsed[-1][1] == {"message": message}
    assert SENSITIVE_SENTINEL not in response.text


async def test_failure_after_plan_marks_task_failed() -> None:
    plan = _plan()
    events = [
        _event("start", "prepare_request"),
        _event("end", "prepare_request", {"status": "understood"}),
        _event("start", "plan_document"),
        _event("end", "plan_document", {"status": "planned", "plan": plan}),
        _event("start", "select_tools"),
        _event(
            "end",
            "select_tools",
            {"status": "failed", "planning_error": ""},
        ),
        _event("start", "tool_failed"),
        _event(
            "end",
            "tool_failed",
            {"status": "failed", "terminal_reason": "tool_failed"},
        ),
        _event("start", "final_response"),
        _event(
            "end",
            "final_response",
            {"final_response": TOOL_FAILURE_RESPONSE},
        ),
    ]
    app = create_app(_settings(), graph_factory=lambda: FakeGraph(events))

    async with _client(app) as client:
        response = await client.post("/agent", json={"request": "Create a report"})

    parsed = _parse_sse(response.text)
    progress = [data for event, data in parsed if event == "progress"]
    failed = [item for item in progress if item["status"] == "failed"]
    assert failed
    assert failed[0]["stage"] == "tool_selection"
    assert parsed[-1][0] == "error"


async def test_failed_docx_generation_does_not_emit_placeholders() -> None:
    plan = _plan()
    events = [
        _event("start", "prepare_request"),
        _event("end", "prepare_request", {"status": "understood"}),
        _event("start", "plan_document"),
        _event("end", "plan_document", {"status": "planned", "plan": plan}),
        _event("start", "generate_docx"),
        _event(
            "end",
            "generate_docx",
            {
                "status": "failed",
                "output_filename": "",
                "rendered_placeholders": [],
            },
        ),
        _event("start", "docx_failed"),
        _event(
            "end",
            "docx_failed",
            {"status": "failed", "terminal_reason": "docx_failed"},
        ),
        _event("start", "final_response"),
        _event(
            "end",
            "final_response",
            {"final_response": DOCUMENT_FAILURE_RESPONSE},
        ),
    ]
    app = create_app(_settings(), graph_factory=lambda: FakeGraph(events))

    async with _client(app) as client:
        response = await client.post("/agent", json={"request": "Create a report"})

    parsed = _parse_sse(response.text)
    names = [event for event, _ in parsed]
    assert "metadata" not in names
    assert "placeholders" not in names
    assert names[-1] == "error"
    assert parsed[-1][1] == {"message": DOCUMENT_FAILURE_RESPONSE}


async def test_exception_after_completion_keeps_completed_tasks() -> None:
    plan = _plan()
    events = [
        _event("start", "prepare_request"),
        _event("end", "prepare_request", {"status": "understood"}),
        _event("start", "plan_document"),
        _event("end", "plan_document", {"status": "planned", "plan": plan}),
        _event("start", "select_tools"),
        _event("end", "select_tools", {"status": "tool_selection"}),
        _event("start", "write_document"),
        _event(
            "end",
            "write_document",
            {"status": "writing", "document_error": ""},
        ),
    ]

    class FailingGraph(FakeGraph):
        async def astream_events(self, state: dict[str, Any], *, version: str = "v2"):
            for event in self.events:
                yield event
            raise RuntimeError(f"provider failure {SENSITIVE_SENTINEL}")

    app = create_app(_settings(), graph_factory=lambda: FailingGraph(events))

    async with _client(app) as client:
        response = await client.post("/agent", json={"request": "Create a report"})

    parsed = _parse_sse(response.text)
    progress = [data for event, data in parsed if event == "progress"]
    failed = [item for item in progress if item["status"] == "failed"]
    assert failed == []
    assert parsed[-1][0] == "error"
    assert SENSITIVE_SENTINEL not in response.text


async def test_unexpected_graph_exception_fails_active_task_and_is_sanitized() -> None:
    plan = _plan()
    events = [
        _event("start", "prepare_request"),
        _event("end", "prepare_request", {"status": "understood"}),
        _event("start", "plan_document"),
        _event("end", "plan_document", {"status": "planned", "plan": plan}),
        _event("start", "select_tools"),
        _event("end", "select_tools", {"status": "tool_selection"}),
        _event("start", "write_document"),
    ]

    class FailingGraph(FakeGraph):
        async def astream_events(self, state: dict[str, Any], *, version: str = "v2"):
            for event in self.events:
                yield event
            raise RuntimeError(f"provider failure {SENSITIVE_SENTINEL}")

    app = create_app(_settings(), graph_factory=lambda: FailingGraph(events))

    async with _client(app) as client:
        response = await client.post("/agent", json={"request": "Create a report"})

    parsed = _parse_sse(response.text)
    progress = [data for event, data in parsed if event == "progress"]
    failed = [item for item in progress if item["status"] == "failed"]
    assert failed
    assert failed[-1]["stage"] == "document_writing"
    assert parsed[-1][0] == "error"
    assert SENSITIVE_SENTINEL not in response.text
    assert "provider failure" not in response.text


async def test_graph_factory_exception_is_sanitized() -> None:
    def factory() -> FakeGraph:
        raise RuntimeError(f"factory failure {SENSITIVE_SENTINEL}")

    app = create_app(_settings(), graph_factory=factory)

    async with _client(app) as client:
        response = await client.post("/agent", json={"request": "Create a report"})

    parsed = _parse_sse(response.text)
    assert parsed[-1][0] == "error"
    assert SENSITIVE_SENTINEL not in response.text
    assert "factory failure" not in response.text


async def test_concurrent_requests_are_isolated() -> None:
    class Factory:
        def __init__(self) -> None:
            self.count = 0

        def __call__(self) -> FakeGraph:
            self.count += 1
            plan = _plan(title=f"Report {self.count}")
            filename = f"report-{self.count:012x}.docx"
            return FakeGraph(_success_events(plan, filename))

    factory = Factory()
    app = create_app(_settings(), graph_factory=factory)

    async with _client(app) as client:
        first, second = await asyncio.gather(
            client.post("/agent", json={"request": "First"}),
            client.post("/agent", json={"request": "Second"}),
        )

    first_complete = next(
        data for event, data in _parse_sse(first.text) if event == "complete"
    )
    second_complete = next(
        data for event, data in _parse_sse(second.text) if event == "complete"
    )
    assert first_complete["filename"] != second_complete["filename"]
    assert factory.count == 2


async def test_stream_cancellation_propagates() -> None:
    class CancelledGraph:
        async def astream_events(
            self,
            state: dict[str, Any],
            *,
            version: str = "v2",
        ):
            yield _event("start", "prepare_request")
            raise asyncio.CancelledError

    generator = stream_agent_events(lambda: CancelledGraph(), "request")

    with pytest.raises(asyncio.CancelledError):
        async for _ in generator:
            pass


async def test_document_download_success(tmp_path: Path) -> None:
    filename = "report-123456789abc.docx"
    (tmp_path / filename).write_bytes(b"docx-bytes")
    app = create_app(
        _settings(),
        graph_factory=lambda: FakeGraph([]),
        output_dir=tmp_path,
    )

    async with _client(app) as client:
        response = await client.get(f"/documents/{filename}")

    assert response.status_code == 200
    assert response.headers["content-type"] == DOCX_MEDIA_TYPE
    assert filename in response.headers["content-disposition"]
    assert response.content == b"docx-bytes"


async def test_document_download_missing_returns_404(tmp_path: Path) -> None:
    app = create_app(
        _settings(),
        graph_factory=lambda: FakeGraph([]),
        output_dir=tmp_path,
    )

    async with _client(app) as client:
        response = await client.get("/documents/report-123456789abc.docx")

    assert response.status_code == 404


@pytest.mark.parametrize(
    "filename",
    [
        "report.docx",
        "report-123456789abc.pdf",
        "REPORT-123456789ABC.docx",
        "report-123456789ab.docx",
        "report-123456789abcd.docx",
    ],
)
async def test_invalid_document_filenames_return_404(
    tmp_path: Path,
    filename: str,
) -> None:
    (tmp_path / filename).write_bytes(b"docx-bytes")
    app = create_app(
        _settings(),
        graph_factory=lambda: FakeGraph([]),
        output_dir=tmp_path,
    )

    async with _client(app) as client:
        response = await client.get(f"/documents/{filename}")

    assert response.status_code == 404


@pytest.mark.parametrize(
    "url",
    [
        "/documents/..%2Fsecret.docx",
        "/documents/%2e%2e%2fsecret.docx",
        "/documents/sub/report-123456789abc.docx",
    ],
)
async def test_path_traversal_returns_404(tmp_path: Path, url: str) -> None:
    (tmp_path.parent / "secret.docx").write_bytes(b"secret")
    app = create_app(
        _settings(),
        graph_factory=lambda: FakeGraph([]),
        output_dir=tmp_path,
    )

    async with _client(app) as client:
        response = await client.get(url)

    assert response.status_code == 404


async def test_symlink_escaping_output_directory_returns_404(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    outside = tmp_path / "outside.docx"
    outside.write_bytes(b"outside")
    (output_dir / "report-123456789abc.docx").symlink_to(outside)

    app = create_app(
        _settings(),
        graph_factory=lambda: FakeGraph([]),
        output_dir=output_dir,
    )

    async with _client(app) as client:
        response = await client.get("/documents/report-123456789abc.docx")

    assert response.status_code == 404


class _NoIssueReviewer:
    async def review(
        self,
        request: str,
        plan: DocumentPlan,
        content: DocumentContent,
        tool_results: list[str] | tuple[str, ...] = (),
    ) -> ReviewOutput:
        return ReviewOutput()


class _StaticPlanner:
    def __init__(self, plan: DocumentPlan, error: Exception | None = None) -> None:
        self.plan = plan
        self.error = error

    async def create_plan(self, request: str) -> DocumentPlan:
        if self.error is not None:
            raise self.error
        return self.plan


class _NoToolSelector:
    async def select(self, messages: Any) -> AIMessage:
        return AIMessage(content="No tool needed.")


class _StaticWriter:
    def __init__(self, content: DocumentContent) -> None:
        self.content = content

    async def generate(
        self,
        request: str,
        plan: DocumentPlan,
        tool_results: list[str] | tuple[str, ...] = (),
    ) -> DocumentContent:
        return self.content

    async def revise(
        self,
        request: str,
        plan: DocumentPlan,
        content: DocumentContent,
        blocking_issues: Any,
        tool_results: list[str] | tuple[str, ...] = (),
    ) -> DocumentContent:
        return self.content


def _meeting_plan_and_content() -> tuple[DocumentPlan, DocumentContent]:
    headings = ("Agenda", "Decisions", "Action Items")
    plan = DocumentPlan(
        document_type=DocumentType.MEETING_MINUTES,
        title="Atlas Kickoff Minutes",
        objective="Record the kickoff meeting.",
        document_date="2026-11-06",
        document_version="1.0",
        sections=[
            PlanSection(heading=heading, purpose=f"Purpose of {heading}.")
            for heading in headings
        ],
        execution_tasks=make_execution_tasks(),
    )
    content = DocumentContent(
        document_type=DocumentType.MEETING_MINUTES,
        title=plan.title,
        meeting=MeetingDetails(
            date="2026-11-06",
            attendees=["Anna", "Ben"],
        ),
        action_items=[ActionItem(action="Send notes", status="Open")],
        sections=[
            ContentSection(heading=heading, paragraphs=[f"{heading} content."])
            for heading in headings
        ],
    )
    return plan, content


async def test_actual_graph_end_to_end_sse_and_download(tmp_path: Path) -> None:
    plan, content = _meeting_plan_and_content()

    def factory() -> Any:
        return compile_agent_graph(
            planner=_StaticPlanner(plan),
            tool_selector=_NoToolSelector(),
            writer=_StaticWriter(content),
            docx_generator=DocxGenerator(tmp_path),
            validator=DocumentValidator(_NoIssueReviewer()),
        )

    app = create_app(_settings(), graph_factory=factory, output_dir=tmp_path)

    async with _client(app) as client:
        response = await client.post("/agent", json={"request": "Create minutes"})
        parsed = _parse_sse(response.text)
        complete = next(data for event, data in parsed if event == "complete")
        download = await client.get(complete["url"])

    assert response.status_code == 200
    assert download.status_code == 200
    assert download.headers["content-type"] == DOCX_MEDIA_TYPE
    assert SENSITIVE_SENTINEL not in response.text

    names = [event for event, _ in parsed]
    assert names[0] == "plan"
    assert names[-2:] == ["token", "complete"]
    assert names.count("metadata") == 1
    assert names.index("metadata") < names.index("placeholders")
    assert names.index("placeholders") < names.index("token")
    metadata = next(data for event, data in parsed if event == "metadata")
    assert metadata == {
        "document_date": "2026-11-06",
        "document_version": "1.0",
    }
    plan_event = parsed[0][1]
    assert plan_event["document_type"] == "meeting_minutes"
    assert plan_event["document_date"] == "2026-11-06"
    assert plan_event["document_version"] == "1.0"

    progress = [data for event, data in parsed if event == "progress"]
    stages = [item["stage"] for item in progress]
    assert stages.index("document_writing") < stages.index("validation")
    assert stages.index("validation") < stages.index("docx_generation")
    assert stages.index("docx_generation") < stages.index("finalization")

    document = Document(io.BytesIO(download.content))
    assert document.paragraphs[0].text == content.title
    values = {
        cell.text
        for table in document.tables
        for row in table.rows
        for cell in row.cells
    }
    assert "Meeting Date" in values
    assert "Document Date" in values
    assert "Document Version" in values
    assert "Action" in values


async def test_actual_graph_terminal_failure_emits_error_not_complete(
    tmp_path: Path,
) -> None:
    def factory() -> Any:
        return compile_agent_graph(
            planner=_StaticPlanner(
                DocumentPlan(
                    document_type=DocumentType.BUSINESS_REPORT,
                    title="Report",
                    objective="Report.",
                    sections=[PlanSection(heading="Overview", purpose="Overview.")],
                    execution_tasks=make_execution_tasks(),
                ),
                error=PlanningError("planning failed"),
            ),
            tool_selector=_NoToolSelector(),
            writer=_StaticWriter(_meeting_plan_and_content()[1]),
            docx_generator=DocxGenerator(tmp_path),
            validator=DocumentValidator(_NoIssueReviewer()),
        )

    app = create_app(_settings(), graph_factory=factory, output_dir=tmp_path)

    async with _client(app) as client:
        response = await client.post("/agent", json={"request": "Create a report"})

    parsed = _parse_sse(response.text)
    names = [event for event, _ in parsed]
    assert "plan" not in names
    assert "complete" not in names
    assert "token" not in names
    assert names[-1] == "error"
    assert parsed[-1][1]["message"] == PLANNING_FAILURE_RESPONSE


async def test_health_endpoint_contract_is_preserved() -> None:
    app = create_app(_settings(), graph_factory=lambda: FakeGraph([]))

    async with _client(app) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_sse_event_formatting() -> None:
    event = _sse_event("progress", {"message": "Working"})

    assert event == b'event: progress\ndata: {"message": "Working"}\n\n'
