"""Tests for the SSE agent endpoint and secure document download."""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage

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
from backend.app.agent.models import DocumentPlan, DocumentType, PlanSection
from backend.app.api.agent import (
    DOCX_MEDIA_TYPE,
    PROGRESS_MESSAGES,
    _sse_event,
    stream_agent_events,
)
from backend.app.config import Settings
from backend.app.main import create_app

SENSITIVE_SENTINEL = "SECRET_INTERNAL_STATE"


class FakeGraph:
    def __init__(
        self,
        chunks: list[dict[str, Any]],
        error: Exception | None = None,
    ) -> None:
        self.chunks = chunks
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def astream(
        self,
        state: dict[str, Any],
        *,
        stream_mode: str = "updates",
    ):
        self.calls.append(state)
        for chunk in self.chunks:
            yield chunk
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


def _success_chunks(filename: str = "report-123456789abc.docx") -> list[dict[str, Any]]:
    final_text = f"Document generated successfully: {filename}"
    return [
        {"prepare_request": {"status": "understood"}},
        {"plan_document": {"status": "planned"}},
        {"select_tools": {"status": "tool_selection"}},
        {"tools": {"messages": [SENSITIVE_SENTINEL]}},
        {"write_document": {"status": "writing", "content": SENSITIVE_SENTINEL}},
        {
            "validate_document": {
                "status": "validating",
                "validation_report": SENSITIVE_SENTINEL,
            }
        },
        {
            "generate_docx": {
                "status": "docx_generation",
                "output_filename": filename,
            }
        },
        {"finalize": {"status": "completed", "terminal_reason": "completed"}},
        {"final_response": {"final_response": final_text}},
    ]


async def test_successful_sse_stream_has_progress_token_and_complete() -> None:
    app = create_app(
        _settings(),
        graph_factory=lambda: FakeGraph(_success_chunks()),
    )

    async with _client(app) as client:
        response = await client.post("/agent", json={"request": "Create a report"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"

    events = _parse_sse(response.text)
    event_names = [event for event, _ in events]
    assert event_names[-2:] == ["token", "complete"]
    assert "error" not in event_names

    progress_messages = [
        data["message"] for event, data in events if event == "progress"
    ]
    assert "Understanding request..." in progress_messages
    assert "Planning document..." in progress_messages
    assert "Processing tool results..." in progress_messages
    assert "Writing document..." in progress_messages
    assert "Validating document..." in progress_messages
    assert "Generating DOCX document..." in progress_messages
    assert all(message in PROGRESS_MESSAGES.values() for message in progress_messages)

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
    chunks = [
        {"prepare_request": {"status": "rejected"}},
        {"reject": {"status": "rejected", "terminal_reason": "rejected"}},
        {"final_response": {"final_response": REJECTION_RESPONSE}},
    ]
    app = create_app(
        _settings(),
        graph_factory=lambda: FakeGraph(chunks),
    )

    async with _client(app) as client:
        response = await client.post("/agent", json={"request": "   "})

    events = _parse_sse(response.text)
    event_names = [event for event, _ in events]
    assert "token" not in event_names
    assert event_names[-1] == "error"
    assert events[-1][1]["message"] == REJECTION_RESPONSE
    assert SENSITIVE_SENTINEL not in response.text


@pytest.mark.parametrize(
    ("node", "reason", "message"),
    [
        ("planning_failed", "planning_failed", PLANNING_FAILURE_RESPONSE),
        ("tool_failed", "tool_failed", TOOL_FAILURE_RESPONSE),
        (
            "tool_limit_reached",
            "tool_limit_reached",
            TOOL_LIMIT_RESPONSE,
        ),
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
    chunks = [
        {"prepare_request": {"status": "understood"}},
        {node: {"status": "failed", "terminal_reason": reason}},
        {"final_response": {"final_response": message}},
    ]
    app = create_app(
        _settings(),
        graph_factory=lambda: FakeGraph(chunks),
    )

    async with _client(app) as client:
        response = await client.post("/agent", json={"request": "Create a report"})

    events = _parse_sse(response.text)
    assert [event for event, _ in events][-1] == "error"
    assert events[-1][1] == {"message": message}
    assert SENSITIVE_SENTINEL not in response.text


async def test_unexpected_graph_exception_is_sanitized() -> None:
    def factory() -> FakeGraph:
        return FakeGraph(
            [{"prepare_request": {"status": "understood"}}],
            error=RuntimeError(f"provider failure {SENSITIVE_SENTINEL}"),
        )

    app = create_app(_settings(), graph_factory=factory)

    async with _client(app) as client:
        response = await client.post("/agent", json={"request": "Create a report"})

    events = _parse_sse(response.text)
    assert events[-1][0] == "error"
    assert SENSITIVE_SENTINEL not in response.text
    assert "provider failure" not in response.text
    assert events[-1][1]["message"] == (
        "The request could not be completed. Please try again."
    )


async def test_graph_factory_exception_is_sanitized() -> None:
    def factory() -> FakeGraph:
        raise RuntimeError(f"factory failure {SENSITIVE_SENTINEL}")

    app = create_app(_settings(), graph_factory=factory)

    async with _client(app) as client:
        response = await client.post("/agent", json={"request": "Create a report"})

    events = _parse_sse(response.text)
    assert events[-1][0] == "error"
    assert SENSITIVE_SENTINEL not in response.text
    assert "factory failure" not in response.text


async def test_concurrent_requests_are_isolated() -> None:
    class Factory:
        def __init__(self) -> None:
            self.count = 0

        def __call__(self) -> FakeGraph:
            self.count += 1
            filename = f"report-{self.count:012x}.docx"
            return FakeGraph(_success_chunks(filename))

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
        async def astream(self, state: dict[str, Any], *, stream_mode: str = "updates"):
            yield {"prepare_request": {"status": "understood"}}
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


async def test_real_graph_streams_through_sse(
    fake_writer: Any,
    fake_docx_generator: Any,
    fake_validator: Any,
) -> None:
    class StaticPlanner:
        async def create_plan(self, request: str) -> DocumentPlan:
            return DocumentPlan(
                document_type=DocumentType.BUSINESS_REPORT,
                title="Status Report",
                objective="Report progress.",
                sections=[PlanSection(heading="Overview", purpose="Overview.")],
            )

    class NoToolSelector:
        async def select(self, messages: Any) -> AIMessage:
            return AIMessage(content="No tool needed.")

    def factory() -> Any:
        return compile_agent_graph(
            planner=StaticPlanner(),
            tool_selector=NoToolSelector(),
            writer=fake_writer,
            docx_generator=fake_docx_generator,
            validator=fake_validator,
        )

    app = create_app(_settings(), graph_factory=factory)

    async with _client(app) as client:
        response = await client.post("/agent", json={"request": "Create a report"})

    events = _parse_sse(response.text)
    event_names = [event for event, _ in events]
    assert "complete" in event_names
    assert event_names[-2:] == ["token", "complete"]
    complete = next(data for event, data in events if event == "complete")
    assert complete["filename"] == "test-document-000000000000.docx"
    token = next(data for event, data in events if event == "token")
    assert token["text"] == (
        "Document generated successfully: test-document-000000000000.docx"
    )


async def test_health_endpoint_contract_is_preserved() -> None:
    app = create_app(_settings(), graph_factory=lambda: FakeGraph([]))

    async with _client(app) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_sse_event_formatting() -> None:
    event = _sse_event("progress", {"message": "Working"})

    assert event == b'event: progress\ndata: {"message": "Working"}\n\n'
