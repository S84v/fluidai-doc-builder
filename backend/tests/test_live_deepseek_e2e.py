"""Opt-in live DeepSeek end-to-end tests through the FastAPI application.

Run only with credentials and:

    RUN_LIVE_DEEPSEEK_TESTS=1 uv run pytest \
        backend/tests/test_live_deepseek_e2e.py -s -v
"""

import io
import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
from docx import Document

from backend.app.config import Settings
from backend.app.main import create_app

RUN_LIVE = os.getenv("RUN_LIVE_DEEPSEEK_TESTS") == "1"

pytestmark = [
    pytest.mark.skipif(
        not RUN_LIVE,
        reason="Live DeepSeek test disabled; set RUN_LIVE_DEEPSEEK_TESTS=1",
    ),
    pytest.mark.asyncio,
]


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


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        timeout=180,
    )


def _table_headers(document: Document) -> set[str]:
    values: set[str] = set()
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                values.add(cell.text)
    return values


@pytest.mark.parametrize(
    ("case_name", "request_text", "expects_tool"),
    [
        (
            "business_proposal",
            "Create a business proposal for an internal document automation "
            "platform for Atlas leadership. Include an executive summary, "
            "problem or opportunity, proposed solution, scope and deliverables, "
            "implementation approach, timeline, benefits and risks, and next "
            "steps. Project: Atlas. Target launch: Q1 2027. "
            "Use the calculator tool to compute 17 * 23 and include the result.",
            True,
        ),
        (
            "meeting_minutes",
            "Create short meeting minutes for the Atlas project kickoff. "
            "Date: 2026-11-06. Time: 10:00. Location: Room A. "
            "Attendees: Anna, Ben. Facilitator: Anna. "
            "Agenda: project scope and timeline. Discussion: reviewed phase one. "
            "Decision: proceed with phase one. Action items: Anna to send notes "
            "by 2026-11-08; Ben to confirm budget by 2026-11-10.",
            False,
        ),
    ],
)
async def test_live_e2e_agent_request_and_download(
    tmp_path: Path,
    case_name: str,
    request_text: str,
    expects_tool: bool,
) -> None:
    settings = Settings()
    if not settings.deepseek_api_key:
        pytest.skip("Inconclusive: DEEPSEEK_API_KEY is not configured")

    app = create_app(settings, output_dir=tmp_path)

    async with _client(app) as client:
        response = await client.post("/agent", json={"request": request_text})
        events = _parse_sse(response.text)

        event_names = [event for event, _ in events]
        assert "error" not in event_names, events[-1][1]
        assert event_names[0] == "plan"
        assert event_names[-2:] == ["token", "complete"]

        plan_event = events[0][1]
        task_stages = {task["stage"] for task in plan_event["tasks"]}
        assert {"document_writing", "validation", "docx_generation"} <= task_stages

        progress_stages = {
            data["stage"] for event, data in events if event == "progress"
        }
        assert "document_writing" in progress_stages
        assert "validation" in progress_stages
        assert "docx_generation" in progress_stages
        if expects_tool:
            assert "tool_execution" in progress_stages

        token = next(data for event, data in events if event == "token")
        complete = next(data for event, data in events if event == "complete")
        assert token["text"].startswith("Document generated successfully:")
        assert complete["filename"].endswith(".docx")

        download = await client.get(complete["url"])

    assert download.status_code == 200
    assert "reasoning_content" not in response.text
    document = Document(io.BytesIO(download.content))
    assert document.paragraphs[0].text

    headings = {
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.style.name == "Heading 1"
    }
    assert headings
    table_values = _table_headers(document)

    assert "Document Date" in table_values
    assert "Document Version" in table_values

    if case_name == "business_proposal":
        assert "Prepared For" in table_values
        assert any(
            'w:type="page"' in paragraph._p.xml for paragraph in document.paragraphs
        )
    else:
        assert "Meeting Date" in table_values
        assert "Action" in table_values
