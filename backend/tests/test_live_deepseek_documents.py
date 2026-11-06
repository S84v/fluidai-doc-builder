"""Opt-in live DeepSeek test for JSON-mode planning and writing.

Run only with an explicit flag and configured credentials:

    RUN_LIVE_DEEPSEEK_TESTS=1 uv run pytest \
        backend/tests/test_live_deepseek_documents.py -v

The test keeps the request small and writes only to pytest's temporary
directory. It never prints credentials or reasoning content.
"""

import os
from pathlib import Path

import pytest
from docx import Document

from backend.app.agent import (
    DocumentContent,
    DocumentPlan,
    DocumentPlanner,
    DocumentWriter,
    DocxGenerator,
)
from backend.app.config import Settings
from backend.app.services.llm import build_chat_model

RUN_LIVE = os.getenv("RUN_LIVE_DEEPSEEK_TESTS") == "1"

pytestmark = [
    pytest.mark.skipif(
        not RUN_LIVE,
        reason="Live DeepSeek test disabled; set RUN_LIVE_DEEPSEEK_TESTS=1",
    ),
    pytest.mark.asyncio,
]


async def test_live_json_mode_planning_writing_and_docx(tmp_path: Path) -> None:
    settings = Settings()
    if not settings.deepseek_api_key:
        pytest.skip("Inconclusive: DEEPSEEK_API_KEY is not configured")

    request = (
        "Create short meeting minutes for a project kickoff. "
        "Project: Atlas. Date: 2026-11-06. Attendees: Anna and Ben. "
        "Decision: proceed with phase one."
    )
    model = build_chat_model(settings)
    planner = DocumentPlanner(model)
    writer = DocumentWriter(model)

    try:
        plan = await planner.create_plan(request)
    except Exception as exc:
        pytest.fail(f"Live planning failed with {type(exc).__name__}")

    assert isinstance(plan, DocumentPlan)
    assert plan.sections

    try:
        content = await writer.generate(request, plan, tool_results=[])
    except Exception as exc:
        pytest.fail(f"Live writing failed with {type(exc).__name__}")

    assert isinstance(content, DocumentContent)
    assert [section.heading for section in content.sections] == [
        section.heading for section in plan.sections
    ]

    filename = DocxGenerator(tmp_path).generate(content, request)
    destination = tmp_path / filename
    assert destination.exists()

    reopened = Document(destination)
    assert reopened.paragraphs[0].text == content.title
    headings = [
        paragraph.text
        for paragraph in reopened.paragraphs
        if paragraph.style.name == "Heading 1"
    ]
    assert headings == [section.heading for section in content.sections]
