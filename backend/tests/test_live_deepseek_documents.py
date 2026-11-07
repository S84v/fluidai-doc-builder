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
    DocumentType,
    DocumentWriter,
    DocxGenerator,
)
from backend.app.agent.docx_templates import (
    TEMPLATE_RENDERERS,
    render_business_proposal,
    render_meeting_minutes,
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


@pytest.mark.parametrize(
    ("user_request", "expected_type", "expected_renderer"),
    [
        (
            "Create a business proposal for a document automation platform "
            "for Atlas leadership. Include an executive summary, proposed "
            "solution, scope, implementation approach, timeline, benefits and "
            "next steps. Project: Atlas. Target launch: Q1 2027. Budget is not "
            "yet approved.",
            DocumentType.BUSINESS_PROPOSAL,
            render_business_proposal,
        ),
        (
            "Create short meeting minutes for a project kickoff. "
            "Project: Atlas. Date: 2026-11-06. Attendees: Anna and Ben. "
            "Decision: proceed with phase one.",
            DocumentType.MEETING_MINUTES,
            render_meeting_minutes,
        ),
    ],
)
async def test_live_json_mode_planning_writing_and_docx(
    tmp_path: Path,
    user_request: str,
    expected_type: DocumentType,
    expected_renderer: object,
) -> None:
    settings = Settings()
    if not settings.deepseek_api_key:
        pytest.skip("Inconclusive: DEEPSEEK_API_KEY is not configured")

    model = build_chat_model(settings)
    planner = DocumentPlanner(model)
    writer = DocumentWriter(model)

    try:
        plan = await planner.create_plan(user_request)
    except Exception as exc:
        pytest.fail(f"Live planning failed with {type(exc).__name__}")

    assert isinstance(plan, DocumentPlan)
    assert plan.document_type is expected_type
    assert plan.sections

    plan_headings = " ".join(section.heading.casefold() for section in plan.sections)
    if expected_type is DocumentType.BUSINESS_PROPOSAL:
        assert any(
            keyword in plan_headings
            for keyword in ("summary", "solution", "approach", "scope")
        )
    else:
        assert any(
            keyword in plan_headings
            for keyword in ("agenda", "discussion", "decision", "action")
        )

    try:
        content = await writer.generate(user_request, plan, tool_results=[])
    except Exception as exc:
        pytest.fail(f"Live writing failed with {type(exc).__name__}")

    assert isinstance(content, DocumentContent)
    assert content.document_type is expected_type
    assert [section.heading for section in content.sections] == [
        section.heading for section in plan.sections
    ]
    assert TEMPLATE_RENDERERS[content.document_type] is expected_renderer

    generated = DocxGenerator(tmp_path).generate(content, user_request, plan)
    filename = generated.filename
    destination = tmp_path / filename
    assert destination.exists()

    reopened = Document(destination)
    assert reopened.paragraphs[0].text == content.title
    headings = [
        paragraph.text
        for paragraph in reopened.paragraphs
        if paragraph.style.name == "Heading 1"
    ]
    assert all(section.heading in headings for section in content.sections)
    assert reopened.tables
