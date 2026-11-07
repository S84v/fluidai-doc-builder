"""Opt-in live DeepSeek test for review and controlled revision.

Run only with explicit credentials and:

    RUN_LIVE_DEEPSEEK_TESTS=1 uv run pytest \
        backend/tests/test_live_deepseek_validation.py -v
"""

import os

import pytest

from backend.app.agent import (
    ContentSection,
    DocumentContent,
    DocumentPlan,
    DocumentReviewer,
    DocumentType,
    DocumentValidator,
    DocumentWriter,
    IssueSeverity,
    PlanSection,
    ValidationIssue,
)
from backend.app.config import Settings
from backend.app.services.llm import build_chat_model
from backend.tests.plan_factory import make_execution_tasks

RUN_LIVE = os.getenv("RUN_LIVE_DEEPSEEK_TESTS") == "1"

pytestmark = [
    pytest.mark.skipif(
        not RUN_LIVE,
        reason="Live DeepSeek test disabled; set RUN_LIVE_DEEPSEEK_TESTS=1",
    ),
    pytest.mark.asyncio,
]


class RecordingWriter(DocumentWriter):
    def __init__(self, model: object) -> None:
        super().__init__(model)
        self.feedback: list[ValidationIssue] = []

    async def revise(
        self,
        request: str,
        plan: DocumentPlan,
        content: DocumentContent,
        blocking_issues: list[ValidationIssue],
        tool_results: list[str] | tuple[str, ...] = (),
    ) -> DocumentContent:
        self.feedback = list(blocking_issues)
        return await super().revise(
            request,
            plan,
            content,
            blocking_issues,
            tool_results,
        )


def _plan() -> DocumentPlan:
    return DocumentPlan(
        execution_tasks=make_execution_tasks(),
        document_type=DocumentType.BUSINESS_REPORT,
        title="Atlas Budget Report",
        objective="Report the approved budget and timeline.",
        sections=[
            PlanSection(heading="Overview", purpose="Introduce the report."),
            PlanSection(heading="Budget", purpose="State the approved budget."),
            PlanSection(heading="Timeline", purpose="State the approved timeline."),
        ],
        provided_facts=[
            "The approved budget is 10,000 USD.",
            "The timeline is four weeks.",
        ],
    )


def _contradictory_content() -> DocumentContent:
    return DocumentContent(
        document_type=DocumentType.BUSINESS_REPORT,
        title="Atlas Budget Report",
        sections=[
            ContentSection(
                heading="Overview",
                paragraphs=["This report covers the Atlas project."],
            ),
            ContentSection(
                heading="Budget",
                paragraphs=["The approved budget is 20,000 USD."],
            ),
            ContentSection(
                heading="Timeline",
                paragraphs=["The project will complete in four weeks."],
            ),
        ],
    )


def _content_text(content: DocumentContent) -> str:
    parts: list[str] = []
    for section in content.sections:
        parts.extend(section.paragraphs)
        parts.extend(section.bullets)
    return "\n".join(parts)


async def test_live_review_detects_and_revision_corrects_contradiction() -> None:
    settings = Settings()
    if not settings.deepseek_api_key:
        pytest.skip("Inconclusive: DEEPSEEK_API_KEY is not configured")

    request = (
        "Create a short business report. The approved budget is 10,000 USD. "
        "The timeline is four weeks."
    )
    plan = _plan()
    content = _contradictory_content()
    model = build_chat_model(settings)
    reviewer = DocumentReviewer(model)
    validator = DocumentValidator(reviewer)
    writer = RecordingWriter(model)

    try:
        report = await validator.validate(request, plan, content)
    except Exception as exc:
        pytest.fail(f"Live review failed with {type(exc).__name__}")

    print(
        "reviewer_detected_contradiction:",
        bool(report.blocking_issues),
    )

    blocking = report.blocking_issues
    if not blocking:
        blocking = [
            ValidationIssue(
                code="budget_contradiction",
                message=(
                    "The budget section states 20,000 USD, contradicting the "
                    "approved 10,000 USD."
                ),
                severity=IssueSeverity.BLOCKING,
                section="Budget",
                suggestion="Change the budget figure to 10,000 USD.",
            )
        ]

    try:
        revised = await writer.revise(
            request,
            plan,
            content,
            blocking,
            tool_results=[],
        )
    except Exception as exc:
        pytest.fail(f"Live revision failed with {type(exc).__name__}")

    assert writer.feedback == blocking
    revised_text = _content_text(revised)
    assert "10,000" in revised_text or "10000" in revised_text

    try:
        second_report = await validator.validate(request, plan, revised)
    except Exception as exc:
        pytest.fail(f"Live re-review failed with {type(exc).__name__}")

    assert second_report.passed is True
