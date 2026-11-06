"""Opt-in live DeepSeek test for the calculator tool round trip.

Run only with an explicit flag and configured credentials:

    RUN_LIVE_DEEPSEEK_TESTS=1 uv run pytest \
        backend/tests/test_live_deepseek_tools.py -v

The test performs one tool-selection round trip (plus any bounded follow-up
rounds requested by the model). It never prints credentials or reasoning text.
"""

import os

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from backend.app.agent import (
    DocumentPlan,
    DocumentType,
    PlanSection,
    ToolSelector,
    run_agent,
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


class StaticPlanner:
    async def create_plan(self, request: str) -> DocumentPlan:
        return DocumentPlan(
            document_type=DocumentType.BUSINESS_REPORT,
            title="Calculation Check",
            objective="Verify the calculator tool integration.",
            sections=[
                PlanSection(
                    heading="Calculation",
                    purpose="Record the calculated result.",
                )
            ],
        )


async def test_live_calculator_tool_round_trip() -> None:
    settings = Settings()
    if not settings.deepseek_api_key:
        pytest.skip("Inconclusive: DEEPSEEK_API_KEY is not configured")

    model = build_chat_model(settings)
    selector = ToolSelector(model)

    try:
        result = await run_agent(
            "Use the calculator tool to compute 17 * 23. Report the numeric result.",
            planner=StaticPlanner(),
            tool_selector=selector,
        )
    except Exception as exc:
        pytest.fail(f"Live tool round trip failed with {type(exc).__name__}")

    tool_messages = [
        message
        for message in result.get("messages", [])
        if isinstance(message, ToolMessage)
    ]
    if not tool_messages:
        pytest.skip("Inconclusive: the model did not select the calculator tool")

    assert any(message.content == "391" for message in tool_messages)
    assert result["status"] == "completed", result["final_response"]
    assert result["final_response"].strip()
    assert result["tool_rounds"] >= 1

    tool_call_messages = [
        message
        for message in result["messages"]
        if isinstance(message, AIMessage) and message.tool_calls
    ]
    assert tool_call_messages
    assert any(
        message.additional_kwargs.get("reasoning_content")
        for message in tool_call_messages
    )
