"""Offline tests for document-plan models and the planning service."""

from typing import Any

import pytest
from pydantic import ValidationError

from backend.app.agent import (
    DocumentPlan,
    DocumentPlanner,
    DocumentType,
    PlanningError,
    PlanSection,
)
from backend.app.agent.planning import PLANNING_SYSTEM_PROMPT


class FakeRunnable:
    def __init__(self, handler: Any) -> None:
        self._handler = handler

    async def ainvoke(self, messages: Any) -> Any:
        return await self._handler(messages)


class FakeStructuredModel:
    """Minimal model double implementing ``with_structured_output``."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.schema: Any = None
        self.method: str | None = None
        self.include_raw: bool | None = None
        self.calls: list[Any] = []

    def with_structured_output(
        self,
        schema: Any,
        *,
        method: str = "function_calling",
        include_raw: bool = False,
    ) -> FakeRunnable:
        self.schema = schema
        self.method = method
        self.include_raw = include_raw

        async def ainvoke(messages: Any) -> Any:
            self.calls.append(messages)
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

        return FakeRunnable(ainvoke)


def _plan(**overrides: Any) -> DocumentPlan:
    values: dict[str, Any] = {
        "document_type": DocumentType.BUSINESS_REPORT,
        "title": "Quarterly Status Report",
        "objective": "Report progress to stakeholders.",
        "audience": None,
        "sections": [
            PlanSection(
                heading="Executive Summary",
                purpose="Summarize the report.",
                key_points=["Highlights"],
            )
        ],
        "provided_facts": [],
        "assumptions": [],
        "missing_information": [],
        "clarification_questions": [],
        **overrides,
    }
    return DocumentPlan(**values)


def test_planning_prompt_lists_all_supported_document_types() -> None:
    for document_type in DocumentType:
        assert document_type.value in PLANNING_SYSTEM_PROMPT
    assert "JSON" in PLANNING_SYSTEM_PROMPT


def test_document_plan_requires_at_least_one_section() -> None:
    with pytest.raises(ValidationError):
        _plan(sections=[])


def test_document_plan_rejects_unknown_document_type() -> None:
    with pytest.raises(ValidationError):
        _plan(document_type="unrecognized_type")


def test_document_plan_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        _plan(unexpected_field="not allowed")


def test_document_plan_accepts_all_supported_types() -> None:
    for document_type in DocumentType:
        assert _plan(document_type=document_type).document_type is document_type


async def test_planner_uses_json_mode_and_returns_structured_plan() -> None:
    plan = _plan()
    model = FakeStructuredModel(plan)
    planner = DocumentPlanner(model)

    result = await planner.create_plan("Create a quarterly status report")

    assert result == plan
    assert model.schema is DocumentPlan
    assert model.method == "json_mode"
    assert model.include_raw is False
    assert len(model.calls) == 1
    assert [message.type for message in model.calls[0]] == ["system", "human"]


@pytest.mark.parametrize("document_type", list(DocumentType))
async def test_planner_preserves_each_document_type(
    document_type: DocumentType,
) -> None:
    plan = _plan(document_type=document_type)
    planner = DocumentPlanner(FakeStructuredModel(plan))

    result = await planner.create_plan("Plan the requested document")

    assert result.document_type is document_type


async def test_planner_preserves_facts_assumptions_and_questions() -> None:
    plan = _plan(
        provided_facts=["The project is called Atlas."],
        assumptions=["A one-page summary is appropriate."],
        missing_information=["Project deadline", "Budget"],
        clarification_questions=["What is the project deadline?"],
        sections=[
            PlanSection(
                heading="Overview",
                purpose="Introduce Atlas.",
                key_points=["Known project name"],
                placeholders=["[Project deadline]"],
            )
        ],
    )
    planner = DocumentPlanner(FakeStructuredModel(plan))

    result = await planner.create_plan("Write a project plan for Atlas")

    assert result.provided_facts == ["The project is called Atlas."]
    assert result.assumptions == ["A one-page summary is appropriate."]
    assert result.missing_information == ["Project deadline", "Budget"]
    assert result.clarification_questions == ["What is the project deadline?"]
    assert result.sections[0].placeholders == ["[Project deadline]"]


async def test_planner_wraps_model_failure() -> None:
    cause = RuntimeError("provider unavailable")
    planner = DocumentPlanner(FakeStructuredModel(cause))

    with pytest.raises(PlanningError, match="valid document plan") as exc_info:
        await planner.create_plan("Create a report")

    assert exc_info.value.__cause__ is cause


async def test_planner_rejects_invalid_structured_output() -> None:
    invalid_plan = {
        "document_type": "business_report",
        "title": "Missing required fields",
    }
    planner = DocumentPlanner(FakeStructuredModel(invalid_plan))

    with pytest.raises(PlanningError, match="invalid document plan"):
        await planner.create_plan("Create a report")


async def test_planner_rejects_empty_request_without_calling_model() -> None:
    model = FakeStructuredModel(_plan())
    planner = DocumentPlanner(model)

    with pytest.raises(PlanningError, match="empty request"):
        await planner.create_plan("   ")

    assert model.calls == []
