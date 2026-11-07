"""Offline tests for document-plan models and the planning service."""

from typing import Any

import pytest
from pydantic import ValidationError

from backend.app.agent import (
    DocumentPlan,
    DocumentPlanner,
    DocumentType,
    ExecutionStage,
    ExecutionTask,
    PlanningError,
    PlanPlaceholder,
    PlanSection,
    RequirementConflict,
)
from backend.app.agent.planning import (
    DOCUMENT_STRUCTURE_GUIDANCE,
    DOCUMENT_TYPE_GUIDANCE,
    PLANNING_SYSTEM_PROMPT,
)
from backend.tests.plan_factory import make_execution_tasks


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
        "execution_tasks": make_execution_tasks(),
        **overrides,
    }
    return DocumentPlan(**values)


def test_planning_prompt_lists_all_supported_document_types() -> None:
    for document_type in DocumentType:
        assert document_type.value in PLANNING_SYSTEM_PROMPT
    assert "JSON" in PLANNING_SYSTEM_PROMPT


def test_planning_prompt_includes_explicit_classification_guidance() -> None:
    for document_type, guidance in DOCUMENT_TYPE_GUIDANCE.items():
        assert document_type.value in PLANNING_SYSTEM_PROMPT
        assert guidance in PLANNING_SYSTEM_PROMPT

    assert "primary purpose" in PLANNING_SYSTEM_PROMPT
    assert "intended audience" in PLANNING_SYSTEM_PROMPT
    assert "desired outcome" in PLANNING_SYSTEM_PROMPT
    assert "Do not classify using isolated keywords alone" in PLANNING_SYSTEM_PROMPT
    assert "Use other when no existing category is suitable" in PLANNING_SYSTEM_PROMPT


def test_planning_prompt_includes_type_specific_section_guidance() -> None:
    assert set(DOCUMENT_STRUCTURE_GUIDANCE) == set(DocumentType)
    for sections in DOCUMENT_STRUCTURE_GUIDANCE.values():
        for heading in sections:
            assert heading in PLANNING_SYSTEM_PROMPT

    assert DOCUMENT_STRUCTURE_GUIDANCE[DocumentType.MEETING_MINUTES] == (
        "Meeting Details",
        "Agenda",
        "Discussion Points",
        "Decisions",
        "Action Items",
    )
    assert (
        "Acceptance Criteria"
        in DOCUMENT_STRUCTURE_GUIDANCE[DocumentType.PRODUCT_SPECIFICATION]
    )
    assert "Procedure Steps" in DOCUMENT_STRUCTURE_GUIDANCE[DocumentType.SOP]


def test_planning_prompt_includes_execution_conflict_and_placeholder_guidance() -> None:
    assert "request-specific list of meaningful work tasks" in PLANNING_SYSTEM_PROMPT
    assert "Multiple tasks may share the same stage" in PLANNING_SYSTEM_PROMPT
    assert "document_writing, validation and" in PLANNING_SYSTEM_PROMPT
    assert "docx_generation" in PLANNING_SYSTEM_PROMPT
    assert "document_date" in PLANNING_SYSTEM_PROMPT
    assert "document_version" in PLANNING_SYSTEM_PROMPT
    assert "Do not use meeting dates" in PLANNING_SYSTEM_PROMPT
    assert "For each genuine conflict" in PLANNING_SYSTEM_PROMPT
    assert "reason explaining why the value is" in PLANNING_SYSTEM_PROMPT


def test_execution_plan_requires_core_task_stages() -> None:
    incomplete = [
        task
        for task in make_execution_tasks()
        if task.stage is not ExecutionStage.DOCUMENT_WRITING
    ]

    with pytest.raises(ValidationError, match="missing required stages"):
        _plan(execution_tasks=incomplete)


def test_execution_tasks_allow_multiple_tasks_per_stage_without_finalization() -> None:
    tasks = [
        ExecutionTask(
            id="draft_summary",
            title="Draft summary",
            description="Draft the executive summary.",
            stage=ExecutionStage.DOCUMENT_WRITING,
        ),
        ExecutionTask(
            id="draft_findings",
            title="Draft findings",
            description="Draft the findings.",
            stage=ExecutionStage.DOCUMENT_WRITING,
        ),
        ExecutionTask(
            id="validate",
            title="Validate",
            description="Validate the document.",
            stage=ExecutionStage.VALIDATION,
        ),
        ExecutionTask(
            id="render",
            title="Render",
            description="Render the DOCX.",
            stage=ExecutionStage.DOCX_GENERATION,
        ),
    ]

    plan = _plan(execution_tasks=tasks)

    assert len(plan.execution_tasks) == 4
    assert plan.execution_tasks[0].stage is ExecutionStage.DOCUMENT_WRITING
    assert plan.execution_tasks[1].stage is ExecutionStage.DOCUMENT_WRITING


def test_duplicate_execution_task_ids_are_rejected() -> None:
    tasks = make_execution_tasks()
    tasks[1] = tasks[1].model_copy(update={"id": tasks[0].id})

    with pytest.raises(ValidationError, match="IDs must be unique"):
        _plan(execution_tasks=tasks)


def test_document_date_and_version_are_preserved() -> None:
    plan = _plan(document_date="2026-11-07", document_version="1.2")

    assert plan.document_date == "2026-11-07"
    assert plan.document_version == "1.2"


def test_conflict_requires_two_requirements_and_an_impact() -> None:
    conflict = RequirementConflict(
        requirements=["Four-week rollout", "Three weeks of cleanup"],
        impact="The target cannot be met.",
        resolution="Use a phased pilot.",
    )
    plan = _plan(conflicts=[conflict])
    assert plan.conflicts[0].resolution == "Use a phased pilot."

    with pytest.raises(ValidationError):
        RequirementConflict(
            requirements=["Only one requirement"],
            impact="Invalid.",
        )


def test_placeholder_requires_brackets_and_a_reason() -> None:
    with pytest.raises(ValidationError):
        PlanPlaceholder(text="Budget", reason="Missing budget.")
    with pytest.raises(ValidationError):
        PlanPlaceholder(text="[Budget]", reason="")


async def test_planner_preserves_tasks_conflicts_and_document_metadata() -> None:
    plan = _plan(
        document_date="2026-11-07",
        document_version="1.2",
        conflicts=[
            RequirementConflict(
                requirements=["Four-week rollout", "Three-week cleanup"],
                impact="The target is not achievable as requested.",
                resolution="Recommend a phased rollout.",
            )
        ],
    )
    planner = DocumentPlanner(FakeStructuredModel(plan))

    result = await planner.create_plan("Create the plan")

    assert result.document_date == "2026-11-07"
    assert result.document_version == "1.2"
    assert result.conflicts[0].impact.startswith("The target")
    assert result.execution_tasks == plan.execution_tasks


def test_document_plan_requires_at_least_one_section() -> None:
    with pytest.raises(ValidationError):
        _plan(sections=[])


def test_document_plan_rejects_unknown_document_type() -> None:
    with pytest.raises(ValidationError):
        _plan(document_type="unrecognized_type")


def test_document_plan_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        _plan(unexpected_field="not allowed")


def test_document_plan_rejects_duplicate_section_headings() -> None:
    duplicate_sections = [
        PlanSection(heading="Overview", purpose="First overview."),
        PlanSection(heading="Overview", purpose="Second overview."),
    ]

    with pytest.raises(ValidationError, match="section headings must be unique"):
        _plan(sections=duplicate_sections)


def test_document_plan_rejects_headings_differing_only_by_case_or_whitespace() -> None:
    duplicate_sections = [
        PlanSection(heading="Financial   Review", purpose="First review."),
        PlanSection(heading="  financial review  ", purpose="Second review."),
    ]

    with pytest.raises(ValidationError, match="section headings must be unique"):
        _plan(sections=duplicate_sections)


def test_plan_section_rejects_duplicate_placeholder_text() -> None:
    placeholder = PlanPlaceholder(
        text="[Budget]",
        reason="Budget was not supplied.",
    )

    with pytest.raises(ValidationError, match="placeholder text must be unique"):
        PlanSection(
            heading="Budget",
            purpose="Describe the budget.",
            placeholders=[placeholder, placeholder],
        )


def test_same_placeholder_in_different_sections_remains_valid() -> None:
    sections = [
        PlanSection(
            heading="Alpha",
            purpose="First section.",
            placeholders=[
                PlanPlaceholder(text="[Budget]", reason="Alpha budget reason.")
            ],
        ),
        PlanSection(
            heading="Beta",
            purpose="Second section.",
            placeholders=[
                PlanPlaceholder(text="[Budget]", reason="Beta budget reason.")
            ],
        ),
    ]

    plan = _plan(sections=sections)

    assert plan.sections[0].placeholders[0].reason == "Alpha budget reason."
    assert plan.sections[1].placeholders[0].reason == "Beta budget reason."


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
                placeholders=[
                    PlanPlaceholder(
                        text="[Project deadline]",
                        reason="The project deadline was not provided.",
                    )
                ],
            )
        ],
    )
    planner = DocumentPlanner(FakeStructuredModel(plan))

    result = await planner.create_plan("Write a project plan for Atlas")

    assert result.provided_facts == ["The project is called Atlas."]
    assert result.assumptions == ["A one-page summary is appropriate."]
    assert result.missing_information == ["Project deadline", "Budget"]
    assert result.clarification_questions == ["What is the project deadline?"]
    assert result.sections[0].placeholders == [
        PlanPlaceholder(
            text="[Project deadline]",
            reason="The project deadline was not provided.",
        )
    ]


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

    with pytest.raises(PlanningError, match="valid document plan"):
        await planner.create_plan("Create a report")


async def test_planner_retries_once_after_invalid_output() -> None:
    plan = _plan()

    class RetryModel:
        def __init__(self) -> None:
            self.calls = 0

        def with_structured_output(self, *args: Any, **kwargs: Any) -> FakeRunnable:
            async def ainvoke(messages: Any) -> Any:
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("invalid response")
                return plan

            return FakeRunnable(ainvoke)

    model = RetryModel()
    result = await DocumentPlanner(model).create_plan("Create the report")

    assert result == plan
    assert model.calls == 2


async def test_planner_rejects_empty_request_without_calling_model() -> None:
    model = FakeStructuredModel(_plan())
    planner = DocumentPlanner(model)

    with pytest.raises(PlanningError, match="empty request"):
        await planner.create_plan("   ")

    assert model.calls == []
