"""Tests for mapping graph events to request-specific execution tasks."""

from backend.app.agent import (
    DocumentPlan,
    DocumentType,
    ExecutionStage,
    ExecutionTask,
    PlanPlaceholder,
    PlanSection,
)
from backend.app.api.progress import (
    NODE_STAGE_MAP,
    TERMINAL_FAILURE_STAGES,
    PlanProgressTracker,
)


def _plan() -> DocumentPlan:
    return DocumentPlan(
        document_type=DocumentType.PROJECT_PLAN,
        title="Rollout Plan",
        objective="Plan the rollout.",
        document_date="2026-11-07",
        document_version="1.2",
        sections=[
            PlanSection(
                heading="Overview",
                purpose="Introduce the plan.",
                placeholders=[
                    PlanPlaceholder(
                        text="[Budget]",
                        reason="No budget was supplied.",
                    )
                ],
            )
        ],
        missing_information=["Budget"],
        assumptions=["A phased approach is acceptable."],
        conflicts=[],
        execution_tasks=[
            ExecutionTask(
                id="understand",
                title="Understand request",
                description="Read the request.",
                stage=ExecutionStage.REQUEST_UNDERSTANDING,
            ),
            ExecutionTask(
                id="plan",
                title="Plan document",
                description="Plan the document.",
                stage=ExecutionStage.PLANNING,
            ),
            ExecutionTask(
                id="draft_summary",
                title="Draft summary",
                description="Draft the summary.",
                stage=ExecutionStage.DOCUMENT_WRITING,
            ),
            ExecutionTask(
                id="draft_milestones",
                title="Draft milestones",
                description="Draft milestones.",
                stage=ExecutionStage.DOCUMENT_WRITING,
            ),
            ExecutionTask(
                id="validate",
                title="Validate",
                description="Validate the plan.",
                stage=ExecutionStage.VALIDATION,
            ),
            ExecutionTask(
                id="render",
                title="Render DOCX",
                description="Render the DOCX.",
                stage=ExecutionStage.DOCX_GENERATION,
            ),
            ExecutionTask(
                id="use_tool",
                title="Use a tool",
                description="Run a required tool.",
                stage=ExecutionStage.TOOL_EXECUTION,
            ),
            ExecutionTask(
                id="revise",
                title="Revise",
                description="Revise if needed.",
                stage=ExecutionStage.REVISION,
            ),
        ],
    )


def test_stage_mapping_uses_finalize_and_not_final_response() -> None:
    assert NODE_STAGE_MAP["finalize"] is ExecutionStage.FINALIZATION
    assert "final_response" not in NODE_STAGE_MAP
    assert set(TERMINAL_FAILURE_STAGES.values()) <= set(ExecutionStage)


def test_plan_event_contains_tasks_and_plan_details() -> None:
    tracker = PlanProgressTracker()
    tracker.complete_stage(ExecutionStage.REQUEST_UNDERSTANDING)
    tracker.complete_stage(ExecutionStage.PLANNING)

    payload = tracker.attach_plan(_plan())
    tasks = {task["id"]: task for task in payload["tasks"]}

    assert tasks["understand"]["status"] == "completed"
    assert tasks["plan"]["status"] == "completed"
    assert tasks["draft_summary"]["status"] == "pending"
    assert tasks["use_tool"]["optional"] is True
    assert payload["document_date"] == "2026-11-07"
    assert payload["document_version"] == "1.2"
    assert payload["placeholders"] == [
        {
            "text": "[Budget]",
            "section": "Overview",
            "reason": "No budget was supplied.",
        }
    ]


def test_multiple_tasks_in_one_stage_update_together() -> None:
    tracker = PlanProgressTracker()
    tracker.attach_plan(_plan())

    active = tracker.start_stage(ExecutionStage.DOCUMENT_WRITING)
    completed = tracker.complete_stage(ExecutionStage.DOCUMENT_WRITING)

    assert active == {
        "stage": "document_writing",
        "task_ids": ["draft_summary", "draft_milestones"],
        "status": "active",
        "message": "Writing document...",
    }
    assert completed is not None
    assert completed["task_ids"] == ["draft_summary", "draft_milestones"]
    assert completed["status"] == "completed"
    assert tracker.active_task_ids == []


def test_completed_tasks_are_not_marked_failed_after_unexpected_exception() -> None:
    tracker = PlanProgressTracker()
    tracker.attach_plan(_plan())
    tracker.start_stage(ExecutionStage.DOCUMENT_WRITING)
    tracker.complete_stage(ExecutionStage.DOCUMENT_WRITING)

    assert tracker.fail_active() is None
    assert tracker.statuses["draft_summary"] == "completed"
    assert tracker.statuses["draft_milestones"] == "completed"


def test_multiple_tasks_in_stage_fail_together() -> None:
    tracker = PlanProgressTracker()
    tracker.attach_plan(_plan())
    tracker.start_stage(ExecutionStage.DOCUMENT_WRITING)

    failed = tracker.fail_active()

    assert failed is not None
    assert failed["task_ids"] == ["draft_summary", "draft_milestones"]
    assert tracker.statuses["draft_summary"] == "failed"
    assert tracker.statuses["draft_milestones"] == "failed"
    assert tracker.active_task_ids == []


def test_optional_stage_skip_updates_planned_tasks() -> None:
    tracker = PlanProgressTracker()
    tracker.attach_plan(_plan())

    skipped = tracker.skip_stage(ExecutionStage.TOOL_EXECUTION)

    assert skipped == {
        "stage": "tool_execution",
        "task_ids": ["use_tool"],
        "status": "skipped",
        "message": "No tools were required.",
    }


def test_terminal_failure_marks_matching_stage_failed() -> None:
    tracker = PlanProgressTracker()
    tracker.attach_plan(_plan())

    failed = tracker.fail_stage(TERMINAL_FAILURE_STAGES["validation_failed"])

    assert failed is not None
    assert failed["stage"] == "validation"
    assert failed["task_ids"] == ["validate"]
    assert failed["status"] == "failed"


def test_terminal_failure_marks_genuinely_active_task_failed() -> None:
    plan = _plan()
    plan = plan.model_copy(
        update={
            "execution_tasks": [
                task
                for task in plan.execution_tasks
                if task.stage is not ExecutionStage.TOOL_EXECUTION
            ]
        }
    )
    tracker = PlanProgressTracker()
    tracker.attach_plan(plan)
    tracker.start_stage(ExecutionStage.DOCUMENT_WRITING)

    failed = tracker.fail_stage(ExecutionStage.TOOL_EXECUTION)

    assert failed is not None
    assert failed["stage"] == "document_writing"
    assert failed["task_ids"] == ["draft_summary", "draft_milestones"]
    assert failed["status"] == "failed"


def test_terminal_failure_without_planned_stage_and_no_active_task_is_silent() -> None:
    plan = _plan()
    plan = plan.model_copy(
        update={
            "execution_tasks": [
                task
                for task in plan.execution_tasks
                if task.stage is not ExecutionStage.TOOL_EXECUTION
            ]
        }
    )
    tracker = PlanProgressTracker()
    tracker.attach_plan(plan)

    assert tracker.fail_stage(ExecutionStage.TOOL_EXECUTION) is None
    assert tracker.active_task_ids == []


def test_fail_active_handles_unexpected_exceptions() -> None:
    tracker = PlanProgressTracker()
    tracker.attach_plan(_plan())
    tracker.start_stage(ExecutionStage.DOCUMENT_WRITING)

    failed = tracker.fail_active()

    assert failed is not None
    assert failed["stage"] == "document_writing"
    assert failed["status"] == "failed"
