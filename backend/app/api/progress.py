"""Presentation-layer mapping from actual graph events to plan task progress."""

from typing import Any

from backend.app.agent.models import (
    OPTIONAL_EXECUTION_STAGES,
    DocumentPlan,
    ExecutionStage,
)

NODE_STAGE_MAP: dict[str, ExecutionStage] = {
    "prepare_request": ExecutionStage.REQUEST_UNDERSTANDING,
    "plan_document": ExecutionStage.PLANNING,
    "select_tools": ExecutionStage.TOOL_SELECTION,
    "tools": ExecutionStage.TOOL_EXECUTION,
    "write_document": ExecutionStage.DOCUMENT_WRITING,
    "validate_document": ExecutionStage.VALIDATION,
    "revise_document": ExecutionStage.REVISION,
    "generate_docx": ExecutionStage.DOCX_GENERATION,
    "finalize": ExecutionStage.FINALIZATION,
}

TERMINAL_FAILURE_STAGES: dict[str, ExecutionStage] = {
    "reject": ExecutionStage.REQUEST_UNDERSTANDING,
    "planning_failed": ExecutionStage.PLANNING,
    "tool_failed": ExecutionStage.TOOL_SELECTION,
    "tool_limit_reached": ExecutionStage.TOOL_EXECUTION,
    "content_failed": ExecutionStage.DOCUMENT_WRITING,
    "validation_failed": ExecutionStage.VALIDATION,
    "review_failed": ExecutionStage.VALIDATION,
    "revision_failed": ExecutionStage.REVISION,
    "docx_failed": ExecutionStage.DOCX_GENERATION,
}

STAGE_MESSAGES: dict[ExecutionStage, dict[str, str]] = {
    ExecutionStage.REQUEST_UNDERSTANDING: {
        "active": "Understanding request...",
        "completed": "Request understood.",
        "failed": "The request could not be understood.",
    },
    ExecutionStage.PLANNING: {
        "active": "Planning document...",
        "completed": "Document plan created.",
        "failed": "Document planning failed.",
    },
    ExecutionStage.TOOL_SELECTION: {
        "active": "Selecting tools...",
        "completed": "Tool selection complete.",
        "failed": "Tool selection failed.",
    },
    ExecutionStage.TOOL_EXECUTION: {
        "active": "Executing tools...",
        "completed": "Tool execution complete.",
        "skipped": "No tools were required.",
        "failed": "Tool execution failed.",
    },
    ExecutionStage.DOCUMENT_WRITING: {
        "active": "Writing document...",
        "completed": "Document written.",
        "failed": "Document writing failed.",
    },
    ExecutionStage.VALIDATION: {
        "active": "Validating document...",
        "completed": "Document validation complete.",
        "failed": "Document validation failed.",
    },
    ExecutionStage.REVISION: {
        "active": "Revising document...",
        "completed": "Document revision complete.",
        "skipped": "No revision was required.",
        "failed": "Document revision failed.",
    },
    ExecutionStage.DOCX_GENERATION: {
        "active": "Generating DOCX document...",
        "completed": "DOCX document generated.",
        "failed": "DOCX generation failed.",
    },
    ExecutionStage.FINALIZATION: {
        "active": "Finalizing document...",
        "completed": "Document finalized.",
        "failed": "Document finalization failed.",
    },
}


def _task_optional(stage: ExecutionStage) -> bool:
    return stage in OPTIONAL_EXECUTION_STAGES


def _plan_event_data(
    plan: DocumentPlan,
    statuses: dict[str, str],
) -> dict[str, Any]:
    placeholders = [
        {
            "text": placeholder.text,
            "section": section.heading,
            "reason": placeholder.reason,
        }
        for section in plan.sections
        for placeholder in section.placeholders
    ]
    return {
        "document_type": plan.document_type.value,
        "title": plan.title,
        "objective": plan.objective,
        "audience": plan.audience,
        "document_date": plan.document_date,
        "document_version": plan.document_version,
        "provided_facts": plan.provided_facts,
        "tasks": [
            {
                "id": task.id,
                "title": task.title,
                "description": task.description,
                "stage": task.stage.value,
                "optional": _task_optional(task.stage),
                "status": statuses.get(task.id, "pending"),
            }
            for task in plan.execution_tasks
        ],
        "conflicts": [
            {
                "requirements": conflict.requirements,
                "impact": conflict.impact,
                "resolution": conflict.resolution,
            }
            for conflict in plan.conflicts
        ],
        "missing_information": plan.missing_information,
        "assumptions": plan.assumptions,
        "clarification_questions": plan.clarification_questions,
        "placeholders": placeholders,
    }


class PlanProgressTracker:
    """Track task status against the graph's actual lifecycle nodes."""

    def __init__(self) -> None:
        self.plan: DocumentPlan | None = None
        self.statuses: dict[str, str] = {}
        self.tasks_by_stage: dict[ExecutionStage, list[str]] = {}
        self.started_stages: set[ExecutionStage] = set()
        self.completed_stages: set[ExecutionStage] = set()
        self.active_task_ids: list[str] = []

    def attach_plan(self, plan: DocumentPlan) -> dict[str, Any]:
        self.plan = plan
        self.tasks_by_stage = {}
        for task in plan.execution_tasks:
            self.tasks_by_stage.setdefault(task.stage, []).append(task.id)
            self.statuses[task.id] = (
                "completed" if task.stage in self.completed_stages else "pending"
            )
        return _plan_event_data(plan, self.statuses)

    def _stage_event(
        self,
        stage: ExecutionStage,
        status: str,
        *,
        force: bool = False,
    ) -> dict[str, Any] | None:
        task_ids = self.tasks_by_stage.get(stage, [])
        if not task_ids:
            return None
        if not force and all(
            self.statuses.get(task_id) == status for task_id in task_ids
        ):
            return None
        for task_id in task_ids:
            self.statuses[task_id] = status
        message = STAGE_MESSAGES[stage].get(status)
        if message is None:
            return None
        return {
            "stage": stage.value,
            "task_ids": task_ids,
            "status": status,
            "message": message,
        }

    def start_stage(self, stage: ExecutionStage) -> dict[str, Any] | None:
        self.started_stages.add(stage)
        self.active_task_ids = list(self.tasks_by_stage.get(stage, []))
        return self._stage_event(stage, "active")

    def complete_stage(
        self,
        stage: ExecutionStage,
        output: Any = None,
    ) -> dict[str, Any] | None:
        self.completed_stages.add(stage)
        self.active_task_ids = []
        failed = isinstance(output, dict) and output.get("status") == "failed"
        return self._stage_event(stage, "failed" if failed else "completed")

    def skip_stage(self, stage: ExecutionStage) -> dict[str, Any] | None:
        if self.plan is None or stage in self.started_stages:
            return None
        self.active_task_ids = []
        return self._stage_event(stage, "skipped")

    def fail_stage(self, stage: ExecutionStage) -> dict[str, Any] | None:
        task_ids = self.tasks_by_stage.get(stage, [])
        if task_ids:
            self.active_task_ids = []
            if all(self.statuses.get(task_id) == "failed" for task_id in task_ids):
                return None
            return self._stage_event(stage, "failed", force=True)
        return self.fail_active()

    def fail_active(self) -> dict[str, Any] | None:
        task_ids = list(self.active_task_ids)
        if not task_ids:
            return None
        self.active_task_ids = []
        for task_id in task_ids:
            self.statuses[task_id] = "failed"
        stage = next(
            (
                candidate
                for candidate, ids in self.tasks_by_stage.items()
                if ids == task_ids
            ),
            None,
        )
        message = (
            STAGE_MESSAGES[stage]["failed"]
            if stage is not None
            else "Task execution failed."
        )
        return {
            "stage": stage.value if stage is not None else "unknown",
            "task_ids": task_ids,
            "status": "failed",
            "message": message,
        }
