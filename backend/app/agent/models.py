"""Pydantic models for document planning."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DocumentType(StrEnum):
    """Supported structured business-document types."""

    BUSINESS_PROPOSAL = "business_proposal"
    MEETING_MINUTES = "meeting_minutes"
    PROJECT_PLAN = "project_plan"
    BUSINESS_REPORT = "business_report"
    TECHNICAL_DESIGN = "technical_design"
    SOP = "sop"
    PRODUCT_SPECIFICATION = "product_specification"
    OTHER = "other"


class ExecutionStage(StrEnum):
    """Graph stages that execution tasks can be associated with."""

    REQUEST_UNDERSTANDING = "request_understanding"
    PLANNING = "planning"
    TOOL_SELECTION = "tool_selection"
    TOOL_EXECUTION = "tool_execution"
    DOCUMENT_WRITING = "document_writing"
    VALIDATION = "validation"
    REVISION = "revision"
    DOCX_GENERATION = "docx_generation"
    FINALIZATION = "finalization"


EXECUTION_STAGE_ORDER: tuple[ExecutionStage, ...] = tuple(ExecutionStage)
OPTIONAL_EXECUTION_STAGES: frozenset[ExecutionStage] = frozenset(
    {ExecutionStage.TOOL_EXECUTION, ExecutionStage.REVISION}
)
REQUIRED_TASK_STAGES: frozenset[ExecutionStage] = frozenset(
    {
        ExecutionStage.DOCUMENT_WRITING,
        ExecutionStage.VALIDATION,
        ExecutionStage.DOCX_GENERATION,
    }
)


def _normalize_heading(heading: str) -> str:
    return " ".join(heading.split()).casefold()


class ExecutionTask(BaseModel):
    """One request-specific task tied to a real graph stage."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(
        min_length=2,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_-]*$",
    )
    title: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=500)
    stage: ExecutionStage


class RequirementConflict(BaseModel):
    """A conflict between requested requirements."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    requirements: list[str] = Field(min_length=2)
    impact: str = Field(min_length=1)
    resolution: str | None = None


class PlanPlaceholder(BaseModel):
    """A planned placeholder and the reason it is needed."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(pattern=r"^\[[^\[\]]+\]$")
    reason: str = Field(min_length=1)


class PlanSection(BaseModel):
    """One planned section of a document."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    heading: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    key_points: list[str] = Field(default_factory=list)
    placeholders: list[PlanPlaceholder] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_placeholder_uniqueness(self) -> Self:
        texts = [placeholder.text for placeholder in self.placeholders]
        if len(texts) != len(set(texts)):
            raise ValueError("placeholder text must be unique within a plan section")
        return self


class DocumentPlan(BaseModel):
    """Validated plan produced before document content is generated."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    document_type: DocumentType
    title: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    audience: str | None = None
    document_date: str | None = None
    document_version: str | None = None
    sections: list[PlanSection] = Field(min_length=1)
    provided_facts: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    execution_tasks: list[ExecutionTask] = Field(min_length=1, max_length=20)
    conflicts: list[RequirementConflict] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_execution_tasks(self) -> Self:
        task_ids = [task.id for task in self.execution_tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("execution task IDs must be unique")

        stages = {task.stage for task in self.execution_tasks}
        missing = REQUIRED_TASK_STAGES - stages
        if missing:
            names = ", ".join(sorted(stage.value for stage in missing))
            raise ValueError(f"execution plan is missing required stages: {names}")
        return self

    @model_validator(mode="after")
    def _validate_unique_section_headings(self) -> Self:
        headings = [_normalize_heading(section.heading) for section in self.sections]
        if len(headings) != len(set(headings)):
            raise ValueError(
                "section headings must be unique after whitespace and "
                "case normalization"
            )
        return self
