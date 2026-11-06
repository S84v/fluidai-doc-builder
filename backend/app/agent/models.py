"""Pydantic models for document planning."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


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


class PlanSection(BaseModel):
    """One planned section of a document."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    heading: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    key_points: list[str] = Field(default_factory=list)
    placeholders: list[str] = Field(default_factory=list)


class DocumentPlan(BaseModel):
    """Validated plan produced before document content is generated."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    document_type: DocumentType
    title: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    audience: str | None = None
    sections: list[PlanSection] = Field(min_length=1)
    provided_facts: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
