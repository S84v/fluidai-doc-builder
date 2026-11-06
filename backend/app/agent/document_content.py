"""Pydantic models for generated document content."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.agent.models import DocumentType


class _ContentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ContentSection(_ContentModel):
    """One ordered section of generated document content."""

    heading: str = Field(min_length=1)
    paragraphs: list[str] = Field(default_factory=list)
    bullets: list[str] = Field(default_factory=list)

    @field_validator("paragraphs", "bullets")
    @classmethod
    def _reject_empty_items(cls, values: list[str]) -> list[str]:
        if any(not value for value in values):
            raise ValueError("section content items must not be empty")
        return values

    @model_validator(mode="after")
    def _require_body(self) -> Self:
        if not self.paragraphs and not self.bullets:
            raise ValueError("a section must contain a paragraph or bullet")
        return self


class DocumentMetadata(_ContentModel):
    """Optional document-level metadata."""

    date: str | None = None
    prepared_for: str | None = None
    prepared_by: str | None = None
    version: str | None = None


class MeetingDetails(_ContentModel):
    """Meeting metadata rendered prominently in meeting minutes."""

    date: str | None = None
    time: str | None = None
    location: str | None = None
    facilitator: str | None = None
    attendees: list[str] = Field(default_factory=list)

    @field_validator("attendees")
    @classmethod
    def _reject_empty_attendees(cls, values: list[str]) -> list[str]:
        if any(not value for value in values):
            raise ValueError("attendee names must not be empty")
        return values


class ActionItem(_ContentModel):
    """One meeting action item."""

    action: str = Field(min_length=1)
    owner: str | None = None
    due_date: str | None = None
    status: str | None = None


class Milestone(_ContentModel):
    """One project or proposal milestone."""

    name: str = Field(min_length=1)
    target_date: str | None = None
    owner: str | None = None
    status: str | None = None


class SopStep(_ContentModel):
    """One numbered procedure step."""

    instruction: str = Field(min_length=1)
    note: str | None = None


class Requirement(_ContentModel):
    """One product requirement."""

    identifier: str | None = None
    description: str = Field(min_length=1)
    priority: str | None = None
    acceptance_criteria: str | None = None


class TechnicalComponent(_ContentModel):
    """One technical component and its interfaces."""

    name: str = Field(min_length=1)
    responsibility: str = Field(min_length=1)
    interfaces: list[str] = Field(default_factory=list)

    @field_validator("interfaces")
    @classmethod
    def _reject_empty_interfaces(cls, values: list[str]) -> list[str]:
        if any(not value for value in values):
            raise ValueError("component interfaces must not be empty")
        return values


class DocumentContent(_ContentModel):
    """Validated content for one generated business document."""

    document_type: DocumentType
    title: str = Field(min_length=1)
    audience: str | None = None
    metadata: DocumentMetadata | None = None
    sections: list[ContentSection] = Field(min_length=1)
    meeting: MeetingDetails | None = None
    action_items: list[ActionItem] = Field(default_factory=list)
    milestones: list[Milestone] = Field(default_factory=list)
    sop_steps: list[SopStep] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)
    components: list[TechnicalComponent] = Field(default_factory=list)
