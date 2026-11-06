"""Pydantic models for generated document content."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.agent.models import DocumentType


class ContentSection(BaseModel):
    """One ordered section of generated document content."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

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


class DocumentContent(BaseModel):
    """Validated content for one generated business document."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    document_type: DocumentType
    title: str = Field(min_length=1)
    sections: list[ContentSection] = Field(min_length=1)
