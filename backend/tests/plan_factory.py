"""Shared test helpers for request-specific execution plans."""

from backend.app.agent.models import ExecutionStage, ExecutionTask


def make_execution_tasks() -> list[ExecutionTask]:
    return [
        ExecutionTask(
            id="understand_request",
            title="Understand the request",
            description="Identify the requested deliverable and constraints.",
            stage=ExecutionStage.REQUEST_UNDERSTANDING,
        ),
        ExecutionTask(
            id="plan_document",
            title="Plan the document",
            description="Create the document structure and planning details.",
            stage=ExecutionStage.PLANNING,
        ),
        ExecutionTask(
            id="write_document",
            title="Write the document",
            description="Produce the requested document content.",
            stage=ExecutionStage.DOCUMENT_WRITING,
        ),
        ExecutionTask(
            id="validate_document",
            title="Validate the document",
            description="Check structure, consistency and factual support.",
            stage=ExecutionStage.VALIDATION,
        ),
        ExecutionTask(
            id="generate_docx",
            title="Generate the DOCX",
            description="Render the validated document as a Word file.",
            stage=ExecutionStage.DOCX_GENERATION,
        ),
        ExecutionTask(
            id="finalize_document",
            title="Finalize the document",
            description="Complete the document run.",
            stage=ExecutionStage.FINALIZATION,
        ),
    ]
