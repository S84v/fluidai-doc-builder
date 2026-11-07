"""HTTP and SSE integration for the document agent."""

import asyncio
import json
import re
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from backend.app.agent.graph import GENERIC_FAILURE_RESPONSE
from backend.app.agent.models import ExecutionStage
from backend.app.agent.state import create_initial_state
from backend.app.api.progress import (
    NODE_STAGE_MAP,
    TERMINAL_FAILURE_STAGES,
    PlanProgressTracker,
)

DOCX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
DOCUMENT_FILENAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*-[0-9a-f]{12}\.docx$")
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}

_KNOWN_NODES = (
    set(NODE_STAGE_MAP)
    | set(TERMINAL_FAILURE_STAGES)
    | {
        "final_response",
    }
)

GraphFactory = Callable[[], Any]


class AgentRequest(BaseModel):
    request: str


def _sse_event(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def _serialize_placeholders(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for placeholder in value:
        if isinstance(placeholder, dict):
            result.append(placeholder)
        elif hasattr(placeholder, "model_dump"):
            result.append(placeholder.model_dump())
    return result


async def stream_agent_events(
    graph_factory: GraphFactory,
    request: str,
) -> AsyncIterator[bytes]:
    """Stream safe plan, task progress and terminal events for one run."""
    snapshot: dict[str, Any] = {}
    tracker = PlanProgressTracker()
    plan_sent = False
    try:
        graph = graph_factory()
        async for event in graph.astream_events(
            create_initial_state(request),
            version="v2",
        ):
            if event.get("event") not in ("on_chain_start", "on_chain_end"):
                continue

            metadata = event.get("metadata", {})
            node_name = metadata.get("langgraph_node")
            if (
                not isinstance(node_name, str)
                or node_name not in _KNOWN_NODES
                or event.get("name") != node_name
            ):
                continue

            is_start = event["event"] == "on_chain_start"
            node_output = (
                event.get("data", {}).get("output")
                if event["event"] == "on_chain_end"
                else None
            )

            if isinstance(node_output, dict):
                for key in ("status", "output_filename", "terminal_reason"):
                    if key in node_output:
                        snapshot[key] = node_output[key]

            if node_name == "prepare_request":
                if is_start:
                    tracker.start_stage(ExecutionStage.REQUEST_UNDERSTANDING)
                else:
                    tracker.complete_stage(
                        ExecutionStage.REQUEST_UNDERSTANDING,
                        node_output,
                    )
                continue

            if node_name == "plan_document":
                if is_start:
                    tracker.start_stage(ExecutionStage.PLANNING)
                else:
                    tracker.complete_stage(ExecutionStage.PLANNING, node_output)
                    plan = (
                        node_output.get("plan")
                        if isinstance(node_output, dict)
                        else None
                    )
                    if plan is not None and not plan_sent:
                        yield _sse_event("plan", tracker.attach_plan(plan))
                        plan_sent = True
                continue

            if node_name == "final_response":
                if not is_start and isinstance(node_output, dict):
                    final_text = node_output.get("final_response")
                    if isinstance(final_text, str) and final_text.strip():
                        snapshot["final_response"] = final_text
                continue

            if node_name in TERMINAL_FAILURE_STAGES:
                if not is_start:
                    progress = tracker.fail_stage(TERMINAL_FAILURE_STAGES[node_name])
                    if progress is not None:
                        yield _sse_event("progress", progress)
                continue

            stage = NODE_STAGE_MAP.get(node_name)
            if stage is None:
                continue

            if is_start:
                if stage is ExecutionStage.DOCUMENT_WRITING:
                    skipped = tracker.skip_stage(ExecutionStage.TOOL_EXECUTION)
                    if skipped is not None:
                        yield _sse_event("progress", skipped)
                elif stage is ExecutionStage.DOCX_GENERATION:
                    skipped = tracker.skip_stage(ExecutionStage.REVISION)
                    if skipped is not None:
                        yield _sse_event("progress", skipped)

                progress = tracker.start_stage(stage)
                if progress is not None:
                    yield _sse_event("progress", progress)
            else:
                progress = tracker.complete_stage(stage, node_output)
                if progress is not None:
                    yield _sse_event("progress", progress)

                if node_name == "generate_docx" and isinstance(node_output, dict):
                    generated_filename = node_output.get("output_filename")
                    if (
                        node_output.get("status") == "docx_generation"
                        and isinstance(generated_filename, str)
                        and DOCUMENT_FILENAME_RE.fullmatch(generated_filename)
                    ):
                        document_date = node_output.get("document_date")
                        document_version = node_output.get("document_version")
                        if (
                            isinstance(document_date, str)
                            and document_date
                            and isinstance(document_version, str)
                            and document_version
                        ):
                            yield _sse_event(
                                "metadata",
                                {
                                    "document_date": document_date,
                                    "document_version": document_version,
                                },
                            )
                        yield _sse_event(
                            "placeholders",
                            {
                                "placeholders": _serialize_placeholders(
                                    node_output.get("rendered_placeholders")
                                )
                            },
                        )

        status = snapshot.get("status")
        filename = snapshot.get("output_filename")
        final_text = snapshot.get("final_response")

        if status == "completed" and filename and final_text:
            yield _sse_event("token", {"text": final_text})
            yield _sse_event(
                "complete",
                {
                    "filename": filename,
                    "url": f"/documents/{quote(filename)}",
                },
            )
        else:
            yield _sse_event(
                "error",
                {
                    "message": (
                        final_text
                        if isinstance(final_text, str) and final_text.strip()
                        else GENERIC_FAILURE_RESPONSE
                    )
                },
            )
    except asyncio.CancelledError:
        raise
    except Exception:
        progress = tracker.fail_active()
        if progress is not None:
            yield _sse_event("progress", progress)
        yield _sse_event("error", {"message": GENERIC_FAILURE_RESPONSE})


def _resolve_document(output_dir: Path, filename: str) -> Path | None:
    if not DOCUMENT_FILENAME_RE.fullmatch(filename):
        return None

    resolved_dir = output_dir.expanduser().resolve()
    candidate = (resolved_dir / filename).resolve()
    if not candidate.is_relative_to(resolved_dir):
        return None
    if candidate.parent != resolved_dir:
        return None
    if not candidate.is_file():
        return None
    return candidate


def create_agent_router(
    graph_factory: GraphFactory,
    output_dir: Path,
) -> APIRouter:
    router = APIRouter()

    @router.post("/agent")
    async def run_agent_request(payload: AgentRequest) -> StreamingResponse:
        return StreamingResponse(
            stream_agent_events(graph_factory, payload.request),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    @router.get("/documents/{filename}")
    async def download_document(filename: str) -> FileResponse:
        path = _resolve_document(output_dir, filename)
        if path is None:
            raise HTTPException(status_code=404, detail="Document not found.")
        return FileResponse(
            path,
            media_type=DOCX_MEDIA_TYPE,
            filename=filename,
            content_disposition_type="attachment",
        )

    return router
