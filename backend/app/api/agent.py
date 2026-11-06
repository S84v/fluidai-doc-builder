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
from backend.app.agent.state import create_initial_state

DOCX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
DOCUMENT_FILENAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*-[0-9a-f]{12}\.docx$")
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}

PROGRESS_MESSAGES = {
    "prepare_request": "Understanding request...",
    "plan_document": "Planning document...",
    "select_tools": "Selecting tools...",
    "tools": "Processing tool results...",
    "write_document": "Writing document...",
    "validate_document": "Validating document...",
    "revise_document": "Revising document...",
    "generate_docx": "Generating DOCX document...",
    "final_response": "Finalizing response...",
}

GraphFactory = Callable[[], Any]


class AgentRequest(BaseModel):
    request: str


def _sse_event(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


async def stream_agent_events(
    graph_factory: GraphFactory,
    request: str,
) -> AsyncIterator[bytes]:
    """Stream safe progress and terminal events for one graph invocation."""
    snapshot: dict[str, Any] = {}
    try:
        graph = graph_factory()
        async for update in graph.astream(
            create_initial_state(request),
            stream_mode="updates",
        ):
            if not isinstance(update, dict):
                continue
            for node_name, node_update in update.items():
                if not isinstance(node_update, dict):
                    continue
                for key in ("status", "output_filename", "terminal_reason"):
                    if key in node_update:
                        snapshot[key] = node_update[key]

                progress = PROGRESS_MESSAGES.get(node_name)
                if progress:
                    yield _sse_event("progress", {"message": progress})

                if node_name == "final_response":
                    final_text = node_update.get("final_response")
                    if isinstance(final_text, str) and final_text.strip():
                        snapshot["final_response"] = final_text

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
