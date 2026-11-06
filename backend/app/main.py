from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from backend.app.agent.docx_generator import DEFAULT_OUTPUT_DIR
from backend.app.agent.graph import compile_agent_graph
from backend.app.api.agent import create_agent_router
from backend.app.config import Settings, get_settings


def create_app(
    settings: Settings | None = None,
    graph_factory: Callable[[], Any] | None = None,
    output_dir: Path | str | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    resolved_output = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR

    if graph_factory is None:

        def factory() -> Any:
            return compile_agent_graph(output_dir=resolved_output)

    else:
        factory = graph_factory

    app = FastAPI(title=settings.app_name, version="0.1.0")
    app.include_router(create_agent_router(factory, resolved_output))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
