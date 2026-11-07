# FluidAI Doc Builder

Autonomous AI document-generation agent built with FastAPI, LangGraph, and
DeepSeek.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
cp .env.example .env
```

`DEEPSEEK_API_KEY` is optional. Without it the service still starts, but
agent calls that require the model will fail.

## Run

```bash
uv run uvicorn backend.app.main:app --reload
```

Health check: `GET http://127.0.0.1:8000/health`

## Test and lint

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Configuration

Settings are loaded from the environment or a local `.env` file:

| Variable | Default | Required |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | unset | no |
| `DEEPSEEK_MODEL` | `deepseek-flash` | no |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | no |
| `DEEPSEEK_REASONING_EFFORT` | `low` | no |

## API

- `GET /health` — service health.
- `POST /agent` — accepts `{"request": "..."}` and returns SSE events.
- `GET /documents/{filename}` — downloads a generated DOCX from `var/output/`.

SSE events:

- `plan` — curated execution plan with request-specific tasks, conflicts,
  assumptions, missing information, clarification questions and planned
  placeholders.
- `progress` — task IDs, graph stage, status and authored progress message.
- `metadata` — effective document date and version used for the generated DOCX.
- `placeholders` — final placeholder inventory for the generated DOCX with
  location and justification.
- `token` — final user-facing response text.
- `complete` — generated filename and relative download URL.
- `error` — safe terminal failure message.

Missing document dates default to the system date at generation; missing
document versions default to `1.0`. Meeting dates and other task dates are never
substituted with the document date.

Stream a request locally with `curl -N`:

```bash
curl -N -X POST http://127.0.0.1:8000/agent \
  -H 'Content-Type: application/json' \
  -d '{"request": "Create a short business report for the Atlas project."}'
```

The completion event includes a relative download URL under `/documents/`.

## Layout

```text
backend/       FastAPI application, agent graph and tests
frontend/      React, TypeScript and Vite interface
var/output/    Generated DOCX files (gitignored)
```
