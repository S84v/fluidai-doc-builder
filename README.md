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

## API

- `GET /health` — service health.
- `POST /agent` — planned.
- `GET /documents/{filename}` — planned.

## Layout

```text
backend/       FastAPI application and tests
var/output/    Generated DOCX files (gitignored)
```
