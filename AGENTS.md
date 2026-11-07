# AGENTS.md

Persistent project-specific instructions for FluidAI Doc Builder. Read this file
before making changes.

## 1. Project purpose

FluidAI Doc Builder is a proof-of-concept autonomous AI agent that generates
structured business documents in Microsoft Word format.

The objective is to demonstrate Python AI engineering, autonomous planning,
LangGraph, tool calling and good software engineering practices.

Prioritise simplicity, readability and explainability. Avoid unnecessary
production-scale complexity.

## 2. Development practices

- Inspect relevant existing files before making changes.
- Work on one logical task at a time.
- Modify only files required by the current task.
- Preserve established architecture and interfaces.
- Prefer straightforward Python functions over unnecessary abstractions.
- Use meaningful names, type hints and clear error handling.
- Do not introduce dependencies without justification.
- Do not overwrite unrelated work.
- Do not commit or push changes unless explicitly requested.
- Never claim tests or commands succeeded without running them.

### Code comments and documentation

- Write clear, meaningful comments and docstrings wherever they improve
  understanding.
- Explain why non-obvious logic exists, not merely what the code does.
- Add comments for complex control flow, architectural decisions, important
  constraints, edge cases and error handling.
- Pay particular attention to LangGraph workflows, state transitions, tool
  execution, asynchronous operations and SSE streaming.
- Use docstrings for important modules, classes and functions where useful.
- Avoid redundant comments, excessive documentation and line-by-line narration.
- Keep comments accurate and update them when implementation changes.

## 3. Approved technology stack

Backend:

- Python 3.12
- FastAPI
- LangGraph with explicit `StateGraph` orchestration
- LangChain and `ChatDeepSeek`
- Pydantic
- python-docx

Frontend:

- React
- TypeScript
- Vite
- Plain CSS

Development:

- uv
- Ruff
- Pytest
- HTTPX

Preserve these API contracts:

- `POST /agent`: accepts `{"request": "..."}` and returns SSE.
- `GET /documents/{filename}`: downloads a generated DOCX.
- `GET /health`: application health check.

Keep agent orchestration independent of HTTP endpoint logic.

- Prefer `langchain-core` interfaces and provider-specific integrations.
- Do not install the full `langchain` package unless a concrete
  implementation requirement justifies it.

## 4. Agent architecture

- Use an explicit LangGraph `StateGraph`.
- Keep graph state typed and structured.
- Validate model-generated data using Pydantic.
- Keep execution state isolated between requests.
- Use conditional routing for agent decisions and tool execution.
- Use LangChain tools and `ToolNode` for model-selected actions.
- Keep available tools minimal and purposeful.
- Prefer deterministic Python functions for predictable operations.
- Bound execution iterations and retry attempts.
- Never permit arbitrary shell execution or unrestricted filesystem access.
- Centralise DeepSeek model configuration.
- Verify provider and framework compatibility before relying on features.

Do not introduce multiple agents, RAG, conversation memory, databases, message
queues or other excluded technologies without an explicit requirement and
technical justification.

## 5. Streaming

Use one SSE connection for `POST /agent`.

Distinguish between:

- Agent progress events.
- Final user-facing response tokens.
- Terminal completion or error events.

Never stream internal reasoning, raw planning model output, tool calls, raw tool
results, internal messages or document-generation content.

A curated, validated `DocumentPlan` summary may be streamed through the
dedicated `plan` event. Only sanitized plan fields, task definitions, conflicts,
assumptions, missing information, clarification questions and explained
placeholders may be exposed; never chain-of-thought or raw model output.

Only stream user-facing response text from the dedicated final-response node.

Use asynchronous execution and maintain a consistent event protocol.

The frontend must use Fetch API streaming, not native `EventSource` for the POST
request.

## 6. Document generation and security

- Generate structured document content.
- Use python-docx for deterministic Word document generation.
- Verify generated documents before reporting success.
- Store documents only under `var/output/`.
- Never construct filesystem paths directly from user input.
- Validate filenames and ensure resolved paths remain inside the designated
  output directory.

Generated filenames must contain a short descriptive slug, a unique 12-character
identifier and exactly one valid `.docx` extension.

## 7. Configuration and security

- Use pydantic-settings for application configuration.
- Keep credentials in a local `.env` file.
- Never commit or expose secrets.
- Never expose the DeepSeek API key through frontend variables.
- Keep `.env.example` updated when configuration changes.
- Validate incoming requests, tool arguments and generated plans.
- Make assumptions explicit rather than inventing facts.
- Request clarification when essential information cannot safely be assumed.

## 8. Python quality and testing

Use Python 3.12 and uv for dependency management.

Ruff configuration is maintained in `pyproject.toml`.

Before completing Python implementation tasks, run:

```bash
uv run ruff check .
uv run ruff format --check .
```

Format using:

```bash
uv run ruff format .
```

Run relevant tests using:

```bash
uv run pytest
```

Testing principles:

- Normal tests must not require paid API calls.
- Use fake or mocked LLMs for deterministic tests.
- Keep live provider integration tests separate.
- Test failure cases as well as successful execution.
- Add tests for newly introduced functionality.
- Use HTTPX `ASGITransport` for asynchronous API tests.
- Update `uv.lock` whenever dependencies change.

## 9. Frontend practices

- Verify Node.js and npm versions before frontend setup.
- Use ordinary React components and state.
- Keep frontend responsibilities limited to presentation and API communication.
- Use the Vite development proxy for local backend communication.
- Avoid unnecessary component libraries and state-management frameworks.

## 10. Scope control

This is a local proof-of-concept application.

Do not introduce:

- Microservices.
- Multiple autonomous agents.
- Databases or persistence.
- Authentication.
- Redis or message queues.
- Cloud deployment infrastructure.
- Kubernetes or container orchestration.
- Unnecessary frameworks or dependencies.

Any architectural deviation must have a clear technical justification.
