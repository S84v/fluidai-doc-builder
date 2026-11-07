# Frontend Guide

This is the frontend reference for FluidAI Doc Builder. The frontend is a
single-page React application that submits document requests to the backend,
renders the execution plan and progress, and exposes the completed DOCX download.

The browser is responsible only for presentation and API communication. Document
planning, validation and generation stay in the backend.

## Stack

- React 19
- TypeScript
- Vite 8
- Plain CSS
- Node.js 22.12 or newer and npm

The repository uses an npm lockfile (`package-lock.json`) and has no frontend
environment variables or frontend-held API keys.

## Project layout

```text
frontend/
  index.html
  package.json
  package-lock.json
  tsconfig.json
  vite.config.ts
  src/
    main.tsx              React entry point
    App.tsx               Request, plan, progress and result UI
    api/agent.ts          SSE parser, API types and Fetch streaming client
    placeholderGroups.ts  Placeholder grouping by text and reason
    resultText.ts         Completion-text cleanup
    styles.css            Responsive layout and component styles
  tests/
    agent.test.mjs        SSE parsing and payload validation
    placeholderGroups.test.mjs
    resultText.test.mjs
```

## Prerequisites

The backend must be running at `http://127.0.0.1:8000` for local development.
See the [backend guide](../backend/README.md) for setup.

The current Vite version requires Node.js `^20.19.0` or `>=22.12.0`. The frontend
test script also uses Node's `--experimental-strip-types`, so Node.js 22.12 or
newer is the practical minimum for all frontend scripts.

## Install and run

```bash
cd frontend
npm ci
npm run dev
```

Open <http://127.0.0.1:5173>.

The Vite development server proxies these paths to `http://127.0.0.1:8000`:

- `/agent`
- `/documents`
- `/health`

The proxy is configured in `vite.config.ts`. If the backend port changes, update
the proxy target there.

## Scripts

| Command | What it does |
| --- | --- |
| `npm run dev` | Starts the Vite development server on port 5173. |
| `npm run build` | Runs `tsc --noEmit` and creates a production build in `dist/`. |
| `npm run preview` | Serves the production build locally. |
| `npm test` | Runs the Node test runner over `tests/*.test.mjs`. |

There is no separate lint script; TypeScript strict checking runs during
`npm run build`.

## UI workflow

1. **Request** — enter a business request or select one of the example requests.
2. **Agent** — the plan and task list appear as soon as the `plan` event arrives.
   Progress events update task statuses and the current stage.
3. **Plan details** — confirmed facts, missing information, assumptions,
   conflicts, clarification questions and planned placeholders are displayed.
4. **Result** — streaming text and progress appear while the request runs. On
   completion, the filename and download link are shown.

The page has four phases: `idle`, `running`, `completed` and `failed`. The layout
is two columns on desktop and a single column below 960px, with compact spacing
on smaller screens.

The placeholder section keeps its heading outside a vertically scrollable group
list. Groups show placeholder text, occurrence count, reason and one or more
locations. The bounded list height prevents a large placeholder inventory from
pushing the download result far down the page.

References: `src/App.tsx`, `src/styles.css`.

## SSE client

The frontend uses `fetch()` with a `POST` request to `/agent`; it does not use
native `EventSource`, because the request requires a JSON body.

`src/api/agent.ts` contains an incremental SSE parser:

- buffers partial network chunks;
- accepts `\n\n`, `\r\n\r\n` or `\r\r` event delimiters;
- joins multi-line `data:` fields;
- ignores comments and unknown event names;
- stops reading after `complete` or `error`;
- reports malformed JSON, missing fields and unexpected types as
  `AgentStreamError`;
- reports streams that end without a terminal event.

Known events and their UI handling:

| Event | Frontend behavior |
| --- | --- |
| `plan` | Replaces the execution plan and initializes task statuses. |
| `progress` | Updates the current stage, message and matching task statuses. |
| `metadata` | Updates effective document date and version in the plan summary. |
| `placeholders` | Replaces the rendered placeholder list used by the remaining-placeholders section. |
| `token` | Appends the backend's single final response event to the result text. |
| `complete` | Stores the filename and relative download URL and marks the request complete. |
| `error` | Shows a safe failure message and marks the request failed. |

The backend's `token` event is a single final response event, not incremental
model token streaming.

References: `src/api/agent.ts`, `tests/agent.test.mjs`.

## Completion and download

The `complete` event supplies a relative URL such as
`/documents/report-a1b2c3d4e5f6.docx`. The UI renders that URL in an anchor with
the filename as the download attribute.

The frontend does not construct document paths. The backend validates the
filename and keeps downloads inside its configured output directory.

`src/resultText.ts` removes the redundant success prefix and filename from the
completed display, and suppresses the backend's `Validation notes:` summary. The
validation report and warning classification remain intact in the backend and
SSE response; only the completed download text is cleaned.

References: `src/resultText.ts`, `tests/resultText.test.mjs`.

## Placeholder presentation

`groupPlaceholders()` combines occurrences with the same placeholder text and
reason, increments the count, and deduplicates locations. Different reasons for
the same placeholder stay in separate groups.

References: `src/placeholderGroups.ts`, `tests/placeholderGroups.test.mjs`.

## Tests

```bash
cd frontend
npm test
npm run build
```

`npm test` covers:

- SSE parsing and strict payload validation;
- terminal and error event handling;
- placeholder grouping, counts and locations;
- completion-text cleanup and validation-notes suppression.

`npm run build` verifies TypeScript and produces the production bundle. There is
no committed browser automation or end-to-end test framework; do not treat
Playwright or another browser tool as a project script.

## Related documentation

- [Project overview and quick start](../README.md)
- [Backend guide](../backend/README.md)
- [Coding-agent instructions](../AGENTS.md)
- [Package scripts](package.json)
