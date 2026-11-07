export type TaskStatus =
  | "pending"
  | "active"
  | "completed"
  | "skipped"
  | "failed";

export type PlanTask = {
  id: string;
  title: string;
  description: string;
  stage: string;
  optional: boolean;
  status: TaskStatus;
};

export type PlanConflict = {
  requirements: string[];
  impact: string;
  resolution: string | null;
};

export type PlanPlaceholder = {
  text: string;
  section: string;
  reason: string;
};

export type RenderedPlaceholder = {
  text: string;
  location: string;
  reason: string;
};

export type ExecutionPlan = {
  documentType: string;
  title: string;
  objective: string;
  audience: string | null;
  documentDate: string | null;
  documentVersion: string | null;
  providedFacts: string[];
  tasks: PlanTask[];
  conflicts: PlanConflict[];
  missingInformation: string[];
  assumptions: string[];
  clarificationQuestions: string[];
  placeholders: PlanPlaceholder[];
};

export type AgentEvent =
  | { type: "plan"; plan: ExecutionPlan }
  | {
      type: "progress";
      stage: string;
      taskIds: string[];
      status: TaskStatus;
      message: string;
    }
  | { type: "metadata"; documentDate: string; documentVersion: string }
  | { type: "placeholders"; placeholders: RenderedPlaceholder[] }
  | { type: "token"; text: string }
  | { type: "complete"; filename: string; url: string }
  | { type: "error"; message: string };

export type RawSseEvent = {
  event: string;
  data: string;
};

type EventBoundary = {
  index: number;
  length: number;
};

const TASK_STATUSES: readonly TaskStatus[] = [
  "pending",
  "active",
  "completed",
  "skipped",
  "failed",
];

const SUPPORTED_EVENT_NAMES = new Set([
  "plan",
  "progress",
  "metadata",
  "placeholders",
  "token",
  "complete",
  "error",
]);

export class AgentStreamError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AgentStreamError";
  }
}

function findEventBoundary(buffer: string): EventBoundary | null {
  const delimiters = ["\r\n\r\n", "\n\n", "\r\r"];
  let boundary: EventBoundary | null = null;

  for (const delimiter of delimiters) {
    const index = buffer.indexOf(delimiter);
    if (index === -1) {
      continue;
    }
    if (boundary === null || index < boundary.index) {
      boundary = { index, length: delimiter.length };
    }
  }

  return boundary;
}

function parseEventBlock(block: string): RawSseEvent | null {
  let event = "message";
  const dataLines: string[] = [];

  for (const line of block.split(/\r\n|\r|\n/)) {
    if (line === "" || line.startsWith(":")) {
      continue;
    }

    const separator = line.indexOf(":");
    const field = separator === -1 ? line : line.slice(0, separator);
    let value = separator === -1 ? "" : line.slice(separator + 1);

    if (value.startsWith(" ")) {
      value = value.slice(1);
    }

    if (field === "event") {
      event = value;
    } else if (field === "data") {
      dataLines.push(value);
    }
  }

  if (dataLines.length === 0) {
    return null;
  }

  return { event, data: dataLines.join("\n") };
}

/**
 * Incremental SSE parser. The buffer keeps partial event blocks and split
 * delimiters until the next network chunk arrives.
 */
export class SseParser {
  private buffer = "";

  push(chunk: string): RawSseEvent[] {
    this.buffer += chunk;
    const events: RawSseEvent[] = [];

    while (true) {
      const boundary = findEventBoundary(this.buffer);
      if (boundary === null) {
        break;
      }

      const block = this.buffer.slice(0, boundary.index);
      this.buffer = this.buffer.slice(boundary.index + boundary.length);
      const event = parseEventBlock(block);
      if (event !== null) {
        events.push(event);
      }
    }

    return events;
  }

  flush(): RawSseEvent[] {
    const remainder = this.buffer.trim();
    this.buffer = "";

    if (remainder === "") {
      return [];
    }

    const event = parseEventBlock(remainder);
    return event === null ? [] : [event];
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requireString(value: unknown): string {
  if (typeof value !== "string") {
    throw new AgentStreamError(
      "The server returned an unexpected streaming response.",
    );
  }
  return value;
}

function requireNonEmptyString(value: unknown): string {
  const text = requireString(value);
  if (text.trim().length === 0) {
    throw new AgentStreamError(
      "The server returned an unexpected streaming response.",
    );
  }
  return text;
}

function optionalNonEmptyString(value: unknown): string | null {
  if (value === null) {
    return null;
  }
  return requireNonEmptyString(value);
}

function requireBoolean(value: unknown): boolean {
  if (typeof value !== "boolean") {
    throw new AgentStreamError(
      "The server returned an unexpected streaming response.",
    );
  }
  return value;
}

function requireTaskStatus(value: unknown): TaskStatus {
  if (typeof value === "string" && TASK_STATUSES.includes(value as TaskStatus)) {
    return value as TaskStatus;
  }
  throw new AgentStreamError(
    "The server returned an unexpected task status.",
  );
}

function requireArray(value: unknown): unknown[] {
  if (!Array.isArray(value)) {
    throw new AgentStreamError(
      "The server returned an unexpected streaming response.",
    );
  }
  return value;
}

function requireStringArray(value: unknown): string[] {
  return requireArray(value).map(requireNonEmptyString);
}

function parseTask(value: unknown): PlanTask {
  if (!isRecord(value)) {
    throw new AgentStreamError(
      "The server returned an unexpected execution task.",
    );
  }
  return {
    id: requireNonEmptyString(value.id),
    title: requireNonEmptyString(value.title),
    description: requireNonEmptyString(value.description),
    stage: requireNonEmptyString(value.stage),
    optional: requireBoolean(value.optional),
    status: requireTaskStatus(value.status),
  };
}

function parseConflict(value: unknown): PlanConflict {
  if (!isRecord(value)) {
    throw new AgentStreamError(
      "The server returned an unexpected requirement conflict.",
    );
  }
  return {
    requirements: requireStringArray(value.requirements),
    impact: requireNonEmptyString(value.impact),
    resolution: optionalNonEmptyString(value.resolution),
  };
}

function parsePlanPlaceholder(value: unknown): PlanPlaceholder {
  if (!isRecord(value)) {
    throw new AgentStreamError(
      "The server returned an unexpected planned placeholder.",
    );
  }
  return {
    text: requireNonEmptyString(value.text),
    section: requireNonEmptyString(value.section),
    reason: requireNonEmptyString(value.reason),
  };
}

function parseRenderedPlaceholder(value: unknown): RenderedPlaceholder {
  if (!isRecord(value)) {
    throw new AgentStreamError(
      "The server returned an unexpected rendered placeholder.",
    );
  }
  return {
    text: requireNonEmptyString(value.text),
    location: requireNonEmptyString(value.location),
    reason: requireNonEmptyString(value.reason),
  };
}

function parsePlan(value: unknown): ExecutionPlan {
  if (!isRecord(value)) {
    throw new AgentStreamError(
      "The server returned an unexpected execution plan.",
    );
  }

  return {
    documentType: requireNonEmptyString(value.document_type),
    title: requireNonEmptyString(value.title),
    objective: requireNonEmptyString(value.objective),
    audience: optionalNonEmptyString(value.audience),
    documentDate: optionalNonEmptyString(value.document_date),
    documentVersion: optionalNonEmptyString(value.document_version),
    providedFacts: requireStringArray(value.provided_facts),
    tasks: requireArray(value.tasks).map(parseTask),
    conflicts: requireArray(value.conflicts).map(parseConflict),
    missingInformation: requireStringArray(value.missing_information),
    assumptions: requireStringArray(value.assumptions),
    clarificationQuestions: requireStringArray(value.clarification_questions),
    placeholders: requireArray(value.placeholders).map(parsePlanPlaceholder),
  };
}

export function parseAgentEvent(raw: RawSseEvent): AgentEvent | null {
  if (!SUPPORTED_EVENT_NAMES.has(raw.event)) {
    return null;
  }

  let payload: unknown;
  try {
    payload = JSON.parse(raw.data);
  } catch {
    throw new AgentStreamError(
      "The server returned malformed streaming data. Please try again.",
    );
  }

  if (!isRecord(payload)) {
    throw new AgentStreamError(
      "The server returned an unexpected streaming response.",
    );
  }

  if (raw.event === "plan") {
    return { type: "plan", plan: parsePlan(payload) };
  }

  if (raw.event === "progress") {
    return {
      type: "progress",
      stage: requireNonEmptyString(payload.stage),
      taskIds: requireStringArray(payload.task_ids),
      status: requireTaskStatus(payload.status),
      message: requireNonEmptyString(payload.message),
    };
  }

  if (raw.event === "metadata") {
    return {
      type: "metadata",
      documentDate: requireNonEmptyString(payload.document_date),
      documentVersion: requireNonEmptyString(payload.document_version),
    };
  }

  if (raw.event === "placeholders") {
    return {
      type: "placeholders",
      placeholders: requireArray(payload.placeholders).map(
        parseRenderedPlaceholder,
      ),
    };
  }

  if (raw.event === "token") {
    return { type: "token", text: requireNonEmptyString(payload.text) };
  }

  if (raw.event === "complete") {
    return {
      type: "complete",
      filename: requireNonEmptyString(payload.filename),
      url: requireNonEmptyString(payload.url),
    };
  }

  if (raw.event === "error") {
    return { type: "error", message: requireNonEmptyString(payload.message) };
  }

  return null;
}

function isAbortError(error: unknown): boolean {
  return (
    (error instanceof DOMException && error.name === "AbortError") ||
    (error instanceof Error && error.name === "AbortError")
  );
}

export async function streamAgentRequest(
  request: string,
  onEvent: (event: AgentEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  let response: Response;

  try {
    response = await fetch("/agent", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify({ request }),
      signal,
    });
  } catch (error) {
    if (isAbortError(error)) {
      throw error;
    }
    throw new AgentStreamError(
      "Unable to reach the document service. Please try again.",
    );
  }

  if (!response.ok) {
    throw new AgentStreamError(
      "The document service could not process the request. Please try again.",
    );
  }

  if (response.body === null) {
    throw new AgentStreamError(
      "The document service returned an empty streaming response.",
    );
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  const parser = new SseParser();
  let terminal = false;

  const handleRawEvent = (raw: RawSseEvent): void => {
    const event = parseAgentEvent(raw);
    if (event === null) {
      return;
    }

    if (event.type === "complete" || event.type === "error") {
      terminal = true;
    }
    onEvent(event);
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      const text = decoder.decode(value, { stream: true });
      for (const raw of parser.push(text)) {
        handleRawEvent(raw);
        if (terminal) {
          break;
        }
      }

      if (terminal) {
        try {
          await reader.cancel();
        } catch {
          // The terminal event was already received; cancellation has no UI impact.
        }
        break;
      }
    }

    if (!terminal) {
      const tail = decoder.decode();
      for (const raw of [...parser.push(tail), ...parser.flush()]) {
        handleRawEvent(raw);
        if (terminal) {
          break;
        }
      }
    }
  } catch (error) {
    if (isAbortError(error)) {
      throw error;
    }
    if (error instanceof AgentStreamError) {
      throw error;
    }
    throw new AgentStreamError(
      "The connection to the document service was interrupted. Please try again.",
    );
  } finally {
    reader.releaseLock();
  }

  if (!terminal) {
    throw new AgentStreamError(
      "The connection ended before the agent completed. Please try again.",
    );
  }
}
