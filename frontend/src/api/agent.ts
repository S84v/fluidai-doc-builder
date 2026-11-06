export type AgentEvent =
  | { type: "progress"; message: string }
  | { type: "token"; text: string }
  | { type: "complete"; filename: string; url: string }
  | { type: "error"; message: string };

type RawSseEvent = {
  event: string;
  data: string;
};

type EventBoundary = {
  index: number;
  length: number;
};

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
  return typeof value === "object" && value !== null;
}

function parseAgentEvent(raw: RawSseEvent): AgentEvent | null {
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

  if (raw.event === "progress" && typeof payload.message === "string") {
    return { type: "progress", message: payload.message };
  }

  if (raw.event === "token" && typeof payload.text === "string") {
    return { type: "token", text: payload.text };
  }

  if (
    raw.event === "complete" &&
    typeof payload.filename === "string" &&
    typeof payload.url === "string"
  ) {
    return {
      type: "complete",
      filename: payload.filename,
      url: payload.url,
    };
  }

  if (raw.event === "error" && typeof payload.message === "string") {
    return { type: "error", message: payload.message };
  }

  if (
    raw.event === "progress" ||
    raw.event === "token" ||
    raw.event === "complete" ||
    raw.event === "error"
  ) {
    throw new AgentStreamError(
      "The server returned an unexpected streaming response.",
    );
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
