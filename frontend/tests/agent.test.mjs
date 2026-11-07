import assert from "node:assert/strict";
import test from "node:test";

import {
  SseParser,
  parseAgentEvent,
} from "../src/api/agent.ts";

function decode(chunks) {
  const parser = new SseParser();
  const raw = chunks.flatMap((chunk) => parser.push(chunk));
  return raw.map(parseAgentEvent).filter((event) => event !== null);
}

const PLAN = {
  document_type: "project_plan",
  title: "Inventory Rollout",
  objective: "Plan the rollout.",
  audience: null,
  document_date: "2026-11-07",
  document_version: "1.2",
  provided_facts: ["Three branches are in scope."],
  tasks: [
    {
      id: "analyse_constraints",
      title: "Analyse constraints",
      description: "Compare the rollout constraints.",
      stage: "planning",
      optional: false,
      status: "completed",
    },
    {
      id: "draft_plan",
      title: "Draft plan",
      description: "Write the project plan.",
      stage: "document_writing",
      optional: false,
      status: "pending",
    },
  ],
  conflicts: [
    {
      requirements: ["Four-week rollout", "Three weeks of cleanup"],
      impact: "The dates conflict.",
      resolution: "Use a phased rollout.",
    },
  ],
  missing_information: ["Budget"],
  assumptions: ["A pilot is acceptable."],
  clarification_questions: ["What is the budget?"],
  placeholders: [
    {
      text: "[Budget]",
      section: "Investment",
      reason: "No budget was supplied.",
    },
  ],
};

function sse(event, data) {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

test("parses plan, progress, placeholders and complete events across chunks", () => {
  const payload = sse("plan", PLAN);
  const splitAt = Math.floor(payload.length / 2);
  const chunks = [
    payload.slice(0, splitAt),
    payload.slice(splitAt),
    sse("progress", {
      stage: "document_writing",
      task_ids: ["draft_plan"],
      status: "active",
      message: "Writing document...",
    }),
    sse("placeholders", {
      placeholders: [
        {
          text: "[Budget]",
          location: "Table 1, row 3, column 2",
          reason: "No budget was supplied.",
        },
      ],
    }),
    sse("complete", {
      filename: "project-plan-123456789abc.docx",
      url: "/documents/project-plan-123456789abc.docx",
    }),
  ];

  const events = decode(chunks);

  assert.equal(events.length, 4);
  assert.equal(events[0].type, "plan");
  assert.equal(events[0].plan.documentType, "project_plan");
  assert.equal(events[0].plan.documentDate, "2026-11-07");
  assert.equal(events[0].plan.tasks[0].status, "completed");
  assert.equal(events[0].plan.conflicts[0].resolution, "Use a phased rollout.");
  assert.equal(events[0].plan.placeholders[0].text, "[Budget]");

  assert.deepEqual(events[1], {
    type: "progress",
    stage: "document_writing",
    taskIds: ["draft_plan"],
    status: "active",
    message: "Writing document...",
  });
  assert.equal(events[2].type, "placeholders");
  assert.equal(events[2].placeholders[0].location, "Table 1, row 3, column 2");
  assert.equal(events[3].type, "complete");
});

test("ignores unknown SSE events", () => {
  assert.deepEqual(decode([sse("heartbeat", { ok: true })]), []);
});

test("rejects malformed known event payloads", () => {
  assert.throws(
    () => decode([sse("progress", { stage: "writing" })]),
    /unexpected streaming response|unexpected task status/i,
  );
  assert.throws(
    () => decode([sse("plan", { title: "Missing fields" })]),
    /unexpected streaming response|unexpected execution plan/i,
  );
});

test("supports multiple tasks sharing one stage update", () => {
  const [event] = decode([
    sse("progress", {
      stage: "document_writing",
      task_ids: ["draft_summary", "draft_findings"],
      status: "completed",
      message: "Document written.",
    }),
  ]);

  assert.equal(event.type, "progress");
  assert.deepEqual(event.taskIds, ["draft_summary", "draft_findings"]);
  assert.equal(event.status, "completed");
});

test("parses effective document metadata", () => {
  const [event] = decode([
    sse("metadata", {
      document_date: "2026-11-07",
      document_version: "1.2",
    }),
  ]);

  assert.deepEqual(event, {
    type: "metadata",
    documentDate: "2026-11-07",
    documentVersion: "1.2",
  });
});

test("rejects missing or empty metadata fields", () => {
  assert.throws(
    () => decode([sse("metadata", { document_version: "1.0" })]),
    /unexpected streaming response/i,
  );
  assert.throws(
    () =>
      decode([
        sse("metadata", {
          document_date: "2026-11-07",
          document_version: "   ",
        }),
      ]),
    /unexpected streaming response/i,
  );
});

test("rejects incorrectly typed metadata fields", () => {
  assert.throws(
    () =>
      decode([
        sse("metadata", {
          document_date: 20261107,
          document_version: "1.0",
        }),
      ]),
    /unexpected streaming response/i,
  );
  assert.throws(
    () =>
      decode([
        sse("metadata", {
          document_date: "2026-11-07",
          document_version: null,
        }),
      ]),
    /unexpected streaming response/i,
  );
});
