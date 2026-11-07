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

function clonePlan() {
  return JSON.parse(JSON.stringify(PLAN));
}

const PLAN_ERROR =
  /unexpected streaming response|unexpected execution task|unexpected requirement conflict|unexpected planned placeholder|unexpected task status/i;

function assertPlanRejected(mutator, message) {
  const payload = clonePlan();
  mutator(payload);
  assert.throws(() => decode([sse("plan", payload)]), PLAN_ERROR, message);
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
  assert.equal(events[0].plan.tasks[0].optional, false);
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

test("ignores unknown events with valid JSON", () => {
  assert.deepEqual(decode([sse("heartbeat", { ok: true })]), []);
});

test("ignores unknown events without parsing invalid JSON", () => {
  assert.deepEqual(
    decode(["event: heartbeat\ndata: not-json\n\n"]),
    [],
  );
});

test("rejects malformed JSON in known events", () => {
  assert.throws(
    () => decode(["event: progress\ndata: not-json\n\n"]),
    /malformed streaming data/i,
  );
});

test("rejects known events with non-object payloads", () => {
  for (const payload of [null, [], "text", 42, true]) {
    assert.throws(
      () => decode([sse("token", payload)]),
      /unexpected streaming response/i,
    );
  }
});

test("rejects missing required plan arrays", () => {
  for (const key of [
    "provided_facts",
    "tasks",
    "conflicts",
    "missing_information",
    "assumptions",
    "clarification_questions",
    "placeholders",
  ]) {
    assertPlanRejected((plan) => {
      delete plan[key];
    }, `missing ${key}`);
  }
});

test("rejects incorrectly typed plan arrays", () => {
  const cases = [
    ["provided_facts", "facts"],
    ["provided_facts", [1]],
    ["tasks", {}],
    ["conflicts", "conflicts"],
    ["missing_information", [null]],
    ["assumptions", {}],
    ["clarification_questions", [""]],
    ["placeholders", "placeholders"],
  ];

  for (const [key, value] of cases) {
    assertPlanRejected((plan) => {
      plan[key] = value;
    }, `invalid ${key}`);
  }
});

test("rejects incorrectly typed optional task values", () => {
  for (const optional of ["false", 0, null]) {
    assertPlanRejected((plan) => {
      plan.tasks[0].optional = optional;
    }, `invalid optional ${String(optional)}`);
  }

  assertPlanRejected((plan) => {
    delete plan.tasks[0].optional;
  }, "missing optional");
});

test("rejects missing nullable plan and conflict fields", () => {
  for (const key of ["audience", "document_date", "document_version"]) {
    assertPlanRejected((plan) => {
      delete plan[key];
    }, `missing ${key}`);
  }

  assertPlanRejected((plan) => {
    delete plan.conflicts[0].resolution;
  }, "missing resolution");
});

test("accepts explicit null nullable fields", () => {
  const payload = clonePlan();
  payload.audience = null;
  payload.document_date = null;
  payload.document_version = null;
  payload.conflicts[0].resolution = null;

  const [event] = decode([sse("plan", payload)]);

  assert.equal(event.type, "plan");
  assert.equal(event.plan.audience, null);
  assert.equal(event.plan.documentDate, null);
  assert.equal(event.plan.documentVersion, null);
  assert.equal(event.plan.conflicts[0].resolution, null);
});

test("rejects invalid nested task fields", () => {
  const cases = [
    (task) => {
      task.id = "";
    },
    (task) => {
      task.id = 1;
    },
    (task) => {
      delete task.title;
    },
    (task) => {
      task.description = "";
    },
    (task) => {
      task.stage = "";
    },
    (task) => {
      task.status = "unknown";
    },
  ];

  for (const mutate of cases) {
    assertPlanRejected((plan) => mutate(plan.tasks[0]));
  }
});

test("rejects invalid conflict fields", () => {
  const cases = [
    (conflict) => {
      conflict.requirements = "not-array";
    },
    (conflict) => {
      conflict.requirements = [1];
    },
    (conflict) => {
      conflict.requirements = [""];
    },
    (conflict) => {
      conflict.impact = "";
    },
    (conflict) => {
      conflict.resolution = 5;
    },
    (conflict) => {
      delete conflict.resolution;
    },
  ];

  for (const mutate of cases) {
    assertPlanRejected((plan) => mutate(plan.conflicts[0]));
  }
});

test("rejects invalid planned placeholder fields", () => {
  const cases = [
    (placeholder) => {
      placeholder.text = "";
    },
    (placeholder) => {
      placeholder.text = 1;
    },
    (placeholder) => {
      delete placeholder.section;
    },
    (placeholder) => {
      placeholder.reason = "";
    },
  ];

  for (const mutate of cases) {
    assertPlanRejected((plan) => mutate(plan.placeholders[0]));
  }
});

test("rejects invalid rendered placeholder fields", () => {
  const valid = {
    text: "[Budget]",
    location: "Table 1",
    reason: "No budget was supplied.",
  };
  const cases = [
    [{ ...valid, text: "" }],
    [{ ...valid, text: 1 }],
    [{ ...valid, location: "" }],
    [{ ...valid, reason: "" }],
    [null],
    ["placeholder"],
  ];

  for (const placeholders of cases) {
    assert.throws(
      () => decode([sse("placeholders", { placeholders })]),
      /unexpected streaming response|unexpected rendered placeholder/i,
    );
  }
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

test("parses token, complete and error events", () => {
  const events = decode([
    sse("token", { text: "Done." }),
    sse("complete", {
      filename: "report.docx",
      url: "/documents/report.docx",
    }),
    sse("error", { message: "Nope." }),
  ]);

  assert.deepEqual(events, [
    { type: "token", text: "Done." },
    {
      type: "complete",
      filename: "report.docx",
      url: "/documents/report.docx",
    },
    { type: "error", message: "Nope." },
  ]);
});

test("rejects missing or empty token, complete and error fields", () => {
  const cases = [
    ["token", {}],
    ["token", { text: "" }],
    ["complete", { filename: "report.docx", url: "" }],
    ["complete", { filename: "", url: "/documents/report.docx" }],
    ["error", { message: "  " }],
  ];

  for (const [event, payload] of cases) {
    assert.throws(
      () => decode([sse(event, payload)]),
      /unexpected streaming response/i,
    );
  }
});

test("rejects invalid progress payload fields", () => {
  const valid = {
    stage: "document_writing",
    task_ids: ["draft_plan"],
    status: "active",
    message: "Writing document...",
  };
  const cases = [
    { ...valid, stage: "" },
    { ...valid, task_ids: "draft_plan" },
    { ...valid, task_ids: [""] },
    { ...valid, status: "unknown" },
    { ...valid, message: "" },
  ];

  for (const payload of cases) {
    assert.throws(
      () => decode([sse("progress", payload)]),
      /unexpected streaming response|unexpected task status/i,
    );
  }
});
