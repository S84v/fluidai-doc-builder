import assert from "node:assert/strict";
import test from "node:test";

import {
  buildInputImprovementTip,
  groupPlaceholders,
} from "../src/placeholderGroups.ts";

function placeholder(text, reason, location) {
  return { text, reason, location };
}

test("groups occurrences with the same text and reason", () => {
  const groups = groupPlaceholders([
    placeholder("[Budget]", "No budget was supplied.", "Section A, paragraph 1"),
    placeholder("[Budget]", "No budget was supplied.", "Section B, paragraph 2"),
  ]);

  assert.equal(groups.length, 1);
  assert.equal(groups[0].text, "[Budget]");
  assert.equal(groups[0].reason, "No budget was supplied.");
  assert.equal(groups[0].count, 2);
  assert.deepEqual(groups[0].locations, [
    "Section A, paragraph 1",
    "Section B, paragraph 2",
  ]);
});

test("keeps placeholders with different reasons separate", () => {
  const groups = groupPlaceholders([
    placeholder("[Owner]", "No owner was supplied.", "Section A"),
    placeholder("[Owner]", "The owner was not confirmed.", "Section B"),
  ]);

  assert.equal(groups.length, 2);
  assert.deepEqual(
    groups.map((group) => group.reason),
    ["No owner was supplied.", "The owner was not confirmed."],
  );
});

test("counts occurrences and deduplicates identical locations", () => {
  const groups = groupPlaceholders([
    placeholder("[Budget]", "No budget was supplied.", "Section A"),
    placeholder("[Budget]", "No budget was supplied.", "Section A"),
    placeholder("[Budget]", "No budget was supplied.", "Section B"),
  ]);

  assert.equal(groups.length, 1);
  assert.equal(groups[0].count, 3);
  assert.deepEqual(groups[0].locations, ["Section A", "Section B"]);
});

test("returns no groups for an empty final placeholder list", () => {
  assert.deepEqual(groupPlaceholders([]), []);
});

test("builds one tip from the first usable missing information item", () => {
  const tip = buildInputImprovementTip({
    missingInformation: ["   ", "Budget", "Timeline"],
    clarificationQuestions: ["What is the budget?"],
  });

  assert.equal(
    tip,
    'Tip: Add the missing detail "Budget" to your next request.',
  );
});

test("truncates long missing information with an ellipsis", () => {
  const longDetail =
    "Budget amount or budget context for Ben's action item and the overall project";
  const tip = buildInputImprovementTip({
    missingInformation: [longDetail],
    clarificationQuestions: [],
  });

  assert.ok(tip !== null);
  assert.match(tip, /missing detail/);
  assert.ok(tip.includes("…"));
  assert.equal(tip.includes(longDetail), false);
});

test("falls back to a shortened clarification question", () => {
  const longQuestion =
    "What is the approved budget for the inventory rollout across all three branches?";
  const tip = buildInputImprovementTip({
    missingInformation: ["", "   "],
    clarificationQuestions: [longQuestion],
  });

  assert.ok(tip !== null);
  assert.match(tip, /clarification question/);
  assert.ok(tip.includes("…"));
  assert.equal(tip.includes(longQuestion), false);
});

test("prefers the first usable clarification question", () => {
  const tip = buildInputImprovementTip({
    missingInformation: [],
    clarificationQuestions: ["  ", "What is the budget?", "Who owns it?"],
  });

  assert.equal(
    tip,
    'Tip: Answer the clarification question "What is the budget?" to improve the next request.',
  );
});

test("omits the tip when no usable plan input exists", () => {
  assert.equal(buildInputImprovementTip(null), null);
  assert.equal(
    buildInputImprovementTip({
      missingInformation: [],
      clarificationQuestions: [],
    }),
    null,
  );
  assert.equal(
    buildInputImprovementTip({
      missingInformation: ["", "   "],
      clarificationQuestions: [" ", "\n"],
    }),
    null,
  );
});
