import assert from "node:assert/strict";
import test from "node:test";

import { groupPlaceholders } from "../src/placeholderGroups.ts";

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
