import assert from "node:assert/strict";
import test from "node:test";

import { cleanCompletionText, isCompletionMessage } from "../src/resultText.ts";

const FILENAME = "report-123456789abc.docx";

test("identifies the redundant completion message", () => {
  assert.equal(
    isCompletionMessage(`Document generated successfully: ${FILENAME}`),
    true,
  );
  assert.equal(
    isCompletionMessage("Review the document before sending."),
    false,
  );
  assert.equal(isCompletionMessage(""), false);
});

test("removes the redundant success message and filename", () => {
  assert.equal(
    cleanCompletionText(
      `Document generated successfully: ${FILENAME}`,
      FILENAME,
    ),
    "",
  );
});

test("removes the exact filename from a sentence", () => {
  assert.equal(
    cleanCompletionText(`Review ${FILENAME} before sending.`, FILENAME),
    "Review before sending.",
  );
});

test("keeps a filename embedded inside another word unchanged", () => {
  const embedded = `my${FILENAME}`;

  assert.equal(
    cleanCompletionText(`Review ${embedded} before sending.`, FILENAME),
    `Review ${embedded} before sending.`,
  );
  assert.equal(
    cleanCompletionText(`Keep ${FILENAME}.backup unchanged.`, FILENAME),
    `Keep ${FILENAME}.backup unchanged.`,
  );
});

test("removes every standalone occurrence of the filename", () => {
  assert.equal(
    cleanCompletionText(
      `A ${FILENAME} B ${FILENAME} C.`,
      FILENAME,
    ),
    "A B C.",
  );
});

test("removes a filename that is on its own line", () => {
  assert.equal(
    cleanCompletionText(
      `First paragraph.\n${FILENAME}\nSecond paragraph.`,
      FILENAME,
    ),
    "First paragraph.\nSecond paragraph.",
  );
});

test("preserves multiple paragraphs and line breaks", () => {
  assert.equal(
    cleanCompletionText(
      `Document generated successfully: ${FILENAME}\n\nFirst paragraph.\nStill first.\n\nSecond paragraph.`,
      FILENAME,
    ),
    "First paragraph.\nStill first.\n\nSecond paragraph.",
  );
});

test("preserves substantive text following the success prefix", () => {
  assert.equal(
    cleanCompletionText(
      `Document generated successfully: ${FILENAME}\n\nReview the document before sending.`,
      FILENAME,
    ),
    "Review the document before sending.",
  );
});

test("preserves meaningful punctuation around the removed filename", () => {
  assert.equal(
    cleanCompletionText(`See ${FILENAME}.`, FILENAME),
    "See.",
  );
});

test("preserves content when the filename is not present", () => {
  assert.equal(
    cleanCompletionText("The document is ready.", FILENAME),
    "The document is ready.",
  );
});

test("does not normalise unrelated whitespace", () => {
  assert.equal(
    cleanCompletionText("Keep  these   spaces intact.", FILENAME),
    "Keep  these   spaces intact.",
  );
});

test("returns an empty string for empty response text", () => {
  assert.equal(cleanCompletionText("", FILENAME), "");
  assert.equal(cleanCompletionText("   ", FILENAME), "");
});
