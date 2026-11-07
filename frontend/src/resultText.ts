const SUCCESS_PREFIX = /^Document generated successfully:?\s*/i;
const VALIDATION_NOTES_LINE = /^[ \t]*Validation notes:.*$/gm;
const FILENAME_BOUNDARY_START = "[^\\w./\\\\-]";
const FILENAME_BOUNDARY_END = "(?![\\w/\\\\-]|\\.[\\w/\\\\-])";

export function isCompletionMessage(finalText: string): boolean {
  return SUCCESS_PREFIX.test(finalText.trimStart());
}

export function cleanCompletionText(
  finalText: string,
  filename: string,
): string {
  const text = finalText.trim();
  const name = filename.trim();

  if (text.length === 0) {
    return "";
  }
  if (name.length === 0) {
    return text;
  }

  if (text === `Document generated successfully: ${name}`) {
    return "";
  }

  let cleaned = text.replace(SUCCESS_PREFIX, "").trimStart();
  cleaned = removeStandaloneFilename(cleaned, name);
  cleaned = removeValidationNotes(cleaned).trim();

  if (cleaned.replace(/[\s:;,.–—-]/g, "").length === 0) {
    return "";
  }
  return cleaned;
}

function removeValidationNotes(text: string): string {
  if (!/^[ \t]*Validation notes:/m.test(text)) {
    return text;
  }

  // Validation notes remain in the streamed/backend response for diagnostics,
  // but the completed download result should show only the document outcome.
  return text.replace(VALIDATION_NOTES_LINE, "").replace(/\n{3,}/g, "\n\n");
}

function removeStandaloneFilename(text: string, name: string): string {
  const escapedName = escapeRegExp(name);
  const filenameLine = new RegExp(
    `^[ \\t]*${escapedName}[ \\t]*(?:\\r?\\n|$)`,
    "gm",
  );
  let cleaned = text.replace(filenameLine, "");

  const inlineFilename = new RegExp(
    `(^|${FILENAME_BOUNDARY_START})[ \\t]*${escapedName}${FILENAME_BOUNDARY_END}[ \\t]*`,
    "g",
  );

  // Repeat so adjacent standalone filenames are also handled.
  let previous: string;
  do {
    previous = cleaned;
    cleaned = cleaned.replace(
      inlineFilename,
      (match, boundary: string, offset: number, whole: string) => {
        const next = whole[offset + match.length] ?? "";
        if (boundary === "\n" || boundary === "\r") {
          return boundary;
        }
        if (boundary === "") {
          return "";
        }
        if (/[ \t]/.test(boundary)) {
          const nextEndsClause =
            next === "" || /[\s.,;:!?)\]}]/.test(next);
          return nextEndsClause ? "" : boundary;
        }
        return boundary;
      },
    );
  } while (cleaned !== previous);

  return cleaned;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
