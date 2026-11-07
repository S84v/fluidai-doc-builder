import type {
  ExecutionPlan,
  RenderedPlaceholder,
} from "./api/agent";

export type PlaceholderGroup = {
  text: string;
  reason: string;
  count: number;
  locations: string[];
};

const MAX_TIP_INPUT_LENGTH = 48;

export function groupPlaceholders(
  placeholders: RenderedPlaceholder[],
): PlaceholderGroup[] {
  const groups = new Map<string, PlaceholderGroup>();

  for (const placeholder of placeholders) {
    const key = JSON.stringify([placeholder.text, placeholder.reason]);
    const existing = groups.get(key);
    if (existing === undefined) {
      groups.set(key, {
        text: placeholder.text,
        reason: placeholder.reason,
        count: 1,
        locations: [placeholder.location],
      });
      continue;
    }

    existing.count += 1;
    if (!existing.locations.includes(placeholder.location)) {
      existing.locations.push(placeholder.location);
    }
  }

  return [...groups.values()];
}

export function buildInputImprovementTip(
  plan: ExecutionPlan | null,
): string | null {
  if (plan === null) {
    return null;
  }

  const missingDetail = firstUsableItem(plan.missingInformation);
  if (missingDetail !== null) {
    return `Tip: Add the missing detail "${shortenForTip(missingDetail)}" to your next request.`;
  }

  const clarificationQuestion = firstUsableItem(plan.clarificationQuestions);
  if (clarificationQuestion !== null) {
    return `Tip: Answer the clarification question "${shortenForTip(clarificationQuestion)}" to improve the next request.`;
  }

  return null;
}

function firstUsableItem(items: string[]): string | null {
  for (const item of items) {
    const normalized = item.trim().replace(/\s+/g, " ");
    if (normalized.length > 0) {
      return normalized;
    }
  }
  return null;
}

function shortenForTip(value: string): string {
  if (value.length <= MAX_TIP_INPUT_LENGTH) {
    return value;
  }
  return `${value.slice(0, MAX_TIP_INPUT_LENGTH - 1).trimEnd()}…`;
}
