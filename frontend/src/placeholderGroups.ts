import type { RenderedPlaceholder } from "./api/agent";

export type PlaceholderGroup = {
  text: string;
  reason: string;
  count: number;
  locations: string[];
};

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
