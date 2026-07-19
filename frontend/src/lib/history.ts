import type { PredictionResult } from "@/types/api";

/**
 * Prediction history persisted in localStorage (replaces the old Streamlit
 * repo-file approach). Stores the last 15 successful predictions, most recent
 * first. The full PredictionResult JSON is cached alongside each entry so a
 * history item can be "reused" (re-render the result tabs) without a backend
 * round-trip. Quota errors are handled by first dropping bulky SHAP arrays,
 * then dropping oldest entries.
 */

export interface HistoryEntry {
  smiles: string;
  /** Display name when the molecule came from quick-select/search, else "". */
  name: string;
  logS: number | null;
  pKa: number | null;
  model_used: string;
  /** Unix epoch milliseconds. */
  timestamp: number;
  result: PredictionResult;
}

const STORAGE_KEY = "dissolve_history_v1";
const MAX_ENTRIES = 15;

export function loadHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (e): e is HistoryEntry =>
        typeof e === "object" && e !== null && typeof (e as HistoryEntry).smiles === "string",
    );
  } catch {
    return [];
  }
}

function persist(entries: HistoryEntry[]): boolean {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
    return true;
  } catch {
    return false;
  }
}

export function pushHistory(result: PredictionResult, name: string): HistoryEntry[] {
  const entry: HistoryEntry = {
    smiles: result.smiles,
    name,
    logS: result.logS_final,
    pKa: result.pka,
    model_used: result.model_used,
    timestamp: Date.now(),
    result,
  };
  // De-dupe: a re-predicted molecule moves to the top instead of duplicating.
  const entries = [entry, ...loadHistory().filter((e) => e.smiles !== result.smiles)].slice(
    0,
    MAX_ENTRIES,
  );

  if (persist(entries)) return entries;

  // Quota guard 1: strip bulky SHAP arrays from cached results.
  const slim = entries.map((e) => ({
    ...e,
    result: { ...e.result, shap_values: null, shap_names: null },
  }));
  if (persist(slim)) return slim;

  // Quota guard 2: drop oldest until it fits (or only the new entry remains).
  let trimmed = slim.slice(0, -1);
  while (trimmed.length > 1 && !persist(trimmed)) {
    trimmed = trimmed.slice(0, -1);
  }
  return trimmed;
}

export function clearHistory(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* ignore */
  }
}
