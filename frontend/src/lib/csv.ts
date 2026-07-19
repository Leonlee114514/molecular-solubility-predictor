/**
 * Client-side CSV helpers for the batch upload preview.
 *
 * `sniffSmilesColumn` mirrors backend/routes.py's batch-file endpoint (and
 * app.py's batch flow) exactly so the previewed column matches what the
 * backend will auto-detect.
 */

/** Minimal CSV line splitter: handles "quoted" fields and "" escapes. */
export function splitCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"') {
        if (line[i + 1] === '"') {
          cur += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        cur += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      out.push(cur);
      cur = "";
    } else {
      cur += ch;
    }
  }
  out.push(cur);
  return out;
}

/** Parse the header row the same way the backend does. */
export function parseHeader(text: string): string[] {
  const firstLine = text.replace(/^\uFEFF/, "").split("\n", 1)[0] ?? "";
  return splitCsvLine(firstLine).map((c) => c.trim().replace(/^["']|["']$/g, ""));
}

/** Auto-detect the SMILES column index; identical rules to the backend. */
export function sniffSmilesColumn(header: string[]): number {
  const lower = header.map((h) => h.toLowerCase());
  for (const keyword of ["smiles", "smile", "smi", "canonical_smiles", "isomeric_smiles"]) {
    for (let i = 0; i < lower.length; i++) {
      if (lower[i].includes(keyword)) return i;
    }
  }
  for (let i = 0; i < lower.length; i++) {
    if (lower[i].includes("mol") || lower[i].includes("structure") || lower[i].includes("compound")) {
      return i;
    }
  }
  return 0;
}

export interface CsvPreview {
  header: string[];
  rows: string[][];
  /** Total data rows (excluding header, excluding fully-empty lines). */
  rowCount: number;
}

/** Parse the whole file (good enough for preview + column extraction). */
export function parseCsv(text: string): CsvPreview {
  const clean = text.replace(/^\uFEFF/, "").replace(/\r\n?/g, "\n");
  const lines = clean.split("\n").filter((l) => l.trim().length > 0);
  const header = lines.length > 0 ? splitCsvLine(lines[0]).map((c) => c.trim()) : [];
  const rows = lines.slice(1).map(splitCsvLine);
  return { header, rows, rowCount: rows.length };
}

/** Extract one column's values (skipping empties), like pandas dropna(). */
export function extractColumn(preview: CsvPreview, colIndex: number): string[] {
  return preview.rows
    .map((r) => (r[colIndex] ?? "").trim())
    .filter((v) => v.length > 0);
}
