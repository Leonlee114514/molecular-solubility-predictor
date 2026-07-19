import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { api, ApiError } from "@/api/client";
import { Progress } from "@/components/ui/progress";
import { tf } from "@/i18n/format";
import { extractColumn, parseCsv, sniffSmilesColumn, type CsvPreview } from "@/lib/csv";
import type { BatchRow, ModelMode } from "@/types/api";
import { isBatchRowError } from "@/types/api";

interface BatchSectionProps {
  mode: ModelMode;
}

type Phase = "idle" | "ready" | "running" | "done" | "error";

const OOD_COLORS: Record<string, string> = {
  LOW: "#34d399",
  MEDIUM: "#fbbf24",
  HIGH: "#f87171",
  UNKNOWN: "#6b6b7b",
};

const POLL_MS = 1500;

/** Build the results CSV client-side (columns per Phase-4 spec). */
function buildResultsCsv(rows: BatchRow[]): string {
  const esc = (v: string) => (/[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);
  const lines = ["smiles,logS_final,model_used,pKa,pka_kind,ood_risk,error"];
  for (const row of rows) {
    if (isBatchRowError(row)) {
      lines.push([esc(row.smiles), "", "", "", "", "", esc(row.error)].join(","));
    } else {
      lines.push(
        [
          esc(row.smiles),
          row.logS_final.toFixed(4),
          esc(row.model_used),
          row.pka != null ? row.pka.toFixed(3) : "",
          row.pka_kind ?? "",
          row.ood_risk ?? "",
          "",
        ].join(","),
      );
    }
  }
  return lines.join("\n");
}

export default function BatchSection({ mode }: BatchSectionProps) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<CsvPreview | null>(null);
  const [detectedCol, setDetectedCol] = useState(0);
  const [colIndex, setColIndex] = useState(0);
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [rows, setRows] = useState<BatchRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<number | null>(null);

  const stopPolling = () => {
    if (pollRef.current !== null) {
      window.clearTimeout(pollRef.current);
      pollRef.current = null;
    }
  };
  useEffect(() => stopPolling, []);

  const handleFile = useCallback(
    async (f: File) => {
      stopPolling();
      setError(null);
      setRows([]);
      try {
        const text = await f.text();
        const pv = parseCsv(text);
        if (pv.header.length === 0 || pv.rowCount === 0) {
          setError(tf("app.batch.no_rows"));
          setFile(null);
          setPreview(null);
          setPhase("idle");
          return;
        }
        const detected = sniffSmilesColumn(pv.header);
        setFile(f);
        setPreview(pv);
        setDetectedCol(detected);
        setColIndex(detected);
        setPhase("ready");
      } catch (e) {
        setError(tf("app.batch.parse_error", { err: String(e) }));
        setPhase("error");
      }
    },
    [],
  );

  const start = useCallback(async () => {
    if (!file || !preview) return;
    setError(null);
    setRows([]);
    setProgress({ done: 0, total: 0 });
    try {
      // Column override honored client-side: when the user picked a different
      // column than the backend would auto-detect, extract that column and
      // use the JSON endpoint; otherwise ship the file as-is.
      const start =
        colIndex === detectedCol
          ? await api.predictBatchFile(file, mode)
          : await (() => {
              const smiles = extractColumn(preview, colIndex);
              if (smiles.length === 0) {
                throw new ApiError(400, tf("app.batch.no_rows"));
              }
              return api.predictBatchList(smiles, mode);
            })();
      setProgress({ done: 0, total: start.count });
      setPhase("running");

      const poll = async () => {
        try {
          const status = await api.batchStatus(start.task_id);
          setProgress(status.progress);
          if (status.status === "done") {
            setRows(status.results ?? []);
            setPhase("done");
            pollRef.current = null;
            return;
          }
          if (status.status === "error") {
            setError(tf("app.batch.error", { err: status.error ?? "unknown" }));
            setPhase("error");
            pollRef.current = null;
            return;
          }
        } catch (e) {
          setError(tf("app.batch.error", { err: e instanceof ApiError ? e.detail : String(e) }));
          setPhase("error");
          pollRef.current = null;
          return;
        }
        pollRef.current = window.setTimeout(() => void poll(), POLL_MS);
      };
      pollRef.current = window.setTimeout(() => void poll(), POLL_MS);
    } catch (e) {
      setError(tf("app.batch.error", { err: e instanceof ApiError ? e.detail : String(e) }));
      setPhase("error");
    }
  }, [file, preview, colIndex, detectedCol, mode]);

  const download = useCallback(() => {
    const csv = buildResultsCsv(rows);
    const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = t("app.batch.download_filename");
    a.click();
    URL.revokeObjectURL(url);
  }, [rows, t]);

  const okCount = rows.filter((r) => !isBatchRowError(r)).length;

  return (
    <section className="glass-card overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center justify-between px-5 py-4 text-left transition-colors hover:bg-nebula/10"
      >
        <span className="text-sm font-semibold text-ob-text">{t("app.batch.title")}</span>
        <span
          className={`text-ob-muted transition-transform ${open ? "rotate-180" : ""}`}
          aria-hidden
        >
          ▾
        </span>
      </button>

      {open && (
        <div className="flex flex-col gap-4 border-t border-ob-border px-5 py-4">
          <p className="text-xs text-ob-muted">{t("app.batch.desc")}</p>

          {/* Upload drop zone */}
          <button
            type="button"
            aria-label={t("app.batch.upload_label")}
            disabled={phase === "running"}
            onClick={() => fileRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const f = e.dataTransfer.files?.[0];
              if (f) void handleFile(f);
            }}
            className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-ob-border bg-ob-bg/40 px-4 py-6 text-sm text-ob-muted transition-colors hover:border-nebula/60 hover:text-ob-text disabled:opacity-50"
          >
            <span className="text-xl" aria-hidden>
              📊
            </span>
            <span>{file ? file.name : t("app.batch.upload_label")}</span>
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void handleFile(f);
              e.target.value = "";
            }}
          />

          {/* Preview + column override */}
          {phase !== "idle" && preview && (
            <div className="flex flex-col gap-3">
              <p className="text-xs text-ob-muted">
                {tf("app.batch.file_info", {
                  name: file?.name ?? "",
                  rows: preview.rowCount,
                  col: preview.header[colIndex] ?? "",
                }).replace(/\*\*|`/g, "")}
              </p>
              <label className="flex items-center gap-2 text-xs text-ob-muted">
                {t("app.batch.column_label")}
                <select
                  value={colIndex}
                  disabled={phase === "running"}
                  onChange={(e) => setColIndex(Number(e.target.value))}
                  className="rounded-lg border border-ob-border bg-ob-bg/60 px-2 py-1.5 text-sm text-ob-text outline-none focus:border-nebula"
                >
                  {preview.header.map((h, i) => (
                    <option key={`${h}-${i}`} value={i}>
                      {h || `#${i + 1}`}
                    </option>
                  ))}
                </select>
              </label>
              <div className="overflow-x-auto rounded-lg border border-ob-border">
                <table className="w-full text-left text-xs">
                  <caption className="px-3 py-1.5 text-left text-ob-faint">
                    {t("app.batch.preview_rows")}
                  </caption>
                  <thead>
                    <tr className="border-b border-ob-border bg-ob-surface/60">
                      {preview.header.map((h, i) => (
                        <th
                          key={`${h}-${i}`}
                          className={`px-3 py-1.5 font-medium ${
                            i === colIndex ? "text-nebula-light" : "text-ob-muted"
                          }`}
                        >
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.rows.slice(0, 5).map((r, ri) => (
                      <tr key={ri} className="border-b border-ob-border/50 last:border-0">
                        {preview.header.map((_, ci) => (
                          <td
                            key={ci}
                            className={`max-w-[220px] truncate px-3 py-1.5 font-mono ${
                              ci === colIndex ? "text-nebula-light" : "text-ob-text"
                            }`}
                          >
                            {r[ci] ?? ""}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <button
                type="button"
                disabled={phase === "running"}
                onClick={() => void start()}
                className="self-start rounded-xl bg-nebula px-6 py-2.5 text-sm font-semibold text-white shadow-glow transition-opacity hover:opacity-90 disabled:opacity-50"
              >
                {t("app.batch.start_btn")}
              </button>
            </div>
          )}

          {/* Progress */}
          {phase === "running" && (
            <div className="flex flex-col gap-2">
              <p className="text-xs text-ob-muted">
                {tf("app.batch.running", { done: progress.done, total: progress.total })}
              </p>
              <Progress
                value={progress.total > 0 ? (progress.done / progress.total) * 100 : 0}
                className="h-2"
              />
            </div>
          )}

          {/* Error */}
          {error && <p className="text-sm text-red-400">{error}</p>}

          {/* Results */}
          {phase === "done" && (
            <div className="flex flex-col gap-3">
              <div className="flex flex-wrap items-center gap-3">
                <p className="text-sm text-emerald-300">
                  {tf("app.batch.complete", { n: rows.length })}
                  <span className="ml-2 text-xs text-ob-muted">
                    {tf("app.batch.rows_ok", { ok: okCount, total: rows.length })}
                  </span>
                </p>
                <button
                  type="button"
                  onClick={download}
                  className="rounded-lg border border-cyan-500/50 bg-cyan-500/10 px-4 py-1.5 text-xs text-cyan-300 transition-colors hover:bg-cyan-500/20"
                >
                  {t("app.batch.download_btn")}
                </button>
              </div>
              <div className="max-h-96 overflow-auto rounded-lg border border-ob-border">
                <table className="w-full text-left text-xs">
                  <thead className="sticky top-0 bg-ob-surface">
                    <tr className="border-b border-ob-border">
                      <th className="px-3 py-2 font-medium text-ob-muted">
                        {t("app.batch.table.index")}
                      </th>
                      <th className="px-3 py-2 font-medium text-ob-muted">
                        {t("app.batch.table.smiles")}
                      </th>
                      <th className="px-3 py-2 font-medium text-ob-muted">
                        {t("app.batch.table.logs")}
                      </th>
                      <th className="px-3 py-2 font-medium text-ob-muted">
                        {t("app.batch.table.model")}
                      </th>
                      <th className="px-3 py-2 font-medium text-ob-muted">
                        {t("app.batch.table.pka")}
                      </th>
                      <th className="px-3 py-2 font-medium text-ob-muted">
                        {t("app.batch.table.ood")}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row, i) =>
                      isBatchRowError(row) ? (
                        <tr key={i} className="border-b border-ob-border/50 bg-red-500/5">
                          <td className="px-3 py-1.5 text-ob-faint">{i + 1}</td>
                          <td className="max-w-[240px] truncate px-3 py-1.5 font-mono text-red-300">
                            {row.smiles}
                          </td>
                          <td colSpan={4} className="px-3 py-1.5 text-red-400">
                            {t("app.batch.table.error")}: {row.error}
                          </td>
                        </tr>
                      ) : (
                        <tr key={i} className="border-b border-ob-border/50 last:border-0">
                          <td className="px-3 py-1.5 text-ob-faint">{i + 1}</td>
                          <td className="max-w-[240px] truncate px-3 py-1.5 font-mono text-ob-text">
                            {row.smiles}
                          </td>
                          <td className="px-3 py-1.5 tabular-nums text-nebula-light">
                            {row.logS_final.toFixed(3)}
                          </td>
                          <td className="px-3 py-1.5 text-ob-muted">{row.model_used}</td>
                          <td className="px-3 py-1.5 tabular-nums text-cyan-300">
                            {row.pka != null ? row.pka.toFixed(2) : "—"}
                          </td>
                          <td className="px-3 py-1.5">
                            {row.ood_risk ? (
                              <span
                                className="rounded-full px-2 py-0.5 text-[10px] font-semibold"
                                style={{
                                  color: OOD_COLORS[row.ood_risk] ?? OOD_COLORS.UNKNOWN,
                                  background: `${OOD_COLORS[row.ood_risk] ?? OOD_COLORS.UNKNOWN}22`,
                                }}
                              >
                                {row.ood_risk}
                              </span>
                            ) : (
                              <span className="text-ob-faint">—</span>
                            )}
                          </td>
                        </tr>
                      ),
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
