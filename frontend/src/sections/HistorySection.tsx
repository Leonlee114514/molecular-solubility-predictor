import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { tf } from "@/i18n/format";
import type { HistoryEntry } from "@/lib/history";

interface HistorySectionProps {
  entries: HistoryEntry[];
  onReuse: (entry: HistoryEntry) => void;
  onClear: () => void;
}

function logSColor(logS: number | null): string {
  if (logS == null) return "#6b6b7b";
  if (logS > 0) return "#34d399";
  if (logS > -2) return "#fbbf24";
  return "#f87171";
}

function pkaColor(pka: number | null): string {
  if (pka == null) return "#6b6b7b";
  if (pka < 5) return "#a78bfa";
  if (pka > 9) return "#22d3ee";
  return "#fbbf24";
}

function relativeTime(timestamp: number, justNow: string): string {
  const minutes = Math.floor((Date.now() - timestamp) / 60000);
  if (minutes < 1) return justNow;
  if (minutes < 60) return tf("history.time.minutes_ago", { n: minutes });
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return tf("history.time.hours_ago", { n: hours });
  return tf("history.time.days_ago", { n: Math.floor(hours / 24) });
}

export default function HistorySection({ entries, onReuse, onClear }: HistorySectionProps) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [confirming, setConfirming] = useState(false);

  // Auto-cancel the clear-confirm state after 3s.
  useEffect(() => {
    if (!confirming) return;
    const id = window.setTimeout(() => setConfirming(false), 3000);
    return () => window.clearTimeout(id);
  }, [confirming]);

  if (entries.length === 0) return null;

  return (
    <section className="glass-card overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center justify-between px-5 py-4 text-left transition-colors hover:bg-nebula/10"
      >
        <span className="text-sm font-semibold text-ob-text">
          {tf("history.title", { n: entries.length })}
        </span>
        <span
          className={`text-ob-muted transition-transform ${open ? "rotate-180" : ""}`}
          aria-hidden
        >
          ▾
        </span>
      </button>

      {open && (
        <div className="flex flex-col gap-3 border-t border-ob-border px-5 py-4">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {entries.map((entry, i) => {
              const sColor = logSColor(entry.logS);
              const pColor = pkaColor(entry.pKa);
              return (
                <div
                  key={`${entry.timestamp}-${entry.smiles}`}
                  className="flex flex-col gap-2 rounded-xl border border-ob-border bg-ob-surface/40 px-3 py-2.5"
                >
                  <div className="flex items-center gap-2">
                    <span className="shrink-0 rounded-full bg-gradient-to-br from-nebula to-nebula-light px-2 py-0.5 text-[10px] font-bold text-white">
                      #{i + 1}
                    </span>
                    <span
                      className="min-w-0 flex-1 truncate text-sm text-ob-text"
                      title={entry.smiles}
                    >
                      {entry.name || entry.smiles}
                    </span>
                    <span className="shrink-0 text-[10px] text-ob-faint">
                      {relativeTime(entry.timestamp, t("history.time.just_now"))}
                    </span>
                  </div>
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span
                      className="rounded-full px-2 py-0.5 text-[10px] font-semibold tabular-nums"
                      style={{ color: sColor, background: `${sColor}26` }}
                    >
                      {entry.logS != null
                        ? tf("history.logS_val", { val: entry.logS })
                        : t("history.logS_unknown")}
                    </span>
                    <span
                      className="rounded-full px-2 py-0.5 text-[10px] font-semibold tabular-nums"
                      style={{ color: pColor, background: `${pColor}26` }}
                    >
                      {entry.pKa != null
                        ? tf("history.pka_val", { val: entry.pKa })
                        : t("history.pka_unknown")}
                    </span>
                    <span className="rounded-full bg-ob-bg/60 px-2 py-0.5 text-[10px] text-ob-muted">
                      {entry.model_used}
                    </span>
                    <button
                      type="button"
                      onClick={() => onReuse(entry)}
                      className="ml-auto rounded-lg border border-nebula/50 bg-nebula/15 px-2.5 py-0.5 text-[10px] font-medium text-nebula-light transition-colors hover:bg-nebula/30"
                    >
                      {t("history.reuse_btn")}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>

          <button
            type="button"
            onClick={() => {
              if (confirming) {
                onClear();
                setConfirming(false);
              } else {
                setConfirming(true);
              }
            }}
            className={`self-end rounded-lg border px-3 py-1.5 text-xs transition-colors ${
              confirming
                ? "border-red-500/60 bg-red-500/15 text-red-300 hover:bg-red-500/25"
                : "border-ob-border bg-ob-surface/60 text-ob-muted hover:text-ob-text"
            }`}
          >
            {confirming ? t("history.clear_confirm") : t("history.clear_btn")}
          </button>
        </div>
      )}
    </section>
  );
}
