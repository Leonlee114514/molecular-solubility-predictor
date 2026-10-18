import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { tf } from "@/i18n/format";
import type { HistoryEntry } from "@/lib/history";

interface HistorySectionProps {
  entries: HistoryEntry[];
  onReuse: (entry: HistoryEntry) => void;
  onClear: () => void;
}

type Tone = "neutral" | "brand" | "ok" | "warn" | "danger";

/**
 * Band colour for a history entry. Verdict tones come from the resolved
 * pka_kind so they match the badges in the result tabs; entries stored before
 * pka_kind existed fall back to neutral rather than re-deriving a band from the
 * raw number, which is what used to drift away from the backend's thresholds
 * (core/chemistry.py).
 */
function logSTone(logS: number | null): Tone {
  if (logS == null) return "neutral";
  if (logS > 0) return "ok";
  if (logS > -2) return "warn";
  return "danger";
}

function pkaTone(entry: HistoryEntry): Tone {
  if (entry.pKa == null) return "neutral";
  switch (entry.result?.pka_kind) {
    case "acid":
      return "brand";
    case "base":
    case "amphoteric":
      return "warn";
    default:
      return "neutral";
  }
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

  const empty = entries.length === 0;

  // Rendered even when empty: the card keeps its 56px header so the first
  // prediction grows the page below the fold instead of shoving the viewport.
  return (
    <Card className="overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center justify-between px-5 py-4 text-left transition-colors duration-fast ease-out hover:bg-ob-sunken/60"
      >
        <span className="text-sm font-semibold text-ob-text">
          {tf("history.title", { n: entries.length })}
        </span>
        <span
          className={`text-ob-muted transition-transform duration-fast ease-out ${open ? "rotate-180" : ""}`}
          aria-hidden
        >
          ▾
        </span>
      </button>

      {open && (
        <div className="flex flex-col gap-3 border-t border-ob-rule px-5 py-4">
          {empty ? (
            <p className="text-xs text-ob-faint">{t("history.empty")}</p>
          ) : (
            <>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {entries.map((entry, i) => (
                  <div
                    key={`${entry.timestamp}-${entry.smiles}`}
                    className="flex flex-col gap-2 rounded-card border border-ob-rule bg-ob-bg/40 px-3 py-2.5"
                  >
                    <div className="flex items-center gap-2">
                      <Badge tone="neutral" className="shrink-0">
                        #{i + 1}
                      </Badge>
                      <span
                        className="min-w-0 flex-1 truncate text-sm text-ob-text"
                        title={entry.smiles}
                      >
                        {entry.name || entry.smiles}
                      </span>
                      <span className="shrink-0 text-xs text-ob-faint">
                        {relativeTime(entry.timestamp, t("history.time.just_now"))}
                      </span>
                    </div>
                    <div className="flex flex-wrap items-center gap-1.5">
                      <Badge tone={logSTone(entry.logS)}>
                        {entry.logS != null
                          ? tf("history.logS_val", { val: entry.logS })
                          : t("history.logS_unknown")}
                      </Badge>
                      <Badge tone={pkaTone(entry)}>
                        {entry.pKa != null
                          ? tf("history.pka_val", { val: entry.pKa })
                          : t("history.pka_unknown")}
                      </Badge>
                      <Badge tone="neutral">{entry.model_used}</Badge>
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => onReuse(entry)}
                        className="ml-auto px-2.5"
                      >
                        {t("history.reuse_btn")}
                      </Button>
                    </div>
                  </div>
                ))}
              </div>

              <Button
                variant={confirming ? "danger" : "secondary"}
                size="sm"
                onClick={() => {
                  if (confirming) {
                    onClear();
                    setConfirming(false);
                  } else {
                    setConfirming(true);
                  }
                }}
                className="self-end"
              >
                {confirming ? t("history.clear_confirm") : t("history.clear_btn")}
              </Button>
            </>
          )}
        </div>
      )}
    </Card>
  );
}
