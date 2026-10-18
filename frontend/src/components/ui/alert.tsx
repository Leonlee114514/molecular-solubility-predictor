import * as React from "react";

import { cn } from "@/lib/utils";

/**
 * Alert — a margin note, not a coloured box.
 *
 * A 2px left rule carries the tone; the fill stays at 5% so a page with four
 * notes still reads as paper. `danger` is announced to screen readers; the
 * other tones are status only.
 */

type AlertTone = "info" | "ok" | "warn" | "danger";

const TONE: Record<AlertTone, string> = {
  info: "border-brand/60 bg-brand/5",
  ok: "border-ok/60 bg-ok/5",
  warn: "border-warn/60 bg-warn/5",
  danger: "border-danger/60 bg-danger/5",
};

const TITLE_TONE: Record<AlertTone, string> = {
  info: "text-brand",
  ok: "text-ok",
  warn: "text-warn",
  danger: "text-danger",
};

export interface AlertProps
  extends Omit<React.HTMLAttributes<HTMLDivElement>, "title"> {
  tone?: AlertTone;
  title?: React.ReactNode;
}

export function Alert({
  tone = "info",
  title,
  children,
  className,
  ...props
}: AlertProps) {
  return (
    <div
      role={tone === "danger" ? "alert" : "status"}
      className={cn(
        "border-l-2 py-2 pl-3 pr-3 text-sm",
        TONE[tone],
        className,
      )}
      {...props}
    >
      {title != null && (
        <p className={cn("font-semibold", TITLE_TONE[tone])}>{title}</p>
      )}
      {children != null && (
        <div
          className={cn(
            "text-ob-muted",
            title != null && "mt-1 text-xs leading-relaxed",
          )}
        >
          {children}
        </div>
      )}
    </div>
  );
}
