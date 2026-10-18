import * as React from "react";

import { cn } from "@/lib/utils";

/**
 * Badge — a printed label.
 *
 * Square-ish (2px), 12px, tabular numerals. Tone carries meaning: `brand` for
 * identity (model, pKa kind), `ok`/`warn`/`danger` for a verdict about the
 * prediction, `neutral` for plain metadata.
 */

type BadgeTone = "neutral" | "brand" | "ok" | "warn" | "danger";

const TONE: Record<BadgeTone, string> = {
  neutral: "bg-ob-sunken text-ob-muted",
  brand: "bg-brand/10 text-brand",
  ok: "bg-ok/10 text-ok",
  warn: "bg-warn/10 text-warn",
  danger: "bg-danger/10 text-danger",
};

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  tone?: BadgeTone;
}

export function Badge({ tone = "neutral", className, children, ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-sm px-2 py-0.5 text-xs font-medium tabular-nums",
        TONE[tone],
        className,
      )}
      {...props}
    >
      {children}
    </span>
  );
}
