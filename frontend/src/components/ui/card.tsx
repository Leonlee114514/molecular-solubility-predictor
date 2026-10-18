import * as React from "react";

import { cn } from "@/lib/utils";

/**
 * Card — the paper surface.
 *
 * One border, one radius, no shadow, no blur. Padding is the caller's job
 * (p-4 for dense panels, p-5 for the result masthead) so the same surface can
 * carry a chart, a table or a sentence without a second component.
 */
export function Card({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-card border border-ob-rule bg-ob-surface",
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}

/** Card title row: 14px semibold ink, optional right-aligned action. */
export function CardHeader({
  title,
  action,
  className,
}: {
  title: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex items-baseline justify-between gap-3",
        className,
      )}
    >
      <h3 className="text-sm font-semibold text-ob-text">{title}</h3>
      {action}
    </div>
  );
}
