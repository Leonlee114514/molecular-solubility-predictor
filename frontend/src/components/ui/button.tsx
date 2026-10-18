import * as React from "react";

import { cn } from "@/lib/utils";

/**
 * Button — the only control primitive in the app.
 *
 * Four variants, four sizes, one state model. States are exhaustive on
 * purpose: hover darkens (never fades), active sinks 1px, focus draws a 2px
 * brand ring, disabled drops to 45% and stops hover/active, loading swaps in a
 * spinner and sets aria-busy while keeping the label readable.
 */

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
type ButtonSize = "sm" | "md" | "lg" | "icon";

const BASE =
  "inline-flex shrink-0 items-center justify-center gap-2 whitespace-nowrap rounded-control font-medium " +
  "transition-colors duration-fast ease-out " +
  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/35 focus-visible:ring-offset-2 focus-visible:ring-offset-ob-bg " +
  "disabled:pointer-events-none disabled:opacity-45 " +
  "active:translate-y-px";

const VARIANT: Record<ButtonVariant, string> = {
  primary: "bg-brand text-white shadow-hairline hover:bg-brand-strong",
  secondary:
    "border border-ob-rule bg-ob-surface text-ob-muted hover:border-brand/45 hover:text-ob-text",
  ghost: "text-ob-muted hover:bg-ob-sunken hover:text-ob-text",
  danger: "border border-danger/40 bg-danger/5 text-danger hover:bg-danger/10",
};

const SIZE: Record<ButtonSize, string> = {
  sm: "h-8 px-3 text-xs",
  md: "h-9 px-4 text-sm",
  lg: "h-11 px-6 text-sm font-semibold",
  icon: "h-9 w-9",
};

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  function Button(
    {
      className,
      variant = "primary",
      size = "md",
      loading = false,
      disabled,
      type = "button",
      children,
      ...props
    },
    ref,
  ) {
    return (
      <button
        ref={ref}
        type={type}
        disabled={disabled || loading}
        aria-busy={loading || undefined}
        className={cn(BASE, VARIANT[variant], SIZE[size], className)}
        {...props}
      >
        {loading && (
          <span
            aria-hidden="true"
            className="size-3.5 animate-spin rounded-full border-2 border-current border-t-transparent"
          />
        )}
        {children}
      </button>
    );
  },
);

/** Inline text link. Kept as a component so underline/hover can never drift. */
export function TextLink({
  className,
  children,
  ...props
}: React.AnchorHTMLAttributes<HTMLAnchorElement>) {
  return (
    <a
      className={cn(
        "rounded-xs text-brand-soft underline decoration-brand/30 underline-offset-2",
        "transition-colors duration-fast ease-out hover:text-brand hover:decoration-brand/60",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/35",
        className,
      )}
      {...props}
    >
      {children}
    </a>
  );
}
