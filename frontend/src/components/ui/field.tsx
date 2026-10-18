import * as React from "react";

import { cn } from "@/lib/utils";

/**
 * Field + Input — the only text-input primitive.
 *
 * Field owns the label / hint / error wiring (id, aria-describedby,
 * aria-invalid, role="alert"), so call sites cannot forget it. Input owns the
 * single visual state model: hairline border, brand border + ring on focus,
 * sunken surface when disabled, danger border when invalid.
 */

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  /** Renders the danger border/ring. Normally set automatically by Field. */
  invalid?: boolean;
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  function Input({ className, invalid = false, ...props }, ref) {
    return (
      <input
        ref={ref}
        aria-invalid={invalid || undefined}
        className={cn(
          "h-9 w-full rounded-control border bg-ob-surface px-3 text-sm text-ob-text",
          "placeholder:text-ob-faint",
          "transition-colors duration-fast ease-out",
          "focus-visible:outline-none focus-visible:ring-2",
          "disabled:cursor-not-allowed disabled:bg-ob-sunken disabled:text-ob-muted",
          invalid
            ? "border-danger/60 focus-visible:border-danger focus-visible:ring-danger/25"
            : "border-ob-rule focus-visible:border-brand focus-visible:ring-brand/25",
          className,
        )}
        {...props}
      />
    );
  },
);

type FieldControlProps = {
  id?: string;
  "aria-describedby"?: string;
  "aria-invalid"?: boolean;
};

export interface FieldProps {
  label: React.ReactNode;
  hint?: React.ReactNode;
  error?: string | null;
  htmlFor?: string;
  className?: string;
  /** A single Input (or any element that accepts id / aria-*). */
  children: React.ReactElement<FieldControlProps>;
}

export function Field({
  label,
  hint,
  error,
  htmlFor,
  className,
  children,
}: FieldProps) {
  const autoId = React.useId();
  const fieldId = htmlFor ?? autoId;
  const controlId = children.props.id ?? fieldId;
  const hintId = hint ? `${fieldId}-hint` : undefined;
  const errorId = error ? `${fieldId}-error` : undefined;
  const describedBy =
    [error ? errorId : null, hint ? hintId : null].filter(Boolean).join(" ") ||
    undefined;

  const control = React.cloneElement(children, {
    id: controlId,
    "aria-describedby": describedBy,
    "aria-invalid": Boolean(error),
  });

  return (
    <div className={cn("flex flex-col gap-1.5", className)}>
      <label htmlFor={controlId} className="text-xs text-ob-muted">
        {label}
      </label>
      {control}
      {error ? (
        <p id={errorId} role="alert" className="text-xs text-danger">
          {error}
        </p>
      ) : hint ? (
        <p id={hintId} className="text-xs text-ob-faint">
          {hint}
        </p>
      ) : null}
    </div>
  );
}
