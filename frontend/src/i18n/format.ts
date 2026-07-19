import i18n from "@/i18n";

/**
 * Format a dumped locale string that may contain Python-style placeholders
 * (e.g. "{logS:.3f}", "{pct:.0f}%", "{name}").
 *
 * i18next interpolation can't handle the `:.<fmt>` spec, so this helper reads
 * the raw template from the active resource bundle and interpolates manually.
 * Numeric specs `:.Nf` apply toFixed(N); anything else stringifies as-is.
 */
export function tf(key: string, vars?: Record<string, unknown>): string {
  const lang = i18n.language?.startsWith("zh") ? "zh" : "en";
  let template = readBundle(lang)[key];
  if (typeof template !== "string" && lang !== "zh") {
    template = readBundle("zh")[key];
  }
  if (typeof template !== "string") return key;
  if (!vars) return template;

  return template.replace(
    /\{([^{}:]+)(?::([^{}]+))?\}/g,
    (match, rawName: string, spec: string | undefined) => {
      const value = vars[rawName.trim()];
      if (value === undefined || value === null) return match;
      if (typeof value === "number" && spec) {
        const fixed = /^\.(\d+)f$/.exec(spec);
        if (fixed) return value.toFixed(Number(fixed[1]));
      }
      return String(value);
    },
  );
}

function readBundle(lang: string): Record<string, string> {
  return (i18n.getResourceBundle(lang, "translation") ?? {}) as Record<string, string>;
}
