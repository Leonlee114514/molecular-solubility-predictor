import { useTranslation } from "react-i18next";

import type { ModelMode } from "@/types/api";

interface ModelSelectorProps {
  mode: ModelMode;
  onChange: (mode: ModelMode) => void;
  gnnAvailable: boolean;
}

const OPTIONS: { value: ModelMode; labelKey: string; needsGnn: boolean }[] = [
  { value: "auto", labelKey: "app.model.auto", needsGnn: true },
  { value: "rf", labelKey: "app.model.rf", needsGnn: false },
  { value: "gnn", labelKey: "app.model.gnn", needsGnn: true },
  { value: "ensemble", labelKey: "app.model.ensemble", needsGnn: true },
];

export default function ModelSelector({ mode, onChange, gnnAvailable }: ModelSelectorProps) {
  const { t } = useTranslation();
  const missingTip = t("app.model.gnn_missing");

  return (
    <div className="glass-card flex flex-col gap-3 p-4">
      <span className="text-sm font-medium text-ob-muted">{t("app.model.label")}</span>
      <div className="flex flex-wrap gap-2">
        {OPTIONS.map((opt) => {
          const disabled = opt.needsGnn && !gnnAvailable;
          const active = mode === opt.value;
          return (
            <button
              key={opt.value}
              type="button"
              disabled={disabled}
              title={disabled ? missingTip : undefined}
              onClick={() => onChange(opt.value)}
              className={`rounded-xl border px-4 py-2 text-sm transition-all ${
                active
                  ? "border-nebula bg-nebula/30 text-white shadow-glow"
                  : disabled
                    ? "cursor-not-allowed border-ob-border bg-ob-surface/40 text-ob-faint"
                    : "border-ob-border bg-ob-surface/70 text-ob-muted hover:border-nebula/60 hover:text-ob-text"
              }`}
            >
              {t(opt.labelKey)}
            </button>
          );
        })}
      </div>
      {!gnnAvailable && (
        <p className="text-xs text-ob-faint">{missingTip}</p>
      )}
    </div>
  );
}
