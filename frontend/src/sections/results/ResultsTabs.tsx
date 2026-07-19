import { useState } from "react";
import { useTranslation } from "react-i18next";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { descriptorLabel } from "@/data/descriptors";
import AiPanel from "@/sections/results/AiPanel";
import PharmacologyPanel from "@/sections/results/PharmacologyPanel";
import PkaPanel from "@/sections/results/PkaPanel";
import PreviewPanel from "@/sections/results/PreviewPanel";
import SolubilityPanel from "@/sections/results/SolubilityPanel";
import { useAnalysis } from "@/sections/results/useAnalysis";
import type { PredictionResult } from "@/types/api";

interface ResultsTabsProps {
  result: PredictionResult;
}

function solubilityLevel(logS: number): { key: string; color: string } {
  if (logS > 0) return { key: "result.solubility.high", color: "#34d399" };
  if (logS > -2) return { key: "result.solubility.moderate", color: "#fbbf24" };
  return { key: "result.solubility.poor", color: "#f87171" };
}

const MODEL_BADGE_KEYS: Record<string, string> = {
  RF: "result.solubility.badge_rf",
  GNN: "result.solubility.badge_gnn",
  Ensemble: "result.solubility.badge_ensemble",
  "Ensemble(W)": "result.solubility.badge_weighted",
};

const MODEL_COLORS: Record<string, string> = {
  RF: "#34d399",
  GNN: "#a78bfa",
  Ensemble: "#fbbf24",
  "Ensemble(W)": "#f97316",
};

const PKA_KIND_KEYS: Record<string, string> = {
  acid: "model.pka.type.acidic_display",
  base: "model.pka.type.basic_display",
  amphoteric: "model.pka.type.amphoteric_display",
};

const PKA_KIND_COLORS: Record<string, string> = {
  acid: "#a78bfa",
  base: "#22d3ee",
  amphoteric: "#fbbf24",
};

/** Strip simple **bold** markdown markers from backend strings. */
export function plain(s: string): string {
  return s.replace(/\*\*/g, "");
}

function OodBanner({ result }: { result: PredictionResult }) {
  const { t } = useTranslation();
  const risk = result.ood_risk;
  if (!risk || risk === "UNKNOWN") return null;

  const renderDetails = () => {
    const parts: { label: string; names: string[] }[] = [];
    if (result.ood_out_of_range.length > 0) {
      parts.push({
        label: t("result.ood.out_of_range_label"),
        names: result.ood_out_of_range,
      });
    }
    if (result.ood_extreme.length > 0) {
      parts.push({ label: t("result.ood.extreme_label"), names: result.ood_extreme });
    }
    if (parts.length === 0) return null;
    return (
      <ul className="mt-2 space-y-1 text-xs opacity-90">
        {parts.map((p) => (
          <li key={p.label}>
            <span className="font-medium">{p.label}: </span>
            {p.names.map((n) => descriptorLabel(t, n)).join(" · ")}
          </li>
        ))}
      </ul>
    );
  };

  if (risk === "LOW") {
    return (
      <div className="rounded-xl border border-emerald-500/40 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-300">
        {plain(t("result.ood.low"))}
      </div>
    );
  }

  const isHigh = risk === "HIGH";
  const titleKey = isHigh ? "result.ood.high.title" : "result.ood.medium.title";
  const descKey = isHigh ? "result.ood.high.desc" : "result.ood.medium.desc";
  const cls = isHigh
    ? "border-red-500/50 bg-red-500/10 text-red-300"
    : "border-amber-500/50 bg-amber-500/10 text-amber-300";

  return (
    <div className={`rounded-xl border px-4 py-3 text-sm ${cls}`}>
      <p className="font-semibold">{plain(t(titleKey))}</p>
      <p className="mt-1 text-xs opacity-90">{plain(t(descKey))}</p>
      {renderDetails()}
    </div>
  );
}

const TAB_IDS = ["preview", "solubility", "pka", "pharmacology", "ai"] as const;

export default function ResultsTabs({ result }: ResultsTabsProps) {
  const { t } = useTranslation();
  const [tab, setTab] = useState<string>("preview");
  const analysis = useAnalysis(result);

  const level = solubilityLevel(result.logS_final);
  const badgeKey = MODEL_BADGE_KEYS[result.model_used];
  const badgeColor = MODEL_COLORS[result.model_used] ?? "#a78bfa";

  return (
    <section className="flex flex-col gap-4">
      <OodBanner result={result} />

      {/* Headline: logS + model badge + pKa chip */}
      <div className="glass-card flex flex-wrap items-center gap-x-6 gap-y-3 p-5">
        <div>
          <p className="text-xs text-ob-faint">{t("result.solubility.metric_logs")}</p>
          <p className="flex items-baseline gap-3">
            <span
              className="text-5xl font-bold tabular-nums"
              style={{ color: level.color, textShadow: `0 0 24px ${level.color}55` }}
            >
              {result.logS_final.toFixed(3)}
            </span>
            <span className="text-base font-medium" style={{ color: level.color }}>
              {t(level.key)}
            </span>
          </p>
        </div>
        {badgeKey && (
          <span
            className="rounded-full px-3 py-1 text-xs"
            style={{
              color: badgeColor,
              border: `1px solid ${badgeColor}66`,
              background: `${badgeColor}1f`,
            }}
          >
            {t(badgeKey)}
          </span>
        )}
        {result.pka != null && (
          <span className="flex items-center gap-2 rounded-full border border-cyan-500/40 bg-cyan-500/10 px-3 py-1 text-xs text-cyan-300">
            <span className="font-semibold tabular-nums">pKa {result.pka.toFixed(2)}</span>
            {result.pka_kind && PKA_KIND_KEYS[result.pka_kind] && (
              <span style={{ color: PKA_KIND_COLORS[result.pka_kind] }}>
                {t(PKA_KIND_KEYS[result.pka_kind])}
              </span>
            )}
          </span>
        )}
      </div>

      <Tabs value={tab} onValueChange={setTab}>
        <TabsList className="glass-card h-auto w-full flex-wrap justify-start gap-1 p-1.5">
          {TAB_IDS.map((id) => (
            <TabsTrigger
              key={id}
              value={id}
              className="rounded-lg px-3 py-1.5 text-xs data-[state=active]:bg-nebula data-[state=active]:text-white sm:text-sm"
            >
              {t(`result.tab.${id}`)}
            </TabsTrigger>
          ))}
        </TabsList>

        <TabsContent value="preview">
          <PreviewPanel result={result} />
        </TabsContent>
        <TabsContent value="solubility">
          <SolubilityPanel result={result} />
        </TabsContent>
        <TabsContent value="pka">
          <PkaPanel result={result} analysis={analysis} />
        </TabsContent>
        <TabsContent value="pharmacology">
          <PharmacologyPanel result={result} analysis={analysis} />
        </TabsContent>
        <TabsContent value="ai">
          <AiPanel result={result} />
        </TabsContent>
      </Tabs>
    </section>
  );
}
