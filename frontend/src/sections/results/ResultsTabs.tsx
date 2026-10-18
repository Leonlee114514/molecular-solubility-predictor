import { useState } from "react";
import { useTranslation } from "react-i18next";

import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
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

type Verdict = "ok" | "warn" | "danger";

const VERDICT_TEXT: Record<Verdict, string> = {
  ok: "text-ok",
  warn: "text-warn",
  danger: "text-danger",
};

function solubilityLevel(logS: number): { key: string; verdict: Verdict } {
  if (logS > 0) return { key: "result.solubility.high", verdict: "ok" };
  if (logS > -2) return { key: "result.solubility.moderate", verdict: "warn" };
  return { key: "result.solubility.poor", verdict: "danger" };
}

const MODEL_BADGE_KEYS: Record<string, string> = {
  RF: "result.solubility.badge_rf",
  GNN: "result.solubility.badge_gnn",
  Ensemble: "result.solubility.badge_ensemble",
  "Ensemble(W)": "result.solubility.badge_weighted",
};

const PKA_KIND_KEYS: Record<string, string> = {
  acid: "model.pka.type.acidic_display",
  base: "model.pka.type.basic_display",
  amphoteric: "model.pka.type.amphoteric_display",
};

/** Strip simple **bold** markdown markers from backend strings. */
function plain(s: string): string {
  return s.replace(/\*\*/g, "");
}

function OodBanner({ result }: { result: PredictionResult }) {
  const { t } = useTranslation();
  const risk = result.ood_risk;
  if (!risk || risk === "UNKNOWN") return null;

  const details = () => {
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
      <ul className="mt-1 space-y-1">
        {parts.map((p) => (
          <li key={p.label}>
            <span className="font-medium text-ob-text">{p.label}: </span>
            {p.names.map((n) => descriptorLabel(t, n)).join(" · ")}
          </li>
        ))}
      </ul>
    );
  };

  if (risk === "LOW") {
    return <Alert tone="ok">{plain(t("result.ood.low"))}</Alert>;
  }

  const isHigh = risk === "HIGH";
  const titleKey = isHigh ? "result.ood.high.title" : "result.ood.medium.title";
  const descKey = isHigh ? "result.ood.high.desc" : "result.ood.medium.desc";

  return (
    <Alert
      tone={isHigh ? "danger" : "warn"}
      title={plain(t(titleKey))}
    >
      <p>{plain(t(descKey))}</p>
      {details()}
    </Alert>
  );
}

const TAB_IDS = ["preview", "solubility", "pka", "pharmacology", "ai"] as const;

export default function ResultsTabs({ result }: ResultsTabsProps) {
  const { t } = useTranslation();
  const [tab, setTab] = useState<string>("preview");
  const analysis = useAnalysis(result);

  const level = solubilityLevel(result.logS_final);
  const badgeKey = MODEL_BADGE_KEYS[result.model_used];

  return (
    <section className="flex flex-col gap-4">
      <OodBanner result={result} />

      {/* Headline: logS + model badge + pKa chip */}
      <Card className="flex flex-wrap items-center gap-x-6 gap-y-3 p-5">
        <div>
          <p className="text-xs text-ob-faint">{t("result.solubility.metric_logs")}</p>
          <p className="flex items-baseline gap-3">
            <span
              className={`text-5xl font-bold tabular-nums ${VERDICT_TEXT[level.verdict]}`}
            >
              {result.logS_final.toFixed(3)}
            </span>
            <span className={`text-base font-medium ${VERDICT_TEXT[level.verdict]}`}>
              {t(level.key)}
            </span>
          </p>
        </div>
        {badgeKey && (
          <Badge tone="brand">{t(badgeKey)}</Badge>
        )}
        {result.pka != null && (
          <Badge tone="brand">
            <span className="font-semibold">pKa {result.pka.toFixed(2)}</span>
            {result.pka_kind && PKA_KIND_KEYS[result.pka_kind] && (
              <span>{t(PKA_KIND_KEYS[result.pka_kind])}</span>
            )}
          </Badge>
        )}
      </Card>

      <Tabs value={tab} onValueChange={setTab}>
        <TabsList className="h-auto w-full flex-wrap justify-start gap-1 rounded-card border border-ob-rule bg-ob-sunken p-1">
          {TAB_IDS.map((id) => (
            <TabsTrigger
              key={id}
              value={id}
              className="rounded-control px-3 py-1.5 text-xs text-ob-muted transition-colors duration-fast ease-out hover:bg-ob-surface/70 hover:text-ob-text data-[state=active]:bg-brand data-[state=active]:text-white sm:text-sm"
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
