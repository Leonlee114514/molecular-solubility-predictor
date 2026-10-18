import { useMemo } from "react";
import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import {
  Bar,
  BarChart,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import GlossaryText from "@/components/GlossaryText";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { tf } from "@/i18n/format";
import type { AnalysisResponse, DruglikenessResult, PredictionResult } from "@/types/api";

interface PharmacologyPanelProps {
  result: PredictionResult;
  analysis: {
    data: AnalysisResponse | null;
    error: string | null;
    loading: boolean;
  };
}

/* ---------- Lipinski ---------- */

const LIPINSKI_ROWS: { i18nKey: string; threshold: string }[] = [
  { i18nKey: "result.pharma.lipinski_prop_mw", threshold: "≤ 500 Da" },
  { i18nKey: "result.pharma.lipinski_prop_logp", threshold: "≤ 5" },
  { i18nKey: "result.pharma.lipinski_prop_hbd", threshold: "≤ 5" },
  { i18nKey: "result.pharma.lipinski_prop_hba", threshold: "≤ 10" },
  { i18nKey: "result.pharma.lipinski_prop_rotb", threshold: "≤ 10" },
];

function LipinskiSection({ analysis }: { analysis: AnalysisResponse }) {
  const { t } = useTranslation();
  const lip = analysis.lipinski;
  const passedTone = lip.passed >= 4 ? "text-ok" : "text-warn";

  return (
    <div className="glass-card flex flex-col gap-3 p-4">
      <h3 className="text-sm font-semibold text-ob-text">
        {t("result.pharma.lipinski_title")}
      </h3>
      <p className={`text-sm font-semibold ${passedTone}`}>
        <GlossaryText
          text={tf("result.pharma.lipinski_score_title", {
            score: String(lip.passed),
            total: "5",
            text: lip.interpretation,
          })}
          glossary={false}
        />
      </p>
      <div className="flex flex-col gap-2">
        {lip.rules.map((rule, i) => {
          const [, , passed, actual] = rule;
          const meta = LIPINSKI_ROWS[i];
          const value = Number.parseFloat(actual);
          const limit = Number.parseFloat(meta.threshold.replace(/[^\d.]/g, ""));
          // The bar shows how close the value is to its own limit, not whether
          // the rule passed — the old fixed 100% made every row look perfect.
          const ratio =
            Number.isFinite(value) && Number.isFinite(limit) && limit > 0
              ? Math.min(1, value / limit)
              : 0;
          return (
            <div
              key={meta.i18nKey}
              className="rounded-card border border-ob-rule bg-ob-bg/40 px-3 py-2"
            >
              <div className="flex items-baseline justify-between gap-3">
                <span className="min-w-0 text-xs text-ob-muted">
                  {t(meta.i18nKey)}
                </span>
                <span
                  className={`shrink-0 text-xs font-semibold tabular-nums ${
                    passed ? "text-ok" : "text-danger"
                  }`}
                >
                  {actual}{" "}
                  {passed
                    ? t("result.pharma.lipinski_pass")
                    : t("result.pharma.lipinski_fail")}
                </span>
              </div>
              <div className="mt-1.5 flex items-center gap-3">
                <div className="h-1.5 min-w-0 flex-1 overflow-hidden rounded-full bg-ob-well">
                  <div
                    className={`h-full rounded-full transition-[width] duration-slow ease-out ${
                      passed ? "bg-ok/70" : "bg-danger/70"
                    }`}
                    style={{ width: `${ratio * 100}%` }}
                  />
                </div>
                <span className="shrink-0 text-xs tabular-nums text-ob-faint">
                  {meta.threshold}
                </span>
              </div>
            </div>
          );
        })}
      </div>
      <div className="rounded-control border-l-2 border-brand/40 bg-brand/10 px-4 py-3 text-xs leading-relaxed text-ob-muted">
        <GlossaryText text={t("result.pharma.lipinski_history_html")} />
      </div>
    </div>
  );
}

/* ---------- Drug-likeness gauges ---------- */

function GaugeCard({
  title,
  note,
  value,
  display,
  pct,
  color,
  level,
}: {
  title: string;
  note: string;
  value: number;
  display: string;
  pct: number;
  color: string;
  level: string;
}) {
  return (
    <div className="glass-card flex flex-col items-center gap-2 p-4 text-center">
      <p className="text-2xs uppercase tracking-[0.12em] text-ob-faint">{title}</p>
      <p className="text-3xl font-bold tabular-nums" style={{ color }}>
        {display}
      </p>
      <div className="h-1.5 w-full max-w-[200px] overflow-hidden rounded-full bg-ob-well">
        <div
          className="h-full rounded-full transition-[width]"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <p className="text-sm font-semibold" style={{ color }}>
        {level}
      </p>
      <p className="text-xs leading-relaxed text-ob-faint">{note}</p>
      <span className="hidden">{value}</span>
    </div>
  );
}

function DruglikenessSection({ dl }: { dl: DruglikenessResult }) {
  const { t } = useTranslation();
  const saPct = Math.max(5, Math.min(100, (1 - (dl.sascore - 1) / 9) * 100));
  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-sm font-semibold text-ob-text">
        {t("result.pharma.druglikeness_title")}
      </h3>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <GaugeCard
          title={t("result.pharma.qed_label")}
          note={t("result.pharma.qed_note")}
          value={dl.qed}
          display={dl.qed.toFixed(3)}
          pct={dl.qed * 100}
          color={dl.qed_color}
          level={dl.qed_level}
        />
        <GaugeCard
          title={t("result.pharma.sascore_label")}
          note={t("result.pharma.sascore_note")}
          value={dl.sascore}
          display={dl.sascore.toFixed(2)}
          pct={saPct}
          color={dl.sa_color}
          level={dl.sa_level}
        />
        <GaugeCard
          title={t("result.pharma.fsp3_label")}
          note={`${t("result.pharma.fsp3_note")} ${tf("result.pharma.fsp3_detail", {
            n_sp3: dl.n_sp3,
            n_carbon: dl.n_carbons,
          })}`}
          value={dl.fsp3}
          display={dl.fsp3.toFixed(3)}
          pct={dl.fsp3 * 100}
          color={dl.fsp3_color}
          level={dl.fsp3_level}
        />
      </div>
    </div>
  );
}

/* ---------- Ionization chart ---------- */

/* Tones only — the fill comes from `currentColor`, so the palette lives in
   one place (the token layer) instead of four hex literals here. */
const IONIZATION_TONES = ["text-danger", "text-warn", "text-ok", "text-brand"];

function IonizationSection({ analysis, pka }: { analysis: AnalysisResponse; pka: number }) {
  const { t } = useTranslation();
  const rows = useMemo(
    () =>
      (analysis.ionization ?? []).map((point, i) => ({
        // zh locale stores "Stomach\n胃": the last line is the active-language
        // label, the first is the English one. Taking [0] showed English in a
        // Chinese UI and made four long ticks collide at 375px.
        env: t(`result.pharma.ionization_env_${point.env}`).split("\n").pop() ?? point.env,
        pct: point.pct,
        ph: point.ph,
        color: IONIZATION_TONES[i % IONIZATION_TONES.length],
      })),
    [analysis.ionization, t],
  );
  if (rows.length === 0) return null;

  return (
    <div className="glass-card flex flex-col gap-2 p-4">
      <h3 className="text-sm font-semibold text-ob-text">
        {t("result.pharma.ionization_title")}
      </h3>
      <p className="text-xs text-ob-faint">
        {tf("result.pharma.ionization_chart_title", { val: pka })}
      </p>
      <div className="h-56 w-full overflow-hidden text-ob-muted">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ left: 0, right: 8, top: 12, bottom: 4 }}>
            <XAxis
              dataKey="env"
              stroke="currentColor"
              tick={{ fill: "currentColor", fontSize: 11 }}
              axisLine={{ stroke: "currentColor", strokeOpacity: 0.3 }}
              tickLine={false}
            />
            <YAxis
              domain={[0, 105]}
              stroke="currentColor"
              tick={{ fill: "currentColor", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v: number) => `${v}%`}
            />
            <Tooltip
              cursor={{ fill: "hsl(var(--ob-text) / 0.05)" }}
              contentStyle={{
                background: "hsl(var(--ob-surface))",
                border: "1px solid hsl(var(--ob-rule))",
                borderRadius: 4,
                fontSize: 12,
                boxShadow: "var(--shadow-pop)",
              }}
              labelStyle={{ color: "hsl(var(--ob-text))", fontWeight: 600 }}
              itemStyle={{ color: "hsl(var(--brand-soft))" }}
              formatter={(value: number) => [
                `${value.toFixed(1)}%`,
                t("result.pharma.ionization_ylabel"),
              ]}
            />
            <Bar dataKey="pct" radius={[4, 4, 0, 0]} barSize={44}>
              {rows.map((row) => (
                <Cell
                  key={row.env}
                  className={row.color}
                  fill="currentColor"
                />
              ))}
              <LabelList
                dataKey="pct"
                position="top"
                formatter={(v: number) => `${v.toFixed(1)}%`}
                style={{ fill: "hsl(var(--ob-muted))", fontSize: 11 }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/* ---------- Pharma analysis + linkage ---------- */

const PHARMA_ANALYSIS_STYLE: Record<string, string> = {
  strong_acid: "border-ok/50 bg-ok/5 text-ok",
  mid_acid: "border-brand/40 bg-brand/10 text-brand",
  strong_base: "border-warn/50 bg-warn/5 text-warn",
  weak_base: "border-brand/40 bg-brand/10 text-brand",
  amphoteric: "border-brand/40 bg-brand/10 text-brand",
};

function PharmaAnalysisSection({ analysis }: { analysis: AnalysisResponse }) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-3">
      {analysis.pharma_analysis && (
        <div
          className={`rounded-xl border px-4 py-3 text-sm ${
            PHARMA_ANALYSIS_STYLE[analysis.pharma_analysis] ??
            "border-brand/40 bg-brand/10 text-brand"
          }`}
        >
          <GlossaryText text={t(`result.pharma.analysis.${analysis.pharma_analysis}`)} />
        </div>
      )}
      {analysis.linkage && (
        <div className="rounded-xl border border-brand/30 bg-brand/10 px-4 py-3 text-sm leading-relaxed text-ob-muted">
          <p className="mb-1 text-xs font-semibold uppercase tracking-wider text-brand">
            {t("result.pharma.linkage.title")}
          </p>
          <GlossaryText text={analysis.linkage} />
        </div>
      )}
    </div>
  );
}

/* ---------- ADME/Tox accordion ---------- */

type AdmetTone = "ok" | "brand" | "warn" | "danger";

const ADMET_TONE: Record<AdmetTone, string> = {
  ok: "text-ok",
  brand: "text-brand",
  warn: "text-warn",
  danger: "text-danger",
};

const TOX_STYLE: Record<AdmetTone, string> = {
  ok: "border-ok/40 bg-ok/5",
  brand: "border-brand/40 bg-brand/5",
  warn: "border-warn/40 bg-warn/5",
  danger: "border-danger/40 bg-danger/5",
};

function toxicityTone(level: string): AdmetTone {
  const key = level.trim().toLowerCase();
  if (key === "高" || key === "high") return "danger";
  if (key === "中" || key === "medium") return "warn";
  return "ok";
}

function AdmetSection({ analysis }: { analysis: AnalysisResponse }) {
  const { t } = useTranslation();
  const admet = analysis.admet;

  const sections: { id: string; title: string; tone: AdmetTone; body: ReactNode }[] = [
    {
      id: "absorption",
      title: t("result.pharma.admet_absorption"),
      tone: "ok",
      body: (
        <>
          <p className="text-sm leading-relaxed text-ob-muted">
            <GlossaryText text={admet.absorption} />
          </p>
          <Caption text={t("result.pharma.admet.absorption_caption")} />
        </>
      ),
    },
    {
      id: "distribution",
      title: t("result.pharma.admet_distribution"),
      tone: "brand",
      body: (
        <>
          <div className="mb-2 grid grid-cols-1 gap-2 sm:grid-cols-2">
            <div className="rounded-control border border-brand/20 bg-brand/10 px-3 py-2 text-xs">
              <b className="text-brand">{t("result.pharma.admet_vd")}</b>
              <p className="mt-1 text-ob-muted">{admet.distribution.vd_estimate}</p>
            </div>
            <div className="rounded-control border border-brand/20 bg-brand/10 px-3 py-2 text-xs">
              <b className="text-brand">{t("result.pharma.admet_ppb")}</b>
              <p className="mt-1 text-ob-muted">{admet.distribution.ppb}</p>
            </div>
          </div>
          <p className="text-sm leading-relaxed text-ob-muted">
            <GlossaryText text={admet.distribution.summary} />
          </p>
          <Caption text={t("result.pharma.admet.distribution_caption")} />
        </>
      ),
    },
    {
      id: "metabolism",
      title: t("result.pharma.admet_metabolism"),
      tone: "warn",
      body: (
        <>
          <p className="text-sm leading-relaxed text-ob-muted">
            <GlossaryText text={admet.metabolism.summary} />
          </p>
          <div className="mt-2 rounded-control border border-warn/20 bg-warn/10 px-3 py-2 text-xs">
            <b className="text-warn">{t("result.pharma.admet_cyp")}</b>
            <p className="mt-1 text-ob-muted">{admet.metabolism.cyp_enzymes}</p>
          </div>
          <Caption text={t("result.pharma.admet.metabolism_caption")} />
        </>
      ),
    },
    {
      id: "excretion",
      title: t("result.pharma.admet_excretion"),
      tone: "brand",
      body: (
        <>
          <p className="text-sm font-semibold text-brand">
            {t("result.pharma.admet_excretion_route")}：{admet.excretion.route}
          </p>
          <p className="mt-1 text-sm leading-relaxed text-ob-muted">
            <GlossaryText text={admet.excretion.summary} />
          </p>
          <Caption text={t("result.pharma.admet.excretion_caption")} />
        </>
      ),
    },
    {
      id: "toxicity",
      title: t("result.pharma.admet_toxicity"),
      tone: "danger",
      body: (
        <>
          <div className="flex flex-col gap-2">
            {admet.toxicity.map(([level, desc], i) => {
              const tone = toxicityTone(level);
              return (
                <div
                  key={i}
                  className={`rounded-control border px-3 py-2 text-sm ${TOX_STYLE[tone]}`}
                >
                  <b className={ADMET_TONE[tone]}>
                    {t("result.pharma.admet_risk", { level })}
                  </b>{" "}
                  <span className="text-ob-muted">{desc}</span>
                </div>
              );
            })}
          </div>
          <Caption text={t("result.pharma.admet.toxicity_caption")} />
        </>
      ),
    },
  ];

  return (
    <div className="glass-card p-4">
      <h3 className="mb-2 text-sm font-semibold text-ob-text">
        {t("result.pharma.admet_title")}
      </h3>
      <Accordion type="single" collapsible className="w-full">
        {sections.map((section) => (
          <AccordionItem key={section.id} value={section.id} className="border-ob-border">
            <AccordionTrigger
              className={`text-sm hover:no-underline ${
                section.tone === "danger" ? "text-danger" : "text-ob-text"
              }`}
            >
              {section.title}
            </AccordionTrigger>
            <AccordionContent>{section.body}</AccordionContent>
          </AccordionItem>
        ))}
      </Accordion>
    </div>
  );
}

function Caption({ text }: { text: string }) {
  return (
    <p className="mt-2 text-xs leading-relaxed text-ob-faint">
      <GlossaryText text={text} />
    </p>
  );
}

/* ---------- panel ---------- */

export default function PharmacologyPanel({ result, analysis }: PharmacologyPanelProps) {
  const { t } = useTranslation();

  if (analysis.loading && !analysis.data) {
    return <div className="glass-card p-4 text-sm text-ob-faint">{t("common.loading")}</div>;
  }
  if (analysis.error && !analysis.data) {
    return <div className="glass-card p-4 text-sm text-danger">{analysis.error}</div>;
  }
  if (!analysis.data) return null;

  const data = analysis.data;
  return (
    <div className="flex flex-col gap-4">
      <LipinskiSection analysis={data} />
      {data.druglikeness && <DruglikenessSection dl={data.druglikeness} />}
      {result.pka != null && data.ionization && (
        <IonizationSection analysis={data} pka={result.pka} />
      )}
      <PharmaAnalysisSection analysis={data} />
      <AdmetSection analysis={data} />
    </div>
  );
}
