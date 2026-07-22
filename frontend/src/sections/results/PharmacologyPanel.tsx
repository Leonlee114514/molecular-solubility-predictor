import { useMemo } from "react";
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
  const titleColor = lip.passed >= 4 ? "#34d399" : "#fbbf24";

  return (
    <div className="glass-card flex flex-col gap-3 p-4">
      <h3 className="text-sm font-semibold text-ob-text">
        {t("result.pharma.lipinski_title")}
      </h3>
      <p className="text-sm font-semibold" style={{ color: titleColor }}>
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
          return (
            <div
              key={meta.i18nKey}
              className="flex items-center gap-3 rounded-xl border border-ob-border bg-ob-bg/40 px-3 py-2"
            >
              <span className="w-40 shrink-0 text-xs text-ob-muted">{t(meta.i18nKey)}</span>
              <span className="w-16 shrink-0 text-xs text-ob-faint">{meta.threshold}</span>
              <div
                className="h-2 flex-1 overflow-hidden rounded-full"
                style={{ background: "#1e1e30" }}
              >
                <div
                  className="h-full rounded-full"
                  style={{
                    width: "100%",
                    background: passed ? "#34d399" : "#f87171",
                    opacity: 0.75,
                  }}
                />
              </div>
              <span
                className="w-24 shrink-0 text-right text-xs font-semibold tabular-nums"
                style={{ color: passed ? "#34d399" : "#f87171" }}
              >
                {actual} {passed ? t("result.pharma.lipinski_pass") : t("result.pharma.lipinski_fail")}
              </span>
            </div>
          );
        })}
      </div>
      <div className="rounded-lg border-l-2 border-nebula/40 bg-nebula/6 px-4 py-3 text-xs leading-relaxed text-ob-muted">
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
      <p className="text-[0.68rem] uppercase tracking-[0.12em] text-ob-faint">{title}</p>
      <p className="text-3xl font-bold tabular-nums" style={{ color }}>
        {display}
      </p>
      <div className="h-1.5 w-full max-w-[200px] overflow-hidden rounded-full bg-ob-bg">
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

const IONIZATION_COLORS = ["#f87171", "#fbbf24", "#34d399", "#60a5fa"];

function IonizationSection({ analysis, pka }: { analysis: AnalysisResponse; pka: number }) {
  const { t } = useTranslation();
  const rows = useMemo(
    () =>
      (analysis.ionization ?? []).map((point, i) => ({
        env: t(`result.pharma.ionization_env_${point.env}`).split("\n")[0],
        pct: point.pct,
        ph: point.ph,
        color: IONIZATION_COLORS[i % IONIZATION_COLORS.length],
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
      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ left: 0, right: 8, top: 12, bottom: 4 }}>
            <XAxis
              dataKey="env"
              stroke="#6b6b7b"
              tick={{ fill: "#d0d0e0", fontSize: 11 }}
              axisLine={{ stroke: "#2a2a3a" }}
              tickLine={false}
            />
            <YAxis
              domain={[0, 105]}
              stroke="#6b6b7b"
              tick={{ fill: "#a0a0b0", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v: number) => `${v}%`}
            />
            <Tooltip
              cursor={{ fill: "rgba(124,58,237,0.08)" }}
              contentStyle={{
                background: "rgba(26,26,46,0.95)",
                border: "1px solid rgba(255,255,255,0.12)",
                borderRadius: 10,
                fontSize: 12,
              }}
              labelStyle={{ color: "#f0f0f5" }}
              formatter={(value: number) => [
                `${value.toFixed(1)}%`,
                t("result.pharma.ionization_ylabel"),
              ]}
            />
            <Bar dataKey="pct" radius={[4, 4, 0, 0]} barSize={44}>
              {rows.map((row) => (
                <Cell key={row.env} fill={row.color} />
              ))}
              <LabelList
                dataKey="pct"
                position="top"
                formatter={(v: number) => `${v.toFixed(1)}%`}
                style={{ fill: "#a0a0b0", fontSize: 11 }}
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
  strong_acid: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
  mid_acid: "border-cyan-500/40 bg-cyan-500/10 text-cyan-300",
  strong_base: "border-amber-500/50 bg-amber-500/10 text-amber-300",
  weak_base: "border-cyan-500/40 bg-cyan-500/10 text-cyan-300",
  amphoteric: "border-cyan-500/40 bg-cyan-500/10 text-cyan-300",
};

function PharmaAnalysisSection({ analysis }: { analysis: AnalysisResponse }) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-3">
      {analysis.pharma_analysis && (
        <div
          className={`rounded-xl border px-4 py-3 text-sm ${
            PHARMA_ANALYSIS_STYLE[analysis.pharma_analysis] ??
            "border-cyan-500/40 bg-cyan-500/10 text-cyan-300"
          }`}
        >
          <GlossaryText text={t(`result.pharma.analysis.${analysis.pharma_analysis}`)} />
        </div>
      )}
      {analysis.linkage && (
        <div className="rounded-xl border border-cyan-500/30 bg-cyan-500/8 px-4 py-3 text-sm leading-relaxed text-ob-muted">
          <p className="mb-1 text-xs font-semibold uppercase tracking-wider text-cyan-300">
            {t("result.pharma.linkage.title")}
          </p>
          <GlossaryText text={analysis.linkage} />
        </div>
      )}
    </div>
  );
}

/* ---------- ADME/Tox accordion ---------- */

function toxicityColor(level: string): string {
  const key = level.trim().toLowerCase();
  if (key === "高" || key === "high") return "#f87171";
  if (key === "中" || key === "medium") return "#fbbf24";
  return "#34d399";
}

function AdmetSection({ analysis }: { analysis: AnalysisResponse }) {
  const { t } = useTranslation();
  const admet = analysis.admet;

  const sections = [
    {
      id: "absorption",
      title: t("result.pharma.admet_absorption"),
      color: "#34d399",
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
      color: "#60a5fa",
      body: (
        <>
          <div className="mb-2 grid grid-cols-1 gap-2 sm:grid-cols-2">
            <div className="rounded-lg border border-blue-400/20 bg-blue-400/8 px-3 py-2 text-xs">
              <b style={{ color: "#60a5fa" }}>{t("result.pharma.admet_vd")}</b>
              <p className="mt-1 text-ob-muted">{admet.distribution.vd_estimate}</p>
            </div>
            <div className="rounded-lg border border-blue-400/20 bg-blue-400/8 px-3 py-2 text-xs">
              <b style={{ color: "#60a5fa" }}>{t("result.pharma.admet_ppb")}</b>
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
      color: "#fbbf24",
      body: (
        <>
          <p className="text-sm leading-relaxed text-ob-muted">
            <GlossaryText text={admet.metabolism.summary} />
          </p>
          <div className="mt-2 rounded-lg border border-stargold/20 bg-stargold/8 px-3 py-2 text-xs">
            <b className="text-stargold">{t("result.pharma.admet_cyp")}</b>
            <p className="mt-1 text-ob-muted">{admet.metabolism.cyp_enzymes}</p>
          </div>
          <Caption text={t("result.pharma.admet.metabolism_caption")} />
        </>
      ),
    },
    {
      id: "excretion",
      title: t("result.pharma.admet_excretion"),
      color: "#a78bfa",
      body: (
        <>
          <p className="text-sm font-semibold" style={{ color: "#a78bfa" }}>
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
      color: "#f87171",
      body: (
        <>
          <div className="flex flex-col gap-2">
            {admet.toxicity.map(([level, desc], i) => {
              const color = toxicityColor(level);
              return (
                <div
                  key={i}
                  className="rounded-lg px-3 py-2 text-sm"
                  style={{ background: `${color}14`, border: `1px solid ${color}33` }}
                >
                  <b style={{ color }}>{t("result.pharma.admet_risk", { level })}</b>{" "}
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
            <AccordionTrigger className="text-sm hover:no-underline" style={{ color: section.color }}>
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
    return <div className="glass-card p-4 text-sm text-red-400">{analysis.error}</div>;
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
