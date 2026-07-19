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
import { tf } from "@/i18n/format";
import type { AnalysisResponse, PredictionResult } from "@/types/api";

interface PkaPanelProps {
  result: PredictionResult;
  analysis: {
    data: AnalysisResponse | null;
    error: string | null;
    loading: boolean;
  };
}

const KIND_META: Record<string, { key: string; descKey: string; color: string }> = {
  acid: {
    key: "model.pka.type.acidic_display",
    descKey: "model.pka.type.acidic_desc",
    color: "#a78bfa",
  },
  base: {
    key: "model.pka.type.basic_display",
    descKey: "model.pka.type.basic_desc",
    color: "#22d3ee",
  },
  amphoteric: {
    key: "model.pka.type.amphoteric_display",
    descKey: "model.pka.type.amphoteric_desc",
    color: "#fbbf24",
  },
};

interface FactorRow {
  name: string;
  value: number;
}

export default function PkaPanel({ result, analysis }: PkaPanelProps) {
  const { t } = useTranslation();
  const pka = result.pka;
  const kind = result.pka_kind ? KIND_META[result.pka_kind] : null;

  const rows = useMemo<FactorRow[]>(() => {
    const factors = analysis.data?.pka_factors;
    if (!factors) return [];
    return Object.entries(factors)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .reverse(); // largest on top with vertical layout
  }, [analysis.data]);

  if (pka == null) {
    return (
      <div className="glass-card p-4 text-sm text-ob-muted">
        {t("result.pka.model_unavailable_short")}
      </div>
    );
  }

  const isAcid = pka < 7;
  const unit = isAcid ? t("result.pka.unit_acid") : t("result.pka.unit_base");
  const legendType = isAcid ? t("result.pka.legend_type_acid") : t("result.pka.legend_type_base");

  return (
    <div className="flex flex-col gap-4">
      {/* Value + kind badge */}
      <div className="glass-card flex flex-wrap items-center gap-6 p-5">
        <div>
          <p className="text-xs text-ob-faint">{t("result.pka.metric")}</p>
          <p className="mt-1 text-5xl font-bold tabular-nums text-cyan-300">
            {pka.toFixed(2)}
          </p>
        </div>
        {kind && (
          <div className="flex flex-col gap-1">
            <span
              className="w-fit rounded-full px-3 py-1 text-sm font-semibold"
              style={{
                color: kind.color,
                border: `1px solid ${kind.color}66`,
                background: `${kind.color}1a`,
              }}
            >
              {t(kind.key)}
            </span>
            <p className="max-w-md text-xs text-ob-muted">{t(kind.descKey)}</p>
          </div>
        )}
      </div>

      {/* Factor decomposition */}
      <div className="glass-card flex flex-col gap-2 p-4">
        <h3 className="text-sm font-semibold text-ob-text">
          {t("result.pka.decomp_title")}
        </h3>
        {analysis.loading && <p className="text-sm text-ob-faint">{t("common.loading")}</p>}
        {analysis.error && <p className="text-sm text-red-400">{analysis.error}</p>}
        {!analysis.loading && !analysis.error && rows.length === 0 && (
          <p className="text-sm text-ob-muted">{t("result.pka.unavailable_short")}</p>
        )}
        {rows.length > 0 && (
          <>
            <p className="text-xs text-ob-faint">
              {tf("result.pka.chart_title", { val: pka })} · {tf("result.pka.chart_xlabel", { unit })}
            </p>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 48, top: 4, bottom: 4 }}>
                  <XAxis
                    type="number"
                    stroke="#6b6b7b"
                    tick={{ fill: "#a0a0b0", fontSize: 11 }}
                    axisLine={{ stroke: "#2a2a3a" }}
                    tickLine={false}
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    width={170}
                    stroke="#6b6b7b"
                    tick={{ fill: "#d0d0e0", fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
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
                    formatter={(value: number) => [value.toFixed(2), unit]}
                  />
                  <Bar dataKey="value" radius={[3, 3, 3, 3]} barSize={20}>
                    {rows.map((row) => (
                      <Cell key={row.name} fill={row.value > 0 ? "#a78bfa" : "#22d3ee"} />
                    ))}
                    <LabelList
                      dataKey="value"
                      position="right"
                      formatter={(v: number) => (v > 0 ? `+${v.toFixed(2)}` : v.toFixed(2))}
                      style={{ fill: "#a0a0b0", fontSize: 11 }}
                    />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="text-xs text-ob-faint">
              <span style={{ color: "#a78bfa" }}>■</span> {tf("result.pka.legend_enhance", { type: legendType })}
              {"  "}
              <span style={{ color: "#22d3ee" }}>■</span> {tf("result.pka.legend_weaken", { type: legendType })}
            </p>
            <p className="text-xs text-ob-muted">
              <GlossaryText text={t("result.pka.factor_guide")} />
            </p>
          </>
        )}

        {/* Glossary hints */}
        <div className="mt-2 rounded-lg border-l-2 border-nebula/40 bg-nebula/6 px-4 py-3 text-xs leading-loose text-ob-muted">
          <b className="text-nebula-light">{t("result.pka.glossary_title")}</b>
          <ul className="mt-1 space-y-1">
            {(["inductive", "resonance", "intra_hb", "steric", "hybrid"] as const).map((k) => (
              <li key={k}>
                • <GlossaryText text={t(`result.pka.glossary_${k}`)} />
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
