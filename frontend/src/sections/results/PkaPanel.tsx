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
    color: "#2c5282",
  },
  base: {
    key: "model.pka.type.basic_display",
    descKey: "model.pka.type.basic_desc",
    color: "#b45309",
  },
  amphoteric: {
    key: "model.pka.type.amphoteric_display",
    descKey: "model.pka.type.amphoteric_desc",
    color: "#a16207",
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

  // The acid/base unit copy follows the resolved kind (structural evidence from
  // the backend). pka_kind is always present whenever pka is - both come from
  // the same resolution step in services/prediction.py - so no numeric fallback
  // is needed here, and the copy can no longer contradict the badge below.
  const isAcid = result.pka_kind === "acid";
  const unit = isAcid ? t("result.pka.unit_acid") : t("result.pka.unit_base");
  const legendType = isAcid ? t("result.pka.legend_type_acid") : t("result.pka.legend_type_base");

  return (
    <div className="flex flex-col gap-4">
      {/* Value + kind badge */}
      <div className="glass-card flex flex-wrap items-center gap-6 p-5">
        <div>
          <p className="text-xs text-ob-faint">{t("result.pka.metric")}</p>
          <p className="mt-1 text-5xl font-bold tabular-nums text-nebula">
            {pka.toFixed(2)}
          </p>
        </div>
        {result.pka_acidic != null && result.pka_basic != null && (
          <div className="flex items-end gap-6">
            <div>
              <p className="text-xs text-ob-faint">{t("result.pka.metric_acidic")}</p>
              <p className="mt-1 text-2xl font-semibold tabular-nums text-nebula">
                {result.pka_acidic.toFixed(2)}
              </p>
            </div>
            <div>
              <p className="text-xs text-ob-faint">{t("result.pka.metric_basic")}</p>
              <p className="mt-1 text-2xl font-semibold tabular-nums text-nebula">
                {result.pka_basic.toFixed(2)}
              </p>
            </div>
          </div>
        )}
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
            {result.pka_acidic != null && result.pka_basic != null && (
              <p className="max-w-md text-xs text-ob-faint">{t("result.pka.dual_note")}</p>
            )}
          </div>
        )}
      </div>

      {/* Factor decomposition */}
      <div className="glass-card flex flex-col gap-2 p-4">
        <h3 className="text-sm font-semibold text-ob-text">
          {t("result.pka.decomp_title")}
        </h3>
        {analysis.loading && <p className="text-sm text-ob-faint">{t("common.loading")}</p>}
        {analysis.error && <p className="text-sm text-red-700">{analysis.error}</p>}
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
                    stroke="#b8b2a8"
                    tick={{ fill: "#5c574f", fontSize: 11 }}
                    axisLine={{ stroke: "#ded8cd" }}
                    tickLine={false}
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    width={170}
                    stroke="#b8b2a8"
                    tick={{ fill: "#5c574f", fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip
                    cursor={{ fill: "rgba(28,26,23,0.05)" }}
                    contentStyle={{
                      background: "#fffdf9",
                      border: "1px solid rgba(28,26,23,0.12)",
                      borderRadius: 6,
                      fontSize: 12,
                      boxShadow: "0 2px 8px rgba(28,26,23,0.08)",
                    }}
                    labelStyle={{ color: "#1c1a17", fontWeight: 600 }}
                    itemStyle={{ color: "#2c5282" }}
                    formatter={(value: number) => [value.toFixed(2), unit]}
                  />
                  <Bar dataKey="value" radius={[3, 3, 3, 3]} barSize={20}>
                    {rows.map((row) => (
                      <Cell key={row.name} fill={row.value > 0 ? "#2c5282" : "#b45309"} />
                    ))}
                    <LabelList
                      dataKey="value"
                      position="right"
                      formatter={(v: number) => (v > 0 ? `+${v.toFixed(2)}` : v.toFixed(2))}
                      style={{ fill: "#5c574f", fontSize: 11 }}
                    />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="text-xs text-ob-faint">
              <span style={{ color: "#2c5282" }}>■</span> {tf("result.pka.legend_enhance", { type: legendType })}
              {"  "}
              <span style={{ color: "#b45309" }}>■</span> {tf("result.pka.legend_weaken", { type: legendType })}
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
