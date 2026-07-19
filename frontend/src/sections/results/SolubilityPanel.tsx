import { useMemo, useState } from "react";
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

import { api, ApiError } from "@/api/client";
import GlossaryText from "@/components/GlossaryText";
import { descriptorLabel, GNN_FEATURE_NAMES } from "@/data/descriptors";
import { tf } from "@/i18n/format";
import type { GnnExplainResponse, PredictionResult } from "@/types/api";

interface SolubilityPanelProps {
  result: PredictionResult;
}

/* ---------- descriptor grid ---------- */

const DESCRIPTOR_ITEMS: { key: string; i18nKey: string; format: (v: number) => string }[] = [
  { key: "MolWt", i18nKey: "result.solubility.desc_mw", format: (v) => v.toFixed(1) },
  { key: "LogP", i18nKey: "result.solubility.desc_logp", format: (v) => v.toFixed(2) },
  { key: "NumHDonors", i18nKey: "result.solubility.desc_hbd", format: (v) => String(v) },
  { key: "NumHAcceptors", i18nKey: "result.solubility.desc_hba", format: (v) => String(v) },
  { key: "TPSA", i18nKey: "result.solubility.desc_tpsa", format: (v) => v.toFixed(1) },
  { key: "NumRotatableBonds", i18nKey: "result.solubility.desc_rotb", format: (v) => String(v) },
  { key: "NumAromaticRings", i18nKey: "result.solubility.desc_arom", format: (v) => String(v) },
  { key: "NumAliphaticRings", i18nKey: "result.solubility.desc_aliph", format: (v) => String(v) },
];

/* ---------- SHAP chart ---------- */

interface ShapRow {
  name: string;
  value: number;
}

function ShapChart({ result }: { result: PredictionResult }) {
  const { t } = useTranslation();
  const rows = useMemo<ShapRow[]>(() => {
    if (!result.shap_values || !result.shap_names) return [];
    const pairs = result.shap_names.map((name, i) => ({
      name: descriptorLabel(t, name),
      value: result.shap_values![i],
    }));
    pairs.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    // Recharts renders the first item at the bottom; reverse so largest is on top.
    return pairs.slice(0, 8).reverse();
  }, [result.shap_values, result.shap_names, t]);

  if (rows.length === 0) return null;
  const base = result.shap_base_value;

  return (
    <div className="glass-card flex flex-col gap-2 p-4">
      <h3 className="text-sm font-semibold text-ob-text">
        {t("result.solubility.shap_title")}
      </h3>
      <p className="text-xs text-ob-faint">
        {base != null &&
          tf("result.solubility.shap_title_pred", {
            pred: result.logS_final,
            base,
          })}
      </p>
      <div className="h-72 w-full">
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
              width={190}
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
              itemStyle={{ color: "#a78bfa" }}
              formatter={(value: number) => [value.toFixed(4), t("result.solubility.shap_xlabel")]}
            />
            <Bar dataKey="value" radius={[3, 3, 3, 3]} barSize={16}>
              {rows.map((row) => (
                <Cell key={row.name} fill={row.value > 0 ? "#a78bfa" : "#06b6d4"} />
              ))}
              <LabelList
                dataKey="value"
                position="right"
                formatter={(v: number) => (v > 0 ? `+${v.toFixed(3)}` : v.toFixed(3))}
                style={{ fill: "#a0a0b0", fontSize: 11 }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-ob-faint">
        <span style={{ color: "#a78bfa" }}>■</span> {t("result.solubility.shap_legend_pos")}
        {"  "}
        <span style={{ color: "#06b6d4" }}>■</span> {t("result.solubility.shap_legend_neg")}
      </p>
      <p className="text-xs text-ob-muted">{t("result.solubility.shap_guide")}</p>
    </div>
  );
}

/* ---------- SHAP insight (ported from ui/results.py) ---------- */

function ShapInsight({ result }: { result: PredictionResult }) {
  const { t } = useTranslation();
  const text = useMemo(() => {
    if (!result.shap_values || !result.shap_names || result.shap_base_value == null) {
      return null;
    }
    const prediction = result.logS_final;
    const base = result.shap_base_value;
    const pairs = result.shap_names
      .map((name, i) => ({ name, value: result.shap_values![i] }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, 3);

    const levelKey =
      prediction > 0
        ? "result.solubility.high"
        : prediction > -2
          ? "result.solubility.moderate"
          : "result.solubility.poor";

    const supporting: string[] = [];
    const resisting: string[] = [];
    for (const { name, value } of pairs) {
      const label = descriptorLabel(t, name);
      if (prediction <= -2) {
        if (value < 0) supporting.push(`**${label}**（${value.toFixed(3)}）`);
        else resisting.push(`**${label}**（+${value.toFixed(3)}）`);
      } else if (prediction >= 0) {
        if (value > 0) supporting.push(`**${label}**（+${value.toFixed(3)}）`);
        else resisting.push(`**${label}**（${value.toFixed(3)}）`);
      } else {
        const dir = value > 0 ? t("ai.shap.direction_up") : t("ai.shap.direction_down");
        supporting.push(`**${label}**（${value > 0 ? "+" : ""}${value.toFixed(3)}，${dir}）`);
      }
    }

    const parts: string[] = [
      tf("result.solubility.shap_insight_leading", {
        level: t(levelKey),
        logS: prediction,
      }),
    ];
    if (supporting.length > 0) {
      parts.push(tf("result.solubility.shap_supporting", { factors: supporting.join(", ") }));
    }
    if (resisting.length > 0) {
      const target =
        prediction <= -2
          ? t("result.solubility.shap_target_soluble")
          : t("result.solubility.shap_target_insoluble");
      parts.push(tf("result.solubility.shap_resisting", { target, factors: resisting.join(", ") }));
    }
    parts.push(
      tf("result.solubility.shap_shift", {
        base,
        direction:
          prediction > base
            ? t("result.solubility.shap_dir_up")
            : t("result.solubility.shap_dir_down"),
        shift: Math.abs(prediction - base),
      }),
    );
    return parts.join(" ");
  }, [result, t]);

  if (!text) return null;
  return (
    <div className="rounded-xl border border-nebula/25 bg-nebula/8 px-4 py-3 text-sm leading-relaxed text-ob-muted">
      <span className="font-semibold text-ob-text">💡 {t("result.solubility.shap_insight_title")}</span>
      <br />
      <GlossaryText text={text} />
    </div>
  );
}

/* ---------- GNN explainer ---------- */

function GnnSection({ smiles }: { smiles: string }) {
  const { t } = useTranslation();
  const [busy, setBusy] = useState(false);
  const [data, setData] = useState<GnnExplainResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      setData(await api.gnnExplain(smiles));
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : String(err));
    } finally {
      setBusy(false);
    }
  };

  const topBonds = useMemo(
    () => (data ? [...data.bond_details].sort((a, b) => b[2] - a[2]).slice(0, 5) : []),
    [data],
  );
  const topFeatures = useMemo(
    () =>
      data
        ? data.feature_importance
            .map((value, idx) => ({
              name: GNN_FEATURE_NAMES[idx] ?? `Dim_${idx}`,
              value,
            }))
            .sort((a, b) => b.value - a.value)
            .slice(0, 6)
        : [],
    [data],
  );
  const bondsParam = useMemo(
    () =>
      data
        ? data.bond_importance.map(([a, b, imp]) => `${a}-${b}:${imp.toFixed(3)}`).join(",")
        : null,
    [data],
  );

  return (
    <div className="glass-card flex flex-col gap-3 p-4">
      <h3 className="text-sm font-semibold text-ob-text">{t("result.gnn.title")}</h3>
      <p className="text-xs text-ob-faint">{t("result.gnn.desc")}</p>

      {!data && (
        <button
          type="button"
          disabled={busy}
          onClick={() => void run()}
          className="self-start rounded-lg bg-nebula px-4 py-2 text-sm text-white transition-opacity hover:opacity-90 disabled:opacity-40"
        >
          {busy ? t("result.gnn.spinner") : `⚛ ${t("result.gnn.run_btn")}`}
        </button>
      )}
      {error && <p className="text-xs text-red-400">{error}</p>}

      {data && (
        <>
          <p className="text-xs text-ob-faint">
            <GlossaryText text={tf("result.gnn.elapsed_text", { t: data.elapsed })} glossary={false} />
          </p>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1.2fr_1fr]">
            <div className="flex flex-col items-center gap-2">
              <img
                src={api.mol2dUrl(smiles, bondsParam ?? undefined)}
                alt={t("result.gnn.img_caption")}
                className="w-full max-w-[500px] rounded-lg border border-ob-border"
              />
              <p className="text-xs text-ob-faint">{t("result.gnn.img_caption")}</p>
            </div>
            <div className="flex flex-col gap-3">
              <p className="text-sm font-semibold text-ob-text">
                <GlossaryText text={t("result.gnn.top_bonds_title")} glossary={false} />
              </p>
              {topBonds.map(([a, b, imp, label], rank) => (
                <div key={`${a}-${b}`} className="text-sm">
                  <span className="text-ob-faint">#{rank + 1}</span>{" "}
                  <span className="font-medium text-ob-text">{label}</span>
                  <div className="mt-1 h-3 overflow-hidden rounded-full bg-ob-bg">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${Math.min(100, imp * 100)}%`,
                        background: `rgba(${Math.round(140 + 115 * imp)},${Math.round(
                          60 + 175 * imp,
                        )},${Math.round(230 - 200 * imp)},0.6)`,
                      }}
                    />
                  </div>
                  <span className="text-xs text-ob-faint">
                    {tf("result.gnn.top_bonds_importance", { imp, pct: imp * 100 })}
                  </span>
                </div>
              ))}

              {topFeatures.length > 0 && (
                <>
                  <p className="mt-2 text-sm font-semibold text-ob-text">
                    <GlossaryText text={t("result.gnn.feature_title")} glossary={false} />
                  </p>
                  {topFeatures.map((f) => (
                    <div key={f.name} className="text-xs">
                      <span className="text-ob-muted">{f.name}</span>
                      <div className="mt-0.5 h-2 w-4/5 overflow-hidden rounded-full bg-ob-bg">
                        <div
                          className="h-full rounded-full bg-cyan-400"
                          style={{ width: `${Math.min(100, f.value * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </>
              )}
            </div>
          </div>

          <details className="rounded-xl border border-ob-border bg-ob-bg/40 px-4 py-2 text-sm">
            <summary className="cursor-pointer text-ob-muted hover:text-ob-text">
              {t("result.gnn.how_to_read_title")}
            </summary>
            <div className="mt-2 space-y-3 text-xs leading-relaxed text-ob-muted">
              <GlossaryText text={t("result.gnn.how_to_read_html")} glossary={false} />
              <GlossaryText text={t("result.gnn.how_to_read_tech")} glossary={false} />
            </div>
          </details>
        </>
      )}
    </div>
  );
}

/* ---------- panel ---------- */

export default function SolubilityPanel({ result }: SolubilityPanelProps) {
  const { t } = useTranslation();
  const rf = result.logS_rf;
  const gnn = result.logS_gnn;
  const diff = rf != null && gnn != null ? Math.abs(rf - gnn) : null;
  const isEnsemble = result.model_used === "Ensemble" || result.model_used === "Ensemble(W)";
  const hasGnn = result.model_used.includes("GNN") || isEnsemble;

  return (
    <div className="flex flex-col gap-4">
      {/* Interpretation guide */}
      <div className="glass-card p-4 font-mono text-xs leading-relaxed text-ob-muted">
        <b className="text-ob-text">{t("result.solubility.guide")}</b>
        <br />
        <span className="text-emerald-400">&gt;</span> logS &gt; 0: {t("result.solubility.high_hint")}
        <br />
        <span className="text-stargold">&gt;</span> -2 &lt; logS &lt; 0: {t("result.solubility.moderate")}
        <br />
        <span className="text-red-400">&gt;</span> logS &lt; -2: {t("result.solubility.poor_hint")}
      </div>

      {/* Disagreement warning */}
      {diff != null && diff > 0.5 && (
        <div className="rounded-xl border border-red-500/50 bg-red-500/10 px-4 py-3 text-sm text-red-300">
          {diff > 1.0
            ? tf("result.solubility.severe_disagree", { diff })
            : tf("result.solubility.notable_disagree", { diff })}
        </div>
      )}

      {/* Ensemble components */}
      {isEnsemble && rf != null && gnn != null && diff != null && (
        <div className="rounded-xl border border-stargold/25 bg-stargold/8 px-4 py-3 text-sm">
          <b className="text-stargold">
            {result.model_used === "Ensemble(W)"
              ? t("result.solubility.badge_weighted")
              : t("result.solubility.badge_ensemble")}
          </b>
          {result.model_used === "Ensemble(W)" && (
            <span className="ml-2 text-xs text-ob-faint">
              {t("result.solubility.ensemble_weights")}
            </span>
          )}
          <br />
          <span className="text-emerald-400">{t("result.solubility.ensemble_rf")}</span>{" "}
          {rf.toFixed(3)}
          {"  |  "}
          <span className="text-nebula-light">{t("result.solubility.ensemble_gnn")}</span>{" "}
          {gnn.toFixed(3)}
          <br />
          <span className="text-stargold">{t("result.solubility.disagreement")}:</span>{" "}
          {diff.toFixed(3)}{" "}
          <span className="text-xs text-ob-faint">
            {diff < 0.5
              ? t("result.solubility.good_agreement")
              : diff < 1.0
                ? t("result.solubility.notable_disagreement")
                : t("result.solubility.large_divergence")}
          </span>
        </div>
      )}

      {/* Descriptor grid */}
      <div className="glass-card p-4">
        <h3 className="mb-3 text-sm font-semibold text-ob-text">
          {t("result.solubility.descriptors")}
        </h3>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {DESCRIPTOR_ITEMS.map((item) => {
            const value = result.features[item.key];
            return (
              <div key={item.key} className="rounded-xl border border-ob-border bg-ob-bg/40 p-3">
                <p className="text-xs text-ob-faint">{t(item.i18nKey)}</p>
                <p className="mt-1 text-xl font-semibold tabular-nums text-ob-text">
                  {value != null ? item.format(value) : "—"}
                </p>
              </div>
            );
          })}
        </div>
        <div className="mt-3 rounded-xl border border-cyan-500/20 bg-cyan-500/5 px-4 py-3 text-xs leading-relaxed text-ob-muted">
          <b className="text-ob-text">{t("result.solubility.insight_title")}:</b>
          <ul className="mt-1 list-disc space-y-1 pl-5">
            <li><GlossaryText text={t("result.solubility.insight_tpsa")} /></li>
            <li><GlossaryText text={t("result.solubility.insight_hbond")} /></li>
            <li><GlossaryText text={t("result.solubility.insight_logp")} /></li>
          </ul>
        </div>
      </div>

      {/* GNN explainer (before SHAP, like the old app) */}
      {hasGnn && <GnnSection smiles={result.smiles} />}

      {/* SHAP (RF-based predictions only) */}
      {result.shap_values && result.shap_names && (
        <>
          <ShapChart result={result} />
          <ShapInsight result={result} />
        </>
      )}
    </div>
  );
}
