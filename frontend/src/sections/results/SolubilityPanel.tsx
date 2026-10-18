import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Lightbulb } from "lucide-react";
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
import { Button } from "@/components/ui/button";
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
      <div className="h-72 w-full overflow-hidden text-ob-muted">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 48, top: 4, bottom: 4 }}>
            <XAxis
              type="number"
              stroke="currentColor"
              tick={{ fill: "currentColor", fontSize: 11 }}
              axisLine={{ stroke: "currentColor", strokeOpacity: 0.3 }}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={132}
              stroke="currentColor"
              tick={{ fill: "currentColor", fontSize: 11 }}
              tickFormatter={shortDescriptor}
              axisLine={false}
              tickLine={false}
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
              formatter={(value: number) => [value.toFixed(4), t("result.solubility.shap_xlabel")]}
            />
            <Bar dataKey="value" radius={[3, 3, 3, 3]} barSize={16}>
              {rows.map((row) => (
                <Cell
                  key={row.name}
                  className={row.value > 0 ? "text-brand" : "text-warn"}
                  fill="currentColor"
                />
              ))}
              <LabelList
                dataKey="value"
                position="right"
                formatter={(v: number) => (v > 0 ? `+${v.toFixed(3)}` : v.toFixed(3))}
                style={{ fill: "hsl(var(--ob-muted))", fontSize: 11 }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-ob-faint">
        <span className="text-brand">■</span> {t("result.solubility.shap_legend_pos")}
        {"  "}
        <span className="text-warn">■</span> {t("result.solubility.shap_legend_neg")}
      </p>
      <p className="text-xs text-ob-muted">{t("result.solubility.shap_guide")}</p>
    </div>
  );
}

/**
 * Axis labels arrive as full descriptor names ("Molecular Flexibility (Hall-Kier α)").
 * Axis space is tight, so the tick shows the trailing parenthesised short form and
 * the full name stays available in the tooltip - recharts' tickFormatter only
 * changes the tick text, not the value the tooltip labels itself with.
 */
function shortDescriptor(name: string): string {
  const match = name.match(/[（(]\s*([^（()）]+?)\s*[)）]\s*$/);
  const base = match ? match[1] : name;
  return base.length > 13 ? `${base.slice(0, 12)}…` : base;
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
    <div className="rounded-card border border-brand/30 bg-brand/10 px-4 py-3 text-sm leading-relaxed text-ob-muted">
      <span className="flex items-center gap-1.5 font-semibold text-ob-text">
        <Lightbulb className="size-4 text-brand" aria-hidden="true" />
        {t("result.solubility.shap_insight_title")}
      </span>
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
        <Button
          size="sm"
          loading={busy}
          onClick={() => void run()}
          className="self-start"
        >
          {busy ? t("result.gnn.spinner") : t("result.gnn.run_btn")}
        </Button>
      )}
      {error && <p className="text-xs text-danger">{error}</p>}

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
                  <div className="mt-1 h-3 overflow-hidden rounded-full bg-ob-well">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${Math.min(100, imp * 100)}%`,
                        // Importance is depth of ink, not a hue shift: the old
                        // blue→red ramp read as "safe→dangerous" when it only
                        // meant "weak→strong bond".
                        background: `hsl(var(--brand) / ${(0.35 + 0.55 * imp).toFixed(2)})`,
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
                      <div className="mt-0.5 h-2 w-4/5 overflow-hidden rounded-full bg-ob-well">
                        <div
                          className="h-full rounded-full bg-brand"
                          style={{ width: `${Math.min(100, f.value * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </>
              )}
            </div>
          </div>

          <details className="rounded-card border border-ob-border bg-ob-bg/40 px-4 py-2 text-sm">
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
      <div className="glass-card p-4 text-xs leading-relaxed text-ob-muted">
        <b className="text-ob-text">{t("result.solubility.guide")}</b>
        <br />
        <span className="text-ok">&gt;</span> logS &gt; 0: {t("result.solubility.high_hint")}
        <br />
        <span className="text-warn">&gt;</span> -2 &lt; logS &lt; 0: {t("result.solubility.moderate")}
        <br />
        <span className="text-danger">&gt;</span> logS &lt; -2: {t("result.solubility.poor_hint")}
      </div>

      {/* Disagreement warning */}
      {diff != null && diff > 0.5 && (
        <div className="rounded-card border border-danger/50 bg-danger/5 px-4 py-3 text-sm text-danger">
          {diff > 1.0
            ? (result.model_selected === "auto"
                ? tf("result.solubility.severe_disagree_auto", { diff })
                : tf("result.solubility.severe_disagree", { diff }))
            : tf("result.solubility.notable_disagree", { diff })}
        </div>
      )}

      {/* Ensemble components */}
      {isEnsemble && rf != null && gnn != null && diff != null && (
        <div className="rounded-card border border-ob-rule bg-ob-bg/40 px-4 py-3 text-sm">
          <b className="text-ob-text">
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
          <span className="text-ob-muted">{t("result.solubility.ensemble_rf")}</span>{" "}
          <span className="tabular-nums">{rf.toFixed(3)}</span>
          {"  |  "}
          <span className="text-ob-muted">{t("result.solubility.ensemble_gnn")}</span>{" "}
          <span className="tabular-nums">{gnn.toFixed(3)}</span>
          <br />
          <span className="text-ob-muted">{t("result.solubility.disagreement")}:</span>{" "}
          <span className="tabular-nums">{diff.toFixed(3)}</span>{" "}
          <span
            className={`text-xs ${
              diff < 0.5 ? "text-ok" : diff < 1 ? "text-warn" : "text-danger"
            }`}
          >
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
              <div key={item.key} className="rounded-card border border-ob-border bg-ob-bg/40 p-3">
                <p className="text-xs text-ob-faint">{t(item.i18nKey)}</p>
                <p className="mt-1 text-xl font-semibold tabular-nums text-ob-text">
                  {value != null ? item.format(value) : "—"}
                </p>
              </div>
            );
          })}
        </div>
        <div className="mt-3 rounded-card border border-brand/20 bg-brand/5 px-4 py-3 text-xs leading-relaxed text-ob-muted">
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
