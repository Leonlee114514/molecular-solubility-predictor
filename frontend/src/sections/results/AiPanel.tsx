import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import ReactMarkdown from "react-markdown";

import { api, ApiError } from "@/api/client";
import type { PredictionResult } from "@/types/api";

interface AiPanelProps {
  result: PredictionResult;
}

export default function AiPanel({ result }: AiPanelProps) {
  const { t, i18n } = useTranslation();
  const lang = i18n.language.startsWith("zh") ? "zh" : "en";

  const [markdown, setMarkdown] = useState<string | null>(null);
  const [generatedLang, setGeneratedLang] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [noKey, setNoKey] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generate = useCallback(
    async (targetLang: string) => {
      setBusy(true);
      setError(null);
      setNoKey(false);
      try {
        const resp = await api.explain({
          smiles: result.smiles,
          prediction: result.logS_final,
          features: result.features,
          shap_features: result.shap_names,
          shap_values: result.shap_values,
          pka_value: result.pka,
          pka_type: result.pka_kind,
          lang: targetLang,
        });
        setMarkdown(resp.markdown);
        setGeneratedLang(targetLang);
      } catch (err) {
        if (err instanceof ApiError && err.status === 503) {
          setNoKey(true);
        } else {
          setError(
            t("ai.error.generic", {
              err: err instanceof ApiError ? err.detail : String(err),
            }),
          );
        }
      } finally {
        setBusy(false);
      }
    },
    [result, t],
  );

  // Prose is translated by the backend: regenerate when the UI language
  // switches after an explanation was already produced.
  useEffect(() => {
    if (markdown && generatedLang && generatedLang !== lang && !busy) {
      void generate(lang);
    }
  }, [lang, markdown, generatedLang, busy, generate]);

  return (
    <div className="glass-card flex flex-col gap-3 p-4">
      <h3 className="text-sm font-semibold text-ob-text">{t("result.ai.title")}</h3>

      {markdown ? (
        <>
          <div className="md-body text-sm">
            <ReactMarkdown>{markdown}</ReactMarkdown>
          </div>
          <button
            type="button"
            onClick={() => {
              setMarkdown(null);
              setGeneratedLang(null);
            }}
            className="self-start rounded-lg border border-ob-border bg-ob-surface/70 px-3 py-1.5 text-xs text-ob-muted transition-colors hover:border-nebula/60 hover:text-ob-text"
          >
            {t("result.ai.clear_btn")}
          </button>
        </>
      ) : (
        <>
          <p className="text-xs text-ob-faint">{t("result.ai.need_manual")}</p>
          {noKey && (
            <div className="rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-300">
              {t("ai.error.no_key")}
            </div>
          )}
          {error && (
            <div className="rounded-xl border border-red-500/50 bg-red-500/10 px-4 py-3 text-sm text-red-300">
              {error}
            </div>
          )}
          <button
            type="button"
            disabled={busy}
            onClick={() => void generate(lang)}
            className="w-full rounded-xl bg-gradient-to-r from-nebula to-orbit py-2.5 text-sm font-semibold text-white shadow-glow transition-opacity hover:opacity-90 disabled:opacity-40"
          >
            {busy ? t("result.ai.generating") : `✨ ${t("result.ai.generate_btn")}`}
          </button>
        </>
      )}
    </div>
  );
}
