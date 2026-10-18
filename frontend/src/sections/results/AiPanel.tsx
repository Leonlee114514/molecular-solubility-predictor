import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import ReactMarkdown from "react-markdown";

import { api, ApiError } from "@/api/client";
import { Alert } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
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
          <Button
            variant="secondary"
            size="sm"
            onClick={() => {
              setMarkdown(null);
              setGeneratedLang(null);
            }}
            className="self-start"
          >
            {t("result.ai.clear_btn")}
          </Button>
        </>
      ) : (
        <>
          <p className="text-xs text-ob-faint">{t("result.ai.need_manual")}</p>
          {noKey && <Alert tone="warn">{t("ai.error.no_key")}</Alert>}
          {error && <Alert tone="danger">{error}</Alert>}
          <Button
            loading={busy}
            onClick={() => void generate(lang)}
            className="w-full"
          >
            {busy ? t("result.ai.generating") : t("result.ai.generate_btn")}
          </Button>
        </>
      )}
    </div>
  );
}
