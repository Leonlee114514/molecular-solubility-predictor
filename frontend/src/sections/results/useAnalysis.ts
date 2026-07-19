import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { api, ApiError } from "@/api/client";
import type { AnalysisResponse, PredictionResult } from "@/types/api";

/**
 * Fetch the analysis bundle for a prediction. The prose is translated by the
 * BACKEND, so the bundle is refetched whenever the UI language changes.
 */
export function useAnalysis(result: PredictionResult) {
  const { i18n } = useTranslation();
  const lang = i18n.language.startsWith("zh") ? "zh" : "en";
  const [data, setData] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const smiles = result.smiles;
  const pka = result.pka;
  const logs = result.logS_final;

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .analysis(smiles, pka, logs, lang)
      .then((resp) => {
        if (!cancelled) setData(resp);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof ApiError ? err.detail : String(err));
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [smiles, pka, logs, lang]);

  return { data, error, loading };
}
