import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { api, ApiError } from "@/api/client";
import { clearHistory, loadHistory, pushHistory, type HistoryEntry } from "@/lib/history";
import BatchSection from "@/sections/BatchSection";
import HeaderSection from "@/sections/HeaderSection";
import HistorySection from "@/sections/HistorySection";
import InputSection from "@/sections/InputSection";
import ModelSelector from "@/sections/ModelSelector";
import ResultsTabs from "@/sections/results/ResultsTabs";
import type { ModelMode, PredictionResult } from "@/types/api";

export default function Home() {
  const { t, i18n } = useTranslation();
  const [smiles, setSmiles] = useState("");
  const [selectedName, setSelectedName] = useState("");
  const [mode, setMode] = useState<ModelMode>("auto");
  const [gnnAvailable, setGnnAvailable] = useState(true);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState<HistoryEntry[]>(() => loadHistory());
  // Bumped by the predict button so the effect below runs even when the
  // inputs (smiles/mode/lang) are unchanged since the last prediction.
  const [requestSeq, setRequestSeq] = useState(0);

  useEffect(() => {
    api
      .health()
      .then((h) => setGnnAvailable(h.models.gnn.available))
      .catch(() => setGnnAvailable(false));
  }, []);

  const lang = i18n.language.startsWith("zh") ? "zh" : "en";
  const langRef = useRef(lang);
  useEffect(() => {
    if (langRef.current !== lang && result) {
      // Language switched with results on screen: re-predict so any
      // backend-translated strings refresh (analysis/AI panels refetch
      // themselves; this keeps the prediction bundle consistent).
      setRequestSeq((n) => n + 1);
    }
    langRef.current = lang;
  }, [lang, result]);

  const handleSelect = useCallback((s: string, name?: string) => {
    setSmiles(s);
    setSelectedName(name ?? "");
  }, []);

  const runPrediction = useCallback(async () => {
    const query = smiles.trim();
    if (!query) {
      setError(t("app.predict.empty"));
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await api.predict(query, mode, lang);
      setResult(res);
      setHistory(pushHistory(res, selectedName));
    } catch (err) {
      setResult(null);
      if (err instanceof ApiError && err.status === 400) {
        setError(t("app.error.invalid_smiles", { smiles: query }));
      } else if (err instanceof ApiError && err.status === 0) {
        setError(t("app.error.network"));
      } else {
        setError(err instanceof ApiError ? err.detail : String(err));
      }
    } finally {
      setLoading(false);
    }
  }, [smiles, selectedName, mode, lang, t]);

  useEffect(() => {
    if (requestSeq > 0) void runPrediction();
  }, [requestSeq, runPrediction]);

  const handleReuse = useCallback((entry: HistoryEntry) => {
    setSmiles(entry.smiles);
    setSelectedName(entry.name);
    setResult(entry.result);
    setError(null);
  }, []);

  const handleClearHistory = useCallback(() => {
    clearHistory();
    setHistory([]);
  }, []);

  return (
    <div className="relative min-h-screen bg-ob-bg text-ob-text">
      <main className="relative z-10 mx-auto flex max-w-6xl flex-col gap-6 px-4 pb-16">
        <HeaderSection />
        <InputSection selectedSmiles={smiles} onSelect={handleSelect} />
        <HistorySection entries={history} onReuse={handleReuse} onClear={handleClearHistory} />
        <ModelSelector mode={mode} onChange={setMode} gnnAvailable={gnnAvailable} />

        <div className="flex flex-col items-center gap-2">
          <button
            type="button"
            disabled={loading}
            onClick={() => setRequestSeq((n) => n + 1)}
            className="rounded-xl bg-nebula px-8 py-3 text-base font-semibold text-white shadow-glow transition-opacity hover:opacity-90 disabled:opacity-50"
          >
            {loading ? t("app.predict.status") : t("app.predict_btn")}
          </button>
          {error && <p className="max-w-xl text-center text-sm text-red-400">{error}</p>}
        </div>

        {result && <ResultsTabs result={result} />}

        <BatchSection mode={mode} />
      </main>
    </div>
  );
}
