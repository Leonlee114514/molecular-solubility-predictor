import { lazy, Suspense, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { api } from "@/api/client";
import type { MolInfo, PredictionResult } from "@/types/api";

// 3dmol is ~1.7 MB; split it out of the initial bundle.
const Molecule3D = lazy(() => import("@/components/Molecule3D"));

interface PreviewPanelProps {
  result: PredictionResult;
}

export default function PreviewPanel({ result }: PreviewPanelProps) {
  const { t } = useTranslation();
  const [info, setInfo] = useState<MolInfo | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .molInfo(result.smiles)
      .then((d) => {
        if (!cancelled) setInfo(d);
      })
      .catch(() => {
        if (!cancelled) setInfo(null);
      });
    return () => {
      cancelled = true;
    };
  }, [result.smiles]);

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {/* 2D structure */}
        <div className="glass-card flex flex-col items-center gap-2 p-4">
          <h3 className="self-start text-sm font-semibold text-ob-muted">
            {t("result.preview.card_title")}
          </h3>
          <img
            src={api.mol2dUrl(result.smiles)}
            alt={t("result.preview.img_caption")}
            className="w-full max-w-[460px] rounded-lg"
          />
          <p className="text-xs text-ob-faint">{t("result.preview.img_caption")}</p>
        </div>

        {/* Identity */}
        <div className="glass-card flex flex-col gap-4 p-5">
          <div>
            <p className="text-xs text-ob-faint">{t("result.preview.formula")}</p>
            <p className="mt-1 text-3xl font-bold text-ob-text">
              {info?.formula ?? "…"}
            </p>
          </div>
          <div>
            <p className="text-xs text-ob-faint">{t("result.preview.mol_weight")}</p>
            <p className="mt-1 text-2xl font-semibold tabular-nums text-cyan-300">
              {info ? `${info.mw.toFixed(1)} Da` : "…"}
            </p>
          </div>
          <div>
            <p className="text-xs text-ob-faint">{t("result.preview.smiles_label")}</p>
            <code className="mt-1 block break-all rounded-lg border border-ob-border bg-ob-bg/60 px-3 py-2 font-mono text-xs text-nebula-light">
              {result.smiles}
            </code>
          </div>
        </div>
      </div>

      {/* 3D model */}
      <div className="glass-card flex flex-col gap-2 p-4">
        <h3 className="text-sm font-semibold text-ob-muted">
          {t("result.preview.model3d")}
        </h3>
        <Suspense
          fallback={
            <div className="flex h-[420px] items-center justify-center rounded-lg bg-ob-surface/40 text-sm text-ob-faint">
              {t("result.preview.model3d_loading")}
            </div>
          }
        >
          <Molecule3D smiles={result.smiles} />
        </Suspense>
      </div>
    </div>
  );
}
