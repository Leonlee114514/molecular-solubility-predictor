import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { createViewer } from "3dmol";
import type { GLViewer } from "3dmol";
import { Pause, Play } from "lucide-react";

import { api } from "@/api/client";
import { Button } from "@/components/ui/button";

interface Molecule3DProps {
  smiles: string;
}

/**
 * 3dmol's `backgroundColor` wants a concrete colour string, not a CSS custom
 * property, so the token is resolved through the browser once per mount.
 */
function paperColor(): string {
  const probe = document.createElement("span");
  probe.style.cssText =
    "position:absolute;visibility:hidden;background:hsl(var(--ob-surface))";
  document.body.appendChild(probe);
  const value = getComputedStyle(probe).backgroundColor;
  probe.remove();
  return value;
}

export default function Molecule3D({ smiles }: Molecule3DProps) {
  const { t } = useTranslation();
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<GLViewer | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [spinning, setSpinning] = useState(false);

  useEffect(() => {
    let disposed = false;
    setLoading(true);
    setError(null);
    setSpinning(false);

    api
      .mol3d(smiles)
      .then(({ molblock }) => {
        if (disposed || !containerRef.current) return;
        containerRef.current.innerHTML = "";
        const viewer = createViewer(containerRef.current, {
          backgroundColor: paperColor(),
        });
        viewer.addModel(molblock, "mol");
        // Ball-and-stick, same style as the old py3Dmol rendering.
        viewer.setStyle({}, { stick: { radius: 0.18 }, sphere: { scale: 0.3 } });
        viewer.zoomTo();
        viewer.render();
        viewerRef.current = viewer;
        setLoading(false);
      })
      .catch((err: unknown) => {
        if (disposed) return;
        setError(err instanceof Error ? err.message : String(err));
        setLoading(false);
      });

    return () => {
      disposed = true;
      viewerRef.current = null;
    };
  }, [smiles]);

  // Keep the canvas sized to its container.
  useEffect(() => {
    const el = containerRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => {
      viewerRef.current?.resize();
      viewerRef.current?.render();
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const toggleSpin = () => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    const next = !spinning;
    viewer.spin(next ? "y" : false);
    viewer.render();
    setSpinning(next);
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="relative">
        <div
          ref={containerRef}
          className="h-[420px] w-full overflow-hidden rounded-panel border border-ob-border"
          style={{ background: "hsl(var(--ob-surface))" }}
        />
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center text-sm text-ob-muted">
            {t("common.loading")}
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center px-4 text-center text-sm text-ob-muted">
            {t("result.preview.model3d_fail")}
          </div>
        )}
      </div>
      <Button
        variant="secondary"
        size="sm"
        onClick={toggleSpin}
        disabled={loading || !!error}
        className="self-start"
      >
        {spinning ? (
          <Pause className="size-3.5" aria-hidden="true" />
        ) : (
          <Play className="size-3.5" aria-hidden="true" />
        )}
        {spinning
          ? t("result.preview.spin_stop")
          : t("result.preview.spin_start")}
      </Button>
    </div>
  );
}
