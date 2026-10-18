import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { ChevronDown, FileUp } from "lucide-react";

import { api, ApiError } from "@/api/client";
import { Alert } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Field, Input } from "@/components/ui/field";
import { cn } from "@/lib/utils";
import type { MoleculeEntry, SearchResponse } from "@/types/api";

interface InputSectionProps {
  selectedSmiles: string;
  onSelect: (smiles: string, name?: string) => void;
}

/* ---------- shared bits ---------- */

/** A clickable search hit: left rule carries the match quality. */
function ResultRow({
  tone,
  onClick,
  children,
}: {
  tone: "ok" | "warn" | "brand";
  onClick: () => void;
  children: ReactNode;
}) {
  const toneClass = {
    ok: "border-ok/70 bg-ok/5 text-ok hover:bg-ok/10",
    warn: "border-warn/70 bg-warn/5 text-warn hover:bg-warn/10",
    brand: "border-brand/60 bg-brand/5 text-brand hover:bg-brand/10",
  }[tone];
  return (
    <button
      type="button"
      onClick={onClick}
      className={`w-full border-l-2 px-3 py-2 text-left text-sm transition-colors duration-fast ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/35 ${toneClass}`}
    >
      {children}
    </button>
  );
}

/* ---------- Card 1: quick select from local library ---------- */

function QuickSelectCard({ onSelect }: InputSectionProps) {
  const { t } = useTranslation();
  const [molecules, setMolecules] = useState<MoleculeEntry[]>([]);
  const [loadError, setLoadError] = useState(false);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    let cancelled = false;
    api
      .listMolecules()
      .then((list) => {
        if (!cancelled) setMolecules(list);
      })
      .catch(() => {
        if (!cancelled) setLoadError(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return molecules;
    const hits = molecules.filter((m) => m.name.toLowerCase().includes(q));
    // Old app behavior: fall back to the full list when nothing matches.
    return hits.length > 0 ? hits : molecules;
  }, [molecules, filter]);

  return (
    <div className="glass-card flex flex-col gap-3 p-4">
      <h3 className="text-sm font-semibold text-ob-text">{t("input.method1.title")}</h3>
      <Field label={t("input.method1.search_label")}>
        <Input
          type="text"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder={t("input.method1.search_placeholder")}
        />
      </Field>
      <div className="flex flex-col gap-1.5">
        <span className="text-xs text-ob-muted">{t("input.method1.select_label")}</span>
        <div className="max-h-44 overflow-y-auto rounded-control border border-ob-rule bg-ob-well/40">
          {loadError && (
            <Alert tone="danger" className="border-l-0 bg-transparent py-2 text-xs">
              {t("app.error.network")}
            </Alert>
          )}
          {!loadError && molecules.length === 0 && (
            <p className="px-2 py-2 text-xs text-ob-faint">…</p>
          )}
          {filtered.map((m) => (
            <button
              key={`${m.name}|${m.smiles}`}
              type="button"
              onClick={() => onSelect(m.smiles, m.name)}
              title={m.smiles}
              className="block w-full px-2 py-1.5 text-left text-sm text-ob-text transition-colors duration-fast ease-out hover:bg-ob-sunken focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-brand/35"
            >
              {m.name}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ---------- Card 2: name search (local + PubChem) ---------- */

function NameSearchCard({ onSelect }: InputSectionProps) {
  const { t, i18n } = useTranslation();
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [searching, setSearching] = useState(false);
  const [pubchemBusy, setPubchemBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const lang = i18n.language.startsWith("zh") ? "zh" : "en";

  const runSearch = async (pubchem: boolean) => {
    const q = query.trim();
    if (!q) return;
    if (pubchem) {
      setPubchemBusy(true);
    } else {
      setSearching(true);
      setResult(null);
    }
    setError(null);
    try {
      const resp = await api.searchMolecules(q, lang, pubchem);
      if (pubchem) {
        // Merge pubchem outcome into existing result view.
        setResult((prev) => ({
          exact: prev?.exact ?? [],
          fuzzy: prev?.fuzzy ?? [],
          pubchem: resp.pubchem,
        }));
      } else {
        setResult(resp);
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : String(err));
    } finally {
      setSearching(false);
      setPubchemBusy(false);
    }
  };

  const hasLocalHits = (result?.exact.length ?? 0) + (result?.fuzzy.length ?? 0) > 0;

  return (
    <div className="glass-card flex flex-col gap-3 p-4">
      <h3 className="text-sm font-semibold text-ob-text">{t("input.method2.title")}</h3>
      <p className="text-xs text-ob-muted">{t("input.method2.desc")}</p>
      <div className="flex items-end gap-2">
        <Field label={t("input.method2.search_btn")} className="min-w-0 flex-1">
          <Input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void runSearch(false);
            }}
            placeholder={t("input.method2.placeholder")}
            invalid={Boolean(error)}
          />
        </Field>
        <Button
          disabled={searching || !query.trim()}
          onClick={() => void runSearch(false)}
        >
          {searching ? "…" : t("input.method2.search_btn")}
        </Button>
      </div>

      {error && <Alert tone="danger" className="text-xs">{error}</Alert>}

      {result && (
        <div className="flex flex-col gap-2 text-sm">
          {result.exact.map((m) => (
            <ResultRow key={`exact-${m.name}`} tone="ok" onClick={() => onSelect(m.smiles, m.name)}>
              {t("input.method2.exact_match", { name: m.name, smiles: m.smiles })}
            </ResultRow>
          ))}
          {result.fuzzy.map((m) => (
            <ResultRow key={`fuzzy-${m.name}`} tone="warn" onClick={() => onSelect(m.smiles, m.name)}>
              {t("input.method2.fuzzy_match", { name: query.trim(), best: m.name })}
            </ResultRow>
          ))}
          {result.pubchem &&
            (result.pubchem.smiles ? (
              <ResultRow
                tone="brand"
                onClick={() => onSelect(result.pubchem!.smiles!, query.trim())}
              >
                {t("input.method2.pubchem_success", { status: result.pubchem.status })}
              </ResultRow>
            ) : (
              <p className="text-xs text-ob-muted">
                {t("input.method2.pubchem_fail", { status: result.pubchem.status })}
              </p>
            ))}
          {!result.pubchem && (
            <Button
              variant="secondary"
              size="sm"
              disabled={pubchemBusy || !query.trim()}
              onClick={() => void runSearch(true)}
              className="self-start"
            >
              {pubchemBusy ? t("input.method2.pubchem_status") : t("input.method2.skip_btn")}
            </Button>
          )}
          {!hasLocalHits && !result.pubchem && (
            <p className="text-xs text-ob-muted">
              {t("input.method2.not_found", { name: query.trim() })}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

/* ---------- Card 3: direct SMILES input ---------- */

function DirectSmilesCard({ selectedSmiles, onSelect }: InputSectionProps) {
  const { t } = useTranslation();
  const [draft, setDraft] = useState(selectedSmiles);

  // Mirror external selections (cards 1/2/4) into the editor, like the old app.
  useEffect(() => {
    setDraft(selectedSmiles);
  }, [selectedSmiles]);

  return (
    <div className="glass-card flex flex-col gap-3 p-4">
      <h3 className="text-sm font-semibold text-ob-text">
        <span className="md:hidden">{t("input.main.title")}</span>
        <span className="hidden md:inline">{t("input.method3.title")}</span>
      </h3>
      <p className="hidden text-xs text-ob-muted md:block">{t("input.method3.desc")}</p>
      <div className="flex items-end gap-2">
        <Field label={t("input.method3.label")} className="min-w-0 flex-1">
          <Input
            type="text"
            value={draft}
            onChange={(e) => {
              setDraft(e.target.value);
              // Keep the prediction input in sync so a pasted SMILES does not
              // need the ✓ press before "predict" will accept it. The ✓ button
              // stays as the explicit "use this" affordance.
              onSelect(e.target.value);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && draft.trim()) onSelect(draft.trim());
            }}
            placeholder="CCO, c1ccccc1, CC(=O)Oc1ccccc1C(=O)O …"
            spellCheck={false}
            className="font-mono"
          />
        </Field>
        <Button
          size="icon"
          disabled={!draft.trim()}
          aria-label={t("input.method3.label")}
          onClick={() => onSelect(draft.trim())}
        >
          ✓
        </Button>
      </div>
    </div>
  );
}

/* ---------- Card 4: file upload ---------- */

const ACCEPTED = ".mol,.sdf,.mol2,.pdb,.xyz";

function FileUploadCard({ onSelect }: InputSectionProps) {
  const { t } = useTranslation();
  const fileRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<{ ok: boolean; text: string } | null>(null);

  const handleFile = async (file: File) => {
    setBusy(true);
    setStatus(null);
    try {
      const parsed = await api.parseFile(file);
      onSelect(parsed.smiles);
      setStatus({
        ok: true,
        text: `${file.name} → ${parsed.formula} (${parsed.mw.toFixed(1)} Da)`,
      });
    } catch (err) {
      const detail = err instanceof ApiError ? err.detail : String(err);
      setStatus({
        ok: false,
        text: `${t("input.method4.parse_fail", { name: file.name })} — ${detail}`,
      });
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  return (
    <div className="glass-card flex flex-col gap-3 p-4">
      <h3 className="text-sm font-semibold text-ob-text">{t("input.method4.title")}</h3>
      <p className="text-xs text-ob-muted">{t("input.method4.desc")}</p>
      <Button
        variant="secondary"
        disabled={busy}
        onClick={() => fileRef.current?.click()}
        className="h-auto w-full flex-col gap-2 border-dashed py-5 font-normal"
      >
        <FileUp className="size-5" aria-hidden="true" />
        <span>{busy ? "…" : t("input.method4.upload_label")}</span>
      </Button>
      <input
        ref={fileRef}
        type="file"
        accept={ACCEPTED}
        className="hidden"
        disabled={busy}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) void handleFile(file);
        }}
      />
      {status && (
        <Alert tone={status.ok ? "ok" : "danger"} className="py-1.5 text-xs">
          {status.text}
        </Alert>
      )}
    </div>
  );
}

/* ---------- Section ---------- */

/**
 * Phones get one job: enter a molecule. SMILES is the primary card and the
 * other three methods collapse behind a single button. From `md` up all four
 * render in the original 2×2 grid — the wrapper `grid` keeps every card the
 * same height as its row neighbour, which a plain div would not.
 */
export default function InputSection(props: InputSectionProps) {
  const { t } = useTranslation();
  const [more, setMore] = useState(false);

  return (
    <section className="grid grid-cols-1 gap-3 md:grid-cols-2 md:gap-4">
      <div className="order-1 grid md:order-3">
        <DirectSmilesCard {...props} />
      </div>

      <div
        className={cn(
          "order-2 grid md:order-1 md:grid",
          more ? "grid" : "hidden",
        )}
      >
        <QuickSelectCard {...props} />
      </div>

      <div
        className={cn(
          "order-3 grid md:order-2 md:grid",
          more ? "grid" : "hidden",
        )}
      >
        <NameSearchCard {...props} />
      </div>

      <div
        className={cn("order-4 grid md:grid", more ? "grid" : "hidden")}
      >
        <FileUploadCard {...props} />
      </div>

      <Button
        variant="secondary"
        aria-expanded={more}
        onClick={() => setMore((v) => !v)}
        className="order-5 w-full border-dashed md:hidden"
      >
        <ChevronDown
          aria-hidden="true"
          className={cn(
            "size-4 transition-transform duration-fast ease-out",
            more && "rotate-180",
          )}
        />
        {more ? t("input.more.hide") : t("input.more.toggle")}
      </Button>
    </section>
  );
}
