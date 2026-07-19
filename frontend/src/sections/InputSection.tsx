import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { api, ApiError } from "@/api/client";
import type { MoleculeEntry, SearchResponse } from "@/types/api";

interface InputSectionProps {
  selectedSmiles: string;
  onSelect: (smiles: string, name?: string) => void;
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
      <label className="flex flex-col gap-1 text-xs text-ob-muted">
        {t("input.method1.search_label")}
        <input
          type="text"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder={t("input.method1.search_placeholder")}
          className="rounded-lg border border-ob-border bg-ob-bg/60 px-3 py-2 text-sm text-ob-text outline-none placeholder:text-ob-faint focus:border-nebula"
        />
      </label>
      <label className="flex flex-col gap-1 text-xs text-ob-muted">
        {t("input.method1.select_label")}
        <div className="max-h-44 overflow-y-auto rounded-lg border border-ob-border bg-ob-bg/60">
          {loadError && (
            <p className="px-3 py-2 text-xs text-red-400">{t("app.error.network")}</p>
          )}
          {!loadError && molecules.length === 0 && (
            <p className="px-3 py-2 text-xs text-ob-faint">…</p>
          )}
          {filtered.map((m) => (
            <button
              key={`${m.name}|${m.smiles}`}
              type="button"
              onClick={() => onSelect(m.smiles, m.name)}
              className="block w-full px-3 py-1.5 text-left text-sm text-ob-text transition-colors hover:bg-nebula/20"
              title={m.smiles}
            >
              {m.name}
            </button>
          ))}
        </div>
      </label>
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
      <div className="flex gap-2">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void runSearch(false);
          }}
          placeholder={t("input.method2.placeholder")}
          className="min-w-0 flex-1 rounded-lg border border-ob-border bg-ob-bg/60 px-3 py-2 text-sm text-ob-text outline-none placeholder:text-ob-faint focus:border-nebula"
        />
        <button
          type="button"
          disabled={searching || !query.trim()}
          onClick={() => void runSearch(false)}
          className="shrink-0 rounded-lg bg-nebula px-4 py-2 text-sm text-white transition-opacity hover:opacity-90 disabled:opacity-40"
        >
          {searching ? "…" : t("input.method2.search_btn")}
        </button>
      </div>

      {error && <p className="text-xs text-red-400">{error}</p>}

      {result && (
        <div className="flex flex-col gap-2 text-sm">
          {result.exact.map((m) => (
            <button
              key={`exact-${m.name}`}
              type="button"
              onClick={() => onSelect(m.smiles, m.name)}
              className="rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-left text-emerald-300 transition-colors hover:bg-emerald-500/20"
            >
              {t("input.method2.exact_match", { name: m.name, smiles: m.smiles })}
            </button>
          ))}
          {result.fuzzy.map((m) => (
            <button
              key={`fuzzy-${m.name}`}
              type="button"
              onClick={() => onSelect(m.smiles, m.name)}
              className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-left text-amber-300 transition-colors hover:bg-amber-500/20"
            >
              {t("input.method2.fuzzy_match", { name: query.trim(), best: m.name })}
            </button>
          ))}
          {result.pubchem &&
            (result.pubchem.smiles ? (
              <button
                type="button"
                onClick={() => onSelect(result.pubchem!.smiles!, query.trim())}
                className="rounded-lg border border-cyan-500/40 bg-cyan-500/10 px-3 py-2 text-left text-cyan-300 transition-colors hover:bg-cyan-500/20"
              >
                {t("input.method2.pubchem_success", { status: result.pubchem.status })}
              </button>
            ) : (
              <p className="text-xs text-ob-muted">
                {t("input.method2.pubchem_fail", { status: result.pubchem.status })}
              </p>
            ))}
          {!result.pubchem && (
            <button
              type="button"
              disabled={pubchemBusy || !query.trim()}
              onClick={() => void runSearch(true)}
              className="self-start rounded-lg border border-ob-border bg-ob-surface/70 px-3 py-1.5 text-xs text-ob-muted transition-colors hover:border-nebula/60 hover:text-ob-text disabled:opacity-40"
            >
              {pubchemBusy ? t("input.method2.pubchem_status") : t("input.method2.skip_btn")}
            </button>
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
      <h3 className="text-sm font-semibold text-ob-text">{t("input.method3.title")}</h3>
      <p className="text-xs text-ob-muted">{t("input.method3.desc")}</p>
      <label className="flex flex-col gap-1 text-xs text-ob-muted">
        {t("input.method3.label")}
        <div className="flex gap-2">
          <input
            type="text"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && draft.trim()) onSelect(draft.trim());
            }}
            placeholder="CCO, c1ccccc1, CC(=O)Oc1ccccc1C(=O)O …"
            spellCheck={false}
            className="min-w-0 flex-1 rounded-lg border border-ob-border bg-ob-bg/60 px-3 py-2 font-mono text-sm text-ob-text outline-none placeholder:text-ob-faint focus:border-nebula"
          />
          <button
            type="button"
            disabled={!draft.trim()}
            aria-label={t("input.method3.label")}
            onClick={() => onSelect(draft.trim())}
            className="shrink-0 rounded-lg bg-nebula px-4 py-2 text-sm text-white transition-opacity hover:opacity-90 disabled:opacity-40"
          >
            ✓
          </button>
        </div>
      </label>
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
      <label className="flex cursor-pointer flex-col items-center gap-2 rounded-lg border border-dashed border-ob-border bg-ob-bg/40 px-3 py-5 text-center text-sm text-ob-muted transition-colors hover:border-nebula/60 hover:text-ob-text">
        <span className="text-xl">📄</span>
        <span>{busy ? "…" : t("input.method4.upload_label")}</span>
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
      </label>
      {status && (
        <p className={`text-xs ${status.ok ? "text-emerald-400" : "text-red-400"}`}>
          {status.text}
        </p>
      )}
    </div>
  );
}

/* ---------- Section ---------- */

export default function InputSection(props: InputSectionProps) {
  return (
    <section className="grid grid-cols-1 gap-4 md:grid-cols-2">
      <QuickSelectCard {...props} />
      <NameSearchCard {...props} />
      <DirectSmilesCard {...props} />
      <FileUploadCard {...props} />
    </section>
  );
}
