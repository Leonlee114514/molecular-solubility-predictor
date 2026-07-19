import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { createPortal } from "react-dom";
import { useTranslation } from "react-i18next";

import glossaryData from "@/data/glossary.json";

interface GlossaryEntry {
  keys: string[];
  en: string;
  cn: string;
  def: string;
  defEn: string;
}

const ENTRIES = glossaryData as GlossaryEntry[];

// key -> entry, keys sorted longest-first so long terms win at match time.
const KEY_TO_ENTRY = new Map<string, GlossaryEntry>();
for (const entry of ENTRIES) {
  for (const key of entry.keys) {
    if (!KEY_TO_ENTRY.has(key)) KEY_TO_ENTRY.set(key, entry);
  }
}
const SORTED_KEYS = [...KEY_TO_ENTRY.keys()].sort((a, b) => b.length - a.length);
const TERM_PATTERN = new RegExp(
  `(${SORTED_KEYS.map((k) => k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`,
  "g",
);

/* ---------- rich-text parsing (bold / <b> / <br> / entities) ---------- */

type Segment = { kind: "text" | "bold" | "br"; text: string };

const ENTITY_MAP: Record<string, string> = {
  "&bull;": "•",
  "&amp;": "&",
  "&lt;": "<",
  "&gt;": ">",
  "&sup3;": "³",
  "&sup2;": "²",
  "&quot;": '"',
  "&#39;": "'",
  "&nbsp;": " ",
};

function decodeEntities(text: string): string {
  return text.replace(/&[a-z0-9#]+;/g, (m) => ENTITY_MAP[m] ?? m);
}

/** Parse the old app's lightweight markup into segments. */
function parseRichText(raw: string): Segment[] {
  const text = decodeEntities(raw);
  // Split by explicit line-break markup first.
  const lines = text.split(/<br\s*\/?>|\n/g);
  const segments: Segment[] = [];
  lines.forEach((line, li) => {
    if (li > 0) segments.push({ kind: "br", text: "" });
    // **bold** and <b>bold</b> in one pass.
    const pattern = /\*\*([^*]+)\*\*|<b>([^<]+)<\/b>/g;
    let last = 0;
    let m: RegExpExecArray | null;
    while ((m = pattern.exec(line)) !== null) {
      if (m.index > last) segments.push({ kind: "text", text: line.slice(last, m.index) });
      segments.push({ kind: "bold", text: m[1] ?? m[2] ?? "" });
      last = m.index + m[0].length;
    }
    if (last < line.length) segments.push({ kind: "text", text: line.slice(last) });
  });
  return segments;
}

/* ---------- glossary linkification ---------- */

function linkify(text: string, keyPrefix: string, onTerm: (entry: GlossaryEntry, x: number, y: number) => void) {
  const parts = text.split(TERM_PATTERN);
  if (parts.length === 1) return text;
  return parts.map((part, i) => {
    const entry = KEY_TO_ENTRY.get(part);
    if (!entry) return part;
    return (
      <button
        key={`${keyPrefix}-${i}`}
        type="button"
        className="gloss-term"
        onClick={(e) => {
          e.stopPropagation();
          onTerm(entry, e.clientX, e.clientY);
        }}
      >
        {part}
      </button>
    );
  });
}

/* ---------- popup ---------- */

interface PopupState {
  entry: GlossaryEntry;
  x: number;
  y: number;
}

function GlossaryPopup({ popup, onClose }: { popup: PopupState; onClose: () => void }) {
  const { i18n } = useTranslation();
  const zh = i18n.language.startsWith("zh");
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Clamp inside the viewport.
  const style: CSSProperties = { position: "fixed", zIndex: 70 };
  const width = 340;
  const heightGuess = 220;
  style.left = Math.min(Math.max(8, popup.x + 14), window.innerWidth - width - 8);
  style.top = Math.min(Math.max(8, popup.y - 10), window.innerHeight - heightGuess - 8);
  style.width = width;

  return createPortal(
    <>
      <div className="fixed inset-0 z-[60] bg-black/30" onClick={onClose} />
      <div
        ref={ref}
        style={style}
        className="glass-card max-h-[60vh] overflow-y-auto p-4 shadow-glow"
        role="dialog"
        aria-label={popup.entry.en}
      >
        <div className="text-sm font-semibold text-nebula-light">{popup.entry.en}</div>
        <div className="mb-2 text-xs text-ob-muted">{popup.entry.cn}</div>
        <p className="text-sm leading-relaxed text-ob-text">
          {zh ? popup.entry.def : popup.entry.defEn}
        </p>
        <p className="mt-2 border-t border-ob-border pt-2 text-xs leading-relaxed text-ob-faint">
          {zh ? popup.entry.defEn : popup.entry.def}
        </p>
      </div>
    </>,
    document.body,
  );
}

/* ---------- main component ---------- */

interface GlossaryTextProps {
  text: string;
  className?: string;
  /** Set false to render plain rich text without term links. */
  glossary?: boolean;
}

export default function GlossaryText({ text, className, glossary = true }: GlossaryTextProps) {
  const [popup, setPopup] = useState<PopupState | null>(null);
  const segments = useMemo(() => parseRichText(text), [text]);

  const onTerm = useCallback((entry: GlossaryEntry, x: number, y: number) => {
    setPopup({ entry, x, y });
  }, []);

  return (
    <span className={className}>
      {segments.map((seg, i) => {
        if (seg.kind === "br") return <br key={i} />;
        const content = glossary ? linkify(seg.text, `${i}`, onTerm) : seg.text;
        if (seg.kind === "bold") {
          return (
            <strong key={i} className="font-semibold text-ob-text">
              {content}
            </strong>
          );
        }
        return <span key={i}>{content}</span>;
      })}
      {popup && <GlossaryPopup popup={popup} onClose={() => setPopup(null)} />}
    </span>
  );
}
