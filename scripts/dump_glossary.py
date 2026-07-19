"""Extract GLOSS_ENTRIES from assets/scripts.py into frontend/src/data/glossary.json.

Each entry: {keys, en, cn, def, defEn} — bilingual chemistry glossary used by
the React <GlossaryText> component. Run from the repo root:

    venv/Scripts/python.exe scripts/dump_glossary.py
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "assets" / "scripts.py"
OUT = ROOT / "frontend" / "src" / "data" / "glossary.json"


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    match = re.search(r"var GLOSS_ENTRIES = \[(.*?)\];", text, re.DOTALL)
    if not match:
        raise SystemExit("GLOSS_ENTRIES array not found in assets/scripts.py")
    body = match.group(1)

    entries = []
    # Each entry is a JS object literal { keys: [...], en: "...", ... }.
    for obj in re.finditer(r"\{(.*?)\}", body, re.DOTALL):
        chunk = obj.group(1)
        entry = {}
        for field in ("keys", "en", "cn", "def", "defEn"):
            if field == "keys":
                m = re.search(r"keys:\s*\[(.*?)\]", chunk, re.DOTALL)
                if not m:
                    continue
                entry["keys"] = [
                    s.strip() for s in re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
                ]
            else:
                m = re.search(rf'{field}:\s*"((?:[^"\\]|\\.)*)"', chunk)
                if m:
                    entry[field] = m.group(1)
        if entry.get("keys") and entry.get("def"):
            entries.append(entry)

    if len(entries) < 30:
        raise SystemExit(f"Suspiciously few entries extracted: {len(entries)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUT} ({len(entries)} entries)")


if __name__ == "__main__":
    main()
