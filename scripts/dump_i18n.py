"""One-off: dump core/i18n.py's _ALL dict to frontend/src/i18n/{zh,en}.json.

Run from the project root:
    venv/Scripts/python.exe scripts/dump_i18n.py

Flat dot-keys, both languages. Re-run whenever core/i18n.py gains new keys.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.i18n import _ALL  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "src", "i18n")


def main():
    zh = {}
    en = {}
    for key, entry in _ALL.items():
        zh[key] = entry.get("zh") or entry.get("en") or key
        en[key] = entry.get("en") or entry.get("zh") or key

    os.makedirs(OUT_DIR, exist_ok=True)
    for lang, data in (("zh", zh), ("en", en)):
        path = os.path.join(OUT_DIR, f"{lang}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"wrote {path} ({len(data)} keys)")


if __name__ == "__main__":
    main()
