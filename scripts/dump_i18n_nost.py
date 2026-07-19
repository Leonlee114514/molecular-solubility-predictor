"""Managed-python fallback for scripts/dump_i18n.py.

The project venv is the canonical runner; this variant stubs out `streamlit`
(core/i18n.py imports it at module level but only uses it inside functions) so
the dump can run under any Python. Output is byte-identical.
"""

import json
import os
import sys
import types

sys.modules.setdefault("streamlit", types.ModuleType("streamlit"))
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
