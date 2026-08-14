"""Audit language purity of core/i18n.py _ALL (source of truth).

Flags:
  (A) zh values containing no CJK characters  -> English-only zh (needs Chinese),
      unless the value is a technical acronym / symbol.
  (B) en values containing CJK characters     -> Chinese leak in en (needs cleanup).

Chart-bilingual zh values (contain both CJK and Latin) are NOT flagged as
violations; they fall under the "charts may be bilingual" exception.
"""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "core" / "i18n.py"

_CJK = re.compile(r"[一-鿿]")
_HAS_LATIN = re.compile(r"[A-Za-z0-9]")

# Technical terms that are legitimately written in English/Latin in BOTH languages.
ACRONYMS = {
    "SMILES", "pKa", "pka", "logS", "LogP", "TPSA", "RF", "GNN", "SHAP",
    "QED", "SAscore", "Fsp", "ADME", "Tox", "ADMET", "MW", "Mol", "Da",
    "Kimi", "PubChem", "Lipinski", "RDKit", "MorganFP", "OOD", "SHAP",
}


def contains_cjk(s):
    return bool(_CJK.search(s))


def only_acronyms(s):
    """True if the string is purely technical tokens + punctuation (no words)."""
    # Remove common punctuation/symbols used inside labels.
    tokens = re.split(r"[\s（）()【】\[\]:：·|/\\,.+\-*=''\"\"!?;；&]", s)
    tokens = [t for t in tokens if t]
    if not tokens:
        return False
    for t in tokens:
        # Allow numeric floats like 0.45, 0.55, 500, 1.5
        if re.fullmatch(r"[\d.]+", t):
            continue
        if t in ACRONYMS or t.upper() in ACRONYMS:
            continue
        return False
    return True


def main():
    src = SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    all_dict = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_ALL":
                    all_dict = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "_ALL":
                all_dict = node.value
    if all_dict is None or not isinstance(all_dict, ast.Dict):
        print("ERROR: could not find _ALL dict in core/i18n.py")
        sys.exit(1)

    def lit(x):
        if isinstance(x, ast.Constant) and isinstance(x.value, str):
            return x.value
        return None

    rows = []
    for key_node, val_node in zip(all_dict.keys, all_dict.values):
        key = lit(key_node)
        if not key or not isinstance(val_node, ast.Dict):
            continue
        vals = {}
        for k, v in zip(val_node.keys, val_node.values):
            kk = lit(k)
            vv = lit(v)
            if kk:
                vals[kk] = vv or ""
        zh = vals.get("zh", "")
        en = vals.get("en", "")

        # (A) zh is English-only
        if zh and not contains_cjk(zh) and not only_acronyms(zh):
            rows.append(("A", key, zh, en))
        # (B) en contains CJK
        if en and contains_cjk(en):
            rows.append(("B", key, zh, en))

    rows.sort(key=lambda r: (r[0], r[1]))
    print(f"Found {len(rows)} flagged entries (A=zh English-only, B=en has CJK)\n")
    for kind, key, zh, en in rows:
        flag = "A" if kind == "A" else "B"
        print(f"[{flag}] {key}")
        print(f"   zh: {zh!r}")
        print(f"   en: {en!r}")


if __name__ == "__main__":
    main()
