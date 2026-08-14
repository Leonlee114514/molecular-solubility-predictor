"""Deeper audit:
  (B2) en values containing CJK punctuation (【】（）：；等) or Han chars.
  (M)  zh values whose Latin word tokens are NOT technical/known terms
       (likely leftover English that should be localized).
"""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "core" / "i18n.py"

_CJK_HAN = re.compile(r"[一-鿿]")
_CJK_PUNCT = re.compile(r"[，。；：？!！“”‘’（）【】《》·—…、]")
_WORD = re.compile(r"[A-Za-z]{3,}")

# Technical / proper-noun / molecule-name terms allowed to appear in zh values.
KNOWN = {
    "SMILES", "RDKit", "Streamlit", "PubChem", "Kimi", "Moonshot", "Lipinski",
    "Aspirin", "Ibuprofen", "Metformin", "Caffeine", "Bickerton", "Ertl",
    "Schuffenhauer", "Lovering", "Christopher", "Morgan", "LogP", "TPSA",
    "QED", "SAscore", "Fsp", "GNN", "SHAP", "OOD", "ADME", "BBB", "CYP",
    "RF", "AI", "KIMI", "API", "pH", "pKa", "logS", "MolWt", "DA", "Protac",
    "PROTAC", "Ames", "hERG", "Michael", "Ligand", "Vd", "MW",
}


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

    def lit(x):
        return x.value if isinstance(x, ast.Constant) and isinstance(x.value, str) else None

    b2 = []
    m = []
    for key_node, val_node in zip(all_dict.keys, all_dict.values):
        key = lit(key_node)
        if not key or not isinstance(val_node, ast.Dict):
            continue
        vals = {lit(k): (lit(v) or "") for k, v in zip(val_node.keys, val_node.values)}
        zh, en = vals.get("zh", ""), vals.get("en", "")

        if en and (_CJK_HAN.search(en) or _CJK_PUNCT.search(en)):
            b2.append((key, zh, en))

        if zh and _CJK_HAN.search(zh):
            for w in _WORD.findall(zh):
                # skip when followed/embedded inside backticks (code) or is a known term
                if w in KNOWN:
                    continue
                # Skip words inside `...` code spans
                if f"`{w}" in zh or f"{w}`" in zh:
                    continue
                m.append((key, zh, w))

    print(f"B2 (en with CJK punctuation/Han): {len(b2)}")
    for key, zh, en in b2:
        print(f"  {key}\n    zh: {zh!r}\n    en: {en!r}")

    print(f"\nM (zh with unknown Latin word): {len(m)}")
    for key, zh, w in sorted(set(m)):
        print(f"  {key}: ...{w}... in {zh!r}")


if __name__ == "__main__":
    main()
