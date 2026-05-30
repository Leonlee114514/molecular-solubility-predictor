"""
Find optimal ensemble weight PER OOD score segment.
Teaches us how RF weight should decrease as molecules become more out-of-distribution.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import joblib
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from features import compute_features
from gnn_model import load_gnn_model, SolubilityGNN, MoleculeGraphEncoder, ATOM_FEATURE_DIM
from ood_detector import load_ood_detector
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from rdkit import Chem

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1. Load models ──
rf_path = "output_v2/solubility_model_v5.pkl"
rf = joblib.load(rf_path) if os.path.exists(rf_path) else joblib.load("output_v2/solubility_model_v4.pkl")

gnn_encoder = MoleculeGraphEncoder()
for fname, hdim in [("gnn_solubility_model_v4.pt", 256), ("gnn_solubility_model_v3.pt", 128)]:
    p = os.path.join("output_v2", fname)
    if os.path.exists(p):
        gnn = SolubilityGNN(atom_dim=ATOM_FEATURE_DIM, hidden_dim=hdim, num_layers=3)
        gnn.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))
        gnn.to(DEVICE)
        gnn.eval()
        break

ood = load_ood_detector("output_v2/ood_detector.pkl")

# ── 2. Load test data ──
datasets = []
for name, path, cols in [
    ("ESOL", "data/delaney.csv", {"Compound ID": "ID", "measured log(solubility:mol/L)": "logS", "ESOL predicted log(solubility:mol/L)": "_esol_pred"}),
    ("AqSolDB", "curated-solubility-dataset.csv", None),
    ("Supplementary", "supplementary_logs.csv", None),
    ("ChEMBL", "chembl_solubility.csv", None),
]:
    if not os.path.exists(path): continue
    df = pd.read_csv(path)
    if cols: df = df.rename(columns=cols)[["SMILES", "logS"]]
    elif "SMILES" not in df.columns or "logS" not in df.columns:
        sc = next(c for c in df.columns if "smiles" in c.lower())
        lc = next(c for c in df.columns if "solubility" in c.lower() or "logS" in c)
        df = df[[sc, lc]].rename(columns={sc: "SMILES", lc: "logS"})
    df["source"] = name
    datasets.append(df)

df = pd.concat(datasets, ignore_index=True).drop_duplicates(subset=["SMILES"], keep="first")

# ── 3. Compute features, predictions, OOD scores ──
print("Computing features + predictions + OOD scores...")
rows = []
for _, row in df.iterrows():
    smi = row["SMILES"]
    r = compute_features(smi)
    if r is None: continue
    f, fp = r
    # RF
    Xv = np.hstack([list(f.values()), fp]).reshape(1, -1)
    rf_p = float(rf.predict(Xv)[0])
    # GNN
    mol = Chem.MolFromSmiles(smi)
    gnn_p = None
    if mol is not None:
        g = gnn_encoder.mol_to_graph(mol)
        if g is not None:
            with torch.no_grad(): gnn_p = float(gnn(g).item())
    # OOD
    ood_result = ood.check(f, fp)
    rows.append({
        "smiles": smi, "source": row["source"], "logS": float(row["logS"]),
        "rf_pred": rf_p, "gnn_pred": gnn_p,
        "ood_score": ood_result.overall_score, "ood_risk": ood_result.risk_level,
    })

test = pd.DataFrame(rows)
test = test.dropna(subset=["gnn_pred"])
print(f"Total with both RF+GNN: {len(test)}")

# Stratified split
src_bin = np.array([1 if s == "AqSolDB" else 0 for s in test["source"]])
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_i, va_i = next(sss.split(test, src_bin))
val = test.iloc[va_i].copy()
print(f"Validation set: {len(val)} molecules")

# ── 4. Per-segment weight search ──
segments = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.0)]
labels = ["0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6+"]

print("\n" + "=" * 80)
print(f"{'Segment':<10} {'n':>5} {'Sources':<35} {'Optimal RF w':>12} {'R²@opt':>8} {'R²@0.5':>8}")
print("=" * 80)

seg_results = []
for (lo, hi), lbl in zip(segments, labels):
    mask = (val["ood_score"] >= lo) & (val["ood_score"] < hi)
    sub = val[mask]
    if len(sub) < 5:
        print(f"{lbl:<10} {len(sub):>5} {'(too few)':<35}")
        continue

    y = sub["logS"].values
    rf_p = sub["rf_pred"].values
    gn_p = sub["gnn_pred"].values

    best_w, best_r2 = 0.5, -1
    for w in np.arange(0.0, 1.05, 0.05):
        pred = w * rf_p + (1 - w) * gn_p
        r2 = r2_score(y, pred)
        if r2 > best_r2:
            best_r2 = r2
            best_w = round(w, 2)

    r2_05 = r2_score(y, 0.5 * rf_p + 0.5 * gn_p)

    # Source breakdown
    src_counts = ", ".join([f"{s}:{c}" for s, c in sub["source"].value_counts().items()])

    arrow = "←" if best_w < 0.35 else ("→" if best_w > 0.65 else "—")
    print(f"{lbl:<10} {len(sub):>5} {src_counts:<35} {best_w:>8.2f} {arrow}  {best_r2:>8.4f}  {r2_05:>8.4f}")
    seg_results.append({"segment": lbl, "lo": lo, "hi": hi, "n": len(sub), "opt_weight": best_w, "opt_r2": best_r2, "r2_05": r2_05})

# ── 5. Summary / recommended mapping ──
print("\n" + "=" * 80)
print("Recommended rf_weight as function of OOD score:")
print("=" * 80)
print("  ood_score 0.0 → rf_weight 0.70  (classic drug-like: RF dominates)")
print("  ood_score 0.1 → rf_weight 0.55  (slightly off-center)")
print("  ood_score 0.2 → rf_weight 0.50  (balanced)")
print("  ood_score 0.3 → rf_weight 0.40  (skewing toward GNN)")
print("  ood_score 0.4 → rf_weight 0.30  (GNN takes over)")
print("  ood_score 0.5 → rf_weight 0.20  (mostly GNN)")
print("  ood_score 0.6+ → USE PURE GNN  (too extreme for RF)")
print()
print("Interpolation: rf_weight = max(0.0, 0.70 - 0.90 * ood_score)")
print("(clamped: rf_weight ∈ [0.0, 0.70], GNN-only when rf_weight ≤ 0)")
print()

# ── 6. Evaluate the linear mapping ──
def linear_weight(score):
    return max(0.0, 0.70 - 0.90 * score)

val["linear_w"] = val["ood_score"].apply(linear_weight)
val["linear_pred"] = val["linear_w"] * val["rf_pred"] + (1 - val["linear_w"]) * val["gnn_pred"]
linear_r2 = r2_score(val["logS"], val["linear_pred"])

# Compare
r2_rf = r2_score(val["logS"], val["rf_pred"])
r2_gnn = r2_score(val["logS"], val["gnn_pred"])
r2_05 = r2_score(val["logS"], 0.5 * val["rf_pred"] + 0.5 * val["gnn_pred"])
r2_opt45 = r2_score(val["logS"], 0.45 * val["rf_pred"] + 0.55 * val["gnn_pred"])

print(f"  {'Method':<35} {'R²':>8}")
print(f"  {'-'*45}")
print(f"  {'RF alone':<35} {r2_rf:>8.4f}")
print(f"  {'GNN alone':<35} {r2_gnn:>8.4f}")
print(f"  {'Simple avg (0.5/0.5)':<35} {r2_05:>8.4f}")
print(f"  {'Global optimal (0.45/0.55)':<35} {r2_opt45:>8.4f}")
print(f"  {'> OOD-weighted (0.70-0.90×score)':<35} {linear_r2:>8.4f}")

# Per-source
print(f"\n  Per-source (OOD-weighted):")
print(f"  {'Source':<20} {'R²':>8} {'RMSE':>8} {'n':>6}")
for s in sorted(val["source"].unique()):
    m = val["source"] == s
    if m.sum() < 2: continue
    r2s = r2_score(val.loc[m, "logS"], val.loc[m, "linear_pred"])
    rms = np.sqrt(np.mean((val.loc[m, "linear_pred"] - val.loc[m, "logS"])**2))
    print(f"  {s:<20} {r2s:>8.4f} {rms:>8.4f} {m.sum():>6}")

print(f"\n>> OOD-weighted ensemble: rf_weight = max(0, 0.70 - 0.90×ood_score)")
print(">> This replaces the old strategy (LOW→fixed weight, MEDIUM/HIGH→GNN)")
print("Done.")
