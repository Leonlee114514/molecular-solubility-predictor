"""
Search for the optimal ensemble weight that maximizes validation R².
Tests rf_weight from 0.0 to 1.0 in steps of 0.05.
Loads the latest trained RF + GNN models.
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
from sklearn.metrics import r2_score, mean_squared_error
from rdkit import Chem

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1. Load latest models ──
# Try V5 first, fall back to V4
rf_path = "output_v2/solubility_model_v5.pkl"
if not os.path.exists(rf_path):
    rf_path = "output_v2/solubility_model_v4.pkl"
rf = joblib.load(rf_path)
print(f"RF: {rf_path}")

gnn_path = "output_v2/gnn_solubility_model_v4.pt"
gnn_encoder = MoleculeGraphEncoder()
for fname, hdim in [("gnn_solubility_model_v4.pt", 256), ("gnn_solubility_model_v3.pt", 128)]:
    p = os.path.join("output_v2", fname)
    if os.path.exists(p):
        gnn_path = p
        gnn_hidden = hdim
        break
gnn = SolubilityGNN(atom_dim=ATOM_FEATURE_DIM, hidden_dim=gnn_hidden, num_layers=3)
gnn.load_state_dict(torch.load(gnn_path, map_location=DEVICE, weights_only=True))
gnn.to(DEVICE)
gnn.eval()
print(f"GNN: {gnn_path} (hidden={gnn_hidden})")

# ── 2. Load test data ──
datasets = []
for name, path, cols in [
    ("ESOL", "data/delaney.csv", {"Compound ID": "ID", "measured log(solubility:mol/L)": "logS", "ESOL predicted log(solubility:mol/L)": "_esol_pred"}),
    ("AqSolDB", "curated-solubility-dataset.csv", None),
    ("Supplementary", "supplementary_logs.csv", None),
    ("ChEMBL", "chembl_solubility.csv", None),
]:
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    if cols:
        df = df.rename(columns=cols)[["SMILES", "logS"]]
    elif "SMILES" not in df.columns or "logS" not in df.columns:
        sc = next(c for c in df.columns if "smiles" in c.lower())
        lc = next(c for c in df.columns if "solubility" in c.lower() or "logS" in c)
        df = df[[sc, lc]].rename(columns={sc: "SMILES", lc: "logS"})
    df["source"] = name
    datasets.append(df)

df = pd.concat(datasets, ignore_index=True).drop_duplicates(subset=["SMILES"], keep="first")
print(f"\nData: {len(df)} unique molecules")

# ── 3. Compute RF features + GNN graphs ──
print("Computing features...")
X_list, smiles_list = [], []
for _, row in df.iterrows():
    r = compute_features(row["SMILES"])
    if r is None:
        continue
    f, fp = r
    X_list.append(np.hstack([list(f.values()), fp]))
    smiles_list.append(row["SMILES"])

X = np.array(X_list, dtype=np.float32)
y_true = df.loc[df["SMILES"].isin(smiles_list), "logS"].values.astype(np.float32)
print(f"  RF: {len(X)} valid")

# Stratified split
from sklearn.model_selection import StratifiedShuffleSplit
src = df[df["SMILES"].isin(smiles_list)]["source"].values
src_bin = np.array([1 if s == "AqSolDB" else 0 for s in src])
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_i, va_i = next(sss.split(X, src_bin))

# RF predictions
rf_pred = rf.predict(X[va_i])
rf_r2 = r2_score(y_true[va_i], rf_pred)
print(f"  RF Val R²: {rf_r2:.4f}")

# GNN predictions (on va set)
print("Computing GNN predictions...")
va_smiles = [smiles_list[i] for i in va_i]
gnn_preds = []
for smi in va_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        gnn_preds.append(np.nan)
        continue
    g = gnn_encoder.mol_to_graph(mol)
    if g is None:
        gnn_preds.append(np.nan)
        continue
    with torch.no_grad():
        p = gnn(g)
    gnn_preds.append(float(p.item()))

gnn_pred = np.array(gnn_preds)
valid = ~np.isnan(gnn_pred)
rf_v, gn_v, y_v = rf_pred[valid], gnn_pred[valid], y_true[va_i][valid]
gnn_r2 = r2_score(y_v, gn_v)
print(f"  GNN Val R²: {gnn_r2:.4f}")
print(f"  Valid molecules for ensemble: {len(y_v)}")

# ── 4. Weight search ──
print("\n" + "=" * 60)
print("Optimal Ensemble Weight Search")
print("=" * 60)
print(f"\n  {'Weight':>8} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
print(f"  {'-'*36}")

results = []
best_r2 = -1
best_w = 0.5

for w in np.arange(0.0, 1.05, 0.05):
    w = round(w, 2)
    pred = w * rf_v + (1.0 - w) * gn_v
    r2 = r2_score(y_v, pred)
    rmse = np.sqrt(np.mean((pred - y_v) ** 2))
    mae = np.mean(np.abs(pred - y_v))
    results.append((w, r2, rmse, mae))
    marker = " ← BEST" if r2 > best_r2 else ""
    if r2 > best_r2:
        best_r2 = r2
        best_w = w
    print(f"  rf_weight={w:<5} {r2:>8.4f} {rmse:>8.4f} {mae:>8.4f}{marker}")

# ── 5. Compare with simple average ──
simple_pred = 0.5 * rf_v + 0.5 * gn_v
simple_r2 = r2_score(y_v, simple_pred)

best_pred = best_w * rf_v + (1.0 - best_w) * gn_v
best_rmse = np.sqrt(np.mean((best_pred - y_v) ** 2))
best_mae = np.mean(np.abs(best_pred - y_v))

print(f"\n{'Method':<30} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
print(f"  {'-'*54}")
print(f"  {'RF alone':<30} {r2_score(y_v, rf_v):>8.4f} {np.sqrt(np.mean((rf_v-y_v)**2)):>8.4f} {np.mean(np.abs(rf_v-y_v)):>8.4f}")
print(f"  {'GNN alone':<30} {r2_score(y_v, gn_v):>8.4f} {np.sqrt(np.mean((gn_v-y_v)**2)):>8.4f} {np.mean(np.abs(gn_v-y_v)):>8.4f}")
print(f"  {'Simple avg (0.5RF+0.5GNN)':<30} {simple_r2:>8.4f} {np.sqrt(np.mean((simple_pred-y_v)**2)):>8.4f} {np.mean(np.abs(simple_pred-y_v)):>8.4f}")
print(f"  {'> Optimal weighted':<30} {best_r2:>8.4f} {best_rmse:>8.4f} {best_mae:>8.4f}")

# Per-source at best weight
print(f"\n  Per-source R² at rf_weight={best_w:.2f}:")
print(f"  {'Source':<20} {'R²':>8} {'RMSE':>8} {'n':>6}")
va_src = src[va_i][valid]
for s in sorted(np.unique(va_src)):
    mask = va_src == s
    if mask.sum() < 2:
        continue
    p = best_w * rf_v[mask] + (1.0 - best_w) * gn_v[mask]
    r2s = r2_score(y_v[mask], p)
    rms = np.sqrt(np.mean((p - y_v[mask]) ** 2))
    print(f"  {s:<20} {r2s:>8.4f} {rms:>8.4f} {mask.sum():>6}")

print(f"\n>> Optimal: rf_weight = {best_w:.2f}  (R² = {best_r2:.4f})")
print("Done.")
