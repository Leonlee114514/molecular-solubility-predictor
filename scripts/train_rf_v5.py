"""
Retrain RF with V5 features (13 descriptors + 1024 Morgan fingerprint).
Trains on full V4 dataset (ESOL + AqSolDB + Supplementary + ChEMBL).
Saves model + rebuilds OOD detector.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from features import compute_features
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 42
print("=" * 60)
print("RF V5 Training — 13 descriptors + 1024 Morgan fingerprint")
print("=" * 60)

# ── 1. Load all datasets ──
datasets = []
for name, path, cols in [
    ("ESOL", "data/delaney.csv", {"Compound ID": "ID", "measured log(solubility:mol/L)": "logS", "ESOL predicted log(solubility:mol/L)": "_esol_pred"}),
    ("AqSolDB", "curated-solubility-dataset.csv", None),
    ("Supplementary", "supplementary_logs.csv", None),
    ("ChEMBL", "chembl_solubility.csv", None),
]:
    if not os.path.exists(path):
        print(f"  {name}: not found, skip")
        continue
    df = pd.read_csv(path)
    if cols:
        df = df.rename(columns=cols)[["SMILES", "logS"]]
    elif "SMILES" not in df.columns or "logS" not in df.columns:
        # Auto-detect
        sc = next(c for c in df.columns if "smiles" in c.lower())
        lc = next(c for c in df.columns if "solubility" in c.lower() or "logS" in c)
        df = df[[sc, lc]].rename(columns={sc: "SMILES", lc: "logS"})
    df["source"] = name
    datasets.append(df)
    print(f"  {name}: {len(df)}")

df = pd.concat(datasets, ignore_index=True).drop_duplicates(subset=["SMILES"], keep="first")
print(f"  Total (dedup): {len(df)}")

# ── 2. Compute V5 features ──
print("\nComputing V5 features (13 desc + 1024 Morgan)...")
t0 = time.time()
X_list, y_list, src_list = [], [], []
for _, row in df.iterrows():
    r = compute_features(row["SMILES"])
    if r is None:
        continue
    f, fp = r
    X_list.append(np.hstack([list(f.values()), fp]))
    y_list.append(row["logS"])
    src_list.append(row["source"])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)
src = np.array(src_list)
feat_sample, _ = compute_features("CCO")
desc_names = list(feat_sample.keys())

elapsed = time.time() - t0
print(f"  {len(X)} molecules, {X.shape[1]} features ({len(desc_names)} desc + 1024 fp) in {elapsed:.1f}s")

# ── 3. Stratified split ──
src_binary = np.array([1 if s == "AqSolDB" else 0 for s in src])
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_i, va_i = next(sss.split(X, src_binary))
X_tr, X_va = X[tr_i], X[va_i]
y_tr, y_va = y[tr_i], y[va_i]
src_va = src[va_i]
print(f"  Train: {len(X_tr)}  Val: {len(X_va)}")

# ── 4. Train RF ──
rf_params = {
    "n_estimators": 800, "max_depth": 30,
    "min_samples_split": 5, "min_samples_leaf": 2,
    "max_features": None, "random_state": SEED, "n_jobs": -1,
}
print(f"\nTraining RF (params: {rf_params})...")
t0 = time.time()
rf = RandomForestRegressor(**rf_params)
rf.fit(X_tr, y_tr)
elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s")

# ── 5. Evaluate ──
yp = rf.predict(X_va)
r2 = r2_score(y_va, yp)
rmse = np.sqrt(mean_squared_error(y_va, yp))
mae = mean_absolute_error(y_va, yp)
print(f"\n  {'Model':<22} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
print(f"  {'-'*48}")
print(f"  {'RF V5':<22} {r2:>8.4f} {rmse:>8.4f} {mae:>8.4f}")

# Per-source
print(f"\n  {'Source':<20} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'n':>6}")
for s in sorted(np.unique(src_va)):
    mask = src_va == s
    if mask.sum() < 2:
        continue
    rs = r2_score(y_va[mask], yp[mask])
    rms = np.sqrt(mean_squared_error(y_va[mask], yp[mask]))
    ms = mean_absolute_error(y_va[mask], yp[mask])
    print(f"  {s:<20} {rs:>8.4f} {rms:>8.4f} {ms:>8.4f} {mask.sum():>6}")

# ── 6. Feature importance ──
importances = rf.feature_importances_
top_idx = np.argsort(importances)[::-1][:15]
names = desc_names + [f"FP_bit_{i}" for i in range(1024)]
print(f"\n  Top 15 features:")
for rank, i in enumerate(top_idx, 1):
    print(f"    {rank:2d}. {names[i]:30s} {importances[i]:.4f}")

# ── 7. Save model ──
os.makedirs("output_v2", exist_ok=True)
joblib.dump(rf, "output_v2/solubility_model_v5.pkl")
joblib.dump(desc_names, "output_v2/descriptor_names_v5.pkl")
print(f"\n  Saved: output_v2/solubility_model_v5.pkl")

# ── 8. Rebuild OOD detector ──
print("\n" + "=" * 60)
print("Rebuilding OOD detector with V5 feature set...")
print("=" * 60)
from ood_detector import OODDetector, save_ood_detector, DESCRIPTOR_ORDER

all_descs = {n: [] for n in DESCRIPTOR_ORDER}
all_fps = []
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
rdBase = Chem.rdBase

# Recompute features with OOD descriptor order
for _, row in df.iterrows():
    mol = Chem.MolFromSmiles(row["SMILES"])
    if mol is None:
        continue
    feat = {}
    feat['MolWt'] = Descriptors.MolWt(mol)
    feat['LogP'] = Descriptors.MolLogP(mol)
    feat['NumHDonors'] = Descriptors.NumHDonors(mol)
    feat['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
    feat['TPSA'] = Descriptors.TPSA(mol)
    feat['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
    feat['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
    feat['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
    feat['FractionCSP3'] = Descriptors.FractionCSP3(mol)
    feat['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
    feat['HallKierAlpha'] = Descriptors.HallKierAlpha(mol)
    feat['Chi0v'] = Descriptors.Chi0v(mol)
    feat['Chi1v'] = Descriptors.Chi1v(mol)
    for n in DESCRIPTOR_ORDER:
        all_descs[n].append(feat[n])
    rdBase.DisableLog("rdApp.warning")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fa = np.zeros((1,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, fa)
    rdBase.EnableLog("rdApp.warning")
    all_fps.append(fa)

desc_stats = {}
for n in DESCRIPTOR_ORDER:
    vals = np.array(all_descs[n])
    desc_stats[n] = {
        "mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
        "min": float(vals.min()), "max": float(vals.max()),
    }
    print(f"  {n:22s} mean={desc_stats[n]['mean']:8.2f}  std={desc_stats[n]['std']:8.2f}  "
          f"range=[{desc_stats[n]['min']:.2f}, {desc_stats[n]['max']:.2f}]")

rng = np.random.RandomState(SEED)
all_fp_arr = np.array(all_fps)
n_sample = min(3000, len(all_fp_arr))
idx = rng.choice(len(all_fp_arr), size=n_sample, replace=False)
fp_samples = all_fp_arr[idx]

detector = OODDetector(desc_stats=desc_stats, fp_samples=fp_samples)
save_ood_detector(detector, "output_v2/ood_detector.pkl")
size_mb = os.path.getsize("output_v2/ood_detector.pkl") / (1024 * 1024)
print(f"\n  Saved: output_v2/ood_detector.pkl ({size_mb:.1f} MB)")

# ── 9. Config ──
config = {
    "model_version": "v5",
    "rf_params": rf_params,
    "n_features": X.shape[1],
    "n_desc": len(desc_names),
    "descriptors": desc_names,
    "data": {"n_total": len(df), "n_valid": len(X)},
    "metrics": {"val_r2": float(r2), "val_rmse": float(rmse), "val_mae": float(mae)},
}
with open("output_v2/v5_config.json", "w") as f:
    import json
    json.dump(config, f, indent=2)
print(f"  Saved: output_v2/v5_config.json")

print("\n" + "=" * 60)
print("V5 training complete!")
print("=" * 60)
