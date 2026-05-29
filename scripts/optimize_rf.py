"""
RF Hyperparameter Optimization with 5-fold Cross-Validation.
Searches over n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features.
Uses all available data: ESOL + AqSolDB + Supplementary + ChEMBL (if available).
Saves best model as solubility_model_v4.pkl.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from features import compute_features

PYTHON = r"C:\Users\Lyonl\AppData\Local\Python\bin\python.exe"
SEED = 42

# ── 1. Load all datasets ──
print("=" * 60)
print("Loading datasets...")
print("=" * 60)

datasets = []

# ESOL (local copy, downloaded from dataprofessor/delaney)
df_esol = pd.read_csv("data/delaney.csv")
df_esol = df_esol.rename(columns={
    "measured log(solubility:mol/L)": "logS", "Compound ID": "ID"
})[["SMILES", "logS"]]
df_esol["source"] = "ESOL"
datasets.append(df_esol)
print(f"  ESOL: {len(df_esol)}")

# AqSolDB
df_aqsol = pd.read_csv("curated-solubility-dataset.csv")
if "SMILES" in df_aqsol.columns and "Solubility" in df_aqsol.columns:
    df_aqsol = df_aqsol[["SMILES", "Solubility"]].rename(columns={"Solubility": "logS"})
elif "smiles" in df_aqsol.columns and "solubility" in df_aqsol.columns:
    df_aqsol = df_aqsol[["smiles", "solubility"]].rename(columns={"smiles": "SMILES", "solubility": "logS"})
else:
    smiles_col = next(c for c in df_aqsol.columns if "smiles" in c.lower())
    sol_col = next(c for c in df_aqsol.columns if "solubility" in c.lower())
    df_aqsol = df_aqsol[[smiles_col, sol_col]].rename(columns={smiles_col: "SMILES", sol_col: "logS"})
df_aqsol["source"] = "AqSolDB"
datasets.append(df_aqsol)
print(f"  AqSolDB: {len(df_aqsol)}")

# Supplementary
df_supp = pd.read_csv("supplementary_logs.csv")
df_supp["source"] = "Supplementary"
datasets.append(df_supp)
print(f"  Supplementary: {len(df_supp)}")

# ChEMBL (if available)
chembl_path = "chembl_solubility.csv"
if os.path.exists(chembl_path):
    df_chembl = pd.read_csv(chembl_path)
    df_chembl["source"] = "ChEMBL"
    datasets.append(df_chembl)
    print(f"  ChEMBL: {len(df_chembl)}")
else:
    print(f"  ChEMBL: not found (will train without it)")

# Merge
df = pd.concat(datasets, ignore_index=True)
n_before = len(df)
df = df.drop_duplicates(subset=["SMILES"], keep="first")
print(f"\n  Total (deduplicated): {n_before} → {len(df)} unique molecules")

# ── 2. Compute features ──
print("\n" + "=" * 60)
print("Computing molecular features...")
print("=" * 60)

X_list, y_list, src_list = [], [], []
skipped = 0

t0 = time.time()
for idx, row in df.iterrows():
    result = compute_features(row["SMILES"])
    if result is None:
        skipped += 1
        continue
    feat, fp = result
    X_list.append(np.hstack([list(feat.values()), fp]))
    y_list.append(row["logS"])
    src_list.append(row.get("source", "unknown"))

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)
src = np.array(src_list)

elapsed = time.time() - t0
print(f"  Computed: {len(X)} molecules ({skipped} skipped) in {elapsed:.1f}s")
print(f"  Feature dim: {X.shape[1]}")

# ── 3. Train/Test split ──
print("\n" + "=" * 60)
print("Train/Test split (stratified by source)...")
print("=" * 60)

from sklearn.model_selection import StratifiedShuffleSplit

# Create binary source labels for stratification
src_binary = np.array([1 if s == "AqSolDB" else 0 for s in src])

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(sss.split(X, src_binary))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
src_train, src_test = src[train_idx], src[test_idx]

print(f"  Train: {len(X_train)}  Test: {len(X_test)}")
for s in np.unique(src):
    print(f"    {s}: train={np.sum(src_train==s)} test={np.sum(src_test==s)}")

# ── 4. Hyperparameter search ──
print("\n" + "=" * 60)
print("RandomizedSearchCV for Random Forest...")
print("=" * 60)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error

param_dist = {
    "n_estimators": [200, 300, 500, 800, 1000],
    "max_depth": [10, 15, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}

base_rf = RandomForestRegressor(random_state=SEED, n_jobs=-1, verbose=0)

search = RandomizedSearchCV(
    estimator=base_rf,
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    random_state=SEED,
    verbose=1,
)

t0 = time.time()
search.fit(X_train, y_train)
elapsed = time.time() - t0

print(f"\n  Search completed in {elapsed:.1f}s")
print(f"  Best CV R²: {search.best_score_:.4f}")
print(f"  Best params: {search.best_params_}")

# ── 5. Evaluate on test set ──
print("\n" + "=" * 60)
print("Test set evaluation...")
print("=" * 60)

best_rf = search.best_estimator_
y_pred = best_rf.predict(X_test)

test_r2 = r2_score(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_mae = np.mean(np.abs(y_test - y_pred))

print(f"  {'Metric':<20} {'Value':>10}")
print(f"  {'-'*32}")
print(f"  {'R²':<20} {test_r2:>10.4f}")
print(f"  {'RMSE':<20} {test_rmse:>10.4f}")
print(f"  {'MAE':<20} {test_mae:>10.4f}")

# Per-source
print(f"\n  Per-source results:")
print(f"  {'Source':<20} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'n':>6}")
for s in np.unique(src_test):
    mask = src_test == s
    if mask.sum() < 2:
        continue
    r2_s = r2_score(y_test[mask], y_pred[mask])
    rmse_s = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))
    mae_s = np.mean(np.abs(y_test[mask] - y_pred[mask]))
    print(f"  {s:<20} {r2_s:>8.4f} {rmse_s:>8.4f} {mae_s:>8.4f} {mask.sum():>6}")

# ── 6. Feature importance ──
print("\n" + "=" * 60)
print("Top 15 feature importances...")
print("=" * 60)

descriptor_names = list(next(df.iterrows())[1].keys())  # placeholder
# Load actual descriptor names
from features import compute_features as cf
dummy = cf("CCO")
desc_names = list(dummy[0].keys()) if dummy else [f"desc_{i}" for i in range(8)]
feature_names = desc_names + [f"FP_bit_{i}" for i in range(1024)]

importances = best_rf.feature_importances_
top_indices = np.argsort(importances)[::-1][:15]
for rank, idx in enumerate(top_indices, 1):
    print(f"  {rank:2d}. {feature_names[idx]:30s} {importances[idx]:.4f}")

# ── 7. Compare with old model (V3) ──
print("\n" + "=" * 60)
print("Comparison with old V3 model...")
print("=" * 60)

old_path = "output_v2/solubility_model_v3.pkl"
if os.path.exists(old_path):
    old_rf = joblib.load(old_path)
    old_pred = old_rf.predict(X_test)
    old_r2 = r2_score(y_test, old_pred)
    old_rmse = np.sqrt(mean_squared_error(y_test, old_pred))
    old_mae = np.mean(np.abs(y_test - old_pred))

    print(f"  {'Model':<20} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
    print(f"  {'-'*46}")
    print(f"  {'Old V3':<20} {old_r2:>8.4f} {old_rmse:>8.4f} {old_mae:>8.4f}")
    print(f"  {'New V4 (optimized)':<20} {test_r2:>8.4f} {test_rmse:>8.4f} {test_mae:>8.4f}")
    print(f"  {'Improvement':<20} {test_r2 - old_r2:>+8.4f} {old_rmse - test_rmse:>+8.4f} {old_mae - test_mae:>+8.4f}")
else:
    print("  Old V3 model not found, skipping comparison.")

# ── 8. Save model ──
print("\n" + "=" * 60)
print("Saving optimized model...")
print("=" * 60)

os.makedirs("output_v2", exist_ok=True)
joblib.dump(best_rf, "output_v2/solubility_model_v4.pkl")
joblib.dump(desc_names, "output_v2/descriptor_names_v4.pkl")

print(f"  Model saved to output_v2/solubility_model_v4.pkl")
print(f"  Best params: {search.best_params_}")
print("=" * 60)
print("RF optimization complete!")
print("=" * 60)
