"""
Build the OOD detector from V4 training data (ESOL + AqSolDB + Supplementary + ChEMBL).

Computes descriptor statistics and fingerprint reference samples, then saves them
as output_v2/ood_detector.pkl.

Run after training V4 or whenever training data changes:
    python scripts/build_ood_detector.py
"""

import os, sys

# Ensure project root is on sys.path BEFORE any project imports
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
from ood_detector import OODDetector, save_ood_detector, DESCRIPTOR_ORDER
from features import compute_features

N_FP_SAMPLES = 3000  # more reference samples for expanded dataset
RANDOM_SEED = 42

print("=" * 60)
print("Building OOD Detector from V4 Training Data")
print("=" * 60)

# ── Step 1: Load all V4 datasets ─────────────────────────────────────
datasets = []

print("\nLoading ESOL (local)...")
df_esol = pd.read_csv("data/delaney.csv")
df_esol = df_esol.rename(columns={
    "measured log(solubility:mol/L)": "logS", "Compound ID": "ID"
})[["SMILES", "logS"]]
df_esol["source"] = "ESOL"
datasets.append(df_esol)
print(f"  OK: {len(df_esol)}")

print("Loading AqSolDB...")
df_aqsol = pd.read_csv("curated-solubility-dataset.csv")
if "SMILES" in df_aqsol.columns and "Solubility" in df_aqsol.columns:
    df_aqsol = df_aqsol[["SMILES", "Solubility"]].rename(columns={"Solubility": "logS"})
else:
    smiles_col = next(c for c in df_aqsol.columns if "smiles" in c.lower())
    sol_col = next(c for c in df_aqsol.columns if "solubility" in c.lower())
    df_aqsol = df_aqsol[[smiles_col, sol_col]].rename(columns={smiles_col: "SMILES", sol_col: "logS"})
df_aqsol["source"] = "AqSolDB"
datasets.append(df_aqsol)
print(f"  OK: {len(df_aqsol)}")

print("Loading Supplementary...")
df_supp = pd.read_csv("supplementary_logs.csv")
df_supp["source"] = "Supplementary"
datasets.append(df_supp)
print(f"  OK: {len(df_supp)}")

print("Loading ChEMBL...")
if os.path.exists("chembl_solubility.csv"):
    df_chembl = pd.read_csv("chembl_solubility.csv")
    df_chembl["source"] = "ChEMBL"
    datasets.append(df_chembl)
    print(f"  OK: {len(df_chembl)}")
else:
    print("  SKIP: not found")

df = pd.concat(datasets, ignore_index=True)
df = df.drop_duplicates(subset=["SMILES"], keep="first")
print(f"\n  Total unique molecules: {len(df)}")
for s, c in df["source"].value_counts().items():
    print(f"    {s}: {c}")

# ── Step 2: Compute features ─────────────────────────────────────────
print("\nComputing molecular features...")
all_descriptors = {name: [] for name in DESCRIPTOR_ORDER}
all_fingerprints = []
valid_count = 0

for i, (_, row) in enumerate(df.iterrows()):
    result = compute_features(row["SMILES"])
    if result is None:
        continue
    feat, fp = result
    for name in DESCRIPTOR_ORDER:
        all_descriptors[name].append(feat[name])
    all_fingerprints.append(fp)
    valid_count += 1
    if (i + 1) % 3000 == 0:
        print(f"  processed {i + 1}/{len(df)}...")

print(f"  Done: {valid_count} molecules")

# ── Step 3: Descriptor statistics ────────────────────────────────────
print("\nDescriptor statistics (V4, n={}):".format(valid_count))
desc_stats = {}
for name in DESCRIPTOR_ORDER:
    vals = np.array(all_descriptors[name])
    desc_stats[name] = {
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }
    print(f"  {name:22s}  mean={desc_stats[name]['mean']:8.2f}  "
          f"std={desc_stats[name]['std']:8.2f}  "
          f"range=[{desc_stats[name]['min']:.2f}, {desc_stats[name]['max']:.2f}]")

# ── Step 4: Sample fingerprints ──────────────────────────────────────
print(f"\nSampling {N_FP_SAMPLES} fingerprints for reference set...")
rng = np.random.RandomState(RANDOM_SEED)
all_fps = np.array(all_fingerprints)
n_total = len(all_fps)
n_sample = min(N_FP_SAMPLES, n_total)
indices = rng.choice(n_total, size=n_sample, replace=False)
fp_samples = all_fps[indices]
print(f"  Reference set: {fp_samples.shape[0]} fingerprints × {fp_samples.shape[1]} bits")

# ── Step 5: Save ─────────────────────────────────────────────────────
os.makedirs("output_v2", exist_ok=True)
detector = OODDetector(desc_stats=desc_stats, fp_samples=fp_samples)
save_ood_detector(detector, "output_v2/ood_detector.pkl")

size_mb = os.path.getsize("output_v2/ood_detector.pkl") / (1024 * 1024)
print(f"\nSaved: output_v2/ood_detector.pkl ({size_mb:.1f} MB)")
print("Done.")
