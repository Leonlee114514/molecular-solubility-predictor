"""
Build the OOD detector from training data.

Computes descriptor statistics and fingerprint reference samples from the same
datasets used to train the solubility model (ESOL + AqSolDB), then saves them
as output_v2/ood_detector.pkl.

Run once after training or when the training data changes:
    python build_ood_detector.py
"""

import os
import sys
import numpy as np
import pandas as pd
from ood_detector import OODDetector, save_ood_detector, DESCRIPTOR_ORDER

# Add the project root to path so we can import features
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from features import compute_features

N_FP_SAMPLES = 2000  # number of fingerprint reference samples to store
RANDOM_SEED = 42

print("=" * 60)
print("Building OOD Detector from Training Data")
print("=" * 60)

# ── Step 1: Load datasets ────────────────────────────────────────────
datasets = []

print("\nLoading ESOL dataset...")
try:
    url = "https://raw.githubusercontent.com/dataprofessor/data/master/delaney.csv"
    df_esol = pd.read_csv(url)
    df_esol = df_esol.rename(columns={
        "measured log(solubility:mol/L)": "logS",
        "Compound ID": "ID",
    })[["SMILES", "logS"]]
    df_esol["source"] = "ESOL"
    datasets.append(df_esol)
    print(f"  OK: {len(df_esol)} molecules")
except Exception as e:
    print(f"  FAIL: {e}")

print("\nLoading AqSolDB dataset...")
try:
    df_aqsol = pd.read_csv("curated-solubility-dataset.csv")
    if "SMILES" in df_aqsol.columns and "Solubility" in df_aqsol.columns:
        df_aqsol = df_aqsol[["SMILES", "Solubility"]].rename(columns={"Solubility": "logS"})
    elif "smiles" in df_aqsol.columns and "solubility" in df_aqsol.columns:
        df_aqsol = df_aqsol[["smiles", "solubility"]].rename(
            columns={"smiles": "SMILES", "solubility": "logS"}
        )
    else:
        smiles_col = [c for c in df_aqsol.columns if "smiles" in c.lower() or "SMILES" in c][0]
        sol_col = [c for c in df_aqsol.columns if "solubility" in c.lower() or "Solubility" in c][0]
        df_aqsol = df_aqsol[[smiles_col, sol_col]].rename(
            columns={smiles_col: "SMILES", sol_col: "logS"}
        )
    df_aqsol["source"] = "AqSolDB"
    datasets.append(df_aqsol)
    print(f"  OK: {len(df_aqsol)} molecules")
except FileNotFoundError:
    print("  SKIP: curated-solubility-dataset.csv not found")
except Exception as e:
    print(f"  FAIL: {e}")

df = pd.concat(datasets, ignore_index=True)
df = df.drop_duplicates(subset=["SMILES"], keep="first")
print(f"\nTotal unique molecules: {len(df)}")

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
    if (i + 1) % 2000 == 0:
        print(f"  processed {i + 1}/{len(df)} molecules...")

print(f"Successfully processed: {valid_count} molecules")

# ── Step 3: Compute descriptor statistics ────────────────────────────
print("\nComputing descriptor statistics...")
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
