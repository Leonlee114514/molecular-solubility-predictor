
"""Clean comparison: original models (v2) vs retrained models (v3) on test_batch.csv."""

import numpy as np, pandas as pd, torch, joblib, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rdkit import Chem
from gnn_model import MoleculeGraphEncoder, load_gnn_model
from features import compute_features

SEED = 42
np.random.seed(SEED)

# Load test batch
df = pd.read_csv("test_batch.csv")
smiles_list = df["SMILES"].tolist()
names = df["Name"].tolist()

# ── Load models ──
# Original (v2)
rf_v2 = joblib.load("output_v2/solubility_model_v2.pkl")
gnn_v2 = load_gnn_model("output_v2/gnn_solubility_model.pt", device="cpu")
encoder = MoleculeGraphEncoder()

# Retrained (v3)
rf_v3 = joblib.load("output_v2/solubility_model_v3.pkl")
gnn_v3 = load_gnn_model("output_v2/gnn_solubility_model_v3.pt", device="cpu")

results = []
for smi, name in zip(smiles_list, names):
    result = compute_features(smi)
    if result is None:
        continue
    f, fp = result
    X = np.hstack([list(f.values()), fp]).reshape(1, -1)

    rf2 = float(rf_v2.predict(X)[0])
    rf3 = float(rf_v3.predict(X)[0])

    mol = Chem.MolFromSmiles(smi)
    g2 = encoder.mol_to_graph(mol)
    g3 = encoder.mol_to_graph(mol)
    with torch.no_grad():
        g2_pred = float(gnn_v2(g2).item())
        g3_pred = float(gnn_v3(g3).item())

    results.append({
        "Name": name, "SMILES": smi,
        "RF_v2": round(rf2, 3), "RF_v3": round(rf3, 3),
        "RF_chg": round(rf3 - rf2, 3),
        "GNN_v2": round(g2_pred, 3), "GNN_v3": round(g3_pred, 3),
        "GNN_chg": round(g3_pred - g2_pred, 3),
        "Ens_v2": round((rf2 + g2_pred)/2, 3),
        "Ens_v3": round((rf3 + g3_pred)/2, 3),
        "|RF-GNN|_v2": round(abs(rf2 - g2_pred), 3),
        "|RF-GNN|_v3": round(abs(rf3 - g3_pred), 3),
    })

res = pd.DataFrame(results)

print("=" * 100)
print("Model comparison: Original (v2) vs Retrained (v3) on test_batch.csv")
print("=" * 100)

# Overall stats
print(f"\nOverall:")
for model in ["RF", "GNN", "Ens"]:
    v2_col = f"{model}_v2"
    v3_col = f"{model}_v3"
    if model == "Ens":
        mean_v2 = res[v2_col].mean()
        mean_v3 = res[v3_col].mean()
    else:
        mean_v2 = res[v2_col].mean()
        mean_v3 = res[v3_col].mean()
    print(f"  {model}: v2_mean={mean_v2:.3f}  v3_mean={mean_v3:.3f}  Δ={mean_v3-mean_v2:+.3f}")

# Mean disagreement
for version in ["v2", "v3"]:
    col = f"|RF-GNN|_{version}"
    print(f"  Mean |RF-GNN| ({version}): {res[col].mean():.3f}")

# GNN changes detail
print(f"\nTop 10 largest GNN changes (v2 → v3):")
chg = res.reindex(res["GNN_chg"].abs().sort_values(ascending=False).index)
for _, r in chg.head(10).iterrows():
    print(f"  {r['Name']:<28s} GNN: {r['GNN_v2']:>6.3f} → {r['GNN_v3']:>6.3f} (Δ={r['GNN_chg']:+.3f})")

print(f"\nTop 10 largest RF changes (v2 → v3):")
for _, r in chg.sort_values("RF_chg", key=lambda x: x.abs(), ascending=False).head(10).iterrows():
    print(f"  {r['Name']:<28s} RF: {r['RF_v2']:>6.3f} → {r['RF_v3']:>6.3f} (Δ={r['RF_chg']:+.3f})")

# Check specific hard cases
print(f"\nHard case check (cyclohexane, where GNN was most wrong):")
cyc = res[res["Name"].str.contains("环己", na=False)]
if len(cyc):
    r = cyc.iloc[0]
    print(f"  {r['Name']}: RF_v2={r['RF_v2']} RF_v3={r['RF_v3']}  GNN_v2={r['GNN_v2']} GNN_v3={r['GNN_v3']}")
    print(f"  (best guess for cyclohexane logS ≈ -3.1, GNN_v2 was {r['GNN_v2']:.1f}, GNN_v3 is {r['GNN_v3']:.1f})")

print(f"\nHard case check (ATP):")
atp = res[res["Name"].str.contains("ATP", na=False)]
if len(atp):
    r = atp.iloc[0]
    print(f"  {r['Name']}: RF_v2={r['RF_v2']} RF_v3={r['RF_v3']}  GNN_v2={r['GNN_v2']} GNN_v3={r['GNN_v3']}")

print(f"\nHard case check (Aniline):")
aniline = res[res["Name"].str.contains("苯胺", na=False)]
if len(aniline):
    r = aniline.iloc[0]
    print(f"  {r['Name']}: RF_v2={r['RF_v2']} RF_v3={r['RF_v3']}  GNN_v2={r['GNN_v2']} GNN_v3={r['GNN_v3']}")

# Summary
print(f"\n{'='*60}")
print(f"Summary statistics:")
print(f"{'Metric':<30} {'v2':>8} {'v3':>8} {'Δ':>8}")
print(f"{'-'*56}")
print(f"{'Mean |RF-GNN| disagreement':<30} {res['|RF-GNN|_v2'].mean():>8.3f} {res['|RF-GNN|_v3'].mean():>8.3f} {res['|RF-GNN|_v3'].mean()-res['|RF-GNN|_v2'].mean():>+8.3f}")
g2_min, g2_max = res["GNN_v2"].min(), res["GNN_v2"].max()
g3_min, g3_max = res["GNN_v3"].min(), res["GNN_v3"].max()
print(f"{'GNN range (min-max)':<30} {g2_min:.2f}~{g2_max:.2f}  {g3_min:.2f}~{g3_max:.2f}")
