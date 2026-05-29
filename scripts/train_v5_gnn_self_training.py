"""
v5: GNN self-training with disagreement-based selection.

Key insight from v4: RF pseudo-labels don't add independent information
to the RF+GNN ensemble. This version uses the GNN (which has different
inductive biases) to pseudo-label, and selects molecules where RF and
GNN disagree most — directly targeting the ensemble's weak points.

Strategy:
1. Load original training data (ESOL + AqSolDB)
2. Load existing v3 GNN and RF models (best so far)
3. Predict on pKa molecules with both models
4. Select top N molecules by |RF-GNN| disagreement (hard cases)
5. Pseudo-label with ensemble average (RF+GNN)/2 for robust targets
6. Add to GNN training set, retrain GNN from scratch
7. Evaluate |RF-GNN| on test_batch.csv
"""

import os, sys, random, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from features import compute_features
from gnn_model import (
    SolubilityGNN, MoleculeGraphEncoder, collate_graphs,
    save_gnn_model, load_gnn_model, ATOM_FEATURE_DIM,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cpu"

N_PSEUDO = 3000  # high-disagreement molecules to add (fewer but higher quality)

# ── 1. Load original training data ──
print("Loading original training data...")
datasets = []

try:
    df_esol = pd.read_csv(
        "https://raw.githubusercontent.com/dataprofessor/data/master/delaney.csv"
    )
    df_esol = df_esol.rename(columns={
        "measured log(solubility:mol/L)": "logS", "Compound ID": "ID"
    })[["SMILES", "logS"]]
    print(f"  ESOL: {len(df_esol)}")
    datasets.append(df_esol)
except:
    print("  ESOL: skipped (network)")

df_aqsol = pd.read_csv("curated-solubility-dataset.csv")
sc = next(c for c in df_aqsol.columns if "smiles" in c.lower())
sl = next(c for c in df_aqsol.columns if "solubility" in c.lower())
df_aqsol = df_aqsol[[sc, sl]].rename(columns={sc: "SMILES", sl: "logS"})
datasets.append(df_aqsol)

df = pd.concat(datasets, ignore_index=True)
df = df.drop_duplicates(subset=["SMILES"], keep="first")
print(f"  Total: {len(df)}")

# ── 2. Train RF on full original data ──
print("\nTraining RF on full original data...")
X_list, y_list = [], []
for smi, logS in zip(df["SMILES"], df["logS"]):
    r = compute_features(smi)
    if r is None:
        continue
    f, fp = r
    X_list.append(np.hstack([list(f.values()), fp]))
    y_list.append(logS)

X = np.array(X_list)
y = np.array(y_list, dtype=np.float32)
rf = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=SEED, n_jobs=-1)
rf.fit(X, y)
print(f"  RF trained on {len(X)} molecules")

# ── 3. Load v3 GNN model ──
print("\nLoading v3 GNN model...")
encoder = MoleculeGraphEncoder()
gnn_v3 = load_gnn_model("output_v2/gnn_solubility_model_v3.pt", device=DEVICE)
print("  v3 GNN loaded")

# ── 4. Load pKa molecules as unlabeled pool ──
print("\nLoading pKa molecules for pseudo-labeling...")
df_pka = pd.concat([
    pd.read_csv("data/pretrain_pka_acidic.csv", usecols=["smiles"]).rename(columns={"smiles": "SMILES"}),
    pd.read_csv("data/pretrain_pka_basic.csv", usecols=["smiles"]).rename(columns={"smiles": "SMILES"}),
], ignore_index=True)
df_pka = df_pka.drop_duplicates(subset=["SMILES"], keep="first")
existing = set(df["SMILES"].tolist())
df_pka = df_pka[~df_pka["SMILES"].isin(existing)]
print(f"  Non-overlapping pKa molecules: {len(df_pka)}")

# Cap pool size for memory
MAX_POOL = 20000
if len(df_pka) > MAX_POOL:
    df_pka = df_pka.sample(n=MAX_POOL, random_state=SEED)
    print(f"  Sampled {MAX_POOL} for computation budget")

# ── 5. Compute RF + GNN predictions, measure disagreement ──
print(f"\nPredicting with RF and GNN, measuring disagreement...")
scored = []
for i, smi in enumerate(df_pka["SMILES"]):
    # RF features
    r = compute_features(smi)
    if r is None:
        continue
    f, fp = r

    # RF predict
    X_p = np.hstack([list(f.values()), fp]).reshape(1, -1)
    rf_pred = float(rf.predict(X_p)[0])

    # GNN predict
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    g = encoder.mol_to_graph(mol)
    if g is None:
        continue
    with torch.no_grad():
        gnn_pred = float(gnn_v3(g).item())

    # Disagreement
    disagreement = abs(rf_pred - gnn_pred)
    ensemble_pred = (rf_pred + gnn_pred) / 2.0

    scored.append({
        "SMILES": smi,
        "RF_pred": rf_pred,
        "GNN_pred": gnn_pred,
        "disagreement": disagreement,
        "ensemble_pred": ensemble_pred,
    })

    if (i + 1) % 10000 == 0:
        print(f"  Processed {i+1}/{len(df_pka)} — current max disagreement: {max(s['disagreement'] for s in scored[-1000:]):.3f}")

print(f"  Total scored: {len(scored)}")

# ── 6. Select top N by disagreement ──
scored.sort(key=lambda x: x["disagreement"], reverse=True)
selected = scored[:min(N_PSEUDO, len(scored))]

# Also ensure MW diversity among selected
for s in selected:
    mol = Chem.MolFromSmiles(s["SMILES"])
    s["MW"] = rdMolDescriptors.CalcExactMolWt(mol) if mol else 0

print(f"\nSelected {len(selected)} highest-disagreement molecules:")
disagreements = [s["disagreement"] for s in selected]
print(f"  Disagreement range: {min(disagreements):.3f} — {max(disagreements):.3f}")
print(f"  MW range: {min(s['MW'] for s in selected):.0f} — {max(s['MW'] for s in selected):.0f}")
print(f"  Mean disagreement: {np.mean(disagreements):.3f}")

# Show top 5
print(f"\n  Top 5 most-disagreed molecules:")
for s in selected[:5]:
    smi_short = s["SMILES"][:45]
    print(f"    {smi_short:48s} RF={s['RF_pred']:+.2f} GNN={s['GNN_pred']:+.2f} Δ={s['disagreement']:.2f}")

# ── 7. Build GNN training set with original + pseudo-labeled data ──
print("\nBuilding combined GNN training set...")

# Original data
orig_graphs, orig_labels = [], []
for smi, logS in zip(df["SMILES"], df["logS"]):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    g = encoder.mol_to_graph(mol)
    if g is None:
        continue
    orig_graphs.append(g)
    orig_labels.append(logS)

# Pseudo-labeled data (selected by disagreement)
pseudo_graphs, pseudo_labels = [], []
for s in selected:
    mol = Chem.MolFromSmiles(s["SMILES"])
    if mol is None:
        continue
    g = encoder.mol_to_graph(mol)
    if g is None:
        continue
    pseudo_graphs.append(g)
    pseudo_labels.append(s["ensemble_pred"])  # ensemble average as pseudo-label

print(f"  Original graphs: {len(orig_graphs)}")
print(f"  Pseudo-labeled graphs: {len(pseudo_graphs)}")

all_graphs = orig_graphs + pseudo_graphs
all_labels = np.array(orig_labels + pseudo_labels, dtype=np.float32)
print(f"  Total: {len(all_graphs)}")

# ── 8. Train/val split ──
train_i, val_i = train_test_split(
    np.arange(len(orig_graphs)), test_size=0.2, random_state=SEED
)
# Add pseudo-labeled data to training set
pseudo_start = len(orig_graphs)
pseudo_indices = list(range(pseudo_start, pseudo_start + len(pseudo_graphs)))
train_i = np.concatenate([train_i, pseudo_indices])

train_graphs = [all_graphs[i] for i in train_i]
train_labels = all_labels[train_i]
val_graphs = [all_graphs[i] for i in val_i]
val_labels = all_labels[val_i]
print(f"  Train: {len(train_graphs)} (incl. {len(pseudo_indices)} pseudo)  Val: {len(val_graphs)}")

# ── 9. Train GNN from scratch ──
print("\n" + "=" * 50)
print("Training GNN v5 with disagreement-based self-training...")
print("=" * 50)

model = SolubilityGNN(atom_dim=ATOM_FEATURE_DIM, hidden_dim=128, num_layers=3).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
criterion = nn.MSELoss()

BATCH_SIZE = 64
EPOCHS = 200
PATIENCE = 30
best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    idx = np.random.permutation(len(train_graphs))
    train_losses = []

    for start in range(0, len(train_graphs), BATCH_SIZE):
        batch_idx = idx[start:start + BATCH_SIZE]
        batch_gs = [train_graphs[i] for i in batch_idx]
        batch_y = torch.tensor(train_labels[batch_idx], dtype=torch.float32, device=DEVICE)
        batch_data = collate_graphs(batch_gs)
        if batch_data is None:
            continue
        batch_data = {k: v.to(DEVICE) for k, v in batch_data.items()}

        pred = model(batch_data)
        loss = criterion(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(loss.item())

    # Validate
    model.eval()
    val_preds = []
    with torch.no_grad():
        for start in range(0, len(val_graphs), BATCH_SIZE):
            batch_gs = val_graphs[start:start + BATCH_SIZE]
            batch_y = val_labels[start:start + BATCH_SIZE]
            batch_data = collate_graphs(batch_gs)
            if batch_data is None:
                continue
            batch_data = {k: v.to(DEVICE) for k, v in batch_data.items()}
            val_preds.append((model(batch_data).cpu(), torch.tensor(batch_y, dtype=torch.float32)))

    vp = torch.cat([p[0] for p in val_preds])
    vt = torch.cat([p[1] for p in val_preds])
    vl = criterion(vp, vt).item()
    r2 = 1 - ((vt - vp) ** 2).sum() / ((vt - vt.mean()) ** 2).sum() + 1e-8
    ta = np.mean(train_losses)
    scheduler.step(vl)

    if epoch <= 5 or epoch % 10 == 0 or vl < best_val_loss:
        print(f"  Epoch {epoch:3d} | train: {ta:.4f} | val: {vl:.4f} | R²: {r2:.4f} | lr: {optimizer.param_groups[0]['lr']:.2e}")

    if vl < best_val_loss - 1e-5:
        best_val_loss = vl
        best_epoch = epoch
        patience_counter = 0
        save_gnn_model(model, "output_v2/gnn_solubility_model_v5.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

print(f"\nBest epoch: {best_epoch} | val loss: {best_val_loss:.4f}")

# ── 10. Evaluate ──
print("\nFinal evaluation:")
gnn_best = load_gnn_model("output_v2/gnn_solubility_model_v5.pt", device=DEVICE)
gnn_best.eval()

all_preds = []
with torch.no_grad():
    for start in range(0, len(val_graphs), BATCH_SIZE):
        batch_gs = val_graphs[start:start + BATCH_SIZE]
        batch_data = collate_graphs(batch_gs)
        if batch_data is None:
            continue
        batch_data = {k: v.to(DEVICE) for k, v in batch_data.items()}
        all_preds.append(gnn_best(batch_data).cpu())

final_pred = torch.cat(all_preds).numpy()
final_true = val_labels[:len(final_pred)]

print(f"  Val R² = {r2_score(final_true, final_pred):.3f}")
print(f"  Val RMSE = {np.sqrt(mean_squared_error(final_true, final_pred)):.3f}")

# ── 11. Compare v2 vs v3 vs v5 on test_batch.csv ──
print("\n" + "=" * 50)
print("3-way comparison on test_batch.csv (v2 vs v3 vs v5)")
print("=" * 50)

df_test = pd.read_csv("test_batch.csv")
rf_v2 = joblib.load("output_v2/solubility_model_v2.pkl")
gnn_v2 = load_gnn_model("output_v2/gnn_solubility_model.pt", device=DEVICE)
gnn_v3 = load_gnn_model("output_v2/gnn_solubility_model_v3.pt", device=DEVICE)
gnn_v5 = gnn_best

rows = []
for smi in df_test["SMILES"]:
    r = compute_features(smi)
    if r is None:
        continue
    f, fp = r
    Xt = np.hstack([list(f.values()), fp]).reshape(1, -1)
    rf2 = float(rf_v2.predict(Xt)[0])

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    g2 = encoder.mol_to_graph(mol)
    g3 = encoder.mol_to_graph(mol)
    g5 = encoder.mol_to_graph(mol)
    if g2 is None or g3 is None or g5 is None:
        continue

    with torch.no_grad():
        g2p = float(gnn_v2(g2).item())
        g3p = float(gnn_v3(g3).item())
        g5p = float(gnn_v5(g5).item())

    rows.append({
        "SMILES": smi,
        "RF_v2": rf2,
        "GNN_v2": g2p, "GNN_v3": g3p, "GNN_v5": g5p,
        "|RF-GNN|_v2": abs(rf2 - g2p),
        "|RF-GNN|_v3": abs(rf2 - g3p),
        "|RF-GNN|_v5": abs(rf2 - g5p),
    })

res = pd.DataFrame(rows)
print(f"\n  {'Model':<20} {'Mean |RF-GNN|':>15}")
print(f"  {'-'*35}")
for ver in ["v2", "v3", "v5"]:
    col = f"|RF-GNN|_{ver}"
    print(f"  {'GNN ' + ver:<20} {res[col].mean():>15.3f}")

print(f"\n  Improvement:")
imp_v3 = res['|RF-GNN|_v2'].mean() - res['|RF-GNN|_v3'].mean()
imp_v5 = res['|RF-GNN|_v2'].mean() - res['|RF-GNN|_v5'].mean()
print(f"  v3 reduction from v2: {imp_v3:+.3f}")
print(f"  v5 reduction from v2: {imp_v5:+.3f}")

print(f"\n  Hardest cases (by v2 |RF-GNN|):")
for _, r in res.sort_values("|RF-GNN|_v2", ascending=False).head(5).iterrows():
    smi_short = r["SMILES"][:42]
    print(f"    {smi_short:44s} v2={r['GNN_v2']:+.2f} v3={r['GNN_v3']:+.2f} v5={r['GNN_v5']:+.2f} |RF|={r['RF_v2']:+.2f}")

print("\nModel saved: output_v2/gnn_solubility_model_v5.pt")
