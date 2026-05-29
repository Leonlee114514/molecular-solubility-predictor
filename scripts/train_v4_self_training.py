"""
Self-training: use RF to pseudo-label diverse molecules from pKa dataset,
filter by OOD confidence, augment GNN training set.
"""

import os, sys, random, math, csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
import joblib

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cpu"

N_PSEUDO = 5000  # pseudo-labeled molecules to add

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
except:
    print("  ESOL: skipped (network)")
    df_esol = None

if df_esol is not None:
    datasets.append(df_esol)

df_aqsol = pd.read_csv("curated-solubility-dataset.csv")
sc = next(c for c in df_aqsol.columns if "smiles" in c.lower())
sl = next(c for c in df_aqsol.columns if "solubility" in c.lower())
df_aqsol = df_aqsol[[sc, sl]].rename(columns={sc: "SMILES", sl: "logS"})
datasets.append(df_aqsol)

df_supp = pd.read_csv("supplementary_logs.csv") if os.path.exists("supplementary_logs.csv") else None
if df_supp is not None:
    datasets.append(df_supp)

df = pd.concat(datasets, ignore_index=True)
df = df.drop_duplicates(subset=["SMILES"], keep="first")
print(f"  Total: {len(df)}")

# ── 2. Train RF on full original data ──
print("\nTraining RF on full data...")
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

# ── 3. Load pKa molecules as unlabeled pool ──
print("\nLoading pKa molecules for pseudo-labeling...")
df_pka = pd.concat([
    pd.read_csv("data/pretrain_pka_acidic.csv", usecols=["smiles"]).rename(columns={"smiles": "SMILES"}),
    pd.read_csv("data/pretrain_pka_basic.csv", usecols=["smiles"]).rename(columns={"smiles": "SMILES"}),
], ignore_index=True)
df_pka = df_pka.drop_duplicates(subset=["SMILES"], keep="first")
# Remove any that overlap with training set
existing = set(df["SMILES"].tolist())
df_pka = df_pka[~df_pka["SMILES"].isin(existing)]
print(f"  Non-overlapping pKa molecules: {len(df_pka)}")

# ── 4. Filter and pseudo-label ──
print(f"\nPseudo-labeling with RF (targeting {N_PSEUDO} molecules)...")
pseudo_results = []
for i, smi in enumerate(df_pka["SMILES"]):
    if i >= N_PSEUDO * 3:  # process up to 3x target, then filter
        break
    r = compute_features(smi)
    if r is None:
        continue
    f, fp = r
    X_p = np.hstack([list(f.values()), fp]).reshape(1, -1)
    pred = float(rf.predict(X_p)[0])
    pseudo_results.append({"SMILES": smi, "logS": pred, "features": f, "fp": fp})
    if (i+1) % 5000 == 0:
        print(f"  Processed {i+1}/{len(df_pka)}")

print(f"  Pseudo-labeled: {len(pseudo_results)}")

# Sort by diversity (MW as simple proxy) and pick top N_PSEUDO
# Get MW for diversity filter
for pr in pseudo_results:
    mol = Chem.MolFromSmiles(pr["SMILES"])
    pr["MW"] = rdMolDescriptors.CalcExactMolWt(mol) if mol else 0

# Stratified selection by MW decile to ensure diversity
mw_vals = np.array([pr["MW"] for pr in pseudo_results])
bins = pd.qcut(mw_vals, q=10, labels=False, duplicates="drop")
selected = []
for bin_id in range(len(np.unique(bins))):
    bin_indices = np.where(bins == bin_id)[0]
    n_from_bin = max(1, N_PSEUDO // len(np.unique(bins)))
    chosen = np.random.choice(bin_indices, min(n_from_bin, len(bin_indices)), replace=False)
    for idx in chosen:
        selected.append(pseudo_results[idx])

selected = selected[:N_PSEUDO]
print(f"  Selected: {len(selected)} (MW range: {min(pr['MW'] for pr in selected):.0f}-{max(pr['MW'] for pr in selected):.0f})")

# ── 5. Combine original + pseudo-labeled data for GNN training ──
print("\nBuilding combined GNN training set...")

encoder = MoleculeGraphEncoder()

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

# Pseudo-labeled data
pseudo_graphs, pseudo_labels = [], []
for pr in selected:
    mol = Chem.MolFromSmiles(pr["SMILES"])
    if mol is None:
        continue
    g = encoder.mol_to_graph(mol)
    if g is None:
        continue
    pseudo_graphs.append(g)
    pseudo_labels.append(pr["logS"])

print(f"  Original graphs: {len(orig_graphs)}")
print(f"  Pseudo-labeled graphs: {len(pseudo_graphs)}")

# Combine (use all original + pseudo)
all_graphs = orig_graphs + pseudo_graphs
all_labels = np.array(orig_labels + pseudo_labels, dtype=np.float32)
print(f"  Total: {len(all_graphs)}")

# ── 6. Train/val split ──
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

# ── 7. Train GNN ──
print("\n" + "=" * 50)
print("Training GNN with self-training augmentation...")
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
        save_gnn_model(model, "output_v2/gnn_solubility_model_v4.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

print(f"\nBest epoch: {best_epoch} | val loss: {best_val_loss:.4f}")

# ── 8. Evaluate ──
print("\nFinal evaluation:")
gnn_best = load_gnn_model("output_v2/gnn_solubility_model_v4.pt", device=DEVICE)
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

# ── 9. Compare on test_batch.csv ──
print("\n" + "=" * 50)
print("Comparison on test_batch.csv (v2 vs v4)")
print("=" * 50)

df_test = pd.read_csv("test_batch.csv")
rf_v2 = joblib.load("output_v2/solubility_model_v2.pkl")
gnn_v2 = load_gnn_model("output_v2/gnn_solubility_model.pt", device=DEVICE)
gnn_v4 = gnn_best

rows = []
for smi in df_test["SMILES"]:
    r = compute_features(smi)
    if r is None:
        continue
    f, fp = r
    Xt = np.hstack([list(f.values()), fp]).reshape(1, -1)
    rf2 = float(rf_v2.predict(Xt)[0])
    mol = Chem.MolFromSmiles(smi)
    g2 = encoder.mol_to_graph(mol)
    g4 = encoder.mol_to_graph(mol)
    if g2 is None or g4 is None:
        continue
    with torch.no_grad():
        g2p = float(gnn_v2(g2).item())
        g4p = float(gnn_v4(g4).item())
    rows.append({"SMILES": smi, "RF_v2": rf2, "GNN_v2": g2p, "GNN_v4": g4p,
                 "|RF-GNN|v2": abs(rf2-g2p), "|RF-GNN|v4": abs(rf2-g4p)})

res = pd.DataFrame(rows)
print(f"  Mean |RF-GNN| v2: {res['|RF-GNN|v2'].mean():.3f}")
print(f"  Mean |RF-GNN| v4: {res['|RF-GNN|v4'].mean():.3f}")
print(f"  Δ: {res['|RF-GNN|v4'].mean() - res['|RF-GNN|v2'].mean():+.3f}")
print(f"\n  Hard cases:")
for _, r in res.sort_values("|RF-GNN|v2", ascending=False).head(5).iterrows():
    smi_short = r["SMILES"][:40]
    print(f"    {smi_short:42s} v2={r['GNN_v2']:+.2f} v4={r['GNN_v4']:+.2f} |RF|={res.loc[res['SMILES']==r['SMILES'],'RF_v2'].values[0]:+.2f}")

print("\nModels saved: output_v2/gnn_solubility_model_v4.pt")
