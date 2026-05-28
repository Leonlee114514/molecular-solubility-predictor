"""
Retrain both RF + GNN with supplementary data added.
1. Merge ESOL + AqSolDB + supplementary_logs.csv
2. Train RF (same config)
3. Train GNN (same config)
4. Compare validation metrics
"""

import os, sys, math, time, random, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from features import compute_features
from gnn_model import (
    SolubilityGNN, MoleculeGraphEncoder, collate_graphs,
    save_gnn_model, load_gnn_model, ATOM_FEATURE_DIM,
)
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Step 1: Load & merge all datasets ──
print("\nLoading datasets...")

# ESOL
df_esol = pd.read_csv(
    "https://raw.githubusercontent.com/dataprofessor/data/master/delaney.csv"
)
df_esol = df_esol.rename(columns={
    "measured log(solubility:mol/L)": "logS", "Compound ID": "ID"
})[["SMILES", "logS"]]
df_esol["source"] = "ESOL"
print(f"  ESOL: {len(df_esol)}")

# AqSolDB
df_aqsol = pd.read_csv("curated-solubility-dataset.csv")
smiles_col = next(c for c in df_aqsol.columns if "smiles" in c.lower())
sol_col = next(c for c in df_aqsol.columns if "solubility" in c.lower())
df_aqsol = df_aqsol[[smiles_col, sol_col]].rename(columns={smiles_col: "SMILES", sol_col: "logS"})
df_aqsol["source"] = "AqSolDB"
print(f"  AqSolDB: {len(df_aqsol)}")

# Supplementary
df_supp = pd.read_csv("supplementary_logs.csv")
df_supp["source"] = "Supplementary"
print(f"  Supplementary: {len(df_supp)}")

# Merge
df = pd.concat([df_esol, df_aqsol, df_supp], ignore_index=True)
df = df.drop_duplicates(subset=["SMILES"], keep="first")
print(f"  Total (deduplicated): {len(df)} molecules")

# ── Step 2: Compute features for RF + graphs for GNN ──
print("\nComputing features and graphs...")
encoder = MoleculeGraphEncoder()

feature_list, fingerprint_list = [], []
graphs, labels, sources = [], [], []
rf_skipped = 0
gnn_skipped = 0

for _, row in df.iterrows():
    smi = row["SMILES"]
    logS = row["logS"]
    src = row["source"]

    # RF features
    rf_result = compute_features(smi)
    if rf_result is None:
        rf_skipped += 1
    else:
        f, fp = rf_result
        feature_list.append(list(f.values()))
        fingerprint_list.append(fp)

    # GNN graph
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        gnn_skipped += 1
        continue
    g = encoder.mol_to_graph(mol)
    if g is None:
        gnn_skipped += 1
        continue

    graphs.append(g)
    labels.append(logS)
    sources.append(src)

print(f"  RF: {len(feature_list)} valid ({rf_skipped} skipped)")
print(f"  GNN: {len(graphs)} valid ({gnn_skipped} skipped)")

# ── Step 3: Stratified train/val split ──
src_binary = np.array([1 if s == "AqSolDB" else 0 for s in sources])
labels_arr = np.array(labels, dtype=np.float32)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, val_idx = next(sss.split(np.arange(len(graphs)), src_binary))

# For RF: filter to molecules that have both RF features and GNN graphs
train_set = set(train_idx)
val_set = set(val_idx)
rf_train_idx = [i for i in range(len(feature_list)) if i in train_set]
rf_val_idx = [i for i in range(len(feature_list)) if i in val_set]

print(f"\nSplit:")
print(f"  RF train: {len(rf_train_idx)} | RF val: {len(rf_val_idx)}")
print(f"  GNN train: {len(train_idx)} | GNN val: {len(val_idx)}")

# ════════════════════════════════════════════════════════════════════
# PART A: Train Random Forest
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("Training Random Forest...")
print("=" * 50)

X_rf = np.hstack([
    np.array(feature_list),
    np.array(fingerprint_list),
])
y_rf = np.array([labels[i] for i in range(len(graphs)) if i < len(feature_list)], dtype=np.float32)

# Wait, the indexing mismatch between RF features and GNN graphs is wrong.
# Let me restructure properly.

# Actually, let me just compute everything from the deduplicated SMILES list
print("\nRecomputing with proper alignment...")

all_smiles = df["SMILES"].tolist()
all_logs = df["logS"].values
all_sources = df["source"].values

X_list, y_list_rf, src_rf = [], [], []
graph_list, y_list_gnn, src_gnn = [], [], []
rf_skip, gnn_skip = 0, 0

for i, smi in enumerate(all_smiles):
    # RF
    result = compute_features(smi)
    if result is not None:
        f, fp = result
        X_list.append(np.hstack([list(f.values()), fp]))
        y_list_rf.append(all_logs[i])
        src_rf.append(all_sources[i])
    else:
        rf_skip += 1

    # GNN
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        g = encoder.mol_to_graph(mol)
        if g is not None:
            graph_list.append(g)
            y_list_gnn.append(all_logs[i])
            src_gnn.append(all_sources[i])
    else:
        gnn_skip += 1

print(f"  RF: {len(X_list)} valid ({rf_skip} skipped)")
print(f"  GNN: {len(graph_list)} valid ({gnn_skip} skipped)")

# ── Train RF ──
X = np.array(X_list)
y_rf = np.array(y_list_rf, dtype=np.float32)
src_rf = np.array(src_rf)

# Split for RF
sss_rf = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
src_binary_rf = np.array([1 if s == "AqSolDB" else 0 for s in src_rf])
rf_train_i, rf_val_i = next(sss_rf.split(X, src_binary_rf))

X_tr, X_va = X[rf_train_i], X[rf_val_i]
y_tr, y_va = y_rf[rf_train_i], y_rf[rf_val_i]

print(f"\nRF training set: {len(X_tr)}, validation: {len(X_va)}")
t0 = time.time()
rf_model = RandomForestRegressor(
    n_estimators=300, max_depth=20, random_state=SEED, n_jobs=-1
)
rf_model.fit(X_tr, y_tr)
t1 = time.time()
print(f"  RF trained in {t1-t0:.1f}s")

rf_pred = rf_model.predict(X_va)
rf_r2 = r2_score(y_va, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_va, rf_pred))
print(f"  RF val R² = {rf_r2:.3f}, RMSE = {rf_rmse:.3f}")

# ── Train GNN ──
print("\n" + "=" * 50)
print("Training GNN...")
print("=" * 50)

y_gnn = np.array(y_list_gnn, dtype=np.float32)
src_gnn_arr = np.array(src_gnn)
src_binary_gnn = np.array([1 if s == "AqSolDB" else 0 for s in src_gnn_arr])

sss_gnn = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
gnn_train_i, gnn_val_i = next(sss_gnn.split(np.arange(len(graph_list)), src_binary_gnn))

train_graphs = [graph_list[i] for i in gnn_train_i]
train_labels = y_gnn[gnn_train_i]
val_graphs = [graph_list[i] for i in gnn_val_i]
val_labels = y_gnn[gnn_val_i]
val_sources = src_gnn_arr[gnn_val_i]

print(f"  Train: {len(train_graphs)}  |  Val: {len(val_graphs)}")

BATCH_SIZE = 64
HIDDEN_DIM = 128
NUM_LAYERS = 3
EPOCHS = 200
PATIENCE = 30

model = SolubilityGNN(
    atom_dim=ATOM_FEATURE_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS
).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
)
criterion = nn.MSELoss()

best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    idx = np.random.permutation(len(train_graphs))
    train_losses = []

    for start in range(0, len(train_graphs), BATCH_SIZE):
        batch_idx = idx[start:start + BATCH_SIZE]
        batch_graphs = [train_graphs[i] for i in batch_idx]
        batch_y = torch.tensor(train_labels[batch_idx], dtype=torch.float32, device=DEVICE)

        batch_data = collate_graphs(batch_graphs)
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
            batch_graphs = val_graphs[start:start + BATCH_SIZE]
            batch_y = val_labels[start:start + BATCH_SIZE]
            batch_data = collate_graphs(batch_graphs)
            if batch_data is None:
                continue
            batch_data = {k: v.to(DEVICE) for k, v in batch_data.items()}
            pred = model(batch_data)
            val_preds.append((pred.cpu(), torch.tensor(batch_y, dtype=torch.float32)))

    val_pred = torch.cat([p[0] for p in val_preds])
    val_true = torch.cat([p[1] for p in val_preds])
    val_loss = criterion(val_pred, val_true).item()
    ss_res = ((val_true - val_pred) ** 2).sum()
    ss_tot = ((val_true - val_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    train_avg = np.mean(train_losses)
    scheduler.step(val_loss)

    if epoch <= 5 or epoch % 10 == 0 or val_loss < best_val_loss:
        print(f"  Epoch {epoch:3d}/{EPOCHS} | train: {train_avg:.4f} | val: {val_loss:.4f} | R²: {r2:.4f} | lr: {optimizer.param_groups[0]['lr']:.2e}")

    if val_loss < best_val_loss - 1e-5:
        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0
        os.makedirs("output_v2", exist_ok=True)
        save_gnn_model(model, "output_v2/gnn_solubility_model_v3.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

print(f"\nBest GNN epoch: {best_epoch} | val loss: {best_val_loss:.4f}")

# ── Final evaluation ──
print("\n" + "=" * 50)
print("Final Evaluation (best models)")
print("=" * 50)

# Load best GNN
gnn_best = load_gnn_model("output_v2/gnn_solubility_model_v3.pt", device=DEVICE)
gnn_best.eval()

# GNN predictions
all_gnn_preds = []
with torch.no_grad():
    for start in range(0, len(val_graphs), BATCH_SIZE):
        batch_graphs = val_graphs[start:start + BATCH_SIZE]
        batch_data = collate_graphs(batch_graphs)
        if batch_data is None:
            continue
        batch_data = {k: v.to(DEVICE) for k, v in batch_data.items()}
        all_gnn_preds.append(gnn_best(batch_data).cpu())

gnn_val_pred = torch.cat(all_gnn_preds).numpy()
gnn_val_true = y_gnn[gnn_val_i]

# RF predictions on same molecules via SMILES matching
# Use the val indices + compute from SMILES
val_smiles = [all_smiles[gnn_val_i[j]] for j in range(len(gnn_val_i))]
rf_on_val = []
for smi in val_smiles:
    result = compute_features(smi)
    if result is None:
        rf_on_val.append(np.nan)
    else:
        f, fp = result
        Xv = np.hstack([list(f.values()), fp]).reshape(1, -1)
        rf_on_val.append(rf_model.predict(Xv)[0])
rf_on_val = np.array(rf_on_val)

valid = ~np.isnan(rf_on_val)
gnn_v = gnn_val_pred[valid]
rf_v = rf_on_val[valid]
ens_v = (gnn_v + rf_v) / 2.0
y_v = gnn_val_true[valid]

print(f"\nValidation set (n={len(y_v)}):")
print(f"  {'Model':<22} {'R²':>7} {'RMSE':>8} {'MAE':>7}")
print(f"  {'-'*48}")
for name, preds in [("RF", rf_v), ("GNN", gnn_v), ("Ensemble", ens_v)]:
    r2 = r2_score(y_v, preds)
    rmse = np.sqrt(mean_squared_error(y_v, preds))
    mae = np.mean(np.abs(y_v - preds))
    print(f"  {name:<22} {r2:>7.3f} {rmse:>8.4f} {mae:>7.4f}")

# Per-source
print(f"\nPer-source R²:")
print(f"  {'Source':<20} {'RF':>8} {'GNN':>8} {'Ensemble':>8}")
for src in ["ESOL", "AqSolDB", "Supplementary"]:
    mask = val_sources[valid] == src
    if mask.sum() == 0:
        continue
    rf_src = r2_score(y_v[mask], rf_v[mask])
    gnn_src = r2_score(y_v[mask], gnn_v[mask])
    ens_src = r2_score(y_v[mask], ens_v[mask])
    print(f"  {src:<20} {rf_src:>8.3f} {gnn_src:>8.3f} {ens_src:>8.3f}")

# ── Save updated RF model ──
joblib.dump(rf_model, "output_v2/solubility_model_v3.pkl")
print(f"\nModels saved to output_v2/ (solubility_model_v3.pkl, gnn_solubility_model_v3.pt)")
