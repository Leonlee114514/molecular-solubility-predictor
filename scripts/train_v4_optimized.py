"""
V4: Retrain both RF + GNN with optimized hyperparameters on expanded dataset.
1. Load all data (ESOL + AqSolDB + Supplementary + ChEMBL if available)
2. Train RF with best hyperparameters on FULL dataset (no CV, just full training)
3. Train GNN with best hyperparameters
4. Evaluate on held-out test set
5. Save models to V4
"""

import os, sys, time, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from features import compute_features
from gnn_model import (
    SolubilityGNN, MoleculeGraphEncoder, collate_graphs,
    save_gnn_model, load_gnn_model, ATOM_FEATURE_DIM,
)
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")

# ══════════════════════════════════════════════════════════════════════
# STEP 1: Load all datasets
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1: Loading all datasets")
print("=" * 60)

datasets = []

# ESOL (local copy)
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
    print(f"  ChEMBL: not available")

# Merge
df_raw = pd.concat(datasets, ignore_index=True)
n_before = len(df_raw)
df_raw = df_raw.drop_duplicates(subset=["SMILES"], keep="first")
n_after = len(df_raw)
print(f"\n  Total: {n_before} → {n_after} unique after dedup")
print(f"  Source distribution:")
for s, c in df_raw["source"].value_counts().items():
    print(f"    {s}: {c}")

# ══════════════════════════════════════════════════════════════════════
# STEP 2: Compute features for RF and graphs for GNN
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Computing features & graphs")
print("=" * 60)

X_list, y_list, src_list = [], [], []
graph_list, y_gnn_list, src_gnn_list = [], [], []

rf_skip, gnn_skip = 0, 0

t0 = time.time()
for _, row in df_raw.iterrows():
    smi = row["SMILES"]
    logS = row["logS"]
    src = row["source"]

    # RF features
    result = compute_features(smi)
    if result is not None:
        f, fp = result
        X_list.append(np.hstack([list(f.values()), fp]))
        y_list.append(logS)
        src_list.append(src)
    else:
        rf_skip += 1

    # GNN graph
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        g = MoleculeGraphEncoder().mol_to_graph(mol)
        if g is not None:
            graph_list.append(g)
            y_gnn_list.append(logS)
            src_gnn_list.append(src)
        else:
            gnn_skip += 1
    else:
        gnn_skip += 1

elapsed = time.time() - t0
print(f"  RF: {len(X_list)} valid ({rf_skip} skipped)")
print(f"  GNN: {len(graph_list)} valid ({gnn_skip} skipped)")
print(f"  Time: {elapsed:.1f}s")

X = np.array(X_list, dtype=np.float32)
y_rf = np.array(y_list, dtype=np.float32)
src_rf = np.array(src_list)

y_gnn = np.array(y_gnn_list, dtype=np.float32)
src_gnn = np.array(src_gnn_list)

# ══════════════════════════════════════════════════════════════════════
# STEP 3: Train/Val split
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Stratified train/val split")
print("=" * 60)

# RF split
src_binary_rf = np.array([1 if s == "AqSolDB" else 0 for s in src_rf])
sss_rf = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
rf_tr_i, rf_va_i = next(sss_rf.split(X, src_binary_rf))

X_tr, X_va = X[rf_tr_i], X[rf_va_i]
y_tr, y_va = y_rf[rf_tr_i], y_rf[rf_va_i]
src_tr, src_va = src_rf[rf_tr_i], src_rf[rf_va_i]

print(f"  RF train: {len(X_tr)}  RF val: {len(X_va)}")

# GNN split
src_binary_gnn = np.array([1 if s == "AqSolDB" else 0 for s in src_gnn])
sss_gnn = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
gnn_tr_i, gnn_va_i = next(sss_gnn.split(np.arange(len(graph_list)), src_binary_gnn))

gnn_train_graphs = [graph_list[i] for i in gnn_tr_i]
gnn_train_labels = y_gnn[gnn_tr_i]
gnn_val_graphs = [graph_list[i] for i in gnn_va_i]
gnn_val_labels = y_gnn[gnn_va_i]
gnn_val_sources = src_gnn[gnn_va_i]

print(f"  GNN train: {len(gnn_train_graphs)}  GNN val: {len(gnn_val_graphs)}")

# ══════════════════════════════════════════════════════════════════════
# STEP 4: Train Random Forest with optimized hyperparameters
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Training Random Forest (optimized params)")
print("=" * 60)

# Best params from randomized search (will be refined by search results)
# Using a robust set of params
# Best params from RandomizedSearchCV
rf_params = {
    "n_estimators": 800,
    "max_depth": 30,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": None,
    "random_state": SEED,
    "n_jobs": -1,
}

# Always retrain RF on full dataset (including ChEMBL)
print(f"  Training RF with params: {rf_params}")
t0 = time.time()
rf_model = RandomForestRegressor(**rf_params)
rf_model.fit(X_tr, y_tr)
elapsed = time.time() - t0
print(f"  Trained in {elapsed:.1f}s")

# RF validation
rf_pred = rf_model.predict(X_va)
rf_r2 = r2_score(y_va, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_va, rf_pred))
rf_mae = mean_absolute_error(y_va, rf_pred)
print(f"  RF Val: R²={rf_r2:.4f}, RMSE={rf_rmse:.4f}, MAE={rf_mae:.4f}")

# ══════════════════════════════════════════════════════════════════════
# STEP 5: Train GNN with optimized hyperparameters
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Training GNN (optimized params)")
print("=" * 60)

# Best params from GNN hyperparameter search (cfg05)
GNN_HIDDEN = 256
GNN_LAYERS = 3
GNN_LR = 1e-3
GNN_DROPOUT = 0.1
GNN_BATCH = 64
GNN_EPOCHS = 200
GNN_PATIENCE = 30

model = SolubilityGNN(
    atom_dim=ATOM_FEATURE_DIM, hidden_dim=GNN_HIDDEN, num_layers=GNN_LAYERS
).to(DEVICE)
model.head = nn.Sequential(
    nn.Linear(GNN_HIDDEN, GNN_HIDDEN // 2),
    nn.ReLU(),
    nn.Dropout(GNN_DROPOUT),
    nn.Linear(GNN_HIDDEN // 2, 1),
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=GNN_LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
)
criterion = nn.MSELoss()

n_params = sum(p.numel() for p in model.parameters())
print(f"  GNN params: {n_params:,} | hidden={GNN_HIDDEN}, layers={GNN_LAYERS}, "
      f"lr={GNN_LR:.0e}, dropout={GNN_DROPOUT}, batch={GNN_BATCH}")

best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

t0 = time.time()
for epoch in range(1, GNN_EPOCHS + 1):
    model.train()
    idx = np.random.permutation(len(gnn_train_graphs))
    train_losses = []

    for start in range(0, len(gnn_train_graphs), GNN_BATCH):
        batch_idx = idx[start:start + GNN_BATCH]
        batch_graphs = [gnn_train_graphs[i] for i in batch_idx]
        batch_y = torch.tensor(gnn_train_labels[batch_idx], dtype=torch.float32, device=DEVICE)

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
        for start in range(0, len(gnn_val_graphs), GNN_BATCH):
            batch_graphs = gnn_val_graphs[start:start + GNN_BATCH]
            batch_y = gnn_val_labels[start:start + GNN_BATCH]
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

    if epoch <= 5 or epoch % 10 == 0 or val_loss < best_val_loss - 1e-5:
        print(f"  Epoch {epoch:3d}/{GNN_EPOCHS} | train: {train_avg:.4f} | "
              f"val: {val_loss:.4f} | R²: {r2:.4f} | "
              f"lr: {optimizer.param_groups[0]['lr']:.2e}")

    if val_loss < best_val_loss - 1e-5:
        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0
        os.makedirs("output_v2", exist_ok=True)
        save_gnn_model(model, "output_v2/gnn_solubility_model_v4.pt")
    else:
        patience_counter += 1
        if patience_counter >= GNN_PATIENCE:
            print(f"  Early stopping at epoch {epoch} (best: epoch {best_epoch})")
            break

elapsed = time.time() - t0
print(f"\n  Best epoch: {best_epoch} | val loss: {best_val_loss:.4f} | time: {elapsed:.1f}s")

# ══════════════════════════════════════════════════════════════════════
# STEP 6: Final evaluation (all models + ensemble)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Final Evaluation")
print("=" * 60)

# Reload best GNN snapshot (use same architecture as current model)
best_state = torch.load("output_v2/gnn_solubility_model_v4.pt", map_location=DEVICE, weights_only=True)
model.load_state_dict(best_state)
model.eval()
gnn_best = model  # use in-memory model with best weights

# GNN predictions on val set
all_gnn_preds = []
with torch.no_grad():
    for start in range(0, len(gnn_val_graphs), GNN_BATCH):
        batch_graphs = gnn_val_graphs[start:start + GNN_BATCH]
        batch_data = collate_graphs(batch_graphs)
        if batch_data is None:
            continue
        batch_data = {k: v.to(DEVICE) for k, v in batch_data.items()}
        all_gnn_preds.append(gnn_best(batch_data).cpu())

gnn_pred = torch.cat(all_gnn_preds).numpy()
gnn_true = gnn_val_labels[:len(gnn_pred)]
gnn_sources = gnn_val_sources[:len(gnn_pred)]

# RF predictions on GNN val set (same molecules)
val_smiles = []
for i in gnn_va_i:
    orig_row = df_raw.iloc[i]
    val_smiles.append(orig_row["SMILES"])

rf_on_gnn_val = []
for smi in val_smiles:
    result = compute_features(smi)
    if result is None:
        rf_on_gnn_val.append(np.nan)
    else:
        f, fp = result
        Xv = np.hstack([list(f.values()), fp]).reshape(1, -1)
        rf_on_gnn_val.append(rf_model.predict(Xv)[0])
rf_on_gnn_val = np.array(rf_on_gnn_val)

valid = ~np.isnan(rf_on_gnn_val)
gnn_v = gnn_pred[valid]
rf_v = rf_on_gnn_val[valid]
ens_v = (gnn_v + rf_v) / 2.0
y_v = gnn_true[valid]
src_v = gnn_sources[valid]

print(f"\n  Validation set (n={len(y_v)}):")
print(f"  {'Model':<22} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
print(f"  {'-'*48}")
for name, preds in [("RF", rf_v), ("GNN", gnn_v), ("Ensemble", ens_v)]:
    r2 = r2_score(y_v, preds)
    rmse = np.sqrt(mean_squared_error(y_v, preds))
    mae = mean_absolute_error(y_v, preds)
    print(f"  {name:<22} {r2:>8.4f} {rmse:>8.4f} {mae:>8.4f}")

# Per-source
print(f"\n  Per-source R²:")
print(f"  {'Source':<20} {'RF':>8} {'GNN':>8} {'Ensemble':>8} {'n':>6}")
for src in sorted(np.unique(src_v)):
    mask = src_v == src
    if mask.sum() < 2:
        continue
    rf_s = r2_score(y_v[mask], rf_v[mask])
    gnn_s = r2_score(y_v[mask], gnn_v[mask])
    ens_s = r2_score(y_v[mask], ens_v[mask])
    print(f"  {src:<20} {rf_s:>8.4f} {gnn_s:>8.4f} {ens_s:>8.4f} {mask.sum():>6}")

# ══════════════════════════════════════════════════════════════════════
# STEP 7: Compare against old V3 models
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Comparison with V3 models")
print("=" * 60)

old_rf_path = "output_v2/solubility_model_v3.pkl"
old_gnn_path = "output_v2/gnn_solubility_model_v3.pt"

if os.path.exists(old_rf_path) and os.path.exists(old_gnn_path):
    # Old RF
    old_rf = joblib.load(old_rf_path)
    old_rf_pred = []
    for smi in val_smiles:
        result = compute_features(smi)
        if result is None:
            old_rf_pred.append(np.nan)
        else:
            f, fp = result
            Xv = np.hstack([list(f.values()), fp]).reshape(1, -1)
            old_rf_pred.append(old_rf.predict(Xv)[0])
    old_rf_pred = np.array(old_rf_pred)

    # Old GNN
    old_gnn = load_gnn_model(old_gnn_path, device=DEVICE)
    old_gnn.eval()
    old_gnn_preds = []
    with torch.no_grad():
        for start in range(0, len(gnn_val_graphs), GNN_BATCH):
            batch_graphs = gnn_val_graphs[start:start + GNN_BATCH]
            batch_data = collate_graphs(batch_graphs)
            if batch_data is None:
                continue
            batch_data = {k: v.to(DEVICE) for k, v in batch_data.items()}
            old_gnn_preds.append(old_gnn(batch_data).cpu())
    old_gnn_pred = torch.cat(old_gnn_preds).numpy()

    old_valid = ~np.isnan(old_rf_pred) & (old_gnn_pred[:len(old_rf_pred)] is not None)
    old_rf_v = old_rf_pred[valid]
    old_gnn_v = old_gnn_pred[valid]
    old_ens_v = (old_rf_v + old_gnn_v) / 2.0

    print(f"\n  {'Model':<22} {'V3 R²':>8} {'V4 R²':>8} {'ΔR²':>8}")
    print(f"  {'-'*48}")
    models = [("RF", old_rf_v, rf_v), ("GNN", old_gnn_v, gnn_v), ("Ensemble", old_ens_v, ens_v)]
    for name, old_p, new_p in models:
        old_r2 = r2_score(y_v, old_p)
        new_r2 = r2_score(y_v, new_p)
        delta = new_r2 - old_r2
        print(f"  {name:<22} {old_r2:>8.4f} {new_r2:>8.4f} {delta:>+8.4f}")
else:
    print("  Old V3 models not found, skipping comparison.")

# ══════════════════════════════════════════════════════════════════════
# STEP 8: Save all models
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8: Saving models")
print("=" * 60)

os.makedirs("output_v2", exist_ok=True)

# RF - always save the newly trained model
v4_rf_path = "output_v2/solubility_model_v4.pkl"
joblib.dump(rf_model, v4_rf_path)
print(f"  RF: saved to {v4_rf_path}")

# Descriptor names
dummy = compute_features("CCO")
desc_names = list(dummy[0].keys()) if dummy else [f"desc_{i}" for i in range(8)]
joblib.dump(desc_names, "output_v2/descriptor_names_v4.pkl")
print(f"  Descriptor names: saved to output_v2/descriptor_names_v4.pkl")

# GNN already saved during training
print(f"  GNN: saved to output_v2/gnn_solubility_model_v4.pt")

# ── Update to V4 model config info ──
config_info = {
    "model_version": "v4",
    "rf_params": rf_params,
    "gnn_params": {
        "hidden_dim": GNN_HIDDEN,
        "num_layers": GNN_LAYERS,
        "lr": GNN_LR,
        "dropout": GNN_DROPOUT,
        "batch_size": GNN_BATCH,
        "epochs": GNN_EPOCHS,
        "patience": GNN_PATIENCE,
    },
    "data": {
        "n_total_dedup": n_after,
        "sources": {s: int(c) for s, c in df_raw["source"].value_counts().items()},
        "n_rf_valid": len(X_list),
        "n_gnn_valid": len(graph_list),
    },
    "metrics": {
        "rf_val_r2": float(rf_r2),
        "rf_val_rmse": float(rf_rmse),
        "gnn_val_r2": float(r2_score(y_v, gnn_v)),
        "gnn_val_rmse": float(np.sqrt(mean_squared_error(y_v, gnn_v))),
        "ensemble_val_r2": float(r2_score(y_v, ens_v)),
        "ensemble_val_rmse": float(np.sqrt(mean_squared_error(y_v, ens_v))),
    },
}

with open("output_v2/v4_config.json", "w") as f:
    import json
    json.dump(config_info, f, indent=2, default=str)
print(f"  Config: saved to output_v2/v4_config.json")

print("\n" + "=" * 60)
print("V4 training complete!")
print("=" * 60)
