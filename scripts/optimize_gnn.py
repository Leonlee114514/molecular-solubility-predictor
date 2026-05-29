"""
GNN Hyperparameter Search.
Tests combinations of hidden_dim, num_layers, lr, dropout, batch_size.
Uses all available data. Reports best configuration.
"""

import os, sys, time, random, itertools, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gnn_model import (
    SolubilityGNN, MoleculeGraphEncoder, collate_graphs,
    save_gnn_model, ATOM_FEATURE_DIM,
)
from rdkit import Chem

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── 1. Load data ──
print("=" * 60)
print("Loading datasets...")
datasets = []

df_esol = pd.read_csv("data/delaney.csv")
df_esol = df_esol.rename(columns={
    "measured log(solubility:mol/L)": "logS", "Compound ID": "ID"
})[["SMILES", "logS"]]
df_esol["source"] = "ESOL"
datasets.append(df_esol)
print(f"  ESOL: {len(df_esol)}")

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

df_supp = pd.read_csv("supplementary_logs.csv")
df_supp["source"] = "Supplementary"
datasets.append(df_supp)
print(f"  Supplementary: {len(df_supp)}")

chembl_path = "chembl_solubility.csv"
if os.path.exists(chembl_path):
    df_chembl = pd.read_csv(chembl_path)
    df_chembl["source"] = "ChEMBL"
    datasets.append(df_chembl)
    print(f"  ChEMBL: {len(df_chembl)}")

df = pd.concat(datasets, ignore_index=True)
df = df.drop_duplicates(subset=["SMILES"], keep="first")
print(f"  Total (deduplicated): {len(df)}")

# ── 2. Convert to graphs ──
print("\nConverting molecules to graphs...")
encoder = MoleculeGraphEncoder()
graphs, labels, sources = [], [], []

for _, row in df.iterrows():
    mol = Chem.MolFromSmiles(row["SMILES"])
    if mol is None:
        continue
    g = encoder.mol_to_graph(mol)
    if g is None:
        continue
    graphs.append(g)
    labels.append(row["logS"])
    sources.append(row["source"])

print(f"  Converted: {len(graphs)} molecules")

# ── 3. Train/Val split ──
from sklearn.model_selection import StratifiedShuffleSplit

labels_arr = np.array(labels, dtype=np.float32)
src_binary = np.array([1 if s == "AqSolDB" else 0 for s in sources])

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, val_idx = next(sss.split(np.arange(len(graphs)), src_binary))

train_graphs = [graphs[i] for i in train_idx]
train_labels = labels_arr[train_idx]
val_graphs = [graphs[i] for i in val_idx]
val_labels = labels_arr[val_idx]

print(f"  Train: {len(train_graphs)}  Val: {len(val_graphs)}")

# ── 4. Hyperparameter search ──
print("\n" + "=" * 60)
print("GNN Hyperparameter Search")
print("=" * 60)

# Configurations to try
configs = [
    # (hidden_dim, num_layers, lr, dropout, batch_size)
    (128, 3, 1e-3, 0.1, 64),   # baseline (current)
    (128, 3, 5e-4, 0.1, 64),
    (128, 4, 1e-3, 0.1, 64),
    (128, 2, 1e-3, 0.1, 64),
    (256, 3, 1e-3, 0.1, 64),
    (256, 3, 5e-4, 0.1, 64),
    (64,  3, 1e-3, 0.1, 64),
    (128, 3, 1e-3, 0.2, 64),
    (128, 3, 1e-3, 0.0, 64),
    (128, 3, 1e-3, 0.1, 32),
    (256, 4, 1e-3, 0.1, 64),
    (256, 2, 1e-3, 0.1, 64),
]

EPOCHS = 100
PATIENCE = 15

results = []

for cfg_idx, (hidden_dim, num_layers, lr, dropout, batch_size) in enumerate(configs):
    cfg_name = f"cfg{cfg_idx+1:02d}_h{hidden_dim}_l{num_layers}_lr{lr:.0e}_d{dropout}_b{batch_size}"
    print(f"\n[{cfg_idx+1}/{len(configs)}] {cfg_name}")

    # Build model
    model = SolubilityGNN(
        atom_dim=ATOM_FEATURE_DIM, hidden_dim=hidden_dim, num_layers=num_layers
    ).to(DEVICE)
    model.head = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim // 2, 1),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        idx = np.random.permutation(len(train_graphs))
        train_losses = []

        for start in range(0, len(train_graphs), batch_size):
            batch_idx = idx[start:start + batch_size]
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
            for start in range(0, len(val_graphs), batch_size):
                batch_graphs = val_graphs[start:start + batch_size]
                batch_y = val_labels[start:start + batch_size]
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

        if epoch <= 3 or epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | train: {train_avg:.4f} | val: {val_loss:.4f} | R²: {r2:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    t_elapsed = time.time() - t_start

    # Final val metrics
    model.eval()
    all_pred = []
    with torch.no_grad():
        for start in range(0, len(val_graphs), batch_size):
            batch_graphs = val_graphs[start:start + batch_size]
            batch_data = collate_graphs(batch_graphs)
            if batch_data is None:
                continue
            batch_data = {k: v.to(DEVICE) for k, v in batch_data.items()}
            all_pred.append(model(batch_data).cpu())

    final_pred = torch.cat(all_pred).numpy()
    final_true = val_labels[:len(final_pred)]

    val_r2 = 1 - ((final_true - final_pred) ** 2).sum() / (((final_true - final_true.mean()) ** 2).sum() + 1e-8)
    val_rmse = np.sqrt(np.mean((final_true - final_pred) ** 2))
    val_mae = np.mean(np.abs(final_true - final_pred))

    results.append({
        "config": cfg_name,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "lr": lr,
        "dropout": dropout,
        "batch_size": batch_size,
        "val_r2": val_r2,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "time": t_elapsed,
    })

    print(f"  >> Best: epoch={best_epoch}, val_loss={best_val_loss:.4f}")
    print(f"  >> Final: R²={val_r2:.4f}, RMSE={val_rmse:.4f}, MAE={val_mae:.4f} ({t_elapsed:.1f}s)")

# ── 5. Results summary ──
print("\n" + "=" * 60)
print("Results Summary (sorted by val R²)")
print("=" * 60)

results.sort(key=lambda r: -r["val_r2"])

print(f"\n{'Rank':<6} {'Config':<35} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'Time':>8}")
print(f"  {'-'*75}")
for rank, r in enumerate(results, 1):
    print(f"  {rank:<4} {r['config']:<35} {r['val_r2']:>8.4f} {r['val_rmse']:>8.4f} {r['val_mae']:>8.4f} {r['time']:>7.1f}s")

print(f"\nBest config: {results[0]['config']}")
print(f"Best val R²: {results[0]['val_r2']:.4f}")

# ── 6. Save results ──
import json
os.makedirs("output_v2", exist_ok=True)
with open("output_v2/gnn_hparam_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to output_v2/gnn_hparam_results.json")
print("=" * 60)
print("GNN hyperparameter search complete!")
print("=" * 60)
