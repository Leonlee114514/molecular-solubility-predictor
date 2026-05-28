"""
Train SolubilityGNN on ESOL + AqSolDB (≈11K molecules).
Pure PyTorch GIN — no PyG/DGL needed.
"""

import os, sys, math, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
BATCH_SIZE = 64
HIDDEN_DIM = 128
NUM_LAYERS = 3
LR = 1e-3
EPOCHS = 200
PATIENCE = 25  # early stopping

print("=" * 60)
print("Training SolubilityGNN on ESOL + AqSolDB")
print(f"Device: {DEVICE}  |  Batch: {BATCH_SIZE}  |  Hidden: {HIDDEN_DIM}")
print("=" * 60)

# ── Step 1: Load datasets ──
datasets = []

print("\nLoading ESOL...")
df_esol = pd.read_csv(
    "https://raw.githubusercontent.com/dataprofessor/data/master/delaney.csv"
)
df_esol = df_esol.rename(columns={
    "measured log(solubility:mol/L)": "logS", "Compound ID": "ID"
})[["SMILES", "logS"]]
df_esol["source"] = "ESOL"
datasets.append(df_esol)
print(f"  ESOL: {len(df_esol)} molecules")

print("\nLoading AqSolDB...")
df_aqsol = pd.read_csv("curated-solubility-dataset.csv")
if "SMILES" in df_aqsol.columns and "Solubility" in df_aqsol.columns:
    df_aqsol = df_aqsol[["SMILES", "Solubility"]].rename(columns={"Solubility": "logS"})
elif "smiles" in df_aqsol.columns and "solubility" in df_aqsol.columns:
    df_aqsol = df_aqsol[["smiles", "solubility"]].rename(
        columns={"smiles": "SMILES", "solubility": "logS"}
    )
else:
    smiles_col = next(c for c in df_aqsol.columns if "smiles" in c.lower())
    sol_col = next(c for c in df_aqsol.columns if "solubility" in c.lower())
    df_aqsol = df_aqsol[[smiles_col, sol_col]].rename(
        columns={smiles_col: "SMILES", sol_col: "logS"}
    )
df_aqsol["source"] = "AqSolDB"
datasets.append(df_aqsol)
print(f"  AqSolDB: {len(df_aqsol)} molecules")

df = pd.concat(datasets, ignore_index=True)
df = df.drop_duplicates(subset=["SMILES"], keep="first")
print(f"  Total (deduplicated): {len(df)} molecules")

# ── Step 2: Convert all molecules to graphs ──
print("\nConverting molecules to graphs...")
encoder = MoleculeGraphEncoder()

graphs = []
labels = []
sources = []
skipped = 0

for _, row in df.iterrows():
    mol = Chem.MolFromSmiles(row["SMILES"])
    if mol is None:
        skipped += 1
        continue
    g = encoder.mol_to_graph(mol)
    if g is None:
        skipped += 1
        continue
    graphs.append(g)
    labels.append(row["logS"])
    sources.append(row["source"])

print(f"  Converted: {len(graphs)}  |  Skipped: {skipped}")

# ── Step 3: Stratified train/val split by source ──
from sklearn.model_selection import StratifiedShuffleSplit

labels_arr = np.array(labels, dtype=np.float32)
src_labels = np.array([1 if s == "AqSolDB" else 0 for s in sources])

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, val_idx = next(sss.split(np.arange(len(graphs)), src_labels))

train_graphs = [graphs[i] for i in train_idx]
train_labels = labels_arr[train_idx]
val_graphs = [graphs[i] for i in val_idx]
val_labels = labels_arr[val_idx]

print(f"\nSplit: train={len(train_graphs)}  val={len(val_graphs)}")

# ── Step 4: Training loop ──
model = SolubilityGNN(
    atom_dim=ATOM_FEATURE_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS
).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
)
criterion = nn.MSELoss()

best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

print(f"\nTraining {EPOCHS} epochs...")
print("-" * 50)

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # ── Train ──
    model.train()
    idx = np.random.permutation(len(train_graphs))
    train_losses = []

    for start in range(0, len(train_graphs), BATCH_SIZE):
        batch_idx = idx[start:start + BATCH_SIZE]
        batch_graphs = [train_graphs[i] for i in batch_idx]
        batch_y = torch.tensor(
            train_labels[batch_idx], dtype=torch.float32, device=DEVICE
        )

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

    # ── Validate ──
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

    # R²
    ss_res = ((val_true - val_pred) ** 2).sum()
    ss_tot = ((val_true - val_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    train_avg = np.mean(train_losses)
    scheduler.step(val_loss)

    elapsed = time.time() - t0

    if epoch <= 5 or epoch % 10 == 0 or val_loss < best_val_loss:
        print(
            f"  Epoch {epoch:3d}/{EPOCHS} | "
            f"train_loss: {train_avg:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_r²: {r2:.4f} | "
            f"lr: {optimizer.param_groups[0]['lr']:.2e} | "
            f"time: {elapsed:.1f}s"
        )

    # Early stopping
    if val_loss < best_val_loss - 1e-5:
        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0
        # Save best model
        os.makedirs("output_v2", exist_ok=True)
        save_gnn_model(model, "output_v2/gnn_solubility_model.pt")
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
        break

print("-" * 50)
print(f"Best epoch: {best_epoch}  |  Best val loss: {best_val_loss:.4f}")

# ── Step 5: Final evaluation ──
print("\nLoading best model for final evaluation...")
from gnn_model import load_gnn_model

model = load_gnn_model("output_v2/gnn_solubility_model.pt", device=DEVICE)

# Per-source evaluation
model.eval()
all_preds = []
with torch.no_grad():
    for start in range(0, len(val_graphs), BATCH_SIZE):
        batch_graphs = val_graphs[start:start + BATCH_SIZE]
        batch_data = collate_graphs(batch_graphs)
        if batch_data is None:
            continue
        batch_data = {k: v.to(DEVICE) for k, v in batch_data.items()}
        all_preds.append(model(batch_data).cpu())

val_pred_final = torch.cat(all_preds).numpy()
val_sources = np.array([sources[i] for i in val_idx])

print("\nFinal Evaluation (best model):")
for src in ["ESOL", "AqSolDB"]:
    mask = val_sources == src
    if mask.sum() == 0:
        continue
    src_pred = val_pred_final[mask]
    src_true = labels_arr[val_idx][mask]
    rmse = np.sqrt(np.mean((src_pred - src_true) ** 2))
    ss_res = ((src_true - src_pred) ** 2).sum()
    ss_tot = ((src_true - src_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    print(f"  {src:>8s}: R²={r2:.3f}  RMSE={rmse:.3f}  n={mask.sum()}")

overall_rmse = np.sqrt(np.mean((val_pred_final - labels_arr[val_idx]) ** 2))
ss_res = ((labels_arr[val_idx] - val_pred_final) ** 2).sum()
ss_tot = ((labels_arr[val_idx] - labels_arr[val_idx].mean()) ** 2).sum()
overall_r2 = 1 - ss_res / (ss_tot + 1e-8)
print(f"  {'Overall':>8s}: R²={overall_r2:.3f}  RMSE={overall_rmse:.3f}  n={len(val_idx)}")

n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {n_params:,}")
print("Model saved to output_v2/gnn_solubility_model.pt")
print("=" * 60)
print("Training complete!")
print("=" * 60)
