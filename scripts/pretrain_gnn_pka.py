"""
Pre-train GNN on pKa data (413K molecules) to learn general molecular representations.
Then transfer backbone weights for solubility fine-tuning.
"""
import os, sys, random, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rdkit import Chem
from gnn_model import (
    SolubilityGNN, MoleculeGraphEncoder, collate_graphs,
    save_gnn_model, ATOM_FEATURE_DIM,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cpu"

# Use a balanced sample for faster training
MAX_MOLECULES = 60000
PRETRAIN_EPOCHS = 60
PATIENCE = 10

# ── 1. Load pKa data ──
print("Loading pKa data...")
df_acidic = pd.read_csv("data/pretrain_pka_acidic.csv")
df_basic = pd.read_csv("data/pretrain_pka_basic.csv")

# Sample balanced dataset
n_acidic = min(MAX_MOLECULES // 2, len(df_acidic))
n_basic = min(MAX_MOLECULES // 2, len(df_basic))

df_acidic = df_acidic.sample(n=n_acidic, random_state=SEED)
df_basic = df_basic.sample(n=n_basic, random_state=SEED)

# Combine and remove duplicates (keep first occurrence)
df_acidic = df_acidic.rename(columns={"pka_acidic": "pka"})
df_basic = df_basic.rename(columns={"pka_basic": "pka"})
df_acidic["source"] = "acidic"
df_basic["source"] = "basic"

df = pd.concat([df_acidic, df_basic], ignore_index=True)
# Keep duplicate SMILES — same molecule may have both acidic and basic pKa
print(f"  Acidic: {n_acidic}, Basic: {n_basic}, Total: {len(df)}")
print(f"  pKa range: {df['pka'].min():.2f} — {df['pka'].max():.2f}")

# ── 2. Build graphs ──
print("\nBuilding molecular graphs...")
encoder = MoleculeGraphEncoder()
graphs, labels = [], []
skipped = 0
for i, (smi, pka) in enumerate(zip(df["smiles"], df["pka"])):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        skipped += 1
        continue
    g = encoder.mol_to_graph(mol)
    if g is None:
        skipped += 1
        continue
    graphs.append(g)
    labels.append(pka)
    if (i + 1) % 20000 == 0:
        print(f"  [{i+1}/{len(df)}] {len(graphs)} valid ({skipped} skipped)")

print(f"  Total valid graphs: {len(graphs)} ({skipped} skipped)")

# ── 3. Train/val split ──
from sklearn.model_selection import train_test_split
indices = np.arange(len(graphs))
train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=SEED)

train_graphs = [graphs[i] for i in train_idx]
train_labels = np.array([labels[i] for i in train_idx], dtype=np.float32)
val_graphs = [graphs[i] for i in val_idx]
val_labels = np.array([labels[i] for i in val_idx], dtype=np.float32)
print(f"\n  Train: {len(train_graphs)}  Val: {len(val_graphs)}")

# ── 4. Pre-train GNN on pKa ──
print("\n" + "=" * 50)
print("Pre-training GNN on pKa data...")
print("=" * 50)

model = SolubilityGNN(atom_dim=ATOM_FEATURE_DIM, hidden_dim=128, num_layers=3).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
criterion = nn.MSELoss()

BATCH_SIZE = 256  # larger batches for speed
best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

for epoch in range(1, PRETRAIN_EPOCHS + 1):
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

    if epoch <= 3 or epoch % 5 == 0 or vl < best_val_loss:
        print(f"  Epoch {epoch:2d}/{PRETRAIN_EPOCHS} | train: {ta:.4f} | val: {vl:.4f} | R²: {r2:.4f} | lr: {optimizer.param_groups[0]['lr']:.2e}")

    if vl < best_val_loss - 1e-5:
        best_val_loss = vl
        best_epoch = epoch
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break

print(f"\nBest pre-training epoch: {best_epoch} | val loss: {best_val_loss:.4f}")

# Save pre-trained model
os.makedirs("output_v2", exist_ok=True)
save_gnn_model(model, "output_v2/gnn_pka_pretrained.pt")
print("Pre-trained model saved: output_v2/gnn_pka_pretrained.pt")
