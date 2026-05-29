"""
Fine-tune GNN on solubility data using pre-trained pKa backbone weights.
Compares v2/v3/v6 on test_batch.csv.
"""
import os, sys, random, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rdkit import Chem
from features import compute_features
from gnn_model import (
    SolubilityGNN, MoleculeGraphEncoder, collate_graphs,
    save_gnn_model, load_gnn_model, transfer_backbone, ATOM_FEATURE_DIM,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cpu"

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

# ── 2. Train RF for comparison ──
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

# ── 3. Build GNN graphs from solubility data ──
print("\nBuilding GNN graphs...")
encoder = MoleculeGraphEncoder()
graphs, labels = [], []
for smi, logS in zip(df["SMILES"], df["logS"]):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    g = encoder.mol_to_graph(mol)
    if g is None:
        continue
    graphs.append(g)
    labels.append(logS)
print(f"  {len(graphs)} graphs")

# ── 4. Train/val split ──
train_i, val_i = train_test_split(
    np.arange(len(graphs)), test_size=0.2, random_state=SEED
)
train_graphs = [graphs[i] for i in train_i]
train_labels = np.array([labels[i] for i in train_i], dtype=np.float32)
val_graphs = [graphs[i] for i in val_i]
val_labels = np.array([labels[i] for i in val_i], dtype=np.float32)
print(f"  Train: {len(train_graphs)}  Val: {len(val_graphs)}")

# ── 5. Load pre-trained pKa weights ──
print("\nLoading pre-trained pKa backbone...")
pretrained_dict = torch.load("output_v2/gnn_pka_pretrained.pt", map_location=DEVICE, weights_only=True)

model = SolubilityGNN(atom_dim=ATOM_FEATURE_DIM, hidden_dim=128, num_layers=3).to(DEVICE)
n_loaded, n_total = transfer_backbone(model, pretrained_dict)
print(f"  Loaded {n_loaded}/{n_total} backbone layers (head re-initialized)")

# ── 6. Fine-tune on solubility ──
print("\n" + "=" * 50)
print("Fine-tuning GNN on solubility data...")
print("=" * 50)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
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
        save_gnn_model(model, "output_v2/gnn_solubility_model_v6.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

print(f"\nBest epoch: {best_epoch} | val loss: {best_val_loss:.4f}")

# ── 7. Evaluate on test_batch ──
print("\n" + "=" * 50)
print("3-way comparison on test_batch.csv (v2 vs v3 vs v6)")
print("=" * 50)

df_test = pd.read_csv("test_batch.csv")
rf_v2 = joblib.load("output_v2/solubility_model_v2.pkl")
gnn_v2 = load_gnn_model("output_v2/gnn_solubility_model.pt", device=DEVICE)
gnn_v3 = load_gnn_model("output_v2/gnn_solubility_model_v3.pt", device=DEVICE)
gnn_v6 = model  # best fine-tuned model

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
    with torch.no_grad():
        g2p = float(gnn_v2(encoder.mol_to_graph(mol)).item())
        g3p = float(gnn_v3(encoder.mol_to_graph(mol)).item())
        g6p = float(gnn_v6(encoder.mol_to_graph(mol)).item())

    rows.append({
        "SMILES": smi,
        "RF_v2": rf2,
        "GNN_v2": g2p, "GNN_v3": g3p, "GNN_v6": g6p,
        "|RF-GNN|_v2": abs(rf2 - g2p),
        "|RF-GNN|_v3": abs(rf2 - g3p),
        "|RF-GNN|_v6": abs(rf2 - g6p),
    })

res = pd.DataFrame(rows)
print(f"\n  {'Model':<20} {'Mean |RF-GNN|':>15}")
print(f"  {'-'*35}")
for ver in ["v2", "v3", "v6"]:
    col = f"|RF-GNN|_{ver}"
    print(f"  {'GNN ' + ver:<20} {res[col].mean():>15.3f}")

print(f"\n  Reduction from v2:")
for ver in ["v3", "v6"]:
    col = f"|RF-GNN|_{ver}"
    red = res['|RF-GNN|_v2'].mean() - res[col].mean()
    pct = red / res['|RF-GNN|_v2'].mean() * 100
    print(f"  {ver}: {red:+.3f} ({pct:+.1f}%)")

print(f"\n  Hard cases:")
for _, r in res.sort_values("|RF-GNN|_v2", ascending=False).head(5).iterrows():
    smi_short = r["SMILES"][:40]
    print(f"    {smi_short:42s} v2={r['GNN_v2']:+.2f} v3={r['GNN_v3']:+.2f} v6={r['GNN_v6']:+.2f} |RF|={r['RF_v2']:+.2f}")

print("\nModel saved: output_v2/gnn_solubility_model_v6.pt")
