"""
Clean retraining of the solubility GNN for independent evaluation.

Why this script exists
----------------------
The shipped GNN (`gnn_solubility_model_v4.pt`) was trained with a seed=42
validation split and may have been selected/tuned against that split. To get
a trustworthy comparison, this script:

  1. Loads the exact same 4 datasets as V4 (ESOL + AqSolDB + Supplementary + ChEMBL).
  2. Creates the SAME seed=2026 stratified test split used by
     scripts/evaluate_models.py, so the test molecules are untouched.
  3. Splits the remaining 80% into train/val for early stopping.
  4. Trains the V4-optimised GNN config (hidden=256, 3 layers, lr=1e-3).
  5. Saves a candidate model and a machine-readable report:
       output_v2/gnn_solubility_model_v5_clean.pt
       output_v2/gnn_clean_config.json

This does NOT overwrite any production model. The candidate is only promoted
after comparing it against the shipped models on the held-out test set.

Usage:
    python scripts/train_gnn_clean.py
    python scripts/train_gnn_clean.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem  # noqa: E402
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # noqa: E402
from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402
from gnn_model import (  # noqa: E402
    ATOM_FEATURE_DIM,
    MoleculeGraphEncoder,
    SolubilityGNN,
    collate_graphs,
    save_gnn_model,
)

# Same split used by scripts/evaluate_models.py for the clean RF hold-out.
TEST_SEED = 2026
VAL_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.20  # fraction of the non-test pool

# V4-optimised config (cfg05 in output_v2/gnn_hparam_results.json).
GNN_HIDDEN = 256
GNN_LAYERS = 3
GNN_LR = 1e-3
GNN_DROPOUT = 0.1
GNN_BATCH = 64
GNN_EPOCHS = 200
GNN_PATIENCE = 30

MODEL_OUT = PROJECT_ROOT / "output_v2" / "gnn_solubility_model_v5_clean.pt"
CONFIG_OUT = PROJECT_ROOT / "output_v2" / "gnn_clean_config.json"

DATASETS = [
    ("ESOL", "data/delaney.csv", {"Compound ID": "ID", "measured log(solubility:mol/L)": "logS"}),
    ("AqSolDB", "curated-solubility-dataset.csv", None),
    ("Supplementary", "supplementary_logs.csv", None),
    ("ChEMBL", "chembl_solubility.csv", None),
]


def _load_data(limit: int | None) -> pd.DataFrame:
    frames = []
    for name, rel_path, cols in DATASETS:
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            print(f"  [skip] {name}: not found", flush=True)
            continue
        df = pd.read_csv(path)
        if cols:
            df = df.rename(columns=cols)[["SMILES", "logS"]]
        elif "SMILES" not in df.columns or "logS" not in df.columns:
            smiles_col = next(c for c in df.columns if "smiles" in c.lower())
            sol_col = next(c for c in df.columns if "solubility" in c.lower() or "logs" in c.lower())
            df = df[[smiles_col, sol_col]].rename(columns={smiles_col: "SMILES", sol_col: "logS"})
        df["source"] = name
        frames.append(df)
        print(f"  {name}: {len(df)}", flush=True)

    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(subset=["SMILES"], keep="first").reset_index(drop=True)
    if limit:
        data = data.head(limit)
    print(f"  unique molecules: {len(data)}", flush=True)
    return data


def _build_graphs(data: pd.DataFrame):
    encoder = MoleculeGraphEncoder()
    graphs, labels, sources = [], [], []
    for _, row in data.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            continue
        graph = encoder.mol_to_graph(mol)
        if graph is None:
            continue
        graphs.append(graph)
        labels.append(float(row["logS"]))
        sources.append(row["source"])
    return graphs, np.array(labels, dtype=np.float32), np.array(sources)


def _metrics(y_true, y_pred):
    return {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _per_source_metrics(y_true, y_pred, sources):
    out = {}
    for src in sorted(set(sources)):
        mask = sources == src
        if mask.sum() >= 2:
            out[str(src)] = _metrics(y_true[mask], y_pred[mask])
    return out


def _predict_batches(model, graphs, labels, batch_size=64):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch_data = collate_graphs(graphs[start:start + batch_size])
            if batch_data is None:
                continue
            pred = model(batch_data)
            preds.append(pred.cpu())
    return torch.cat(preds).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="use 4000 molecules")
    parser.add_argument("--epochs", type=int, default=GNN_EPOCHS)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72, flush=True)
    print("Clean SolubilityGNN training", flush=True)
    print(f"device={device} threads={args.threads} quick={args.quick} epochs={args.epochs}", flush=True)
    print("=" * 72, flush=True)

    print("\n[1/5] Load datasets", flush=True)
    data = _load_data(4000 if args.quick else None)

    print("\n[2/5] Build molecular graphs", flush=True)
    t0 = time.time()
    graphs, labels, sources = _build_graphs(data)
    print(f"  graphs={len(graphs)} in {time.time() - t0:.1f}s", flush=True)

    strata = np.array([1 if s == "AqSolDB" else 0 for s in sources])
    outer = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=TEST_SEED)
    pool_idx, test_idx = next(outer.split(np.arange(len(graphs)), strata))
    print(f"  pool={len(pool_idx)} test={len(test_idx)} (seed={TEST_SEED})", flush=True)

    inner = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=VAL_SEED)
    train_idx, val_idx = next(inner.split(pool_idx, strata[pool_idx]))
    train_idx, val_idx = pool_idx[train_idx], pool_idx[val_idx]
    print(f"  train={len(train_idx)} val={len(val_idx)} (inner seed={VAL_SEED})", flush=True)

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    print("\n[3/5] Build model", flush=True)
    model = SolubilityGNN(atom_dim=ATOM_FEATURE_DIM, hidden_dim=GNN_HIDDEN, num_layers=GNN_LAYERS)
    model.head = nn.Sequential(
        nn.Linear(GNN_HIDDEN, GNN_HIDDEN // 2),
        nn.ReLU(),
        nn.Dropout(GNN_DROPOUT),
        nn.Linear(GNN_HIDDEN // 2, 1),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,} hidden={GNN_HIDDEN} layers={GNN_LAYERS} batch={GNN_BATCH}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=GNN_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    criterion = nn.MSELoss()

    print("\n[4/5] Train", flush=True)
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    train_start = time.time()
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = np.random.permutation(len(train_graphs))
        train_losses = []

        for start in range(0, len(train_graphs), GNN_BATCH):
            batch_idx = perm[start:start + GNN_BATCH]
            batch_data = collate_graphs([train_graphs[i] for i in batch_idx])
            batch_y = torch.tensor(train_labels[batch_idx], dtype=torch.float32)
            if batch_data is None:
                continue
            pred = model(batch_data)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        val_pred = _predict_batches(model, val_graphs, val_labels, GNN_BATCH)
        val_true = val_labels[:len(val_pred)]
        val_loss = float(criterion(torch.tensor(val_pred), torch.tensor(val_true)).item())
        val_r2 = float(r2_score(val_true, val_pred))
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        history.append({"epoch": epoch, "val_loss": val_loss, "val_r2": val_r2, "seconds": round(elapsed, 1)})

        if epoch <= 3 or epoch % 10 == 0 or val_loss < best_val_loss:
            print(
                f"  epoch {epoch:3d}/{args.epochs} | train_loss={np.mean(train_losses):.4f} "
                f"val_loss={val_loss:.4f} val_r2={val_r2:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} time={elapsed:.1f}s",
                flush=True,
            )

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            save_gnn_model(model, str(MODEL_OUT))
        else:
            patience_counter += 1
            if patience_counter >= GNN_PATIENCE:
                print(f"  early stop at epoch {epoch}", flush=True)
                break

    train_seconds = time.time() - train_start
    print(f"\n  best_epoch={best_epoch} best_val_loss={best_val_loss:.4f} total={train_seconds:.1f}s", flush=True)

    print("\n[5/5] Evaluate candidate on held-out test", flush=True)
    state = torch.load(str(MODEL_OUT), map_location=device, weights_only=True)
    model = SolubilityGNN(
        atom_dim=ATOM_FEATURE_DIM, hidden_dim=GNN_HIDDEN, num_layers=GNN_LAYERS
    )
    model.load_state_dict(state)
    model.eval()
    test_pred = _predict_batches(model, test_graphs, test_labels, GNN_BATCH)
    test_true = test_labels[:len(test_pred)]
    test_sources = sources[test_idx][:len(test_pred)]

    val_idx_arr = np.asarray(val_idx)
    val_pred = _predict_batches(model, val_graphs, val_labels, GNN_BATCH)
    val_true = val_labels[:len(val_pred)]
    val_sources = sources[val_idx_arr[:len(val_pred)]]

    config = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "torch_version": torch.__version__,
        "seed": 42,
        "split": {
            "outer_test_seed": TEST_SEED,
            "outer_test_size": TEST_SIZE,
            "inner_val_seed": VAL_SEED,
            "inner_val_size": VAL_SIZE,
            "same_test_split_as_evaluate_models": True,
        },
        "data": {
            "n_unique": int(len(data)),
            "n_graphs": int(len(graphs)),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "sources": {str(s): int((sources == s).sum()) for s in sorted(set(sources))},
        },
        "model": {
            "hidden_dim": GNN_HIDDEN,
            "num_layers": GNN_LAYERS,
            "dropout": GNN_DROPOUT,
            "batch_size": GNN_BATCH,
            "lr": GNN_LR,
            "max_epochs": args.epochs,
            "patience": GNN_PATIENCE,
            "n_params": n_params,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "train_seconds": round(train_seconds, 1),
        },
        "metrics": {
            "val": _metrics(val_true, val_pred),
            "val_per_source": _per_source_metrics(val_true, val_pred, val_sources),
            "test": _metrics(test_true, test_pred),
            "test_per_source": _per_source_metrics(test_true, test_pred, test_sources),
        },
        "model_file": MODEL_OUT.name,
    }
    CONFIG_OUT.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  val metrics: {config['metrics']['val']}", flush=True)
    print(f"  test metrics: {config['metrics']['test']}", flush=True)
    print(f"\nSaved: {MODEL_OUT}\nSaved: {CONFIG_OUT}", flush=True)


if __name__ == "__main__":
    main()
