"""
DisSolve - Independent model evaluation report.

Creates a stratified hold-out test set (NOT the split used in train_rf_v5.py)
and reports:

  * RF metrics after retraining on the new train split (uncontaminated)
  * The shipped RF / GNN artifact metrics on the same test split
    (clearly labelled as potentially contaminated, because the shipped
    models may have seen some of these molecules during training)
  * Per-source R2 / RMSE / MAE
  * Strategy comparison: RF, GNN, simple average, 0.45/0.55 ensemble,
    and the Auto+ (OOD + disagreement) selection rule

Usage:
    python scripts/evaluate_models.py
    python scripts/evaluate_models.py --quick
    python scripts/evaluate_models.py --skip-retrain
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from features import compute_features  # noqa: E402
from ood_detector import load_ood_detector  # noqa: E402

RF_ARTIFACT_CANDIDATES = [
    PROJECT_ROOT / "output_v2" / "solubility_model_v6_clean.pkl.gz",
    PROJECT_ROOT / "output_v2" / "solubility_model_v5.pkl.gz",
]
RF_ARTIFACT = next((p for p in RF_ARTIFACT_CANDIDATES if p.exists()), RF_ARTIFACT_CANDIDATES[0])
GNN_CLEAN_CANDIDATE = PROJECT_ROOT / "output_v2" / "gnn_solubility_model_v5_clean.pt"
GNN_CANDIDATES = [
    (GNN_CLEAN_CANDIDATE, 256),
    (PROJECT_ROOT / "output_v2" / "gnn_solubility_model_v4.pt", 256),
    (PROJECT_ROOT / "output_v2" / "gnn_solubility_model_v3.pt", 128),
    (PROJECT_ROOT / "output_v2" / "gnn_solubility_model.pt", 128),
]
OOD_ARTIFACT = PROJECT_ROOT / "output_v2" / "ood_detector.pkl.gz"
REPORT_PATH = PROJECT_ROOT / "output_v2" / "evaluation_report.json"

RF_PARAMS = {
    "n_estimators": 800,
    "max_depth": 30,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": None,
    "random_state": 42,
    "n_jobs": -1,
}

DATASETS = [
    (
        "ESOL",
        "data/delaney.csv",
        {
            "Compound ID": "ID",
            "measured log(solubility:mol/L)": "logS",
            "ESOL predicted log(solubility:mol/L)": "_esol_pred",
        },
    ),
    ("AqSolDB", "curated-solubility-dataset.csv", None),
    ("Supplementary", "supplementary_logs.csv", None),
    ("ChEMBL", "chembl_solubility.csv", None),
]


def _load_joblib(path: Path):
    if str(path).endswith(".gz"):
        import gzip
        with gzip.open(path, "rb") as f:
            return joblib.load(f)
    return joblib.load(path)


def load_dataset(limit=None):
    frames = []
    for name, rel_path, cols in DATASETS:
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            print(f"  [skip] {name}: {path} not found")
            continue
        df = pd.read_csv(path)
        if cols:
            df = df.rename(columns=cols)[["SMILES", "logS"]]
        elif "SMILES" not in df.columns or "logS" not in df.columns:
            smiles_col = next(c for c in df.columns if "smiles" in c.lower())
            sol_col = next(
                c for c in df.columns
                if "solubility" in c.lower() or "logs" in c.lower()
            )
            df = df[[smiles_col, sol_col]].rename(
                columns={smiles_col: "SMILES", sol_col: "logS"}
            )
        df["source"] = name
        frames.append(df)
        print(f"  {name}: {len(df)}")

    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(subset=["SMILES"], keep="first").reset_index(drop=True)
    if limit:
        data = data.head(limit)
    print(f"  unique molecules: {len(data)}")
    return data


def compute_all_features(data):
    t0 = time.time()
    features_list, fps, valid_rows = [], [], []
    for idx, row in data.iterrows():
        result = compute_features(row["SMILES"])
        if result is None:
            continue
        features, fp = result
        features_list.append(features)
        fps.append(fp)
        valid_rows.append(idx)
    valid = data.loc[valid_rows].reset_index(drop=True)
    print(f"  features: {len(features_list)} molecules in {time.time() - t0:.1f}s")
    return features_list, fps, valid


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return {"n": int(len(y_true)), "r2": None, "rmse": None, "mae": None}
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"n": int(len(y_true)), "r2": r2, "rmse": rmse, "mae": mae}


def load_gnn_artifact():
    import torch
    from gnn_model import ATOM_FEATURE_DIM, MoleculeGraphEncoder, SolubilityGNN

    for path, hidden_dim in GNN_CANDIDATES:
        if path.exists():
            encoder = MoleculeGraphEncoder()
            model = SolubilityGNN(
                atom_dim=ATOM_FEATURE_DIM, hidden_dim=hidden_dim, num_layers=3
            )
            model.load_state_dict(
                torch.load(str(path), map_location="cpu", weights_only=True)
            )
            model.eval()
            source = "clean_candidate" if path == GNN_CLEAN_CANDIDATE else "shipped_artifact"
            print(f"  GNN {source}: {path.name} (hidden={hidden_dim})")
            return model, encoder, path.name, source
    return None, None, None, None


@dataclass
class StrategyRow:
    strategy: str
    r2: float | None
    rmse: float | None
    mae: float | None
    n: int
    note: str


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="limit to 4000 molecules")
    parser.add_argument(
        "--skip-retrain", action="store_true",
        help="do not retrain RF; shipped artifacts only",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    print("=" * 70)
    print("DisSolve model evaluation")
    print(f"test_size={args.test_size} seed={args.seed} quick={args.quick}")
    print("=" * 70)

    print("\n[1/5] Load datasets")
    limit = 4000 if args.quick else None
    data = load_dataset(limit=limit)

    print("\n[2/5] Compute features")
    features_list, fps, valid = compute_all_features(data)
    X = np.vstack([
        np.hstack([list(f.values()), fp]) for f, fp in zip(features_list, fps)
    ])
    y = valid["logS"].to_numpy(dtype=float)
    source = valid["source"].to_numpy()

    from sklearn.model_selection import StratifiedShuffleSplit

    strata = np.array([1 if s == "AqSolDB" else 0 for s in source])
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.seed
    )
    train_idx, test_idx = next(splitter.split(X, strata))
    print(f"  train={len(train_idx)} test={len(test_idx)}")

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "n_valid_molecules": int(len(valid)),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "sources": {str(s): int((source == s).sum()) for s in sorted(set(source))},
        },
        "split": {
            "method": "StratifiedShuffleSplit (stratum: AqSolDB vs other)",
            "seed": args.seed,
            "test_size": args.test_size,
        },
        "models": {},
        "strategies": {},
        "per_source": {},
        "caveats": [
            "The retrained RF metric is a clean hold-out evaluation for the RF model.",
            "The deployed RF artifact loaded by this script (V6 clean is preferred, "
            "V5 legacy is the fallback) may have seen some of these molecules during "
            "training if it is the legacy V5 artifact.",
            "The GNN source is recorded in strategies_gnn_source: clean_candidate "
            "(seed=2026 outer test split) or shipped_artifact (may be contaminated).",
        ],
    }

    print("\n[3/5] Load shipped RF and GNN artifacts")
    shipped_rf = None
    if RF_ARTIFACT.exists():
        shipped_rf = _load_joblib(RF_ARTIFACT)
        print(f"  RF artifact: {RF_ARTIFACT.name}")
    else:
        print("  RF artifact missing")

    gnn_model, gnn_encoder, gnn_name, gnn_source = load_gnn_artifact()

    rf_test_pred = None
    rf_retrained_test_pred = None
    if not args.skip_retrain:
        print("\n[4/5] Retrain RF on the new train split (clean hold-out)")
        from sklearn.ensemble import RandomForestRegressor

        t0 = time.time()
        rf_new = RandomForestRegressor(**RF_PARAMS)
        rf_new.fit(X[train_idx], y[train_idx])
        print(f"  RF trained in {time.time() - t0:.1f}s")
        rf_retrained_test_pred = rf_new.predict(X[test_idx])
        report["models"]["rf_retrained"] = {
            "train": metrics(y[train_idx], rf_new.predict(X[train_idx])),
            "test": metrics(y[test_idx], rf_retrained_test_pred),
            "params": RF_PARAMS,
        }
        print("  RF retrained train:", report["models"]["rf_retrained"]["train"])
        print("  RF retrained test :", report["models"]["rf_retrained"]["test"])

    if shipped_rf is not None:
        shipped_rf_pred = shipped_rf.predict(X[test_idx])
        report["models"]["rf_shipped_artifact"] = {
            "test": metrics(y[test_idx], shipped_rf_pred),
            "note": "same test set as rf_retrained; artifact may be contaminated",
        }
        print(
            "  RF shipped artifact test:",
            report["models"]["rf_shipped_artifact"]["test"],
        )
        if rf_test_pred is None:
            rf_test_pred = shipped_rf_pred

    gnn_test_pred = None
    if gnn_model is not None:
        import torch
        from rdkit import Chem

        t0 = time.time()
        gnn_preds = []
        for i in test_idx:
            mol = Chem.MolFromSmiles(valid.iloc[i]["SMILES"])
            if mol is None:
                gnn_preds.append(np.nan)
                continue
            graph = gnn_encoder.mol_to_graph(mol)
            if graph is None:
                gnn_preds.append(np.nan)
                continue
            with torch.no_grad():
                gnn_preds.append(float(gnn_model(graph).item()))
        gnn_test_pred = np.array(gnn_preds)
        print(
            f"  GNN inferred {np.isfinite(gnn_test_pred).sum()} test molecules "
            f"in {time.time() - t0:.1f}s"
        )
        gnn_key = "gnn_clean_candidate" if gnn_source == "clean_candidate" else "gnn_shipped_artifact"
        gnn_note = (
            "clean hold-out model; trained on the non-test pool only"
            if gnn_source == "clean_candidate"
            else "artifact may be contaminated"
        )
        report["models"][gnn_key] = {
            "file": gnn_name,
            "source": gnn_source,
            "test": metrics(y[test_idx], gnn_test_pred),
            "note": gnn_note,
        }
        report["strategies_gnn_source"] = gnn_source
        print(
            f"  GNN {gnn_source} test:",
            report["models"][gnn_key]["test"],
        )

    if rf_test_pred is not None and gnn_test_pred is not None:
        print("\n[5/5] Strategy comparison on hold-out test")
        # Prefer the clean retrained RF for the strategy table. If a clean GNN
        # candidate exists it is loaded first; otherwise the shipped artifact is
        # used and its contamination caveat is recorded in the report.
        rf_v = rf_retrained_test_pred if rf_retrained_test_pred is not None else rf_test_pred
        strategy_rf_source = (
            "rf_retrained (clean hold-out)"
            if rf_retrained_test_pred is not None
            else "rf_shipped_artifact"
        )
        report["strategies_rf_source"] = strategy_rf_source
        gn_v = gnn_test_pred
        finite = np.isfinite(rf_v) & np.isfinite(gn_v)

        ood = load_ood_detector(str(OOD_ARTIFACT)) if OOD_ARTIFACT.exists() else None
        auto_pred, auto_used = [], []
        # Current Auto strategy: weighted 0.5/0.5 ensemble whenever both models
        # are available. OOD is still evaluated for warning purposes only.
        for j, i in enumerate(test_idx):
            if not finite[j]:
                auto_pred.append(np.nan)
                auto_used.append("nan")
            else:
                auto_pred.append(0.5 * rf_v[j] + 0.5 * gn_v[j])
                auto_used.append("Ensemble(W)")

        rows = [
            StrategyRow("RF", **metrics(y[test_idx], rf_v), note=""),
            StrategyRow("GNN", **metrics(y[test_idx], gn_v), note=""),
            StrategyRow(
                "Ensemble (0.5/0.5)",
                **metrics(y[test_idx], 0.5 * rf_v + 0.5 * gn_v),
                note="",
            ),
            StrategyRow(
                "Auto (0.5/0.5 ensemble)",
                **metrics(y[test_idx], np.array(auto_pred)),
                note="OOD/disagreement are warnings, not routing",
            ),
        ]
        report["strategies"] = [asdict(r) for r in rows]
        if ood is not None:
            from collections import Counter
            report["strategies_auto_model_usage"] = dict(Counter(auto_used))
        for r in rows:
            print(
                f"  {r.strategy:<32} R2={r.r2 if r.r2 is not None else float('nan'):.4f} "
                f"RMSE={r.rmse if r.rmse is not None else float('nan'):.4f} "
                f"MAE={r.mae if r.mae is not None else float('nan'):.4f} n={r.n}"
            )

    print("\n  Per-source metrics:")
    per_source = {}
    pred_sets = [
        ("shipped_rf", rf_test_pred),
        ("retrained_rf", rf_retrained_test_pred),
    ]
    for key, preds in pred_sets:
        if preds is None:
            continue
        per_source[key] = {}
        print(f"  [{key}]")
        for s in sorted(set(source)):
            mask = np.array([s == source[i] for i in test_idx])
            if mask.sum() >= 2:
                m = metrics(y[test_idx][mask], preds[mask])
                per_source[key][str(s)] = m
                print(
                    f"    {s:<14} n={m['n']:>5} "
                    f"R2={m['r2']:.4f} RMSE={m['rmse']:.4f} MAE={m['mae']:.4f}"
                )
    report["per_source"] = per_source

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
