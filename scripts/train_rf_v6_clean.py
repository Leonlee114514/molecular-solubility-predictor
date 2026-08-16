"""
Train and persist the clean RF solubility model (V6).

Uses the same seed=2026 outer test split as scripts/evaluate_models.py and
scripts/train_gnn_clean.py. The RF is trained only on the non-test pool and
evaluated on the untouched test set.

Outputs:
    output_v2/solubility_model_v6_clean.pkl.gz
    output_v2/descriptor_names_v6_clean.pkl
    output_v2/v6_clean_config.json
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402

from features import compute_features  # noqa: E402

TEST_SEED = 2026
TEST_SIZE = 0.2
RF_PARAMS = {
    "n_estimators": 800,
    "max_depth": 30,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": None,
    "random_state": 42,
    "n_jobs": -1,
}

MODEL_OUT = PROJECT_ROOT / "output_v2" / "solubility_model_v6_clean.pkl.gz"
DESC_OUT = PROJECT_ROOT / "output_v2" / "descriptor_names_v6_clean.pkl"
CONFIG_OUT = PROJECT_ROOT / "output_v2" / "v6_clean_config.json"

DATASETS = [
    ("ESOL", "data/delaney.csv", {"Compound ID": "ID", "measured log(solubility:mol/L)": "logS"}),
    ("AqSolDB", "curated-solubility-dataset.csv", None),
    ("Supplementary", "supplementary_logs.csv", None),
    ("ChEMBL", "chembl_solubility.csv", None),
]


def _load_data() -> pd.DataFrame:
    frames = []
    for name, rel_path, cols in DATASETS:
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            print(f"  [skip] {name}", flush=True)
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
    data = pd.concat(frames, ignore_index=True)
    return data.drop_duplicates(subset=["SMILES"], keep="first").reset_index(drop=True)


def _metrics(y_true, y_pred):
    return {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _per_source(y_true, y_pred, sources):
    out = {}
    for src in sorted(set(sources)):
        mask = sources == src
        if mask.sum() >= 2:
            out[str(src)] = _metrics(y_true[mask], y_pred[mask])
    return out


def main() -> None:
    print("Clean RF V6 training", flush=True)
    data = _load_data()
    print(f"  unique molecules: {len(data)}", flush=True)

    t0 = time.time()
    features_list, fps, valid_idx = [], [], []
    for idx, row in data.iterrows():
        result = compute_features(row["SMILES"])
        if result is None:
            continue
        features, fp = result
        features_list.append(features)
        fps.append(fp)
        valid_idx.append(idx)
    valid = data.loc[valid_idx].reset_index(drop=True)
    X = np.vstack([np.hstack([list(f.values()), fp]) for f, fp in zip(features_list, fps)])
    y = valid["logS"].to_numpy(dtype=float)
    source = valid["source"].to_numpy()
    desc_names = list(features_list[0].keys())
    print(f"  valid molecules: {len(valid)} in {time.time() - t0:.1f}s", flush=True)

    strata = np.array([1 if s == "AqSolDB" else 0 for s in source])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=TEST_SEED)
    train_idx, test_idx = next(splitter.split(X, strata))
    print(f"  train={len(train_idx)} test={len(test_idx)} (seed={TEST_SEED})", flush=True)

    t0 = time.time()
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X[train_idx], y[train_idx])
    print(f"  RF trained in {time.time() - t0:.1f}s", flush=True)

    train_metrics = _metrics(y[train_idx], model.predict(X[train_idx]))
    test_metrics = _metrics(y[test_idx], model.predict(X[test_idx]))
    test_pred = model.predict(X[test_idx])
    print(f"  train: {train_metrics}", flush=True)
    print(f"  test : {test_metrics}", flush=True)
    print(f"  per-source test: {_per_source(y[test_idx], test_pred, source[test_idx])}", flush=True)

    with gzip.open(MODEL_OUT, "wb") as f:
        joblib.dump(model, f, compress=3)
    joblib.dump(desc_names, DESC_OUT)

    config = {
        "model_version": "v6_clean",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rf_params": RF_PARAMS,
        "n_features": int(X.shape[1]),
        "n_desc": len(desc_names),
        "descriptors": desc_names,
        "split": {"test_seed": TEST_SEED, "test_size": TEST_SIZE},
        "data": {
            "n_total": int(len(data)),
            "n_valid": int(len(valid)),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        },
        "metrics": {
            "train": train_metrics,
            "test": test_metrics,
            "test_per_source": _per_source(y[test_idx], test_pred, source[test_idx]),
        },
        "model_file": MODEL_OUT.name,
        "descriptor_file": DESC_OUT.name,
    }
    CONFIG_OUT.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {MODEL_OUT} ({MODEL_OUT.stat().st_size / 1e6:.1f} MB)", flush=True)
    print(f"Saved: {DESC_OUT}", flush=True)
    print(f"Saved: {CONFIG_OUT}", flush=True)


if __name__ == "__main__":
    main()
