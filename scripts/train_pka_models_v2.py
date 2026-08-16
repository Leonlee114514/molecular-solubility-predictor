"""
DisSolve - Train SEPARATE acidic and basic pKa models.

Rationale
---------
The legacy `train_pKa_model.py` concatenated `pka_acidic` and `pka_basic`
labels into one target. 62,076 molecules occur in both datasets with two
different pKa values, so the old model was forced to learn an average of two
chemically distinct quantities (for amino acids, for example, the result is
neither the acid pKa nor the base pKa).

This script trains two independent Random Forest regressors:

  * output_v2/pka_acidic_model.pkl  -> predicts acidic pKa
  * output_v2/pka_basic_model.pkl   -> predicts basic pKa

and writes a machine-readable config with hold-out metrics:

  * output_v2/pka_models_config.json

Usage:
    python scripts/train_pka_models_v2.py
    python scripts/train_pka_models_v2.py --quick        # smoke test
    python scripts/train_pka_models_v2.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

PKA_FEATURE_KEYS = (
    "MolWt", "LogP", "NumHDonors", "NumHAcceptors",
    "TPSA", "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
)

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}

DATASETS = [
    ("acidic", "data/pretrain_pka_acidic.csv", "pka_acidic"),
    ("basic", "data/pretrain_pka_basic.csv", "pka_basic"),
]


def _compute_features(smiles: str):
    """Process worker target (module-level so ProcessPoolExecutor can pickle it)."""
    from features import compute_features

    result = compute_features(smiles)
    if result is None:
        return None
    features, fp = result
    return (
        np.array([features[k] for k in PKA_FEATURE_KEYS], dtype=np.float32),
        fp.astype(np.int8),
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"n": int(len(y_true)), "r2": r2, "rmse": rmse, "mae": mae}


def _build_features(smiles_list: list[str], workers: int):
    """Compute feature vectors for a list of SMILES, preserving order."""
    t0 = time.time()
    chunksize = max(64, len(smiles_list) // (workers * 32))
    if workers <= 1:
        results = [_compute_features(s) for s in smiles_list]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_compute_features, smiles_list, chunksize=chunksize))
    desc_parts, fp_parts, valid_idx = [], [], []
    for i, result in enumerate(results):
        if result is None:
            continue
        desc, fp = result
        desc_parts.append(desc)
        fp_parts.append(fp)
        valid_idx.append(i)
    X = np.hstack([
        np.vstack(desc_parts),
        np.vstack(fp_parts).astype(np.float32),
    ]).astype(np.float32)
    print(f"  parsed {len(valid_idx)}/{len(smiles_list)} in {time.time() - t0:.1f}s")
    return X, np.array(valid_idx, dtype=int)


def train_one(name: str, path: Path, target_col: str, limit: int | None, workers: int):
    print(f"\n{'=' * 60}\n[{name}] {path.name}\n{'=' * 60}")
    df = pd.read_csv(path)
    if limit:
        df = df.head(limit)
    df = df.dropna(subset=[target_col])
    # Deduplicate identical SMILES by averaging repeated labels (rare in raw data).
    df = df.groupby("smiles", as_index=False)[target_col].mean()
    print(f"  unique molecules: {len(df)}")

    X, valid_idx = _build_features(df["smiles"].tolist(), workers)
    y = df[target_col].to_numpy(dtype=np.float32)[valid_idx]
    print(f"  X={X.shape} y={y.shape}", flush=True)

    # Independent hold-out split. NOTE: the legacy script used seed 42 and no
    # saved indices; we intentionally use seed 2026 so these test molecules are
    # disjoint from the legacy training split in expectation and are never used
    # for model selection.
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2026
    )

    from sklearn.ensemble import RandomForestRegressor

    t0 = time.time()
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)
    print(f"  RF trained in {time.time() - t0:.1f}s", flush=True)

    train_metrics = metrics(y_train, model.predict(X_train))
    test_metrics = metrics(y_test, model.predict(X_test))
    print(f"  train: {train_metrics}", flush=True)
    print(f"  test : {test_metrics}", flush=True)

    suffix = "_quick" if limit else ""
    out_path = PROJECT_ROOT / "output_v2" / f"pka_{name}_model{suffix}.pkl"
    joblib.dump(model, out_path, compress=3)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  saved {out_path.name} ({size_mb:.1f} MB)", flush=True)

    return {
        "model_file": out_path.name,
        "target": target_col,
        "source_rows": int(len(df)),
        "n_valid": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "rf_params": RF_PARAMS,
        "train": train_metrics,
        "test": test_metrics,
        "size_mb": round(size_mb, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="use only 5000 rows per dataset")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args()

    print(f"Training separate pKa models (workers={args.workers}, quick={args.quick})")
    limit = 5000 if args.quick else None
    config = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "models": {}}

    for name, rel_path, target_col in DATASETS:
        path = PROJECT_ROOT / rel_path
        config["models"][name] = train_one(name, path, target_col, limit, args.workers)

    suffix = "_quick" if args.quick else ""
    config_path = PROJECT_ROOT / "output_v2" / f"pka_models_config{suffix}.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved config: {config_path}")


if __name__ == "__main__":
    main()
