"""
DisSolve - framework-free unified prediction service.

Single entry point for molecular property prediction (aqueous solubility logS,
pKa, OOD detection, SHAP explainability). This layer backs both the future
FastAPI backend and the legacy Streamlit app (Phase 0 of the FastAPI + React
migration).

Hard constraints for this module:
  * NO streamlit imports, NO core.i18n imports, NO presentation strings
    (no colors, no CSS classes, no translated text). Raw enums only.
  * Importable and fully functional in a plain python process.
  * Mirrors app.py's single-prediction semantics exactly:
      - RF always runs;
      - pKa is lazy (failure -> None);
      - "auto" uses OOD -> auto model-selection strategy;
      - "gnn"/"ensemble" fall back to RF when GNN is unavailable;
      - SHAP is skipped when the resolved model is GNN-only;
      - OOD also runs for non-auto modes.
  * Single and batch predictions share ONE code path (_predict_one).

Model artifacts (read-only, under <project root>/output_v2/):
  * solubility_model_v5.pkl.gz + descriptor_names_v5.pkl  (Random Forest)
  * pka_model.pkl                                         (pKa regressor)
  * gnn_solubility_model_v4.pt / _v3.pt / .pt             (GNN, optional)
  * ood_detector.pkl.gz                                   (OOD reference stats)
"""

from __future__ import annotations

import gzip
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from features import compute_features

logger = logging.getLogger(__name__)

# ── Paths (resolved from the project root, not the CWD) ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _PROJECT_ROOT / "output_v2"

_SOLUBILITY_MODEL_PATH = _OUTPUT_DIR / "solubility_model_v5.pkl.gz"
_DESCRIPTOR_NAMES_PATH = _OUTPUT_DIR / "descriptor_names_v5.pkl"
_PKA_MODEL_PATH = _OUTPUT_DIR / "pka_model.pkl"
_OOD_DETECTOR_PATH = _OUTPUT_DIR / "ood_detector.pkl.gz"

# Same priority / hidden-dim mapping as model.load_gnn_model (V4 -> V3 -> V2).
_GNN_CANDIDATES = (
    ("gnn_solubility_model_v4.pt", 256),
    ("gnn_solubility_model_v3.pt", 128),
    ("gnn_solubility_model.pt", 128),
)

# pKa model was trained on these 8 core descriptors + 1024-bit Morgan FP
# (extracted from app.py's pKa feature vector construction).
_PKA_FEATURE_KEYS = (
    "MolWt", "LogP", "NumHDonors", "NumHAcceptors",
    "TPSA", "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
)

_VALID_MODES = ("auto", "rf", "gnn", "ensemble")

# Resolved models for which SHAP is skipped (app.py: shap_disabled_models).
_SHAP_DISABLED_MODELS = frozenset({"GNN"})


@dataclass
class PredictionResult:
    """Framework-free prediction payload for one molecule (raw enums, no i18n)."""

    smiles: str                       # canonical input smiles (stripped input)
    features: dict[str, float]        # 13 descriptors, raw English keys
    logS_rf: float
    logS_gnn: float | None
    logS_final: float
    model_selected: str               # "auto"|"rf"|"gnn"|"ensemble" (echo of request)
    model_used: str                   # "RF"|"GNN"|"Ensemble"|"Ensemble(W)"
    model_disagreement: float | None  # abs(rf-gnn) when both available, else None
    pka: float | None
    pka_kind: str | None              # "acid"|"base"|"amphoteric" (raw enum)
    ood_risk: str                     # "LOW"|"MEDIUM"|"HIGH"|"UNKNOWN"
    ood_score: float | None
    ood_max_tanimoto: float | None
    ood_out_of_range: list[str]       # raw descriptor keys outside training min/max
    ood_extreme: list[str]            # raw descriptor keys with |z| > 3
    shap_values: list[float] | None   # None when model_used is GNN-only
    shap_names: list[str] | None      # 13 raw descriptor keys + "MorganFP"
    shap_base_value: float | None


# ── Lazy module-level singletons ──
# Streamlit-free replacements for model.py's @st.cache_resource loaders,
# using the same file paths and loading logic. A single lock is sufficient:
# loading is idempotent and infrequent.
_lock = threading.Lock()
_solubility_model = None
_descriptor_names = None
_pka_model = None
_pka_loaded = False
_ood_detector = None
_ood_loaded = False
_gnn_model = None
_gnn_encoder = None
_gnn_loaded = False
_shap_explainer = None


def _load_joblib(path):
    """Load a joblib file, transparently handling .gz compression."""
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return joblib.load(f)
    return joblib.load(path)


def _get_solubility_model():
    """Lazy-load the RF solubility model (V5) + descriptor names."""
    global _solubility_model, _descriptor_names
    if _solubility_model is None:
        with _lock:
            if _solubility_model is None:
                logger.info("Loading RF solubility model from %s", _SOLUBILITY_MODEL_PATH)
                _solubility_model = _load_joblib(_SOLUBILITY_MODEL_PATH)
                _descriptor_names = joblib.load(_DESCRIPTOR_NAMES_PATH)
    return _solubility_model, _descriptor_names


def _get_pka_model():
    """Lazy-load the pKa model. Returns None if the file is missing/unreadable."""
    global _pka_model, _pka_loaded
    if not _pka_loaded:
        with _lock:
            if not _pka_loaded:
                try:
                    if _PKA_MODEL_PATH.exists():
                        logger.info("Loading pKa model from %s", _PKA_MODEL_PATH)
                        _pka_model = joblib.load(_PKA_MODEL_PATH)
                    else:
                        logger.warning("pKa model not found at %s", _PKA_MODEL_PATH)
                except Exception:
                    logger.exception("Failed to load pKa model")
                    _pka_model = None
                _pka_loaded = True
    return _pka_model


def _get_ood_detector():
    """Lazy-load the OOD detector. Returns None if unavailable."""
    global _ood_detector, _ood_loaded
    if not _ood_loaded:
        with _lock:
            if not _ood_loaded:
                try:
                    from ood_detector import load_ood_detector as _load_ood
                    logger.info("Loading OOD detector from %s", _OOD_DETECTOR_PATH)
                    _ood_detector = _load_ood(str(_OOD_DETECTOR_PATH))
                except Exception:
                    logger.exception("Failed to load OOD detector")
                    _ood_detector = None
                _ood_loaded = True
    return _ood_detector


def gnn_files_exist():
    """Quick file-existence check for GNN weights (mirrors app.py's gnn_ready)."""
    return any((_OUTPUT_DIR / name).exists() for name, _ in _GNN_CANDIDATES)


def _get_gnn():
    """Lazy-load the GNN model + graph encoder. Returns (model, encoder) or (None, None)."""
    global _gnn_model, _gnn_encoder, _gnn_loaded
    if not _gnn_loaded:
        with _lock:
            if not _gnn_loaded:
                try:
                    import torch

                    from gnn_model import ATOM_FEATURE_DIM, MoleculeGraphEncoder, SolubilityGNN

                    model_path = None
                    hidden_dim = 128
                    for fname, hdim in _GNN_CANDIDATES:
                        candidate = _OUTPUT_DIR / fname
                        if candidate.exists():
                            model_path, hidden_dim = candidate, hdim
                            break
                    if model_path is None:
                        logger.warning("No GNN model file found in %s", _OUTPUT_DIR)
                    else:
                        logger.info(
                            "Loading GNN model from %s (hidden_dim=%d)", model_path, hidden_dim
                        )
                        encoder = MoleculeGraphEncoder()
                        model = SolubilityGNN(
                            atom_dim=ATOM_FEATURE_DIM, hidden_dim=hidden_dim, num_layers=3
                        )
                        state = torch.load(str(model_path), map_location="cpu", weights_only=True)
                        model.load_state_dict(state)
                        model.eval()
                        _gnn_model, _gnn_encoder = model, encoder
                except Exception:
                    logger.exception("Failed to load GNN model")
                    _gnn_model, _gnn_encoder = None, None
                _gnn_loaded = True
    return _gnn_model, _gnn_encoder


def _get_shap_explainer(rf_model):
    """Lazy-create the SHAP TreeExplainer for the RF model."""
    global _shap_explainer
    if _shap_explainer is None:
        with _lock:
            if _shap_explainer is None:
                import shap
                logger.info("Creating SHAP TreeExplainer")
                _shap_explainer = shap.TreeExplainer(rf_model)
    return _shap_explainer


# ── Pure inference helpers (mirrored from model.py, kept streamlit-free) ──

def _gnn_predict(smiles):
    """GNN prediction for a single SMILES; None when unavailable or failed."""
    model, encoder = _get_gnn()
    if model is None:
        return None
    try:
        import torch
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        graph = encoder.mol_to_graph(mol)
        if graph is None:
            return None
        with torch.no_grad():
            pred = model(graph)
        return float(pred.item())
    except Exception:
        logger.exception("GNN prediction failed for SMILES %r", smiles)
        return None


def _ensemble(rf_pred, gnn_pred):
    """Weighted average 0.45*RF + 0.55*GNN (model.predict_solubility_ensemble)."""
    return 0.45 * rf_pred + 0.55 * gnn_pred


def _auto_select(ood_risk, rf_pred, gnn_pred):
    """Auto+ strategy (model.predict_solubility_auto):
      - GNN missing     -> ("RF", rf_pred)
      - |RF-GNN| > 1.0  -> pure GNN (models fundamentally disagree, GNN is safer)
      - OOD != "LOW"    -> pure GNN (RF unreliable on outliers)
      - else            -> weighted ensemble ("Ensemble(W)")
    Returns (prediction, actual_model_label).
    """
    if gnn_pred is None:
        return rf_pred, "RF"
    disagreement = abs(rf_pred - gnn_pred)
    if disagreement > 1.0:
        return gnn_pred, "GNN"
    if ood_risk != "LOW":
        return gnn_pred, "GNN"
    return _ensemble(rf_pred, gnn_pred), "Ensemble(W)"


def _pka_kind(pka_val):
    """Raw acid/base/amphoteric enum (thresholds from model.get_pka_type)."""
    if pka_val < 6:
        return "acid"
    if pka_val > 8:
        return "base"
    return "amphoteric"


def _run_ood_check(features, fp_array):
    """Returns (risk_level, OODResult|None); ("UNKNOWN", None) when unavailable."""
    try:
        detector = _get_ood_detector()
        if detector is None:
            return "UNKNOWN", None
        result = detector.check(features, fp_array)
        return result.risk_level, result
    except Exception:
        logger.exception("OOD check failed")
        return "UNKNOWN", None


def _compute_shap(rf_model, features, fp_array):
    """SHAP contributions with RAW English names: 13 descriptor keys + "MorganFP".

    Replicates the aggregation logic of model.get_shap_contributions (descriptor
    values first, fingerprint bits summed into one entry) without translation.
    Returns (values, names, base_value) or (None, None, None) on failure.
    """
    try:
        X = np.hstack([list(features.values()), fp_array]).reshape(1, -1)
        explainer = _get_shap_explainer(rf_model)
        shap_values = explainer.shap_values(X)[0]
        n_desc = len(features)  # descriptor values come first in the RF input vector
        desc_shap = shap_values[:n_desc]
        fp_shap_sum = shap_values[n_desc:].sum()
        values = [float(v) for v in desc_shap] + [float(fp_shap_sum)]
        names = list(features.keys()) + ["MorganFP"]
        base = float(np.atleast_1d(explainer.expected_value).ravel()[0])
        return values, names, base
    except Exception:
        logger.exception("SHAP computation failed")
        return None, None, None


# ── Core prediction path (shared by single and batch) ──

def _normalize_mode(mode):
    """Normalize a mode string to one of "auto"|"rf"|"gnn"|"ensemble"."""
    normalized = (mode or "auto").strip().lower()
    if normalized not in _VALID_MODES:
        raise ValueError(f"Invalid mode {mode!r}; expected one of {_VALID_MODES}")
    return normalized


def _predict_one(smiles, mode):
    """Single-molecule prediction. mode must already be normalized.

    Raises ValueError for empty/invalid SMILES.
    """
    smiles = (smiles or "").strip()
    if not smiles:
        raise ValueError("Empty SMILES string")

    result = compute_features(smiles)
    if result is None:
        raise ValueError(f"Invalid SMILES (RDKit could not parse): {smiles!r}")
    features, fp_array = result
    # Dict insertion order is load-bearing: the RF input vector is
    # np.hstack([list(features.values()), fp_array]) exactly as in app.py.
    features = {k: float(v) for k, v in features.items()}
    X_input = np.hstack([list(features.values()), fp_array]).reshape(1, -1)

    # ── RF prediction (always needed) ──
    rf_model, _ = _get_solubility_model()
    rf_pred = float(rf_model.predict(X_input)[0])

    # ── pKa prediction (lazy; failure -> None, mirrors app.py) ──
    pka_val = None
    try:
        pka_model = _get_pka_model()
        if pka_model is not None:
            pka_feat = np.hstack([
                [features[k] for k in _PKA_FEATURE_KEYS],
                fp_array,
            ]).reshape(1, -1)
            pka_val = float(pka_model.predict(pka_feat)[0])
    except Exception:
        logger.exception("pKa prediction failed")
        pka_val = None

    # ── Model resolution + OOD (mirrors app.py single-prediction flow) ──
    gnn_pred = None
    if mode == "auto":
        ood_risk, ood_result = _run_ood_check(features, fp_array)
        # Auto always wants GNN when available (weighted ensemble and pure GNN both use it)
        if gnn_files_exist():
            gnn_pred = _gnn_predict(smiles)
        prediction, model_used = _auto_select(ood_risk, rf_pred, gnn_pred)
    else:
        if mode in ("gnn", "ensemble") and gnn_files_exist():
            gnn_pred = _gnn_predict(smiles)
        if mode == "gnn":
            prediction = gnn_pred if gnn_pred is not None else rf_pred
        elif mode == "ensemble":
            prediction = _ensemble(rf_pred, gnn_pred) if gnn_pred is not None else rf_pred
        else:
            prediction = rf_pred
        # For explicit modes app.py records the requested model as the actual one
        model_used = {"rf": "RF", "gnn": "GNN", "ensemble": "Ensemble"}[mode]
        # OOD also runs for non-auto modes
        ood_risk, ood_result = _run_ood_check(features, fp_array)

    disagreement = abs(rf_pred - gnn_pred) if gnn_pred is not None else None

    # ── SHAP (available for RF, Ensemble, Ensemble(W); skipped for GNN-only) ──
    if model_used in _SHAP_DISABLED_MODELS:
        shap_values = shap_names = shap_base = None
    else:
        shap_values, shap_names, shap_base = _compute_shap(rf_model, features, fp_array)

    return PredictionResult(
        smiles=smiles,
        features=features,
        logS_rf=rf_pred,
        logS_gnn=gnn_pred,
        logS_final=float(prediction),
        model_selected=mode,
        model_used=model_used,
        model_disagreement=disagreement,
        pka=pka_val,
        pka_kind=_pka_kind(pka_val) if pka_val is not None else None,
        ood_risk=ood_risk,
        ood_score=ood_result.overall_score if ood_result is not None else None,
        ood_max_tanimoto=ood_result.max_tanimoto if ood_result is not None else None,
        ood_out_of_range=list(ood_result.desc_out_of_range) if ood_result is not None else [],
        ood_extreme=list(ood_result.desc_extreme) if ood_result is not None else [],
        shap_values=shap_values,
        shap_names=shap_names,
        shap_base_value=shap_base,
    )


# ── Public API ──

def run_prediction(smiles: str, mode: str = "auto") -> PredictionResult:
    """Predict logS / pKa / OOD / SHAP for a single SMILES.

    Args:
        smiles: SMILES string (stripped before use).
        mode:   "auto" | "rf" | "gnn" | "ensemble" (case-insensitive).

    Returns:
        PredictionResult with raw enums (no translated/presentation strings).

    Raises:
        ValueError: for empty/invalid SMILES or an unknown mode.
    """
    return _predict_one(smiles, _normalize_mode(mode))


def predict_batch(smiles_list: list[str], mode: str = "auto") -> list:
    """Batch prediction sharing the exact single-prediction code path.

    Per-row failures are captured as {"smiles": s, "error": str(e)} dicts
    without aborting the batch; successful rows are PredictionResult objects.
    """
    normalized = _normalize_mode(mode)
    results = []
    for smi in smiles_list:
        try:
            results.append(_predict_one(smi, normalized))
        except Exception as e:
            logger.warning("Batch row failed for SMILES %r: %s", smi, e)
            results.append({"smiles": smi, "error": str(e)})
    return results
