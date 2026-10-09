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
  * solubility_model_v6_clean.pkl.gz + descriptor_names_v6_clean.pkl  (RF)
  * solubility_model_v5.pkl.gz + descriptor_names_v5.pkl  (legacy RF fallback)
  * pka_acidic_model.pkl / pka_basic_model.pkl            (pKa regressors)
  * pka_model.pkl                                         (legacy mixed pKa, fallback)
  * gnn_solubility_model_v4.pt / _v3.pt / .pt             (GNN, optional)
  * ood_detector.pkl.gz                                   (OOD reference stats)
"""

from __future__ import annotations

import gzip
import logging
import threading
from dataclasses import dataclass

import joblib
import numpy as np

from core.artifacts import (
    OOD_DETECTOR,
    OUTPUT_DIR,
    PKA_ACIDIC_MODEL,
    PKA_BASIC_MODEL,
    PKA_LEGACY_MODEL,
    RF_CANDIDATES,
    existing_gnn,
    gnn_available,
)
from features import PKA_FEATURE_KEYS, compute_features
from pka_resolve import resolve_pka_pair

logger = logging.getLogger(__name__)

# ── Artifact paths: declared once in core.artifacts, resolved from the project
# root rather than the CWD. The module-level aliases below are kept because
# backend/routes.py reads them by attribute name for its health report.
_SOLUBILITY_MODEL_PATH = RF_CANDIDATES[0][0]
_DESCRIPTOR_NAMES_PATH = RF_CANDIDATES[0][1]
_PKA_ACIDIC_MODEL_PATH = PKA_ACIDIC_MODEL
_PKA_BASIC_MODEL_PATH = PKA_BASIC_MODEL
_PKA_LEGACY_MODEL_PATH = PKA_LEGACY_MODEL
_OOD_DETECTOR_PATH = OOD_DETECTOR

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
    pka: float | None               # primary pKa (closest to physiological pH 7)
    pka_kind: str | None              # "acid"|"base"|"amphoteric" (raw enum)
    pka_acidic: float | None          # acidic pKa prediction (separate model)
    pka_basic: float | None           # basic pKa prediction (separate model)
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
_pka_acidic_model = None
_pka_basic_model = None
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
    """Lazy-load the RF solubility model (clean V6 preferred) + descriptor names."""
    global _solubility_model, _descriptor_names
    if _solubility_model is None:
        with _lock:
            if _solubility_model is None:
                for model_path, desc_path in RF_CANDIDATES:
                    if model_path.exists() and desc_path.exists():
                        logger.info("Loading RF solubility model from %s", model_path)
                        _solubility_model = _load_joblib(model_path)
                        _descriptor_names = joblib.load(desc_path)
                        break
                if _solubility_model is None:
                    raise FileNotFoundError("No solubility model found in output_v2/")
    return _solubility_model, _descriptor_names


def _get_pka_models():
    """Lazy-load the separate acidic/basic pKa models.

    Returns (acidic_model, basic_model); missing models are None. Falls back
    to the legacy mixed pKa model only when both new files are absent.
    """
    global _pka_acidic_model, _pka_basic_model, _pka_loaded
    if not _pka_loaded:
        with _lock:
            if not _pka_loaded:
                try:
                    if _PKA_ACIDIC_MODEL_PATH.exists():
                        logger.info("Loading acidic pKa model from %s", _PKA_ACIDIC_MODEL_PATH)
                        _pka_acidic_model = joblib.load(_PKA_ACIDIC_MODEL_PATH)
                    else:
                        logger.warning("Acidic pKa model not found at %s", _PKA_ACIDIC_MODEL_PATH)
                except Exception:
                    logger.exception("Failed to load acidic pKa model")
                    _pka_acidic_model = None
                try:
                    if _PKA_BASIC_MODEL_PATH.exists():
                        logger.info("Loading basic pKa model from %s", _PKA_BASIC_MODEL_PATH)
                        _pka_basic_model = joblib.load(_PKA_BASIC_MODEL_PATH)
                    else:
                        logger.warning("Basic pKa model not found at %s", _PKA_BASIC_MODEL_PATH)
                except Exception:
                    logger.exception("Failed to load basic pKa model")
                    _pka_basic_model = None
                if _pka_acidic_model is None and _pka_basic_model is None:
                    try:
                        if _PKA_LEGACY_MODEL_PATH.exists():
                            logger.info(
                                "New pKa models missing; loading legacy mixed model %s",
                                _PKA_LEGACY_MODEL_PATH,
                            )
                            _pka_acidic_model = _pka_basic_model = joblib.load(_PKA_LEGACY_MODEL_PATH)
                    except Exception:
                        logger.exception("Failed to load legacy pKa model")
                _pka_loaded = True
    return _pka_acidic_model, _pka_basic_model


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
    """Quick file-existence check for GNN weights (see core.artifacts)."""
    return gnn_available()


def _get_gnn():
    """Lazy-load the GNN model + graph encoder. Returns (model, encoder) or (None, None)."""
    global _gnn_model, _gnn_encoder, _gnn_loaded
    if not _gnn_loaded:
        with _lock:
            if not _gnn_loaded:
                try:
                    import torch

                    from gnn_model import ATOM_FEATURE_DIM, MoleculeGraphEncoder, SolubilityGNN

                    model_path, hidden_dim = existing_gnn()
                    if model_path is None:
                        logger.warning("No GNN model file found in %s", OUTPUT_DIR)
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


def get_shap_explainer(rf_model):
    """Lazy-create the SHAP TreeExplainer for the RF model (process-wide singleton).

    Public because model.get_shap_explainer and ui/results.py both reuse this
    instance instead of building a second explainer.
    """
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


# ── Ensemble / Auto strategy: the single implementation ──
# The simple 0.5/0.5 average beat both single models and the earlier 0.45/0.55
# weighting on the clean seed=2026 hold-out (output_v2/evaluation_report.json).
RF_WEIGHT = 0.5


def ensemble_predict(rf_pred, gnn_pred):
    """Weighted RF/GNN average using the shared default weight."""
    return RF_WEIGHT * rf_pred + (1.0 - RF_WEIGHT) * gnn_pred


def auto_predict(ood_risk, rf_pred, gnn_pred):
    """Auto strategy: weighted ensemble whenever the GNN is available.

    OOD risk and model disagreement are surfaced as warnings by the caller and
    are deliberately NOT used for routing - routing on them did not improve
    accuracy on the clean hold-out.

    Returns (prediction, model_label, disagreement); disagreement is 0.0 when
    no GNN prediction was produced.
    """
    if gnn_pred is None:
        return rf_pred, "RF", 0.0
    return ensemble_predict(rf_pred, gnn_pred), "Ensemble(W)", abs(rf_pred - gnn_pred)


def _resolve_pka_pair(pka_acidic, pka_basic, smiles=None):
    """Delegate to the single shared implementation (see pka_resolve).

    Kept as a thin wrapper so the name stays available to existing importers.
    Returns (primary_pka, kind); kind is acid/base/amphoteric, or None when no
    model produced a value.
    """
    return resolve_pka_pair(pka_acidic, pka_basic, smiles=smiles)


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


def compute_shap(rf_model, features, fp_array):
    """SHAP contributions with RAW English names: 13 descriptor keys + "MorganFP".

    The aggregation (descriptor values first, fingerprint bits summed into one
    entry) lives here; model.get_shap_contributions delegates to it and only
    translates the names for display.
    Returns (values, names, base_value) or (None, None, None) on failure.
    """
    try:
        X = np.hstack([list(features.values()), fp_array]).reshape(1, -1)
        explainer = get_shap_explainer(rf_model)
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

    # ── pKa prediction (separate acidic/basic models; lazy, failure -> None) ──
    pka_acidic = None
    pka_basic = None
    pka_val = None
    pka_kind = None
    try:
        acidic_model, basic_model = _get_pka_models()
        if acidic_model is not None or basic_model is not None:
            pka_feat = np.hstack([
                [features[k] for k in PKA_FEATURE_KEYS],
                fp_array,
            ]).reshape(1, -1)
            if acidic_model is not None:
                pka_acidic = float(acidic_model.predict(pka_feat)[0])
            if basic_model is not None:
                pka_basic = float(basic_model.predict(pka_feat)[0])
            pka_val, pka_kind = _resolve_pka_pair(pka_acidic, pka_basic, smiles=smiles)
    except Exception:
        logger.exception("pKa prediction failed")
        pka_acidic = pka_basic = pka_val = pka_kind = None

    # ── Model resolution + OOD (mirrors app.py single-prediction flow) ──
    gnn_pred = None
    if mode == "auto":
        ood_risk, ood_result = _run_ood_check(features, fp_array)
        # Auto always wants GNN when available (weighted ensemble and pure GNN both use it)
        if gnn_files_exist():
            gnn_pred = _gnn_predict(smiles)
        prediction, model_used, _ = auto_predict(ood_risk, rf_pred, gnn_pred)
    else:
        if mode in ("gnn", "ensemble") and gnn_files_exist():
            gnn_pred = _gnn_predict(smiles)
        if mode == "gnn":
            prediction = gnn_pred if gnn_pred is not None else rf_pred
        elif mode == "ensemble":
            prediction = ensemble_predict(rf_pred, gnn_pred) if gnn_pred is not None else rf_pred
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
        shap_values, shap_names, shap_base = compute_shap(rf_model, features, fp_array)

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
        pka_kind=pka_kind,
        pka_acidic=pka_acidic,
        pka_basic=pka_basic,
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
