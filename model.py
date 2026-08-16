"""
DisSolve - Model loading and inference utilities.
Uses Streamlit caching for efficient model serving.
"""

import streamlit as st
import joblib
import shap
import numpy as np
import os
import gzip
from ood_detector import OODDetector, load_ood_detector as _load_ood_from_disk
from core.i18n import t
from features import make_pka_feature_vector


def _load_joblib(path):
    """Load a joblib file, transparently handling .gz compression."""
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return joblib.load(f)
    return joblib.load(path)


@st.cache_resource
def load_solubility_model():
    """Load the Random Forest solubility prediction model (V5+)."""
    v5_path = "output_v2/solubility_model_v5.pkl.gz"
    if os.path.exists(v5_path):
        model = _load_joblib(v5_path)
        desc_names = joblib.load("output_v2/descriptor_names_v5.pkl")
        return model, desc_names
    raise FileNotFoundError(
        "No solubility model found (expected output_v2/solubility_model_v5.pkl.gz)"
    )


@st.cache_resource
def load_pka_models():
    """Load the separate acidic and basic pKa models.

    Returns (acidic_model, basic_model). Each element may be None when the
    corresponding file is missing. Falls back to the legacy single mixed-pKa
    model only when both new models are unavailable (kept for old checkouts).
    """
    acidic_path = "output_v2/pka_acidic_model.pkl"
    basic_path = "output_v2/pka_basic_model.pkl"
    legacy_path = "output_v2/pka_model.pkl"

    acidic = joblib.load(acidic_path) if os.path.exists(acidic_path) else None
    basic = joblib.load(basic_path) if os.path.exists(basic_path) else None
    if acidic is None and basic is None and os.path.exists(legacy_path):
        legacy = joblib.load(legacy_path)
        return legacy, legacy
    return acidic, basic


@st.cache_resource
def load_pka_model():
    """Deprecated compatibility loader for the legacy mixed pKa model.

    New code should use load_pka_models() and predict_pka_pair().
    """
    legacy_path = "output_v2/pka_model.pkl"
    if not os.path.exists(legacy_path):
        return None
    return joblib.load(legacy_path)


@st.cache_resource
def get_shap_explainer(_model):
    """Create a SHAP TreeExplainer for the given model."""
    return shap.TreeExplainer(_model)


def warmup_shap():
    """Pre-warm the SHAP TreeExplainer at startup so first prediction is fast.

    Call this once during app initialization after the solubility model is loaded.
    The explainer is cached via @st.cache_resource, so subsequent calls are instant.
    """
    model, _ = load_solubility_model()
    get_shap_explainer(model)


def get_shap_contributions(model, features_dict, fp_array):
    """Compute SHAP values and return combined descriptor + fingerprint contributions."""
    import numpy as np
    X = np.hstack([list(features_dict.values()), fp_array]).reshape(1, -1)
    explainer = get_shap_explainer(model)
    shap_values = explainer.shap_values(X)[0]

    n_desc = len(features_dict)  # auto-detect: 8 (legacy) or 13 (V5)
    desc_shap = shap_values[:n_desc]
    fp_shap_sum = shap_values[n_desc:].sum()
    combined_shap = list(desc_shap) + [fp_shap_sum]
    combined_names = list(features_dict.keys()) + ["摩根指纹 (Morgan FP)"]
    # Translate to Chinese for display
    from ood_detector import DESCRIPTOR_NAMES_CN
    combined_names = [DESCRIPTOR_NAMES_CN.get(n, n) for n in combined_names[:-1]] + [t("model.shap.morgan_fp")]
    return combined_shap, combined_names


def get_pka_type(pka_val, kind=None):
    """Classify a pKa value into acid/base/amphoteric.

    ``kind`` may be provided explicitly when the caller already resolved the
    acid/base state from the pair of acidic/basic pKa predictions
    (see resolve_pka_pair). Without it, the legacy single-value thresholds
    (<6 acid, >8 base) are preserved for backwards compatibility.
    """
    if kind is None:
        kind = "acid" if pka_val < 6 else ("base" if pka_val > 8 else "amphoteric")

    if kind == "acid":
        return "acid", t("model.pka.type.acidic_display"), "pka-acid", "#a78bfa", \
               t("model.pka.type.acidic_desc")
    if kind == "base":
        return "base", t("model.pka.type.basic_display"), "pka-base", "#22d3ee", \
               t("model.pka.type.basic_desc")
    return "amphoteric", t("model.pka.type.amphoteric_display"), "pka-amphoteric", "#fbbf24", \
           t("model.pka.type.amphoteric_desc")


def resolve_pka_pair(pka_acidic, pka_basic):
    """Resolve separate acidic/basic pKa predictions into one primary value + kind.

    A molecule is classified as amphoteric when it has a meaningful acidic pKa
    (<7) AND a meaningful basic pKa (>7), e.g. amino acids. Otherwise the
    relevant acid or base prediction is selected. The primary value is the one
    closest to physiological pH 7 when both are available.

    Returns (primary_pka, kind). ``kind`` is "acid" | "base" | "amphoteric" | None.
    """
    if pka_acidic is None and pka_basic is None:
        return None, None

    acidic_relevant = pka_acidic is not None and pka_acidic < 7.0
    basic_relevant = pka_basic is not None and pka_basic > 7.0

    if acidic_relevant and basic_relevant:
        kind = "amphoteric"
    elif acidic_relevant:
        kind = "acid"
    elif basic_relevant:
        kind = "base"
    elif pka_acidic is not None and pka_basic is not None:
        # Both predictions sit in the weak/neutral zone.
        kind = "amphoteric"
    elif pka_acidic is not None:
        kind = "acid" if pka_acidic < 7 else "base"
    else:
        kind = "base" if pka_basic > 7 else "acid"

    values = [v for v in (pka_acidic, pka_basic) if v is not None]
    if kind == "acid" and pka_acidic is not None:
        primary = pka_acidic
    elif kind == "base" and pka_basic is not None:
        primary = pka_basic
    else:
        primary = min(values, key=lambda v: abs(v - 7.0))
    return float(primary), kind


def predict_pka_pair(pka_models, features_dict, fp_array):
    """Predict (pka_acidic, pka_basic) with the loaded pair of models.

    Missing models yield None for that quantity.
    """
    acidic_model, basic_model = pka_models if pka_models is not None else (None, None)
    X = make_pka_feature_vector(features_dict, fp_array)
    pka_acidic = float(acidic_model.predict(X)[0]) if acidic_model is not None else None
    pka_basic = float(basic_model.predict(X)[0]) if basic_model is not None else None
    return pka_acidic, pka_basic


def get_solubility_level(prediction):
    """Classify logS prediction into solubility level."""
    if prediction > 0:
        return t("model.solubility.high"), "#34d399", "result-high"
    elif prediction > -2:
        return t("model.solubility.moderate"), "#fbbf24", "result-moderate"
    else:
        return t("model.solubility.poor"), "#f87171", "result-low"


@st.cache_resource
def load_ood_detector():
    """Load the OOD detector (training-data statistics + fingerprint references)."""
    try:
        return _load_ood_from_disk("output_v2/ood_detector.pkl.gz")
    except FileNotFoundError:
        return None


def run_ood_check(features_dict, fp_array):
    """Run OOD detection and return (risk_level, result_or_None)."""
    detector = load_ood_detector()
    if detector is None:
        return "UNKNOWN", None
    result = detector.check(features_dict, fp_array)
    return result.risk_level, result


# ── GNN model loading & inference ──

@st.cache_resource
def load_gnn_model():
    """Load the trained GNN solubility model and encoder."""
    from gnn_model import SolubilityGNN, MoleculeGraphEncoder, ATOM_FEATURE_DIM
    import torch

    import os
    # Try V4 first, then V3, then V2
    for model_file, hidden_dim in [
        ("gnn_solubility_model_v4.pt", 256),
        ("gnn_solubility_model_v3.pt", 128),
        ("gnn_solubility_model.pt", 128),
    ]:
        model_path = os.path.join("output_v2", model_file)
        if os.path.exists(model_path):
            break
    if not os.path.exists(model_path):
        return None, None

    encoder = MoleculeGraphEncoder()
    model = SolubilityGNN(atom_dim=ATOM_FEATURE_DIM, hidden_dim=hidden_dim, num_layers=3)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model, encoder


def predict_solubility_gnn(model, encoder, smiles):
    """Run GNN prediction for a single SMILES."""
    from rdkit import Chem
    import torch

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    graph = encoder.mol_to_graph(mol)
    if graph is None:
        return None

    with torch.no_grad():
        pred = model(graph)
    return float(pred.item())


def predict_solubility_ensemble(rf_pred, gnn_pred):
    """Return (ensemble_pred, rf_pred, gnn_pred). Weighted average (0.45RF+0.55GNN)."""
    ensemble = 0.45 * rf_pred + 0.55 * gnn_pred
    return ensemble, rf_pred, gnn_pred


def predict_solubility_weighted(rf_pred, gnn_pred, rf_weight=0.45):
    """Weighted ensemble: rf_weight × RF + (1-rf_weight) × GNN.
    Optimal weight 0.45:RF + 0.55:GNN (found via grid search on V5)."""
    return rf_pred * rf_weight + gnn_pred * (1.0 - rf_weight)


def predict_solubility_auto(ood_risk, rf_pred, gnn_pred):
    """Auto+ strategy: select model based on OOD risk + model disagreement.

    Args:
        ood_risk: "LOW", "MEDIUM", or "HIGH" from OOD detector.
        rf_pred: Random Forest prediction value.
        gnn_pred: GNN prediction value (may be None).

    Returns (prediction_value, actual_model_label, disagreement):
      disagreement = abs(rf_pred - gnn_pred) or 0 if gnn is None.

    Strategy:
      - RF/GNN disagree > 1.0 → pure GNN (models can't agree, GNN is safer)
      - OOD LOW + agree ≤ 1.0 → 0.45×RF + 0.55×GNN (weighted ensemble)
      - OOD MEDIUM/HIGH       → pure GNN (RF unreliable on outliers)
    """
    if gnn_pred is None:
        return rf_pred, "RF", 0.0

    disagreement = abs(rf_pred - gnn_pred)

    # Severe disagreement: models fundamentally disagree → trust GNN
    if disagreement > 1.0:
        return gnn_pred, "GNN", disagreement

    # OOD outlier → GNN is more reliable
    if ood_risk != "LOW":
        return gnn_pred, "GNN", disagreement

    # Normal case: weighted ensemble
    return predict_solubility_weighted(rf_pred, gnn_pred), "Ensemble(W)", disagreement


# ── GNN Explainability ──

@st.cache_resource(ttl=600)
def get_gnn_explainer(_model, lr=0.01, epochs=300):
    """Create a cached GNNExplainer for the given model."""
    from gnn_explainer import GNNExplainer
    return GNNExplainer(_model, lr=lr, epochs=epochs)


def explain_gnn_prediction(model, encoder, smiles):
    """Run GNNExplainer on a single SMILES and return bond + feature importance.

    Args:
        model: Loaded SolubilityGNN instance.
        encoder: MoleculeGraphEncoder instance.
        smiles: SMILES string.

    Returns:
        dict from GNNExplainer.explain(), or None if parsing/encoding fails.
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    graph = encoder.mol_to_graph(mol)
    if graph is None:
        return None

    x = graph["x"]
    edge_index = graph["edge_index"]
    if edge_index.size(1) == 0:
        return None  # single-atom molecule, no bonds to explain

    explainer = get_gnn_explainer(model, lr=0.01, epochs=300)
    result = explainer.explain(x, edge_index)
    result["mol"] = mol  # attach RDKit Mol for plotting
    return result
