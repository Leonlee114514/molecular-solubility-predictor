"""
DisSolve - Model loading and inference utilities.
Uses Streamlit caching for efficient model serving.
"""

import streamlit as st
import joblib
import shap
import numpy as np
import os
from ood_detector import OODDetector, load_ood_detector as _load_ood_from_disk


@st.cache_resource
def load_solubility_model():
    """Load the Random Forest solubility prediction model (V5+)."""
    import os
    v5_path = "output_v2/solubility_model_v5.pkl"
    if os.path.exists(v5_path):
        model = joblib.load(v5_path)
        desc_names = joblib.load("output_v2/descriptor_names_v5.pkl")
        return model, desc_names
    v4_path = "output_v2/solubility_model_v4.pkl"
    if os.path.exists(v4_path):
        model = joblib.load(v4_path)
        desc_names = joblib.load("output_v2/descriptor_names_v4.pkl")
        return model, desc_names
    v3_path = "output_v2/solubility_model_v3.pkl"
    if os.path.exists(v3_path):
        model = joblib.load(v3_path)
        desc_names = joblib.load("output_v2/descriptor_names_v2.pkl")  # v3 uses same desc names
        return model, desc_names
    model = joblib.load("output_v2/solubility_model_v2.pkl")
    desc_names = joblib.load("output_v2/descriptor_names_v2.pkl")
    return model, desc_names


@st.cache_resource
def load_pka_model():
    """Load the pKa prediction model."""
    model = joblib.load("output_v2/pka_model.pkl")
    return model


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


def predict_solubility(model, features_dict, fp_array):
    """Run solubility prediction given features and fingerprint."""
    import numpy as np
    X = np.hstack([list(features_dict.values()), fp_array]).reshape(1, -1)
    return float(model.predict(X)[0])


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
    combined_names = [DESCRIPTOR_NAMES_CN.get(n, n) for n in combined_names[:-1]] + ["摩根指纹 (Morgan FP)"]
    return combined_shap, combined_names


def get_pka_type(pka_val):
    """Classify pKa value into acid/base/amphoteric."""
    if pka_val < 6:
        return "acid", "酸性分子 (Acidic)", "pka-acid", "#a78bfa", \
               "pKa 较低，在酸性环境中以分子态为主，脂溶性高"
    elif pka_val > 8:
        return "base", "碱性分子 (Basic)", "pka-base", "#22d3ee", \
               "pKa 较高，在碱性环境中以分子态为主"
    else:
        return "amphoteric", "两性/中性 (Amphoteric/Neutral)", "pka-amphoteric", "#fbbf24", \
               "pKa 接近中性，电离行为随 pH 变化剧烈"


def get_solubility_level(prediction):
    """Classify logS prediction into solubility level."""
    if prediction > 0:
        return "Highly soluble (易溶于水)", "#34d399", "result-high"
    elif prediction > -2:
        return "Moderately soluble (中等溶解)", "#fbbf24", "result-moderate"
    else:
        return "Poorly soluble (难溶于水)", "#f87171", "result-low"


@st.cache_resource
def load_ood_detector():
    """Load the OOD detector (training-data statistics + fingerprint references)."""
    try:
        return _load_ood_from_disk("output_v2/ood_detector.pkl")
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
