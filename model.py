"""
DisSolve - Model loading and inference utilities.
Uses Streamlit caching for efficient model serving.
"""

import streamlit as st
import joblib
import shap
from ood_detector import OODDetector, load_ood_detector as _load_ood_from_disk


@st.cache_resource
def load_solubility_model():
    """Load the Random Forest solubility prediction model (V2)."""
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

    desc_shap = shap_values[:8]
    fp_shap_sum = shap_values[8:].sum()
    combined_shap = list(desc_shap) + [fp_shap_sum]
    combined_names = [
        "分子量 (MolWt)", "脂水分配系数 (LogP)", "氢键供体 (H-Donors)",
        "氢键受体 (H-Acceptors)", "极性表面积 (TPSA)", "可旋转键 (Rotatable Bonds)",
        "芳香环 (Aromatic Rings)", "脂肪环 (Aliphatic Rings)", "摩根指纹 (Morgan FP)"
    ]
    return combined_shap, combined_names


def get_pka_type(pka_val):
    """Classify pKa value into acid/base/amphoteric."""
    if pka_val < 5:
        return "acid", "酸性分子 (Acidic)", "pka-acid", "#a78bfa", \
               "pKa 较低，在酸性环境中以分子态为主，脂溶性高"
    elif pka_val > 9:
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
