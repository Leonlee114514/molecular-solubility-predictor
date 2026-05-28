"""
DisSolve - Streamlit cached wrapper functions.
Separated from app.py so UI modules can import them without circular dependencies.
"""

import streamlit as st
from features import compute_features
from core.analysis import analyze_pka_chemistry, analyze_lipinski, analyze_admet, analyze_druglikeness
from ui.plots import show_3d_molecule
from model import load_solubility_model, get_shap_contributions


@st.cache_data(show_spinner=False, ttl=3600)
def cached_compute_features(smiles_string):
    """Cached wrapper around features.compute_features."""
    return compute_features(smiles_string)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_show_3d(smiles):
    """Cached wrapper around features.show_3d_molecule."""
    return show_3d_molecule(smiles)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_pka_analysis(smiles, pka_val):
    """Cached wrapper around features.analyze_pka_chemistry."""
    return analyze_pka_chemistry(smiles, pka_val)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_shap_contributions(smiles):
    """Cached SHAP computation keyed by SMILES. Reuses cached compute_features."""
    result = cached_compute_features(smiles)
    if result is None:
        return None, None
    features, fp_array = result
    model, _ = load_solubility_model()
    return get_shap_contributions(model, features, fp_array)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_lipinski(features_tuple):
    """Cached Lipinski Rule of Five evaluation."""
    features = dict(features_tuple)
    return analyze_lipinski(features)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_admet(smiles, features_tuple, pka_val):
    """Cached ADME/Tox analysis."""
    features = dict(features_tuple)
    return analyze_admet(smiles, features, pka_val)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_druglikeness(smiles):
    """Cached QED, SAscore, Fsp³ drug-likeness metrics."""
    return analyze_druglikeness(smiles)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_gnn_predict(smiles):
    """Cached GNN prediction. Returns float logS or None."""
    from model import load_gnn_model, predict_solubility_gnn
    gnn_model, encoder = load_gnn_model()
    if gnn_model is None:
        return None
    return predict_solubility_gnn(gnn_model, encoder, smiles)
