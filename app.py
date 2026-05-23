"""
DisSolve - AI-Powered Molecular Property Prediction.
Thin orchestrator: imports modules, manages session state, and coordinates rendering.
"""

import streamlit as st

# CRITICAL: st.set_page_config() MUST be the first Streamlit command.
# Importing core.cache or other modules with @st.cache_data triggers
# Streamlit commands at import time, violating the ordering requirement
# and causing CSS/layout/rendering issues.
st.set_page_config(
    page_title="DisSolve - Molecular Property Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from model import (
    load_solubility_model, load_pka_model,
    load_ood_detector, run_ood_check,
)
from core.cache import (
    cached_compute_features, cached_show_3d, cached_pka_analysis,
    cached_shap_contributions, cached_lipinski, cached_admet, cached_druglikeness,
)
from assets.theme import inject_theme_css
from assets.scripts import inject_all_scripts
from ui.components import render_header, render_footer, render_input_area
from ui.results import render_results


# ========== Inject CSS and JS ==========
inject_theme_css()
inject_all_scripts()


# ========== Load models ==========
try:
    model, descriptor_names = load_solubility_model()
    model_ready = True
except Exception as e:
    st.error(f"模型加载失败: {e}")
    st.info("请先运行 'python train_model_v2.py' 训练模型")
    model_ready = False

try:
    pka_model = load_pka_model()
    pka_ready = True
except Exception:
    pka_ready = False

try:
    ood_detector = load_ood_detector()
    ood_ready = ood_detector is not None
except Exception:
    ood_ready = False


# ========== Session state init ==========
if "smiles_input_box" not in st.session_state:
    st.session_state.smiles_input_box = ""
if "predicted_smiles" not in st.session_state:
    st.session_state.predicted_smiles = None
if "predicted_logS" not in st.session_state:
    st.session_state.predicted_logS = None
if "ai_explanation" not in st.session_state:
    st.session_state.ai_explanation = None


# ========== Render header ==========
render_header()


# ========== Render input area (3 methods) ==========
render_input_area()


# ========== Predict button ==========
st.markdown("<br>", unsafe_allow_html=True)
btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
with btn_col2:
    predict_button = st.button("&#128302; Predict Solubility", use_container_width=True)
st.markdown("<br>", unsafe_allow_html=True)


# ========== Execute prediction ==========
if predict_button and model_ready:
    current = st.session_state.smiles_input_box.strip()

    if not current:
        st.warning("请先输入或选择一个分子的 SMILES")
    else:
        with st.status("正在分析分子结构...", expanded=False) as status:
            status.update(label="Step 1/4: 解析分子结构...")
            result = cached_compute_features(current)

            if result is None:
                status.update(label="解析失败", state="error")
                st.error(f"Invalid SMILES: `{current}`")
                st.info("该 SMILES 无法被 RDKit 解析。可能原因：")
                st.markdown("""
                - 分子含有金属/配位键，RDKit 不支持
                - SMILES 语法错误（括号不匹配）
                - 输入为空或含有非法字符
                """)
            else:
                features, fp_array = result
                st.session_state.cached_features = features
                X_input = np.hstack([list(features.values()), fp_array]).reshape(1, -1)

                status.update(label="Step 2/4: Random Forest 预测溶解度...")
                prediction = model.predict(X_input)[0]
                st.session_state.predicted_smiles = current
                st.session_state.predicted_logS = float(prediction)

                if pka_ready:
                    status.update(label="Step 3/4: 预测 pKa 与电离行为...")
                    pka_pred = pka_model.predict(X_input)[0]
                    st.session_state.predicted_pka = float(pka_pred)

                status.update(label="Step 4/4: SHAP 可解释性分析...")
                try:
                    combined_shap, combined_names = cached_shap_contributions(current)
                    st.session_state.shap_values = combined_shap
                    st.session_state.shap_names = combined_names
                except Exception:
                    st.session_state.shap_values = None
                    st.session_state.shap_names = None
                st.session_state.ai_explanation = None

                if ood_ready:
                    status.update(label="Step 4+/4: OOD 分布检测...")
                    ood_risk, ood_result = run_ood_check(features, fp_array)
                    st.session_state.ood_risk = ood_risk
                    st.session_state.ood_result = ood_result
                else:
                    st.session_state.ood_risk = "UNKNOWN"
                    st.session_state.ood_result = None

                status.update(label=f"分析完成！预测 logS = {float(prediction):.3f}", state="complete")


# ========== Display results (5 tabs) ==========
if st.session_state.predicted_smiles and st.session_state.predicted_logS is not None:
    render_results(model)


# ========== Render footer ==========
render_footer()
