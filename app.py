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

from core.state_keys import StateKey
from model import (
    load_solubility_model, load_pka_model,
    load_ood_detector, run_ood_check,
    warmup_shap, get_solubility_level,
)
from core.cache import (
    cached_compute_features, cached_show_3d, cached_pka_analysis,
    cached_shap_contributions, cached_lipinski, cached_admet, cached_druglikeness,
)
from assets.theme import inject_theme_css
from assets.scripts import inject_all_scripts
from ui.components import render_header, render_footer, render_input_area, render_file_upload_input, render_prediction_history
from ui.results import render_results


# ========== Inject CSS and JS ==========
inject_theme_css()
inject_all_scripts()


# ========== Load models ==========
# Models are cached by @st.cache_resource — first load reads from disk,
# subsequent runs return instantly from cache. SHAP explainer is pre-warmed
# so the first prediction is fast.
model_ready = False
pka_ready = False
ood_ready = False
try:
    model, descriptor_names = load_solubility_model()
    model_ready = True
except Exception as e:
    st.error(f"模型加载失败: {e}")
    st.info("请先运行 'python train_model_v2.py' 训练模型")

try:
    pka_model = load_pka_model()
    pka_ready = True
except Exception:
    pass

try:
    ood_detector = load_ood_detector()
    ood_ready = ood_detector is not None
except Exception:
    pass

# Pre-warm SHAP TreeExplainer so first prediction doesn't lag
if model_ready:
    try:
        warmup_shap()
    except Exception:
        pass


# ========== Session state init ==========
if StateKey.SMILES_INPUT not in st.session_state:
    st.session_state[StateKey.SMILES_INPUT] = ""
if StateKey.PREDICTED_SMILES not in st.session_state:
    st.session_state[StateKey.PREDICTED_SMILES] = None
if StateKey.PREDICTED_LOGS not in st.session_state:
    st.session_state[StateKey.PREDICTED_LOGS] = None
if StateKey.AI_EXPLANATION not in st.session_state:
    st.session_state[StateKey.AI_EXPLANATION] = None

# Apply pending history selection (must happen before widget renders)
if "_pending_history_smiles" in st.session_state:
    st.session_state[StateKey.SMILES_INPUT] = st.session_state.pop("_pending_history_smiles")
    st.session_state[StateKey.CURRENT_MOLECULE_NAME] = st.session_state.pop("_pending_history_name", "")
    st.session_state[StateKey.PREDICTED_SMILES] = None
    st.session_state[StateKey.PREDICTED_LOGS] = None
    st.session_state[StateKey.AI_EXPLANATION] = None


# ========== Render header ==========
render_header()


# ========== Render input area (3 methods) ==========
render_input_area()


# ========== Render file upload (4th input method) ==========
render_file_upload_input()


# ========== Prediction history ==========
render_prediction_history()


# ========== Predict button ==========
st.markdown("<br>", unsafe_allow_html=True)
btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
with btn_col2:
    predict_button = st.button("&#128302; Predict Solubility", use_container_width=True)
st.markdown("<br>", unsafe_allow_html=True)


# ========== Execute prediction ==========
if predict_button and model_ready:
    current = st.session_state[StateKey.SMILES_INPUT].strip()

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
                st.session_state[StateKey.CACHED_FEATURES] = features
                X_input = np.hstack([list(features.values()), fp_array]).reshape(1, -1)

                status.update(label="Step 2/4: Random Forest 预测溶解度...")
                prediction = model.predict(X_input)[0]
                st.session_state[StateKey.PREDICTED_SMILES] = current
                st.session_state[StateKey.PREDICTED_LOGS] = float(prediction)

                if pka_ready:
                    status.update(label="Step 3/4: 预测 pKa 与电离行为...")
                    pka_pred = pka_model.predict(X_input)[0]
                    st.session_state[StateKey.PREDICTED_PKA] = float(pka_pred)

                status.update(label="Step 4/4: SHAP 可解释性分析...")
                try:
                    combined_shap, combined_names = cached_shap_contributions(current)
                    st.session_state[StateKey.SHAP_VALUES] = combined_shap
                    st.session_state[StateKey.SHAP_NAMES] = combined_names
                except Exception:
                    st.session_state[StateKey.SHAP_VALUES] = None
                    st.session_state[StateKey.SHAP_NAMES] = None
                st.session_state[StateKey.AI_EXPLANATION] = None

                if ood_ready:
                    status.update(label="Step 4+/4: OOD 分布检测...")
                    ood_risk, ood_result = run_ood_check(features, fp_array)
                    st.session_state[StateKey.OOD_RISK] = ood_risk
                    st.session_state[StateKey.OOD_RESULT] = ood_result
                else:
                    st.session_state[StateKey.OOD_RISK] = "UNKNOWN"
                    st.session_state[StateKey.OOD_RESULT] = None

                status.update(label=f"分析完成！预测 logS = {float(prediction):.3f}", state="complete")

                # ── Save to prediction history ──
                try:
                    from molecules import MOLECULE_DB
                    mol_name = st.session_state.get(StateKey.CURRENT_MOLECULE_NAME, "")
                    if not mol_name:
                        # Fall back to radio selection if it matches current SMILES
                        radio_key = st.session_state.get("molecule_select_radio")
                        if radio_key and radio_key in MOLECULE_DB and MOLECULE_DB[radio_key] == current:
                            mol_name = radio_key
                    if not mol_name:
                        # Last resort: reverse SMILES lookup
                        mol_name = next(
                            (k for k, v in MOLECULE_DB.items() if v == current and k != "(自定义输入)"),
                            "",
                        )
                except Exception:
                    mol_name = ""
                import datetime, copy
                history_entry = {
                    "smiles": current,
                    "name": mol_name,
                    "logS": float(prediction),
                    "pKa": st.session_state.get(StateKey.PREDICTED_PKA),
                    "timestamp": datetime.datetime.now().strftime("%H:%M"),
                }
                if StateKey.PREDICTION_HISTORY not in st.session_state:
                    st.session_state[StateKey.PREDICTION_HISTORY] = []
                history = st.session_state[StateKey.PREDICTION_HISTORY]
                # Avoid duplicate consecutive entries
                if not history or history[0].get("smiles") != current:
                    history.insert(0, copy.deepcopy(history_entry))
                    if len(history) > 15:
                        history.pop()
                    # Explicitly reassign so Streamlit detects the in-place mutation
                    st.session_state[StateKey.PREDICTION_HISTORY] = history
                    st.rerun()


# ========== Display results (5 tabs) ==========
if st.session_state.get(StateKey.PREDICTED_SMILES) and st.session_state.get(StateKey.PREDICTED_LOGS) is not None:
    render_results(model)


# ========== Batch CSV prediction ==========
st.markdown("---")
with st.expander("&#128230; 批量预测（上传 CSV）", expanded=False):
    st.caption("上传包含 SMILES 列的 CSV 文件，批量预测所有分子的溶解度与 pKa")

    batch_file = st.file_uploader(
        "选择 CSV 文件",
        type=["csv"],
        key="batch_csv_uploader",
        label_visibility="collapsed",
    )

    if batch_file is not None:
        try:
            import pandas as pd

            raw = batch_file.getvalue().decode("utf-8-sig")
            # Peek at header row to find SMILES column
            header_line = raw.split("\n", 1)[0]
            header = [c.strip().strip('"').strip("'") for c in header_line.split(",")]
            header_lower = [h.lower() for h in header]

            # Auto-detect SMILES column
            smiles_col = None
            for keyword in ("smiles", "smile", "smi", "canonical_smiles", "isomeric_smiles"):
                for i, h in enumerate(header_lower):
                    if keyword in h:
                        smiles_col = i
                        break
                if smiles_col is not None:
                    break

            if smiles_col is None:
                # Fallback: first column with "smiles-like" content
                for i, h in enumerate(header_lower):
                    if "mol" in h or "structure" in h or "compound" in h:
                        smiles_col = i
                        break

            if smiles_col is None:
                smiles_col = 0  # default to first column

            df = pd.read_csv(io.StringIO(raw))
            st.info(
                f"文件: `{batch_file.name}` | "
                f"{len(df)} 行 | "
                f"检测到 SMILES 列: **{header[smiles_col]}**"
            )
            st.dataframe(df.head(5), use_container_width=True)

            if st.button("&#128302; 开始批量预测", key="batch_predict_btn", use_container_width=True):
                # Validate SMILES column content
                if header[smiles_col] not in df.columns:
                    st.error(f"列 '{header[smiles_col]}' 不存在，请检查文件格式")
                else:
                    results = []
                    smiles_list = df[header[smiles_col]].dropna().astype(str).tolist()
                    progress_bar = st.progress(0, text=f"正在预测 0/{len(smiles_list)}...")

                    for idx, smi in enumerate(smiles_list):
                        progress_bar.progress(
                            (idx + 1) / len(smiles_list),
                            text=f"正在预测 {idx+1}/{len(smiles_list)}...",
                        )
                        feat_result = cached_compute_features(smi)
                        if feat_result is None:
                            results.append({
                                "SMILES": smi,
                                "logS": None,
                                "Solubility Level": "Invalid SMILES",
                                "pKa": None,
                                "MolWt": None,
                                "LogP": None,
                            })
                            continue
                        features, fp = feat_result
                        X = np.hstack([list(features.values()), fp]).reshape(1, -1)
                        try:
                            logS = float(model.predict(X)[0])
                        except Exception:
                            logS = None
                        try:
                            pKa_val = float(pka_model.predict(X)[0]) if pka_ready else None
                        except Exception:
                            pka_val = None

                        level = get_solubility_level(logS)[0] if logS is not None else "?"
                        results.append({
                            "SMILES": smi,
                            "logS": f"{logS:.3f}" if logS is not None else "?",
                            "Solubility Level": level,
                            "pKa": f"{pKa_val:.2f}" if pKa_val is not None else "?",
                            "MolWt": f"{features['MolWt']:.1f}",
                            "LogP": f"{features['LogP']:.2f}",
                        })

                    progress_bar.empty()
                    result_df = pd.DataFrame(results)
                    st.success(f"批量预测完成！共 {len(results)} 个分子")
                    st.dataframe(result_df, use_container_width=True, height=400)

                    # Download button
                    csv_out = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "&#128229; 下载结果 CSV",
                        data=csv_out,
                        file_name="dissolve_batch_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
        except Exception as batch_err:
            st.error(f"批量处理出错: {batch_err}")
            import traceback
            st.code(traceback.format_exc())

# ========== Render footer ==========
render_footer()
