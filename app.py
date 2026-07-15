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
    page_title=t("app.title"),
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

import numpy as np
import io
import os
import json
import concurrent.futures
from dotenv import load_dotenv

load_dotenv()

from core.state_keys import StateKey
from core.i18n import init_language, t, render_language_selector
from model import (
    load_solubility_model, load_pka_model,
    load_ood_detector, run_ood_check,
    warmup_shap, get_solubility_level,
    load_gnn_model, predict_solubility_gnn, predict_solubility_ensemble,
    predict_solubility_auto, predict_solubility_weighted,
)
from core.cache import (
    cached_compute_features, cached_show_3d, cached_pka_analysis,
    cached_shap_contributions, cached_lipinski, cached_admet, cached_druglikeness,
    cached_gnn_predict,
)
from assets.theme import inject_theme_css
from assets.scripts import inject_all_scripts
from ui.components import render_header, render_footer, render_input_area, render_file_upload_input, render_prediction_history
from ui.results import render_results


# ========== Inject CSS and JS ==========
inject_theme_css()
inject_all_scripts()

# ========== Init i18n + language selector ==========
init_language()


# ========== Load models (RF loaded eagerly; others lazy on first prediction) ==========
model_ready = False
try:
    model, descriptor_names = load_solubility_model()
    model_ready = True
except Exception as e:
    st.error(t("app.model.load_error", e=e))
    st.info(t("app.model.load_help"))

# GNN availability: quick file-existence check instead of loading the full model
_GNN_FILES = [
    os.path.join("output_v2", f)
    for f in ["gnn_solubility_model_v4.pt", "gnn_solubility_model_v3.pt", "gnn_solubility_model.pt"]
]
gnn_ready = any(os.path.exists(p) for p in _GNN_FILES)

# SHAP explainer pre-warmed in background so first prediction is fast
if model_ready and not st.session_state.get("_shap_warmed"):
    import threading
    def _warmup():
        try:
            warmup_shap()
            st.session_state["_shap_warmed"] = True
        except Exception:
            pass
    threading.Thread(target=_warmup, daemon=True).start()


# ========== Session state init ==========
if StateKey.SMILES_INPUT not in st.session_state:
    st.session_state[StateKey.SMILES_INPUT] = ""
if StateKey.PREDICTED_SMILES not in st.session_state:
    st.session_state[StateKey.PREDICTED_SMILES] = None
if StateKey.PREDICTED_LOGS not in st.session_state:
    st.session_state[StateKey.PREDICTED_LOGS] = None
if StateKey.AI_EXPLANATION not in st.session_state:
    st.session_state[StateKey.AI_EXPLANATION] = None
if StateKey.SELECTED_MODEL not in st.session_state:
    st.session_state[StateKey.SELECTED_MODEL] = "Auto"
if StateKey.ACTUAL_MODEL not in st.session_state:
    st.session_state[StateKey.ACTUAL_MODEL] = None
if StateKey.MODEL_DISAGREEMENT not in st.session_state:
    st.session_state[StateKey.MODEL_DISAGREEMENT] = 0.0
if StateKey.PREDICTED_LOGS_RF not in st.session_state:
    st.session_state[StateKey.PREDICTED_LOGS_RF] = None
if StateKey.PREDICTED_LOGS_GNN not in st.session_state:
    st.session_state[StateKey.PREDICTED_LOGS_GNN] = None
if StateKey.PREDICTION_HISTORY not in st.session_state:
    hist_path = os.path.join(os.path.dirname(__file__), '.prediction_history.json')
    loaded = []
    try:
        if os.path.exists(hist_path):
            with open(hist_path, 'r', encoding='utf-8') as hf:
                loaded = json.load(hf)
    except Exception:
        pass
    st.session_state[StateKey.PREDICTION_HISTORY] = loaded
if "__history_dirty" not in st.session_state:
    st.session_state["__history_dirty"] = False

# Apply pending history selection (must happen before widget renders)
if "_pending_history_smiles" in st.session_state:
    st.session_state[StateKey.SMILES_INPUT] = st.session_state.pop("_pending_history_smiles")
    st.session_state[StateKey.CURRENT_MOLECULE_NAME] = st.session_state.pop("_pending_history_name", "")
    st.session_state[StateKey.PREDICTED_SMILES] = None
    st.session_state[StateKey.PREDICTED_LOGS] = None
    st.session_state[StateKey.PREDICTED_PKA] = None
    st.session_state[StateKey.AI_EXPLANATION] = None


# ========== Render header ==========
render_header()

# Language selector
render_language_selector()

# ========== Render input area (3 methods) ==========
render_input_area()


# ========== Render file upload (4th input method) ==========
render_file_upload_input()


# ========== Prediction history ==========
render_prediction_history()


# ========== Model selector ==========
st.markdown("<br>", unsafe_allow_html=True)
sel_col1, sel_col2, sel_col3 = st.columns([1, 2, 1])
with sel_col2:
    model_options = [t("app.model.auto"), t("app.model.rf"), t("app.model.gnn"), t("app.model.ensemble")]
    if not gnn_ready:
        model_options = [t("app.model.rf")]
        st.caption(t("app.model.gnn_missing"))
    model_choice = st.selectbox(
        t("app.model.label"),
        model_options,
        key="model_select_widget",
        index=0 if not gnn_ready else
              (0 if st.session_state[StateKey.SELECTED_MODEL] == "Auto" else
               1 if st.session_state[StateKey.SELECTED_MODEL] == "RF" else
               2 if st.session_state[StateKey.SELECTED_MODEL] == "GNN" else 3),
    )
    choice = st.session_state[StateKey.SELECTED_MODEL]
    if model_choice == t("app.model.auto"):
        st.session_state[StateKey.SELECTED_MODEL] = "Auto"
    elif model_choice == t("app.model.rf"):
        st.session_state[StateKey.SELECTED_MODEL] = "RF"
    elif model_choice == t("app.model.gnn"):
        st.session_state[StateKey.SELECTED_MODEL] = "GNN"
    else:
        st.session_state[StateKey.SELECTED_MODEL] = "Ensemble"
st.markdown("<br>", unsafe_allow_html=True)

# ========== Predict button ==========
btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
with btn_col2:
    predict_button = st.button(t("app.predict_btn"), use_container_width=True)
st.markdown("<br>", unsafe_allow_html=True)


# ========== Execute prediction ==========
if predict_button and model_ready:
    current = st.session_state[StateKey.SMILES_INPUT].strip()

    if not current:
        st.warning(t("app.predict.empty"))
    else:
        with st.status(t("app.predict.status"), expanded=False) as status:
            status.update(label=t("app.predict.step.parse"))
            result = cached_compute_features(current)

            if result is None:
                status.update(label=t("app.predict.step.parse_fail"), state="error")
                st.error(t("app.error.invalid_smiles", smiles=current))
                st.info(t("app.error.parse_fail_info"))
                st.markdown(f"""
- {t('app.error.parse_reason1')}
- {t('app.error.parse_reason2')}
- {t('app.error.parse_reason3')}
                """)
            else:
                features, fp_array = result
                st.session_state[StateKey.CACHED_FEATURES] = features
                X_input = np.hstack([list(features.values()), fp_array]).reshape(1, -1)

                model_type = st.session_state[StateKey.SELECTED_MODEL]
                rf_pred = None
                gnn_pred = None

                # ── RF prediction (always needed) ──
                status.update(label=t("app.predict.step.rf"))
                rf_pred = float(model.predict(X_input)[0])
                st.session_state[StateKey.PREDICTED_SMILES] = current
                st.session_state[StateKey.PREDICTED_LOGS_RF] = rf_pred

                # ── pKa prediction (lazy-loaded on first use) ──
                try:
                    _pka_model = load_pka_model()
                    if _pka_model is not None:
                        status.update(label=t("app.predict.step.pka"))
                        # pKa model was trained on 8 core descriptors + 1024-bit Morgan FP only
                        pka_feat = np.hstack([
                            [features[k] for k in ["MolWt", "LogP", "NumHDonors", "NumHAcceptors",
                                                    "TPSA", "NumRotatableBonds", "NumAromaticRings",
                                                    "NumAliphaticRings"]],
                            fp_array,
                        ]).reshape(1, -1)
                        pka_pred = _pka_model.predict(pka_feat)[0]
                        st.session_state[StateKey.PREDICTED_PKA] = float(pka_pred)
                except Exception:
                    st.session_state[StateKey.PREDICTED_PKA] = None

                # ── Auto+: OOD 动态模型选择 ──
                actual_model = model_type
                if model_type == "Auto":
                    status.update(label=t("app.predict.step.ood"))
                    ood_risk, ood_result = run_ood_check(features, fp_array)
                    st.session_state[StateKey.OOD_RISK] = ood_risk
                    st.session_state[StateKey.OOD_RESULT] = ood_result

                    # Always need GNN for Auto+ (both weighted ensemble and pure GNN use it)
                    if gnn_ready:
                        status.update(label=t("app.predict.step.gnn"))
                        gnn_pred = cached_gnn_predict(current)
                        st.session_state[StateKey.PREDICTED_LOGS_GNN] = gnn_pred

                    prediction, actual_model, disagreement = predict_solubility_auto(ood_risk, rf_pred, gnn_pred)
                    st.session_state[StateKey.PREDICTED_LOGS] = float(prediction)
                    st.session_state[StateKey.ACTUAL_MODEL] = actual_model
                    st.session_state[StateKey.MODEL_DISAGREEMENT] = disagreement
                    ood_already_done = True
                else:
                    ood_already_done = False
                    # ── GNN prediction (for GNN and Ensemble modes) ──
                    if model_type in ("GNN", "Ensemble") and gnn_ready:
                        status.update(label=t("app.predict.step.gnn"))
                        gnn_pred = cached_gnn_predict(current)
                        st.session_state[StateKey.PREDICTED_LOGS_GNN] = gnn_pred

                    # ── Final prediction ──
                    if model_type == "GNN":
                        prediction = gnn_pred if gnn_pred is not None else rf_pred
                    elif model_type == "Ensemble":
                        if gnn_pred is not None:
                            ensemble, _, _ = predict_solubility_ensemble(rf_pred, gnn_pred)
                            prediction = ensemble
                        else:
                            prediction = rf_pred
                    else:
                        prediction = rf_pred
                    st.session_state[StateKey.PREDICTED_LOGS] = float(prediction)
                    st.session_state[StateKey.ACTUAL_MODEL] = model_type
                    st.session_state[StateKey.MODEL_DISAGREEMENT] = abs(rf_pred - gnn_pred) if gnn_pred is not None else 0.0

                # ── SHAP (available for RF, Ensemble, and Ensemble(W); skipped for GNN-only) ──
                status.update(label=t("app.predict.step.shap"))
                shap_disabled_models = {"GNN"}
                if actual_model in shap_disabled_models:
                    st.session_state[StateKey.SHAP_VALUES] = None
                    st.session_state[StateKey.SHAP_NAMES] = None
                else:
                    try:
                        combined_shap, combined_names = cached_shap_contributions(current)
                        st.session_state[StateKey.SHAP_VALUES] = combined_shap
                        st.session_state[StateKey.SHAP_NAMES] = combined_names
                    except Exception:
                        st.session_state[StateKey.SHAP_VALUES] = None
                        st.session_state[StateKey.SHAP_NAMES] = None
                st.session_state[StateKey.AI_EXPLANATION] = None

                if not ood_already_done:
                    status.update(label=t("app.predict.step.ood_post"))
                    try:
                        ood_risk, ood_result = run_ood_check(features, fp_array)
                    except Exception:
                        ood_risk, ood_result = "UNKNOWN", None
                    st.session_state[StateKey.OOD_RISK] = ood_risk
                    st.session_state[StateKey.OOD_RESULT] = ood_result

                # ── Disagreement warning ──
                disagreement = st.session_state.get(StateKey.MODEL_DISAGREEMENT, 0.0)
                if disagreement > 1.0:
                    st.warning(t("app.warn.disagreement.severe", diff=disagreement))
                elif disagreement > 0.5:
                    st.info(t("app.warn.disagreement.moderate", diff=disagreement))

                _model_label_map = {
                    "RF": t("app.model_label.rf"),
                    "GNN": t("app.model_label.gnn"),
                    "Ensemble": t("app.model_label.ensemble"),
                    "Auto": t("app.model_label.auto", actual=actual_model),
                }
                model_label = _model_label_map.get(model_type, model_type)
                status.update(label=t("app.predict.complete", model=model_label, logS=float(prediction)),
                              state=t("app.predict.complete_state"))

                # ── Save to prediction history (in-memory, persisted lazily) ──
                try:
                    from molecules import MOLECULE_DB
                    mol_name = st.session_state.get(StateKey.CURRENT_MOLECULE_NAME, "")
                    if mol_name:
                        known_smiles = MOLECULE_DB.get(mol_name)
                        if known_smiles and known_smiles != current:
                            mol_name = ""
                            st.session_state[StateKey.CURRENT_MOLECULE_NAME] = ""
                    if not mol_name:
                        radio_key = st.session_state.get("molecule_select_radio")
                        if radio_key and radio_key in MOLECULE_DB and MOLECULE_DB[radio_key] == current:
                            mol_name = radio_key
                    if not mol_name:
                        mol_name = next(
                            (k for k, v in MOLECULE_DB.items() if v == current and k != "(自定义输入)"),
                            "",
                        )
                except Exception:
                    mol_name = ""
                import datetime
                from copy import deepcopy
                _logS = float(prediction)
                _pKa = st.session_state.get(StateKey.PREDICTED_PKA)
                history_entry = {
                    "smiles": current,
                    "name": mol_name,
                    "logS": _logS,
                    "pKa": _pKa,
                    "timestamp": datetime.datetime.now().strftime("%H:%M"),
                }
                history = st.session_state[StateKey.PREDICTION_HISTORY]
                st.toast(t("app.toast.prediction", name=mol_name, logS=_logS, pKa=_pKa))
                if not history or history[0].get("smiles") != current:
                    new_history = [deepcopy(history_entry)] + [deepcopy(e) for e in history][:14]  # keep max 15
                    st.session_state[StateKey.PREDICTION_HISTORY] = new_history
                    st.session_state["__history_dirty"] = True


# ========== Display results (5 tabs) ==========
if st.session_state.get(StateKey.PREDICTED_SMILES) and st.session_state.get(StateKey.PREDICTED_LOGS) is not None:
    render_results(model)


# ========== Batch CSV prediction ==========
st.markdown("---")
with st.expander(t("app.batch.title"), expanded=False):
    st.caption(t("app.batch.desc"))

    batch_file = st.file_uploader(
        t("app.batch.upload_label"),
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
            st.info(t("app.batch.file_info", name=batch_file.name, rows=len(df), col=header[smiles_col]))
            st.dataframe(df.head(5), use_container_width=True)

            if st.button(t("app.batch.start_btn"), key="batch_predict_btn", use_container_width=True):
                # Validate SMILES column content
                if header[smiles_col] not in df.columns:
                    st.error(t("app.batch.col_error", col=header[smiles_col]))
                else:
                    results = []
                    smiles_list = df[header[smiles_col]].dropna().astype(str).tolist()
                    progress_bar = st.progress(0, text=t("app.batch.progress_start", total=len(smiles_list)))

                    from features import compute_features

                    # Step 1: compute features for all molecules (RDKit is fast)
                    features_list = []
                    valid_indices = []
                    for i, smi in enumerate(smiles_list):
                        feat_result = compute_features(smi)
                        if feat_result is None:
                            results.append({
                                "SMILES": smi,
                                "logS": None,
                                "Solubility Level": "Invalid SMILES",
                                "pKa": None,
                                "MolWt": None,
                                "LogP": None,
                            })
                        else:
                            features_list.append(feat_result)
                            valid_indices.append(i)
                        progress_bar.progress(
                            (i + 1) / (len(smiles_list) + 1),
                            text=t("app.batch.progress_feature", i=i+1, total=len(smiles_list)),
                        )

                    # Step 2: vectorized batch prediction for all valid molecules
                    if features_list:
                        X_batch = np.vstack([
                            np.hstack([list(f.values()), fp])
                            for f, fp in features_list
                        ])
                        batch_model_type = st.session_state.get(StateKey.SELECTED_MODEL, "Auto")

                        # RF prediction (always needed)
                        rf_batch = model.predict(X_batch)

                        # pKa prediction (lazy-loaded, cached after first use)
                        try:
                            _pka_model = load_pka_model()
                        except Exception:
                            _pka_model = None
                        if _pka_model is not None:
                            X_batch_pka = np.vstack([
                                np.hstack([
                                    [f[k] for k in ["MolWt", "LogP", "NumHDonors", "NumHAcceptors",
                                                     "TPSA", "NumRotatableBonds", "NumAromaticRings",
                                                     "NumAliphaticRings"]],
                                    fp,
                                ])
                                for f, fp in features_list
                            ])
                            pKa_batch = _pka_model.predict(X_batch_pka)
                        else:
                            pKa_batch = [None] * len(features_list)

                        # For Auto+ mode: compute OOD for all, GNN only where needed
                        need_gnn = [False] * len(features_list)
                        ood_risks = [None] * len(features_list)
                        if batch_model_type == "Auto" and gnn_ready:
                            for j in range(len(features_list)):
                                features, fp = features_list[j]
                                ood_risk, _ = run_ood_check(features, fp)
                                ood_risks[j] = ood_risk
                                # Auto+: LOW→weighted (needs GNN), MEDIUM/HIGH→pure GNN
                                need_gnn[j] = True  # always need GNN for Auto+ weighting
                        elif batch_model_type in ("GNN", "Ensemble") and gnn_ready:
                            need_gnn = [True] * len(features_list)

                        # GNN prediction (parallelized with ThreadPoolExecutor)
                        gnn_batch = [None] * len(features_list)
                        if any(need_gnn) and gnn_ready:
                            indices_to_predict = [
                                (j, smiles_list[valid_indices[j]])
                                for j in range(len(features_list))
                                if need_gnn[j]
                            ]
                            total_gnn = len(indices_to_predict)
                            completed = 0
                            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                                future_map = {
                                    pool.submit(cached_gnn_predict, smi): j
                                    for j, smi in indices_to_predict
                                }
                                for future in concurrent.futures.as_completed(future_map):
                                    j = future_map[future]
                                    try:
                                        gnn_batch[j] = future.result()
                                    except Exception:
                                        gnn_batch[j] = None
                                    completed += 1
                                    progress_bar.progress(
                                        (len(smiles_list) + completed) / (len(smiles_list) + len(features_list) + 1),
                                        text=t("app.batch.progress_gnn", i=completed, total=total_gnn),
                                    )

                        for j, idx in enumerate(valid_indices):
                            features, _ = features_list[j]
                            rf_val = float(rf_batch[j])
                            gnn_val = gnn_batch[j]

                            if batch_model_type == "Auto":
                                ood_risk = ood_risks[j]
                                if ood_risk == "LOW" and gnn_val is not None:
                                    logS = predict_solubility_weighted(rf_val, gnn_val)
                                elif gnn_val is not None:
                                    logS = gnn_val
                                else:
                                    logS = rf_val
                            elif batch_model_type == "GNN":
                                logS = gnn_val if gnn_val is not None else rf_val
                            elif batch_model_type == "Ensemble":
                                if gnn_val is not None:
                                    ensemble_val, _, _ = predict_solubility_ensemble(rf_val, gnn_val)
                                    logS = ensemble_val
                                else:
                                    logS = rf_val
                            else:
                                logS = rf_val

                            pKa_val = float(pKa_batch[j]) if _pka_model is not None else None
                            level = get_solubility_level(logS)[0]

                            if batch_model_type == "Auto":
                                ood_risk = ood_risks[j]
                                actual_m = "Ensemble(W)" if ood_risk == "LOW" else "GNN"
                                row = {
                                    "SMILES": smiles_list[idx],
                                    "logS": f"{logS:.3f}",
                                    "Model": f"Auto→{actual_m}",
                                    "Solubility Level": level,
                                    "pKa": f"{pKa_val:.2f}" if pKa_val is not None else "?",
                                    "MolWt": f"{features['MolWt']:.1f}",
                                    "LogP": f"{features['LogP']:.2f}",
                                    "RF_pred": f"{rf_val:.3f}",
                                    "GNN_pred": f"{gnn_val:.3f}" if gnn_val is not None else "?",
                                    "|RF-GNN|": f"{abs(rf_val - gnn_val):.3f}" if gnn_val is not None else "?",
                                }
                            elif batch_model_type == "Ensemble":
                                row = {
                                    "SMILES": smiles_list[idx],
                                    "logS": f"{logS:.3f}",
                                    "Model": batch_model_type,
                                    "Solubility Level": level,
                                    "pKa": f"{pKa_val:.2f}" if pKa_val is not None else "?",
                                    "MolWt": f"{features['MolWt']:.1f}",
                                    "LogP": f"{features['LogP']:.2f}",
                                    "RF_pred": f"{rf_val:.3f}",
                                    "GNN_pred": f"{gnn_val:.3f}" if gnn_val is not None else "?",
                                    "|RF-GNN|": f"{abs(rf_val - gnn_val):.3f}" if gnn_val is not None else "?",
                                }
                            else:
                                row = {
                                    "SMILES": smiles_list[idx],
                                    "logS": f"{logS:.3f}",
                                    "Model": batch_model_type,
                                    "Solubility Level": level,
                                    "pKa": f"{pKa_val:.2f}" if pKa_val is not None else "?",
                                    "MolWt": f"{features['MolWt']:.1f}",
                                    "LogP": f"{features['LogP']:.2f}",
                                }
                            results.insert(idx, row)

                    progress_bar.empty()
                    result_df = pd.DataFrame(results)
                    st.success(t("app.batch.complete", n=len(results)))
                    st.dataframe(result_df, use_container_width=True, height=400)

                    # Download button
                    csv_out = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        t("app.batch.download_btn"),
                        data=csv_out,
                        file_name="dissolve_batch_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
        except Exception as batch_err:
            st.error(t("app.batch.error", err=batch_err))
            import traceback
            st.code(traceback.format_exc())

# ========== Lazy-persist prediction history (only write when dirty) ==========
if st.session_state.get("__history_dirty") and StateKey.PREDICTION_HISTORY in st.session_state:
    try:
        hist_path = os.path.join(os.path.dirname(__file__), '.prediction_history.json')
        with open(hist_path, 'w', encoding='utf-8') as hf:
            json.dump(st.session_state[StateKey.PREDICTION_HISTORY], hf, ensure_ascii=False)
        st.session_state["__history_dirty"] = False
    except Exception:
        pass

# ========== Render footer ==========
render_footer()
