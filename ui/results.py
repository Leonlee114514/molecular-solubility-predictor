"""
DisSolve - Prediction results tab display (5 tabs).
"""

import numpy as np
import streamlit as st
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ui.plots import mol_to_dark_image, mol_to_dark_image_with_importance
from model import get_pka_type, get_solubility_level, get_shap_explainer
from core.ai_client import explain_with_kimi
from core.i18n import t, get_lang
from core.cache import (
    cached_compute_features, cached_show_3d, cached_pka_analysis,
    cached_shap_contributions, cached_lipinski, cached_admet, cached_druglikeness,
    cached_gnn_explanation,
)
from core.state_keys import StateKey
from ui.components import render_html
from ui.plots import setup_plt_dark


def render_results(model):
    """Render the 5-tab prediction results. Must be called only when predictions exist."""
    # ========== 显示预测结果（Tab分组版）==========
    if st.session_state.get(StateKey.PREDICTED_SMILES) and st.session_state.get(StateKey.PREDICTED_LOGS) is not None:

        features = st.session_state.get(StateKey.CACHED_FEATURES)
        if features is None:
            result_display = cached_compute_features(st.session_state[StateKey.PREDICTED_SMILES])
            if result_display is None:
                st.error(t("result.display_error"))
                st.stop()
            features, _ = result_display
        prediction = st.session_state[StateKey.PREDICTED_LOGS]

        # ── 预计算pKa相关变量（供多Tab使用）──
        pka_val = st.session_state.get(StateKey.PREDICTED_PKA)
        pka_type = pka_label = pka_css = pka_text_color = pka_desc = None
        if pka_val is not None:
            pka_type, pka_label, pka_css, pka_text_color, pka_desc = get_pka_type(pka_val)

        # ── 溶解度判定（供多Tab使用）──
        interp, color, css_class = get_solubility_level(prediction)

        # ── OOD 分布检测结果显示 ──
        ood_risk = st.session_state.get(StateKey.OOD_RISK, "UNKNOWN")
        ood_result = st.session_state.get(StateKey.OOD_RESULT)

        if ood_risk == "HIGH":
            st.error(t("result.ood.high.title"))
            with st.container(border=True):
                st.markdown(f"""
                <div style="font-size:0.9rem;line-height:1.7;color:#fca5a5;">
                <strong>{t('result.ood.high.desc')}</strong>
                </div>
                """, unsafe_allow_html=True)
                if ood_result and ood_result.warnings:
                    for w in ood_result.warnings:
                        st.markdown(f"- {w}")
        elif ood_risk == "MEDIUM":
            st.warning(t("result.ood.medium.title"))
            with st.container(border=True):
                st.markdown(f"""
                <div style="font-size:0.9rem;line-height:1.7;color:#fcd34d;">
                <strong>{t('result.ood.medium.desc')}</strong>
                </div>
                """, unsafe_allow_html=True)
                if ood_result and ood_result.warnings:
                    for w in ood_result.warnings:
                        st.markdown(f"- {w}")
        elif ood_risk == "LOW":
            st.success(t("result.ood.low"))

        # ═════════════════════════════════════════
        # TAB 分组
        # ═════════════════════════════════════════
        tab_labels = [
            t("result.tab.preview"),
            t("result.tab.solubility"),
            t("result.tab.pka"),
            t("result.tab.pharmacology"),
            t("result.tab.ai"),
        ]
        tab_preview, tab_sol, tab_pka, tab_pharm, tab_ai = st.tabs(tab_labels)

        # 记住 rerun 后的目标 Tab（解决 st.rerun() 导致 Tab 重置的问题）
        target_tab = st.session_state.pop(StateKey.TARGET_TAB, None)
        if target_tab is not None:
            target_idx = tab_labels.index(target_tab) if target_tab in tab_labels else 0
            render_html(f"""
            <script>
            (function() {{
                var tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
                if (tabs.length > {target_idx}) {{
                    setTimeout(function() {{ tabs[{target_idx}].click(); }}, 80);
                }}
            }})();
            </script>
            """, height=1)

        with tab_preview:
            _tab_preview(features)

        with tab_sol:
            _tab_solubility(features, prediction, interp, color, css_class, model)

        with tab_pka:
            _tab_pka(pka_val, pka_type, pka_label, pka_css, pka_text_color, pka_desc, features)

        with tab_pharm:
            _tab_pharmacology(features, prediction, pka_val, pka_type, pka_label, pka_css, pka_text_color, pka_desc)

        with tab_ai:
            _tab_ai(features, prediction, pka_val, pka_type)


# ════════════════════════════════════════════════════════════════════════════════
# Per-tab extraction functions (called from render_results)
# ════════════════════════════════════════════════════════════════════════════════

def _tab_preview(features):
    """Tab 0: Molecule Preview (2D structure, formula, 3D model)."""
    st.markdown(f"""<div class="card-title">{t('result.preview.card_title')}</div>""", unsafe_allow_html=True)

    mol = Chem.MolFromSmiles(st.session_state[StateKey.PREDICTED_SMILES])

    col_pv_left, col_pv_right = st.columns([1, 1])

    with col_pv_left:
        try:
            img = mol_to_dark_image(mol, size=(460, 380))
            if img is not None:
                st.image(img, caption=t("result.preview.img_caption"), use_container_width=True)
            else:
                st.info(t("result.preview.img_fail"))
        except Exception as e:
            st.warning(t("result.preview.img_error", err=e))

    with col_pv_right:
        if mol is not None:
            formula = rdMolDescriptors.CalcMolFormula(mol)
            mw = features["MolWt"]
            st.metric(t("result.preview.formula"), formula)
            st.metric(t("result.preview.mol_weight"), f"{mw:.1f} Da")
            st.markdown(f"""<div style="font-size:0.82rem;color:var(--ob-text-tertiary);margin-top:0.5rem;margin-bottom:0.2rem;">{t('result.preview.smiles_label')}</div>""", unsafe_allow_html=True)
            st.code(st.session_state[StateKey.PREDICTED_SMILES], language=None)
        else:
            st.warning(t("result.preview.struct_fail"))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""<div class="card-title">{t('result.preview.model3d')}</div>""", unsafe_allow_html=True)
    html_3d = cached_show_3d(st.session_state[StateKey.PREDICTED_SMILES])
    if html_3d:
        render_html(html_3d, height=420)
    else:
        st.info(t("result.preview.model3d_fail"))


def _tab_solubility(features, prediction, interp, color, css_class, model):
    """Tab 1: Solubility prediction details + SHAP explainability."""
    st.markdown(f"""<div class="card-title">{t('result.solubility.card_title')}</div>""", unsafe_allow_html=True)

    model_type = st.session_state.get(StateKey.SELECTED_MODEL, "Auto")
    if model_type == "Auto":
        actual_model = st.session_state.get(StateKey.ACTUAL_MODEL, "GNN")
        model_type = actual_model if actual_model in ("RF", "GNN", "Ensemble", "Ensemble(W)") else "GNN"
    model_colors = {"RF": "#34d399", "GNN": "#a78bfa", "Ensemble": "#fbbf24", "Ensemble(W)": "#f97316"}
    model_badge = {
        "RF": t("result.solubility.badge_rf"),
        "GNN": t("result.solubility.badge_gnn"),
        "Ensemble": t("result.solubility.badge_ensemble"),
        "Ensemble(W)": t("result.solubility.badge_weighted"),
    }.get(model_type, model_type)
    st.markdown(f"""
    <div style="display:inline-block;padding:0.15rem 0.7rem;background:rgba({model_colors.get(model_type, 'a78bfa').lstrip('#')},0.15);border:1px solid {model_colors.get(model_type, '#a78bfa')};border-radius:20px;font-size:0.78rem;color:{model_colors.get(model_type, '#a78bfa')};margin-bottom:0.6rem;">
        Model: {model_badge}
    </div>
    """, unsafe_allow_html=True)

    col_sol1, col_sol2 = st.columns([1, 1.2])
    with col_sol1:
        st.metric(label=t("result.solubility.metric_logs"), value=f"{prediction:.3f}")
        st.markdown(f"""
        <div class="{css_class}">
            <div style="font-size: 1.1rem; font-weight: 700; color: {color};">-> {interp}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Disagreement warning (all modes) ──
        rf_val = st.session_state.get(StateKey.PREDICTED_LOGS_RF)
        gnn_val = st.session_state.get(StateKey.PREDICTED_LOGS_GNN)
        selected_model = st.session_state.get(StateKey.SELECTED_MODEL, "Auto")
        if rf_val is not None and gnn_val is not None:
            diff = abs(rf_val - gnn_val)
            if diff > 1.0:
                if selected_model == "Auto":
                    st.error(t("result.solubility.severe_disagree_auto", diff=diff))
                else:
                    st.error(t("result.solubility.severe_disagree", diff=diff))
            elif diff > 0.5:
                st.error(t("result.solubility.notable_disagree", diff=diff))

        # ── Ensemble component display ──
        if model_type in ("Ensemble", "Ensemble(W)"):
            if rf_val is not None and gnn_val is not None:
                diff = abs(rf_val - gnn_val)
                weighted = 0.45 * rf_val + 0.55 * gnn_val
                is_weighted = model_type == "Ensemble(W)"
                title = t("result.solubility.badge_weighted")
                title_color = "#f97316" if is_weighted else "#fbbf24"
                _agree = t("result.solubility.good_agreement") if diff < 0.5 else (t("result.solubility.notable_disagreement") if diff < 1.0 else t("result.solubility.large_divergence"))
                st.markdown(f"""
                <div style="margin-top:0.8rem;padding:0.7rem 0.9rem;background:rgba({251 if is_weighted else 251},191,36,0.08);border-radius:10px;border:1px solid rgba({251 if is_weighted else 251},191,36,0.2);font-size:0.82rem;">
                    <b style="color:{title_color};">{title}</b><br>
                    <span style="color:#34d399;">{t('result.solubility.ensemble_rf')}</span> {rf_val:.3f} &nbsp;|&nbsp;
                    <span style="color:#a78bfa;">{t('result.solubility.ensemble_gnn')}</span> {gnn_val:.3f}<br>
                    <span style="color:#fbbf24;">{t('result.solubility.disagreement')}:</span> {diff:.3f}
                    <span style="color:#8b8b9b;font-size:0.75rem;">{_agree}</span>
                </div>
                """, unsafe_allow_html=True)

    with col_sol2:
        _high_str = t("result.solubility.high_hint")
        _poor_str = t("result.solubility.poor_hint")
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.03); border-radius: 14px; padding: 1rem; font-size: 0.85rem; color: var(--ob-text-tertiary); border: 1px solid var(--ob-border); font-family: 'Cascadia Code', 'Consolas', monospace;">
        <b style="color: var(--ob-text-secondary);">{t('result.solubility.guide')}</b><br>
        <span style="color: #34d399;">&gt;</span> logS > 0: {_high_str}<br>
        <span style="color: #fbbf24;">&gt;</span> -2 < logS < 0: {t('result.solubility.moderate')}<br>
        <span style="color: #f87171;">&gt;</span> logS < -2: {_poor_str}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""<div class="card-title">{t('result.solubility.descriptors')}</div>""", unsafe_allow_html=True)
    with st.container(border=True):
        desc_col1, desc_col2, desc_col3, desc_col4 = st.columns(4)
        with desc_col1:
            st.metric(t("result.solubility.desc_mw"), f"{features['MolWt']:.1f}")
            st.metric(t("result.solubility.desc_logp"), f"{features['LogP']:.2f}")
        with desc_col2:
            st.metric(t("result.solubility.desc_hbd"), f"{features['NumHDonors']}")
            st.metric(t("result.solubility.desc_hba"), f"{features['NumHAcceptors']}")
        with desc_col3:
            st.metric(t("result.solubility.desc_tpsa"), f"{features['TPSA']:.1f}")
            st.metric(t("result.solubility.desc_rotb"), f"{features['NumRotatableBonds']}")
        with desc_col4:
            st.metric(t("result.solubility.desc_arom"), f"{features['NumAromaticRings']}")
            st.metric(t("result.solubility.desc_aliph"), f"{features['NumAliphaticRings']}")
    st.info(f"""
    **{t('result.solubility.insight_title')}:**
    - {t('result.solubility.insight_tpsa')}
    - {t('result.solubility.insight_hbond')}
    - {t('result.solubility.insight_logp')}
    """)

    # ── SHAP / GNN Explainability section ──
    raw_sel = st.session_state.get(StateKey.SELECTED_MODEL, "Auto")
    if raw_sel == "Auto":
        shap_model_type = st.session_state.get(StateKey.ACTUAL_MODEL, "GNN")
    else:
        shap_model_type = raw_sel

    # GNNExplainer: show whenever GNN is part of the prediction
    if shap_model_type in ("GNN", "Ensemble", "Ensemble(W)"):
        _display_gnn_explanation()

    # SHAP: show whenever RF is part of the prediction (GNN-only has no SHAP data)
    if shap_model_type != "GNN":
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""<div class="card-title">{t('result.solubility.shap_title')}</div>""", unsafe_allow_html=True)
        st.caption(t("result.solubility.shap_guide"))

        if st.session_state.get(StateKey.SHAP_VALUES):
            shap_vals = np.array(st.session_state[StateKey.SHAP_VALUES])
            names = st.session_state[StateKey.SHAP_NAMES]
            abs_vals = np.abs(shap_vals)
            sorted_idx = np.argsort(abs_vals)[::-1][:8]
            top_shap = shap_vals[sorted_idx]
            top_names = [names[i] for i in sorted_idx]
            colors = ['#a78bfa' if v > 0 else '#06b6d4' for v in top_shap]
            setup_plt_dark()
            fig, ax = plt.subplots(figsize=(8, 4.5))
            bars = ax.barh(range(len(top_shap)), top_shap, color=colors, edgecolor="white", height=0.6)
            ax.invert_yaxis()
            for i, (bar, val) in enumerate(zip(bars, top_shap)):
                width = bar.get_width()
                label_x = width * 0.5
                ax.text(label_x, i, f"{val:+.3f}", va="center", ha="center", fontsize=10, fontweight="bold",
                        color="#ffffff",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=(0, 0, 0, 0.5),
                                  edgecolor=(1, 1, 1, 0.15), linewidth=0.5))
            ax.set_yticks(range(len(top_names)))
            ax.set_yticklabels(top_names, fontsize=11)
            ax.axvline(x=0, color="#f0f0f5", linewidth=1.0, alpha=0.4)
            ax.set_xlabel(t("result.solubility.shap_xlabel"), fontsize=11)
            ev = get_shap_explainer(model).expected_value
            if isinstance(ev, (list, tuple, np.ndarray)):
                base_value = float(np.array(ev).flatten()[0])
            else:
                base_value = float(ev)
            ax.set_title(t("result.solubility.shap_title_pred", pred=prediction, base=base_value), fontsize=12, pad=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            legend_elements = [
                Patch(facecolor="#a78bfa", label=t("result.solubility.shap_legend_pos")),
                Patch(facecolor="#06b6d4", label=t("result.solubility.shap_legend_neg"))
            ]
            ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            if prediction > 0:
                solubility_level = t("result.solubility.high")
            elif prediction > -2:
                solubility_level = t("result.solubility.moderate")
            else:
                solubility_level = t("result.solubility.poor")
            supporting = []
            resisting = []
            for i in range(min(3, len(top_names))):
                name = top_names[i]
                val = top_shap[i]
                if prediction <= -2:
                    if val < 0:
                        supporting.append("**" + name + "**（" + f"{val:.3f}" + "）")
                    else:
                        resisting.append("**" + name + "**（+" + f"{val:.3f}" + "）")
                elif prediction >= 0:
                    if val > 0:
                        supporting.append("**" + name + "**（+" + f"{val:.3f}" + "）")
                    else:
                        resisting.append("**" + name + "**（" + f"{val:.3f}" + "）")
                else:
                    direction = t("ai.shap.direction_up") if val > 0 else t("ai.shap.direction_down")
                    supporting.append("**" + name + "**（" + f"{val:+.3f}" + "，" + direction + "）")
            parts = [t("result.solubility.shap_insight_leading", level=solubility_level, logS=prediction)]
            if supporting:
                parts.append(t("result.solubility.shap_supporting", factors=", ".join(supporting)))
            if resisting:
                target = t("result.solubility.shap_target_soluble") if prediction <= -2 else t("result.solubility.shap_target_insoluble")
                parts.append(t("result.solubility.shap_resisting", target=target, factors=", ".join(resisting)))
            shift = abs(prediction - base_value)
            direction = t("result.solubility.shap_dir_up") if prediction > base_value else t("result.solubility.shap_dir_down")
            parts.append(t("result.solubility.shap_shift", base=base_value, direction=direction, shift=shift))
            insight_text = " ".join(parts)
            st.info(insight_text)


def _tab_pka(pka_val, pka_type, pka_label, pka_css, pka_text_color, pka_desc, features):
    """Tab 2: pKa acidity/basicity prediction."""
    if pka_val is not None:
        st.markdown(f"""<div class="card-title">{t('result.pka.card_title')}</div>""", unsafe_allow_html=True)
        col_pka1, col_pka2 = st.columns(2)
        with col_pka1:
            st.metric(t("result.pka.metric"), f"{pka_val:.2f}")
        with col_pka2:
            st.markdown(f"""
            <div class="{pka_css}" style="margin-top: 0.2rem;">
                <div style="font-size: 1rem; font-weight: 700; color: {pka_text_color};">-> {pka_label}</div>
                <div style="font-size: 0.8rem; color: var(--ob-text-tertiary); margin-top: 0.3rem;">{pka_desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""<div class="card-title">{t('result.pka.decomp_title')}</div>""", unsafe_allow_html=True)
        chem_factors = cached_pka_analysis(st.session_state[StateKey.PREDICTED_SMILES], pka_val)
        if chem_factors:
            names = list(chem_factors.keys())
            vals = list(chem_factors.values())
            colors = ['#a78bfa' if v > 0 else '#22d3ee' for v in vals]
            setup_plt_dark()
            fig, ax = plt.subplots(figsize=(8, 4.5))
            bars = ax.barh(range(len(vals)), vals, color=colors, edgecolor=(1, 1, 1, 0.15), height=0.6, linewidth=0.5)
            ax.invert_yaxis()
            ax.axvline(x=0, color='#f0f0f5', linewidth=1.0, alpha=0.4)
            for bar, val in zip(bars, vals):
                width = bar.get_width()
                label_x = width * 0.5
                ax.text(label_x, bar.get_y() + bar.get_height()/2,
                        f'{val:+.2f}', va='center', ha='center', fontsize=10, fontweight='bold',
                        color='#ffffff',
                        bbox=dict(boxstyle='round,pad=0.25', facecolor=(0, 0, 0, 0.35),
                                  edgecolor='none', alpha=0.9))
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=10)
            pka_unit = t("result.pka.unit_acid") if pka_val < 7 else t("result.pka.unit_base")
            ax.set_xlabel(t("result.pka.chart_xlabel", unit=pka_unit), fontsize=11)
            ax.set_title(t("result.pka.chart_title", val=pka_val), fontsize=12, pad=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('#33334d')
            legend_type = t("result.pka.legend_type_acid") if pka_val < 7 else t("result.pka.legend_type_base")
            legend_elements = [
                Patch(facecolor='#a78bfa', label=t("result.pka.legend_enhance", type=legend_type)),
                Patch(facecolor='#22d3ee', label=t("result.pka.legend_weaken", type=legend_type))
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
                      framealpha=0.8, facecolor='#1a1a2e', edgecolor=(1, 1, 1, 0.1))
            plt.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            st.caption(t("result.pka.factor_guide"))
            st.markdown(f"""
            <div style="margin-top: 0.3rem; padding: 0.65rem 0.9rem; background: rgba(124, 58, 237, 0.06); border-left: 2px solid rgba(124, 58, 237, 0.3); border-radius: 4px; font-size: 0.82rem; color: #a0a0b5; line-height: 1.9;">
            <b style="color: #c4b5fd;">{t('result.pka.glossary_title')}</b> &nbsp;{'Click terms to see bilingual definition:' if get_lang() == 'en' else '点击术语查看中英双语定义：'}<br>
            &bull; {t('result.pka.glossary_inductive')}<br>
            &bull; {t('result.pka.glossary_resonance')}<br>
            &bull; {t('result.pka.glossary_intra_hb')}<br>
            &bull; {t('result.pka.glossary_steric')}<br>
            &bull; {t('result.pka.glossary_hybrid')}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(t("result.pka.unavailable_short"))
    else:
        st.info(t("result.pka.model_unavailable_short"))


def _tab_pharmacology(features, prediction, pka_val, pka_type, pka_label, pka_css, pka_text_color, pka_desc):
    """Tab 3: Pharmacology (Lipinski, ADME/Tox, ionization profile)."""
    st.markdown(f"""<div class="card-title">{t('result.pharma.lipinski_title')}</div>""", unsafe_allow_html=True)
    lipinski_result = cached_lipinski(tuple(features.items()))
    rules = lipinski_result["rules"]
    setup_plt_dark()

    fig, ax = plt.subplots(figsize=(10, 3.6))

    property_labels = [
        (t("result.pharma.lipinski_prop_mw"), "≤ 500 Da"),
        (t("result.pharma.lipinski_prop_logp"), "≤ 5"),
        (t("result.pharma.lipinski_prop_hbd"), "≤ 5"),
        (t("result.pharma.lipinski_prop_hba"), "≤ 10"),
        (t("result.pharma.lipinski_prop_rotb"), "≤ 10"),
    ]

    y_positions = list(range(5))
    for i, ((prop_name, threshold), rule) in enumerate(zip(property_labels, rules)):
        _, _, passed, actual = rule
        color = '#34d399' if passed else '#f87171'
        ax.barh(i, 1, color='#1e1e30', edgecolor=(1, 1, 1, 0.06), height=0.65, zorder=1)
        ax.barh(i, 1, color=color, edgecolor=(1, 1, 1, 0.12), height=0.65, alpha=0.82, zorder=2)
        ax.text(-0.02, i, prop_name, va='center', ha='right', fontsize=10,
                fontweight='600', color='#d0d0e0', zorder=5)
        ax.text(0.03, i, threshold, va='center', ha='left', fontsize=9,
                color='#7b7b8b', zorder=5)
        icon = "PASS" if passed else "FAIL"
        ax.text(0.53, i, f"{actual}  [{icon}]", va='center', ha='center',
                fontsize=11, fontweight='bold', color='#ffffff',
                bbox=dict(boxstyle='round,pad=0.35', facecolor=(0, 0, 0, 0.55),
                          edgecolor=(1, 1, 1, 0.10)), zorder=6)

    legend_elements = [
        Patch(facecolor='#34d399', alpha=0.82, label=t("result.pharma.lipinski_legend_pass")),
        Patch(facecolor='#f87171', alpha=0.82, label=t("result.pharma.lipinski_legend_fail")),
    ]
    ax.legend(handles=legend_elements, loc='lower center', fontsize=9,
              ncol=2, framealpha=0.7, facecolor='#1a1a2e', edgecolor=(1, 1, 1, 0.08),
              bbox_to_anchor=(0.5, -0.35))

    ax.set_xlim(-0.42, 1.05)
    ax.set_ylim(-0.6, 4.6)
    ax.axis('off')

    title_color = '#34d399' if lipinski_result['passed'] >= 4 else '#fbbf24'
    ax.set_title(
        t("result.pharma.lipinski_score_title", score=str(lipinski_result["passed"]), total="5", text=lipinski_result["interpretation"]),
        fontsize=12, pad=16, color=title_color, fontweight='600')

    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.markdown(f"""
    <div style="margin-top: 0.3rem; padding: 0.65rem 0.9rem; background: rgba(124, 58, 237, 0.06); border-left: 2px solid rgba(124, 58, 237, 0.3); border-radius: 4px; font-size: 0.82rem; color: #a0a0b5; line-height: 1.9;">
    <b style="color: #c4b5fd;">{'About Lipinski' if get_lang() == 'en' else '关于 Lipinski'}</b><br>
    {t('result.pharma.lipinski_history_html')}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""<div class="card-title">{t('result.pharma.druglikeness_title')}</div>""", unsafe_allow_html=True)
    dl = cached_druglikeness(st.session_state[StateKey.PREDICTED_SMILES])
    if dl:
        col_qed, col_sa, col_fsp3 = st.columns(3)
        with col_qed:
            qed_val = dl["qed"]
            st.markdown(f"""
            <div style="text-align:center;padding:1.2rem 1rem;background:linear-gradient(155deg,rgba(30,30,50,0.75),rgba(18,18,35,0.65));border-radius:14px;border:1px solid rgba(124,58,237,0.12);backdrop-filter:blur(8px);">
                <div style="font-size:0.7rem;color:#8b8b9b;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.3rem;">Quantitative Estimate of Drug-likeness</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="text-align:center;margin-top:-0.5rem;">
                <div style="font-size:2.2rem;font-weight:700;font-family:'Cascadia Code',monospace;color:{dl['qed_color']};">{qed_val:.3f}</div>
                <div style="margin:0.5rem 0;height:6px;background:#1e1e30;border-radius:3px;overflow:hidden;max-width:200px;margin-left:auto;margin-right:auto;">
                    <div style="height:100%;width:{qed_val*100:.0f}%;background:{dl['qed_color']};border-radius:3px;transition:width 0.4s ease;"></div>
                </div>
                <div style="font-size:0.85rem;font-weight:600;color:{dl['qed_color']};margin-bottom:0.5rem;">{dl['qed_level']}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_sa:
            sa_val = dl["sascore"]
            sa_pct = max(5, min(100, (1 - (sa_val - 1) / 9) * 100))
            st.markdown(f"""
            <div style="text-align:center;padding:1.2rem 1rem;background:linear-gradient(155deg,rgba(30,30,50,0.75),rgba(18,18,35,0.65));border-radius:14px;border:1px solid rgba(6,182,212,0.12);backdrop-filter:blur(8px);">
                <div style="font-size:0.7rem;color:#8b8b9b;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.3rem;">Synthetic Accessibility Score</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="text-align:center;margin-top:-0.5rem;">
                <div style="font-size:2.2rem;font-weight:700;font-family:'Cascadia Code',monospace;color:{dl['sa_color']};">{sa_val:.2f}</div>
                <div style="margin:0.5rem 0;height:6px;background:#1e1e30;border-radius:3px;overflow:hidden;max-width:200px;margin-left:auto;margin-right:auto;">
                    <div style="height:100%;width:{sa_pct:.0f}%;background:{dl['sa_color']};border-radius:3px;transition:width 0.4s ease;"></div>
                </div>
                <div style="font-size:0.85rem;font-weight:600;color:{dl['sa_level']};margin-bottom:0.5rem;">{dl['sa_level']}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_fsp3:
            fsp3_val = dl["fsp3"]
            st.markdown(f"""
            <div style="text-align:center;padding:1.2rem 1rem;background:linear-gradient(155deg,rgba(30,30,50,0.75),rgba(18,18,35,0.65));border-radius:14px;border:1px solid rgba(251,191,36,0.12);backdrop-filter:blur(8px);">
                <div style="font-size:0.7rem;color:#8b8b9b;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.3rem;">Fraction sp&sup3; Carbons</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="text-align:center;margin-top:-0.5rem;">
                <div style="font-size:2.2rem;font-weight:700;font-family:'Cascadia Code',monospace;color:{dl['fsp3_color']};">{fsp3_val:.3f}</div>
                <div style="margin:0.5rem 0;height:6px;background:#1e1e30;border-radius:3px;overflow:hidden;max-width:200px;margin-left:auto;margin-right:auto;">
                    <div style="height:100%;width:{fsp3_val*100:.0f}%;background:{dl['fsp3_color']};border-radius:3px;transition:width 0.4s ease;"></div>
                </div>
                <div style="font-size:0.85rem;font-weight:600;color:{dl['fsp3_level']};margin-bottom:0.5rem;">{dl['fsp3_level']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:0.6rem;text-align:center;font-size:0.72rem;color:#6b6b7b;">
        {t('result.pharma.fsp3_detail', n_sp3=dl['n_sp3'], n_carbon=dl['n_carbons'])}<br>
        Refs: Bickerton et al. (2012), Ertl &amp; Schuffenhauer (2009), Lovering et al. (2009)
        </div>
        """, unsafe_allow_html=True)

    if pka_val is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""<div class="card-title">{t('result.pharma.ionization_title')}</div>""", unsafe_allow_html=True)
        env_ph = [1.5, 4.5, 6.8, 7.4]
        env_names = [t("result.pharma.ionization_env_stomach"), t("result.pharma.ionization_env_duodenum"),
                     t("result.pharma.ionization_env_intestine"), t("result.pharma.ionization_env_blood")]
        if pka_type == "acid":
            fractions = [1 / (1 + 10**(ph - pka_val)) for ph in env_ph]
        else:
            fractions = [1 / (1 + 10**(pka_val - ph)) for ph in env_ph]
        setup_plt_dark()
        fig, ax = plt.subplots(figsize=(7, 3.2))
        colors_bar = ['#f87171', '#fbbf24', '#34d399', '#60a5fa']
        bars = ax.bar(env_names, [f*100 for f in fractions], color=colors_bar, edgecolor='white', width=0.6)
        for bar, frac in zip(bars, fractions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{frac*100:.1f}%', ha='center', va='bottom', fontsize=10, fontfamily='monospace')
        ax.set_ylabel(t("result.pharma.ionization_ylabel"), fontsize=11)
        ax.set_ylim(0, 105)
        ax.set_title(t("result.pharma.ionization_chart_title", val=pka_val), fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""<div class="card-title">{t('result.pharma.pharma_analysis')}</div>""", unsafe_allow_html=True)
        with st.container(border=True):
            if pka_type == "acid":
                if pka_val < 4:
                    st.success(t("result.pharma.analysis.strong_acid"))
                else:
                    st.info(t("result.pharma.analysis.mid_acid"))
            elif pka_type == "base":
                if pka_val > 9:
                    st.warning(t("result.pharma.analysis.strong_base"))
                else:
                    st.info(t("result.pharma.analysis.weak_base"))
            else:
                st.info(t("result.pharma.analysis.amphoteric"))

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""<div class="card-title">{t('result.pharma.linkage.title')}</div>""", unsafe_allow_html=True)
        logS = prediction
        parts = []
        if logS > 0:
            parts.append(t("result.pharma.linkage.soluble"))
        elif logS > -2:
            parts.append(t("result.pharma.linkage.moderate"))
        else:
            parts.append(t("result.pharma.linkage.poor"))
        if pka_type == "acid":
            if pka_val < 4:
                parts.append(t("result.pharma.linkage.pka_weak_acid", val=pka_val))
            else:
                parts.append(t("result.pharma.linkage.pka_mid_acid", val=pka_val))
        elif pka_type == "base":
            if pka_val > 9:
                parts.append(t("result.pharma.linkage.pka_strong_base", val=pka_val))
            else:
                parts.append(t("result.pharma.linkage.pka_weak_base", val=pka_val))
        else:
            parts.append(t("result.pharma.linkage.pka_neutral", val=pka_val))
        if logS > 0 and pka_type == "acid" and pka_val < 4:
            parts.append(t("result.pharma.linkage.combo_good"))
        elif logS < -2 and pka_type == "base" and pka_val > 9:
            parts.append(t("result.pharma.linkage.combo_challenging"))
        elif logS > 0 and pka_type == "base" and pka_val > 9:
            parts.append(t("result.pharma.linkage.combo_acceptable"))
        st.info(" | ".join(parts))
    else:
        st.info(t("result.pharma.linkage.not_loaded"))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""<div class="card-title">{t('result.pharma.admet_title')}</div>""", unsafe_allow_html=True)

    admet = cached_admet(
        st.session_state[StateKey.PREDICTED_SMILES],
        tuple(features.items()),
        pka_val
    )

    adme_tabs = st.tabs([
        t("result.pharma.admet_absorption"),
        t("result.pharma.admet_distribution"),
        t("result.pharma.admet_metabolism"),
        t("result.pharma.admet_excretion"),
        t("result.pharma.admet_toxicity"),
    ])

    with adme_tabs[0]:
        st.markdown(f"""
        <div style="padding: 1rem; background: rgba(52, 211, 153, 0.06); border-radius: 12px; border: 1px solid rgba(52, 211, 153, 0.15);">
        <b style="color: #34d399;">{t('result.pharma.admet_absorption_title')}</b><br><br>
        <span style="color: #c0c0d0; font-size: 0.9rem; line-height: 1.7;">{admet['absorption']}</span>
        </div>
        """, unsafe_allow_html=True)
        st.caption(t("result.pharma.admet.absorption_caption"))

    with adme_tabs[1]:
        d = admet["distribution"]
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown(f"""
            <div style="padding: 0.8rem 1rem; background: rgba(96, 165, 250, 0.08); border-radius: 10px; border: 1px solid rgba(96, 165, 250, 0.15);">
            <b style="color: #60a5fa;">{t('result.pharma.admet_vd')}</b><br>
            <span style="color: #c0c0d0; font-size: 0.85rem;">{d['vd_estimate']}</span>
            </div>
            """, unsafe_allow_html=True)
        with col_d2:
            st.markdown(f"""
            <div style="padding: 0.8rem 1rem; background: rgba(96, 165, 250, 0.08); border-radius: 10px; border: 1px solid rgba(96, 165, 250, 0.15);">
            <b style="color: #60a5fa;">{t('result.pharma.admet_ppb')}</b><br>
            <span style="color: #c0c0d0; font-size: 0.85rem;">{d['ppb']}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="margin-top: 0.6rem; padding: 1rem; background: rgba(96, 165, 250, 0.06); border-radius: 12px; border: 1px solid rgba(96, 165, 250, 0.15);">
        <span style="color: #c0c0d0; font-size: 0.9rem; line-height: 1.7;">{d['summary']}</span>
        </div>
        """, unsafe_allow_html=True)
        st.caption(t("result.pharma.admet.distribution_caption"))

    with adme_tabs[2]:
        m = admet["metabolism"]
        col_m1, col_m2 = st.columns([1.5, 1])
        with col_m1:
            st.markdown(f"""
            <div style="padding: 1rem; background: rgba(251, 191, 36, 0.06); border-radius: 12px; border: 1px solid rgba(251, 191, 36, 0.15);">
            <b style="color: #fbbf24;">{t('result.pharma.admet_metab_sites')}</b><br><br>
            <span style="color: #c0c0d0; font-size: 0.9rem; line-height: 1.7;">{m['summary']}</span>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div style="padding: 1rem; background: rgba(251, 191, 36, 0.06); border-radius: 12px; border: 1px solid rgba(251, 191, 36, 0.15);">
            <b style="color: #fbbf24;">{t('result.pharma.admet_cyp')}</b><br><br>
            <span style="color: #c0c0d0; font-size: 0.9rem;">{m['cyp_enzymes']}</span>
            </div>
            """, unsafe_allow_html=True)
        st.caption(t("result.pharma.admet.metabolism_caption"))

    with adme_tabs[3]:
        e = admet["excretion"]
        st.markdown(f"""
        <div style="padding: 1rem; background: rgba(167, 139, 250, 0.06); border-radius: 12px; border: 1px solid rgba(167, 139, 250, 0.15);">
        <b style="color: #a78bfa;">{t('result.pharma.admet_excretion_route')}：{e['route']}</b><br><br>
        <span style="color: #c0c0d0; font-size: 0.9rem; line-height: 1.7;">{e['summary']}</span>
        </div>
        """, unsafe_allow_html=True)
        st.caption(t("result.pharma.admet.excretion_caption"))

    with adme_tabs[4]:
        alerts = admet["toxicity"]
        for risk_level, desc in alerts:
            # risk_level is a translated level word (高/中/低 or High/Medium/Low);
            # match case-insensitively against both languages so colors work in either UI language.
            risk_key = str(risk_level).strip().lower()
            if risk_key in ("高", "high"):
                bg = "rgba(248, 113, 113, 0.08)"
                border = "rgba(248, 113, 113, 0.2)"
                color = "#f87171"
            elif risk_key in ("中", "medium"):
                bg = "rgba(251, 191, 36, 0.08)"
                border = "rgba(251, 191, 36, 0.2)"
                color = "#fbbf24"
            else:
                bg = "rgba(52, 211, 153, 0.08)"
                border = "rgba(52, 211, 153, 0.2)"
                color = "#34d399"
            st.markdown(f"""
            <div style="padding: 0.7rem 1rem; margin-bottom: 0.5rem; background: {bg}; border-radius: 10px; border: 1px solid {border};">
            <b style="color: {color};">{t("result.pharma.admet_risk", level=risk_level)}</b>
            <span style="color: #c0c0d0; font-size: 0.85rem; margin-left: 0.5rem;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)
        st.caption(t("result.pharma.admet.toxicity_caption"))


# ════════════════════════════════════════════════════════════════════════════════
# GNN Explanation (called from _tab_solubility when model is GNN-only)
# ════════════════════════════════════════════════════════════════════════════════

def _display_gnn_explanation():
    """Display GNNExplainer bond importance + feature importance in the Solubility tab."""

    st.markdown(f"""<div class="card-title">{t('result.gnn.title')}</div>""", unsafe_allow_html=True)
    st.caption(t("result.gnn.desc"))

    smiles = st.session_state.get(StateKey.PREDICTED_SMILES)
    if not smiles:
        st.info(t("result.gnn.no_mol"))
        return

    # Try to load cached explanation
    explanation = None
    placeholder = st.empty()
    with placeholder:
        with st.spinner(t("result.gnn.spinner")):
            try:
                explanation = cached_gnn_explanation(smiles)
            except Exception as e:
                st.error(t("result.gnn.fail", err=e))
                return

    if explanation is None:
        st.warning(t("result.gnn.model_unavailable"))
        return

    bond_imp = explanation.get("bond_importance", [])
    feat_imp = explanation.get("feature_importance", [])
    elapsed = explanation.get("elapsed", 0.0)
    mol = explanation.get("mol")

    if not bond_imp:
        st.info(t("result.gnn.no_bonds"))
        return

    # ── Bond importance to RDKit bond weights ──
    from gnn_explainer import GNNExplainer as GE
    bond_weights = GE.bond_importance_to_smarts_weights(mol, bond_imp, threshold=0.05)
    top_bonds = GE.get_top_bonds(bond_imp, top_k=5)
    atom_imp = GE.get_atom_importance_from_bonds(mol, bond_imp, threshold=0.05)

    # ── Layout: two columns ──
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        # Draw highlighted molecule
        try:
            img = mol_to_dark_image_with_importance(mol, bond_weights, size=(500, 400))
            if img is not None:
                st.image(img, caption=t("result.gnn.img_caption"), use_container_width=True)
            else:
                # fallback
                from ui.plots import mol_to_dark_image
                st.image(mol_to_dark_image(mol, (500, 400)), use_container_width=True)
        except Exception as e:
            st.warning(f"Highlight image failed: {e}")
            # fallback to standard image
            from ui.plots import mol_to_dark_image
            try:
                st.image(mol_to_dark_image(mol, (500, 400)), use_container_width=True)
            except Exception:
                pass

    with col_right:
        st.markdown(f"""
        <div style="font-size:0.82rem;color:var(--ob-text-tertiary);margin-bottom:0.5rem;">
            {t('result.gnn.elapsed_text', t=elapsed)}
        </div>
        """, unsafe_allow_html=True)

        # Top-K most important bonds
        st.markdown(t("result.gnn.top_bonds_title"))
        bond_descriptions = []
        for rank, (a, b, imp) in enumerate(top_bonds, 1):
            if mol:
                atom_a = mol.GetAtomWithIdx(a)
                atom_b = mol.GetAtomWithIdx(b)
                symbol_a = atom_a.GetSymbol()
                symbol_b = atom_b.GetSymbol()
                bond = mol.GetBondBetweenAtoms(a, b)
                bond_type = str(bond.GetBondType()) if bond else "?"
                label = f"{symbol_a}{a}—{symbol_b}{b} ({bond_type})"
            else:
                label = f"Atom {a} — Atom {b}"
            pct = imp * 100
            bar_color = f"rgba({int(140+115*imp)},{int(60+175*imp)},{int(230-200*imp)},0.6)"
            st.markdown(f"""
            <div style="margin-bottom:0.4rem;font-size:0.85rem;">
                <span style="color:#c0c0d0;">#{rank}</span> <b>{label}</b><br>
                <div style="height:14px;background:#1e1e30;border-radius:7px;overflow:hidden;">
                    <div style="height:100%;width:{pct:.0f}%;background:{bar_color};border-radius:7px;"></div>
                </div>
                <span style="font-size:0.75rem;color:#8b8b9b;">{t('result.gnn.top_bonds_importance', imp=imp, pct=pct)}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Atom importance (second row) ──
    if atom_imp:
        st.markdown("<br>", unsafe_allow_html=True)
        col_atom_left, col_atom_right = st.columns([1, 1])
        with col_atom_left:
            st.markdown(t("result.gnn.atom_title"))
            sorted_atoms = sorted(atom_imp.items(), key=lambda x: x[1], reverse=True)[:8]
            for idx, imp_val in sorted_atoms:
                if mol:
                    atom = mol.GetAtomWithIdx(idx)
                    symbol = atom.GetSymbol()
                    label = f"{symbol}{idx}"
                else:
                    label = f"Atom {idx}"
                pct = imp_val * 100
                st.markdown(f"""
                <div style="margin-bottom:0.2rem;font-size:0.82rem;">
                    <span style="color:#c0c0d0;">{label}</span>
                    <div style="height:10px;background:#1e1e30;border-radius:5px;overflow:hidden;width:80%;display:inline-block;margin-left:0.5rem;">
                        <div style="height:100%;width:{pct:.0f}%;background:#a78bfa;border-radius:5px;"></div>
                    </div>
                    <span style="font-size:0.7rem;color:#8b8b9b;margin-left:0.3rem;">{imp_val:.2f}</span>
                </div>
                """, unsafe_allow_html=True)

        with col_atom_right:
            # Feature importance
            if feat_imp and len(feat_imp) > 0:
                st.markdown(t("result.gnn.feature_title"))
                # Feature names for the 37-dim atom feature vector
                feat_names = [
                    "AtomicNum_H", "AtomicNum_C", "AtomicNum_N", "AtomicNum_O",
                    "AtomicNum_F", "AtomicNum_P", "AtomicNum_S", "AtomicNum_Cl",
                    "AtomicNum_Br", "AtomicNum_I", "AtomicNum_Other",
                    "Degree_0", "Degree_1", "Degree_2", "Degree_3", "Degree_4", "Degree_5+",
                    "Charge_n2", "Charge_n1", "Charge_0", "Charge_p1", "Charge_p2",
                    "Hyb_SP", "Hyb_SP2", "Hyb_SP3", "Hyb_SP3D", "Hyb_Other",
                    "IsAromatic",
                    "Hs_0", "Hs_1", "Hs_2", "Hs_3", "Hs_4+",
                    "Chiral_None", "Chiral_R", "Chiral_S",
                    "IsInRing",
                ]
                # Sort by importance
                feat_pairs = list(enumerate(feat_imp))
                feat_pairs.sort(key=lambda x: x[1], reverse=True)
                top_feats = feat_pairs[:6]
                for dim_idx, imp_val in top_feats:
                    name = feat_names[dim_idx] if dim_idx < len(feat_names) else f"Dim_{dim_idx}"
                    pct = imp_val * 100
                    st.markdown(f"""
                    <div style="margin-bottom:0.2rem;font-size:0.78rem;">
                        <span style="color:#c0c0d0;">{name}</span>
                        <div style="height:8px;background:#1e1e30;border-radius:4px;overflow:hidden;width:70%;display:inline-block;margin-left:0.3rem;">
                            <div style="height:100%;width:{pct:.0f}%;background:#22d3ee;border-radius:4px;"></div>
                        </div>
                        <span style="font-size:0.65rem;color:#8b8b9b;">{imp_val:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Interpretation ──
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander(t("result.gnn.how_to_read_title"), expanded=False):
        st.markdown(f"""
        <div style="font-size:0.85rem;line-height:1.8;color:#a0a0b5;">
        {t('result.gnn.how_to_read_html')}
        <br><br>
        {t('result.gnn.how_to_read_tech')}
        <br><br>
        </div>
        """, unsafe_allow_html=True)


def _tab_ai(features, prediction, pka_val, pka_type):
    """Tab 4: AI chemistry explanation."""
    st.markdown(f"""<div class="card-title">{t('result.ai.title')}</div>""", unsafe_allow_html=True)
    with st.container(border=True):
        if st.session_state[StateKey.AI_EXPLANATION]:
            st.markdown(st.session_state[StateKey.AI_EXPLANATION])
            if st.button(t("result.ai.clear_btn"), key="clear_ai"):
                st.session_state[StateKey.AI_EXPLANATION] = None
                st.session_state[StateKey.TARGET_TAB] = t("result.tab.ai")
                st.rerun()
        else:
            st.caption(t("result.ai.need_manual"))
            if st.button(t("result.ai.generate_btn"), key="gen_ai", use_container_width=True):
                with st.spinner(t("result.ai.generating")):
                    pka_val_gen = st.session_state.get(StateKey.PREDICTED_PKA)
                    pka_type_gen = None
                    if pka_val_gen is not None:
                        pka_type_gen, _, _, _, _ = get_pka_type(pka_val_gen)
                    explanation = explain_with_kimi(
                        st.session_state[StateKey.PREDICTED_SMILES],
                        prediction,
                        features,
                        shap_features=st.session_state.get("shap_names"),
                        shap_values=st.session_state.get("shap_values"),
                        pka_value=pka_val_gen,
                        pka_type=pka_type_gen
                    )
                st.session_state[StateKey.AI_EXPLANATION] = explanation
                st.session_state[StateKey.TARGET_TAB] = t("result.tab.ai")
                st.rerun()
