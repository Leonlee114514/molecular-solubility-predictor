"""
DisSolve - Prediction results tab display (5 tabs).
"""

import numpy as np
import streamlit as st
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ui.plots import mol_to_dark_image
from model import get_pka_type, get_solubility_level, get_shap_explainer
from core.ai_client import explain_with_kimi
from core.cache import (
    cached_compute_features, cached_show_3d, cached_pka_analysis,
    cached_shap_contributions, cached_lipinski, cached_admet, cached_druglikeness,
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
                st.error("显示时解析失败，请重新输入 SMILES")
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
            st.error(f"**Out-of-Distribution 警告 — 预测可能不可靠**")
            with st.container(border=True):
                st.markdown("""
                <div style="font-size:0.9rem;line-height:1.7;color:#fca5a5;">
                <strong>该分子严重偏离训练数据分布</strong>，预测值仅供参考，不应作为实验依据。
                </div>
                """, unsafe_allow_html=True)
                if ood_result and ood_result.warnings:
                    for w in ood_result.warnings:
                        st.markdown(f"- {w}")
        elif ood_risk == "MEDIUM":
            st.warning(f"**Out-of-Distribution 注意 — 预测可能有一定误差**")
            with st.container(border=True):
                st.markdown("""
                <div style="font-size:0.9rem;line-height:1.7;color:#fcd34d;">
                <strong>该分子部分偏离训练数据分布</strong>，建议谨慎解读预测结果。
                </div>
                """, unsafe_allow_html=True)
                if ood_result and ood_result.warnings:
                    for w in ood_result.warnings:
                        st.markdown(f"- {w}")
        elif ood_risk == "LOW":
            # Show a subtle green badge for in-distribution molecules
            st.success(f"In-Distribution — 该分子在训练数据化学空间内，预测较为可靠")

        # ═════════════════════════════════════════
        # TAB 分组
        # ═════════════════════════════════════════
        tab_labels = [
            "Preview",
            "Solubility",
            "pKa",
            "Pharmacology",
            "AI Explanation"
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
    st.markdown("""<div class="card-title">Molecule Preview</div>""", unsafe_allow_html=True)

    mol = Chem.MolFromSmiles(st.session_state[StateKey.PREDICTED_SMILES])

    col_pv_left, col_pv_right = st.columns([1, 1])

    with col_pv_left:
        try:
            img = mol_to_dark_image(mol, size=(460, 380))
            if img is not None:
                st.image(img, caption="2D Molecular Structure", use_container_width=True)
            else:
                st.info("无法显示2D结构图")
        except Exception as e:
            st.warning(f"结构图生成失败: {e}")

    with col_pv_right:
        if mol is not None:
            formula = rdMolDescriptors.CalcMolFormula(mol)
            mw = features["MolWt"]
            st.metric("Molecular Formula", formula)
            st.metric("Molecular Weight", f"{mw:.1f} Da")
            st.markdown("""<div style="font-size:0.82rem;color:var(--ob-text-tertiary);margin-top:0.5rem;margin-bottom:0.2rem;">SMILES</div>""", unsafe_allow_html=True)
            st.code(st.session_state[StateKey.PREDICTED_SMILES], language=None)
        else:
            st.warning("无法解析分子结构")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="card-title">3D Ball-and-Stick Model</div>""", unsafe_allow_html=True)
    html_3d = cached_show_3d(st.session_state[StateKey.PREDICTED_SMILES])
    if html_3d:
        render_html(html_3d, height=420)
    else:
        st.info("3D 模型生成失败（需安装 py3Dmol）")


def _tab_solubility(features, prediction, interp, color, css_class, model):
    """Tab 1: Solubility prediction details + SHAP explainability."""
    st.markdown("""<div class="card-title">Solubility Prediction</div>""", unsafe_allow_html=True)

    model_type = st.session_state.get(StateKey.SELECTED_MODEL, "Auto")
    # Resolve Auto → actual model used
    if model_type == "Auto":
        gnn_val = st.session_state.get(StateKey.PREDICTED_LOGS_GNN)
        model_type = "GNN" if gnn_val is not None else "RF"
    model_colors = {"RF": "#34d399", "GNN": "#a78bfa", "Ensemble": "#fbbf24"}
    model_badge = {
        "RF": "Random Forest", "GNN": "Graph Neural Network", "Ensemble": "Ensemble (RF+GNN)"
    }.get(model_type, model_type)
    st.markdown(f"""
    <div style="display:inline-block;padding:0.15rem 0.7rem;background:rgba({model_colors[model_type].lstrip('#')},0.15);border:1px solid {model_colors[model_type]};border-radius:20px;font-size:0.78rem;color:{model_colors[model_type]};margin-bottom:0.6rem;">
        Model: {model_badge}
    </div>
    """, unsafe_allow_html=True)

    col_sol1, col_sol2 = st.columns([1, 1.2])
    with col_sol1:
        st.metric(label="Predicted Solubility (logS)", value=f"{prediction:.3f}")
        st.markdown(f"""
        <div class="{css_class}">
            <div style="font-size: 1.1rem; font-weight: 700; color: {color};">-> {interp}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Ensemble component display ──
        if model_type == "Ensemble":
            rf_val = st.session_state.get(StateKey.PREDICTED_LOGS_RF)
            gnn_val = st.session_state.get(StateKey.PREDICTED_LOGS_GNN)
            if rf_val is not None and gnn_val is not None:
                diff = abs(rf_val - gnn_val)
                st.markdown(f"""
                <div style="margin-top:0.8rem;padding:0.7rem 0.9rem;background:rgba(251,191,36,0.08);border-radius:10px;border:1px solid rgba(251,191,36,0.2);font-size:0.82rem;">
                    <b style="color:#fbbf24;">Ensemble Components:</b><br>
                    <span style="color:#34d399;">RF:</span> {rf_val:.3f} &nbsp;|&nbsp;
                    <span style="color:#a78bfa;">GNN:</span> {gnn_val:.3f}<br>
                    <span style="color:#fbbf24;">Disagreement |RF−GNN|:</span> {diff:.3f}
                    <span style="color:#8b8b9b;font-size:0.75rem;">{' (good agreement)' if diff < 0.5 else ' (notable disagreement)' if diff < 1.0 else ' (large divergence — treat prediction with caution)'}</span>
                </div>
                """, unsafe_allow_html=True)

    with col_sol2:
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.03); border-radius: 14px; padding: 1rem; font-size: 0.85rem; color: var(--ob-text-tertiary); border: 1px solid var(--ob-border); font-family: 'Cascadia Code', 'Consolas', monospace;">
        <b style="color: var(--ob-text-secondary);">Interpretation guide:</b><br>
        <span style="color: #34d399;">&gt;</span> logS > 0: Very soluble (like ethanol)<br>
        <span style="color: #fbbf24;">&gt;</span> -2 < logS < 0: Moderately soluble<br>
        <span style="color: #f87171;">&gt;</span> logS < -2: Poorly soluble (like many drug molecules)
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="card-title">Molecular Descriptors</div>""", unsafe_allow_html=True)
    with st.container(border=True):
        desc_col1, desc_col2, desc_col3, desc_col4 = st.columns(4)
        with desc_col1:
            st.metric("Molecular Weight", f"{features['MolWt']:.1f}")
            st.metric("LogP (Hydrophobicity)", f"{features['LogP']:.2f}")
        with desc_col2:
            st.metric("H-Bond Donors", f"{features['NumHDonors']}")
            st.metric("H-Bond Acceptors", f"{features['NumHAcceptors']}")
        with desc_col3:
            st.metric("TPSA (Å²)", f"{features['TPSA']:.1f}")
            st.metric("Rotatable Bonds", f"{features['NumRotatableBonds']}")
        with desc_col4:
            st.metric("Aromatic Rings", f"{features['NumAromaticRings']}")
            st.metric("Aliphatic Rings", f"{features['NumAliphaticRings']}")
    st.info("""
    **Chemistry Insight:**
    - **TPSA** (Topological Polar Surface Area) measures how much of the molecule is polar.
       Higher TPSA usually means better water solubility.
    - **H-Bond Donors/Acceptors** tell us how well the molecule can form hydrogen bonds with water.
    - **LogP** measures lipophilicity. Lower LogP means the molecule prefers water over oil.
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="card-title">SHAP Explainability</div>""", unsafe_allow_html=True)

    # Re-resolve Auto for this section (model_type may have been overridden above)
    raw_model_type = st.session_state.get(StateKey.SELECTED_MODEL, "Auto")
    if raw_model_type == "Auto":
        gnn_check = st.session_state.get(StateKey.PREDICTED_LOGS_GNN)
        shap_model_type = "GNN" if gnn_check is not None else "RF"
    else:
        shap_model_type = raw_model_type

    if shap_model_type == "GNN":
        st.info(
            "SHAP 可解释性分析基于 Random Forest 特征重要性，"
            "对 GNN-only 模式不可用。GNN 模型使用图结构学习，"
            "其可解释性可通过原子/边注意力权重进行分析（GNNExplainer，未来版本支持）。"
            "如需 SHAP 分析，请切换到 RF 或 Ensemble 模式。"
        )
    elif model_type == "Ensemble":
        st.caption(
            "基于 SHAP (SHapley Additive exPlanations) 分析 RF 分量中每个特征对预测的贡献。"
            "GNN 的结构贡献体现在 Ensemble 的差值中。"
        )
    else:
        st.caption("基于 SHAP (SHapley Additive exPlanations) 分析每个特征对预测的贡献")

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
        ax.set_xlabel("对溶解度的贡献值 (logS)", fontsize=11)
        ev = get_shap_explainer(model).expected_value
        if isinstance(ev, (list, tuple, np.ndarray)):
            base_value = float(np.array(ev).flatten()[0])
        else:
            base_value = float(ev)
        ax.set_title(f"预测值: {prediction:.3f}  (基准值: {base_value:.3f})", fontsize=12, pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        legend_elements = [
            Patch(facecolor="#a78bfa", label="推动易溶 (正贡献)"),
            Patch(facecolor="#06b6d4", label="推动难溶 (负贡献)")
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        if prediction > 0:
            solubility_level = "易溶于水"
        elif prediction > -2:
            solubility_level = "中等溶解"
        else:
            solubility_level = "难溶于水"
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
                direction = "推动易溶" if val > 0 else "推动难溶"
                supporting.append("**" + name + "**（" + f"{val:+.3f}" + "，" + direction + "）")
        parts = ["**关键分析**：模型预测该分子 **" + solubility_level + "**（logS = " + f"{prediction:.3f}" + "）。"]
        if supporting:
            parts.append("推动这一结果的主要因素：" + ", ".join(supporting) + "。")
        if resisting:
            target = "更易溶" if prediction <= -2 else "更难溶"
            parts.append("但以下因素在抵抗这一趋势、试图让分子" + target + "：" + ", ".join(resisting) + "。")
        shift = abs(prediction - base_value)
        direction = "向上" if prediction > base_value else "向下"
        parts.append("相比训练集平均分子（基准值 " + f"{base_value:.3f}" + "），该分子的结构特征将预测值" + direction + "拉动了 " + f"{shift:.3f}" + " 个单位。")
        insight_text = " ".join(parts)
        st.info(insight_text)
    else:
        st.info("SHAP 可解释性分析暂不可用，但预测结果仍然有效。")


def _tab_pka(pka_val, pka_type, pka_label, pka_css, pka_text_color, pka_desc, features):
    """Tab 2: pKa acidity/basicity prediction."""
    if pka_val is not None:
        st.markdown("""<div class="card-title">pKa Prediction</div>""", unsafe_allow_html=True)
        col_pka1, col_pka2 = st.columns(2)
        with col_pka1:
            st.metric("Predicted pKa", f"{pka_val:.2f}")
        with col_pka2:
            st.markdown(f"""
            <div class="{pka_css}" style="margin-top: 0.2rem;">
                <div style="font-size: 1rem; font-weight: 700; color: {pka_text_color};">-> {pka_label}</div>
                <div style="font-size: 0.8rem; color: var(--ob-text-tertiary); margin-top: 0.3rem;">{pka_desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div class="card-title">Chemical Factor Decomposition</div>""", unsafe_allow_html=True)
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
            unit = "增强酸性" if pka_val < 7 else "增强碱性"
            ax.set_xlabel(f"对 {unit} 的贡献", fontsize=11)
            ax.set_title(f"pKa = {pka_val:.2f} | 化学因素分解", fontsize=12, pad=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('#33334d')
            legend_elements = [
                Patch(facecolor='#a78bfa', label=f'增强{"酸性" if pka_val < 7 else "碱性"}'),
                Patch(facecolor='#22d3ee', label=f'减弱{"酸性" if pka_val < 7 else "碱性"}')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
                      framealpha=0.8, facecolor='#1a1a2e', edgecolor=(1, 1, 1, 0.1))
            plt.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            st.caption("""
            **如何读懂这张图**：
            紫色条越长 = 该因素越推动分子**释放/结合质子**；
            青色条越长 = 该因素越**抵抗**质子转移。
            和 SHAP 不同，这些不是机器学习权重，而是**真实的结构化学效应**。
            """)
            st.markdown("""
            <div style="margin-top: 0.3rem; padding: 0.65rem 0.9rem; background: rgba(124, 58, 237, 0.06); border-left: 2px solid rgba(124, 58, 237, 0.3); border-radius: 4px; font-size: 0.82rem; color: #a0a0b5; line-height: 1.9;">
            <b style="color: #c4b5fd;">图表术语速查</b> &nbsp;点击术语查看中英双语定义：<br>
            &bull; <b>诱导效应</b>（Inductive Effect）&mdash; 电负性原子通过 σ 键吸引或排斥电子，从而影响质子的结合与释放<br>
            &bull; <b>共轭效应</b>（Resonance / Conjugation）&mdash; π 电子在共轭体系中离域分布，稳定电离后的离子形式<br>
            &bull; <b>分子内氢键</b>（Intramolecular H-Bond）&mdash; 同一分子内不同基团间形成氢键，屏蔽极性、调节 pKa<br>
            &bull; <b>空间位阻</b>（Steric Hindrance）&mdash; 大体积原子或基团阻碍质子的接近与离去，改变反应活性<br>
            &bull; <b>杂化/芳香性</b>（Hybridization / Aromaticity）&mdash; sp² 碳比例与芳香环共轭体系带来的额外稳定性
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("化学因素分析暂不可用")
    else:
        st.info("pKa 模型未加载，pKa 分析不可用。")


def _tab_pharmacology(features, prediction, pka_val, pka_type, pka_label, pka_css, pka_text_color, pka_desc):
    """Tab 3: Pharmacology (Lipinski, ADME/Tox, ionization profile)."""
    st.markdown("""<div class="card-title">Drug-likeness: Lipinski's Rule of Five</div>""", unsafe_allow_html=True)
    lipinski_result = cached_lipinski(tuple(features.items()))
    rules = lipinski_result["rules"]
    setup_plt_dark()

    fig, ax = plt.subplots(figsize=(10, 3.6))

    property_labels = [
        ("分子量\nMol. Weight", "≤ 500 Da"),
        ("脂水分配系数\nLogP", "≤ 5"),
        ("氢键供体数\nH-Bond Donors", "≤ 5"),
        ("氢键受体数\nH-Bond Acceptors", "≤ 10"),
        ("可旋转键数\nRotatable Bonds", "≤ 10"),
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
        Patch(facecolor='#34d399', alpha=0.82, label='PASS — 符合规则'),
        Patch(facecolor='#f87171', alpha=0.82, label='FAIL — 超出阈值'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', fontsize=9,
              ncol=2, framealpha=0.7, facecolor='#1a1a2e', edgecolor=(1, 1, 1, 0.08),
              bbox_to_anchor=(0.5, -0.35))

    ax.set_xlim(-0.42, 1.05)
    ax.set_ylim(-0.6, 4.6)
    ax.axis('off')

    title_color = '#34d399' if lipinski_result['passed'] >= 4 else '#fbbf24'
    ax.set_title(
        f'Lipinski 五规则评分：{lipinski_result["passed"]} / 5   —   {lipinski_result["interpretation"]}',
        fontsize=12, pad=16, color=title_color, fontweight='600')

    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.markdown(f"""
    <div style="margin-top: 0.3rem; padding: 0.65rem 0.9rem; background: rgba(124, 58, 237, 0.06); border-left: 2px solid rgba(124, 58, 237, 0.3); border-radius: 4px; font-size: 0.82rem; color: #a0a0b5; line-height: 1.9;">
    <b style="color: #c4b5fd;">About Lipinski's Rule of Five</b><br>
    &bull; <b>Christopher Lipinski (Pfizer, 1997)</b> 分析了 2,245 个进入 II 期临床的药物分子，总结出 5 条口服药物的经验规则<br>
    &bull; 规则认为分子违反 ≤1 条时，其<b>口服吸收和生物利用度</b>更可能达标<br>
    &bull; 但这只是筛选规则，<b>不是绝对标准</b>——许多成功药物也违反五规则（如天然产物、抗生素、抗癌药）<br>
    &bull; 超出规则范围 (bRo5) 的分子仍是现代药物化学的重要方向（如 PROTAC、大环分子）
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="card-title">Drug-likeness Metrics: QED &middot; SAscore &middot; Fsp&sup3;</div>""", unsafe_allow_html=True)
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
        碳原子总数 Total carbons: <b style="color:#a0a0b0;">{dl['n_carbons']}</b> &nbsp;|&nbsp;
        sp&sup3; 碳 sp&sup3; carbons: <b style="color:#a0a0b0;">{dl['n_sp3']}</b> &nbsp;|&nbsp;
        Refs: Bickerton et al. (2012), Ertl &amp; Schuffenhauer (2009), Lovering et al. (2009)
        </div>
        """, unsafe_allow_html=True)

    if pka_val is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div class="card-title">Ionization Profile</div>""", unsafe_allow_html=True)
        env_ph = [1.5, 4.5, 6.8, 7.4]
        env_names = ['Stomach\n胃', 'Duodenum\n十二指肠', 'Small Intestine\n小肠', 'Blood/Brain\n血液/脑']
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
        ax.set_ylabel('分子态比例 (Unionized %)', fontsize=11)
        ax.set_ylim(0, 105)
        ax.set_title(f'不同生理环境下的分子态比例 | pKa = {pka_val:.2f}', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div class="card-title">药理学分析</div>""", unsafe_allow_html=True)
        with st.container(border=True):
            if pka_type == "acid":
                if pka_val < 4:
                    st.success("**胃吸收优势**：pKa < 4，在胃酸（pH 1.5）中大部分以分子态存在，脂溶性高，容易被胃黏膜吸收。代表药物：阿司匹林 (pKa 3.5)、布洛芬 (pKa 4.9)。")
                else:
                    st.info("**全肠道吸收**：pKa 中等，在胃和小肠中都有一定比例的分子态，吸收较均匀。注意：分子态比例高时脂溶性强，可能刺激胃黏膜。")
            elif pka_type == "base":
                if pka_val > 9:
                    st.warning("**肠道吸收为主**：强碱性分子在胃中几乎完全电离，难以吸收；进入小肠（pH 6.8）后分子态增加，主要在小肠吸收。代表药物：二甲双胍 (pKa ~12.4)。")
                else:
                    st.info("**弱碱性分子**：在胃中少量电离，小肠中吸收良好。进入血液（pH 7.4）后可能部分电离，水溶性增加，有利于肾脏排泄。")
            else:
                st.info("**两性分子**：在不同 pH 环境下电离行为复杂，吸收部位取决于具体结构。可能需要特殊制剂（如肠溶片）来优化生物利用度。")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div class="card-title">溶解度 x pKa 联动分析</div>""", unsafe_allow_html=True)
        logS = prediction
        parts = []
        if logS > 0:
            parts.append("**溶解度**：易溶于水，有利于溶出。")
        elif logS > -2:
            parts.append("**溶解度**：中等，可能需要辅料助溶。")
        else:
            parts.append("**溶解度**：较低，生物利用度可能受限。")
        if pka_type == "acid":
            if pka_val < 4:
                parts.append(f"**pKa**：弱酸性 (pKa={pka_val:.1f})，胃吸收好，**空腹服用**效果更佳。")
            else:
                parts.append(f"**pKa**：中等酸性 (pka={pka_val:.1f})，全肠道吸收，对服药时间要求不高。")
        elif pka_type == "base":
            if pka_val > 9:
                parts.append(f"**pKa**：强碱性 (pKa={pka_val:.1f})，胃吸收差，**餐后服用**可减少胃刺激，主要在小肠吸收。")
            else:
                parts.append(f"**pKa**：弱碱性 (pKa={pka_val:.1f})，小肠吸收为主，血液中有利于排泄。")
        else:
            parts.append(f"**pKa**：接近中性 (pKa={pka_val:.1f})，吸收行为较复杂。")
        if logS > 0 and pka_type == "acid" and pka_val < 4:
            parts.append("**综合**：高溶解度 + 胃吸收优势 = **口服生物利用度极佳**，适合做成普通片剂。")
        elif logS < -2 and pka_type == "base" and pka_val > 9:
            parts.append("**综合**：低溶解度 + 强碱性 = **口服吸收双重挑战**，可能需要肠溶片或注射剂型。")
        elif logS > 0 and pka_type == "base" and pka_val > 9:
            parts.append("**综合**：高溶解度弥补了胃吸收劣势，进入小肠后吸收良好，总体生物利用度可接受。")
        st.info(" | ".join(parts))
    else:
        st.info("pKa 模型未加载，药理学分析不可用。")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="card-title">ADME/Tox 药代动力学概览</div>""", unsafe_allow_html=True)

    admet = cached_admet(
        st.session_state[StateKey.PREDICTED_SMILES],
        tuple(features.items()),
        pka_val
    )

    adme_tabs = st.tabs([
        "Absorption 吸收",
        "Distribution 分布",
        "Metabolism 代谢",
        "Excretion 排泄",
        "Toxicity 毒性",
    ])

    with adme_tabs[0]:
        st.markdown(f"""
        <div style="padding: 1rem; background: rgba(52, 211, 153, 0.06); border-radius: 12px; border: 1px solid rgba(52, 211, 153, 0.15);">
        <b style="color: #34d399;">吸收分析</b><br><br>
        <span style="color: #c0c0d0; font-size: 0.9rem; line-height: 1.7;">{admet['absorption']}</span>
        </div>
        """, unsafe_allow_html=True)
        st.caption("""
        **吸收 (Absorption)** 决定了药物从给药部位进入血液循环的效率和程度。
        主要影响因素包括：分子极性 (TPSA)、脂溶性 (LogP)、电离状态 (pKa vs pH)、氢键能力、分子大小。
        口服药物的吸收主要发生在小肠（表面积大、血流丰富），部分酸性药物可在胃中开始吸收。
        """)

    with adme_tabs[1]:
        d = admet["distribution"]
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown(f"""
            <div style="padding: 0.8rem 1rem; background: rgba(96, 165, 250, 0.08); border-radius: 10px; border: 1px solid rgba(96, 165, 250, 0.15);">
            <b style="color: #60a5fa;">表观分布容积 (Vd)</b><br>
            <span style="color: #c0c0d0; font-size: 0.85rem;">{d['vd_estimate']}</span>
            </div>
            """, unsafe_allow_html=True)
        with col_d2:
            st.markdown(f"""
            <div style="padding: 0.8rem 1rem; background: rgba(96, 165, 250, 0.08); border-radius: 10px; border: 1px solid rgba(96, 165, 250, 0.15);">
            <b style="color: #60a5fa;">血浆蛋白结合率</b><br>
            <span style="color: #c0c0d0; font-size: 0.85rem;">{d['ppb']}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="margin-top: 0.6rem; padding: 1rem; background: rgba(96, 165, 250, 0.06); border-radius: 12px; border: 1px solid rgba(96, 165, 250, 0.15);">
        <span style="color: #c0c0d0; font-size: 0.9rem; line-height: 1.7;">{d['summary']}</span>
        </div>
        """, unsafe_allow_html=True)
        st.caption("""
        **分布 (Distribution)** 描述药物从血液进入组织和器官的过程。
        - **表观分布容积 (Vd)**：Vd 越大，药物越倾向于分布到组织；Vd 小则主要留在血浆中
        - **血浆蛋白结合率**：高结合率 = 药物储备在血液中缓慢释放；低结合率 = 游离药物多、作用快但排泄也快
        - **血脑屏障 (BBB)**：低 TPSA + 适中 LogP 的分子更容易进入中枢神经系统
        """)

    with adme_tabs[2]:
        m = admet["metabolism"]
        col_m1, col_m2 = st.columns([1.5, 1])
        with col_m1:
            st.markdown(f"""
            <div style="padding: 1rem; background: rgba(251, 191, 36, 0.06); border-radius: 12px; border: 1px solid rgba(251, 191, 36, 0.15);">
            <b style="color: #fbbf24;">代谢热点</b><br><br>
            <span style="color: #c0c0d0; font-size: 0.9rem; line-height: 1.7;">{m['summary']}</span>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div style="padding: 1rem; background: rgba(251, 191, 36, 0.06); border-radius: 12px; border: 1px solid rgba(251, 191, 36, 0.15);">
            <b style="color: #fbbf24;">相关代谢酶</b><br><br>
            <span style="color: #c0c0d0; font-size: 0.9rem;">{m['cyp_enzymes']}</span>
            </div>
            """, unsafe_allow_html=True)
        st.caption("""
        **代谢 (Metabolism)** 主要在肝脏进行，分为两相：
        - **I 相代谢**（氧化/还原/水解）：主要由 CYP450 酶系催化，引入或暴露极性基团
        - **II 相代谢**（结合反应）：将葡萄糖醛酸、硫酸、氨基酸等连接到分子上，增加水溶性便于排泄
        - CYP3A4 代谢约 50% 的临床药物；CYP2D6 有显著的基因多态性（人群差异大）
        """)

    with adme_tabs[3]:
        e = admet["excretion"]
        st.markdown(f"""
        <div style="padding: 1rem; background: rgba(167, 139, 250, 0.06); border-radius: 12px; border: 1px solid rgba(167, 139, 250, 0.15);">
        <b style="color: #a78bfa;">排泄途径：{e['route']}</b><br><br>
        <span style="color: #c0c0d0; font-size: 0.9rem; line-height: 1.7;">{e['summary']}</span>
        </div>
        """, unsafe_allow_html=True)
        st.caption("""
        **排泄 (Excretion)** 是药物及其代谢物从体内清除的过程：
        - **肾脏排泄**：分子量 < 350 Da 且极性适中的分子主要通过肾小球滤过进入尿液
        - **肝胆排泄**：分子量 > 500 Da 或高度亲脂的分子倾向于通过胆汁排入肠道
        - 肾小管重吸收会使亲脂性分子重新进入血液，延长药物半衰期
        """)

    with adme_tabs[4]:
        alerts = admet["toxicity"]
        for risk_level, desc in alerts:
            if risk_level == "高":
                bg = "rgba(248, 113, 113, 0.08)"
                border = "rgba(248, 113, 113, 0.2)"
                color = "#f87171"
            elif risk_level == "中":
                bg = "rgba(251, 191, 36, 0.08)"
                border = "rgba(251, 191, 36, 0.2)"
                color = "#fbbf24"
            else:
                bg = "rgba(52, 211, 153, 0.08)"
                border = "rgba(52, 211, 153, 0.2)"
                color = "#34d399"
            st.markdown(f"""
            <div style="padding: 0.7rem 1rem; margin-bottom: 0.5rem; background: {bg}; border-radius: 10px; border: 1px solid {border};">
            <b style="color: {color};">[风险：{risk_level}]</b>
            <span style="color: #c0c0d0; font-size: 0.85rem; margin-left: 0.5rem;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)
        st.caption("""
        **毒性 (Toxicity)** 评估基于结构警报 (Structural Alerts) —— 某些官能团或子结构在历史上与毒性事件相关：
        - 以上分析仅基于<b>结构特征</b>，不代表实际毒性——毒性受剂量、代谢、个体差异等多因素影响
        - 结构警报是药物设计初筛的重要工具，但存在许多假阳性——含警报结构的药物仍可能安全上市
        - 实际毒性需要通过 Ames 试验、hERG 测试、动物实验和临床试验逐级验证
        """)


def _tab_ai(features, prediction, pka_val, pka_type):
    """Tab 4: AI chemistry explanation."""
    st.markdown("""<div class="card-title">AI Chemistry Explanation</div>""", unsafe_allow_html=True)
    with st.container(border=True):
        if st.session_state[StateKey.AI_EXPLANATION]:
            st.markdown(st.session_state[StateKey.AI_EXPLANATION])
            if st.button("清除解释", key="clear_ai"):
                st.session_state[StateKey.AI_EXPLANATION] = None
                st.session_state[StateKey.TARGET_TAB] = "AI Explanation"
                st.rerun()
        else:
            st.caption("AI 解释需要手动调用（消耗 API 额度）")
            if st.button("生成 AI 解释", key="gen_ai", use_container_width=True):
                with st.spinner("正在分析分子结构..."):
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
                st.session_state[StateKey.TARGET_TAB] = "AI Explanation"
                st.rerun()
