"""
DisSolve - Reusable UI components (header, footer, input area, CJK font).
"""

import base64
import streamlit as st
from molecules import MOLECULE_DB, SEARCH_INDEX, search_pubchem as search_pubchem_final


def render_html(html_content, height=1):
    """Render HTML/JS via st.iframe data URL."""
    encoded = base64.b64encode(html_content.encode("utf-8")).decode("utf-8")
    st.iframe(f"data:text/html;charset=utf-8;base64,{encoded}", height=height)


def get_cjk_font():
    """Detect and cache a CJK-capable matplotlib font. Returns font name or None."""
    import glob
    import matplotlib.font_manager as fm
    font_paths = (
        glob.glob('/usr/share/fonts/opentype/noto/*.ttc') +
        glob.glob('/usr/share/fonts/truetype/noto/*.ttc') +
        glob.glob('/usr/share/fonts/noto-cjk/*.ttc') +
        glob.glob('/usr/share/fonts/truetype/wqy/*.ttf') +
        glob.glob('/usr/share/fonts/opentype/source-han-sans/*.otf')
    )
    for fp in font_paths:
        try:
            fm.fontManager.addfont(fp)
        except Exception:
            pass
    for font in fm.fontManager.ttflist:
        if font.name in ('Noto Sans CJK SC', 'Noto Sans CJK'):
            return font.name
        if 'WenQuanYi' in font.name or 'Source Han Sans SC' in font.name:
            return font.name
    return None


def render_header():
    """Render the DisSolve page title and introduction."""
    st.markdown("""
    <div style="text-align:center; margin-top:1rem; margin-bottom:0.5rem;">
        <div class="tagline">AI-POWERED MOLECULAR PROPERTY PREDICTION</div>
        <h1 class="gradient-title">DisSolve</h1>
        <p class="subtitle">Predict Aqueous Solubility, pKa & Pharmacological Profiles from Molecular Structure</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card-container" style="padding: 1.2rem 1.5rem; margin-bottom: 2rem;">
        <p style="margin: 0; color: var(--ob-text-secondary); line-height: 1.7;">
            <b style="color: var(--ob-text-primary);">Welcome!</b> This app predicts aqueous solubility (logS), acid-base behavior (pKa),
            and pharmacological profiles from molecular structure using a <b>Machine Learning</b> model
            trained on <b>11,000+ organic compounds</b>.
            Explore 2D & 3D molecular structures, pKa chemistry insights, and AI-generated explanations.
        </p>
        <div style="display: flex; gap: 1rem; margin-top: 1rem; flex-wrap: wrap;">
            <span class="badge badge-primary"><span style="margin-right:4px;">&#128071;</span> 快速选择</span>
            <span class="badge badge-success"><span style="margin-right:4px;">&#128269;</span> 名称搜索</span>
            <span class="badge badge-warn"><span style="margin-right:4px;">&#9997;</span> SMILES 输入</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_footer():
    """Render the page footer."""
    st.markdown("""
    <div class="footer">
        <div style="font-weight: 600; color: var(--ob-text-secondary); margin-bottom: 0.3rem; font-family: 'Space Grotesk', sans-serif; font-size: 1rem;">DisSolve</div>
        <div>Built with Streamlit | ML: Random Forest + RDKit | Solubility: 11,000+ molecules | pKa: 410,000+ molecules | AI: Kimi (Moonshot AI) | DB: 100+ local + PubChem API</div>
        <div style="margin-top: 0.5rem; font-size: 0.75rem; color: #6b6b7b;">溶解度预测 · pKa分析 · 药理学评估 | AI-Powered Chemistry Platform</div>
    </div>
    """, unsafe_allow_html=True)


def render_input_area():
    """Render the three input methods: quick-select, name search, and SMILES input."""

    # --- 方式1：可搜索单选列表 ---
    with st.container(border=True):
        st.markdown("""<div class="card-title">&#128071; 方式 1：快速选择常见分子</div>""", unsafe_allow_html=True)

        mol_search = st.text_input(
            "搜索分子",
            placeholder="输入中文或英文名称过滤...",
            key="mol_search_filter",
            label_visibility="collapsed"
        ).strip().lower()

        all_mols = list(MOLECULE_DB.keys())
        if mol_search:
            filtered_mols = [m for m in all_mols if mol_search in m.lower()]
            if not filtered_mols:
                st.caption("未找到匹配分子，显示全部选项")
                filtered_mols = all_mols
        else:
            filtered_mols = all_mols

        current_idx = 0
        if "molecule_select_radio" in st.session_state and st.session_state.molecule_select_radio in filtered_mols:
            current_idx = filtered_mols.index(st.session_state.molecule_select_radio)
        elif st.session_state.get("molecule_select_radio") is None:
            current_idx = 0

        selected_molecule = st.radio(
            "选择分子",
            filtered_mols,
            index=current_idx,
            key="molecule_select_radio",
            label_visibility="collapsed"
        )

        if selected_molecule != "(自定义输入)":
            new_smiles = MOLECULE_DB[selected_molecule]
            if new_smiles != st.session_state.smiles_input_box:
                st.session_state.smiles_input_box = new_smiles
                st.session_state.predicted_smiles = None
                st.session_state.predicted_logS = None
                st.session_state.ai_explanation = None
                st.rerun()

    # --- 方式2：三层搜索 ---
    with st.container(border=True):
        st.markdown("""<div class="card-title">&#128269; 方式 2：名称搜索（本地库 + PubChem API）</div>""", unsafe_allow_html=True)
        st.caption("支持中英文，如 阿司匹林 / Aspirin / Ibuprofen / 咖啡因")
        search_col1, search_col2 = st.columns([4, 1])
        with search_col1:
            search_name = st.text_input(
                "输入名称",
                placeholder="例如 阿司匹林 或 Aspirin",
                key="search_name",
                label_visibility="collapsed"
            )
        with search_col2:
            search_clicked = st.button("&#128269; 搜索", key="search_btn", use_container_width=True)

        if "search_state" not in st.session_state:
            st.session_state.search_state = None
            st.session_state.search_query = ""
            st.session_state.fuzzy_matches = []
            st.session_state.fuzzy_best = ""
            st.session_state.fuzzy_smiles = ""

        if search_clicked and search_name:
            st.session_state.search_query = search_name.strip().lower()
            st.session_state.search_state = None
            st.session_state.fuzzy_matches = []
            st.rerun()

        if not search_name and st.session_state.search_query:
            st.session_state.search_query = ""
            st.session_state.search_state = None
            st.session_state.fuzzy_matches = []
            st.rerun()

        query = st.session_state.search_query
        if query:
            if query in SEARCH_INDEX:
                found_smiles = SEARCH_INDEX[query]
                st.success(f"本地精确匹配：`{search_name}` -> `{found_smiles}`")
                if found_smiles != st.session_state.smiles_input_box:
                    st.session_state.smiles_input_box = found_smiles
                    st.session_state.predicted_smiles = None
                    st.session_state.predicted_logS = None
                    st.session_state.ai_explanation = None
                st.info("点击下方的 **Predict** 按钮查看结果")
                st.session_state.search_state = "exact"
            else:
                matches = [k for k in SEARCH_INDEX.keys() if query in k or k in query]
                if matches:
                    matches.sort(key=lambda x: (0 if x.startswith(query) else 1, len(x)))
                    best_match = matches[0]
                    found_smiles = SEARCH_INDEX[best_match]

                    if st.session_state.search_state not in ("fuzzy_pending", "fuzzy_confirmed", "pubchem_pending", "pubchem_done", "no_match"):
                        st.session_state.search_state = "fuzzy_pending"
                        st.session_state.fuzzy_matches = matches
                        st.session_state.fuzzy_best = best_match
                        st.session_state.fuzzy_smiles = found_smiles

                    matches = st.session_state.fuzzy_matches
                    best_match = st.session_state.fuzzy_best
                    found_smiles = st.session_state.fuzzy_smiles

                    if st.session_state.search_state == "fuzzy_pending":
                        st.info(f"本地模糊匹配：`{search_name}` → `{best_match}`")
                        if len(matches) > 1:
                            with st.expander(f"查看全部 {len(matches)} 个模糊匹配结果", expanded=False):
                                for m in matches[:8]:
                                    st.caption(f"**{m}**")
                                    st.code(SEARCH_INDEX[m], language=None)

                        confirm_col1, confirm_col2 = st.columns(2)
                        with confirm_col1:
                            use_fuzzy = st.button("✅ 确认使用此结果", key="use_fuzzy_match", use_container_width=True)
                        with confirm_col2:
                            use_pubchem = st.button("🔍 不是我要的，搜 PubChem", key="skip_to_pubchem", use_container_width=True)

                        if use_fuzzy:
                            st.session_state.search_state = "fuzzy_confirmed"
                            if found_smiles != st.session_state.smiles_input_box:
                                st.session_state.smiles_input_box = found_smiles
                                st.session_state.predicted_smiles = None
                                st.session_state.predicted_logS = None
                                st.session_state.ai_explanation = None
                            st.rerun()
                        elif use_pubchem:
                            st.session_state.search_state = "pubchem_pending"
                            st.rerun()

                    if st.session_state.search_state == "fuzzy_confirmed":
                        st.success(f"已采用：`{best_match}` → `{found_smiles}`")
                        st.info("点击下方的 **Predict** 按钮查看结果")

                    if st.session_state.search_state == "pubchem_pending":
                        with st.status("正在查询 PubChem API...", expanded=True) as pub_status:
                            found_smiles, pub_status_str = search_pubchem_final(search_name)
                            if found_smiles:
                                pub_status.update(label=f"PubChem 匹配成功：{pub_status_str}", state="complete")
                            else:
                                pub_status.update(label=f"PubChem 未找到：{pub_status_str}", state="error")
                        if found_smiles:
                            st.session_state.search_state = "pubchem_done"
                            st.session_state.fuzzy_smiles = found_smiles
                            if found_smiles != st.session_state.smiles_input_box:
                                st.session_state.smiles_input_box = found_smiles
                                st.session_state.predicted_smiles = None
                                st.session_state.predicted_logS = None
                                st.session_state.ai_explanation = None
                        else:
                            st.session_state.search_state = "no_match"
                        st.rerun()

                    if st.session_state.search_state == "pubchem_done":
                        st.success(f"PubChem 匹配：`{search_name}` → `{st.session_state.fuzzy_smiles}`")
                        st.info("点击下方的 **Predict** 按钮查看结果")

                    if st.session_state.search_state == "no_match":
                        st.error(f"未找到：`{search_name}`")
                else:
                    if st.session_state.search_state not in ("pubchem_done", "no_match"):
                        st.session_state.search_state = "pubchem_pending"
                        with st.status("本地未找到，正在查询 PubChem API...", expanded=True) as pub_status:
                            found_smiles, pub_status_str = search_pubchem_final(search_name)
                            if found_smiles:
                                pub_status.update(label=f"PubChem 匹配成功：{pub_status_str}", state="complete")
                            else:
                                pub_status.update(label=f"PubChem 未找到：{pub_status_str}", state="error")
                        if found_smiles:
                            st.session_state.search_state = "pubchem_done"
                            st.session_state.fuzzy_smiles = found_smiles
                        else:
                            st.session_state.search_state = "no_match"

                    if st.session_state.search_state == "pubchem_done":
                        found_smiles = st.session_state.fuzzy_smiles
                        st.success(f"PubChem 匹配：`{search_name}` -> `{found_smiles}`")
                        if found_smiles != st.session_state.smiles_input_box:
                            st.session_state.smiles_input_box = found_smiles
                            st.session_state.predicted_smiles = None
                            st.session_state.predicted_logS = None
                            st.session_state.ai_explanation = None
                        st.info("点击下方的 **Predict** 按钮查看结果")
                    elif st.session_state.search_state == "no_match":
                        st.error(f"未找到：`{search_name}`")
                        st.info("尝试建议：")
                        st.markdown("""
                        - 检查拼写（如 **Aspirin** 而非 **Aspriin**）
                        - 尝试更常见的名称
                        - 直接输入 SMILES（方式3）
                        """)
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, rgba(124, 58, 237, 0.08), rgba(124, 58, 237, 0.02)); padding: 18px; border-radius: 16px; border-left: 3px solid #7c3aed;">
                        <h4 style="color: #a78bfa; margin-top: 0; font-family: 'Space Grotesk', sans-serif;">如何手动获取 SMILES？</h4>
                        <ol style="color: var(--ob-text-secondary); margin-bottom: 0;">
                            <li>访问 <a href="https://pubchem.ncbi.nlm.nih.gov" target="_blank" style="color: #a78bfa;"><b>https://pubchem.ncbi.nlm.nih.gov</b></a></li>
                            <li>在搜索框输入分子名称（英文，如 <b>Aspirin</b>）</li>
                            <li>进入化合物页面，找到 <b>Canonical SMILES</b> 字段</li>
                            <li>复制 SMILES 字符串（如 <code style="background: rgba(124,58,237,0.1); padding: 2px 6px; border-radius: 4px;">CC(=O)Oc1ccccc1C(=O)O</code>）</li>
                            <li>粘贴到下方的 "方式 3" 文本框中，点击 Predict</li>
                        </ol>
                        </div>
                        """, unsafe_allow_html=True)

    # --- 方式3：SMILES 直接输入 ---
    with st.container(border=True):
        st.markdown("""<div class="card-title">&#9997; 方式 3：直接输入 SMILES</div>""", unsafe_allow_html=True)
        st.caption("可从下拉菜单自动填入，也可手动编辑或粘贴外部 SMILES")

        smiles_input = st.text_input(
            "当前 SMILES",
            key="smiles_input_box",
            label_visibility="collapsed"
        )

        if smiles_input != st.session_state.get("smiles_input_box", ""):
            st.session_state.predicted_smiles = None
            st.session_state.predicted_logS = None
            st.session_state.ai_explanation = None
