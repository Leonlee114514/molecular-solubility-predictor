"""
DisSolve - Reusable UI components (header, footer, input area, CJK font).
"""

import streamlit as st
from molecules import MOLECULE_DB, SEARCH_INDEX, search_pubchem as search_pubchem_final
from core.state_keys import StateKey


def render_html(html_content, height=1):
    """Render HTML/JS via st.components.v1.html (same-origin, window.parent access works)."""
    st.components.v1.html(html_content, height=height)


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


def _resolve_display_name(smiles, fallback=""):
    """Reverse-lookup the display name for a SMILES in MOLECULE_DB."""
    if not smiles:
        return fallback
    return next(
        (k for k, v in MOLECULE_DB.items() if v == smiles and k != "(自定义输入)"),
        fallback,
    )


def _set_current_smiles(smiles, name=None):
    """Update SMILES_INPUT and track the molecule name for history."""
    if smiles != st.session_state.get(StateKey.SMILES_INPUT, ""):
        st.session_state[StateKey.SMILES_INPUT] = smiles
        st.session_state[StateKey.PREDICTED_SMILES] = None
        st.session_state[StateKey.PREDICTED_LOGS] = None
        st.session_state[StateKey.PREDICTED_PKA] = None
        st.session_state[StateKey.AI_EXPLANATION] = None
    if name:
        st.session_state[StateKey.CURRENT_MOLECULE_NAME] = name
    elif smiles:
        st.session_state[StateKey.CURRENT_MOLECULE_NAME] = _resolve_display_name(smiles)


def _on_radio_select():
    """Callback when user actively selects a molecule from the radio list.
    Only fires on explicit user interaction, not on every script run.
    """
    selected = st.session_state["molecule_select_radio"]
    if selected != "(自定义输入)":
        smiles = MOLECULE_DB.get(selected)
        if smiles:
            st.session_state[StateKey.SMILES_INPUT] = smiles
            st.session_state[StateKey.CURRENT_MOLECULE_NAME] = selected
            st.session_state[StateKey.PREDICTED_SMILES] = None
            st.session_state[StateKey.PREDICTED_LOGS] = None
            st.session_state[StateKey.PREDICTED_PKA] = None
            st.session_state[StateKey.AI_EXPLANATION] = None


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
        if StateKey.MOLECULE_SELECT_RADIO in st.session_state and st.session_state[StateKey.MOLECULE_SELECT_RADIO] in filtered_mols:
            current_idx = filtered_mols.index(st.session_state[StateKey.MOLECULE_SELECT_RADIO])
        elif st.session_state.get(StateKey.MOLECULE_SELECT_RADIO) is None:
            current_idx = 0

        selected_molecule = st.radio(
            "选择分子",
            filtered_mols,
            index=current_idx,
            key="molecule_select_radio",
            label_visibility="collapsed",
            on_change=_on_radio_select,
        )

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

        if StateKey.SEARCH_STATE not in st.session_state:
            st.session_state[StateKey.SEARCH_STATE] = None
            st.session_state[StateKey.SEARCH_QUERY] = ""
            st.session_state[StateKey.FUZZY_MATCHES] = []
            st.session_state[StateKey.FUZZY_BEST] = ""
            st.session_state[StateKey.FUZZY_SMILES] = ""

        if search_clicked and search_name:
            st.session_state[StateKey.SEARCH_QUERY] = search_name.strip().lower()
            st.session_state[StateKey.SEARCH_STATE] = None
            st.session_state[StateKey.FUZZY_MATCHES] = []
            st.rerun()

        if not search_name and st.session_state[StateKey.SEARCH_QUERY]:
            st.session_state[StateKey.SEARCH_QUERY] = ""
            st.session_state[StateKey.SEARCH_STATE] = None
            st.session_state[StateKey.FUZZY_MATCHES] = []
            st.rerun()

        query = st.session_state[StateKey.SEARCH_QUERY]
        if query:
            if query in SEARCH_INDEX:
                found_smiles = SEARCH_INDEX[query]
                st.success(f"本地精确匹配：`{search_name}` -> `{found_smiles}`")
                _set_current_smiles(found_smiles)
                st.info("点击下方的 **Predict** 按钮查看结果")
                st.session_state[StateKey.SEARCH_STATE] = "exact"
            else:
                matches = [k for k in SEARCH_INDEX.keys() if query in k or k in query]
                if matches:
                    matches.sort(key=lambda x: (0 if x.startswith(query) else 1, len(x)))
                    best_match = matches[0]
                    found_smiles = SEARCH_INDEX[best_match]

                    if st.session_state[StateKey.SEARCH_STATE] not in ("fuzzy_pending", "fuzzy_confirmed", "pubchem_pending", "pubchem_done", "no_match"):
                        st.session_state[StateKey.SEARCH_STATE] = "fuzzy_pending"
                        st.session_state[StateKey.FUZZY_MATCHES] = matches
                        st.session_state[StateKey.FUZZY_BEST] = best_match
                        st.session_state[StateKey.FUZZY_SMILES] = found_smiles

                    matches = st.session_state[StateKey.FUZZY_MATCHES]
                    best_match = st.session_state[StateKey.FUZZY_BEST]
                    found_smiles = st.session_state[StateKey.FUZZY_SMILES]

                    if st.session_state[StateKey.SEARCH_STATE] == "fuzzy_pending":
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
                            st.session_state[StateKey.SEARCH_STATE] = "fuzzy_confirmed"
                            _set_current_smiles(found_smiles)
                            st.rerun()
                        elif use_pubchem:
                            st.session_state[StateKey.SEARCH_STATE] = "pubchem_pending"
                            st.rerun()

                    if st.session_state[StateKey.SEARCH_STATE] == "fuzzy_confirmed":
                        st.success(f"已采用：`{best_match}` → `{found_smiles}`")
                        st.info("点击下方的 **Predict** 按钮查看结果")

                    if st.session_state[StateKey.SEARCH_STATE] == "pubchem_pending":
                        with st.status("正在查询 PubChem API...", expanded=True) as pub_status:
                            found_smiles, pub_status_str = search_pubchem_final(search_name)
                            if found_smiles:
                                pub_status.update(label=f"PubChem 匹配成功：{pub_status_str}", state="complete")
                            else:
                                pub_status.update(label=f"PubChem 未找到：{pub_status_str}", state="error")
                        if found_smiles:
                            st.session_state[StateKey.SEARCH_STATE] = "pubchem_done"
                            st.session_state[StateKey.FUZZY_SMILES] = found_smiles
                            _set_current_smiles(found_smiles, name=search_name.strip())
                        else:
                            st.session_state[StateKey.SEARCH_STATE] = "no_match"
                        st.rerun()

                    if st.session_state[StateKey.SEARCH_STATE] == "pubchem_done":
                        st.success(f"PubChem 匹配：`{search_name}` → `{st.session_state[StateKey.FUZZY_SMILES]}`")
                        st.info("点击下方的 **Predict** 按钮查看结果")

                    if st.session_state[StateKey.SEARCH_STATE] == "no_match":
                        st.error(f"未找到：`{search_name}`")
                else:
                    if st.session_state[StateKey.SEARCH_STATE] not in ("pubchem_done", "no_match"):
                        st.session_state[StateKey.SEARCH_STATE] = "pubchem_pending"
                        with st.status("本地未找到，正在查询 PubChem API...", expanded=True) as pub_status:
                            found_smiles, pub_status_str = search_pubchem_final(search_name)
                            if found_smiles:
                                pub_status.update(label=f"PubChem 匹配成功：{pub_status_str}", state="complete")
                            else:
                                pub_status.update(label=f"PubChem 未找到：{pub_status_str}", state="error")
                        if found_smiles:
                            st.session_state[StateKey.SEARCH_STATE] = "pubchem_done"
                            st.session_state[StateKey.FUZZY_SMILES] = found_smiles
                        else:
                            st.session_state[StateKey.SEARCH_STATE] = "no_match"

                    if st.session_state[StateKey.SEARCH_STATE] == "pubchem_done":
                        found_smiles = st.session_state[StateKey.FUZZY_SMILES]
                        st.success(f"PubChem 匹配：`{search_name}` -> `{found_smiles}`")
                        _set_current_smiles(found_smiles, name=search_name.strip())
                        st.info("点击下方的 **Predict** 按钮查看结果")
                    elif st.session_state[StateKey.SEARCH_STATE] == "no_match":
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

        if smiles_input != st.session_state.get(StateKey.SMILES_INPUT, ""):
            st.session_state[StateKey.PREDICTED_SMILES] = None
            st.session_state[StateKey.PREDICTED_LOGS] = None
            st.session_state[StateKey.PREDICTED_PKA] = None
            st.session_state[StateKey.AI_EXPLANATION] = None
        # Keep molecule name in sync with the current SMILES
        if smiles_input:
            resolved = _resolve_display_name(smiles_input, "")
            if resolved:
                st.session_state[StateKey.CURRENT_MOLECULE_NAME] = resolved
            else:
                st.session_state[StateKey.CURRENT_MOLECULE_NAME] = ""


def render_file_upload_input():
    """Render the 4th input method: upload .mol/.sdf/.pdb files."""
    with st.container(border=True):
        st.markdown("""<div class="card-title">&#128196; 方式 4：上传分子文件</div>""", unsafe_allow_html=True)
        st.caption("支持 .mol .sdf .mol2 .pdb .xyz 格式")

        uploaded = st.file_uploader(
            "选择分子文件",
            type=["mol", "sdf", "mol2", "pdb", "xyz"],
            key="mol_file_uploader",
            label_visibility="collapsed",
        )

        if uploaded is not None:
            _file_upload_key = f"_parsed_{uploaded.name}"
            if st.session_state.get(_file_upload_key) != uploaded.getvalue():
                # New file — parse it
                from features import smiles_from_file
                result = smiles_from_file(uploaded)
                if result is not None:
                    parsed_smiles, formula, mw = result
                    st.success(f"解析成功：{uploaded.name} → {formula} ({mw:.1f} Da)")
                    st.code(parsed_smiles, language=None)
                    if parsed_smiles != st.session_state.get(StateKey.SMILES_INPUT):
                        # Widget already rendered — use pending pattern
                        st.session_state["_pending_history_smiles"] = parsed_smiles
                        st.session_state[StateKey.PREDICTED_SMILES] = None
                        st.session_state[StateKey.PREDICTED_LOGS] = None
                        st.session_state[StateKey.PREDICTED_PKA] = None
                        st.session_state[StateKey.AI_EXPLANATION] = None
                    st.session_state[_file_upload_key] = uploaded.getvalue()
                    st.info("点击下方的 **Predict** 按钮查看结果")
                    st.rerun()
                else:
                    st.error(f"文件解析失败：{uploaded.name} 无法被 RDKit 识别")
                    st.info("请确保文件包含有效的 2D/3D 分子结构")


def render_prediction_history():
    """Show a collapsible list of past predictions for quick re-predict."""
    history = st.session_state.get(StateKey.PREDICTION_HISTORY, [])
    if not history:
        return

    with st.expander(f"&#128203; 预测历史记录 ({len(history)} 条)", expanded=False):
        for i, entry in enumerate(history):
            ts = entry.get("timestamp", "")
            smiles = entry.get("smiles", "")
            logS = entry.get("logS")
            pka = entry.get("pKa")
            name = entry.get("name", "")

            label_parts = [f"#{i + 1}"]
            if name:
                label_parts.append(name)
            label_parts.append(f"logS={logS:.3f}" if logS is not None else "logS=?")
            if pka is not None:
                label_parts.append(f"pKa={pka:.2f}")
            label_parts.append(ts)

            cols = st.columns([5, 1])
            with cols[0]:
                st.markdown(
                    f'<div style="font-size:0.85rem;color:var(--ob-text-secondary);'
                    f'font-family:monospace;overflow:hidden;text-overflow:ellipsis;">'
                    f'{" | ".join(label_parts)}<br>'
                    f'<span style="color:var(--ob-text-tertiary);font-size:0.75rem;">{smiles}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with cols[1]:
                if st.button("复用", key=f"hist_reuse_{i}", use_container_width=True):
                    # Can't set widget key directly — widget already rendered this run.
                    # Store pending value; app.py will apply it before the widget renders.
                    st.session_state["_pending_history_smiles"] = smiles
                    st.session_state["_pending_history_name"] = name
                    st.session_state[StateKey.PREDICTED_SMILES] = None
                    st.session_state[StateKey.PREDICTED_LOGS] = None
                    st.session_state[StateKey.PREDICTED_PKA] = None
                    st.session_state[StateKey.AI_EXPLANATION] = None
                    st.rerun()
