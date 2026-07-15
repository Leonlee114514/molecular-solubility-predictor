"""
DisSolve - Internationalization (i18n) module.
Provides t() translation function and language selector widget.
"""

import streamlit as st

_LANG_KEY = "language"

# ── Engine ──

def init_language():
    """Initialize language in session state (must be called early in app.py)."""
    if _LANG_KEY not in st.session_state:
        st.session_state[_LANG_KEY] = "zh"


def get_lang():
    """Get current language code: 'zh' or 'en'."""
    return st.session_state.get(_LANG_KEY, "zh")


def t(key, **kwargs):
    """Translate a dot-notation key to the current language.

    Usage:
        t("result.solubility.high")  → "Highly soluble (易溶于水)" or "Highly soluble"
        t("prediction.step", n=2, total=5)  → "Step 2/5: ..."
    """
    lang = get_lang()
    entry = _ALL.get(key)
    if entry is None:
        return key
    text = entry.get(lang) or entry.get("zh", key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass
    return text


def render_language_selector():
    """Render language toggle buttons in the page header."""
    current = get_lang()
    cols = st.columns([1, 1, 10])
    with cols[0]:
        if st.button("🇨🇳 中文" if current == "en" else "✅ 中文",
                     key="lang_zh", use_container_width=True):
            st.session_state[_LANG_KEY] = "zh"
            st.rerun()
    with cols[1]:
        if st.button("✅ English" if current == "en" else "🇬🇧 English",
                     key="lang_en", use_container_width=True):
            st.session_state[_LANG_KEY] = "en"
            st.rerun()


# ── Translation dictionary ──

_ALL: dict[str, dict[str, str]] = {
    # ============================================================
    # Language selector
    # ============================================================
    "lang.zh_btn": {"zh": "🇨🇳 中文", "en": "🇨🇳 中文"},
    "lang.en_btn": {"zh": "🇬🇧 English", "en": "🇬🇧 English"},

    # ============================================================
    # app.py – Page / Top-level
    # ============================================================
    "app.title": {
        "zh": "DisSolve - 分子性质预测平台",
        "en": "DisSolve - Molecular Property Predictor",
    },
    "app.model.load_error": {"zh": "模型加载失败: {e}", "en": "Model loading failed: {e}"},
    "app.model.load_help": {
        "zh": "请先运行 'python train_model_v2.py' 训练模型",
        "en": "Please run 'python train_model_v2.py' to train the model first",
    },

    # Model selector
    "app.model.label": {"zh": "模型选择", "en": "Model Selection"},
    "app.model.auto": {"zh": "Auto (智能选择)", "en": "Auto (Smart Selection)"},
    "app.model.rf": {"zh": "RF (Random Forest)", "en": "RF (Random Forest)"},
    "app.model.gnn": {"zh": "GNN (Graph Neural Network)", "en": "GNN (Graph Neural Network)"},
    "app.model.ensemble": {"zh": "Ensemble (RF + GNN)", "en": "Ensemble (RF + GNN)"},
    "app.model.gnn_missing": {
        "zh": "GNN 模型未找到，仅 RF 可用。运行 `python scripts/train_gnn.py` 训练 GNN 模型。",
        "en": "GNN model not found. Only RF available. Run `python scripts/train_gnn.py` to train the GNN model.",
    },
    "app.predict_btn": {"zh": "Predict Solubility", "en": "Predict Solubility"},

    # Prediction – warnings / errors
    "app.predict.empty": {
        "zh": "请先输入或选择一个分子的 SMILES",
        "en": "Please enter or select a molecule SMILES first",
    },
    "app.predict.status": {"zh": "正在分析分子结构...", "en": "Analyzing molecular structure..."},
    "app.predict.step.parse": {"zh": "Step 1/4: 解析分子结构...", "en": "Step 1/4: Parsing molecular structure..."},
    "app.predict.step.parse_fail": {"zh": "解析失败", "en": "Parsing failed"},
    "app.predict.step.rf": {"zh": "Step 2/5: Random Forest 预测溶解度...", "en": "Step 2/5: Random Forest predicting solubility..."},
    "app.predict.step.pka": {"zh": "Step 3/5: 预测 pKa 与电离行为...", "en": "Step 3/5: Predicting pKa & ionization..."},
    "app.predict.step.ood": {"zh": "Step 3+/5: OOD 分布检测，智能选择模型...", "en": "Step 3+/5: OOD detection, smart model selection..."},
    "app.predict.step.gnn": {"zh": "Step 4/5: GNN 预测溶解度...", "en": "Step 4/5: GNN predicting solubility..."},
    "app.predict.step.shap": {"zh": "Step 5/5: SHAP 可解释性分析...", "en": "Step 5/5: SHAP explainability analysis..."},
    "app.predict.step.ood_post": {"zh": "Step 5+/5: OOD 分布检测...", "en": "Step 5+/5: OOD distribution detection..."},
    "app.predict.complete": {
        "zh": "分析完成！[{model}] 预测 logS = {logS:.3f}",
        "en": "Analysis complete! [{model}] Predicted logS = {logS:.3f}",
    },
    "app.predict.complete_state": {"zh": "complete", "en": "complete"},

    # SMILES error
    "app.error.invalid_smiles": {"zh": "Invalid SMILES: `{smiles}`", "en": "Invalid SMILES: `{smiles}`"},
    "app.error.parse_fail_info": {
        "zh": "该 SMILES 无法被 RDKit 解析。可能原因：",
        "en": "This SMILES could not be parsed by RDKit. Possible causes:",
    },
    "app.error.parse_reason1": {"zh": "- 分子含有金属/配位键，RDKit 不支持", "en": "- Molecule contains metal/coordination bonds not supported by RDKit"},
    "app.error.parse_reason2": {"zh": "- SMILES 语法错误（括号不匹配）", "en": "- SMILES syntax error (unmatched brackets)"},
    "app.error.parse_reason3": {"zh": "- 输入为空或含有非法字符", "en": "- Empty input or illegal characters"},

    # Disagreement
    "app.warn.disagreement.severe": {
        "zh": "⚠️ **RF 与 GNN 预测严重分歧**（|RF−GNN| = {diff:.2f}），已自动降级为 GNN 预测。此分子的预测可靠性较低，请谨慎参考。",
        "en": "⚠️ **RF & GNN predictions severely disagree** (|RF−GNN| = {diff:.2f}). Using GNN prediction. This result has low reliability.",
    },
    "app.warn.disagreement.moderate": {
        "zh": "📊 **RF 与 GNN 存在显著分歧**（|RF−GNN| = {diff:.2f}），加权集成已偏向 GNN。建议结合分子结构自行判断。",
        "en": "📊 **RF & GNN notably disagree** (|RF−GNN| = {diff:.2f}). Weighted ensemble favors GNN. Interpret with caution.",
    },

    # Model labels in status
    "app.model_label.rf": {"zh": "RF", "en": "RF"},
    "app.model_label.gnn": {"zh": "GNN", "en": "GNN"},
    "app.model_label.ensemble": {"zh": "Ensemble", "en": "Ensemble"},
    "app.model_label.auto": {"zh": "Auto → {actual}", "en": "Auto → {actual}"},

    # Toast
    "app.toast.prediction": {
        "zh": "预测: name={name} logS={logS:.3f} pKa={pKa}",
        "en": "Prediction: name={name} logS={logS:.3f} pKa={pKa}",
    },

    # Batch prediction
    "app.batch.title": {"zh": "批量预测（上传 CSV）", "en": "Batch Prediction (Upload CSV)"},
    "app.batch.desc": {
        "zh": "上传包含 SMILES 列的 CSV 文件，批量预测所有分子的溶解度与 pKa",
        "en": "Upload a CSV file with a SMILES column to batch-predict solubility & pKa for all molecules",
    },
    "app.batch.upload_label": {"zh": "选择 CSV 文件", "en": "Choose CSV file"},
    "app.batch.file_info": {
        "zh": "文件: `{name}` | {rows} 行 | 检测到 SMILES 列: **{col}**",
        "en": "File: `{name}` | {rows} rows | Detected SMILES column: **{col}**",
    },
    "app.batch.start_btn": {"zh": "开始批量预测", "en": "Start Batch Prediction"},
    "app.batch.col_error": {
        "zh": "列 '{col}' 不存在，请检查文件格式",
        "en": "Column '{col}' not found. Please check file format.",
    },
    "app.batch.progress_start": {"zh": "正在预测 0/{total}...", "en": "Predicting 0/{total}..."},
    "app.batch.progress_feature": {"zh": "计算特征 {i}/{total}...", "en": "Computing features {i}/{total}..."},
    "app.batch.progress_gnn": {"zh": "GNN 预测 {i}/{total}...", "en": "GNN prediction {i}/{total}..."},
    "app.batch.complete": {"zh": "批量预测完成！共 {n} 个分子", "en": "Batch prediction complete! {n} molecules"},
    "app.batch.download_btn": {"zh": "下载结果 CSV", "en": "Download Results CSV"},
    "app.batch.download_filename": {"zh": "dissolve_batch_results.csv", "en": "dissolve_batch_results.csv"},
    "app.batch.error": {"zh": "批量处理出错: {err}", "en": "Batch processing error: {err}"},

    # ============================================================
    # ui/components.py – Header / Footer / Input
    # ============================================================
    "header.tagline": {"zh": "AI-POWERED MOLECULAR PROPERTY PREDICTION", "en": "AI-POWERED MOLECULAR PROPERTY PREDICTION"},
    "header.title": {"zh": "DisSolve", "en": "DisSolve"},
    "header.subtitle": {
        "zh": "从分子结构预测水溶解度、pKa 与药理学特征",
        "en": "Predict Aqueous Solubility, pKa & Pharmacological Profiles from Molecular Structure",
    },
    "header.welcome": {"zh": "Welcome!", "en": "Welcome!"},
    "header.welcome_text": {
        "zh": "本应用基于<b>机器学习</b>模型（训练数据包含 <b>11,000+ 有机化合物</b>），从分子结构预测水溶解度(logS)、酸碱行为(pKa)和药理学特征。支持 2D & 3D 分子结构可视化、pKa 化学原理分析和 AI 生成的结构解释。",
        "en": "This app predicts aqueous solubility (logS), acid-base behavior (pKa), and pharmacological profiles from molecular structure using a <b>Machine Learning</b> model trained on <b>11,000+ organic compounds</b>. Explore 2D & 3D molecular structures, pKa chemistry insights, and AI-generated explanations.",
    },
    "header.badge_select": {"zh": "快速选择", "en": "Quick Select"},
    "header.badge_search": {"zh": "名称搜索", "en": "Name Search"},
    "header.badge_smiles": {"zh": "SMILES 输入", "en": "SMILES Input"},

    "footer.subtitle": {"zh": "溶解度预测 · pKa分析 · 药理学评估 | AI-Powered Chemistry Platform", "en": "Solubility · pKa · Pharmacology | AI-Powered Chemistry Platform"},
    "footer.built_with": {
        "zh": "Built with Streamlit | ML: Random Forest + GNN | AI: Kimi (Moonshot AI) | DB: 100+ local + PubChem API",
        "en": "Built with Streamlit | ML: Random Forest + GNN | AI: Kimi (Moonshot AI) | DB: 100+ local + PubChem API",
    },

    # ── Input Method 1: Quick Select ──
    "input.method1.title": {"zh": "方式 1：快速选择常见分子", "en": "Method 1: Quick Select Common Molecules"},
    "input.method1.search_label": {"zh": "搜索分子", "en": "Search molecules"},
    "input.method1.search_placeholder": {"zh": "输入中文或英文名称过滤...", "en": "Type Chinese or English name to filter..."},
    "input.method1.no_match": {"zh": "未找到匹配分子，显示全部选项", "en": "No matching molecules found, showing all"},
    "input.method1.select_label": {"zh": "选择分子", "en": "Select molecule"},

    # ── Input Method 2: Name Search ──
    "input.method2.title": {"zh": "方式 2：名称搜索（本地库 + PubChem API）", "en": "Method 2: Name Search (Local DB + PubChem API)"},
    "input.method2.desc": {"zh": "支持中英文，如 阿司匹林 / Aspirin / Ibuprofen / 咖啡因", "en": "Supports Chinese & English, e.g. Aspirin / Ibuprofen"},
    "input.method2.placeholder": {"zh": "例如 阿司匹林 或 Aspirin", "en": "e.g. Aspirin or Ibuprofen"},
    "input.method2.search_btn": {"zh": "搜索", "en": "Search"},
    "input.method2.exact_match": {"zh": "本地精确匹配：`{name}` -> `{smiles}`", "en": "Local exact match: `{name}` -> `{smiles}`"},
    "input.method2.try_predict": {"zh": "点击下方的 **Predict** 按钮查看结果", "en": "Click the **Predict** button below to view results"},
    "input.method2.fuzzy_match": {"zh": "本地模糊匹配：`{name}` → `{best}`", "en": "Local fuzzy match: `{name}` → `{best}`"},
    "input.method2.fuzzy_see_all": {"zh": "查看全部 {n} 个模糊匹配结果", "en": "View all {n} fuzzy matches"},
    "input.method2.confirm_btn": {"zh": "✅ 确认使用此结果", "en": "✅ Confirm this result"},
    "input.method2.skip_btn": {"zh": "不是我要的，搜 PubChem", "en": "Not what I want, search PubChem"},
    "input.method2.confirmed": {"zh": "已采用：`{best}` → `{smiles}`", "en": "Adopted: `{best}` → `{smiles}`"},
    "input.method2.pubchem_status": {"zh": "正在查询 PubChem API...", "en": "Querying PubChem API..."},
    "input.method2.pubchem_success": {"zh": "PubChem 匹配成功：{status}", "en": "PubChem match successful: {status}"},
    "input.method2.pubchem_fail": {"zh": "PubChem 未找到：{status}", "en": "PubChem not found: {status}"},
    "input.method2.pubchem_done": {"zh": "PubChem 匹配：`{name}` → `{smiles}`", "en": "PubChem match: `{name}` → `{smiles}`"},
    "input.method2.not_found": {"zh": "未找到：`{name}`", "en": "Not found: `{name}`"},
    "input.method2.local_not_found": {"zh": "本地未找到，正在查询 PubChem API...", "en": "Not found locally, querying PubChem API..."},
    "input.method2.suggestions_title": {"zh": "尝试建议：", "en": "Suggestions:"},
    "input.method2.suggestion1": {"zh": "- 检查拼写（如 **Aspirin** 而非 **Aspriin**）", "en": "- Check spelling (e.g. **Aspirin** not **Asprin**)"},
    "input.method2.suggestion2": {"zh": "- 尝试更常见的名称", "en": "- Try a more common name"},
    "input.method2.suggestion3": {"zh": "- 直接输入 SMILES（方式3）", "en": "- Enter SMILES directly (Method 3)"},
    "input.method2.manual_title": {"zh": "如何手动获取 SMILES？", "en": "How to manually obtain SMILES?"},
    "input.method2.manual_step1": {"zh": "访问 ", "en": "Visit "},
    "input.method2.manual_step2": {"zh": "在搜索框输入分子名称（英文，如 **Aspirin**）", "en": "Enter the molecule name in the search box (e.g. **Aspirin**)"},
    "input.method2.manual_step3": {"zh": "进入化合物页面，找到 **Canonical SMILES** 字段", "en": "Go to the compound page and find the **Canonical SMILES** field"},
    "input.method2.manual_step4": {"zh": "复制 SMILES 字符串（如 ", "en": "Copy the SMILES string (e.g. "},
    "input.method2.manual_step5": {"zh": "粘贴到下方的 \"方式 3\" 文本框中，点击 Predict", "en": "Paste it into Method 3 below and click Predict"},

    # ── Input Method 3: SMILES ──
    "input.method3.title": {"zh": "方式 3：直接输入 SMILES", "en": "Method 3: Enter SMILES Directly"},
    "input.method3.desc": {"zh": "可从下拉菜单自动填入，也可手动编辑或粘贴外部 SMILES", "en": "Auto-filled from the dropdown, or manually type/paste any SMILES"},
    "input.method3.label": {"zh": "当前 SMILES", "en": "Current SMILES"},

    # ── Input Method 4: File Upload ──
    "input.method4.title": {"zh": "方式 4：上传分子文件", "en": "Method 4: Upload Molecular File"},
    "input.method4.desc": {"zh": "支持 .mol .sdf .mol2 .pdb .xyz 格式", "en": "Supports .mol .sdf .mol2 .pdb .xyz formats"},
    "input.method4.upload_label": {"zh": "选择分子文件", "en": "Choose molecular file"},
    "input.method4.parse_success": {"zh": "解析成功：{name} → {formula} ({mw:.1f} Da)", "en": "Parsed successfully: {name} → {formula} ({mw:.1f} Da)"},
    "input.method4.parse_fail": {"zh": "文件解析失败：{name} 无法被 RDKit 识别", "en": "File parsing failed: {name} was not recognized by RDKit"},
    "input.method4.parse_fail_info": {"zh": "请确保文件包含有效的 2D/3D 分子结构", "en": "Please ensure the file contains a valid 2D/3D molecular structure"},

    # ── Prediction History ──
    "history.title": {"zh": "预测历史记录 ({n} 条)", "en": "Prediction History ({n} entries)"},
    "history.reuse_btn": {"zh": "复用", "en": "Reuse"},
    "history.custom_input": {"zh": "(自定义输入)", "en": "(Custom input)"},
    "history.logS_val": {"zh": "logS={val:.3f}", "en": "logS={val:.3f}"},
    "history.logS_unknown": {"zh": "logS=?", "en": "logS=?"},
    "history.pka_val": {"zh": "pKa={val:.2f}", "en": "pKa={val:.2f}"},
    "history.pka_unknown": {"zh": "pKa=?", "en": "pKa=?"},

    # ============================================================
    # ui/results.py – Results tabs
    # ============================================================
    # Tab names
    "result.tab.preview": {"zh": "Preview", "en": "Preview"},
    "result.tab.solubility": {"zh": "Solubility", "en": "Solubility"},
    "result.tab.pka": {"zh": "pKa", "en": "pKa"},
    "result.tab.pharmacology": {"zh": "Pharmacology", "en": "Pharmacology"},
    "result.tab.ai": {"zh": "AI Explanation", "en": "AI Explanation"},

    # OOD warnings
    "result.ood.high.title": {"zh": "**Out-of-Distribution 警告 — 预测可能不可靠**", "en": "**Out-of-Distribution Warning — Prediction may be unreliable**"},
    "result.ood.high.desc": {
        "zh": "该分子严重偏离训练数据分布，预测值可能极不可靠。建议仅用作定性参考，不要依赖具体数值。",
        "en": "This molecule deviates significantly from the training data distribution. Prediction may be highly unreliable — use only as a qualitative reference.",
    },
    "result.ood.medium.title": {"zh": "**Out-of-Distribution 注意 — 预测可能有一定误差**", "en": "**Out-of-Distribution Notice — Prediction may have some error**"},
    "result.ood.medium.desc": {
        "zh": "该分子部分偏离训练数据分布，预测值可能存在较大误差。建议谨慎解读预测结果。",
        "en": "This molecule partially deviates from the training data distribution. Results may have notable error. Interpret with caution.",
    },
    "result.ood.low": {
        "zh": "In-Distribution — 该分子在训练数据化学空间内，预测较为可靠",
        "en": "In-Distribution — This molecule falls within the training data chemical space. Prediction is reliable.",
    },

    # Tab 0 – Preview
    "result.preview.card_title": {"zh": "Molecule Preview", "en": "Molecule Preview"},
    "result.preview.img_caption": {"zh": "2D Molecular Structure", "en": "2D Molecular Structure"},
    "result.preview.img_fail": {"zh": "无法显示2D结构图", "en": "Unable to display 2D structure"},
    "result.preview.img_error": {"zh": "结构图生成失败: {err}", "en": "Structure image generation failed: {err}"},
    "result.preview.formula": {"zh": "Molecular Formula", "en": "Molecular Formula"},
    "result.preview.mol_weight": {"zh": "Molecular Weight", "en": "Molecular Weight"},
    "result.preview.smiles_label": {"zh": "SMILES", "en": "SMILES"},
    "result.preview.struct_fail": {"zh": "无法解析分子结构", "en": "Unable to parse molecular structure"},
    "result.preview.model3d": {"zh": "3D Ball-and-Stick Model", "en": "3D Ball-and-Stick Model"},
    "result.preview.model3d_fail": {"zh": "3D 模型生成失败（需安装 py3Dmol）", "en": "3D model generation failed (py3Dmol required)"},

    # Tab 1 – Solubility
    "result.solubility.card_title": {"zh": "Solubility Prediction", "en": "Solubility Prediction"},
    "result.solubility.badge_rf": {"zh": "Random Forest", "en": "Random Forest"},
    "result.solubility.badge_gnn": {"zh": "Graph Neural Network", "en": "Graph Neural Network"},
    "result.solubility.badge_ensemble": {"zh": "Ensemble (RF+GNN)", "en": "Ensemble (RF+GNN)"},
    "result.solubility.badge_weighted": {"zh": "Weighted Ensemble (0.45×RF+0.55×GNN)", "en": "Weighted Ensemble (0.45×RF+0.55×GNN)"},
    "result.solubility.metric_logs": {"zh": "Predicted Solubility (logS)", "en": "Predicted Solubility (logS)"},
    "result.solubility.severe_disagree": {
        "zh": "⚠️ RF 与 GNN 严重分歧（|Δ|={diff:.2f}），已自动降级为 GNN 预测。请谨慎参考。",
        "en": "⚠️ RF & GNN severely disagree (|Δ|={diff:.2f}). Using GNN. Interpret with caution.",
    },
    "result.solubility.notable_disagree": {
        "zh": "📊 RF 与 GNN 存在显著分歧（|Δ|={diff:.2f}），集成偏向 GNN。",
        "en": "📊 RF & GNN notably disagree (|Δ|={diff:.2f}). Ensemble favors GNN.",
    },
    "result.solubility.ensemble_weighted": {"zh": "Weighted Ensemble", "en": "Weighted Ensemble"},
    "result.solubility.ensemble_rf": {"zh": "RF:", "en": "RF:"},
    "result.solubility.ensemble_gnn": {"zh": "GNN:", "en": "GNN:"},
    "result.solubility.disagreement": {"zh": "Disagreement", "en": "Disagreement"},
    "result.solubility.good_agreement": {"zh": "(good agreement)", "en": "(good agreement)"},
    "result.solubility.notable_disagreement": {"zh": "(notable disagreement)", "en": "(notable disagreement)"},
    "result.solubility.large_divergence": {"zh": "(large divergence — treat prediction with caution)", "en": "(large divergence — treat prediction with caution)"},
    "result.solubility.guide": {"zh": "Interpretation guide:", "en": "Interpretation guide:"},
    "result.solubility.high": {"zh": "易溶于水", "en": "Highly soluble"},
    "result.solubility.moderate": {"zh": "中等溶解", "en": "Moderately soluble"},
    "result.solubility.poor": {"zh": "难溶于水", "en": "Poorly soluble"},

    "result.solubility.descriptors": {"zh": "Molecular Descriptors", "en": "Molecular Descriptors"},
    "result.solubility.desc_mw": {"zh": "Molecular Weight", "en": "Molecular Weight"},
    "result.solubility.desc_logp": {"zh": "LogP (Hydrophobicity)", "en": "LogP (Hydrophobicity)"},
    "result.solubility.desc_hbd": {"zh": "H-Bond Donors", "en": "H-Bond Donors"},
    "result.solubility.desc_hba": {"zh": "H-Bond Acceptors", "en": "H-Bond Acceptors"},
    "result.solubility.desc_tpsa": {"zh": "TPSA (Å²)", "en": "TPSA (Å²)"},
    "result.solubility.desc_rotb": {"zh": "Rotatable Bonds", "en": "Rotatable Bonds"},
    "result.solubility.desc_arom": {"zh": "Aromatic Rings", "en": "Aromatic Rings"},
    "result.solubility.desc_aliph": {"zh": "Aliphatic Rings", "en": "Aliphatic Rings"},

    "result.solubility.shap_title": {"zh": "SHAP Explainability", "en": "SHAP Explainability"},
    "result.solubility.shap_guide": {
        "zh": "SHAP 值显示各分子特征对溶解度预测的贡献方向与幅度，红色=推动易溶，蓝色=推动难溶。",
        "en": "SHAP values show the contribution of each molecular feature to the solubility prediction. Red = increases solubility, blue = decreases solubility.",
    },
    "result.solubility.shap_positive": {"zh": "推动易溶 (正贡献)", "en": "Increases solubility (+)"},
    "result.solubility.shap_negative": {"zh": "推动难溶 (负贡献)", "en": "Decreases solubility (−)"},
    "result.solubility.shap_pred_vs_base": {"zh": "预测值: {pred:.3f} (基准值: {base:.3f})", "en": "Prediction: {pred:.3f} (Base value: {base:.3f})"},
    "result.solubility.shap_insight_title": {"zh": "💡 SHAP 洞察", "en": "💡 SHAP Insight"},
    "result.solubility.shap_insight_generic": {
        "zh": "LogP 是影响溶解度最重要的特征，其次是 TPSA 和氢键计数。如果 LogP 贡献指向难溶且 TPSA 贡献指向易溶，说明该分子存在亲水-疏水竞争。",
        "en": "LogP is the most important feature for solubility, followed by TPSA and H-bond counts. If LogP pushes toward insolubility but TPSA pushes toward solubility, the molecule has a hydrophilic-hydrophobic competition.",
    },

    # pKa result
    "result.pka.card_title": {"zh": "pKa Prediction", "en": "pKa Prediction"},
    "result.pka.metric": {"zh": "Predicted pKa", "en": "Predicted pKa"},
    "result.pka.decomp_title": {"zh": "Chemical Factor Decomposition", "en": "Chemical Factor Decomposition"},
    "result.pka.chart_title": {"zh": "pKa = {val:.2f} | 化学因素分解", "en": "pKa = {val:.2f} | Chemical Factor Decomposition"},
    "result.pka.enhance_acid": {"zh": "增强酸性", "en": "Enhances acidity"},
    "result.pka.enhance_base": {"zh": "增强碱性", "en": "Enhances basicity"},
    "result.pka.weaken_acid": {"zh": "减弱酸性", "en": "Weakens acidity"},
    "result.pka.weaken_base": {"zh": "减弱碱性", "en": "Weakens basicity"},
    "result.pka.how_to_read": {"zh": "如何读懂这张图", "en": "How to read this chart"},
    "result.pka.how_to_read_desc": {
        "zh": "紫色条表示增强酸性的因素（拉电子/共轭稳定），青色条表示增强碱性的因素（推电子）。",
        "en": "Purple bars show factors that enhance acidity (electron-withdrawing / resonance stabilization). Cyan bars show factors that enhance basicity (electron-donating).",
    },
    "result.pka.legend_enhance_acid": {"zh": "增强酸性", "en": "Enhances acidity"},
    "result.pka.legend_weaken_acid": {"zh": "减弱酸性", "en": "Weakens acidity"},
    "result.pka.unavailable": {"zh": "化学因素分析暂不可用", "en": "Chemical factor analysis temporarily unavailable"},
    "result.pka.model_unavailable": {"zh": "pKa 模型未加载，pKa 分析不可用。", "en": "pKa model not loaded. pKa analysis unavailable."},

    # Pharmacology
    "result.pharma.lipinski_title": {"zh": "Drug-likeness: Lipinski's Rule of Five", "en": "Drug-likeness: Lipinski's Rule of Five"},
    "result.pharma.lipinski_mw": {"zh": "分子量\nMol. Weight", "en": "Mol. Weight"},
    "result.pharma.lipinski_logp": {"zh": "脂水分配系数\nLogP", "en": "LogP"},
    "result.pharma.lipinski_hbd": {"zh": "氢键供体数\nH-Bond Donors", "en": "H-Bond Donors"},
    "result.pharma.lipinski_hba": {"zh": "氢键受体数\nH-Bond Acceptors", "en": "H-Bond Acceptors"},
    "result.pharma.lipinski_rotb": {"zh": "可旋转键数\nRotatable Bonds", "en": "Rotatable Bonds"},
    "result.pharma.lipinski_pass": {"zh": "PASS", "en": "PASS"},
    "result.pharma.lipinski_fail": {"zh": "FAIL", "en": "FAIL"},
    "result.pharma.lipinski_legend_pass": {"zh": "PASS — 符合规则", "en": "PASS — Meets rule"},
    "result.pharma.lipinski_legend_fail": {"zh": "FAIL — 超出阈值", "en": "FAIL — Exceeds threshold"},
    "result.pharma.lipinski_score": {
        "zh": "Lipinski 五规则评分：{pass}/{total} — {interpretation}",
        "en": "Lipinski Rule of Five: {pass}/{total} — {interpretation}",
    },
    "result.pharma.lipinski_history": {
        "zh": "Lipinski 五规则由 Christopher Lipinski 于 1997 年提出，用于评估化合物的口服药物相似性。该规则是药物化学中最广泛使用的 ADME/成药性预筛选标准之一。",
        "en": "Lipinski's Rule of Five was proposed by Christopher Lipinski in 1997 to evaluate oral drug-likeness of compounds. It remains one of the most widely used ADME/drug-likeness pre-screening filters in medicinal chemistry.",
    },
    "result.pharma.druglikeness_title": {"zh": "Drug-likeness Metrics: QED · SAscore · Fsp³", "en": "Drug-likeness Metrics: QED · SAscore · Fsp³"},
    "result.pharma.qed_label": {"zh": "Quantitative Estimate of Drug-likeness", "en": "Quantitative Estimate of Drug-likeness"},
    "result.pharma.qed_note": {
        "zh": "QED 综合分子量、LogP、氢键、电荷等 8 个描述符，数值越接近 1 表示越像药物。",
        "en": "QED integrates 8 descriptors including MW, LogP, H-bonds, and charge. Values closer to 1 indicate higher drug-likeness.",
    },
    "result.pharma.sascore_label": {"zh": "Synthetic Accessibility Score", "en": "Synthetic Accessibility Score"},
    "result.pharma.sascore_note": {
        "zh": "SAscore 基于分子片段频率和复杂度，1=极易合成，10=极难合成。",
        "en": "SAscore estimates synthetic difficulty based on fragment frequency & complexity. 1 = very easy, 10 = very hard.",
    },
    "result.pharma.fsp3_label": {"zh": "Fraction sp³ Carbons", "en": "Fraction sp³ Carbons"},
    "result.pharma.fsp3_note": {
        "zh": "Fsp³ 反映分子的三维度/饱和度。Fsp³ ≥ 0.45 通常与更好的临床候选药物性质相关。",
        "en": "Fsp³ reflects molecular 3D character / saturation. Fsp³ ≥ 0.45 is generally associated with better clinical candidate properties.",
    },
    "result.pharma.fsp3_detail": {
        "zh": "饱和碳数: {n_sp3}/{n_carbon} | 提示：增加 sp³ 比例可提高溶解度和三维多样性",
        "en": "sp³ carbons: {n_sp3}/{n_carbon} | Higher sp³ fraction improves solubility & 3D diversity",
    },
    "result.pharma.ionization_title": {"zh": "Ionization Profile", "en": "Ionization Profile"},
    "result.pharma.ionization_chart_title": {"zh": "不同生理环境下的分子态比例 | pKa = {val:.2f}", "en": "Unionized fraction across physiological pH | pKa = {val:.2f}"},
    "result.pharma.ionization_yaxis": {"zh": "分子态比例 (Unionized %)", "en": "Unionized fraction (%)"},
    "result.pharma.pharma_analysis": {"zh": "药理学分析", "en": "Pharmacological Analysis"},
    "result.pharma.pka_acid_low": {
        "zh": "**pKa**：弱酸性 (pKa={val:.1f})，胃吸收好，**空腹服用**效果更佳。",
        "en": "**pKa**: Weak acid (pKa={val:.1f}), good gastric absorption, best taken **on an empty stomach**.",
    },
    "result.pharma.pka_acid_mid": {
        "zh": "**pKa**：中等酸性 (pKa={val:.1f})，全肠道吸收，对服药时间要求不高。",
        "en": "**pKa**: Moderate acid (pKa={val:.1f}), absorbed throughout the GI tract. No strict timing requirement.",
    },
    "result.pharma.pka_base_high": {
        "zh": "**pKa**：强碱性 (pKa={val:.1f})，胃吸收差，**餐后服用**可减少胃刺激，主要在小肠吸收。",
        "en": "**pKa**: Strong base (pKa={val:.1f}), poor gastric absorption. Take **after meals** to reduce gastric irritation.",
    },
    "result.pharma.pka_base_low": {
        "zh": "**pKa**：弱碱性 (pKa={val:.1f})，小肠吸收为主，血液中有利于排泄。",
        "en": "**pKa**: Weak base (pKa={val:.1f}), primarily absorbed in the small intestine. Excretion favored in blood.",
    },
    "result.pharma.pka_neutral": {
        "zh": "**pKa**：接近中性 (pKa={val:.1f})，吸收行为较复杂。",
        "en": "**pKa**: Near neutral (pKa={val:.1f}), complex absorption behavior.",
    },
    "result.pharma.linkage_title": {"zh": "溶解度 x pKa 联动分析", "en": "Solubility × pKa Joint Analysis"},

    # ADME/Tox
    "result.pharma.admet_title": {"zh": "ADME/Tox 药代动力学概览", "en": "ADME/Tox Pharmacokinetics Overview"},
    "result.pharma.admet_absorption": {"zh": "Absorption 吸收", "en": "Absorption"},
    "result.pharma.admet_distribution": {"zh": "Distribution 分布", "en": "Distribution"},
    "result.pharma.admet_metabolism": {"zh": "Metabolism 代谢", "en": "Metabolism"},
    "result.pharma.admet_excretion": {"zh": "Excretion 排泄", "en": "Excretion"},
    "result.pharma.admet_toxicity": {"zh": "Toxicity 毒性", "en": "Toxicity"},
    "result.pharma.admet_absorption_title": {"zh": "吸收分析", "en": "Absorption Analysis"},
    "result.pharma.admet_distribution_title": {"zh": "分布分析", "en": "Distribution Analysis"},
    "result.pharma.admet_metabolism_title": {"zh": "代谢分析", "en": "Metabolism Analysis"},
    "result.pharma.admet_excretion_title": {"zh": "排泄分析", "en": "Excretion Analysis"},
    "result.pharma.admet_toxicity_title": {"zh": "毒性分析", "en": "Toxicity Analysis"},
    "result.pharma.admet_vd": {"zh": "表观分布容积 (Vd)", "en": "Volume of Distribution (Vd)"},
    "result.pharma.admet_ppb": {"zh": "血浆蛋白结合率", "en": "Plasma Protein Binding"},
    "result.pharma.admet_metab_sites": {"zh": "代谢热点", "en": "Metabolism Hotspots"},
    "result.pharma.admet_cyp": {"zh": "相关代谢酶", "en": "Related CYP Enzymes"},
    "result.pharma.admet_excretion_route": {"zh": "排泄途径", "en": "Excretion Route"},
    "result.pharma.admet_tox_alerts": {"zh": "毒性警报", "en": "Toxicity Alerts"},
    "result.pharma.admet_risk": {"zh": "[风险：{level}]", "en": "[Risk: {level}]"},

    # GNN Explainer
    "result.gnn.title": {"zh": "GNN Explainability — 原子/边注意力分析", "en": "GNN Explainability — Atom/Bond Attention Analysis"},
    "result.gnn.desc": {
        "zh": "GNNExplainer 识别分子中对该分子溶解度预测最重要的化学键和原子特征。暖色 = 更高重要性。",
        "en": "GNNExplainer identifies the bonds and atom features most important for this molecule's solubility prediction. Warmer colors = higher importance.",
    },
    "result.gnn.no_mol": {"zh": "无可解释的分子", "en": "No molecule to explain"},
    "result.gnn.running": {"zh": "正在运行 GNNExplainer 分析（约 10-30 秒）...", "en": "Running GNNExplainer analysis (~10-30 seconds)..."},
    "result.gnn.fail": {"zh": "GNN 解释生成失败: {err}", "en": "GNN explanation failed: {err}"},
    "result.gnn.model_unavailable": {"zh": "GNN 解释生成失败（模型未加载或分子无法解析）", "en": "GNN explanation unavailable (model not loaded or molecule invalid)"},
    "result.gnn.no_bonds": {"zh": "该分子无非氢键，无法进行边重要性分析", "en": "This molecule has no non-hydrogen bonds, cannot analyze bond importance"},
    "result.gnn.img_caption": {"zh": "关键化学键高亮（暖色 = 更重要）", "en": "Key bond highlighting (warmer = more important)"},
    "result.gnn.elapsed": {"zh": "分析耗时：", "en": "Analysis time: "},
    "result.gnn.top_bonds": {"zh": "Top 最重要的化学键", "en": "Top most important bonds"},
    "result.gnn.bond_importance": {"zh": "重要性: {imp:.3f} ({pct:.0f}%)", "en": "Importance: {imp:.3f} ({pct:.0f}%)"},
    "result.gnn.atom_title": {"zh": "原子重要性（基于相连键汇总）", "en": "Atom Importance (aggregated from bond importance)"},
    "result.gnn.feature_title": {"zh": "原子特征重要性", "en": "Atom Feature Importance"},
    "result.gnn.how_to_read": {"zh": "如何读懂 GNN 解释", "en": "How to read GNN explanations"},
    "result.gnn.how_to_read_desc": {
        "zh": "GNNExplainer 通过分析消息传递路径，识别哪些化学键和原子特征对模型的溶解度预测贡献最大。高重要性（暖色）的键表示删除或改性该键会显著改变预测结果，是分子中结构-性质关系的关键位点。",
        "en": "GNNExplainer identifies which bonds and atom features contribute most to the solubility prediction by analyzing message-passing paths. High-importance (warm-colored) bonds are key structure-property relationship sites.",
    },

    # AI Explanation tab
    "result.ai.title": {"zh": "AI Chemistry Explanation", "en": "AI Chemistry Explanation"},
    "result.ai.clear_btn": {"zh": "清除解释", "en": "Clear"},
    "result.ai.need_manual": {"zh": "AI 解释需要手动调用（消耗 API 额度）", "en": "AI explanation requires manual invocation (API credits consumed)"},
    "result.ai.generate_btn": {"zh": "生成 AI 解释", "en": "Generate AI Explanation"},
    "result.ai.generating": {"zh": "正在分析分子结构...", "en": "Analyzing molecular structure..."},

    # Results – display error
    "result.display_error": {"zh": "显示时解析失败，请重新输入 SMILES", "en": "Parsing failed during display. Please re-enter SMILES."},

    # ============================================================
    # core/analysis.py – Analysis text
    # ============================================================
    "analysis.lipinski.good": {
        "zh": "符合 Lipinski 五规则，具有良好口服生物利用度潜力",
        "en": "Meets Lipinski's Rule of Five, good oral bioavailability potential",
    },
    "analysis.lipinski.violated": {
        "zh": "违反 {n} 条规则，口服吸收可能受限，但仍有成为药物的可能（许多成功药物也违反五规则）",
        "en": "Violates {n} rule(s). Oral absorption may be limited, but many successful drugs also violate the rule of five.",
    },

    # Drug-likeness levels
    "analysis.qed.attractive": {"zh": "Attractive (有吸引力)", "en": "Attractive"},
    "analysis.qed.moderate": {"zh": "Moderate (中等)", "en": "Moderate"},
    "analysis.qed.low": {"zh": "Low (偏低)", "en": "Low"},
    "analysis.sa.easy": {"zh": "Easy (容易合成)", "en": "Easy (to synthesize)"},
    "analysis.sa.moderate": {"zh": "Moderate (中等难度)", "en": "Moderate"},
    "analysis.sa.hard": {"zh": "Difficult (难以合成)", "en": "Difficult"},
    "analysis.fsp3.high": {"zh": "High 3D complexity (高三维复杂度)", "en": "High 3D complexity"},
    "analysis.fsp3.moderate": {"zh": "Moderate (中等)", "en": "Moderate"},
    "analysis.fsp3.planar": {"zh": "Mostly planar (偏平面)", "en": "Mostly planar"},

    # pKa factor names (used in plots)
    "analysis.pka.inductive": {"zh": "Inductive\n(诱导效应)", "en": "Inductive\n(Inductive Effect)"},
    "analysis.pka.resonance": {"zh": "Resonance\n(共轭效应)", "en": "Resonance\n(Conjugation Effect)"},
    "analysis.pka.intra_hb": {"zh": "Intra-HB\n(分子内氢键)", "en": "Intra-HB\n(Intramolecular H-Bond)"},
    "analysis.pka.steric": {"zh": "Steric\n(空间位阻)", "en": "Steric\n(Steric Hindrance)"},
    "analysis.pka.hybridization": {"zh": "Hybridization\n(杂化/芳香性)", "en": "Hybridization\n(Hybridization/Aromaticity)"},

    # ADME/Tox – Absorption
    "analysis.admet.absorption.tpsa_low": {"zh": "TPSA < 60 Å²，易于穿过肠道细胞膜", "en": "TPSA < 60 Å², good intestinal membrane permeability"},
    "analysis.admet.absorption.tpsa_mid": {"zh": "TPSA 中等 (60-140 Å²)，吸收尚可", "en": "TPSA moderate (60-140 Å²), acceptable absorption"},
    "analysis.admet.absorption.tpsa_high": {"zh": "TPSA > 140 Å²，极性表面积较大，可能限制被动跨膜吸收", "en": "TPSA > 140 Å², large polar surface may limit passive trans-membrane absorption"},
    "analysis.admet.absorption.pka_acid": {"zh": "pKa = {val:.1f}（酸性），胃中(pH 1.5)以分子态为主，胃吸收良好", "en": "pKa = {val:.1f} (acidic), unionized in stomach (pH 1.5), good gastric absorption"},
    "analysis.admet.absorption.pka_base": {"zh": "pKa = {val:.1f}（碱性），胃中离子化，主要在小肠(pH 6.8)吸收", "en": "pKa = {val:.1f} (basic), ionized in stomach, mainly absorbed in small intestine (pH 6.8)"},
    "analysis.admet.absorption.pka_neutral": {"zh": "pKa = {val:.1f}（近中性），全肠道均有吸收", "en": "pKa = {val:.1f} (near neutral), absorbed throughout GI tract"},
    "analysis.admet.absorption.hbd_high": {"zh": "H-Bond Donors > 5，跨膜需要脱溶剂化，可能降低吸收", "en": "H-Bond Donors > 5, desolvation required for membrane crossing, may reduce absorption"},
    "analysis.admet.absorption.hba_high": {"zh": "H-Bond Acceptors > 10，过多的氢键受体降低膜通透性", "en": "H-Bond Acceptors > 10, excessive H-bond acceptors reduce membrane permeability"},
    "analysis.admet.absorption.fallback": {"zh": "吸收特性待评估", "en": "Absorption characteristics pending evaluation"},

    # ADME/Tox – Distribution
    "analysis.admet.distribution.lipophilic": {"zh": "分子亲脂性强 (LogP > 3)，倾向于分布到脂肪组织和通过血脑屏障", "en": "Highly lipophilic (LogP > 3),倾向于 distribute to adipose tissue and cross BBB"},
    "analysis.admet.distribution.hydrophilic": {"zh": "分子亲水性强 (LogP < 0)，主要分布在血浆和细胞外液", "en": "Highly hydrophilic (LogP < 0), mainly distributed in plasma and extracellular fluid"},
    "analysis.admet.distribution.moderate": {"zh": "LogP 适中，组织分布较均衡", "en": "Moderate LogP, balanced tissue distribution"},
    "analysis.admet.distribution.bbb": {"zh": "低 TPSA + 中等 LogP = 可能通过血脑屏障 (BBB)", "en": "Low TPSA + moderate LogP = may cross Blood-Brain Barrier (BBB)"},
    "analysis.admet.distribution.high_mw": {"zh": "分子量 > 500，组织渗透能力下降", "en": "MW > 500, reduced tissue penetration"},
    "analysis.admet.distribution.vd_high": {"zh": "较高（亲脂性强，易分布至组织）", "en": "High (lipophilic, readily distributes to tissues)"},
    "analysis.admet.distribution.vd_low": {"zh": "较低（亲水性强，主要留在血浆中）", "en": "Low (hydrophilic, mainly stays in plasma)"},
    "analysis.admet.distribution.ppb_high": {"zh": "较高（亲脂性强，与血浆蛋白结合率高）", "en": "High (lipophilic, high plasma protein binding)"},
    "analysis.admet.distribution.ppb_low": {"zh": "较低（亲水性强，游离药物比例高）", "en": "Low (hydrophilic, high free fraction)"},
    "analysis.admet.distribution.fallback": {"zh": "分布特性待评估", "en": "Distribution characteristics pending evaluation"},

    # ADME/Tox – Metabolism
    "analysis.admet.metabolism.aromatic": {"zh": "芳香环（可能被 CYP450 氧化为环氧化物/酚类）", "en": "Aromatic rings (may undergo CYP450 oxidation to epoxides/phenols)"},
    "analysis.admet.metabolism.phenol": {"zh": "酚羟基（易被 II 相代谢：葡萄糖醛酸化/硫酸化）", "en": "Phenolic OH (prone to Phase II metabolism: glucuronidation/sulfation)"},
    "analysis.admet.metabolism.carboxylic": {"zh": "羧基（可能与氨基酸结合或形成酰基葡萄糖醛酸）", "en": "Carboxyl group (may undergo amino acid conjugation or acyl glucuronidation)"},
    "analysis.admet.metabolism.ester": {"zh": "酯键（被酯酶水解，可能首过效应显著）", "en": "Ester bond (hydrolyzed by esterases, may have significant first-pass effect)"},
    "analysis.admet.metabolism.amide": {"zh": "酰胺键（代谢较稳定，但可被酰胺酶水解）", "en": "Amide bond (metabolically stable, but may be cleaved by amidases)"},
    "analysis.admet.metabolism.methyl": {"zh": "甲基（可被 CYP450 氧化为羟甲基 -> 醛 -> 羧酸）", "en": "Methyl group (CYP450 oxidation: hydroxymethyl → aldehyde → carboxylic acid)"},
    "analysis.admet.metabolism.amine_sec": {"zh": "伯/仲胺基（可能发生 N-脱烷基或 N-氧化）", "en": "Primary/secondary amine (may undergo N-dealkylation or N-oxidation)"},
    "analysis.admet.metabolism.amine_tert": {"zh": "叔胺基（易发生 N-脱甲基化）", "en": "Tertiary amine (prone to N-demethylation)"},
    "analysis.admet.metabolism.fallback": {"zh": "结构较简单，主要代谢途径待实验验证", "en": "Simple structure, primary metabolic pathways require experimental validation"},
    "analysis.admet.metabolism.cyp_fallback": {"zh": "待实验确定", "en": "To be determined experimentally"},

    # ADME/Tox – Excretion
    "analysis.admet.excretion.renal": {"zh": "分子量小 + 亲水性适中 → 倾向于肾脏排泄（肾小球滤过）", "en": "Low MW + moderate hydrophilicity → renal excretion (glomerular filtration)"},
    "analysis.admet.excretion.biliary": {"zh": "分子量 > 500 → 倾向于肝胆排泄（胆汁）", "en": "MW > 500 → hepatobiliary excretion (bile)"},
    "analysis.admet.excretion.hepato": {"zh": "LogP > 3 → 在肾小管中易被重吸收，肝胆排泄比例增加", "en": "LogP > 3 →容易 reabsorbed in renal tubules, increased hepatobiliary excretion"},
    "analysis.admet.excretion.tpsa": {"zh": "高 TPSA 有利于肾脏排泄（水溶性代谢物）", "en": "High TPSA favors renal excretion (water-soluble metabolites)"},
    "analysis.admet.excretion.conjugate": {"zh": "含羧基/酚羟基 → 易形成 II 相代谢物经肾脏排出", "en": "Carboxyl/phenolic OH →容易 form Phase II conjugates for renal excretion"},
    "analysis.admet.excretion.route_renal": {"zh": "主要经肾脏排泄", "en": "Primarily renal excretion"},
    "analysis.admet.excretion.route_biliary": {"zh": "倾向于肝胆排泄", "en": "Hepatobiliary excretion"},
    "analysis.admet.excretion.route_dual": {"zh": "肝肾双途径排泄", "en": "Renal + hepatobiliary excretion"},
    "analysis.admet.excretion.fallback": {"zh": "排泄途径待评估", "en": "Excretion route pending evaluation"},

    # ADME/Tox – Toxicity
    "analysis.admet.toxicity.nitro": {"zh": "硝基芳香族 → 可能经 CYP450 还原为有毒的亚硝基/羟胺中间体，有致突变风险", "en": "Nitroaromatic → may be reduced by CYP450 to toxic nitroso/hydroxylamine intermediates, mutagenic risk"},
    "analysis.admet.toxicity.aniline": {"zh": "芳香胺 → 可能经 N-氧化生成致癌性 N-羟基代谢物", "en": "Aromatic amine → may undergo N-oxidation to carcinogenic N-hydroxy metabolites"},
    "analysis.admet.toxicity.epoxide": {"zh": "环氧化物 → 高反应活性，可与 DNA/蛋白质共价结合，可能致癌", "en": "Epoxide → highly reactive, can covalently bind DNA/proteins, potentially carcinogenic"},
    "analysis.admet.toxicity.hydrazine": {"zh": "肼类结构 → 已知的肝毒性警报结构", "en": "Hydrazine structure → known hepatotoxicity alert"},
    "analysis.admet.toxicity.alkyl_halide": {"zh": "卤代烷基 → 可能是烷化剂，与谷胱甘肽结合消耗肝脏保护物质", "en": "Alkyl halide → potential alkylating agent, depletes glutathione"},
    "analysis.admet.toxicity.michael": {"zh": "Michael 受体 (α,β-不饱和羰基) → 可与亲核基团非特异性结合，可能引起肝毒性/皮肤致敏", "en": "Michael acceptor (α,β-unsaturated carbonyl) → non-specific nucleophile binding, potential hepatotoxicity/skin sensitization"},
    "analysis.admet.toxicity.halogen": {"zh": "含卤素取代基 → 代谢较稳定，但可能生物积累", "en": "Halogen substituent → metabolically stable, but may bioaccumulate"},
    "analysis.admet.toxicity.none": {"zh": "未检出常见结构警报，常规毒性风险较低", "en": "No common structural alerts detected, generally low toxicity risk"},
    "analysis.admet.toxicity.level_high": {"zh": "高", "en": "High"},
    "analysis.admet.toxicity.level_medium": {"zh": "中", "en": "Medium"},
    "analysis.admet.toxicity.level_low": {"zh": "低", "en": "Low"},

    # Drug-likeness general
    "analysis.druglikeness.vd": {"zh": "中等", "en": "Medium"},
    "analysis.druglikeness.ppb": {"zh": "中等", "en": "Medium"},

    # ============================================================
    # model.py – Solubility levels & pKa types
    # ============================================================
    "model.solubility.high": {"zh": "Highly soluble (易溶于水)", "en": "Highly soluble"},
    "model.solubility.moderate": {"zh": "Moderately soluble (中等溶解)", "en": "Moderately soluble"},
    "model.solubility.poor": {"zh": "Poorly soluble (难溶于水)", "en": "Poorly soluble"},
    "model.solubility.high_short": {"zh": "Highly soluble", "en": "Highly soluble"},
    "model.solubility.moderate_short": {"zh": "Moderately soluble", "en": "Moderately soluble"},
    "model.solubility.poor_short": {"zh": "Poorly soluble", "en": "Poorly soluble"},

    "model.pka.type.acidic_display": {"zh": "酸性分子 (Acidic)", "en": "Acidic"},
    "model.pka.type.basic_display": {"zh": "碱性分子 (Basic)", "en": "Basic"},
    "model.pka.type.amphoteric_display": {"zh": "两性/中性 (Amphoteric/Neutral)", "en": "Amphoteric/Neutral"},
    "model.pka.type.acidic_desc": {"zh": "pKa 较低，在酸性环境中以分子态为主，脂溶性高", "en": "Low pKa, predominantly unionized in acidic environments, lipophilic"},
    "model.pka.type.basic_desc": {"zh": "pKa 较高，在碱性环境中以分子态为主", "en": "High pKa, predominantly unionized in basic environments"},
    "model.pka.type.amphoteric_desc": {"zh": "pKa 接近中性，电离行为随 pH 变化剧烈", "en": "Near-neutral pKa, ionization behavior varies significantly with pH"},

    "model.shap.morgan_fp": {"zh": "摩根指纹 (Morgan FP)", "en": "Morgan Fingerprint (Morgan FP)"},
    "model.no_model": {"zh": "No solubility model found (expected output_v2/solubility_model_v5.pkl.gz)", "en": "No solubility model found (expected output_v2/solubility_model_v5.pkl.gz)"},

    "model.model_label_rf": {"zh": "RF", "en": "RF"},
    "model.model_label_gnn": {"zh": "GNN", "en": "GNN"},
    "model.model_label_ensemble_w": {"zh": "Ensemble(W)", "en": "Ensemble(W)"},

    # ============================================================
    # ood_detector.py – Descriptor names & warning text
    # ============================================================
    "ood.descriptor.MolWt": {"zh": "分子量 (MolWt)", "en": "Mol. Weight (MolWt)"},
    "ood.descriptor.LogP": {"zh": "脂水分配系数 (LogP)", "en": "LogP (Partition Coefficient)"},
    "ood.descriptor.NumHDonors": {"zh": "氢键供体 (H-Donors)", "en": "H-Bond Donors"},
    "ood.descriptor.NumHAcceptors": {"zh": "氢键受体 (H-Acceptors)", "en": "H-Bond Acceptors"},
    "ood.descriptor.TPSA": {"zh": "极性表面积 (TPSA)", "en": "Polar Surface Area (TPSA)"},
    "ood.descriptor.NumRotatableBonds": {"zh": "可旋转键 (Rotatable Bonds)", "en": "Rotatable Bonds"},
    "ood.descriptor.NumAromaticRings": {"zh": "芳香环 (Aromatic Rings)", "en": "Aromatic Rings"},
    "ood.descriptor.NumAliphaticRings": {"zh": "脂肪环 (Aliphatic Rings)", "en": "Aliphatic Rings"},
    "ood.descriptor.FractionCSP3": {"zh": "碳饱和比例 (FractionCSP3)", "en": "Fraction sp³ (FractionCSP3)"},
    "ood.descriptor.NumSaturatedRings": {"zh": "饱和环数 (Saturated Rings)", "en": "Saturated Rings"},
    "ood.descriptor.HallKierAlpha": {"zh": "分子柔性 (Hall-Kier α)", "en": "Molecular Flexibility (Hall-Kier α)"},
    "ood.descriptor.Chi0v": {"zh": "连接性指数 χ0v (Chi0)", "en": "Connectivity Index χ0v (Chi0)"},
    "ood.descriptor.Chi1v": {"zh": "连接性指数 χ1v (Chi1)", "en": "Connectivity Index χ1v (Chi1)"},

    "ood.risk.high": {"zh": "HIGH", "en": "HIGH"},
    "ood.risk.medium": {"zh": "MEDIUM", "en": "MEDIUM"},
    "ood.risk.low": {"zh": "LOW", "en": "LOW"},
    "ood.risk.unknown": {"zh": "UNKNOWN", "en": "UNKNOWN"},

    "ood.warning.severe": {
        "zh": "该分子严重偏离训练数据分布，预测值可能极不可靠。建议仅用作定性参考，不要依赖具体数值。",
        "en": "This molecule deviates severely from the training data distribution. Predictions may be highly unreliable. Use only as a qualitative reference.",
    },
    "ood.warning.moderate": {
        "zh": "该分子部分偏离训练数据分布，预测值可能存在较大误差。建议谨慎解读结果。",
        "en": "This molecule partially deviates from the training data distribution. Predictions may have significant error. Interpret with caution.",
    },
    "ood.warning.desc_above": {"zh": "高于", "en": "above"},
    "ood.warning.desc_below": {"zh": "低于", "en": "below"},
    "ood.warning.desc_zscore": {
        "zh": "{name} 为 {val:.1f}，{dir}训练均值 ({mean:.1f} ± {std:.1f}) 超过 3 个标准差",
        "en": "{name} = {val:.1f}, {dir} training mean ({mean:.1f} ± {std:.1f}) by >3 standard deviations",
    },
    "ood.warning.desc_range": {
        "zh": "{name} 为 {val:.1f}，超出训练集范围 [{min:.1f}, {max:.1f}]",
        "en": "{name} = {val:.1f}, outside training range [{min:.1f}, {max:.1f}]",
    },
    "ood.warning.fp_low": {
        "zh": "分子指纹最大相似度仅 {sim:.2f}，与训练集中任何分子的结构差异都很大",
        "en": "Max fingerprint similarity only {sim:.2f}, structurally very different from all training molecules",
    },
    "ood.warning.fp_medium": {
        "zh": "分子指纹最大相似度仅 {sim:.2f}，与训练集分子的结构相似度偏低",
        "en": "Max fingerprint similarity only {sim:.2f}, low structural similarity to training molecules",
    },

    # ============================================================
    # molecules.py – PubChem error messages
    # ============================================================
    "molecule.pubchem.empty_name": {"zh": "名称不能为空", "en": "Name cannot be empty"},
    "molecule.pubchem.not_found": {"zh": "PubChem 未找到该化合物", "en": "PubChem did not find this compound"},
    "molecule.pubchem.empty_data": {"zh": "PubChem 返回空数据", "en": "PubChem returned empty data"},
    "molecule.pubchem.not_found_404": {"zh": "PubChem 未找到该化合物 (404)", "en": "PubChem not found (404)"},
    "molecule.pubchem.ssl_error": {"zh": "SSL 连接失败", "en": "SSL connection failed"},
    "molecule.pubchem.timeout": {"zh": "查询超时，PubChem 服务器无响应", "en": "Query timed out, PubChem server unresponsive"},
    "molecule.pubchem.network_error": {"zh": "网络异常: {err}", "en": "Network error: {err}"},
    "molecule.pubchem.unavailable": {"zh": "PubChem 持续不可用，请稍后重试", "en": "PubChem persistently unavailable, please retry later"},
    "molecule.pubchem.cached": {"zh": "success (cached)", "en": "success (cached)"},
    "molecule.pubchem.success": {"zh": "success (PubChem)", "en": "success (PubChem)"},
    "molecule.custom_input": {"zh": "(自定义输入)", "en": "(Custom input)"},

    # ============================================================
    # core/ai_client.py – AI prompts (keys used by the prompt builder)
    # ============================================================
    "ai.solubility.high": {"zh": "易溶于水", "en": "Highly soluble"},
    "ai.solubility.moderate": {"zh": "中等溶解", "en": "Moderately soluble"},
    "ai.solubility.poor": {"zh": "难溶于水", "en": "Poorly soluble"},
    "ai.solubility.desc_high": {"zh": "logS > 0，属于高溶解度", "en": "logS > 0, highly soluble"},
    "ai.solubility.desc_moderate": {"zh": "-2 < logS ≤ 0，属于中等溶解度", "en": "-2 < logS ≤ 0, moderately soluble"},
    "ai.solubility.desc_poor": {"zh": "logS ≤ -2，属于低溶解度", "en": "logS ≤ -2, poorly soluble"},
    "ai.shap.direction_up": {"zh": "推动易溶", "en": "Increases solubility"},
    "ai.shap.direction_down": {"zh": "推动难溶", "en": "Decreases solubility"},
    "ai.shap.unavailable": {"zh": "（SHAP 解释暂不可用，跳过第 4 段）", "en": "(SHAP data unavailable, section 4 will be skipped)"},
    "ai.pka.acid_label": {"zh": "酸性", "en": "Acidic"},
    "ai.pka.base_label": {"zh": "碱性", "en": "Basic"},
    "ai.pka.amphoteric_label": {"zh": "两性/中性", "en": "Amphoteric/Neutral"},
    "ai.pka.acid_desc": {
        "zh": "倾向于释放质子 (H⁺)，在胃 (pH≈1.5) 中以分子态为主易吸收，在血液 (pH≈7.4) 中以离子态为主利于排泄。",
        "en": "Tends to donate protons (H⁺). Unionized in stomach (pH≈1.5) for good absorption, ionized in blood (pH≈7.4) for excretion.",
    },
    "ai.pka.base_desc": {
        "zh": "倾向于结合质子 (H⁺)，在胃中易电离，主要在小肠 (pH≈6.8) 吸收，在血液中以离子态为主。",
        "en": "Tends to accept protons (H⁺). Ionized in stomach, mainly absorbed in small intestine (pH≈6.8), ionized in blood.",
    },
    "ai.pka.amphoteric_desc": {
        "zh": "在不同 pH 环境下既可释放也可结合质子，吸收行为复杂，随给药部位环境 pH 变化。",
        "en": "Can donate or accept protons depending on pH. Absorption is complex and varies with administration site pH.",
    },
    "ai.error.no_key": {
        "zh": "未配置 Kimi API Key。请在 .env 文件中写入：KIMI_API_KEY=sk-你的密钥",
        "en": "Kimi API Key not configured. Add KIMI_API_KEY=sk-xxx to your .env file.",
    },
    "ai.error.generic": {"zh": "AI 解释暂时不可用: {err}", "en": "AI explanation temporarily unavailable: {err}"},
    "ai.shap.title": {"zh": "【SHAP 关键特征】（影响最大的前 3 个）", "en": "【SHAP Key Features】(Top 3 by importance)"},
}

# Add zh-only entries as copies of en for keys that are English-only
for _key, _val in list(_ALL.items()):
    if "zh" not in _val and "en" in _val:
        _val["zh"] = _val["en"]
