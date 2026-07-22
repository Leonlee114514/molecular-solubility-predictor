"""
DisSolve - Internationalization (i18n) module.
Provides t() translation function and language selector widget.
"""

import contextvars
import streamlit as st

_LANG_KEY = "language"

# ── Request-scoped language override (used by the FastAPI backend) ──
# When set (via set_request_language / language_context), get_lang() returns
# this value FIRST, before consulting Streamlit session state. Outside any
# override the Streamlit behavior is unchanged.
_request_lang: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "disolve_request_lang", default=None
)


def set_request_language(lang: str | None):
    """Set the request-scoped language override ('zh'/'en', or None to clear).

    Returns the contextvars Token so callers can reset explicitly if needed.
    """
    if lang is not None:
        lang = str(lang).strip().lower()
        if lang not in ("zh", "en"):
            lang = "zh"
    return _request_lang.set(lang)


class language_context:
    """Context manager: run a block under a specific language override.

    Usage (FastAPI backend):
        with language_context(lang):
            result = analyze_admet(...)   # t() calls resolve in `lang`
    """

    def __init__(self, lang: str):
        self._token = None
        self._lang = lang

    def __enter__(self):
        self._token = set_request_language(self._lang)
        return self

    def __exit__(self, *exc):
        _request_lang.reset(self._token)
        return False


# ── Engine ──

def init_language():
    """Initialize language in session state (must be called early in app.py)."""
    if _LANG_KEY not in st.session_state:
        st.session_state[_LANG_KEY] = "zh"


def get_lang():
    """Get current language code: 'zh' or 'en'.

    Resolution order: request-scoped override (FastAPI) -> Streamlit session
    state -> default 'zh'. Streamlit app behavior is unchanged when no
    override is active.
    """
    override = _request_lang.get()
    if override in ("zh", "en"):
        return override
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
    "app.error.network": {
        "zh": "无法连接后端服务。请先运行：uvicorn backend.main:app --port 8000",
        "en": "Cannot reach the backend server. Start it first: uvicorn backend.main:app --port 8000",
    },

    # Disagreement — Auto mode (system chose model)
    "app.warn.disagreement.severe": {
        "zh": "⚠️ **RF 与 GNN 预测严重分歧**（|RF−GNN| = {diff:.2f}），已自动降级为 GNN 预测。此分子的预测可靠性较低，请谨慎参考。",
        "en": "⚠️ **RF & GNN predictions severely disagree** (|RF−GNN| = {diff:.2f}). Using GNN prediction. This result has low reliability.",
    },
    "app.warn.disagreement.severe_manual": {
        "zh": "⚠️ **RF 与 GNN 预测严重分歧**（|RF−GNN| = {diff:.2f}），预测可靠性较低。请谨慎参考。",
        "en": "⚠️ **RF & GNN predictions severely disagree** (|RF−GNN| = {diff:.2f}). Prediction reliability is low. Interpret with caution.",
    },
    # Disagreement — moderate
    "app.warn.disagreement.moderate": {
        "zh": "📊 **RF 与 GNN 存在显著分歧**（|RF−GNN| = {diff:.2f}），加权集成已偏向 GNN。建议结合分子结构自行判断。",
        "en": "📊 **RF & GNN notably disagree** (|RF−GNN| = {diff:.2f}). Weighted ensemble favors GNN. Interpret with caution.",
    },
    "app.warn.disagreement.moderate_manual": {
        "zh": "📊 **RF 与 GNN 存在显著分歧**（|RF−GNN| = {diff:.2f}）。请谨慎参考预测结果。",
        "en": "📊 **RF & GNN notably disagree** (|RF−GNN| = {diff:.2f}). Interpret predictions with caution.",
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
    "app.batch.column_label": {"zh": "SMILES 列", "en": "SMILES column"},
    "app.batch.preview_rows": {"zh": "预览（前 5 行）", "en": "Preview (first 5 rows)"},
    "app.batch.running": {"zh": "批量预测进行中 {done}/{total}...", "en": "Batch prediction running {done}/{total}..."},
    "app.batch.parse_error": {"zh": "CSV 解析失败: {err}", "en": "Could not parse CSV: {err}"},
    "app.batch.no_rows": {"zh": "所选列没有有效数据行", "en": "No valid data rows in the selected column"},
    "app.batch.table.index": {"zh": "#", "en": "#"},
    "app.batch.table.smiles": {"zh": "SMILES", "en": "SMILES"},
    "app.batch.table.logs": {"zh": "logS", "en": "logS"},
    "app.batch.table.model": {"zh": "模型", "en": "Model"},
    "app.batch.table.pka": {"zh": "pKa", "en": "pKa"},
    "app.batch.table.ood": {"zh": "OOD 风险", "en": "OOD risk"},
    "app.batch.table.error": {"zh": "错误", "en": "Error"},
    "app.batch.rows_ok": {"zh": "成功 {ok} / 共 {total}", "en": "{ok} succeeded / {total} total"},

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
    "history.clear_btn": {"zh": "清空历史", "en": "Clear history"},
    "history.clear_confirm": {"zh": "确认清空？", "en": "Confirm clear?"},
    "history.time.just_now": {"zh": "刚刚", "en": "just now"},
    "history.time.minutes_ago": {"zh": "{n} 分钟前", "en": "{n} min ago"},
    "history.time.hours_ago": {"zh": "{n} 小时前", "en": "{n} h ago"},
    "history.time.days_ago": {"zh": "{n} 天前", "en": "{n} d ago"},

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
    "result.solubility.severe_disagree_auto": {
        "zh": "⚠️ RF 与 GNN 严重分歧（|Δ|={diff:.2f}），已自动降级为 GNN 预测。请谨慎参考。",
        "en": "⚠️ RF & GNN severely disagree (|Δ|={diff:.2f}). Using GNN prediction. Interpret with caution.",
    },
    "result.solubility.severe_disagree": {
        "zh": "⚠️ RF 与 GNN 严重分歧（|Δ|={diff:.2f}），预测可靠性较低。请谨慎参考。",
        "en": "⚠️ RF & GNN severely disagree (|Δ|={diff:.2f}). Prediction reliability is low. Interpret with caution.",
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
    "result.solubility.high_hint": {"zh": "非常易溶（如乙醇）", "en": "Very soluble (like ethanol)"},
    "result.solubility.poor_hint": {"zh": "难溶（如许多药物分子）", "en": "Poorly soluble (like many drug molecules)"},

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
    "analysis.admet.distribution.lipophilic": {"zh": "分子亲脂性强 (LogP > 3)，倾向于分布到脂肪组织和通过血脑屏障", "en": "Highly lipophilic (LogP > 3), tends to distribute to adipose tissue and cross BBB"},
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
    "analysis.admet.excretion.hepato": {"zh": "LogP > 3 → 在肾小管中易被重吸收，肝胆排泄比例增加", "en": "LogP > 3 → prone to reabsorption in renal tubules, increased hepatobiliary excretion"},
    "analysis.admet.excretion.tpsa": {"zh": "高 TPSA 有利于肾脏排泄（水溶性代谢物）", "en": "High TPSA favors renal excretion (water-soluble metabolites)"},
    "analysis.admet.excretion.conjugate": {"zh": "含羧基/酚羟基 → 易形成 II 相代谢物经肾脏排出", "en": "Carboxyl/phenolic OH → readily forms Phase II conjugates for renal excretion"},
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

    # ============================================================
    # ADDITIONAL RESULTS TEXT (previously missed)
    # ============================================================

    # Solubility tab – Chemistry Insight
    "result.solubility.insight_title": {"zh": "Chemistry Insight", "en": "Chemistry Insight"},
    "result.solubility.insight_tpsa": {"zh": "**TPSA** (Topological Polar Surface Area) measures how much of the molecule is polar. Higher TPSA usually means better water solubility.", "en": "**TPSA** (Topological Polar Surface Area) measures how much of the molecule is polar. Higher TPSA usually means better water solubility."},
    "result.solubility.insight_hbond": {"zh": "**H-Bond Donors/Acceptors** tell us how well the molecule can form hydrogen bonds with water.", "en": "**H-Bond Donors/Acceptors** tell us how well the molecule can form hydrogen bonds with water."},
    "result.solubility.insight_logp": {"zh": "**LogP** measures lipophilicity. Lower LogP means the molecule prefers water over oil.", "en": "**LogP** measures lipophilicity. Lower LogP means the molecule prefers water over oil."},

    # Solubility tab – SHAP chart labels
    "result.solubility.shap_xlabel": {"zh": "对溶解度的贡献值 (logS)", "en": "SHAP contribution to solubility (logS)"},
    "result.solubility.shap_title_pred": {"zh": "预测值: {pred:.3f} (基准值: {base:.3f})", "en": "Prediction: {pred:.3f} (Base: {base:.3f})"},
    "result.solubility.shap_legend_pos": {"zh": "推动易溶 (正贡献)", "en": "Increases solubility (+)"},
    "result.solubility.shap_legend_neg": {"zh": "推动难溶 (负贡献)", "en": "Decreases solubility (−)"},
    "result.solubility.shap_insight_leading": {"zh": "**关键分析**：模型预测该分子 **{level}**（logS = {logS:.3f}）。", "en": "**Key Analysis**: The model predicts this molecule is **{level}** (logS = {logS:.3f})."},
    "result.solubility.shap_supporting": {"zh": "推动这一结果的主要因素：{factors}。", "en": "Main factors driving this result: {factors}."},
    "result.solubility.shap_resisting": {"zh": "但以下因素在抵抗这一趋势、试图让分子{target}：{factors}。", "en": "However, the following factors resist this trend, pushing the molecule {target}: {factors}."},
    "result.solubility.shap_target_soluble": {"zh": "更易溶", "en": "more soluble"},
    "result.solubility.shap_target_insoluble": {"zh": "更难溶", "en": "less soluble"},
    "result.solubility.shap_shift": {"zh": "相比训练集平均分子（基准值 {base:.3f}），该分子的结构特征将预测值{direction}拉动了 {shift:.3f} 个单位。", "en": "Compared to the training set average (base {base:.3f}), the molecule's structural features shifted the prediction {direction} by {shift:.3f} units."},
    "result.solubility.shap_dir_up": {"zh": "向上", "en": "upward"},
    "result.solubility.shap_dir_down": {"zh": "向下", "en": "downward"},

    # pKa tab – chart labels
    "result.pka.decomp_title": {"zh": "Chemical Factor Decomposition", "en": "Chemical Factor Decomposition"},
    "result.pka.chart_xlabel": {"zh": "对 {unit} 的贡献", "en": "Contribution to {unit}"},
    "result.pka.unit_acid": {"zh": "增强酸性", "en": "enhance acidity"},
    "result.pka.unit_base": {"zh": "增强碱性", "en": "enhance basicity"},
    "result.pka.legend_enhance": {"zh": "增强{type}", "en": "Enhances {type}"},
    "result.pka.legend_weaken": {"zh": "减弱{type}", "en": "Weakens {type}"},
    "result.pka.legend_type_acid": {"zh": "酸性", "en": "acidity"},
    "result.pka.legend_type_base": {"zh": "碱性", "en": "basicity"},
    "result.pka.factor_guide": {
        "zh": "**如何读懂这张图**：紫色条越长 = 该因素越推动分子**释放/结合质子**；青色条越长 = 该因素越**抵抗**质子转移。和 SHAP 不同，这些不是机器学习权重，而是**真实的结构化学效应**。",
        "en": "**How to read this chart**: Longer purple bars = factor promotes **proton donation/acceptance**; longer cyan bars = factor **resists** proton transfer. Unlike SHAP, these are not ML weights but **real structural chemistry effects**.",
    },
    "result.pka.glossary_title": {"zh": "图表术语速查", "en": "Chart Glossary"},
    "result.pka.glossary_inductive": {"zh": "**诱导效应**（Inductive Effect）— 电负性原子通过 σ 键吸引或排斥电子，从而影响质子的结合与释放", "en": "**Inductive Effect** — electronegative atoms attract or repel electrons through σ bonds, affecting proton binding and release"},
    "result.pka.glossary_resonance": {"zh": "**共轭效应**（Resonance / Conjugation）— π 电子在共轭体系中离域分布，稳定电离后的离子形式", "en": "**Resonance / Conjugation** — π electrons delocalize in conjugated systems, stabilizing ionized forms"},
    "result.pka.glossary_intra_hb": {"zh": "**分子内氢键**（Intramolecular H-Bond）— 同一分子内不同基团间形成氢键，屏蔽极性、调节 pKa", "en": "**Intramolecular H-Bond** — H-bond between groups in the same molecule, shielding polarity and modulating pKa"},
    "result.pka.glossary_steric": {"zh": "**空间位阻**（Steric Hindrance）— 大体积原子或基团阻碍质子的接近与离去，改变反应活性", "en": "**Steric Hindrance** — bulky atoms/groups hinder proton approach and departure, altering reactivity"},
    "result.pka.glossary_hybrid": {"zh": "**杂化/芳香性**（Hybridization / Aromaticity）— sp² 碳比例与芳香环共轭体系带来的额外稳定性", "en": "**Hybridization / Aromaticity** — additional stability from sp² carbon ratio and aromatic conjugation"},
    "result.pka.unavailable_short": {"zh": "化学因素分析暂不可用", "en": "Chemical factor analysis unavailable"},
    "result.pka.model_unavailable_short": {"zh": "pKa 模型未加载，pKa 分析不可用。", "en": "pKa model not loaded. pKa analysis unavailable."},

    # Pharmacology – Lipinski property labels
    "result.pharma.lipinski_prop_mw": {"zh": "分子量\nMol. Weight", "en": "Mol. Weight"},
    "result.pharma.lipinski_prop_logp": {"zh": "脂水分配系数\nLogP", "en": "LogP"},
    "result.pharma.lipinski_prop_hbd": {"zh": "氢键供体数\nH-Bond Donors", "en": "H-Bond Donors"},
    "result.pharma.lipinski_prop_hba": {"zh": "氢键受体数\nH-Bond Acceptors", "en": "H-Bond Acceptors"},
    "result.pharma.lipinski_prop_rotb": {"zh": "可旋转键数\nRotatable Bonds", "en": "Rotatable Bonds"},
    "result.pharma.lipinski_history_html": {
        "zh": "&bull; <b>Christopher Lipinski (Pfizer, 1997)</b> 分析了 2,245 个进入 II 期临床的药物分子，总结出 5 条口服药物的经验规则<br>&bull; 规则认为分子违反 ≤1 条时，其<b>口服吸收和生物利用度</b>更可能达标<br>&bull; 但这只是筛选规则，<b>不是绝对标准</b>——许多成功药物也违反五规则（如天然产物、抗生素、抗癌药）<br>&bull; 超出规则范围 (bRo5) 的分子仍是现代药物化学的重要方向（如 PROTAC、大环分子）",
        "en": "&bull; <b>Christopher Lipinski (Pfizer, 1997)</b> analyzed 2,245 drug candidates that reached Phase II clinical trials, summarizing 5 empirical rules for oral drugs<br>&bull; Molecules with ≤1 violation are more likely to have good <b>oral absorption and bioavailability</b><br>&bull; This is a screening guideline, <b>not an absolute rule</b> — many successful drugs violate it (e.g., natural products, antibiotics, anticancer drugs)<br>&bull; Beyond Ro5 (bRo5) molecules are an important direction in modern medicinal chemistry (e.g., PROTACs, macrocycles)",
    },
    "result.pharma.lipinski_score_title": {"zh": "Lipinski 五规则评分：{score}/{total} — {text}", "en": "Lipinski Rule of Five: {score}/{total} — {text}"},

    # Pharmacology – Ionization profile
    "result.pharma.ionization_env_stomach": {"zh": "Stomach\n胃", "en": "Stomach"},
    "result.pharma.ionization_env_duodenum": {"zh": "Duodenum\n十二指肠", "en": "Duodenum"},
    "result.pharma.ionization_env_intestine": {"zh": "Small Intestine\n小肠", "en": "Small Intestine"},
    "result.pharma.ionization_env_blood": {"zh": "Blood/Brain\n血液/脑", "en": "Blood/Brain"},
    "result.pharma.ionization_ylabel": {"zh": "分子态比例 (Unionized %)", "en": "Unionized fraction (%)"},

    # Pharmacology – Analysis text (pKa-based)
    "result.pharma.analysis.strong_acid": {"zh": "**胃吸收优势**：pKa < 4，在胃酸（pH 1.5）中大部分以分子态存在，脂溶性高，容易被胃黏膜吸收。代表药物：阿司匹林 (pKa 3.5)、布洛芬 (pKa 4.9)。", "en": "**Gastric absorption advantage**: pKa < 4, mostly unionized in stomach acid (pH 1.5), lipophilic, easily absorbed by gastric mucosa. Examples: Aspirin (pKa 3.5), Ibuprofen (pKa 4.9)."},
    "result.pharma.analysis.mid_acid": {"zh": "**全肠道吸收**：pKa 中等，在胃和小肠中都有一定比例的分子态，吸收较均匀。注意：分子态比例高时脂溶性强，可能刺激胃黏膜。", "en": "**Whole GI tract absorption**: moderate pKa, unionized fraction in both stomach and intestine. Note: high unionized fraction means strong lipophilicity, may irritate gastric mucosa."},
    "result.pharma.analysis.strong_base": {"zh": "**肠道吸收为主**：强碱性分子在胃中几乎完全电离，难以吸收；进入小肠（pH 6.8）后分子态增加，主要在小肠吸收。代表药物：二甲双胍 (pKa ~12.4)。", "en": "**Intestinal absorption primary**: strongly basic molecules are fully ionized in the stomach; absorption increases in the small intestine (pH 6.8). Examples: Metformin (pKa ~12.4)."},
    "result.pharma.analysis.weak_base": {"zh": "**弱碱性分子**：在胃中少量电离，小肠中吸收良好。进入血液（pH 7.4）后可能部分电离，水溶性增加，有利于肾脏排泄。", "en": "**Weak base**: slightly ionized in the stomach, well absorbed in the small intestine. May partially ionize in blood (pH 7.4), increasing water solubility for renal excretion."},
    "result.pharma.analysis.amphoteric": {"zh": "**两性分子**：在不同 pH 环境下电离行为复杂，吸收部位取决于具体结构。可能需要特殊制剂（如肠溶片）来优化生物利用度。", "en": "**Amphoteric molecule**: complex ionization behavior across pH ranges; absorption site depends on specific structure. May require special formulations (e.g., enteric-coated tablets)."},

    # Pharmacology – Joint analysis
    "result.pharma.linkage.title": {"zh": "溶解度 x pKa 联动分析", "en": "Solubility × pKa Joint Analysis"},
    "result.pharma.linkage.soluble": {"zh": "**溶解度**：易溶于水，有利于溶出。", "en": "**Solubility**: Highly soluble, favors dissolution."},
    "result.pharma.linkage.moderate": {"zh": "**溶解度**：中等，可能需要辅料助溶。", "en": "**Solubility**: Moderate, may require excipients."},
    "result.pharma.linkage.poor": {"zh": "**溶解度**：较低，生物利用度可能受限。", "en": "**Solubility**: Low, bioavailability may be limited."},
    "result.pharma.linkage.pka_weak_acid": {"zh": "**pKa**：弱酸性 (pKa={val:.1f})，胃吸收好，**空腹服用**效果更佳。", "en": "**pKa**: Weak acid (pKa={val:.1f}), good gastric absorption, best taken **on an empty stomach**."},
    "result.pharma.linkage.pka_mid_acid": {"zh": "**pKa**：中等酸性 (pka={val:.1f})，全肠道吸收，对服药时间要求不高。", "en": "**pKa**: Moderate acid (pKa={val:.1f}), absorbed throughout GI tract, no strict timing requirement."},
    "result.pharma.linkage.pka_strong_base": {"zh": "**pKa**：强碱性 (pKa={val:.1f})，胃吸收差，**餐后服用**可减少胃刺激，主要在小肠吸收。", "en": "**pKa**: Strong base (pKa={val:.1f}), poor gastric absorption, take **after meals**, mainly absorbed in small intestine."},
    "result.pharma.linkage.pka_weak_base": {"zh": "**pKa**：弱碱性 (pKa={val:.1f})，小肠吸收为主，血液中有利于排泄。", "en": "**pKa**: Weak base (pKa={val:.1f}), mainly absorbed in small intestine, favors excretion in blood."},
    "result.pharma.linkage.pka_neutral": {"zh": "**pKa**：接近中性 (pKa={val:.1f})，吸收行为较复杂。", "en": "**pKa**: Near neutral (pKa={val:.1f}), absorption behavior is complex."},
    "result.pharma.linkage.combo_good": {"zh": "**综合**：高溶解度 + 胃吸收优势 = **口服生物利用度极佳**，适合做成普通片剂。", "en": "**Summary**: High solubility + gastric absorption = **excellent oral bioavailability**, suitable for standard tablets."},
    "result.pharma.linkage.combo_challenging": {"zh": "**综合**：低溶解度 + 强碱性 = **口服吸收双重挑战**，可能需要肠溶片或注射剂型。", "en": "**Summary**: Low solubility + strong base = **dual challenge for oral absorption**, may require enteric coating or injection."},
    "result.pharma.linkage.combo_acceptable": {"zh": "**综合**：高溶解度弥补了胃吸收劣势，进入小肠后吸收良好，总体生物利用度可接受。", "en": "**Summary**: High solubility compensates for poor gastric absorption, good absorption in small intestine, acceptable bioavailability."},
    "result.pharma.linkage.not_loaded": {"zh": "pKa 模型未加载，药理学分析不可用。", "en": "pKa model not loaded. Pharmacological analysis unavailable."},

    # ADME/Tox section captions
    "result.pharma.admet.absorption_caption": {
        "zh": "**吸收 (Absorption)** 决定了药物从给药部位进入血液循环的效率和程度。主要影响因素包括：分子极性 (TPSA)、脂溶性 (LogP)、电离状态 (pKa vs pH)、氢键能力、分子大小。口服药物的吸收主要发生在小肠（表面积大、血流丰富），部分酸性药物可在胃中开始吸收。",
        "en": "**Absorption** determines the efficiency of drug entry from the administration site into the bloodstream. Key factors: polarity (TPSA), lipophilicity (LogP), ionization state (pKa vs pH), H-bond capacity, and molecular size. Oral absorption primarily occurs in the small intestine (large surface area, rich blood flow); some acidic drugs begin absorption in the stomach.",
    },
    "result.pharma.admet.distribution_caption": {
        "zh": "**分布 (Distribution)** 描述药物从血液进入组织和器官的过程。\n- **表观分布容积 (Vd)**：Vd 越大，药物越倾向于分布到组织；Vd 小则主要留在血浆中\n- **血浆蛋白结合率**：高结合率 = 药物储备在血液中缓慢释放；低结合率 = 游离药物多、作用快但排泄也快\n- **血脑屏障 (BBB)**：低 TPSA + 适中 LogP 的分子更容易进入中枢神经系统",
        "en": "**Distribution** describes the movement of drugs from blood into tissues and organs.\n- **Volume of Distribution (Vd)**: Higher Vd = drug favors tissue distribution; lower Vd = mainly stays in plasma\n- **Plasma Protein Binding**: High binding = slow-release drug reservoir; low binding = more free drug, faster action and excretion\n- **Blood-Brain Barrier (BBB)**: Low TPSA + moderate LogP favors CNS penetration",
    },
    "result.pharma.admet.metabolism_caption": {
        "zh": "**代谢 (Metabolism)** 主要在肝脏进行，分为两相：\n- **I 相代谢**（氧化/还原/水解）：主要由 CYP450 酶系催化，引入或暴露极性基团\n- **II 相代谢**（结合反应）：将葡萄糖醛酸、硫酸、氨基酸等连接到分子上，增加水溶性便于排泄\n- CYP3A4 代谢约 50% 的临床药物；CYP2D6 有显著的基因多态性（人群差异大）",
        "en": "**Metabolism** primarily occurs in the liver in two phases:\n- **Phase I** (oxidation/reduction/hydrolysis): catalyzed mainly by CYP450 enzymes, introducing or exposing polar groups\n- **Phase II** (conjugation): attaches glucuronic acid, sulfate, amino acids, etc. to increase water solubility for excretion\n- CYP3A4 metabolizes ~50% of clinical drugs; CYP2D6 has significant genetic polymorphism",
    },
    "result.pharma.admet.excretion_caption": {
        "zh": "**排泄 (Excretion)** 是药物及其代谢物从体内清除的过程：\n- **肾脏排泄**：分子量 < 350 Da 且极性适中的分子主要通过肾小球滤过进入尿液\n- **肝胆排泄**：分子量 > 500 Da 或高度亲脂的分子倾向于通过胆汁排入肠道\n- 肾小管重吸收会使亲脂性分子重新进入血液，延长药物半衰期",
        "en": "**Excretion** is the process of eliminating drugs and their metabolites:\n- **Renal excretion**: molecules with MW < 350 Da and moderate polarity are filtered through glomeruli into urine\n- **Hepatobiliary excretion**: molecules with MW > 500 Da or highly lipophilic tend to be excreted via bile\n- Tubular reabsorption can return lipophilic molecules to the blood, prolonging half-life",
    },
    "result.pharma.admet.toxicity_caption": {
        "zh": "**毒性 (Toxicity)** 评估基于结构警报 (Structural Alerts) — 某些官能团或子结构在历史上与毒性事件相关：\n- 以上分析仅基于<b>结构特征</b>，不代表实际毒性——毒性受剂量、代谢、个体差异等多因素影响\n- 结构警报是药物设计初筛的重要工具，但存在许多假阳性——含警报结构的药物仍可能安全上市\n- 实际毒性需要通过 Ames 试验、hERG 测试、动物实验和临床试验逐级验证",
        "en": "**Toxicity** assessment is based on Structural Alerts — functional groups or substructures historically associated with toxicity:\n- This analysis is based on <b>structural features only</b>, not actual toxicity — toxicity depends on dose, metabolism, individual variation, etc.\n- Structural alerts are important drug design screening tools, but have many false positives — drugs with alert structures can still be safe\n- Actual toxicity requires Ames test, hERG assay, animal studies, and clinical trials for validation",
    },

    # GNN Explanation
    "result.gnn.title": {"zh": "GNN Explainability — 原子/边注意力分析", "en": "GNN Explainability — Atom/Bond Attention Analysis"},
    "result.gnn.top_bonds_title": {"zh": "**Top 最重要的化学键**", "en": "**Top most important bonds**"},
    "result.gnn.top_bonds_importance": {"zh": "重要性: {imp:.3f} ({pct:.0f}%)", "en": "Importance: {imp:.3f} ({pct:.0f}%)"},
    "result.gnn.atom_title": {"zh": "**原子重要性（基于相连键汇总）**", "en": "**Atom Importance (aggregated from bond importance)**"},
    "result.gnn.feature_title": {"zh": "**原子特征重要性**", "en": "**Atom Feature Importance**"},
    "result.gnn.img_caption": {"zh": "关键化学键高亮（暖色 = 更重要）", "en": "Bond highlighting (warmer = more important)"},
    "result.gnn.spinner": {"zh": "正在运行 GNNExplainer 分析（约 10-30 秒）...", "en": "Running GNNExplainer analysis (~10-30 seconds)..."},
    "result.gnn.fail": {"zh": "GNN 解释生成失败: {err}", "en": "GNN explanation failed: {err}"},
    "result.gnn.model_unavailable": {"zh": "GNN 解释生成失败（模型未加载或分子无法解析）", "en": "GNN explanation failed (model not loaded or molecule invalid)"},
    "result.gnn.no_bonds": {"zh": "该分子无非氢键，无法进行边重要性分析", "en": "This molecule has no non-hydrogen bonds, cannot analyze bond importance"},
    "result.gnn.no_mol": {"zh": "无可解释的分子", "en": "No molecule to explain"},
    "result.gnn.elapsed_text": {"zh": "分析耗时：**{t:.1f}s**", "en": "Analysis time: **{t:.1f}s**"},
    "result.gnn.how_to_read_title": {"zh": "如何读懂 GNN 解释", "en": "How to read GNN explanations"},
    "result.gnn.how_to_read_html": {
        "zh": "<b>GNNExplainer</b> 通过学习每条化学键的\"重要性权重\"来解释模型预测：<br><br>- <b>暖色高亮边</b> = 模型认为这条化学键及其连接的原子对判断溶解度最重要<br>- <b>冷色/暗淡边</b> = 该化学键对预测影响较小<br><br><b>技术原理</b>：<br>1. 将每条边（化学键）初始化为一个可学习的参数（0~1 的权重）<br>2. 用带权重的图做前向传播，只保留高权重的边<br>3. 优化目标是\"带权重的预测 ≈ 原始预测\"，同时尽量保留最少的边<br>4. 最终权重 = 这条边对模型预测的贡献大小<br><br><b>注意</b>：<br>- GNNExplainer 是近似方法，不同运行可能有微小差异<br>- 重要性高的边 ≠ 该键在化学反应中更容易断裂<br>- 建议结合 SHAP（RF 模式）和 pKa 化学因素分解共同解读",
        "en": "<b>GNNExplainer</b> explains model predictions by learning \"importance weights\" for each chemical bond:<br><br>- <b>Warm-colored edges</b> = the model considers these bonds and their atoms most important for solubility prediction<br>- <b>Cool/dim edges</b> = these bonds have less impact on the prediction<br><br><b>How it works</b>:<br>1. Each edge (bond) starts with a learnable weight (0~1)<br>2. Forward pass uses the weighted graph, keeping only high-weight edges<br>3. Optimization target: \"weighted prediction ≈ original prediction\" while keeping minimal edges<br>4. Final weight = importance of that edge to the model's prediction<br><br><b>Note</b>:<br>- GNNExplainer is an approximation; slight variations across runs are normal<br>- High importance ≠ the bond is more chemically reactive<br>- Combine with SHAP (RF mode) and pKa factor decomposition for full interpretation",
    },
    "result.gnn.how_to_read_tech": {
        "zh": "**原子特征重要性** 显示 37 维原子特征向量中哪些维度最重要：<br>- 原子类型（是 C、N、O 还是其他元素）<br>- 连接度（连了几个原子）<br>- 杂化方式（sp²、sp³）<br>- 芳香性、手性、是否在环上等",
        "en": "**Atom Feature Importance** shows which dimensions of the 37-dim atom feature vector matter most:<br>- Atom type (C, N, O, or other)<br>- Degree (how many atoms it's connected to)<br>- Hybridization (sp², sp³)<br>- Aromaticity, chirality, ring membership",
    },
    "result.gnn.run_btn": {"zh": "运行 GNNExplainer 分析", "en": "Run GNNExplainer Analysis"},

    # ── Web UI additions (React frontend, Phase 3) ──
    "result.preview.spin_start": {"zh": "开始旋转", "en": "Start rotation"},
    "result.preview.spin_stop": {"zh": "停止旋转", "en": "Stop rotation"},
    "result.ood.out_of_range_label": {"zh": "超出训练数据范围", "en": "Outside training range"},
    "result.ood.extreme_label": {"zh": "极端偏离 (|z| > 3)", "en": "Extreme deviation (|z| > 3)"},
    "result.solubility.ensemble_weights": {"zh": "权重 RF 0.45 · GNN 0.55", "en": "Weights RF 0.45 · GNN 0.55"},
    "common.loading": {"zh": "加载中...", "en": "Loading..."},
    "common.retry": {"zh": "重试", "en": "Retry"},

    # ── Web UI additions (React frontend, Phase 4) ──
    "result.preview.model3d_loading": {"zh": "正在加载 3D 查看器...", "en": "Loading 3D viewer..."},
}

# Add zh-only entries as copies of en for keys that are English-only
for _key, _val in list(_ALL.items()):
    if "zh" not in _val and "en" in _val:
        _val["zh"] = _val["en"]
