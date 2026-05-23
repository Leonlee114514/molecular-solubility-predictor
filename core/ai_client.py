"""
DisSolve - Kimi AI chemistry explanation client.
"""

import os
import numpy as np
import openai
import streamlit as st

def _get_api_key():
    """Read Kimi API key from Streamlit Secrets or .env file."""
    return st.secrets.get("KIMI_API_KEY") or os.getenv("KIMI_API_KEY")

def _build_explanation_prompt(smiles, prediction, features, shap_features=None, shap_values=None, pka_value=None, pka_type=None):
    """Build the Kimi API prompt for chemistry explanation. Returns (prompt_str, solubility_level)."""

    if prediction > 0:
        solubility_level = "易溶于水"
        solubility_desc = "logS > 0，属于高溶解度"
    elif prediction > -2:
        solubility_level = "中等溶解"
        solubility_desc = "-2 < logS <= 0，属于中等溶解度"
    else:
        solubility_level = "难溶于水"
        solubility_desc = "logS <= -2，属于低溶解度"

    shap_text = ""
    if shap_features and shap_values and len(shap_features) == len(shap_values):
        abs_vals = np.abs(np.array(shap_values))
        sorted_idx = np.argsort(abs_vals)[::-1][:5]
        top_features = [shap_features[i] for i in sorted_idx]
        top_vals = [shap_values[i] for i in sorted_idx]
        shap_lines = []
        for name, val in zip(top_features, top_vals):
            direction = "推动易溶" if val > 0 else "推动难溶"
            shap_lines.append(f"- {name}: 贡献值 {val:+.3f}（{direction}）")
        shap_text = "\n".join(shap_lines)

    pka_section = ""
    pka_task = ""
    if pka_value is not None and pka_type is not None:
        if pka_type == "acid":
            pka_label = "酸性分子"
            pka_desc_full = f"pKa = {pka_value:.2f} (< 5)，属于酸性分子。在酸性环境（如胃，pH ~1.5）中主要以分子态存在，脂溶性较高，容易被胃黏膜吸收。"
            ionization_desc = "在生理 pH 范围内，该分子倾向于释放质子 (H+)，形成共轭碱。"
        elif pka_type == "base":
            pka_label = "碱性分子"
            pka_desc_full = f"pKa = {pka_value:.2f} (> 9)，属于碱性分子。在碱性环境中主要以分子态存在，在胃中容易电离，主要在小肠吸收。"
            ionization_desc = "在生理 pH 范围内，该分子倾向于结合质子 (H+)，形成共轭酸。"
        else:
            pka_label = "两性/中性分子"
            pka_desc_full = f"pKa = {pka_value:.2f} (5-9 之间)，属于两性或中性分子。电离行为随环境 pH 变化剧烈，在不同生理部位的存在形态差异大。"
            ionization_desc = "该分子既可能释放也可能结合质子，具体取决于所处环境的 pH。"

        pka_section = f"""【pKa 与电离行为分析】
- 预测 pKa: {pka_value:.2f}
- 酸碱性判定: {pka_label}
- 电离特征: {ionization_desc}
- 生理意义: {pka_desc_full}

【溶解度 x pKa 联动提示】
溶解度 (logS) 和 pKa 共同决定药物在体内的吸收行为：
- 分子态（非电离）脂溶性高，易穿透细胞膜被吸收
- 离子态水溶性好，有利于在血液中运输和肾脏排泄
- 当前分子：logS = {prediction:.2f}（{solubility_level}），pKa = {pka_value:.2f}（{pka_label}）"""

        pka_task = f"""5. **pKa 结构化学深度解析**（4-5句话）：
   - 从 SMILES 识别该分子的**可电离基团**（如 -COOH、脂肪胺、芳香胺、酚羟基、杂环氮等），并指出其直接连接的化学环境。
   - 用**电子效应**解释该 pKa = {pka_value:.2f} 的合理性：附近是否有吸电子基团（-I, -M）拉低 pKa / 推电子基团（+I, +M）升高 pKa？是否有共轭稳定化/去稳定化？是否存在分子内氢键或空间位阻影响质子转移？
   - 简要说明该分子在胃 (pH 1.5)、小肠 (pH 6.8)、血液 (pH 7.4) 中的**电离状态趋势**（以分子态比例高低描述即可，不做精确计算）。
   - 联系溶解度分析：该分子的电离状态如何与其亲水/疏水基团分布共同影响体内吸收与排泄。"""

    prompt = f"""你是一位结构化学专家，擅长从分子的 SMILES 表示和理化性质数据中深度剖析其溶解度与电离行为的结构根源。请围绕**分子骨架、官能团、电子效应、空间构型**展开细致分析，避免泛泛而谈的科普介绍。

分子 SMILES: {smiles}
模型预测的水溶解度 (logS): {prediction:.2f}

【分子基本性质】
- 分子量: {features['MolWt']:.1f} g/mol
- 极性表面积 (TPSA): {features['TPSA']:.1f} A2
- 氢键供体数: {features['NumHDonors']}
- 氢键受体数: {features['NumHAcceptors']}
- 脂水分配系数 (LogP): {features['LogP']:.2f}
- 可旋转键数: {features['NumRotatableBonds']}
- 芳香环数: {features['NumAromaticRings']}
- 脂肪环数: {features['NumAliphaticRings']}

【SHAP 模型可解释性分析 - 影响溶解度预测的关键结构特征】
{shap_text if shap_text else "（SHAP 分析暂不可用）"}

【已由程序精确判定的溶解度结论（严禁修改或重新判断）】
该分子的预测溶解度 logS = {prediction:.3f}，判定结果为：**{solubility_level}**。
判定依据：{solubility_desc}。
重要：上述结论已由程序精确计算得出，你只需在回答中直接复述，不可重新判断或做数值比较。
{pka_section}

请用中文回答，严格按以下段落组织，重点放在**结构解析**上：

1. **溶解度结论**（1句话）：直接复述——该分子属于「{solubility_level}」。

2. **分子骨架与官能团识别**（3-4句话）：
   - 从 SMILES 字符串解析该分子的**核心骨架**（如苯环、甾体、糖类、肽链、脂肪链等）。
   - 列出分子中存在的**主要官能团**（如羟基 -OH、羧基 -COOH、氨基 -NH2、酰胺 -CONH-、醚键 -O-、酯基 -COOR、卤素、硝基、磺酸基、杂环氮等）。
   - 指出是否存在**可电离基团**及其直接连接的化学环境。
   - 描述分子的**整体构型特征**（如线性/分支/稠环/大环、刚性 vs 柔性、亲水面与疏水面的空间分布趋势）。

3. **结构-溶解度深度解析**（4-5句话）：
   - 结合具体官能团解释：哪些基团**推动水溶**（如羟基、羧基、氨基形成氢键），哪些**阻碍水溶**（如长烷基链、大芳香疏水面）。
   - 结合 SHAP 分析结果，说明模型最关注的结构特征如何与该分子的实际官能团组成对应。
   - 若分子同时含有亲水与疏水基团，分析二者的**相对比例与空间布局**如何决定整体溶解度。
   - 提及**分子间相互作用**：该分子与水之间能形成多少氢键网络，疏水部分是否导致水分子有序化（疏水效应）。

4. **SHAP 关键特征与结构对应**（2-3句话）：
   - 引用 SHAP 贡献值最高的 1-2 个特征的具体数值。
   - 明确指出这些特征在分子结构上的**物理对应物**。

{pka_task}

要求:
- **以结构化学为核心**，避免空泛的科普描述和简单的生活类比。
- 第2段必须基于 SMILES 识别出**至少2个具体官能团**和**骨架类型**。
- 第3段必须引用分子性质数据（LogP、TPSA、H-Bond 数目等）和 SHAP 贡献值。
- 语言准确但不过度学术，适合具备基础有机化学知识的高中生理解。"""
    return prompt, solubility_level

def explain_with_kimi(smiles, prediction, features, shap_features=None, shap_values=None, pka_value=None, pka_type=None):
    """Generate an AI-powered chemistry explanation using Kimi (Moonshot AI).

    Returns a plain-text explanation string, or an error message on failure.
    """
    api_key = _get_api_key()
    if not api_key:
        return "未配置 Kimi API Key。请在 .env 文件中写入：KIMI_API_KEY=sk-你的密钥"
    prompt, _ = _build_explanation_prompt(smiles, prediction, features, shap_features, shap_values, pka_value, pka_type)

    try:
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1"
        )
        response = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[
                {"role": "system", "content": "你是一位结构化学与药物化学专家。你的核心能力是从分子的 SMILES 表示、理化性质和机器学习特征贡献中，深度解析官能团组成、骨架特征、电子效应与分子性质之间的因果链条。你说话简洁、精准，优先从分子结构切入，避免空泛科普。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=1200
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI 解释暂时不可用: {e}"
