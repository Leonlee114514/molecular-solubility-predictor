"""
DisSolve - Kimi AI chemistry explanation client.
Optimized prompt with caching, reduced temperature, and cleaner structure.
"""

import os
import numpy as np
import openai
import streamlit as st


def _get_api_key():
    """Read Kimi API key from Streamlit Secrets or .env file."""
    return st.secrets.get("KIMI_API_KEY") or os.getenv("KIMI_API_KEY")


_SYSTEM_PROMPT = (
    "你是一位结构化学与药物化学专家。你的核心能力是从分子的 SMILES 表示、"
    "理化性质和 SHAP 特征贡献中，解析官能团、骨架特征、电子效应"
    "与溶解度/电离性质之间的因果链条。回答精准简洁，每段突出核心化学推理，"
    "不做重复性科普。不确定的结构信息不要编造。"
)


def _build_explanation_prompt(smiles, prediction, features,
                              shap_features=None, shap_values=None,
                              pka_value=None, pka_type=None):
    """Build a concise, structured prompt for chemistry explanation.

    Returns (prompt_str, solubility_level).
    """
    # ── Solubility classification ──
    if prediction > 0:
        solubility_level = "易溶于水"
        solubility_desc = "logS > 0，属于高溶解度"
    elif prediction > -2:
        solubility_level = "中等溶解"
        solubility_desc = "-2 < logS ≤ 0，属于中等溶解度"
    else:
        solubility_level = "难溶于水"
        solubility_desc = "logS ≤ -2，属于低溶解度"

    # ── SHAP section ──
    shap_block = ""
    shap_instruction = ""
    if shap_features and shap_values and len(shap_features) == len(shap_values):
        abs_vals = np.abs(np.array(shap_values))
        sorted_idx = np.argsort(abs_vals)[::-1][:3]
        top_features = [shap_features[i] for i in sorted_idx]
        top_vals = [shap_values[i] for i in sorted_idx]
        shap_lines = []
        for name, val in zip(top_features, top_vals):
            direction = "推动易溶" if val > 0 else "推动难溶"
            shap_lines.append(f"- {name}: {val:+.3f}（{direction}）")
        shap_block = "【SHAP 关键特征】（影响最大的前 3 个）\n" + "\n".join(shap_lines)
        shap_instruction = (
            "4. 【SHAP 深入解释】2-3句话：对有最大贡献的特征，解释为什么该特征推动或抑制溶解"
            "（例如：LogP 贡献大是因为长烷基链 → 疏水 → 推动难溶；"
            "TPSA 贡献大是因为羟基多 → 氢键 → 推动易溶）。"
        )
    else:
        shap_block = "（SHAP 解释暂不可用，跳过第 4 段）"
        shap_instruction = ""

    # ── pKa section ──
    pka_block = ""
    pka_instruction = ""
    if pka_value is not None and pka_type is not None:
        if pka_type == "acid":
            pka_label = "酸性"
            ionization_desc = (
                "倾向于释放质子 (H⁺)，在胃 (pH≈1.5) 中以分子态为主易吸收，"
                "在血液 (pH≈7.4) 中以离子态为主利于排泄。"
            )
        elif pka_type == "base":
            pka_label = "碱性"
            ionization_desc = (
                "倾向于结合质子 (H⁺)，在胃中易电离，主要在小肠 (pH≈6.8) 吸收，"
                "在血液中以离子态为主。"
            )
        else:
            pka_label = "两性/中性"
            ionization_desc = (
                "在不同 pH 环境下既可释放也可结合质子，吸收行为复杂，"
                "随给药部位环境 pH 变化。"
            )
        pka_block = (
            f"【pKa 与电离】pKa = {pka_value:.2f}（{pka_label}）\n"
            f"{ionization_desc}"
        )
        pka_instruction = (
            "5. 【pKa 结构分析】2-3句话：从电子效应解释当前 pKa 值的合理性"
            "（吸/推电子基团的影响、共轭稳定化、分子内氢键等），"
            "并简述该分子在不同生理 pH 环境下的电离趋势。"
        )

    # ── Build prompt ──
    prompt = (
        f"分子 SMILES: {smiles}\n"
        f"预测 logS: {prediction:.2f} → {solubility_level}（{solubility_desc}）\n"
        "\n"
        "【理化性质】\n"
        f"分子量: {features['MolWt']:.1f} | LogP: {features['LogP']:.2f} | "
        f"TPSA: {features['TPSA']:.1f} Å²\n"
        f"氢键供体: {features['NumHDonors']} | 氢键受体: {features['NumHAcceptors']}\n"
        f"可旋转键: {features['NumRotatableBonds']} | "
        f"芳香环: {features['NumAromaticRings']} | 脂肪环: {features['NumAliphaticRings']}\n"
        "\n"
        f"{shap_block}\n"
        "\n"
        f"{pka_block}\n"
        "\n"
        "请按以下结构回答（严格遵守每段长度限制）：\n"
        "\n"
        f"1. 【溶解度结论】1句话：该分子属于「{solubility_level}」（logS = {prediction:.2f}），"
        f"由模型精确计算得出。\n"
        "\n"
        "2. 【骨架与官能团】2-3句话：从 SMILES 解析核心骨架类型，"
        "列举至少2个具体官能团及其连接位置，指出是否存在可电离基团。\n"
        "\n"
        "3. 【结构-溶解度解释】2-3句话：哪些基团促进或抑制水溶，"
        "结合 LogP、TPSA、氢键数目说明。SHAP 关键特征对应到分子的哪些实际结构。\n"
        "\n"
        f"{shap_instruction}\n"
        "\n"
        f"{pka_instruction}\n"
        "\n"
        "要求：\n"
        "- 以结构化学为核心但不过度学术，适合学过基础有机化学的大学生阅读\n"
        "- 必须引用至少1个分子性质数据（LogP/TPSA/氢键数等）支持论点\n"
        "- 每段不超过 3 句话\n"
        "- 不确定的结构特征不要编造"
    )
    return prompt, solubility_level


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_kimi_explain(smiles, prediction, features_tuple, shap_tuple,
                         pka_value, pka_type):
    """Cached Kimi API call, keyed by molecular data. 24h TTL."""
    # Rebuild mutable structures from hashable tuples
    features = dict(features_tuple)
    shap_features = list(shap_tuple[0]) if shap_tuple and shap_tuple[0] else None
    shap_values = list(shap_tuple[1]) if shap_tuple and shap_tuple[1] else None

    prompt, _ = _build_explanation_prompt(
        smiles, prediction, features,
        shap_features=shap_features, shap_values=shap_values,
        pka_value=pka_value, pka_type=pka_type,
    )

    api_key = _get_api_key()
    if not api_key:
        return "未配置 Kimi API Key。请在 .env 文件中写入：KIMI_API_KEY=sk-你的密钥"

    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://api.moonshot.cn/v1",
    )
    response = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.35,
        max_tokens=800,
    )
    return response.choices[0].message.content


def explain_with_kimi(smiles, prediction, features,
                      shap_features=None, shap_values=None,
                      pka_value=None, pka_type=None):
    """Generate an AI-powered chemistry explanation using Kimi (Moonshot AI).

    Results are cached for 24 hours per unique molecular input.
    Returns a plain-text explanation string, or an error message on failure.
    """
    api_key = _get_api_key()
    if not api_key:
        return "未配置 Kimi API Key。请在 .env 文件中写入：KIMI_API_KEY=sk-你的密钥"

    # Convert to hashable types for caching
    features_tuple = tuple(sorted(features.items()))
    shap_tuple = (
        tuple(shap_features) if shap_features else None,
        tuple(shap_values) if shap_values else None,
    )

    try:
        return _cached_kimi_explain(
            smiles, prediction, features_tuple, shap_tuple,
            pka_value, pka_type,
        )
    except Exception as e:
        return f"AI 解释暂时不可用: {e}"
