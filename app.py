"""
【分子溶解度预测 - Orbita Theme v1】
保留原有全部功能逻辑，仅改造视觉层为深空宇宙风格
"""

import streamlit as st
import numpy as np
import joblib
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, AllChem, Draw

import openai
from dotenv import load_dotenv
import os

# 加载环境变量（本地开发用）
load_dotenv()

# 优先读取 Streamlit Secrets（Cloud 部署），其次读取 .env（本地开发）
KIMI_API_KEY = st.secrets.get("KIMI_API_KEY") or os.getenv("KIMI_API_KEY")

import streamlit.components.v1 as components

# ========== 本地分子库（100+ 分子，零网络依赖）==========
MOLECULE_DB = {
    "(自定义输入)": "",
    
    # === 基础有机分子 ===
    "乙醇 Ethanol": "CCO",
    "甲醇 Methanol": "CO",
    "异丙醇 Isopropanol": "CC(C)O",
    "乙二醇 Ethylene glycol": "OCCO",
    "甘油 Glycerol": "OCC(O)CO",
    "苯 Benzene": "c1ccccc1",
    "甲苯 Toluene": "Cc1ccccc1",
    "苯酚 Phenol": "Oc1ccccc1",
    "苯甲酸 Benzoic acid": "O=C(O)c1ccccc1",
    "苯乙烯 Styrene": "C=Cc1ccccc1",
    "环己烷 Cyclohexane": "C1CCCCC1",
    "己烷 Hexane": "CCCCCC",
    "辛烷 Octane": "CCCCCCCC",
    
    # === 溶剂与工业 ===
    "乙酸乙酯 Ethyl acetate": "CCOC(=O)C",
    "丙酮 Acetone": "CC(=O)C",
    "乙醚 Diethyl ether": "CCOCC",
    "四氢呋喃 THF": "C1CCOC1",
    "氯仿 Chloroform": "C(Cl)(Cl)Cl",
    "四氯化碳 CCl4": "C(Cl)(Cl)(Cl)Cl",
    "甲醛 Formaldehyde": "C=O",
    "醋酸 Acetic acid": "CC(=O)O",
    "柠檬酸 Citric acid": "C(C(=O)O)C(CC(=O)O)(C(=O)O)O",
    
    # === 生物化学基础 ===
    "尿素 Urea": "NC(=O)N",
    "甘氨酸 Glycine": "NCC(=O)O",
    "丙氨酸 Alanine": "CC(N)C(=O)O",
    "缬氨酸 Valine": "CC(C)C(N)C(=O)O",
    "亮氨酸 Leucine": "CC(C)CC(N)C(=O)O",
    "苯丙氨酸 Phenylalanine": "NC(Cc1ccccc1)C(=O)O",
    "色氨酸 Tryptophan": "NC(Cc1c[nH]c2ccccc12)C(=O)O",
    "酪氨酸 Tyrosine": "NC(Cc1ccc(O)cc1)C(=O)O",
    "谷氨酸 Glutamic acid": "NC(CCC(=O)O)C(=O)O",
    
    # === 糖类 ===
    "葡萄糖 Glucose": "C(C1C(C(C(C(O1)O)O)O)O)O",
    "果糖 Fructose": "C(C1C(C(CO1)(O)O)O)O",
    "蔗糖 Sucrose": "C(C1C(C(C(C(O1)O)O)O)O)OC2OC(C(C(C2O)O)O)CO",
    "乳糖 Lactose": "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)OC[C@@H]2[C@H]([C@@H]([C@H]([C@H](O2)O)O)O)O",
    
    # === 维生素 ===
    "维生素A Vitamin A": "CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CCO)C)C",
    "维生素B2 Riboflavin": "Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(C[C@H](O)[C@H](O)[C@H](O)CO)c2cc1C",
    "维生素B3 Niacin": "O=C(O)c1cccnc1",
    "维生素B6 Pyridoxine": "Cc1ncc(CO)c(CO)c1O",
    "维生素B9 Folic acid": "C1=CC(=CC2=C1C(=NC(=N2)N)N)CNC(=O)NC(CCC(=O)O)C(=O)O",
    "维生素C Ascorbic acid": "C([C@@H]([C@@H]1C(=C(C(=O)O1)O)O)O)O",
    "维生素E Tocopherol": "Cc1c(C)c2C(=C(C1C)CC[C@@](C)(CCCC(C)C)O)CCCC2(C)C",
    
    # === 激素 ===
    "睾酮 Testosterone": "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@@]43C)[C@@H]1CC[C@@H]2O",
    "雌二醇 Estradiol": "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CC[C@@H]2O",
    "孕酮 Progesterone": "CC(=O)[C@H]1CC[C@H]2[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@H]3CC[C@]12C",
    "皮质醇 Cortisol": "C[C@]12CCC(=O)C=C1CC[C@@H]3[C@@H]2[C@H](C[C@]4([C@H]3CC[C@@H]4C(=O)CO)C)O",
    "胆固醇 Cholesterol": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",
    
    # === 常见药物 ===
    "阿司匹林 Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "布洛芬 Ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "萘普生 Naproxen": "COc1ccc2cc(C(C)C(=O)O)ccc2c1",
    "酮洛芬 Ketoprofen": "CC(C(=O)c1ccccc1)c2ccc(C(=O)O)cc2",
    "双氯芬酸 Diclofenac": "O=C(O)Cc1ccccc1Nc2c(Cl)cccc2Cl",
    "对乙酰氨基酚 Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "可待因 Codeine": "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
    "吗啡 Morphine": "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
    "咖啡因 Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "茶碱 Theophylline": "Cn1c2c(c(=O)n(C)c1=O)NC=N2",
    "青霉素G Penicillin G": "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",
    "阿莫西林 Amoxicillin": "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O",
    "四环素 Tetracycline": "C[C@]1(c2cccc(O)c2)C(=O)C(C(=O)NC(C(=O)O)C3CCCCC3)=C(O)C(=O)N1C",
    "多西环素 Doxycycline": "C[C@H]1c2cccc(O)c2C(=O)C3=C(O)[C@](C(=O)NC4C(=O)NC(C(=O)O)C5CCCCC54)(O)C(=O)[C@@H](O)[C@@H]3[C@@H]1C",
    "环丙沙星 Ciprofloxacin": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "甲硝唑 Metronidazole": "Cc1ncc([N+](=O)[O-])n1CCO",
    
    # === 心血管/代谢 ===
    "二甲双胍 Metformin": "CN(C)C(=N)N=C(N)N",
    "阿托伐他汀 Atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
    "辛伐他汀 Simvastatin": "CCC(C)(C)C(=O)O[C@H]1C[C@@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@H]21",
    "硝苯地平 Nifedipine": "COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1[N+](=O)[O-]",
    "氨氯地平 Amlodipine": "CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c2ccccc2Cl",
    "地高辛 Digoxin": "C[C@H]1O[C@@H](O[C@H]2CC[C@@]3(C)[C@@H](CC[C@@H]4[C@@H]3CC[C@]3(C)[C@@H](C5=CC(=O)OC5)CC[C@]43O)C2)C[C@H](O)[C@@H]1O",
    
    # === 精神神经 ===
    "地西泮 Diazepam": "CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc31",
    "劳拉西泮 Lorazepam": "O=C1CN=C(c2ccccc2)c3cc(Cl)ccc3N1",
    "阿普唑仑 Alprazolam": "Cc1nnc2n1-c3ccc(Cl)cc3C(c4ccccc4)=NC2",
    "氟西汀 Fluoxetine": "CNCCC(c1ccc(OC)cc1)c2ccccc2",
    "舍曲林 Sertraline": "CNC1CCC(c2ccc(Cl)cc2)c3cccnc31",
    "奥氮平 Olanzapine": "CN1CCN(C2=Nc3ccccc3Sc4ccc(Cl)cc24)CC1",
    
    # === 消化系统 ===
    "奥美拉唑 Omeprazole": "COc1ccc2nc(S(=O)Cc3ncc(C)c(OC)c3C)[nH]c2c1",
    "雷尼替丁 Ranitidine": "CN(C)CCNC(=O)CSc1ccc(CN/C=C/[N+](=O)[O-])cc1",
    "西咪替丁 Cimetidine": "CC1=C(NC=N1)CSCCN/C(=N/C#N)/NC",
    
    # === 抗过敏/呼吸 ===
    "氯雷他定 Loratadine": "CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc3cccnc32)CC1",
    "西替利嗪 Cetirizine": "O=C(O)C(Cc1ccc(cc1)Cl)CN2CCC(CC2)C(c3ccccc3)c4ccc(Cl)cc4",
    "沙丁胺醇 Salbutamol": "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
    
    # === 抗肿瘤 ===
    "甲氨蝶呤 Methotrexate": "CN(Cc1cnc2nc(N)nc(O)c2n1)c3ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc3",
    "5-氟尿嘧啶 5-FU": "O=c1[nH]cc(F)c(=O)[nH]1",
    "紫杉醇 Paclitaxel": "CC(=O)OC1=C2C(C)[C@@H](OC(=O)C(O)C(NC(=O)c3ccccc3)c3ccccc3)C[C@@](O)(C(=O)C(=O)C4C5COC(=O)C5C(OC(C)=O)C4C2(C)C)C1(C)C",
    "顺铂 Cisplatin": "N[P+](N)(Cl)Cl",
    
    # === 天然产物 ===
    "青蒿素 Artemisinin": "C[C@@H]1CC[C@@H]2[C@@H](C)C3OC(=O)O[C@@H]3C[C@]2(C)O1",
    "白藜芦醇 Resveratrol": "Oc1ccc(C=Cc2cc(O)cc(O)c2)cc1",
    "姜黄素 Curcumin": "COc1cc(C=CC(=O)C=Cc2ccc(O)c(OC)c2)ccc1O",
    "辣椒素 Capsaicin": "COc1cc(CNC(=O)CCCC/C=C/C(C)C)ccc1O",
    "薄荷醇 Menthol": "CC(C)C1CCC(C)CC1O",
    "樟脑 Camphor": "CC12CCC(CC1=O)C2(C)C",
    "香兰素 Vanillin": "COc1cc(C=O)ccc1O",
    "丁香酚 Eugenol": "C=CCc1cc(OC)c(O)cc1",
    
    # === 环境污染物 ===
    "萘 Naphthalene": "c1ccc2ccccc2c1",
    "蒽 Anthracene": "c1ccc2cc3ccccc3cc2c1",
    "菲 Phenanthrene": "c1ccc2c(c1)c3ccccc3cc2",
    "芘 Pyrene": "c1cc2ccc3cccc4ccc(c1)c2c34",
    "苯并芘 Benzo[a]pyrene": "c1ccc2c(c1)cc3ccc4cccc5ccc2c3c45",
    "DDT": "Clc1ccc(C(c2ccc(Cl)cc2)C(Cl)(Cl)Cl)cc1",
    "双酚A BPA": "CC(C)(c1ccc(O)cc1)c2ccc(O)cc2",
    "三聚氰胺 Melamine": "Nc1nc(N)nc(N)n1",
    
    # === 复杂分子 ===
    "三氯蔗糖 Sucralose": "C[C@@H]1O[C@@H](O[C@H]2O[C@H](CCl)[C@@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@@H](O)[C@H]1Cl",
    "胰岛素片段 Insulin (simplified)": "NCCCCC(N)C(=O)N",
}

# 构建本地搜索索引
SEARCH_INDEX = {}
for display_name, smiles in MOLECULE_DB.items():
    SEARCH_INDEX[display_name.lower()] = smiles
    parts = display_name.split()
    for part in parts:
        clean = part.strip().lower()
        if len(clean) > 1:
            SEARCH_INDEX[clean] = smiles

# ========== 保留原有 pubchem_final.py 全部代码（原样复制）==========
import requests
import urllib.parse
import time
import json

CACHE_FILE = "pubchem_cache.json"
pubchem_cache = {}

def load_cache():
    global pubchem_cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                pubchem_cache = json.load(f)
        except (json.JSONDecodeError, OSError, IOError):
            pubchem_cache = {}

def save_cache():
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(pubchem_cache, f, ensure_ascii=False, indent=2)
    except (OSError, IOError):
        pass

load_cache()

def search_pubchem_final(name, max_retries=3):
    """
    最终版 PubChem 搜索
    """
    if not name or not name.strip():
        return None, "名称不能为空"
    
    name_clean = name.strip()
    name_lower = name_clean.lower()
    
    if name_lower in pubchem_cache:
        return pubchem_cache[name_lower], "success (cached)"
    
    time.sleep(1.2)
    encoded = urllib.parse.quote(name_clean)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/CanonicalSMILES/JSON"
    
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=20, verify=False)
            if r.status_code == 200:
                data = r.json()
                if 'Fault' in data:
                    fault = data.get('Fault', {}).get('Message', '')
                    if 'NotFound' in fault or 'not found' in fault.lower():
                        return None, "PubChem 未找到该化合物"
                    time.sleep(1.0 * (attempt + 1))
                    continue
                props = data.get('PropertyTable', {}).get('Properties', [])
                if props:
                    smiles = props[0].get('CanonicalSMILES') or props[0].get('IsomericSMILES')
                    if smiles and smiles.strip():
                        result = smiles.strip()
                        pubchem_cache[name_lower] = result
                        save_cache()
                        return result, "success (PubChem)"
                return None, "PubChem 返回空数据"
            elif r.status_code == 503:
                wait = 2.0 * (attempt + 1)
                time.sleep(wait)
                continue
            elif r.status_code == 404:
                return None, "PubChem 未找到该化合物 (404)"
            else:
                return None, f"PubChem HTTP {r.status_code}: {r.text[:100]}"
        except requests.exceptions.SSLError:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None, "SSL 连接失败"
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None, "查询超时，PubChem 服务器无响应"
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None, f"网络异常: {str(e)}"
    return None, "PubChem 持续不可用，请稍后重试"


# ========== 页面设置 ==========
st.set_page_config(
    page_title="SoluVis - Molecular Solubility Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== ORBITA THEME: Deep Space Universe CSS ==========
st.markdown("""
<style>
/* ═══════════════════════════════════════════════
   ORBITA THEME: Deep Space Molecular Universe
   ═══════════════════════════════════════════════ */

/* Google Fonts removed — system font stack for China accessibility */
/* ─── 0. 粒子背景动画 ─── */
@keyframes nebulaPulse {
    0%, 100% { opacity: 0.3; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.1); }
}

@keyframes gradientShift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

@keyframes glowPulse {
    0%, 100% { box-shadow: 0 0 20px rgba(124, 58, 237, 0.1); }
    50% { box-shadow: 0 0 40px rgba(124, 58, 237, 0.2); }
}

@keyframes starTwinkle {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 1; }
}

@keyframes molecularDiffuse {
    from { opacity: 0; transform: translateY(24px) scale(0.96); filter: blur(2px); }
    to { opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }
}

@keyframes electronJump {
    0% { transform: translateX(-100%); opacity: 0; }
    50% { opacity: 1; }
    100% { transform: translateX(400%); opacity: 0; }
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

/* ─── 1. 核心变量 ─── */
:root {
    --ob-bg-primary: #131328;
    --ob-bg-surface: #1a1a2e;
    --ob-bg-elevated: rgba(30, 30, 46, 0.6);
    --ob-nebula: #7c3aed;
    --ob-nebula-light: #a78bfa;
    --ob-nebula-glow: rgba(124, 58, 237, 0.15);
    --ob-star-gold: #fbbf24;
    --ob-orbit-cyan: #06b6d4;
    --ob-text-primary: #f0f0f5;
    --ob-text-secondary: #a0a0b0;
    --ob-text-tertiary: #6b6b7b;
    --ob-border: rgba(255, 255, 255, 0.08);
    --ob-border-hover: rgba(124, 58, 237, 0.3);
    --ob-radius: 16px;
    --ob-radius-sm: 12px;

    /* Base Web 暗色主题变量覆盖（下拉菜单、弹出框等） */
    --colors-menu-background-color: #1a1a2e;
    --colors-menu-color: #e0e0e0;
    --colors-menu-item-hover-background-color: rgba(124, 58, 237, 0.25);
    --colors-menu-item-selected-background-color: rgba(124, 58, 237, 0.35);
    --colors-popover-background-color: #1a1a2e;
    --colors-dropdown-menu-background-color: #1a1a2e;
    --colors-dropdown-menu-hover-background-color: rgba(124, 58, 237, 0.25);
    --colors-dropdown-menu-selected-background-color: rgba(124, 58, 237, 0.35);
}

/* ─── 2. 全局基础 ─── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .stApp {
    background: var(--ob-bg-primary) !important;
    background-color: var(--ob-bg-primary) !important;
    font-family: 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    color: var(--ob-text-primary);
}

.stApp {
    background: transparent !important;
    background-color: transparent !important;
    position: relative;
    min-height: 100vh;
}

/* 粒子星空层 - 密集多层 */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        /* === 大亮点（紫/青/金，模拟亮星）=== */
        radial-gradient(3.5px 3.5px at 8% 12%, rgba(196, 181, 253, 1.0), transparent),
        radial-gradient(3px 3px at 22% 8%, rgba(103, 232, 249, 0.95), transparent),
        radial-gradient(3.5px 3.5px at 35% 25%, rgba(167, 139, 250, 1.0), transparent),
        radial-gradient(2.5px 2.5px at 48% 5%, rgba(253, 224, 71, 0.9), transparent),
        radial-gradient(3.5px 3.5px at 62% 18%, rgba(34, 211, 238, 1.0), transparent),
        radial-gradient(3px 3px at 78% 10%, rgba(196, 181, 253, 1.0), transparent),
        radial-gradient(3.5px 3.5px at 88% 28%, rgba(103, 232, 249, 0.95), transparent),
        radial-gradient(2.5px 2.5px at 95% 8%, rgba(233, 213, 255, 0.9), transparent),
        radial-gradient(3.5px 3.5px at 15% 40%, rgba(34, 211, 238, 1.0), transparent),
        radial-gradient(3px 3px at 55% 35%, rgba(167, 139, 250, 0.95), transparent),
        radial-gradient(3.5px 3.5px at 72% 42%, rgba(253, 224, 71, 0.85), transparent),
        radial-gradient(2.5px 2.5px at 42% 48%, rgba(233, 213, 255, 0.9), transparent),
        /* === 中等星点 === */
        radial-gradient(2.5px 2.5px at 5% 55%, rgba(139, 92, 246, 0.9), transparent),
        radial-gradient(2px 2px at 18% 62%, rgba(103, 232, 249, 0.85), transparent),
        radial-gradient(2.5px 2.5px at 28% 75%, rgba(196, 181, 253, 0.9), transparent),
        radial-gradient(2px 2px at 40% 68%, rgba(34, 211, 238, 0.8), transparent),
        radial-gradient(2.5px 2.5px at 52% 82%, rgba(167, 139, 250, 0.85), transparent),
        radial-gradient(2px 2px at 65% 58%, rgba(253, 224, 71, 0.75), transparent),
        radial-gradient(2.5px 2.5px at 75% 72%, rgba(139, 92, 246, 0.9), transparent),
        radial-gradient(2px 2px at 85% 55%, rgba(103, 232, 249, 0.85), transparent),
        radial-gradient(2.5px 2.5px at 92% 78%, rgba(233, 213, 255, 0.8), transparent),
        radial-gradient(2px 2px at 12% 88%, rgba(34, 211, 238, 0.75), transparent),
        radial-gradient(2.5px 2.5px at 32% 92%, rgba(167, 139, 250, 0.85), transparent),
        radial-gradient(2px 2px at 58% 95%, rgba(196, 181, 253, 0.8), transparent),
        radial-gradient(2.5px 2.5px at 82% 88%, rgba(103, 232, 249, 0.75), transparent),
        /* === 白色小星点 === */
        radial-gradient(2px 2px at 3% 30%, rgba(255, 255, 255, 0.8), transparent),
        radial-gradient(1.5px 1.5px at 12% 18%, rgba(255, 255, 255, 0.75), transparent),
        radial-gradient(2px 2px at 25% 45%, rgba(255, 255, 255, 0.8), transparent),
        radial-gradient(1.5px 1.5px at 38% 15%, rgba(255, 255, 255, 0.7), transparent),
        radial-gradient(2px 2px at 45% 58%, rgba(255, 255, 255, 0.75), transparent),
        radial-gradient(1.5px 1.5px at 58% 12%, rgba(255, 255, 255, 0.7), transparent),
        radial-gradient(2px 2px at 68% 50%, rgba(255, 255, 255, 0.75), transparent),
        radial-gradient(1.5px 1.5px at 82% 35%, rgba(255, 255, 255, 0.7), transparent),
        radial-gradient(2px 2px at 90% 60%, rgba(255, 255, 255, 0.75), transparent),
        radial-gradient(1.5px 1.5px at 8% 78%, rgba(255, 255, 255, 0.65), transparent),
        radial-gradient(2px 2px at 48% 88%, rgba(255, 255, 255, 0.7), transparent),
        radial-gradient(1.5px 1.5px at 70% 92%, rgba(255, 255, 255, 0.65), transparent),
        /* === 微星点（增加密度）=== */
        radial-gradient(1.5px 1.5px at 7% 5%, rgba(196, 181, 253, 0.65), transparent),
        radial-gradient(1.5px 1.5px at 30% 3%, rgba(103, 232, 249, 0.6), transparent),
        radial-gradient(1.5px 1.5px at 50% 20%, rgba(233, 213, 255, 0.6), transparent),
        radial-gradient(1.5px 1.5px at 60% 8%, rgba(167, 139, 250, 0.65), transparent),
        radial-gradient(1.5px 1.5px at 80% 48%, rgba(34, 211, 238, 0.6), transparent),
        radial-gradient(1.5px 1.5px at 95% 42%, rgba(196, 181, 253, 0.6), transparent),
        radial-gradient(1.5px 1.5px at 20% 52%, rgba(253, 224, 71, 0.55), transparent),
        radial-gradient(1.5px 1.5px at 45% 72%, rgba(196, 181, 253, 0.6), transparent),
        radial-gradient(1.5px 1.5px at 88% 68%, rgba(103, 232, 249, 0.55), transparent);
    background-size: 100% 100%;
    animation: starTwinkle 4s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
    opacity: 1.0;
}

/* 星云光晕叠加 - 多层 */
.stApp::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        /* 主星云：紫色弥散光（左上区域） */
        radial-gradient(ellipse 65% 50% at 25% 35%, rgba(124, 58, 237, 0.35) 0%, rgba(76, 29, 149, 0.15) 35%, transparent 65%),
        /* 副星云：青色弥散光（右上区域） */
        radial-gradient(ellipse 55% 45% at 75% 25%, rgba(6, 182, 212, 0.25) 0%, rgba(8, 145, 178, 0.1) 40%, transparent 70%),
        /* 底部暖光晕 */
        radial-gradient(ellipse 55% 40% at 50% 90%, rgba(251, 191, 36, 0.15) 0%, transparent 55%),
        /* 中央微弱紫光 */
        radial-gradient(ellipse 40% 40% at 50% 50%, rgba(109, 40, 217, 0.06) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
    animation: nebulaPulse 10s ease-in-out infinite;
}

.main .block-container {
    background: transparent !important;
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1200px !important;
    position: relative;
    z-index: 1;
}

code, pre, .mono {
    font-family: 'Cascadia Code', 'Consolas', 'Menlo', monospace !important;
}

/* ─── 3. 排版 ─── */
.gradient-title {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    font-weight: 700;
    font-size: clamp(2.2rem, 5vw, 3.2rem);
    text-align: center;
    letter-spacing: -0.04em;
    line-height: 1.1;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, #a78bfa 0%, #7c3aed 30%, #06b6d4 70%, #22d3ee 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 8s ease infinite;
    filter: drop-shadow(0 0 20px rgba(124, 58, 237, 0.3));
}

.subtitle {
    text-align: center;
    color: var(--ob-text-secondary);
    font-size: 1.05rem;
    font-weight: 400;
    line-height: 1.5;
    margin-bottom: 2rem;
    font-family: 'Segoe UI', system-ui, sans-serif;
    letter-spacing: 0.02em;
}

.tagline {
    text-align: center;
    font-family: 'Cascadia Code', 'Consolas', monospace !important;
    font-size: 0.75rem;
    color: var(--ob-nebula-light);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.card-title {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    color: var(--ob-text-primary);
    font-size: 1.125rem;
    font-weight: 600;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    letter-spacing: -0.01em;
}

.card-title::before {
    content: '';
    display: inline-block;
    width: 4px;
    height: 1.2em;
    background: linear-gradient(180deg, var(--ob-nebula), var(--ob-orbit-cyan));
    border-radius: 4px;
    flex-shrink: 0;
    box-shadow: 0 0 8px rgba(124, 58, 237, 0.4);
}

/* ─── 4. 卡片容器 ─── */
.card-container {
    background: linear-gradient(155deg, rgba(35, 35, 55, 0.55) 0%, rgba(20, 20, 35, 0.45) 50%, rgba(30, 30, 50, 0.40) 100%);
    backdrop-filter: blur(12px) saturate(120%);
    -webkit-backdrop-filter: blur(12px) saturate(120%);
    border: 1px solid var(--ob-border);
    border-radius: var(--ob-radius);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow:
        0 1px 2px rgba(0,0,0,0.2),
        0 4px 12px -2px rgba(0,0,0,0.25),
        inset 0 1px 0 rgba(255,255,255,0.04);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.15s ease;
    will-change: transform, box-shadow;
    position: relative;
    overflow: hidden;
    animation: molecularDiffuse 0.35s ease both;
}

.card-container::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(124, 58, 237, 0.35) 20%, rgba(6, 182, 212, 0.2) 50%, rgba(124, 58, 237, 0.35) 80%, transparent 100%);
    pointer-events: none;
}

.card-container::after {
    content: '';
    position: absolute;
    top: -40%; right: -20%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(124, 58, 237, 0.06) 0%, transparent 70%);
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.3s ease;
}

.card-container:hover {
    transform: translateY(-4px);
    box-shadow:
        0 20px 40px -8px rgba(0,0,0,0.4),
        0 0 0 1px rgba(124, 58, 237, 0.15),
        0 0 40px rgba(124, 58, 237, 0.08),
        inset 0 1px 0 rgba(255,255,255,0.06);
    border-color: rgba(124, 58, 237, 0.2);
}

.card-container:hover::after { opacity: 1; }

.card-container:nth-child(1) { animation-delay: 0.05s; }
.card-container:nth-child(2) { animation-delay: 0.10s; }
.card-container:nth-child(3) { animation-delay: 0.15s; }
.card-container:nth-child(4) { animation-delay: 0.20s; }

/* ─── 5. 按钮 ─── */
.stButton > button {
    background: linear-gradient(135deg, #6d28d9, #7c3aed, #8b5cf6) !important;
    background-size: 200% 200% !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--ob-radius-sm) !important;
    padding: 0.75rem 1.5rem !important;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    font-size: 0.9375rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 4px 15px -3px rgba(124, 58, 237, 0.3), 0 0 0 0 rgba(124, 58, 237, 0) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease, background-position 0.3s ease !important;
    will-change: transform, box-shadow;
    position: relative;
    overflow: hidden;
}

.stButton > button::after {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
    transition: left 0.4s ease;
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 10px 25px -3px rgba(0,0,0,0.3), 0 0 25px rgba(124, 58, 237, 0.2) !important;
    background-position: 100% 0% !important;
}

.stButton > button:hover::after { left: 100%; }

.stButton > button:active {
    transform: translateY(0) scale(0.97);
    box-shadow: 0 2px 8px -1px rgba(0,0,0,0.2), inset 0 2px 4px rgba(0,0,0,0.1) !important;
}

.stButton > button[kind="secondary"] {
    background: linear-gradient(135deg, #0e7490, #06b6d4, #22d3ee) !important;
    box-shadow: 0 4px 15px -3px rgba(6, 182, 212, 0.3) !important;
}

.stButton > button[kind="secondary"]:hover {
    box-shadow: 0 10px 25px -3px rgba(0,0,0,0.3), 0 0 25px rgba(6, 182, 212, 0.2) !important;
}

/* ─── 6. 输入框 ─── */
/* 先彻底清除所有外层容器的默认白边/灰边 */
.stTextInput > div,
.stTextInput > div > div,
.stTextInput [data-baseweb="input"],
.stTextInput [data-baseweb="input"] > div,
.stTextInput [data-baseweb="input"] > div > div,
.stTextInput [data-testid="stTextInput"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* 真正的 input 输入框样式（与 selectbox 统一） */
.stTextInput > div > div > input,
.stTextInput > div > div > textarea,
.stTextInput [data-baseweb="input"] > div,
.stTextInput [data-baseweb="input"] input {
    border-radius: var(--ob-radius-sm) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding: 0.625rem 1rem !important;
    font-size: 0.9375rem !important;
    background: linear-gradient(135deg, rgba(30, 30, 46, 0.9) 0%, rgba(15, 15, 25, 0.7) 100%) !important;
    color: var(--ob-text-primary) !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.2) !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    font-family: 'Cascadia Code', 'Consolas', monospace !important;
}

.stTextInput > div > div > input::placeholder,
.stTextInput [data-baseweb="input"] input::placeholder {
    color: var(--ob-text-tertiary) !important;
}

.stTextInput > div > div > input:focus,
.stTextInput [data-baseweb="input"] > div:focus-within {
    border-color: var(--ob-nebula) !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.15), inset 0 1px 3px rgba(0,0,0,0.2) !important;
    outline: none !important;
}

/* ─── Selectbox ─── */
.stSelectbox [data-baseweb="select"] > div,
.stSelectbox [data-baseweb="select"] > div > div,
.stSelectbox [data-baseweb="select"] input {
    border-radius: var(--ob-radius-sm) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    background: linear-gradient(135deg, rgba(30, 30, 46, 0.9) 0%, rgba(15, 15, 25, 0.7) 100%) !important;
    color: var(--ob-text-primary) !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.2) !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}

.stSelectbox [data-baseweb="select"] > div:hover {
    border-color: rgba(124, 58, 237, 0.35) !important;
}

.stSelectbox [data-baseweb="select"] > div:focus-within {
    border-color: var(--ob-nebula) !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.15), inset 0 1px 3px rgba(0,0,0,0.2) !important;
}

/* 清除 selectbox 原生白色背景 */
.stSelectbox svg,
.stSelectbox [data-baseweb="select"] svg {
    color: var(--ob-text-secondary) !important;
}

/* ─── 下拉菜单容器：全局强制覆盖（不局限在 .stSelectbox 内部） ─── */
/* Streamlit 的下拉菜单位于 body 层级的 portal/overlay 中 */
html body [data-baseweb="menu"],
html body [data-baseweb="popover"],
html body [data-baseweb="select"] [data-baseweb="menu"],
html body div[role="listbox"],
html body ul[role="listbox"] {
    background: #1a1a2e !important;
    background-color: #1a1a2e !important;
    background-image: none !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: var(--ob-radius-sm) !important;
    box-shadow: 0 10px 30px -5px rgba(0,0,0,0.4) !important;
    color: #e0e0e0 !important;
}

/* 菜单内直接子容器 —— 必须显式设回深色，覆盖浏览器/组件默认白底 */
html body [data-baseweb="menu"] > div,
html body [data-baseweb="menu"] > div > div:first-child,
html body [data-baseweb="menu"] [role="listbox"],
html body [data-baseweb="menu"] ul,
html body div[role="listbox"] > div,
html body ul[role="listbox"] > li {
    background: #1a1a2e !important;
    background-color: #1a1a2e !important;
    background-image: none !important;
    color: #e0e0e0 !important;
}

/* 下拉选项行 */
html body [role="option"],
html body [data-baseweb="menu"] [role="option"],
html body [data-baseweb="menu"] div[role="option"],
html body ul li[role="option"],
html body div[role="listbox"] div {
    color: #e0e0e0 !important;
    background: transparent !important;
    background-color: transparent !important;
    transition: background 0.15s ease, color 0.15s ease !important;
}

html body [role="option"]:hover,
html body [data-baseweb="menu"] [role="option"]:hover,
html body div[role="listbox"] div:hover {
    background: rgba(124, 58, 237, 0.25) !important;
    background-color: rgba(124, 58, 237, 0.25) !important;
    color: #ffffff !important;
}

html body [aria-selected="true"],
html body [data-baseweb="menu"] [aria-selected="true"],
html body div[role="listbox"] [aria-selected="true"] {
    background: rgba(124, 58, 237, 0.35) !important;
    background-color: rgba(124, 58, 237, 0.35) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* ─── 7. Metric ─── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(30, 30, 46, 0.55) 0%, rgba(15, 15, 25, 0.4) 100%);
    border: 1px solid var(--ob-border);
    border-radius: var(--ob-radius-sm);
    padding: 1rem 1.25rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 2px 10px rgba(0,0,0,0.15);
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(124, 58, 237, 0.2);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 4px 20px rgba(0,0,0,0.2), 0 0 20px rgba(124, 58, 237, 0.06);
}

[data-testid="stMetricValue"] {
    font-family: 'Cascadia Code', 'Consolas', monospace !important;
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    line-height: 1.15;
    background: linear-gradient(135deg, #f0f0f5 0%, #a0a0b0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

[data-testid="stMetricLabel"] {
    font-family: 'Cascadia Code', 'Consolas', monospace !important;
    font-size: 0.7rem !important;
    color: var(--ob-text-tertiary) !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.5rem;
}

/* ─── 8. 结果状态卡片 ─── */
.result-high, .result-moderate, .result-low {
    border-radius: var(--ob-radius);
    padding: 1.25rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.result-high {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(16, 185, 129, 0.03));
    border: 1px solid rgba(16, 185, 129, 0.2);
    animation: glowPulse 3s ease-in-out infinite;
}
@keyframes glowPulse {
    0%, 100% { box-shadow: 0 4px 12px -2px rgba(0,0,0,0.15), 0 0 0 0 rgba(16, 185, 129, 0); }
    50% { box-shadow: 0 4px 12px -2px rgba(0,0,0,0.15), 0 0 25px rgba(16, 185, 129, 0.08); }
}

.result-high::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #10b981, #34d399, #10b981);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.5);
}
.result-high:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 24px -4px rgba(0,0,0,0.2), 0 0 30px rgba(16, 185, 129, 0.12);
}

.result-moderate {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.08), rgba(245, 158, 11, 0.03));
    border: 1px solid rgba(245, 158, 11, 0.2);
    animation: glowPulseAmber 3s ease-in-out infinite;
}
@keyframes glowPulseAmber {
    0%, 100% { box-shadow: 0 4px 12px -2px rgba(0,0,0,0.15), 0 0 0 0 rgba(245, 158, 11, 0); }
    50% { box-shadow: 0 4px 12px -2px rgba(0,0,0,0.15), 0 0 25px rgba(245, 158, 11, 0.08); }
}
.result-moderate::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #f59e0b, #fbbf24, #f59e0b);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
    box-shadow: 0 0 12px rgba(245, 158, 11, 0.5);
}
.result-moderate:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 24px -4px rgba(0,0,0,0.2), 0 0 30px rgba(245, 158, 11, 0.12);
}

.result-low {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.08), rgba(239, 68, 68, 0.03));
    border: 1px solid rgba(239, 68, 68, 0.2);
    animation: glowPulseRed 3s ease-in-out infinite;
}
@keyframes glowPulseRed {
    0%, 100% { box-shadow: 0 4px 12px -2px rgba(0,0,0,0.15), 0 0 0 0 rgba(239, 68, 68, 0); }
    50% { box-shadow: 0 4px 12px -2px rgba(0,0,0,0.15), 0 0 25px rgba(239, 68, 68, 0.08); }
}
.result-low::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #ef4444, #f87171, #ef4444);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
    box-shadow: 0 0 12px rgba(239, 68, 68, 0.5);
}
.result-low:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 24px -4px rgba(0,0,0,0.2), 0 0 30px rgba(239, 68, 68, 0.12);
}

/* pKa 状态 */
.pka-acid, .pka-base, .pka-amphoteric {
    border-radius: var(--ob-radius);
    padding: 1.25rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.pka-acid {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.08), rgba(124, 58, 237, 0.03));
    border: 1px solid rgba(124, 58, 237, 0.2);
    box-shadow: 0 4px 12px -2px rgba(0,0,0,0.15);
}
.pka-acid::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #7c3aed, #a78bfa, #7c3aed);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
    box-shadow: 0 0 10px rgba(124, 58, 237, 0.5);
}
.pka-acid:hover { transform: scale(1.02); box-shadow: 0 8px 24px -4px rgba(0,0,0,0.2), 0 0 25px rgba(124, 58, 237, 0.12); }

.pka-base {
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.08), rgba(6, 182, 212, 0.03));
    border: 1px solid rgba(6, 182, 212, 0.2);
    box-shadow: 0 4px 12px -2px rgba(0,0,0,0.15);
}
.pka-base::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #06b6d4, #22d3ee, #06b6d4);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
    box-shadow: 0 0 10px rgba(6, 182, 212, 0.5);
}
.pka-base:hover { transform: scale(1.02); box-shadow: 0 8px 24px -4px rgba(0,0,0,0.2), 0 0 25px rgba(6, 182, 212, 0.12); }

.pka-amphoteric {
    background: linear-gradient(135deg, rgba(251, 191, 36, 0.08), rgba(251, 191, 36, 0.03));
    border: 1px solid rgba(251, 191, 36, 0.2);
    box-shadow: 0 4px 12px -2px rgba(0,0,0,0.15);
}
.pka-amphoteric::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #fbbf24, #fde68a, #fbbf24);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
    box-shadow: 0 0 10px rgba(251, 191, 36, 0.5);
}
.pka-amphoteric:hover { transform: scale(1.02); box-shadow: 0 8px 24px -4px rgba(0,0,0,0.2), 0 0 25px rgba(251, 191, 36, 0.12); }

/* ─── 9. Alert ─── */
.stAlert {
    border-radius: var(--ob-radius) !important;
    border: 1px solid !important;
    padding: 1.25rem !important;
    position: relative;
    overflow: hidden;
    animation: molecularDiffuse 0.25s ease both;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}

.stAlert::before {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 4px;
    border-radius: 4px 0 0 4px;
}

.stAlert[data-baseweb="notification"][data-kind="positive"] {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(16, 185, 129, 0.02)) !important;
    border-color: rgba(16, 185, 129, 0.18) !important;
}
.stAlert[data-baseweb="notification"][data-kind="positive"]::before {
    background: linear-gradient(180deg, #10b981, #34d399);
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.4);
}

.stAlert[data-baseweb="notification"][data-kind="info"] {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.08), rgba(124, 58, 237, 0.02)) !important;
    border-color: rgba(124, 58, 237, 0.18) !important;
}
.stAlert[data-baseweb="notification"][data-kind="info"]::before {
    background: linear-gradient(180deg, #7c3aed, #a78bfa);
    box-shadow: 0 0 8px rgba(124, 58, 237, 0.4);
}

.stAlert[data-baseweb="notification"][data-kind="warning"] {
    background: linear-gradient(135deg, rgba(251, 191, 36, 0.08), rgba(251, 191, 36, 0.02)) !important;
    border-color: rgba(251, 191, 36, 0.18) !important;
}
.stAlert[data-baseweb="notification"][data-kind="warning"]::before {
    background: linear-gradient(180deg, #fbbf24, #fde68a);
    box-shadow: 0 0 8px rgba(251, 191, 36, 0.4);
}

.stAlert[data-baseweb="notification"][data-kind="negative"] {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.08), rgba(239, 68, 68, 0.02)) !important;
    border-color: rgba(239, 68, 68, 0.18) !important;
}
.stAlert[data-baseweb="notification"][data-kind="negative"]::before {
    background: linear-gradient(180deg, #ef4444, #f87171);
    box-shadow: 0 0 8px rgba(239, 68, 68, 0.4);
}

.stAlert [data-testid="stMarkdownContainer"] {
    font-size: 0.875rem;
    line-height: 1.6;
    color: var(--ob-text-primary);
}

/* ─── 10. Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    padding-bottom: 0.25rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 12px 12px 0 0 !important;
    padding: 0.5rem 1rem !important;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    color: var(--ob-text-secondary) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.15s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--ob-text-primary) !important;
    background: rgba(124, 58, 237, 0.05) !important;
    text-shadow: 0 0 12px rgba(124, 58, 237, 0.2);
}

.stTabs [aria-selected="true"] {
    color: var(--ob-nebula-light) !important;
    border-bottom-color: var(--ob-nebula) !important;
    background: linear-gradient(180deg, rgba(124, 58, 237, 0.1), transparent) !important;
    text-shadow: 0 0 16px rgba(124, 58, 237, 0.25);
    font-weight: 600 !important;
}

/* ─── 10.5 Radio（方式1 替代 selectbox）─── */
.stRadio > div {
    display: flex;
    flex-direction: column;
    gap: 0 !important;
}

.stRadio > div > div {
    margin: 0 !important;
    padding: 0 !important;
}

.stRadio label {
    display: flex;
    align-items: center;
    padding: 0.5rem 0.875rem !important;
    margin: 0 !important;
    border-radius: 8px !important;
    cursor: pointer;
    transition: background 0.15s ease, color 0.15s ease !important;
    font-size: 0.9rem !important;
    color: var(--ob-text-secondary) !important;
    background: transparent !important;
    border: none !important;
}

.stRadio label:hover {
    background: rgba(124, 58, 237, 0.12) !important;
    color: var(--ob-text-primary) !important;
}

/* 隐藏原生的 radio 圆圈 */
.stRadio label > span:first-child {
    display: none !important;
}

/* 选中的 radio 项 */
.stRadio [aria-checked="true"] + label,
.stRadio input:checked + div label,
.stRadio label[data-baseweb="radio"] [aria-checked="true"] {
    background: rgba(124, 58, 237, 0.18) !important;
    color: var(--ob-nebula-light) !important;
    font-weight: 600 !important;
}

/* radio 容器滚动 */
.stRadio > div {
    max-height: 320px;
    overflow-y: auto;
    padding-right: 4px;
}

/* ─── 11. 媒体与图表 ─── */
.stImage > img {
    border-radius: var(--ob-radius);
    border: 1px solid var(--ob-border);
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.stImage > img:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 24px rgba(0,0,0,0.25), 0 0 30px rgba(124, 58, 237, 0.08);
}

.js-plotly-plot, .stPlotlyChart {
    border-radius: var(--ob-radius);
    overflow: hidden;
}

/* ─── 12. 分隔线与页脚 ─── */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(124, 58, 237, 0.2) 20%, rgba(6, 182, 212, 0.15) 50%, rgba(124, 58, 237, 0.2) 80%, transparent 100%);
    margin: 2rem 0;
    position: relative;
}
hr::after {
    content: '';
    position: absolute;
    left: 50%; top: -2px;
    transform: translateX(-50%);
    width: 6px; height: 6px;
    background: rgba(124, 58, 237, 0.4);
    border-radius: 50%;
    box-shadow: 0 0 12px rgba(124, 58, 237, 0.4);
}

.footer {
    text-align: center;
    padding: 2.5rem 1.5rem;
    color: var(--ob-text-tertiary);
    font-size: 0.8125rem;
    line-height: 1.6;
    border-top: 1px solid var(--ob-border);
    margin-top: 3rem;
    position: relative;
    font-family: 'Cascadia Code', 'Consolas', monospace;
}
.footer::before {
    content: '';
    position: absolute;
    top: -1px; left: 50%;
    transform: translateX(-50%);
    width: 120px; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(124, 58, 237, 0.4), transparent);
}

/* ─── 13. 加载状态 ─── */
.stSpinner > div {
    border-color: var(--ob-nebula) transparent transparent transparent !important;
}

.loading-bar {
    height: 2px;
    background: rgba(124, 58, 237, 0.1);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
}
.loading-bar::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 25%; height: 100%;
    background: linear-gradient(90deg, transparent, var(--ob-nebula), transparent);
    animation: electronJump 1.5s ease-in-out infinite;
}

/* ─── 14. 辅助工具类 ─── */
.text-gradient {
    background: linear-gradient(135deg, #a78bfa, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.badge {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.5rem;
    border-radius: 6px;
    font-family: 'Cascadia Code', 'Consolas', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.badge-primary {
    background: rgba(124, 58, 237, 0.12);
    color: #c4b5fd;
    border: 1px solid rgba(124, 58, 237, 0.15);
}

.badge-success {
    background: rgba(16, 185, 129, 0.12);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.15);
}

.badge-warn {
    background: rgba(251, 191, 36, 0.12);
    color: #fcd34d;
    border: 1px solid rgba(251, 191, 36, 0.15);
}

/* ─── 15. 滚动条 ─── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, rgba(124, 58, 237, 0.3), rgba(6, 182, 212, 0.25));
    border-radius: 10px;
    border: 1px solid transparent;
    background-clip: content-box;
}
::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, rgba(124, 58, 237, 0.5), rgba(6, 182, 212, 0.4));
    background-clip: content-box;
}

/* 文字选中效果 */
::selection {
    background: rgba(124, 58, 237, 0.3);
    color: var(--ob-text-primary);
}

/* Streamlit 原生组件适配 */
.stMarkdown, .stText, p, li, span { color: var(--ob-text-primary); }
.stCaption {
    color: var(--ob-text-tertiary) !important;
    font-size: 0.8125rem !important;
    font-family: 'Cascadia Code', 'Consolas', monospace !important;
}

/* st.container(border=True) */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(155deg, rgba(30, 30, 46, 0.65) 0%, rgba(15, 15, 25, 0.45) 100%) !important;
    border: 1px solid var(--ob-border) !important;
    border-radius: var(--ob-radius) !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.15), 0 4px 12px -2px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.03) !important;
    transition: all 0.2s ease !important;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(124, 58, 237, 0.12) !important;
    box-shadow: 0 4px 12px -2px rgba(0,0,0,0.25), 0 0 20px rgba(124, 58, 237, 0.05), inset 0 1px 0 rgba(255,255,255,0.04) !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: linear-gradient(155deg, rgba(30, 30, 46, 0.6) 0%, rgba(15, 15, 25, 0.4) 100%) !important;
    border: 1px solid var(--ob-border) !important;
    border-radius: var(--ob-radius) !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.12) !important;
}

[data-testid="stExpander"]:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.18), 0 0 0 1px rgba(124, 58, 237, 0.06) !important;
}

/* 隐藏 Streamlit 默认顶栏 */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* 响应式 */
@media (max-width: 768px) {
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .gradient-title { font-size: 2rem; }
    .card-container { padding: 1.25rem; }
}

/* ─── 图表玻璃容器 ─── */
.chart-glass-wrapper {
    border-radius: var(--ob-radius);
    overflow: hidden;
    background: linear-gradient(155deg, rgba(30, 30, 46, 0.50) 0%, rgba(15, 15, 25, 0.35) 100%);
    backdrop-filter: blur(8px) saturate(110%);
    -webkit-backdrop-filter: blur(8px) saturate(110%);
    border: 1px solid var(--ob-border);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.03);
    padding: 0.5rem;
    margin-bottom: 1rem;
}

/* ─── Tab 内容区玻璃底座 ─── */
.tab-glass-panel {
    background: linear-gradient(180deg, rgba(20, 20, 35, 0.40) 0%, rgba(13, 13, 20, 0.25) 100%);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 0 0 var(--ob-radius) var(--ob-radius);
    padding: 1.5rem;
    margin-top: -0.5rem;
    animation: molecularDiffuse 0.3s ease both;
}

/* ─── 骨架屏动画 ─── */
@keyframes skeletonShimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
.skeleton-box {
    background: linear-gradient(90deg, rgba(30,30,46,0.6) 25%, rgba(50,50,70,0.5) 50%, rgba(30,30,46,0.6) 75%);
    background-size: 200% 100%;
    border-radius: var(--ob-radius-sm);
    animation: skeletonShimmer 1.8s ease infinite;
    min-height: 120px;
    margin-bottom: 1rem;
}

/* ─── Tab 标签增强 ─── */
.stTabs [data-baseweb="tab-panel"] {
    animation: molecularDiffuse 0.3s ease both;
}

/* ─── prefers-reduced-motion ─── */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
    .stApp::before, .stApp::after { display: none; }
    .cursor-glow { display: none; }
    .skeleton-box { animation: none; background: rgba(30,30,46,0.6); }
}

/* ─── 文字选中增强 ─── */
::selection {
    background: rgba(124, 58, 237, 0.35);
    color: #ffffff;
    text-shadow: 0 0 8px rgba(124, 58, 237, 0.4);
}

/* ─── 响应式图表容器 ─── */
@media (max-width: 768px) {
    .chart-glass-wrapper { padding: 0.25rem; border-radius: var(--ob-radius-sm); }
    .tab-glass-panel { padding: 1rem; }
}

/* ─── 交互增强：鼠标悬停光效 ─── */
.cursor-glow {
    position: fixed;
    width: 300px;
    height: 300px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(124, 58, 237, 0.22) 0%, rgba(124, 58, 237, 0.08) 40%, transparent 70%);
    pointer-events: none;
    z-index: 9999;
    transform: translate3d(-50%, -50%, 0);
    transition: opacity 0.3s ease;
    will-change: transform;
}
</style>


""", unsafe_allow_html=True)

# ========== JS 强制覆盖下拉菜单样式（合并优化版，无轮询）==========
components.html("""
<script>
(function() {
    'use strict';
    var win = window.parent || window;
    var doc = win.document;
    if (!doc || !doc.body) return;

    var DARK_BG = '#1a1a2e';
    var TEXT = '#e0e0e0';
    var HOVER_BG = 'rgba(124, 58, 237, 0.25)';
    var SELECTED_BG = 'rgba(124, 58, 237, 0.35)';
    var SELECTED_COLOR = '#ffffff';
    var fixedMenus = new WeakSet();

    function applyMenuStyle(menu) {
        if (fixedMenus.has(menu)) return;
        fixedMenus.add(menu);
        menu.style.background = DARK_BG;
        menu.style.backgroundColor = DARK_BG;
        menu.style.border = '1px solid rgba(255, 255, 255, 0.08)';
        menu.style.borderRadius = '12px';
        menu.style.boxShadow = '0 10px 30px -5px rgba(0,0,0,0.4)';
        menu.style.color = TEXT;
    }

    function fixAllMenus() {
        var menus = doc.querySelectorAll('[data-baseweb="menu"], [data-baseweb="popover"], [role="listbox"]');
        menus.forEach(function(menu) {
            applyMenuStyle(menu);
            // clean inner divs
            var divs = menu.querySelectorAll('div');
            for (var i = 0; i < divs.length; i++) {
                divs[i].style.background = 'transparent';
                divs[i].style.backgroundColor = 'transparent';
                divs[i].style.backgroundImage = 'none';
            }
            // fix option rows
            var opts = menu.querySelectorAll('[role="option"]');
            opts.forEach(function(opt) {
                opt.style.color = TEXT;
                opt.style.background = 'transparent';
                opt.style.backgroundColor = 'transparent';
                if (opt.getAttribute('aria-selected') === 'true') {
                    opt.style.background = SELECTED_BG;
                    opt.style.backgroundColor = SELECTED_BG;
                    opt.style.color = SELECTED_COLOR;
                    opt.style.fontWeight = '600';
                }
                if (!opt.dataset.obHover) {
                    opt.dataset.obHover = '1';
                    opt.addEventListener('mouseenter', function() {
                        if (opt.getAttribute('aria-selected') !== 'true') {
                            opt.style.background = HOVER_BG;
                            opt.style.backgroundColor = HOVER_BG;
                            opt.style.color = '#ffffff';
                        }
                    });
                    opt.addEventListener('mouseleave', function() {
                        if (opt.getAttribute('aria-selected') !== 'true') {
                            opt.style.background = 'transparent';
                            opt.style.backgroundColor = 'transparent';
                            opt.style.color = TEXT;
                        }
                    });
                }
            });
        });
    }

    var pending = false;
    function scheduleFix() {
        if (pending) return;
        pending = true;
        requestAnimationFrame(function() {
            pending = false;
            fixAllMenus();
        });
    }

    var observer = new MutationObserver(function(mutations) {
        for (var m = 0; m < mutations.length; m++) {
            var added = mutations[m].addedNodes;
            for (var n = 0; n < added.length; n++) {
                if (added[n].nodeType === 1) {
                    scheduleFix();
                    break;
                }
            }
        }
    });
    observer.observe(doc.documentElement || doc.body, { childList: true, subtree: true });
    doc.body.addEventListener('click', function() { setTimeout(fixAllMenus, 100); });
    fixAllMenus();
})();
</script>
""", height=0)

# ========== Canvas 动态粒子星空背景 ==========
components.html("""
<script>
(function() {
    'use strict';
    console.log('[Orbita] Starfield script loaded');
    
    // 检查是否偏好减弱动画
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        console.log('[Orbita] prefers-reduced-motion is enabled, skipping starfield');
        return;
    }
    
    var win = window.parent || window;
    var doc = win.document;
    
    var canvas = doc.getElementById('ob-starfield');
    if (!canvas) {
        canvas = doc.createElement('canvas');
        canvas.id = 'ob-starfield';
        canvas.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;z-index:1;pointer-events:none;';
        var app = doc.querySelector('.stApp');
        if (app) {
            app.insertBefore(canvas, app.firstChild);
        } else {
            doc.body.insertBefore(canvas, doc.body.firstChild);
        }
        console.log('[Orbita] Starfield canvas created');
    } else {
        console.log('[Orbita] Starfield canvas already exists, reusing');
    }
    
    var ctx = canvas.getContext('2d');
    var W, H;
    function resize() {
        W = canvas.width = (win.innerWidth || 1920);
        H = canvas.height = (win.innerHeight || 1080);
    }
    resize();
    win.addEventListener('resize', resize);
    console.log('[Orbita] Canvas size: ' + W + 'x' + H);
    
    // 防止事件重复绑定
    if (!doc.__obStarMouseBound) {
        doc.__obStarMouseBound = true;
        doc.addEventListener('mousemove', function(e) {
            mouseStarX = e.clientX; mouseStarY = e.clientY;
            mouseTrail.push({x: e.clientX, y: e.clientY, age: 0});
            if (mouseTrail.length > maxTrail) mouseTrail.shift();
        }, { passive: true });
        doc.addEventListener('click', function(e) {
            bursts.push({x: e.clientX, y: e.clientY, radius: 0, maxRadius: 200, life: 1.0});
        }, { passive: true });
    }
    
    // 粒子配置 - 3层（优化数量：从380减至200）
    var layers = [
        { count: 50, minR: 1.8, maxR: 3.5, speed: 0.08, colors: ['rgba(196,181,253,', 'rgba(103,232,249,', 'rgba(233,213,255,', 'rgba(255,255,255,'] },
        { count: 70, minR: 1.0, maxR: 2.2, speed: 0.04, colors: ['rgba(167,139,250,', 'rgba(34,211,238,', 'rgba(139,92,246,', 'rgba(253,224,71,'] },
        { count: 80, minR: 0.5, maxR: 1.2, speed: 0.02, colors: ['rgba(196,181,253,', 'rgba(103,232,249,', 'rgba(255,255,255,'] }
    ];
    
    var stars = [];
    layers.forEach(function(layer, li) {
        for (var i = 0; i < layer.count; i++) {
            var colorBase = layer.colors[Math.floor(Math.random() * layer.colors.length)];
            stars.push({
                x: Math.random() * W,
                y: Math.random() * H,
                r: layer.minR + Math.random() * (layer.maxR - layer.minR),
                baseR: layer.minR + Math.random() * (layer.maxR - layer.minR),
                vx: (Math.random() - 0.5) * layer.speed,
                vy: (Math.random() - 0.5) * layer.speed,
                color: colorBase,
                alpha: 0.5 + Math.random() * 0.5,
                twinkleSpeed: 0.005 + Math.random() * 0.015,
                twinklePhase: Math.random() * Math.PI * 2,
                layer: li
            });
        }
    });
    
    var mouseStarX = W / 2, mouseStarY = H / 2;
    var mouseTrail = [];
    var maxTrail = 20;
    var bursts = [];
    
    var frame = 0;
    function animate() {
        // 绘制深色背景（替代 .stApp 的 CSS 背景）
        var bgGrad = ctx.createLinearGradient(0, 0, 0, H);
        bgGrad.addColorStop(0, '#0f0f1c');
        bgGrad.addColorStop(0.3, '#131328');
        bgGrad.addColorStop(0.5, '#181830');
        bgGrad.addColorStop(0.7, '#131328');
        bgGrad.addColorStop(1, '#0f0f1c');
        ctx.fillStyle = bgGrad;
        ctx.clearRect(0, 0, W, H);

        // 绘制星云光晕
        var nebulas = [
            {x: W*0.18, y: H*0.25, rx: W*0.35, ry: H*0.28, color: 'rgba(124,58,237,'},
            {x: W*0.82, y: H*0.12, rx: W*0.28, ry: H*0.22, color: 'rgba(6,182,212,'},
            {x: W*0.50, y: H*0.80, rx: W*0.25, ry: H*0.20, color: 'rgba(251,191,36,'},
            {x: W*0.50, y: H*0.40, rx: W*0.40, ry: H*0.32, color: 'rgba(67,56,202,'}
        ];
        nebulas.forEach(function(n) {
            var g = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, Math.max(n.rx, n.ry));
            g.addColorStop(0, n.color + '0.25)');
            g.addColorStop(0.5, n.color + '0.08)');
            g.addColorStop(1, n.color + '0)');
            ctx.fillStyle = g;
            ctx.beginPath();
            ctx.ellipse(n.x, n.y, n.rx, n.ry, 0, 0, Math.PI*2);
            ctx.fill();
        });

        frame++;
        
        // 鼠标附近星星连线网络（限制范围，仅检查亮星和中星层）
        var lineCount = 0;
        var maxLines = 30;
        for (var i = 0; i < stars.length && lineCount < maxLines; i++) {
            var dmx = stars[i].x - mouseStarX, dmy = stars[i].y - mouseStarY;
            var dMouse = Math.sqrt(dmx*dmx + dmy*dmy);
            if (dMouse > 200) continue;
            // 只与有限数量的邻居连线
            for (var j = i + 1; j < stars.length && lineCount < maxLines; j++) {
                if (stars[j].layer > 1) continue; // 跳过微星层
                var dx = stars[i].x - stars[j].x, dy = stars[i].y - stars[j].y;
                var dist = Math.sqrt(dx*dx + dy*dy);
                if (dist < 120) {
                    var lineAlpha = 0.1 * (1 - dist/120) * (1 - dMouse/200);
                    ctx.beginPath();
                    ctx.moveTo(stars[i].x, stars[i].y);
                    ctx.lineTo(stars[j].x, stars[j].y);
                    ctx.strokeStyle = 'rgba(167,139,250,' + lineAlpha + ')';
                    ctx.lineWidth = 0.6;
                    ctx.stroke();
                    lineCount++;
                }
            }
        }

        // ── 处理点击爆发效果 ──
        for (var b = bursts.length - 1; b >= 0; b--) {
            var burst = bursts[b];
            burst.radius += 5;
            burst.life = Math.max(0, 1 - burst.radius / burst.maxRadius);
            if (burst.life <= 0) {
                bursts.splice(b, 1);
            }
        }

        // ── 更新所有爆发对星星的脉冲速度 ──
        for (var i = 0; i < stars.length; i++) {
            var s = stars[i];
            for (var b = 0; b < bursts.length; b++) {
                var burst = bursts[b];
                var bdx = s.x - burst.x, bdy = s.y - burst.y;
                var bDist = Math.sqrt(bdx*bdx + bdy*bdy);
                if (bDist < burst.radius && bDist < burst.maxRadius && bDist > 2) {
                    var force = (1 - bDist / burst.maxRadius) * burst.life * 2.5 * (1 - s.layer * 0.3);
                    s.vx += (bdx / bDist) * force;
                    s.vy += (bdy / bDist) * force;
                }
            }
            s.x += s.vx;
            s.y += s.vy;

            // 速度阻尼：让星星逐渐减速，避免屏幕过于混乱
            var damping = 0.992;
            s.vx *= damping;
            s.vy *= damping;

            // 鼠标吸引效果（星座聚集）
            var mdx = mouseStarX - s.x, mdy = mouseStarY - s.y;
            var mDist = Math.sqrt(mdx*mdx + mdy*mdy);
            if (mDist < 250 && mDist > 5) {
                var attract = (250 - mDist) / 250 * 0.35 * (1 - s.layer * 0.3);
                s.x += (mdx / mDist) * attract;
                s.y += (mdy / mDist) * attract;
            }

            // 边界环绕
            if (s.x < -20) s.x = W + 20; if (s.x > W + 20) s.x = -20;
            if (s.y < -20) s.y = H + 20; if (s.y > H + 20) s.y = -20;

            // 闪烁
            var twinkle = Math.sin(frame * s.twinkleSpeed + s.twinklePhase);
            var curAlpha = s.alpha * (0.5 + 0.5 * twinkle);
            s.r = s.baseR * (0.8 + 0.2 * twinkle);

            // 鼠标附近增亮
            if (mDist < 200) {
                curAlpha = Math.min(1, curAlpha + (200 - mDist)/200 * 0.4);
                s.r *= 1 + (200 - mDist)/200 * 0.5;
            }

            // 发光效果
            var glowR = s.r * (s.layer === 0 ? 8 : 5);
            var glow = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, glowR);
            glow.addColorStop(0, s.color + (curAlpha * 0.5) + ')');
            glow.addColorStop(1, s.color + '0)');
            ctx.beginPath();
            ctx.arc(s.x, s.y, glowR, 0, Math.PI*2);
            ctx.fillStyle = glow;
            ctx.fill();

            // 核心星点
            ctx.beginPath();
            ctx.arc(s.x, s.y, s.r, 0, Math.PI*2);
            ctx.fillStyle = s.color + curAlpha + ')';
            ctx.fill();
        }

        // 鼠标轨迹光带
        for (var t = 0; t < mouseTrail.length; t++) {
            var pt = mouseTrail[t];
            pt.age++;
            var life = 1 - pt.age / 40;
            if (life <= 0) continue;
            var tr = 2 + life * 4;
            var tg = ctx.createRadialGradient(pt.x, pt.y, 0, pt.x, pt.y, tr);
            tg.addColorStop(0, 'rgba(196,181,253,' + (life * 0.35) + ')');
            tg.addColorStop(1, 'rgba(124,58,237,0)');
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, tr, 0, Math.PI*2);
            ctx.fillStyle = tg;
            ctx.fill();
        }
        while (mouseTrail.length > 0 && mouseTrail[0].age > 40) mouseTrail.shift();

        // 鼠标核心光晕
        var mGlow = ctx.createRadialGradient(mouseStarX, mouseStarY, 0, mouseStarX, mouseStarY, 60);
        mGlow.addColorStop(0, 'rgba(196,181,253,0.15)');
        mGlow.addColorStop(0.4, 'rgba(124,58,237,0.06)');
        mGlow.addColorStop(1, 'rgba(124,58,237,0)');
        ctx.beginPath();
        ctx.arc(mouseStarX, mouseStarY, 60, 0, Math.PI*2);
        ctx.fillStyle = mGlow;
        ctx.fill();

        // ── 点击爆发涟漪渲染 ──
        for (var b = 0; b < bursts.length; b++) {
            var burst = bursts[b];
            var alpha = burst.life * 0.4;
            // 外圈冲击波
            ctx.beginPath();
            ctx.arc(burst.x, burst.y, burst.radius, 0, Math.PI*2);
            ctx.strokeStyle = 'rgba(167,139,250,' + alpha + ')';
            ctx.lineWidth = 2 * burst.life;
            ctx.stroke();
            // 内圈光晕
            var bGlow = ctx.createRadialGradient(burst.x, burst.y, 0, burst.x, burst.y, burst.radius);
            bGlow.addColorStop(0, 'rgba(196,181,253,' + (alpha * 0.6) + ')');
            bGlow.addColorStop(0.4, 'rgba(124,58,237,' + (alpha * 0.2) + ')');
            bGlow.addColorStop(1, 'rgba(124,58,237,0)');
            ctx.beginPath();
            ctx.arc(burst.x, burst.y, burst.radius, 0, Math.PI*2);
            ctx.fillStyle = bGlow;
            ctx.fill();
        }

        animFrameId = requestAnimationFrame(animate);
    }

    var animFrameId = requestAnimationFrame(animate);

    // 页面不可见时暂停动画，节省 CPU
    document.addEventListener('visibilitychange', function() {
        if (document.hidden) {
            if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
        } else {
            if (!animFrameId) animFrameId = requestAnimationFrame(animate);
        }
    });
})();
</script>
""", height=0)

# ========== 交互增强：鼠标跟随光晕 + 卡片 tilt 效果 ==========
components.html("""
<script>
(function() {
    'use strict';
    let glow = window.parent.document.querySelector('.cursor-glow');
    if (!glow) {
        glow = window.parent.document.createElement('div');
        glow.className = 'cursor-glow';
        window.parent.document.body.appendChild(glow);
    }
    
    let win2 = window.parent || window;
    let doc2 = win2.document;
    let mouseX = win2.innerWidth / 2, mouseY = win2.innerHeight / 2;
    let currentX = mouseX, currentY = mouseY;
    let moveTimeout;
    
    console.log('[Orbita] Cursor glow script loaded, win size: ' + win2.innerWidth + 'x' + win2.innerHeight);
    
    if (!doc2.__obGlowMouseBound) {
        doc2.__obGlowMouseBound = true;
        doc2.addEventListener('mousemove', function(e) {
            mouseX = e.clientX; mouseY = e.clientY;
            glow.style.opacity = '1';
            clearTimeout(moveTimeout);
            moveTimeout = setTimeout(() => { glow.style.opacity = '0'; }, 150);
        }, { passive: true });
    }
    if (!doc2.__obCursorLeaveBound) {
        doc2.__obCursorLeaveBound = true;
        doc2.addEventListener('mouseleave', () => glow.style.opacity = '0');
    }
    
    (function animate() {
        currentX += (mouseX - currentX) * 0.08;
        currentY += (mouseY - currentY) * 0.08;
        glow.style.transform = 'translate3d(' + (currentX - 150) + 'px, ' + (currentY - 150) + 'px, 0)';
        requestAnimationFrame(animate);
    })();
    console.log('[Orbita] Cursor glow initialized at ' + currentX + ',' + currentY);
    
    // 卡片 3D tilt 效果（更克制的角度，避免与CSS hover冲突）
    function bindTilt() {
        const cards = window.parent.document.querySelectorAll('.card-container:not([data-tilt-bound]), [data-testid="stVerticalBlockBorderWrapper"]:not([data-tilt-bound])');
        cards.forEach(card => {
            card.dataset.tiltBound = '1';
            card.addEventListener('mousemove', function(e) {
                const rect = card.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                const centerX = rect.width / 2;
                const centerY = rect.height / 2;
                const rotateX = ((y - centerY) / centerY * -1.5).toFixed(2);
                const rotateY = ((x - centerX) / centerX * 1.5).toFixed(2);
                card.style.transform = 'perspective(1000px) rotateX(' + rotateX + 'deg) rotateY(' + rotateY + 'deg) translateY(-4px)';
            }, { passive: true });
            card.addEventListener('mouseleave', function() {
                card.style.transform = '';
            });
        });
    }
    bindTilt();
    setTimeout(bindTilt, 1500); // Streamlit动态渲染后重绑定
    
    // 按钮点击涟漪效果（品牌紫色 + 数量限制3个）
    function bindRipple() {
        const buttons = window.parent.document.querySelectorAll('.stButton > button:not([data-ripple-bound])');
        buttons.forEach(btn => {
            btn.dataset.rippleBound = '1';
            btn.style.position = 'relative';
            btn.style.overflow = 'hidden';
            btn.addEventListener('click', function(e) {
                const existing = btn.querySelectorAll('.ob-ripple');
                if (existing.length >= 3) existing[0].remove();
                const ripple = window.parent.document.createElement('span');
                ripple.className = 'ob-ripple';
                ripple.style.cssText = `
                    position: absolute; border-radius: 50%;
                    background: radial-gradient(circle, rgba(167,139,250,0.35) 0%, rgba(124,58,237,0.15) 60%, transparent 100%);
                    transform: scale(0); animation: rippleAnim 0.7s ease-out forwards;
                    pointer-events: none;
                `;
                const rect = btn.getBoundingClientRect();
                const size = Math.max(rect.width, rect.height) * 1.4;
                ripple.style.width = ripple.style.height = size + 'px';
                ripple.style.left = (e.clientX - rect.left - size/2) + 'px';
                ripple.style.top = (e.clientY - rect.top - size/2) + 'px';
                btn.appendChild(ripple);
                setTimeout(() => ripple.remove(), 700);
            });
        });
    }
    bindRipple();
    setTimeout(bindRipple, 1500);
    
    // 动态添加涟漪动画（仅添加一次）
    if (!window.parent.document.getElementById('ob-ripple-style')) {
        const style = window.parent.document.createElement('style');
        style.id = 'ob-ripple-style';
        style.textContent = '@keyframes rippleAnim { 0% { transform: scale(0); opacity: 1; } 100% { transform: scale(4); opacity: 0; } }';
        window.parent.document.head.appendChild(style);
    }
})();
</script>
""", height=0)

# ========== 加载 V2 模型 ==========
@st.cache_resource
def load_model():
    model = joblib.load("output_v2/solubility_model_v2.pkl")
    desc_names = joblib.load("output_v2/descriptor_names_v2.pkl")
    return model, desc_names

try:
    model, descriptor_names = load_model()
    model_ready = True
except Exception as e:
    st.error(f"模型加载失败: {e}")
    st.info("请先运行 'python train_model_v2.py' 训练模型")
    model_ready = False

@st.cache_resource
def get_shap_explainer():
    import shap
    return shap.TreeExplainer(model)

# ========== 加载 pKa 模型 ==========
@st.cache_resource
def load_pka_model():
    model = joblib.load("output_v2/pka_model.pkl")
    return model

try:
    pka_model = load_pka_model()
    pka_ready = True
except Exception:
    pka_ready = False

# ========== 特征计算 ==========
@st.cache_data(show_spinner=False)
def compute_features(smiles_string):
    if not smiles_string:
        return None
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None
    features = {}
    features['MolWt'] = Descriptors.MolWt(mol)
    features['LogP'] = Descriptors.MolLogP(mol)
    features['NumHDonors'] = Descriptors.NumHDonors(mol)
    features['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
    features['TPSA'] = Descriptors.TPSA(mol)
    features['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
    features['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
    features['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
    
    rdBase.DisableLog("rdApp.warning")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fp_array = np.zeros((1,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, fp_array)
    rdBase.EnableLog("rdApp.warning")
    return features, fp_array

# ========== 3D 分子展示 ==========
@st.cache_data(show_spinner=False)
def show_3d_molecule(smiles):
    try:
        import py3Dmol
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        mb = Chem.MolToMolBlock(mol)
        view = py3Dmol.view(width=500, height=420)
        view.addModel(mb, 'mol')
        view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.25}})
        view.zoomTo()
        view.setBackgroundColor('#1a1a2e')
        html = view._make_html()
        # 包裹自适应容器，防止溢出截断
        html = f'<div style="width:100%;max-width:100%;overflow:hidden;border-radius:12px;">{html}</div>'
        return html
    except Exception:
        return None


def mol_to_dark_image(mol, size=(500, 400)):
    """使用 MolDraw2DCairo 生成深色主题高清 2D 分子结构图"""
    from io import BytesIO
    from PIL import Image, ImageFilter
    import numpy as np
    from rdkit.Chem.Draw import rdMolDraw2D

    w, h = size
    BG = np.array([26, 26, 46], dtype=np.uint8)  # #1a1a2e

    # 先渲染到透明背景，再手动合成深色背景（比 backgroundColour 更可靠）
    draw = rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = draw.drawOptions()
    opts.clearBackground = False  # 不绘制背景，得到透明底
    opts.bondLineWidth = 3
    opts.multipleBondOffset = 0.18
    opts.padding = 0.08
    opts.legendFontSize = 22

    opts.updateAtomPalette({
        6:  (0.82, 0.82, 0.92),   # C: 亮银灰
        7:  (0.35, 0.65, 1.00),   # N: 亮蓝
        8:  (1.00, 0.40, 0.40),   # O: 亮红
        9:  (0.35, 0.90, 0.55),   # F: 亮绿
        16: (1.00, 0.85, 0.30),   # S: 亮金
        17: (0.35, 0.90, 0.55),   # Cl: 亮绿
        15: (1.00, 0.65, 0.20),   # P: 橙色
    })

    draw.DrawMolecule(mol)
    draw.FinishDrawing()

    png_data = draw.GetDrawingText()
    img = Image.open(BytesIO(png_data)).convert("RGBA")
    arr = np.array(img, dtype=np.float32)

    # 手动合成深色背景：透明区域 → BG，非透明区域保持原色
    alpha = arr[:, :, 3:4] / 255.0
    bg_layer = np.full((h, w, 4), np.append(BG, [255]), dtype=np.float32)
    composed = arr * alpha + bg_layer * (1 - alpha)

    # 微弱外发光（仅对分子结构像素，不影响背景）
    glow = img.filter(ImageFilter.GaussianBlur(radius=2))
    glow_arr = np.array(glow, dtype=np.float32) * alpha * 0.2
    composed = np.clip(composed + glow_arr, 0, 255).astype(np.uint8)

    return Image.fromarray(composed, "RGBA")

# ========== pKa 化学因素分析 ==========
@st.cache_data(show_spinner=False)
def analyze_pka_chemistry(smiles, pka_val):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    is_acidic = pka_val < 7
    factors = {}
    en_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in [7,8,9,17,35])
    inductive = min(en_atoms * 0.4, 3.0)
    factors['诱导效应\n(Inductive)'] = inductive if is_acidic else -inductive * 0.6
    aromatic = Descriptors.NumAromaticRings(mol)
    resonance = min(aromatic * 1.2, 3.0)
    factors['共轭效应\n(Resonance)'] = resonance if is_acidic else resonance * 0.5
    hbond_pat1 = Chem.MolFromSmarts('[OH]c1ccccc1C(=O)[OH]')
    hbond_pat2 = Chem.MolFromSmarts('[OH]c1ccccc1[OH]')
    has_hbond = False
    if hbond_pat1 and mol.HasSubstructMatch(hbond_pat1):
        has_hbond = True
    if hbond_pat2 and mol.HasSubstructMatch(hbond_pat2):
        has_hbond = True
    hbond_score = 1.5 if has_hbond else 0.0
    factors['分子内氢键\n(Intra-HB)'] = hbond_score if is_acidic else -hbond_score * 0.5
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    steric = -min(rot_bonds * 0.25, 2.0)
    factors['空间位阻\n(Steric)'] = steric if is_acidic else -steric
    sp2_score = 1.0 if aromatic > 0 else -0.5
    factors['杂化/芳香性\n(Hybridization)'] = sp2_score if is_acidic else -sp2_score
    return factors

# ========== Kimi AI 解释 ==========
def explain_with_kimi(smiles, prediction, features, shap_features=None, shap_values=None, pka_value=None, pka_type=None):
    if not KIMI_API_KEY:
        return "未配置 Kimi API Key。请在 .env 文件中写入：KIMI_API_KEY=sk-你的密钥"

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

    try:
        client = openai.OpenAI(
            api_key=KIMI_API_KEY,
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

# ========== Session State 初始化 ==========
if "smiles_input_box" not in st.session_state:
    st.session_state.smiles_input_box = ""
if "predicted_smiles" not in st.session_state:
    st.session_state.predicted_smiles = None
if "predicted_logS" not in st.session_state:
    st.session_state.predicted_logS = None
if "ai_explanation" not in st.session_state:
    st.session_state.ai_explanation = None

# ========== 网页界面 ==========
st.markdown("""
<div style="text-align:center; margin-top:1rem; margin-bottom:0.5rem;">
    <div class="tagline">MOLECULAR SOLUBILITY PREDICTION</div>
    <h1 class="gradient-title">SoluVis</h1>
    <p class="subtitle">Predict Aqueous Solubility from Molecular Structure with AI-Powered Insights</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card-container" style="padding: 1.2rem 1.5rem; margin-bottom: 2rem;">
    <p style="margin: 0; color: var(--ob-text-secondary); line-height: 1.7;">
        <b style="color: var(--ob-text-primary);">Welcome!</b> This app predicts how well a molecule dissolves in water (logS)
        using a <b>Machine Learning</b> model trained on <b>11,000+ organic compounds</b>.
        Explore molecular properties, 3D structures, pKa profiles, and AI-generated explanations.
    </p>
    <div style="display: flex; gap: 1rem; margin-top: 1rem; flex-wrap: wrap;">
        <span class="badge badge-primary"><span style="margin-right:4px;">&#128071;</span> 快速选择</span>
        <span class="badge badge-success"><span style="margin-right:4px;">&#128269;</span> 名称搜索</span>
        <span class="badge badge-warn"><span style="margin-right:4px;">&#9997;</span> SMILES 输入</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ========== 输入区域 ==========

# --- 方式1：可搜索单选列表（替换原生 selectbox 以规避 Base Web 白底样式问题）---
with st.container(border=True):
    st.markdown("""<div class="card-title">&#128071; 方式 1：快速选择常见分子</div>""", unsafe_allow_html=True)

    # 搜索过滤
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
            filtered_mols = ["(自定义输入)"]
            st.caption("未找到匹配分子，显示全部选项")
            filtered_mols = all_mols
    else:
        filtered_mols = all_mols

    # 使用 radio 替代 selectbox，选项直接渲染在 DOM 中，CSS 可完全控制样式
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

    if search_clicked and search_name:
        query = search_name.strip().lower()
        
        if query in SEARCH_INDEX:
            found_smiles = SEARCH_INDEX[query]
            st.success(f"本地精确匹配：`{search_name}` -> `{found_smiles}`")
            if found_smiles != st.session_state.smiles_input_box:
                st.session_state.smiles_input_box = found_smiles
                st.session_state.predicted_smiles = None
                st.session_state.predicted_logS = None
                st.session_state.ai_explanation = None
            st.info("点击下方的 **Predict** 按钮查看结果")
        else:
            matches = [k for k in SEARCH_INDEX.keys() if query in k or k in query]
            if matches:
                matches.sort(key=lambda x: (0 if x.startswith(query) else 1, len(x)))
                best_match = matches[0]
                found_smiles = SEARCH_INDEX[best_match]
                st.success(f"本地模糊匹配：`{search_name}` -> `{best_match}` -> `{found_smiles}`")
                if found_smiles != st.session_state.smiles_input_box:
                    st.session_state.smiles_input_box = found_smiles
                    st.session_state.predicted_smiles = None
                    st.session_state.predicted_logS = None
                    st.session_state.ai_explanation = None
                st.info("点击下方的 **Predict** 按钮查看结果")
            else:
                with st.status("本地未找到，正在查询 PubChem API...", expanded=False) as pub_status:
                    found_smiles, pub_status_str = search_pubchem_final(search_name)
                    if found_smiles:
                        pub_status.update(label=f"PubChem 匹配成功：{pub_status_str}", state="complete")
                    else:
                        pub_status.update(label=f"PubChem 未找到：{pub_status_str}", state="error")
                
                if found_smiles:
                    st.success(f"PubChem 匹配：`{search_name}` -> `{found_smiles}` ({pub_status_str})")
                    if found_smiles != st.session_state.smiles_input_box:
                        st.session_state.smiles_input_box = found_smiles
                        st.session_state.predicted_smiles = None
                        st.session_state.predicted_logS = None
                        st.session_state.ai_explanation = None
                    st.info("点击下方的 **Predict** 按钮查看结果")
                else:
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

# ========== 预测按钮 ==========
st.markdown("<br>", unsafe_allow_html=True)
btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
with btn_col2:
    predict_button = st.button("&#128302; Predict Solubility", use_container_width=True)
st.markdown("<br>", unsafe_allow_html=True)

# ========== 执行预测（带进度状态）==========
if predict_button and model_ready:
    current = st.session_state.smiles_input_box.strip()
    
    if not current:
        st.warning("请先输入或选择一个分子的 SMILES")
    else:
        with st.status("正在分析分子结构...", expanded=False) as status:
            status.update(label="Step 1/4: 解析分子结构...")
            result = compute_features(current)
            
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
                st.session_state.cached_features = features
                X_input = np.hstack([list(features.values()), fp_array]).reshape(1, -1)

                status.update(label="Step 2/4: Random Forest 预测溶解度...")
                prediction = model.predict(X_input)[0]
                st.session_state.predicted_smiles = current
                st.session_state.predicted_logS = float(prediction)
                
                if pka_ready:
                    status.update(label="Step 3/4: 预测 pKa 与电离行为...")
                    pka_pred = pka_model.predict(X_input)[0]
                    st.session_state.predicted_pka = float(pka_pred)
                
                status.update(label="Step 4/4: SHAP 可解释性分析...")
                shap_values = get_shap_explainer().shap_values(X_input)[0]
                desc_shap = shap_values[:8]
                fp_shap_sum = shap_values[8:].sum()
                combined_shap = list(desc_shap) + [fp_shap_sum]
                combined_names = [
                    "分子量 (MolWt)", "脂水分配系数 (LogP)", "氢键供体 (H-Donors)",
                    "氢键受体 (H-Acceptors)", "极性表面积 (TPSA)", "可旋转键 (Rotatable Bonds)",
                    "芳香环 (Aromatic Rings)", "脂肪环 (Aliphatic Rings)", "摩根指纹 (Morgan FP)"
                ]
                st.session_state.shap_values = combined_shap
                st.session_state.shap_names = combined_names
                st.session_state.ai_explanation = None
                status.update(label=f"分析完成！预测 logS = {float(prediction):.3f}", state="complete")

# ========== 显示预测结果（Tab分组版）==========
if st.session_state.predicted_smiles and st.session_state.predicted_logS is not None:

    features = st.session_state.get("cached_features")
    if features is None:
        result_display = compute_features(st.session_state.predicted_smiles)
        if result_display is None:
            st.error("显示时解析失败，请重新输入 SMILES")
            st.stop()
        features, _ = result_display
    prediction = st.session_state.predicted_logS

    # ── 预计算pKa相关变量（供多Tab使用）──
    pka_val = st.session_state.get("predicted_pka")
    pka_type = pka_label = pka_css = pka_text_color = pka_desc = None
    if pka_val is not None:
        if pka_val < 5:
            pka_type, pka_label, pka_css, pka_text_color, pka_desc =                     "acid", "酸性分子 (Acidic)", "pka-acid", "#a78bfa", "pKa 较低，在酸性环境中以分子态为主，脂溶性高"
        elif pka_val > 9:
            pka_type, pka_label, pka_css, pka_text_color, pka_desc =                     "base", "碱性分子 (Basic)", "pka-base", "#22d3ee", "pKa 较高，在碱性环境中以分子态为主"
        else:
            pka_type, pka_label, pka_css, pka_text_color, pka_desc =                     "amphoteric", "两性/中性 (Amphoteric/Neutral)", "pka-amphoteric", "#fbbf24", "pKa 接近中性，电离行为随 pH 变化剧烈"

    # ── 溶解度判定（供多Tab使用）──
    if prediction > 0:
        interp, color, css_class = "Highly soluble (易溶于水)", "#34d399", "result-high"
    elif prediction > -2:
        interp, color, css_class = "Moderately soluble (中等溶解)", "#fbbf24", "result-moderate"
    else:
        interp, color, css_class = "Poorly soluble (难溶于水)", "#f87171", "result-low"

    # ═════════════════════════════════════════
    # TAB 分组
    # ═════════════════════════════════════════
    tab_overview, tab_profile, tab_pharma, tab_explain = st.tabs([
        "Prediction Overview",
        "Molecular Profile",
        "Pharmacology",
        "Explainability"
    ])

    # =========================================
    # TAB 1: Prediction Overview
    # =========================================
    with tab_overview:
        st.markdown("""<div class="card-title">&#128202; Prediction Overview</div>""", unsafe_allow_html=True)

        try:
            mol = Chem.MolFromSmiles(st.session_state.predicted_smiles)
            img = mol_to_dark_image(mol, size=(400, 400))
        except Exception as e:
            img = None
            st.warning(f"结构图生成失败: {e}")

        col_left, col_right = st.columns([1, 1.2])

        with col_left:
            if img is not None:
                st.image(img, caption="Molecular Structure", use_container_width=True)
            else:
                st.info("无法显示结构图")

        with col_right:
            st.markdown("<br>", unsafe_allow_html=True)
            st.metric(label="Predicted Solubility (logS)", value=f"{prediction:.3f}")
            st.markdown(f"""
            <div class="{css_class}">
                <div style="font-size: 1.1rem; font-weight: 700; color: {color};">-> {interp}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.03); border-radius: 14px; padding: 1rem; font-size: 0.85rem; color: var(--ob-text-tertiary); border: 1px solid var(--ob-border); font-family: 'Cascadia Code', 'Consolas', monospace;">
            <b style="color: var(--ob-text-secondary);">Interpretation guide:</b><br>
            <span style="color: #34d399;">&#9679;</span> logS > 0: Very soluble (like ethanol)<br>
            <span style="color: #fbbf24;">&#9679;</span> -2 < logS < 0: Moderately soluble<br>
            <span style="color: #f87171;">&#9679;</span> logS < -2: Poorly soluble (like many drug molecules)
            </div>
            """, unsafe_allow_html=True)

        if pka_val is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""<div class="card-title">&#9889; pKa Snapshot</div>""", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Predicted pKa", f"{pka_val:.2f}")
            with c2:
                st.markdown(f"""
                <div class="{pka_css}" style="margin-top: 0.2rem;">
                    <div style="font-size: 1rem; font-weight: 700; color: {pka_text_color};">-> {pka_label}</div>
                    <div style="font-size: 0.8rem; color: var(--ob-text-tertiary); margin-top: 0.3rem;">{pka_desc}</div>
                </div>
                """, unsafe_allow_html=True)

    # =========================================
    # TAB 2: Molecular Profile
    # =========================================
    with tab_profile:
        st.markdown("""<div class="card-title">&#128202; Molecular Descriptors</div>""", unsafe_allow_html=True)
        with st.container(border=True):
            desc_col1, desc_col2, desc_col3, desc_col4 = st.columns(4)
            with desc_col1:
                st.metric("Molecular Weight", f"{features['MolWt']:.1f}")
                st.metric("LogP (Hydrophobicity)", f"{features['LogP']:.2f}")
            with desc_col2:
                st.metric("H-Bond Donors", f"{features['NumHDonors']}")
                st.metric("H-Bond Acceptors", f"{features['NumHAcceptors']}")
            with desc_col3:
                st.metric("TPSA (A2)", f"{features['TPSA']:.1f}")
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

        if pka_val is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""<div class="card-title">&#129516; Structural Chemistry: pKa Deep Dive</div>""", unsafe_allow_html=True)
            chem_factors = analyze_pka_chemistry(st.session_state.predicted_smiles, pka_val)
            col_3d, col_chem = st.columns([1, 1])
            with col_3d:
                st.markdown("<div style='font-weight: 600; color: var(--ob-text-secondary); margin-bottom: 0.5rem; font-family: \"Space Grotesk\", sans-serif;'>&#127919; 3D 球棍模型（可旋转缩放）</div>", unsafe_allow_html=True)
                html_3d = show_3d_molecule(st.session_state.predicted_smiles)
                if html_3d:
                    components.html(html_3d, height=420, scrolling=False)
                else:
                    st.info("3D 模型生成失败（需安装 py3Dmol）")
            with col_chem:
                if chem_factors:
                    import matplotlib.pyplot as plt
                    import matplotlib.font_manager as fm
                    for font in fm.fontManager.ttflist:
                        if font.name in ('Noto Sans CJK SC', 'Noto Sans CJK'):
                            plt.rcParams['font.family'] = font.name
                            break
                    plt.rcParams['axes.unicode_minus'] = False
                    names = list(chem_factors.keys())
                    vals = list(chem_factors.values())
                    colors = ['#a78bfa' if v > 0 else '#22d3ee' for v in vals]
                    plt.rcParams['figure.facecolor'] = '#0d0d14'
                    plt.rcParams['axes.facecolor'] = '#1a1a2e'
                    plt.rcParams['axes.edgecolor'] = '#33334d'
                    plt.rcParams['axes.labelcolor'] = '#a0a0b0'
                    plt.rcParams['xtick.color'] = '#a0a0b0'
                    plt.rcParams['ytick.color'] = '#a0a0b0'
                    plt.rcParams['text.color'] = '#f0f0f5'
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
                    from matplotlib.patches import Patch
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
                else:
                    st.info("化学因素分析暂不可用")

    # =========================================
    # TAB 3: Pharmacology
    # =========================================
    with tab_pharma:
        if pka_val is not None:
            st.markdown("""<div class="card-title">&#9889; pKa & Ionization Profile</div>""", unsafe_allow_html=True)
            col_pka1, col_pka2 = st.columns([1, 1.2])
            with col_pka1:
                st.markdown("<br>", unsafe_allow_html=True)
                st.metric("Predicted pKa", f"{pka_val:.2f}")
                st.markdown(f"""
                <div class="{pka_css}" style="margin-top: 0.8rem;">
                    <div style="font-size: 1.1rem; font-weight: 700; color: {pka_text_color};">-> {pka_label}</div>
                    <div style="font-size: 0.85rem; color: var(--ob-text-tertiary); margin-top: 0.4rem;">{pka_desc}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_pka2:
                import matplotlib.pyplot as plt
                import matplotlib.font_manager as fm
                try:
                    for font in fm.fontManager.ttflist:
                        if font.name in ('Noto Sans CJK SC', 'Noto Sans CJK'):
                            plt.rcParams['font.family'] = font.name
                            break
                except Exception:
                    pass
                plt.rcParams['axes.unicode_minus'] = False
                env_ph = [1.5, 4.5, 6.8, 7.4]
                env_names = ['Stomach\n胃', 'Duodenum\n十二指肠', 'Small Intestine\n小肠', 'Blood/Brain\n血液/脑']
                if pka_type == "acid":
                    fractions = [1 / (1 + 10**(ph - pka_val)) for ph in env_ph]
                else:
                    fractions = [1 / (1 + 10**(pka_val - ph)) for ph in env_ph]
                plt.rcParams['figure.facecolor'] = '#0a0a0f'
                plt.rcParams['axes.facecolor'] = '#1e1e2e'
                plt.rcParams['axes.edgecolor'] = '#2a2a3a'
                plt.rcParams['axes.labelcolor'] = '#a0a0b0'
                plt.rcParams['xtick.color'] = '#a0a0b0'
                plt.rcParams['ytick.color'] = '#a0a0b0'
                plt.rcParams['text.color'] = '#f0f0f5'
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
            st.markdown("""<div class="card-title">&#128138; 药理学分析</div>""", unsafe_allow_html=True)
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
            st.markdown("""<div class="card-title">&#128279; 溶解度 x pKa 联动分析</div>""", unsafe_allow_html=True)
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

    # =========================================
    # TAB 4: Explainability
    # =========================================
    with tab_explain:
        st.markdown("""<div class="card-title">&#128269; SHAP Explainability</div>""", unsafe_allow_html=True)
        st.caption("基于 SHAP (SHapley Additive exPlanations) 分析每个特征对预测的贡献")
        if "shap_values" in st.session_state:
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm
            import numpy as np
            import glob
            fm.fontManager = fm.FontManager()
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
            chinese_font = None
            for font in fm.fontManager.ttflist:
                if font.name in ('Noto Sans CJK SC', 'Noto Sans CJK'):
                    chinese_font = font.name
                    break
                if 'WenQuanYi' in font.name or 'Source Han Sans SC' in font.name:
                    chinese_font = font.name
                    break
            if chinese_font:
                plt.rcParams['font.family'] = chinese_font
            plt.rcParams['axes.unicode_minus'] = False
            shap_vals = np.array(st.session_state.shap_values)
            names = st.session_state.shap_names
            abs_vals = np.abs(shap_vals)
            sorted_idx = np.argsort(abs_vals)[::-1][:8]
            top_shap = shap_vals[sorted_idx]
            top_names = [names[i] for i in sorted_idx]
            colors = ['#a78bfa' if v > 0 else '#06b6d4' for v in top_shap]
            plt.rcParams['figure.facecolor'] = '#0a0a0f'
            plt.rcParams['axes.facecolor'] = '#1e1e2e'
            plt.rcParams['axes.edgecolor'] = '#2a2a3a'
            plt.rcParams['axes.labelcolor'] = '#a0a0b0'
            plt.rcParams['xtick.color'] = '#a0a0b0'
            plt.rcParams['ytick.color'] = '#a0a0b0'
            plt.rcParams['text.color'] = '#f0f0f5'
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
            ev = get_shap_explainer().expected_value
            if isinstance(ev, (list, tuple, np.ndarray)):
                base_value = float(np.array(ev).flatten()[0])
            else:
                base_value = float(ev)
            ax.set_title(f"预测值: {prediction:.3f}  (基准值: {base_value:.3f})", fontsize=12, pad=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            from matplotlib.patches import Patch
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

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div class="card-title">&#129504; AI Chemistry Explanation</div>""", unsafe_allow_html=True)
        with st.container(border=True):
            if st.session_state.ai_explanation:
                st.markdown(st.session_state.ai_explanation)
                if st.button("清除解释", key="clear_ai"):
                    st.session_state.ai_explanation = None
                    st.rerun()
            else:
                st.caption("AI 解释需要手动调用（消耗 API 额度）")
                if st.button("生成 AI 解释", key="gen_ai", use_container_width=True):
                    with st.spinner("正在分析分子结构..."):
                        pka_val_gen = st.session_state.get("predicted_pka")
                        if pka_val_gen is not None:
                            if pka_val_gen < 5:
                                pka_type_gen = "acid"
                            elif pka_val_gen > 9:
                                pka_type_gen = "base"
                            else:
                                pka_type_gen = "amphoteric"
                        else:
                            pka_val_gen = None
                            pka_type_gen = None
                        explanation = explain_with_kimi(
                            st.session_state.predicted_smiles,
                            prediction,
                            features,
                            shap_features=st.session_state.get("shap_names"),
                            shap_values=st.session_state.get("shap_values"),
                            pka_value=pka_val_gen,
                            pka_type=pka_type_gen
                        )
                    st.session_state.ai_explanation = explanation
                    st.rerun()

# ========== 页脚 ==========
st.markdown("""
<div class="footer">
    <div style="font-weight: 600; color: var(--ob-text-secondary); margin-bottom: 0.3rem; font-family: 'Space Grotesk', sans-serif; font-size: 1rem;">SoluVis</div>
    <div>Built with Streamlit | ML: Random Forest + RDKit (V2: 11,000+ molecules) | AI: Kimi (Moonshot AI) | DB: 100+ local + PubChem API</div>
    <div style="margin-top: 0.5rem; font-size: 0.75rem; color: #6b6b7b;">科学计算 - 人工智能 - 药物化学 | Orbita Deep Space Theme</div>
</div>
""", unsafe_allow_html=True)
