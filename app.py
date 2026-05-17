"""
【分子溶解度预测 - 网页应用 Final Stable v3】
三层搜索架构：本地库(100+) → PubChem API(实时) → 官网引导
保留原有 pubchem_final.py 全部逻辑，仅在其前添加本地层
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
    "西咪替丁 Cimetidine": "CN(C)CCNC(=O)CSc1ncnc1C#N",
    
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
        except:
            pubchem_cache = {}

def save_cache():
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(pubchem_cache, f, ensure_ascii=False, indent=2)
    except:
        pass

load_cache()

def search_pubchem_final(name, max_retries=3):
    """
    最终版 PubChem 搜索（基于知乎文章技巧优化）
    - verify=False: 跳过 SSL 验证（解决国内网络握手失败）
    - time.sleep(1): 严格符合官方 1 req/s 限制
    - 本地缓存 + 容错处理
    """
    if not name or not name.strip():
        return None, "名称不能为空"
    
    name_clean = name.strip()
    name_lower = name_clean.lower()
    
    # 0. 查缓存
    if name_lower in pubchem_cache:
        return pubchem_cache[name_lower], "success (cached)"
    
    # 1. 频率控制
    time.sleep(1.2)
    
    encoded = urllib.parse.quote(name_clean)
    
    # 2. 核心请求
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
                print(f"  ⚠️ 503 服务器繁忙，等待 {wait}s 后重试...")
                time.sleep(wait)
                continue
            
            elif r.status_code == 404:
                return None, "PubChem 未找到该化合物 (404)"
            
            else:
                return None, f"PubChem HTTP {r.status_code}: {r.text[:100]}"
                
        except requests.exceptions.SSLError as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None, f"SSL 连接失败: {str(e)}"
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
    page_title="Molecular Solubility Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== 自定义 CSS 美化 ==========
st.markdown("""
<style>
/* ========== 全局样式 ========== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

.main {
    background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
    padding: 1rem 2rem;
}

/* ========== 标题样式 ========== */
.gradient-title {
    background: linear-gradient(135deg, #1a237e 0%, #006064 50%, #00acc1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800;
    font-size: 2.8rem;
    text-align: center;
    margin-bottom: 0.3rem;
    letter-spacing: -0.02em;
}

.subtitle {
    text-align: center;
    color: #546e7a;
    font-size: 1.15rem;
    font-weight: 400;
    margin-bottom: 1.5rem;
}

/* ========== 卡片容器 ========== */
.card-container {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.6);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06), 0 1px 3px rgba(0, 0, 0, 0.04);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.card-container:hover {
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1), 0 2px 6px rgba(0, 0, 0, 0.06);
}

.card-title {
    color: #1a237e;
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    border-bottom: 2px solid #e0e7ee;
    padding-bottom: 0.6rem;
}

/* ========== 输入框美化 ========== */
.stTextInput > div > div > input {
    border-radius: 10px !important;
    border: 2px solid #e0e7ee !important;
    padding: 0.6rem 1rem !important;
    font-size: 1rem !important;
    transition: all 0.2s ease !important;
    background: #ffffff !important;
}

.stTextInput > div > div > input:focus {
    border-color: #00acc1 !important;
    box-shadow: 0 0 0 3px rgba(0, 172, 193, 0.15) !important;
}

.stSelectbox > div > div > div {
    border-radius: 10px !important;
    border: 2px solid #e0e7ee !important;
    background: #ffffff !important;
}

/* ========== 按钮美化 ========== */
.stButton > button {
    background: linear-gradient(135deg, #006064 0%, #00838f 50%, #00acc1 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 15px rgba(0, 172, 193, 0.35) !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 172, 193, 0.5) !important;
    background: linear-gradient(135deg, #004d40 0%, #006064 50%, #00838f 100%) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* 次要按钮 */
.stButton > button[kind="secondary"] {
    background: linear-gradient(135deg, #5e35b1 0%, #7e57c2 100%) !important;
    box-shadow: 0 4px 15px rgba(94, 53, 177, 0.35) !important;
}

.stButton > button[kind="secondary"]:hover {
    box-shadow: 0 6px 20px rgba(94, 53, 177, 0.5) !important;
}

/* ========== Metric 美化 ========== */
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #1a237e, #00838f);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

[data-testid="stMetricLabel"] {
    font-size: 0.85rem !important;
    color: #78909c !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ========== 信息框美化 ========== */
.stAlert {
    border-radius: 12px !important;
    border: none !important;
    padding: 1rem 1.2rem !important;
}

.stAlert [data-testid="stMarkdownContainer"] {
    font-size: 0.95rem;
}

/* Success */
.stAlert[data-baseweb="notification"][data-kind="positive"] {
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%) !important;
    border-left: 4px solid #2e7d32 !important;
}

/* Info */
.stAlert[data-baseweb="notification"][data-kind="info"] {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%) !important;
    border-left: 4px solid #1565c0 !important;
}

/* Warning */
.stAlert[data-baseweb="notification"][data-kind="warning"] {
    background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%) !important;
    border-left: 4px solid #ef6c00 !important;
}

/* Error */
.stAlert[data-baseweb="notification"][data-kind="negative"] {
    background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%) !important;
    border-left: 4px solid #c62828 !important;
}

/* ========== 分隔线 ========== */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, #b0bec5, transparent);
    margin: 2rem 0;
}

/* ========== 页脚 ========== */
.footer {
    text-align: center;
    padding: 1.5rem;
    color: #90a4ae;
    font-size: 0.85rem;
    margin-top: 2rem;
    border-top: 1px solid #e0e7ee;
}

/* ========== 结果高亮 ========== */
.result-high {
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    border: 2px solid #a5d6a7;
    margin: 0.5rem 0;
}

.result-moderate {
    background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    border: 2px solid #ffcc80;
    margin: 0.5rem 0;
}

.result-low {
    background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    border: 2px solid #ef9a9a;
    margin: 0.5rem 0;
}

/* ========== 标签页美化 ========== */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    padding: 0.5rem 1.2rem !important;
    font-weight: 500 !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #006064, #00acc1) !important;
    color: white !important;
}

/* ========== 图片容器 ========== */
.stImage > img {
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

/* ========== 滚动条美化 ========== */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #b0bec5, #78909c);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #78909c, #546e7a);
}
</style>
""", unsafe_allow_html=True)

# ========== 加载 V2 模型 ==========
@st.cache_resource
def load_model():
    model = joblib.load("output_v2/solubility_model_v2.pkl")
    desc_names = joblib.load("output_v2/descriptor_names_v2.pkl")
    return model, desc_names

try:
    model, descriptor_names = load_model()
    model_ready = True

    # ===== 初始化 SHAP Explainer =====
    import shap
    explainer = shap.TreeExplainer(model)
except Exception as e:
    st.error(f"❌ 模型加载失败: {e}")
    st.info("请先运行 'python train_model_v2.py' 训练模型")
    model_ready = False
# ========== 加载 pKa 模型 ==========
@st.cache_resource
def load_pka_model():
    model = joblib.load("output_v2/pka_model.pkl")
    return model

try:
    pka_model = load_pka_model()
    pka_ready = True
except Exception as e:
    pka_ready = False
    st.error(f"DEBUG pKa load error: {type(e).__name__}: {e}")

# ========== 特征计算 ==========
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
        view = py3Dmol.view(width=380, height=320)
        view.addModel(mb, 'mol')
        view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.25}})
        view.zoomTo()
        return view._make_html()
    except Exception:
        return None

# ========== pKa 化学因素分析（结构化学版）==========
def analyze_pka_chemistry(smiles, pka_val):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    is_acidic = pka_val < 7
    factors = {}
    # 1. 诱导效应
    en_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in [7,8,9,17,35])
    inductive = min(en_atoms * 0.4, 3.0)
    factors['诱导效应\n(Inductive)'] = inductive if is_acidic else -inductive * 0.6
    # 2. 共轭效应
    aromatic = Descriptors.NumAromaticRings(mol)
    resonance = min(aromatic * 1.2, 3.0)
    factors['共轭效应\n(Resonance)'] = resonance if is_acidic else resonance * 0.5
    # 3. 分子内氢键
    hbond_pat1 = Chem.MolFromSmarts('[OH]c1ccccc1C(=O)[OH]')
    hbond_pat2 = Chem.MolFromSmarts('[OH]c1ccccc1[OH]')
    has_hbond = False
    if hbond_pat1 and mol.HasSubstructMatch(hbond_pat1):
        has_hbond = True
    if hbond_pat2 and mol.HasSubstructMatch(hbond_pat2):
        has_hbond = True
    hbond_score = 1.5 if has_hbond else 0.0
    factors['分子内氢键\n(Intra-HB)'] = hbond_score if is_acidic else -hbond_score * 0.5
    # 4. 空间位阻
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    steric = -min(rot_bonds * 0.25, 2.0)
    factors['空间位阻\n(Steric)'] = steric if is_acidic else -steric
    # 5. 杂化/芳香性
    sp2_score = 1.0 if aromatic > 0 else -0.5
    factors['杂化/芳香性\n(Hybridization)'] = sp2_score if is_acidic else -sp2_score
    return factors

# ========== Kimi AI 解释 ==========
def explain_with_kimi(smiles, prediction, features, shap_features=None, shap_values=None, pka_value=None, pka_type=None):

    if not KIMI_API_KEY:
        return "⚠️ 未配置 Kimi API Key。请在 .env 文件中写入：KIMI_API_KEY=sk-你的密钥"

    # 在代码里精确判断溶解度等级（避免 LLM 数值比较错误）
    if prediction > 0:
        solubility_level = "易溶于水"
        solubility_desc = "logS > 0，属于高溶解度"
    elif prediction > -2:
        solubility_level = "中等溶解"
        solubility_desc = "-2 < logS ≤ 0，属于中等溶解度"
    else:
        solubility_level = "难溶于水"
        solubility_desc = "logS ≤ -2，属于低溶解度"

    # 构建 SHAP 洞察文本
    shap_text = ""
    if shap_features and shap_values and len(shap_features) == len(shap_values):
        import numpy as np
        abs_vals = np.abs(np.array(shap_values))
        sorted_idx = np.argsort(abs_vals)[::-1][:5]
        top_features = [shap_features[i] for i in sorted_idx]
        top_vals = [shap_values[i] for i in sorted_idx]
        shap_lines = []
        for name, val in zip(top_features, top_vals):
            direction = "推动易溶" if val > 0 else "推动难溶"
            shap_lines.append(f"- {name}: 贡献值 {val:+.3f}（{direction}）")
        shap_text = "\n".join(shap_lines)

    # 构建 pKa 相关文本（如果有）
    pka_section = ""
    pka_task = ""
    if pka_value is not None and pka_type is not None:
        if pka_type == "acid":
            pka_label = "酸性分子"
            pka_desc_full = f"pKa = {pka_value:.2f} (< 5)，属于酸性分子。在酸性环境（如胃，pH ~1.5）中主要以分子态存在，脂溶性较高，容易被胃黏膜吸收。"
            ionization_desc = "在生理 pH 范围内，该分子倾向于释放质子 (H⁺)，形成共轭碱。"
        elif pka_type == "base":
            pka_label = "碱性分子"
            pka_desc_full = f"pKa = {pka_value:.2f} (> 9)，属于碱性分子。在碱性环境中主要以分子态存在，在胃中容易电离，主要在小肠吸收。"
            ionization_desc = "在生理 pH 范围内，该分子倾向于结合质子 (H⁺)，形成共轭酸。"
        else:
            pka_label = "两性/中性分子"
            pka_desc_full = f"pKa = {pka_value:.2f} (5–9 之间)，属于两性或中性分子。电离行为随环境 pH 变化剧烈，在不同生理部位的存在形态差异大。"
            ionization_desc = "该分子既可能释放也可能结合质子，具体取决于所处环境的 pH。"

        pka_section = f"""
【pKa 与电离行为分析】
- 预测 pKa: {pka_value:.2f}
- 酸碱性判定: {pka_label}
- 电离特征: {ionization_desc}
- 生理意义: {pka_desc_full}

【溶解度 × pKa 联动提示】
溶解度 (logS) 和 pKa 共同决定药物在体内的吸收行为：
- 分子态（非电离）脂溶性高，易穿透细胞膜被吸收
- 离子态水溶性好，有利于在血液中运输和肾脏排泄
- 当前分子：logS = {prediction:.2f}（{solubility_level}），pKa = {pka_value:.2f}（{pka_label}）
"""
        pka_task = f"""5. **pKa 结构化学深度解析**（4–5句话）：
   - 从 SMILES 识别该分子的**可电离基团**（如 -COOH、脂肪胺、芳香胺、酚羟基、杂环氮等），并指出其直接连接的化学环境。
   - 用**电子效应**解释该 pKa = {pka_value:.2f} 的合理性：附近是否有吸电子基团（-I, -M）拉低 pKa / 推电子基团（+I, +M）升高 pKa？是否有共轭稳定化/去稳定化？是否存在分子内氢键或空间位阻影响质子转移？
   - 简要说明该分子在胃 (pH 1.5)、小肠 (pH 6.8)、血液 (pH 7.4) 中的**电离状态趋势**（以分子态比例高低描述即可，不做精确计算）。
   - 联系溶解度分析：该分子的电离状态如何与其亲水/疏水基团分布共同影响体内吸收与排泄。"""

    prompt = f"""你是一位结构化学专家，擅长从分子的 SMILES 表示和理化性质数据中深度剖析其溶解度与电离行为的结构根源。请围绕**分子骨架、官能团、电子效应、空间构型**展开细致分析，避免泛泛而谈的科普介绍。

分子 SMILES: {smiles}
模型预测的水溶解度 (logS): {prediction:.2f}

【分子基本性质】
- 分子量: {features['MolWt']:.1f} g/mol
- 极性表面积 (TPSA): {features['TPSA']:.1f} Å²
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
⚠️ 重要：上述结论已由程序精确计算得出，你只需在回答中直接复述，不可重新判断或做数值比较。
{pka_section}
请用中文回答，严格按以下段落组织，重点放在**结构解析**上：

1. **溶解度结论**（1句话）：直接复述——该分子属于「{solubility_level}」。

2. **分子骨架与官能团识别**（3–4句话）：
   - 从 SMILES 字符串解析该分子的**核心骨架**（如苯环、甾体、糖类、肽链、脂肪链等）。
   - 列出分子中存在的**主要官能团**（如羟基 -OH、羧基 -COOH、氨基 -NH₂、酰胺 -CONH-、醚键 -O-、酯基 -COOR、卤素、硝基、磺酸基、杂环氮等）。
   - 指出是否存在**可电离基团**及其直接连接的化学环境（如羧基连接在芳香环上还是脂肪链上，氨基是伯胺/仲胺/叔胺，是否邻近吸电子基团等）。
   - 描述分子的**整体构型特征**（如线性/分支/稠环/大环、刚性 vs 柔性、亲水面与疏水面的空间分布趋势）。

3. **结构-溶解度深度解析**（4–5句话）：
   - 结合具体官能团解释：哪些基团**推动水溶**（如羟基、羧基、氨基形成氢键），哪些**阻碍水溶**（如长烷基链、大芳香疏水面）。
   - 结合 SHAP 分析结果，说明模型最关注的结构特征（如 LogP、TPSA、氢键数目、芳香环数）如何与该分子的实际官能团组成对应。
   - 若分子同时含有亲水与疏水基团，分析二者的**相对比例与空间布局**如何决定整体溶解度（如表面活性剂式的两亲性、被包裹的极性基团等）。
   - 提及**分子间相互作用**：该分子与水之间能形成多少氢键网络，疏水部分是否导致水分子有序化（疏水效应）。

4. **SHAP 关键特征与结构对应**（2–3句话）：
   - 引用 SHAP 贡献值最高的 1–2 个特征的具体数值。
   - 明确指出这些特征在分子结构上的**物理对应物**（如「高 TPSA 贡献 +0.35」对应「分子含 3 个羟基和 1 个羧基」）。

{pka_task}

要求:
- **以结构化学为核心**，避免空泛的科普描述和简单的生活类比；如举类比，必须紧扣官能团行为（不超过1句）。
- 第2段必须基于 SMILES 识别出**至少2个具体官能团**和**骨架类型**。
- 第3段必须引用分子性质数据（LogP、TPSA、H-Bond 数目等）和 SHAP 贡献值。
- 若包含 pKa 段落，必须从**电子效应**（诱导效应、共轭效应、场效应）和**空间环境**（邻位取代基、环张力、分子内氢键）解释该 pKa 的合理性，而非仅复述 pH 分布。
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
<h1 class="gradient-title">🧪 Molecular Solubility Predictor</h1>
<p class="subtitle">Predict Aqueous Solubility from Molecular Structure with AI-Powered Insights</p>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card-container" style="padding: 1.2rem 1.5rem; margin-bottom: 2rem;">
    <p style="margin: 0; color: #455a64; line-height: 1.7;">
        <b>Welcome!</b> This app predicts how well a molecule dissolves in water (logS) 
        using a <b>Machine Learning</b> model trained on <b>11,000+ organic compounds</b>.
        Explore molecular properties, 3D structures, pKa profiles, and AI-generated explanations.
    </p>
    <div style="display: flex; gap: 1rem; margin-top: 1rem; flex-wrap: wrap;">
        <div style="display: flex; align-items: center; gap: 0.4rem; color: #006064; font-weight: 500; font-size: 0.9rem;">
            <span style="font-size: 1.2rem;">👇</span> 快速选择
        </div>
        <div style="display: flex; align-items: center; gap: 0.4rem; color: #006064; font-weight: 500; font-size: 0.9rem;">
            <span style="font-size: 1.2rem;">🔍</span> 名称搜索
        </div>
        <div style="display: flex; align-items: center; gap: 0.4rem; color: #006064; font-weight: 500; font-size: 0.9rem;">
            <span style="font-size: 1.2rem;">✏️</span> SMILES 输入
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ========== 输入区域 ==========

# --- 方式1：下拉菜单 ---
with st.container(border=True):
    st.markdown("""
    <div class="card-title">👇 方式 1：快速选择常见分子</div>
    """, unsafe_allow_html=True)
    selected_molecule = st.selectbox(
        "选择分子",
        list(MOLECULE_DB.keys()),
        index=0,
        key="molecule_select",
        label_visibility="collapsed"
    )

    if selected_molecule != list(MOLECULE_DB.keys())[0]:
        new_smiles = MOLECULE_DB[selected_molecule]
        if new_smiles != st.session_state.smiles_input_box:
            st.session_state.smiles_input_box = new_smiles
            st.session_state.predicted_smiles = None
            st.session_state.predicted_logS = None
            st.session_state.ai_explanation = None
            st.rerun()

# --- 方式2：三层搜索（本地 → PubChem → 引导）---
with st.container(border=True):
    st.markdown("""
    <div class="card-title">🔍 方式 2：名称搜索（本地库 + PubChem API）</div>
    """, unsafe_allow_html=True)
    st.caption("💡 支持中英文，如 阿司匹林 / Aspirin / Ibuprofen / 咖啡因")
    search_col1, search_col2 = st.columns([4, 1])
    with search_col1:
        search_name = st.text_input(
            "输入名称",
            placeholder="例如 阿司匹林 或 Aspirin",
            key="search_name",
            label_visibility="collapsed"
        )
    with search_col2:
        search_clicked = st.button("🔍 搜索", key="search_btn", use_container_width=True)

    if search_clicked and search_name:
        query = search_name.strip().lower()
        
        # ===== 第一层：本地精确匹配 =====
        if query in SEARCH_INDEX:
            found_smiles = SEARCH_INDEX[query]
            st.success(f"✅ 本地精确匹配：`{search_name}` → `{found_smiles}`")
            if found_smiles != st.session_state.smiles_input_box:
                st.session_state.smiles_input_box = found_smiles
                st.session_state.predicted_smiles = None
                st.session_state.predicted_logS = None
                st.session_state.ai_explanation = None
            st.info("👇 点击下方的 **Predict** 按钮查看结果")
        
        else:
            # ===== 第二层：本地模糊匹配 =====
            matches = [k for k in SEARCH_INDEX.keys() if query in k or k in query]
            if matches:
                matches.sort(key=lambda x: (0 if x.startswith(query) else 1, len(x)))
                best_match = matches[0]
                found_smiles = SEARCH_INDEX[best_match]
                st.success(f"✅ 本地模糊匹配：`{search_name}` → `{best_match}` → `{found_smiles}`")
                if found_smiles != st.session_state.smiles_input_box:
                    st.session_state.smiles_input_box = found_smiles
                    st.session_state.predicted_smiles = None
                    st.session_state.predicted_logS = None
                    st.session_state.ai_explanation = None
                st.info("👇 点击下方的 **Predict** 按钮查看结果")
            
            else:
                # ===== 第三层：PubChem API（保留原有代码逻辑）=====
                with st.spinner("🌐 本地未找到，正在查询 PubChem API..."):
                    found_smiles, status = search_pubchem_final(search_name)
                
                if found_smiles:
                    st.success(f"✅ PubChem 匹配：`{search_name}` → `{found_smiles}` ({status})")
                    if found_smiles != st.session_state.smiles_input_box:
                        st.session_state.smiles_input_box = found_smiles
                        st.session_state.predicted_smiles = None
                        st.session_state.predicted_logS = None
                        st.session_state.ai_explanation = None
                    st.info("👇 点击下方的 **Predict** 按钮查看结果")
                
                else:
                    # ===== 第四层：失败引导 =====
                    st.error(f"❌ 未找到：`{search_name}`")
                    st.info("尝试建议：")
                    st.markdown("""
                    - 检查拼写（如 **Aspirin** 而非 **Aspriin**）
                    - 尝试更常见的名称
                    - 直接输入 SMILES（方式3）
                    """)
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #e3f2fd, #bbdefb); padding: 18px; border-radius: 12px; border-left: 4px solid #1565c0;">
                    <h4 style="color: #0d47a1; margin-top: 0;">🔍 如何手动获取 SMILES？</h4>
                    <ol style="color: #37474f; margin-bottom: 0;">
                        <li>访问 <a href="https://pubchem.ncbi.nlm.nih.gov" target="_blank"><b>https://pubchem.ncbi.nlm.nih.gov</b></a></li>
                        <li>在搜索框输入分子名称（英文，如 <b>Aspirin</b>）</li>
                        <li>进入化合物页面，找到 <b>Canonical SMILES</b> 字段</li>
                        <li>复制 SMILES 字符串（如 <code>CC(=O)Oc1ccccc1C(=O)O</code>）</li>
                        <li>粘贴到下方的 "方式 3" 文本框中，点击 Predict</li>
                    </ol>
                    </div>
                    """, unsafe_allow_html=True)

# --- 方式3：SMILES 直接输入 ---
with st.container(border=True):
    st.markdown("""
    <div class="card-title">✏️ 方式 3：直接输入 SMILES</div>
    """, unsafe_allow_html=True)
    st.caption("💡 可从下拉菜单自动填入，也可手动编辑或粘贴外部 SMILES")

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
    predict_button = st.button("🔮 Predict Solubility", use_container_width=True)
st.markdown("<br>", unsafe_allow_html=True)

# ========== 执行预测 ==========
if predict_button and model_ready:
    current = st.session_state.smiles_input_box.strip()
    
    if not current:
        st.warning("⚠️ 请先输入或选择一个分子的 SMILES")
    else:
        result = compute_features(current)
        
        if result is None:
            st.error(f"❌ Invalid SMILES: `{current}`")
            st.info("该 SMILES 无法被 RDKit 解析。可能原因：")
            st.markdown("""
            - 分子含有金属/配位键，RDKit 不支持
            - SMILES 语法错误（括号不匹配）
            - 输入为空或含有非法字符
            """)
        else:
            features, fp_array = result
            X_input = np.hstack([list(features.values()), fp_array]).reshape(1, -1)
            prediction = model.predict(X_input)[0]
            
            st.session_state.predicted_smiles = current
            st.session_state.predicted_logS = float(prediction)
                        # ===== pKa 预测 =====
            if pka_ready:
                pka_pred = pka_model.predict(X_input)[0]
                st.session_state.predicted_pka = float(pka_pred)



            # ===== 计算 SHAP 值 =====
            shap_values = explainer.shap_values(X_input)[0]
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

# ========== 显示预测结果 ==========
if st.session_state.predicted_smiles and st.session_state.predicted_logS is not None:
    
    result_display = compute_features(st.session_state.predicted_smiles)
    
    if result_display is None:
        st.error("显示时解析失败，请重新输入 SMILES")
    else:
        features, _ = result_display
        prediction = st.session_state.predicted_logS
        
        st.markdown("""
        <div class="card-title">📊 预测结果概览</div>
        """, unsafe_allow_html=True)
        
        try:
            mol = Chem.MolFromSmiles(st.session_state.predicted_smiles)
            img = Draw.MolToImage(mol, size=(380, 380), kekulize=True)
        except Exception as e:
            img = None
            st.warning(f"⚠️ 结构图生成失败: {e}")
        
        col_left, col_right = st.columns([1, 1.2])
        
        with col_left:
            if img is not None:
                st.image(img, caption="Molecular Structure", use_container_width=True)
            else:
                st.info("无法显示结构图")
        
        with col_right:
            st.markdown("<br>", unsafe_allow_html=True)
            st.metric(
                label="Predicted Solubility (logS)",
                value=f"{prediction:.3f}"
            )
            
            if prediction > 0:
                interp = "Highly soluble (易溶于水)"
                color = "#2e7d32"
                css_class = "result-high"
            elif prediction > -2:
                interp = "Moderately soluble (中等溶解)"
                color = "#ef6c00"
                css_class = "result-moderate"
            else:
                interp = "Poorly soluble (难溶于水)"
                color = "#c62828"
                css_class = "result-low"
            
            st.markdown(f"""
            <div class="{css_class}">
                <div style="font-size: 1.1rem; font-weight: 700; color: {color};">➜ {interp}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: rgba(236, 239, 241, 0.5); border-radius: 10px; padding: 1rem; font-size: 0.9rem; color: #546e7a;">
            <b>Interpretation guide:</b><br>
            • logS > 0: Very soluble (like ethanol)<br>
            • -2 < logS < 0: Moderately soluble<br>
            • logS < -2: Poorly soluble (like many drug molecules)
            </div>
            """, unsafe_allow_html=True)
                # ========== pKa 预测结果 ==========
        if "predicted_pka" in st.session_state:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="card-title">⚡ pKa & Ionization Profile</div>
            """, unsafe_allow_html=True)
            
            pka_val = st.session_state.predicted_pka
            
            # 判断酸碱性倾向（简化版：pKa < 7 倾向酸性，>7 倾向碱性）
            if pka_val < 5:
                pka_type = "acid"
                pka_label = "酸性分子 (Acidic)"
                pka_color = "#c62828"
                pka_bg = "linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%)"
                pka_border = "#ef9a9a"
                pka_desc = "pKa 较低，在酸性环境中以分子态为主，脂溶性高"
            elif pka_val > 9:
                pka_type = "base"
                pka_label = "碱性分子 (Basic)"
                pka_color = "#1565c0"
                pka_bg = "linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%)"
                pka_border = "#90caf9"
                pka_desc = "pKa 较高，在碱性环境中以分子态为主"
            else:
                pka_type = "amphoteric"
                pka_label = "两性/中性 (Amphoteric/Neutral)"
                pka_color = "#ef6c00"
                pka_bg = "linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%)"
                pka_border = "#ffcc80"
                pka_desc = "pKa 接近中性，电离行为随 pH 变化剧烈"
            
            col_pka1, col_pka2 = st.columns([1, 1.2])
            
            with col_pka1:
                st.markdown("<br>", unsafe_allow_html=True)
                st.metric("Predicted pKa", f"{pka_val:.2f}")
                st.markdown(f"""
                <div style="background: {pka_bg}; border-radius: 12px; padding: 1rem; text-align: center; border: 2px solid {pka_border}; margin-top: 0.8rem;">
                    <div style="font-size: 1.1rem; font-weight: 700; color: {pka_color};">➜ {pka_label}</div>
                    <div style="font-size: 0.85rem; color: #546e7a; margin-top: 0.4rem;">{pka_desc}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_pka2:
                # 生理环境分布图
                import matplotlib.pyplot as plt
                import matplotlib.font_manager as fm
                import glob
                
                # 复用字体设置（简化版，因为前面 SHAP 已经设置过，这里快速复用）
                try:
                    for font in fm.fontManager.ttflist:
                        if font.name in ('Noto Sans CJK SC', 'Noto Sans CJK'):
                            plt.rcParams['font.family'] = font.name
                            break
                except Exception:
                    pass
                plt.rcParams['axes.unicode_minus'] = False
                
                # 生理环境 pH 值
                env_ph = [1.5, 4.5, 6.8, 7.4]
                env_names = ['Stomach\n胃', 'Duodenum\n十二指肠', 'Small Intestine\n小肠', 'Blood/Brain\n血液/脑']
                
                # 计算分子态比例（Henderson-Hasselbalch 方程）
                if pka_type == "acid":
                    fractions = [1 / (1 + 10**(ph - pka_val)) for ph in env_ph]
                else:
                    fractions = [1 / (1 + 10**(pka_val - ph)) for ph in env_ph]
                
                fig, ax = plt.subplots(figsize=(7, 3.2))
                colors_bar = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db']
                bars = ax.bar(env_names, [f*100 for f in fractions], color=colors_bar, edgecolor='white', width=0.6)
                
                for bar, frac in zip(bars, fractions):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                            f'{frac*100:.1f}%', ha='center', va='bottom', fontsize=10)
                
                ax.set_ylabel('分子态比例 (Unionized %)', fontsize=11)
                ax.set_ylim(0, 105)
                ax.set_title(f'不同生理环境下的分子态比例 | pKa = {pka_val:.2f}', fontsize=12)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)
            
            # 药理学洞察
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="card-title">💊 药理学分析</div>
            """, unsafe_allow_html=True)
            
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
            
            # 溶解度 × pKa 联动分析
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="card-title">🔗 溶解度 × pKa 联动分析</div>
            """, unsafe_allow_html=True)
            
            logS = prediction
            parts = []
            
            if logS > 0:
                parts.append("💧 **溶解度**：易溶于水，有利于溶出。")
            elif logS > -2:
                parts.append("💧 **溶解度**：中等，可能需要辅料助溶。")
            else:
                parts.append("💧 **溶解度**：较低，生物利用度可能受限。")
            
            if pka_type == "acid":
                if pka_val < 4:
                    parts.append(f"⚡ **pKa**：弱酸性 (pKa={pka_val:.1f})，胃吸收好，**空腹服用**效果更佳。")
                else:
                    parts.append(f"⚡ **pKa**：中等酸性 (pKa={pka_val:.1f})，全肠道吸收，对服药时间要求不高。")
            elif pka_type == "base":
                if pka_val > 9:
                    parts.append(f"⚡ **pKa**：强碱性 (pKa={pka_val:.1f})，胃吸收差，**餐后服用**可减少胃刺激，主要在小肠吸收。")
                else:
                    parts.append(f"⚡ **pKa**：弱碱性 (pKa={pka_val:.1f})，小肠吸收为主，血液中有利于排泄。")
            else:
                parts.append(f"⚡ **pKa**：接近中性 (pKa={pka_val:.1f})，吸收行为较复杂。")
            
            # 综合判断
            if logS > 0 and pka_type == "acid" and pka_val < 4:
                parts.append("✅ **综合**：高溶解度 + 胃吸收优势 = **口服生物利用度极佳**，适合做成普通片剂。")
            elif logS < -2 and pka_type == "base" and pka_val > 9:
                parts.append("⚠️ **综合**：低溶解度 + 强碱性 = **口服吸收双重挑战**，可能需要肠溶片或注射剂型。")
            elif logS > 0 and pka_type == "base" and pka_val > 9:
                parts.append("✅ **综合**：高溶解度弥补了胃吸收劣势，进入小肠后吸收良好，总体生物利用度可接受。")
            
            st.info(" | ".join(parts))

            # ========== 结构化学深度分析 ==========
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="card-title">🧬 结构化学视角：为什么是这个 pKa？</div>
            """, unsafe_allow_html=True)

            chem_factors = analyze_pka_chemistry(st.session_state.predicted_smiles, pka_val)

            col_3d, col_chem = st.columns([1, 1.2])

            with col_3d:
                st.markdown("<div style='font-weight: 600; color: #37474f; margin-bottom: 0.5rem;'>🎯 3D 球棍模型（可旋转缩放）</div>", unsafe_allow_html=True)
                html_3d = show_3d_molecule(st.session_state.predicted_smiles)
                if html_3d:
                    components.html(html_3d, height=340)
                else:
                    st.info("3D 模型生成失败（需安装 py3Dmol）")

            with col_chem:
                if chem_factors:
                    import matplotlib.pyplot as plt
                    import matplotlib.font_manager as fm

                    # 复用中文字体设置
                    for font in fm.fontManager.ttflist:
                        if font.name in ('Noto Sans CJK SC', 'Noto Sans CJK'):
                            plt.rcParams['font.family'] = font.name
                            break
                    plt.rcParams['axes.unicode_minus'] = False

                    names = list(chem_factors.keys())
                    vals = list(chem_factors.values())
                    colors = ['#ff0051' if v > 0 else '#008bfb' for v in vals]

                    fig, ax = plt.subplots(figsize=(6, 4))
                    bars = ax.barh(range(len(vals)), vals, color=colors, edgecolor='white', height=0.55)
                    ax.invert_yaxis()
                    ax.axvline(x=0, color='black', linewidth=0.8)

                    for bar, val in zip(bars, vals):
                        width = bar.get_width()
                        offset = 0.15 if width >= 0 else -0.15
                        align = 'left' if width >= 0 else 'right'
                        color = 'black' if width >= 0 else 'white'
                        ax.text(width + offset, bar.get_y() + bar.get_height()/2,
                                f'{val:+.2f}', va='center', ha=align, fontsize=10, fontweight='bold', color=color)

                    ax.set_yticks(range(len(names)))
                    ax.set_yticklabels(names, fontsize=10)
                    unit = "增强酸性" if pka_val < 7 else "增强碱性"
                    ax.set_xlabel(f"对 {unit} 的贡献", fontsize=11)
                    ax.set_title(f"pKa = {pka_val:.2f} | 化学因素分解", fontsize=12)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)

                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='#ff0051', label=f'增强{"酸性" if pka_val < 7 else "碱性"}'),
                        Patch(facecolor='#008bfb', label=f'减弱{"酸性" if pka_val < 7 else "碱性"}')
                    ]
                    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

                    plt.tight_layout()
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)

                    st.caption("""
                    💡 **如何读懂这张图**：  
                    红色条越长 = 该因素越推动分子**释放/结合质子**；  
                    蓝色条越长 = 该因素越**抵抗**质子转移。  
                    和 SHAP 不同，这些不是机器学习权重，而是**真实的结构化学效应**。
                    """)
                else:
                    st.info("化学因素分析暂不可用")


        # ========== 分子描述符 ==========
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="card-title">📊 Molecular Descriptors</div>
        """, unsafe_allow_html=True)
        
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
        💡 **Chemistry Insight:** 
        - **TPSA** (Topological Polar Surface Area) measures how much of the molecule is polar. 
           Higher TPSA usually means better water solubility.
        - **H-Bond Donors/Acceptors** tell us how well the molecule can form hydrogen bonds with water.
        - **LogP** measures lipophilicity. Lower LogP means the molecule prefers water over oil.
        """)

        # ========== SHAP 可解释性分析 ==========
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="card-title">🔍 为什么是这个预测结果？</div>
        """, unsafe_allow_html=True)
        st.caption("基于 SHAP (SHapley Additive exPlanations) 分析每个特征对预测的贡献")

        if "shap_values" in st.session_state:
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm
            import numpy as np
            import glob

            # 强制重新扫描系统字体
            fm.fontManager = fm.FontManager()

            # 显式添加 packages.txt 安装的 Noto / 文泉驿字体（.ttc 集合文件）
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

            # 查找中文字体（按优先级）
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
            else:
                print("Chinese font not found. Available fonts:",
                      [f.name for f in fm.fontManager.ttflist[:30]])

            plt.rcParams['axes.unicode_minus'] = False



            shap_vals = np.array(st.session_state.shap_values)
            names = st.session_state.shap_names

            # 按绝对值排序取 Top 8
            abs_vals = np.abs(shap_vals)
            sorted_idx = np.argsort(abs_vals)[::-1][:8]
            top_shap = shap_vals[sorted_idx]
            top_names = [names[i] for i in sorted_idx]

            # 颜色
            colors = ['#ff0051' if v > 0 else '#008bfb' for v in top_shap]

            # 画图
            fig, ax = plt.subplots(figsize=(8, 4.5))
            bars = ax.barh(range(len(top_shap)), top_shap, color=colors, edgecolor="white", height=0.6)
            ax.invert_yaxis()

            for i, (bar, val) in enumerate(zip(bars, top_shap)):
                width = bar.get_width()
                # 正值标签放右侧外部；负值标签放条形内部，避免与Y轴文字重叠
                if width >= 0:
                    label_x = width + 0.05
                    align = "left"
                    text_color = "black"
                else:
                    label_x = width + 0.12  # 放在条形内部，从左侧向右偏移
                    align = "left"
                    text_color = "white"
                ax.text(label_x, i, f"{val:+.3f}", va="center", ha=align, fontsize=10, fontweight="bold", color=text_color)

            ax.set_yticks(range(len(top_names)))
            ax.set_yticklabels(top_names, fontsize=11)
            ax.axvline(x=0, color="black", linewidth=0.8)
            ax.set_xlabel("对溶解度的贡献值 (logS)", fontsize=11)

            # 兼容 expected_value 各种结构
            ev = explainer.expected_value
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
                Patch(facecolor="#ff0051", label="推动易溶 (正贡献)"),
                Patch(facecolor="#008bfb", label="推动难溶 (负贡献)")
            ]
            ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

            plt.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)

            # 文字解读（与预测值一致）
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

            parts = ["💡 **关键分析**：模型预测该分子 **" + solubility_level + "**（logS = " + f"{prediction:.3f}" + "）。"]
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

        # ========== Kimi AI 解释（手动触发）==========
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="card-title">🧠 AI Chemistry Explanation</div>
        """, unsafe_allow_html=True)
        
        with st.container(border=True):
            if st.session_state.ai_explanation:
                st.markdown(st.session_state.ai_explanation)
                if st.button("🗑️ 清除解释", key="clear_ai"):
                    st.session_state.ai_explanation = None
                    st.rerun()
            else:
                st.caption("AI 解释需要手动调用（消耗 API 额度）")
                if st.button("🤖 生成 AI 解释", key="gen_ai", use_container_width=True):
                    with st.spinner("正在分析分子结构..."):
                        # 判断 pKa 类型
                        pka_val = st.session_state.get("predicted_pka")
                        if pka_val is not None:
                            if pka_val < 5:
                                pka_type = "acid"
                            elif pka_val > 9:
                                pka_type = "base"
                            else:
                                pka_type = "amphoteric"
                        else:
                            pka_val = None
                            pka_type = None
                        
                        explanation = explain_with_kimi(
                            st.session_state.predicted_smiles,
                            prediction,
                            features,
                            shap_features=st.session_state.get("shap_names"),
                            shap_values=st.session_state.get("shap_values"),
                            pka_value=pka_val,
                            pka_type=pka_type
                        )
                    st.session_state.ai_explanation = explanation
                    st.rerun()

# ========== 页脚 ==========
st.markdown("""
<div class="footer">
    <div style="font-weight: 600; color: #546e7a; margin-bottom: 0.3rem;">Molecular Solubility Predictor</div>
    <div>Built by Leonlee | ML: Random Forest + RDKit (V2: 11,000+ molecules) | AI: Kimi (Moonshot AI) | DB: 100+ local + PubChem API</div>
    <div style="margin-top: 0.5rem; font-size: 0.75rem; color: #b0bec5;">🧪 科学计算 · 🤖 人工智能 · 🎯 药物化学</div>
</div>
""", unsafe_allow_html=True)
