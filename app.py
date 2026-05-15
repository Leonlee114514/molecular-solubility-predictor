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

# 加载环境变量
load_dotenv()
KIMI_API_KEY = os.getenv("KIMI_API_KEY")

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
    layout="centered"
)

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
    st.error(f"❌ 模型加载失败: {e}")
    st.info("请先运行 'python train_model_v2.py' 训练模型")
    model_ready = False

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
    rdBase.EnableLog("rdApp.warning")
    fp_array = np.zeros((1,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, fp_array)
    return features, fp_array

# ========== Kimi AI 解释 ==========
def explain_with_kimi(smiles, prediction, features):
    if not KIMI_API_KEY:
        return "⚠️ 未配置 Kimi API Key。请在 .env 文件中写入：KIMI_API_KEY=sk-你的密钥"
    
    prompt = f"""你是一位擅长AP Chemistry教学的老师，正在向高中生解释分子溶解度。

分子 SMILES: {smiles}
模型预测的水溶解度 (logS): {prediction:.2f}
关键分子性质:
- 分子量: {features['MolWt']:.1f} g/mol
- 极性表面积 (TPSA): {features['TPSA']:.1f} Å²
- 氢键供体数: {features['NumHDonors']}
- 氢键受体数: {features['NumHAcceptors']}
- 脂水分配系数 (LogP): {features['LogP']:.2f}

请用中文回答，分三段:
1. 这个分子在水中是易溶还是难溶？给出结论。
2. 从分子结构角度解释原因（极性、氢键、疏水性）。引用上面的具体数据。
3. 举一个生活中的类比帮助理解。

要求: 准确、简洁、有亲和力，每段不超过3句话。"""

    try:
        client = openai.OpenAI(
            api_key=KIMI_API_KEY,
            base_url="https://api.moonshot.cn/v1"
        )
        response = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[
                {"role": "system", "content": "你是一位专业的化学教育助手，擅长结合定量数据解释化学现象，用高中生能听懂的语言。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=600
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
st.title("🧪 Molecular Solubility Predictor")
st.subheader("Predict Aqueous Solubility from Molecular Structure")

st.markdown("""
**Welcome!** This app predicts how well a molecule dissolves in water (logS) 
using a Machine Learning model trained on **11,000+ organic compounds**.

**三种输入方式：**
1. 👇 从下拉菜单快速选择（100+ 常见分子）
2. 🔍 输入名称搜索（先查本地库，再查 PubChem API，无需手动切换）
3. ✏️ 直接粘贴 SMILES 字符串
""")

# ========== 输入区域 ==========

# --- 方式1：下拉菜单 ---
st.markdown("**方式 1：快速选择常见分子**")
selected_molecule = st.selectbox(
    "选择分子",
    list(MOLECULE_DB.keys()),
    index=0,
    key="molecule_select"
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
st.markdown("**方式 2：名称搜索（本地库 + PubChem API）**")
st.caption("💡 支持中英文，如 阿司匹林 / Aspirin / Ibuprofen / 咖啡因")
search_col1, search_col2 = st.columns([3, 1])
with search_col1:
    search_name = st.text_input(
        "输入名称",
        placeholder="例如 阿司匹林 或 Aspirin",
        key="search_name"
    )
with search_col2:
    search_clicked = st.button("🔍 搜索", key="search_btn")

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
                <div style="background-color: #f0f8ff; padding: 15px; border-radius: 8px; border-left: 4px solid #4a90e2;">
                <h4>🔍 如何手动获取 SMILES？</h4>
                <ol>
                    <li>访问 <a href="https://pubchem.ncbi.nlm.nih.gov" target="_blank"><b>https://pubchem.ncbi.nlm.nih.gov</b></a></li>
                    <li>在搜索框输入分子名称（英文，如 <b>Aspirin</b>）</li>
                    <li>进入化合物页面，找到 <b>Canonical SMILES</b> 字段</li>
                    <li>复制 SMILES 字符串（如 <code>CC(=O)Oc1ccccc1C(=O)O</code>）</li>
                    <li>粘贴到下方的 "方式 3" 文本框中，点击 Predict</li>
                </ol>
                </div>
                """, unsafe_allow_html=True)

# --- 方式3：SMILES 直接输入 ---
st.markdown("**方式 3：直接输入 SMILES**")
st.caption("💡 可从下拉菜单自动填入，也可手动编辑或粘贴外部 SMILES")

smiles_input = st.text_input(
    "当前 SMILES",
    key="smiles_input_box"
)

if smiles_input != st.session_state.get("smiles_input_box", ""):
    st.session_state.predicted_smiles = None
    st.session_state.predicted_logS = None
    st.session_state.ai_explanation = None

# ========== 预测按钮 ==========
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
with btn_col2:
    predict_button = st.button("🔮 Predict Solubility", width="stretch")

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
            st.session_state.ai_explanation = None

# ========== 显示预测结果 ==========
if st.session_state.predicted_smiles and st.session_state.predicted_logS is not None:
    
    result_display = compute_features(st.session_state.predicted_smiles)
    
    if result_display is None:
        st.error("显示时解析失败，请重新输入 SMILES")
    else:
        features, _ = result_display
        prediction = st.session_state.predicted_logS
        
        st.divider()
        
        try:
            mol = Chem.MolFromSmiles(st.session_state.predicted_smiles)
            img = Draw.MolToImage(mol, size=(350, 350), kekulize=True)
        except Exception as e:
            img = None
            st.warning(f"⚠️ 结构图生成失败: {e}")
        
        col_left, col_right = st.columns([1, 1.2])
        
        with col_left:
            if img is not None:
                st.image(img, caption="Molecular Structure", width="stretch")
            else:
                st.info("无法显示结构图")
        
        with col_right:
            st.metric(
                label="Predicted Solubility (logS)",
                value=f"{prediction:.3f}",
                delta=None
            )
            
            if prediction > 0:
                interp = "Highly soluble (易溶于水)"
                color = "green"
            elif prediction > -2:
                interp = "Moderately soluble (中等溶解)"
                color = "orange"
            else:
                interp = "Poorly soluble (难溶于水)"
                color = "red"
            
            st.markdown(f"<h4 style='color: {color};'>➜ {interp}</h4>", unsafe_allow_html=True)
            
            st.markdown("""
            **Interpretation guide:**
            - logS > 0: Very soluble (like ethanol)
            - -2 < logS < 0: Moderately soluble
            - logS < -2: Poorly soluble (like many drug molecules)
            """)
        
        # ========== 分子描述符 ==========
        st.divider()
        st.subheader("📊 Molecular Descriptors")
        
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
        
        # ========== Kimi AI 解释（手动触发）==========
        st.divider()
        st.subheader("🧠 AI Chemistry Explanation (Powered by Kimi)")
        
        if st.session_state.ai_explanation:
            st.markdown(st.session_state.ai_explanation)
            if st.button("🗑️ 清除解释", key="clear_ai"):
                st.session_state.ai_explanation = None
                st.rerun()
        else:
            st.caption("💡 AI 解释需要手动调用（消耗 API 额度）")
            if st.button("🧠 生成 AI 解释（手动调用 Kimi）", key="gen_ai"):
                with st.spinner("🤖 Kimi 正在分析分子结构..."):
                    explanation = explain_with_kimi(
                        st.session_state.predicted_smiles,
                        prediction,
                        features
                    )
                st.session_state.ai_explanation = explanation
                st.rerun()

# ========== 页脚 ==========
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
Built with ❤️ by a high school chemistry enthusiast | 
ML: Random Forest + RDKit (V2: 11,000+ molecules) | AI: Kimi (Moonshot AI) | DB: 100+ local + PubChem API
</div>
""", unsafe_allow_html=True)
