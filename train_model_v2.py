"""
【分子溶解度预测模型 V2 - 合并多数据集训练】
合并 ESOL (1,144) + AqSolDB (9,982) ≈ 11,000+ 分子
适合基础版本跑通后的进阶扩展
"""

import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

print("=" * 60)
print("🧪 分子溶解度预测模型 V2 - 多数据集训练")
print("=" * 60)

# ========== 第1步：加载多个数据集 ==========

datasets = []

# --- 数据集1：ESOL (Delaney) ---
print("\n📥 加载 ESOL 数据集...")
try:
    df_esol = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/delaney.csv")
    df_esol = df_esol.rename(columns={
        'measured log(solubility:mol/L)': 'logS',
        'Compound ID': 'ID'
    })[['SMILES', 'logS']]
    df_esol['source'] = 'ESOL'
    datasets.append(df_esol)
    print(f"   ✅ ESOL: {len(df_esol)} 个分子")
except Exception as e:
    print(f"   ❌ ESOL 加载失败: {e}")

# --- 数据集2：AqSolDB (需要手动下载放到项目文件夹) ---
print("\n📥 加载 AqSolDB 数据集...")
try:
    # 从本地 CSV 加载（你需要先下载 curated-solubility-dataset.csv）
    df_aqsol = pd.read_csv("curated-solubility-dataset.csv")

    # AqSolDB 的列名可能叫 'SMILES' 和 'Solubility'
    # 我们检查并重命名
    if 'SMILES' in df_aqsol.columns and 'Solubility' in df_aqsol.columns:
        df_aqsol = df_aqsol[['SMILES', 'Solubility']].rename(columns={'Solubility': 'logS'})
    elif 'smiles' in df_aqsol.columns and 'solubility' in df_aqsol.columns:
        df_aqsol = df_aqsol[['smiles', 'solubility']].rename(columns={'smiles': 'SMILES', 'solubility': 'logS'})
    else:
        print(f"   ⚠️ 检测到列名: {list(df_aqsol.columns)}")
        # 尝试自动找到 SMILES 和溶解度列
        smiles_col = [c for c in df_aqsol.columns if 'smiles' in c.lower() or 'SMILES' in c][0]
        sol_col = [c for c in df_aqsol.columns if 'solubility' in c.lower() or 'Solubility' in c][0]
        df_aqsol = df_aqsol[[smiles_col, sol_col]].rename(columns={smiles_col: 'SMILES', sol_col: 'logS'})

    df_aqsol['source'] = 'AqSolDB'
    datasets.append(df_aqsol)
    print(f"   ✅ AqSolDB: {len(df_aqsol)} 个分子")
except FileNotFoundError:
    print("   ❌ 未找到 curated-solubility-dataset.csv")
    print("   💡 请从 https://www.kaggle.com/sorkun/aqsoldb-a-curated-aqueous-solubility-dataset 下载")
    print("   💡 或从 https://github.com/mcsorkun/AqSolDB 下载并放到本文件夹")
except Exception as e:
    print(f"   ❌ AqSolDB 加载失败: {e}")

# 检查是否有数据
if len(datasets) == 0:
    print("\n❌ 没有成功加载任何数据集，退出")
    exit()

# 合并数据集
df = pd.concat(datasets, ignore_index=True)
print(f"\n📊 合并后总数据: {len(df)} 个分子")

# 去重：相同 SMILES 保留第一个（避免重复训练）
df = df.drop_duplicates(subset=['SMILES'], keep='first')
print(f"📊 去重后: {len(df)} 个唯一分子")

# 查看数据来源分布
print("\n📋 数据来源分布:")
print(df['source'].value_counts())

# ========== 第2步：特征计算（和 V1 完全一致）==========
print("\n🔬 正在提取分子特征...")

def compute_features(smiles_string):
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

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fp_array = np.zeros((1,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, fp_array)

    return features, fp_array

feature_list = []
fingerprint_list = []
labels = []
valid_sources = []

for idx, row in df.iterrows():
    smiles = row['SMILES']
    solubility = row['logS']
    source = row['source']

    result = compute_features(smiles)
    if result is None:
        continue

    feat, fp = result
    feature_list.append(list(feat.values()))
    fingerprint_list.append(fp)
    labels.append(solubility)
    valid_sources.append(source)

print(f"✅ 成功处理 {len(feature_list)} 个分子")

X_desc = np.array(feature_list)
X_fp = np.array(fingerprint_list)
X = np.hstack([X_desc, X_fp])
y = np.array(labels)

# ========== 第3步：按来源分层划分训练/测试集 ==========
# 这是进阶技巧：确保两个数据集的分子都出现在训练集和测试集中
# 避免测试集全是某一个来源的分子

from sklearn.model_selection import StratifiedShuffleSplit

# 创建分层标签（二分类：ESOL vs AqSolDB）
source_labels = np.array([1 if s == 'AqSolDB' else 0 for s in valid_sources])

# 分层抽样：保持训练集和测试集中的来源比例一致
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(X, source_labels))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
source_train = np.array(valid_sources)[train_idx]
source_test = np.array(valid_sources)[test_idx]

print(f"\n✂️ 分层划分:")
print(f"   训练集: {len(X_train)} 个分子")
print(f"      - ESOL: {sum(source_train == 'ESOL')}")
print(f"      - AqSolDB: {sum(source_train == 'AqSolDB')}")
print(f"   测试集: {len(X_test)} 个分子")
print(f"      - ESOL: {sum(source_test == 'ESOL')}")
print(f"      - AqSolDB: {sum(source_test == 'AqSolDB')}")

# ========== 第4步：训练模型 ==========
print("\n🌲 训练 Random Forest...")
model = RandomForestRegressor(
    n_estimators=300,      # 增加树的数量，大数据集更稳定
    max_depth=20,          # 适当增加深度
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ========== 第5步：评估（总体 + 分来源）==========
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📈 总体评估:")
print(f"   R²: {r2:.3f}")
print(f"   RMSE: {rmse:.3f}")

# 分来源评估（关键！看模型对不同数据源的泛化能力）
print("\n📈 分来源评估:")
for src in ['ESOL', 'AqSolDB']:
    mask = source_test == src
    if mask.sum() > 0:
        src_r2 = r2_score(y_test[mask], y_pred[mask])
        src_rmse = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))
        print(f"   {src}: R²={src_r2:.3f}, RMSE={src_rmse:.3f}, n={mask.sum()}")

# ========== 第6步：特征重要性 ==========
print("\n🔍 特征重要性 (Top 8 描述符):")
descriptor_importance = model.feature_importances_[:8]
descriptor_names = list(feat.keys())
sorted_indices = np.argsort(descriptor_importance)[::-1]
for rank, idx in enumerate(sorted_indices, 1):
    print(f"   {rank}. {descriptor_names[idx]:20s}: {descriptor_importance[idx]:.3f}")

# ========== 第7步：保存 ==========
os.makedirs("output_v2", exist_ok=True)
joblib.dump(model, "output_v2/solubility_model_v2.pkl")
joblib.dump(descriptor_names, "output_v2/descriptor_names_v2.pkl")
print("\n💾 模型已保存到 output_v2/")

# ========== 第8步：可视化对比 ==========
plt.figure(figsize=(7, 7))

# 用不同颜色标记来源
colors = {'ESOL': 'blue', 'AqSolDB': 'orange'}
for src in ['ESOL', 'AqSolDB']:
    mask = source_test == src
    plt.scatter(y_test[mask], y_pred[mask], 
                alpha=0.5, label=src, color=colors[src], edgecolors='black', linewidth=0.3)

min_val = min(y_test.min(), y_pred.min()) - 0.5
max_val = max(y_test.max(), y_pred.max()) + 0.5
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
plt.xlabel('Experimental logS')
plt.ylabel('Predicted logS')
plt.title(f'Multi-Dataset Model (ESOL + AqSolDB)\nR² = {r2:.3f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output_v2/prediction_vs_actual_v2.png", dpi=200)
print("   ✅ 对比图已保存")

# 新增：按来源的误差分布图
plt.figure(figsize=(8, 5))
for src in ['ESOL', 'AqSolDB']:
    mask = source_test == src
    residuals = y_test[mask] - y_pred[mask]
    plt.hist(residuals, bins=20, alpha=0.5, label=f'{src} (mean={residuals.mean():.2f})')
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.title('Error Distribution by Dataset Source')
plt.legend()
plt.tight_layout()
plt.savefig("output_v2/residual_by_source.png", dpi=200)
print("   ✅ 误差分布图已保存")

print("\n" + "=" * 60)
print("🎉 V2 训练完成！")
print("=" * 60)
print("\n💡 对比 V1:")
print("   - 数据量: 1,144 → ~11,000 分子")
print("   - 来源多样性: 单一 → 多数据集")
print("   - 评估维度: 总体 R² → 分来源 R²（检验跨数据集泛化）")
print("\n⚠️ 注意：如果要让 app.py 使用 V2 模型，需要修改 app.py 中的模型路径")
print("   把 'output/solubility_model.pkl' 改成 'output_v2/solubility_model_v2.pkl'")
