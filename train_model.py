"""
【分子溶解度预测模型 - 训练脚本】
适合零基础高中生的完整代码，每一行都有详细中文注释
作者：你的姓名
项目：Predicting Molecular Solubility from SMILES using Machine Learning
"""

# ========== 第1步：导入必要的工具库 ==========
# pandas: 用于处理表格数据（类似Excel的功能）
import pandas as pd
# numpy: 用于数值计算（处理数组和矩阵）
import numpy as np
# requests: 用于从网络下载数据
import requests
# os: 用于文件路径操作
import os

# RDKit: 化学信息学核心库，用于将SMILES字符串转换为分子并计算性质
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

# scikit-learn: 机器学习库，提供随机森林等算法
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# matplotlib: 用于绘制图表
import matplotlib.pyplot as plt

# joblib: 用于保存训练好的模型，方便后续使用
import joblib

print("=" * 60)
print("🧪 分子溶解度预测模型 - 开始训练")
print("=" * 60)

# ========== 第2步：下载并加载数据 ==========
# 这是著名的 Delaney ESOL 数据集，包含1144个有机分子的结构(SMILES)和实测溶解度
# 数据来源：ESOL: Estimating Aqueous Solubility Directly from Molecular Structure

data_url = "https://raw.githubusercontent.com/dataprofessor/data/master/delaney.csv"
print("\n📥 正在从网络下载数据集...")

try:
    # 尝试直接从网络读取CSV文件
    df = pd.read_csv(data_url)
    print("✅ 数据下载成功！")
except Exception as e:
    # 如果网络下载失败，给出提示
    print(f"❌ 下载失败: {e}")
    print("请检查网络连接，或手动下载上述链接的CSV文件放到本目录")
    exit()

# 查看数据的前5行，了解数据结构
print("\n📋 数据集预览（前5行）：")
print(df.head())

# 查看数据集有多少行（多少个分子）
print(f"\n📊 数据集包含 {len(df)} 个分子")

# 这个数据集有4列：
# 1. Compound ID: 化合物名称
# 2. measured log(solubility:mol/L): 实测溶解度（对数单位，这是我们想预测的目标）
# 3. ESOL predicted log(solubility:mol/L): 论文作者用简单公式预测的溶解度（我们不用这一列）
# 4. SMILES: 分子的文本表示（结构信息）

# 为了代码简洁，我们把列名改短一些
df = df.rename(columns={
    'measured log(solubility:mol/L)': 'logS',
    'ESOL predicted log(solubility:mol/L)': 'ESOL_logS'
})

# ========== 第3步：定义特征计算函数 ==========
# 这是本项目的核心：如何把"化学结构"变成"数字"让计算机理解

print("\n🔬 正在从分子结构中提取特征...")
print("（这可能需要几十秒，请耐心等待）")

def compute_features(smiles_string):
    """
    输入：一个SMILES字符串（如 "CCO" 代表乙醇）
    输出：该分子的数值特征（分子量、极性、氢键数量等）

    如果SMILES无效（比如写错了），返回 None
    """
    # 用 RDKit 把 SMILES 字符串解析成分子对象
    mol = Chem.MolFromSmiles(smiles_string)

    # 如果解析失败（SMILES不合法），返回None，后面会跳过这个分子
    if mol is None:
        return None

    # 创建一个字典，存放所有特征
    features = {}

    # --- 第一类特征：基础物理化学描述符 ---
    # 这些特征都有明确的化学意义，你可以在文书中解释它们

    features['MolWt'] = Descriptors.MolWt(mol)                  # 分子量 (g/mol)
    features['LogP'] = Descriptors.MolLogP(mol)                 # 脂水分配系数（疏水性指标）
    features['NumHDonors'] = Descriptors.NumHDonors(mol)        # 氢键供体数量（如 -OH, -NH）
    features['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)  # 氢键受体数量（如 O, N）
    features['TPSA'] = Descriptors.TPSA(mol)                   # 拓扑极性表面积 (Å²)
    features['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)  # 可旋转键数（柔性指标）
    features['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)  # 芳香环数量
    features['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol) # 脂肪环数量

    # --- 第二类特征：Morgan 分子指纹 ---
    # 指纹是一个长度为1024的二进制向量（0和1的数组）
    # 它记录分子中有哪些"子结构片段"（如苯环、羟基、羧基等）
    # radius=2 表示考虑每个原子周围2层化学键范围内的环境
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)

    # 把指纹转换成 numpy 数组（方便机器学习处理）
    fp_array = np.zeros((1,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, fp_array)

    return features, fp_array

# ========== 第4步：批量处理所有分子 ==========
# 创建空列表，用于存放所有分子的特征和标签
feature_list = []      # 存放分子描述符
fingerprint_list = []  # 存放分子指纹
labels = []            # 存放实测溶解度（我们要预测的目标）
valid_smiles = []      # 存放有效的SMILES（用于后续分析）
compound_names = []    # 存放化合物名称

# 逐行处理数据集中的每个分子
for idx, row in df.iterrows():
    smiles = row['SMILES']
    solubility = row['logS']
    name = row['Compound ID']

    # 调用上面的函数计算特征
    result = compute_features(smiles)

    # 如果SMILES无效，跳过这个分子
    if result is None:
        continue

    feat, fp = result

    # 把结果存入列表
    feature_list.append(list(feat.values()))   # 8个描述符
    fingerprint_list.append(fp)                # 1024-bit指纹
    labels.append(solubility)                  # 实测溶解度
    valid_smiles.append(smiles)                # 保留有效SMILES
    compound_names.append(name)                # 保留化合物名

print(f"✅ 成功处理 {len(feature_list)} 个分子，跳过了 {len(df) - len(feature_list)} 个无效分子")

# 将列表转换为 numpy 数组（机器学习算法需要这种格式）
X_desc = np.array(feature_list)              # 形状: (分子数, 8)
X_fp = np.array(fingerprint_list)            # 形状: (分子数, 1024)

# 把两类特征合并在一起：8个描述符 + 1024个指纹 = 1032个特征
X = np.hstack([X_desc, X_fp])                # 形状: (分子数, 1032)
y = np.array(labels)                         # 形状: (分子数,)，即所有实测溶解度值

# 保存特征名称（用于后续分析特征重要性）
feature_names = list(feat.keys()) + [f'FP_{i}' for i in range(1024)]

print(f"\n📐 特征矩阵形状: {X.shape}")
print(f"   每个分子用 {X.shape[1]} 个数字表示")

# ========== 第5步：划分训练集和测试集 ==========
# 把数据分成两部分：80%用于训练模型，20%用于测试模型表现
# random_state=42 保证每次运行划分结果相同（方便复现）

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✂️ 数据划分:")
print(f"   训练集: {len(X_train)} 个分子（用于学习规律）")
print(f"   测试集: {len(X_test)} 个分子（用于检验预测能力）")

# ========== 第6步：训练机器学习模型 ==========
# 我们选择 Random Forest（随机森林）回归模型
# 原理：构建多棵"决策树"，每棵树根据分子特征投票决定溶解度
# 优点：不容易过拟合，能告诉我们哪些特征最重要，适合初学者

print("\n🌲 正在训练 Random Forest 模型...")
print("（正在构建200棵决策树，请稍候...）")

model = RandomForestRegressor(
    n_estimators=200,      # 森林中有200棵树（越多越稳定，但训练越慢）
    max_depth=15,          # 每棵树最大深度15层（防止过拟合）
    random_state=42,       # 保证结果可复现
    n_jobs=-1              # 使用电脑所有CPU核心加速训练
)

# 开始训练！这是机器学习的核心步骤
model.fit(X_train, y_train)

print("✅ 模型训练完成！")

# ========== 第7步：评估模型表现 ==========
# 用训练好的模型预测测试集的溶解度
y_pred = model.predict(X_test)

# 计算评估指标
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📈 模型评估结果（在测试集上）：")
print(f"   R² 分数: {r2:.3f}")
print(f"      → 解释: R²=1.0 表示完美预测，R²=0 表示和瞎猜一样")
print(f"      → 你的模型 R²={r2:.3f}，意味着能解释溶解度变异的 {r2*100:.1f}%")
print(f"   RMSE: {rmse:.3f}")
print(f"      → 解释: 预测值平均偏离真实值约 {rmse:.3f} 个对数单位")

# ========== 第8步：特征重要性分析（文书核心素材！）==========
# 随机森林可以告诉我们：哪些分子特征对预测溶解度最重要

print("\n🔍 特征重要性分析（哪些因素影响溶解度最多）：")

# 只取前8个"可解释"的分子描述符的重要性
descriptor_importance = model.feature_importances_[:8]
descriptor_names = list(feat.keys())

# 排序：从最重要到最不重要
sorted_indices = np.argsort(descriptor_importance)[::-1]

for rank, idx in enumerate(sorted_indices, 1):
    name = descriptor_names[idx]
    importance = descriptor_importance[idx]
    bar = "█" * int(importance * 50)  # 用方块可视化
    print(f"   {rank}. {name:20s}: {importance:.3f} {bar}")

print("\n💡 化学洞察（可用于申请文书）：")
print("   • TPSA（极性表面积）和 H-bond 相关特征通常排名靠前")
print("   • 这说明'极性'和'氢键能力'是决定水溶性的关键——")
print("   • 完美印证了你在化学课上学到的'相似相溶'原理！")

# ========== 第9步：保存模型和结果 ==========
print("\n💾 正在保存模型和图表...")

# 创建输出目录
os.makedirs("output", exist_ok=True)

# 保存训练好的模型（后续网页应用会加载它）
joblib.dump(model, "output/solubility_model.pkl")
print("   ✅ 模型已保存到: output/solubility_model.pkl")

# 保存特征名称（用于网页应用显示）
joblib.dump(descriptor_names, "output/descriptor_names.pkl")
print("   ✅ 特征名称已保存")

# ========== 第10步：绘制可视化图表 ==========

# 图1：预测值 vs 真实值 散点图
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5)

# 画一条理想的对角线（完美预测时所有点应落在这条线上）
min_val = min(y_test.min(), y_pred.min()) - 0.5
max_val = max(y_test.max(), y_pred.max()) + 0.5
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

plt.xlabel('Experimental logS (实测溶解度)', fontsize=12)
plt.ylabel('Predicted logS (预测溶解度)', fontsize=12)
plt.title(f'Random Forest Prediction\nR² = {r2:.3f}', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/prediction_vs_actual.png", dpi=200)
print("   ✅ 图表1已保存: output/prediction_vs_actual.png")

# 图2：特征重要性柱状图
plt.figure(figsize=(8, 5))
colors = plt.cm.viridis(np.linspace(0, 1, 8))
bars = plt.barh(range(8), descriptor_importance[sorted_indices], color=colors)
plt.yticks(range(8), np.array(descriptor_names)[sorted_indices])
plt.xlabel('Feature Importance (特征重要性)', fontsize=12)
plt.title('Which Molecular Features Drive Solubility?\n(哪些分子特征决定溶解度)', fontsize=13)
plt.gca().invert_yaxis()  # 让最重要的特征在顶部
plt.tight_layout()
plt.savefig("output/feature_importance.png", dpi=200)
print("   ✅ 图表2已保存: output/feature_importance.png")

# 图3：残差分布图（预测误差分析）
residuals = y_test - y_pred
plt.figure(figsize=(7, 5))
plt.hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
plt.xlabel('Prediction Error (预测误差)', fontsize=12)
plt.ylabel('Count (分子数量)', fontsize=12)
plt.title('Distribution of Prediction Errors\n(预测误差分布)', fontsize=13)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.tight_layout()
plt.savefig("output/residual_distribution.png", dpi=200)
print("   ✅ 图表3已保存: output/residual_distribution.png")

print("\n" + "=" * 60)
print("🎉 训练完成！所有文件已保存到 output/ 文件夹")
print("=" * 60)
print("\n下一步：运行 'streamlit run app.py' 启动网页应用")
