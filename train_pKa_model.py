import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os

# ==================== 1. 加载并合并 acidic + basic 数据 ====================
# 修改这里为你的 GR-pKa 数据路径
ACIDIC_PATH = "GR-pKa/data/pre-training/pretrain_pka_acidic.csv"
BASIC_PATH = "GR-pKa/data/pre-training/pretrain_pka_basic.csv"

# 读取 acidic 数据
df_acidic = pd.read_csv(ACIDIC_PATH)[['smiles', 'pka_acidic']].dropna()
df_acidic.columns = ['smiles', 'pka']

# 读取 basic 数据（列名应该是 pka_basic）
df_basic = pd.read_csv(BASIC_PATH)[['smiles', 'pka_basic']].dropna()
df_basic.columns = ['smiles', 'pka']

# 合并
df = pd.concat([df_acidic, df_basic], ignore_index=True)
print(f"✅ 合并完成：acidic {len(df_acidic)} 条 + basic {len(df_basic)} 条 = 共 {len(df)} 条")

# ==================== 2. 特征计算（复用 app.py 逻辑）====================
def compute_features(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None
    features = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumAliphaticRings(mol),
    ]
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fp_array = np.zeros((1,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, fp_array)
    return np.hstack([features, fp_array])

X, y = [], []
for _, row in df.iterrows():
    feat = compute_features(row['smiles'])
    if feat is not None:
        X.append(feat)
        y.append(float(row['pka']))

X = np.array(X)
y = np.array(y)
print(f"✅ 成功解析 {len(y)} 个分子，特征维度: {X.shape[1]}")

# ==================== 3. 训练模型 ====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=400,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"\n✅ pKa Model Trained")
print(f"   RMSE: {rmse:.3f} pH units")
print(f"   MAE:  {mae:.3f} pH units")
print(f"   Data range: {y.min():.1f} ~ {y.max():.1f}")

# ==================== 4. 保存模型 ====================
os.makedirs("output_v2", exist_ok=True)
joblib.dump(model, "output_v2/pka_model.pkl")
joblib.dump([f"f_{i}" for i in range(X.shape[1])], "output_v2/pka_descriptor_names.pkl")
print("💾 已保存到 output_v2/pka_model.pkl")
