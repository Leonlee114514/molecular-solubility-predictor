import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os

print("=" * 50)
print("当前工作目录:", os.getcwd())
print("=" * 50)

# 数据路径（已改成项目内的 data 文件夹）
ACIDIC_PATH = "data/pretrain_pka_acidic.csv"
BASIC_PATH = "data/pretrain_pka_basic.csv"

print(f"读取 acidic 数据: {ACIDIC_PATH}")
df_acidic = pd.read_csv(ACIDIC_PATH)[['smiles', 'pka_acidic']].dropna()
df_acidic.columns = ['smiles', 'pka']
print(f"   -> {len(df_acidic)} 条")

print(f"读取 basic 数据: {BASIC_PATH}")
df_basic = pd.read_csv(BASIC_PATH)[['smiles', 'pka_basic']].dropna()
df_basic.columns = ['smiles', 'pka']
print(f"   -> {len(df_basic)} 条")

df = pd.concat([df_acidic, df_basic], ignore_index=True)
print(f"合并完成: 共 {len(df)} 条")

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
print(f"成功解析 {len(y)} 个分子，特征维度: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
print("开始训练...")
model.fit(X_train, y_train)
print("训练完成！")

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"\n✅ pKa Model Trained")
print(f"   RMSE: {rmse:.3f} pH units")
print(f"   MAE:  {mae:.3f} pH units")

os.makedirs("output_v2", exist_ok=True)
save_path = "output_v2/pka_model.pkl"
joblib.dump(model, save_path, compress=3)
print(f"💾 模型已保存: {os.path.abspath(save_path)}")

size_mb = os.path.getsize(save_path) / (1024 * 1024)
print(f"📦 文件大小: {size_mb:.1f} MB")
if size_mb < 100:
    print("✅ 大小正常，可以提交到 GitHub")
else:
    print("⚠️ 文件 > 100MB，需要减少 n_estimators 重新训练")
