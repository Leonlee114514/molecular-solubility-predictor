import joblib

# 加载原模型
model = joblib.load("output_v2/solubility_model_v2.pkl")
desc_names = joblib.load("output_v2/descriptor_names_v2.pkl")

# 用最高压缩率重新保存
joblib.dump(model, "output_v2/solubility_model_v2.pkl", compress=9)
joblib.dump(desc_names, "output_v2/descriptor_names_v2.pkl", compress=9)

print("压缩完成")
