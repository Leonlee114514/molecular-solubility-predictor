import joblib
import os

model_path = "output_v2/pka_model.pkl"

print("正在加载模型...")
model = joblib.load(model_path)
original_size = os.path.getsize(model_path) / (1024*1024)
print(f"原始大小: {original_size:.1f} MB")

# 方案 1: zlib 最高压缩 (最快)
print("\n尝试 zlib compress=9...")
joblib.dump(model, "output_v2/pka_model_zlib9.pkl", compress=9)
size1 = os.path.getsize("output_v2/pka_model_zlib9.pkl") / (1024*1024)
print(f"zlib-9 后: {size1:.1f} MB")

# 方案 2: lzma 最高压缩 (压缩率最高，但慢)
print("\n尝试 lzma compress=9 (可能需要 3-5 分钟)...")
joblib.dump(model, "output_v2/pka_model_lzma9.pkl", compress=('lzma', 9))
size2 = os.path.getsize("output_v2/pka_model_lzma9.pkl") / (1024*1024)
print(f"lzma-9 后: {size2:.1f} MB")

# 选择最小的可用文件
if size2 < 100:
    best = "output_v2/pka_model_lzma9.pkl"
    best_size = size2
    print(f"\n✅ lzma 压缩成功！{best_size:.1f} MB")
elif size1 < 100:
    best = "output_v2/pka_model_zlib9.pkl"
    best_size = size1
    print(f"\n✅ zlib 压缩成功！{best_size:.1f} MB")
else:
    print(f"\n❌ 两种压缩都 > 100MB，必须减少树的数量重新训练")
    print("建议：把 n_estimators=400 改成 100，重新运行 train_pka_model.py")
    exit()

# 替换为最终文件
import shutil
shutil.copy(best, model_path)
print(f"已替换为: {model_path} ({best_size:.1f} MB)")

# 清理临时文件
os.remove("output_v2/pka_model_zlib9.pkl")
os.remove("output_v2/pka_model_lzma9.pkl")
print("临时文件已清理")
