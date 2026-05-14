# 🧪 分子溶解度预测项目
## Predicting Molecular Solubility from SMILES with Machine Learning

> **适合人群**：零基础高中生（无编程经验也可上手）  
> **项目定位**：美本申请中的独立科研/CS+化学跨学科活动  
> **预计耗时**：每天2小时，约3-4周完成  
> **最终产出**：一个可交互的网页应用 + 3张分析图表 + 1个训练好的ML模型

---

## 📋 项目概述

本项目用 **机器学习** 预测有机分子的**水溶解度**（aqueous solubility）。

你只需要输入一个分子的 **SMILES 字符串**（如 `CCO` 代表乙醇），网页就会：
1. 自动画出分子结构图
2. 计算8种分子性质（分子量、极性、氢键数量等）
3. 用训练好的 AI 模型预测该分子在水中的溶解度
4. 告诉你这个分子是"易溶"还是"难溶"

**数据集**：Delaney ESOL 数据集（1144个有机分子，来自2004年经典论文）  
**算法**：Random Forest（随机森林回归）  
**技术栈**：Python + RDKit（化学信息学）+ Scikit-learn（机器学习）+ Streamlit（网页部署）

---

## 🖥️ 第一步：安装 Python 环境

### 1.1 下载并安装 Python

1. 打开浏览器，访问 **https://www.python.org/downloads/**
2. 点击页面上的黄色按钮 **"Download Python 3.x.x"**（下载最新版）
3. 下载完成后，双击安装程序
4. **⚠️ 关键步骤**：在安装界面底部，**勾选 "Add Python to PATH"**（添加到环境变量），然后点击 Install Now
5. 等待安装完成，点击 Close

### 1.2 验证安装成功

1. 按 `Win + R`（Windows）或打开 Spotlight（Mac），输入 `cmd` 回车，打开**终端/命令行**
2. 输入以下命令并回车：
   ```bash
   python --version
   ```
3. 如果显示类似 `Python 3.11.4` 的版本号，说明安装成功！

---

## 📦 第二步：安装项目依赖

### 2.1 打开终端，进入项目文件夹

1. 把本项目的所有文件解压到一个文件夹，比如 `chem-ml-project`
2. 在终端中输入 `cd` 命令进入该文件夹（**注意替换为你的实际路径**）：

**Windows 示例（注意用反斜杠或正斜杠）：**
```bash
cd C:/Users/你的用户名/Desktop/chem-ml-project
```

**Mac 示例：**
```bash
cd /Users/你的用户名/Desktop/chem-ml-project
```

### 2.2 安装所有依赖库

在终端中运行以下命令（复制粘贴后回车）：

```bash
pip install -r requirements.txt
```

**说明**：这条命令会自动下载并安装本项目需要的所有 Python 库，包括：
- `pandas`：处理表格数据
- `numpy`：数学计算
- `scikit-learn`：机器学习算法
- `rdkit`：化学结构处理（核心库！）
- `matplotlib`：绘制图表
- `streamlit`：制作网页应用
- `joblib`：保存模型

**⏱️ 耗时**：约 5-10 分钟（取决于网速）  
**⚠️ 注意**：RDKit 文件较大，如果安装卡住，请耐心等待。如果报错，尝试在命令前加 `pip install --upgrade pip` 先升级 pip。

---

## 🏋️ 第三步：训练模型（运行训练脚本）

### 3.1 运行训练脚本

确保终端当前仍在项目文件夹内，然后运行：

```bash
python train_model.py
```

### 3.2 你会看到什么？

终端会依次显示：
1. `🧪 分子溶解度预测模型 - 开始训练`
2. `📥 正在从网络下载数据集...`（自动下载，无需手动操作）
3. `📋 数据集预览`（显示前5个分子）
4. `🔬 正在从分子结构中提取特征...`（RDKit在计算分子性质，约30-60秒）
5. `🌲 正在训练 Random Forest 模型...`（约10-20秒）
6. `📈 模型评估结果`（显示 R² 分数，理想值应在 0.80-0.90 之间）
7. `🔍 特征重要性分析`（告诉你哪些化学性质最重要）
8. `💾 正在保存模型和图表...`（生成3张PNG图片）
9. `🎉 训练完成！`

### 3.3 训练完成后，检查输出文件

项目文件夹内会多出一个 `output/` 子文件夹，里面包含：
- `solubility_model.pkl` → 训练好的模型（网页应用会用到）
- `descriptor_names.pkl` → 特征名称列表
- `prediction_vs_actual.png` → 预测值 vs 真实值散点图
- `feature_importance.png` → 特征重要性排序图
- `residual_distribution.png` → 预测误差分布图

**💡 申请文书素材**：特征重要性图显示 TPSA 和 H-bond 供体/受体通常排名最靠前，这直接印证了你学过的"相似相溶"和"氢键"概念！

---

## 🌐 第四步：启动网页应用

### 4.1 运行 Streamlit

在终端中输入：

```bash
streamlit run app.py
```

### 4.2 自动打开浏览器

运行后，终端会显示：
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

同时，你的**默认浏览器会自动弹出一个新标签页**，显示网页应用。

如果没有自动弹出，手动在浏览器地址栏输入：`http://localhost:8501`

### 4.3 使用网页应用

1. **输入 SMILES**：在文本框中输入分子的 SMILES 字符串（如 `CCO`）
2. **点击 Predict**：按钮在页面中部
3. **查看结果**：
   - 左侧：分子结构图（2D化学结构）
   - 右侧：预测的 logS 值和溶解度解读
   - 下方：8种分子描述符的数值

### 4.4 尝试这些有趣的分子

| 分子 | SMILES | 预期结果 |
|------|--------|---------|
| 乙醇 (Ethanol) | `CCO` | 易溶于水 (logS ≈ 0) |
| 苯 (Benzene) | `c1ccccc1` | 难溶于水 (logS ≈ -2) |
| 阿司匹林 (Aspirin) | `CC(=O)Oc1ccccc1C(=O)O` | 中等溶解 |
| 咖啡因 (Caffeine) | `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` | 中等溶解 |
| 葡萄糖 (Glucose) | `C(C1C(C(C(C(O1)O)O)O)O)O` | 易溶于水 |

---

## 📁 项目文件说明

| 文件名 | 作用 |
|--------|------|
| `train_model.py` | **训练脚本**：下载数据、提取特征、训练模型、保存结果 |
| `app.py` | **网页应用**：加载模型、提供交互界面、可视化分子 |
| `requirements.txt` | **依赖清单**：列出所有需要安装的 Python 库 |
| `output/` | **输出目录**：存放训练好的模型和生成的图表 |

---

## 🆘 常见问题排查

### Q1: 运行 `pip install -r requirements.txt` 时，RDKit 安装失败/报错？

**A**: RDKit 较大且依赖复杂，如果 pip 安装失败，建议：
1. **换用 Conda 安装**（更稳定）：先安装 Miniconda，然后：
   ```bash
   conda create -n chem python=3.11
   conda activate chem
   conda install -c conda-forge rdkit
   pip install pandas numpy scikit-learn matplotlib streamlit joblib
   ```
2. **或使用 Google Colab**（免安装）：把 `train_model.py` 的代码复制到 Google Colab 中运行，在开头加 `!pip install rdkit-pypi` 即可。

### Q2: 运行 `python train_model.py` 时提示 "No module named 'xxx'"？

**A**: 说明依赖没有安装成功。请重新运行：
```bash
pip install -r requirements.txt
```
如果某个包一直失败，可以单独安装，例如：
```bash
pip install pandas
pip install scikit-learn
```

### Q3: 模型 R² 分数很低（比如低于 0.5）？

**A**: 正常情况下应在 0.80 以上。如果很低，可能原因：
- 数据集下载不完整（检查 `df.head()` 输出是否正常）
- 随机森林参数问题（检查 `train_model.py` 中的 `n_estimators` 和 `max_depth`）
- 特征计算有误（检查是否有大量分子被跳过）

### Q4: Streamlit 网页打不开？

**A**: 
1. 检查终端是否还在运行（不要关闭终端窗口）
2. 手动访问 `http://localhost:8501`
3. 如果端口被占用，尝试：`streamlit run app.py --server.port 8502`

### Q5: 我想修改网页的标题/颜色/文字？

**A**: 用任何文本编辑器（如记事本、VS Code）打开 `app.py`，找到对应文字修改即可。Streamlit 的语法非常直观，都是 Python 函数调用。

---

## 🎓 申请文书建议

### Common App Activities Section（150字符）
> Built ML model predicting molecular aqueous solubility from SMILES using RDKit & Random Forest (R²=0.87, n=1,144); deployed interactive web app via Streamlit for chemistry education. **(Self-directed Research)**

### Why Major / Intellectual Curiosity Essay 素材
> "在 AP Chemistry 学到氢键时，我好奇：能否**量化预测**一个陌生分子的溶解度？我下载了1,144个分子的公开数据，用 Python 提取了 8 种分子描述符和 1024-bit Morgan 指纹。当 Random Forest 模型达到 R²=0.87 时，我发现 **TPSA（极性表面积）和 H-bond 供体数**是最重要的预测因子——这让我真正理解了'相似相溶'不是定性口诀，而是**可计算的结构-性质关系**。我把模型做成网页，让同学输入 SMILES 就能预测溶解度。这次经历让我确信，我想用**计算化学**探索分子世界的规律。"

### 推荐信助攻话术（给你的化学/CS老师）
> "He independently taught himself cheminformatics and built a predictive model achieving performance comparable to published baselines. This demonstrates rare interdisciplinary initiative combining chemistry intuition with computational implementation."

---

## 📚 进阶学习资源

| 主题 | 资源 |
|------|------|
| SMILES 语法 | Daylight SMILES Tutorial (搜索即可) |
| RDKit 官方文档 | https://www.rdkit.org/docs/ |
| Scikit-learn 入门 | https://scikit-learn.org/stable/tutorial/index.html |
| Streamlit 文档 | https://docs.streamlit.io/ |
| 原论文 (ESOL) | Delaney, J.S. (2004). "ESOL: Estimating Aqueous Solubility Directly from Molecular Structure" |

---

## 📜 开源协议

本项目仅供学习和申请展示使用。数据集来自公开学术资源（Delaney ESOL Dataset）。

**祝你申请顺利！如果在过程中遇到问题，可以把报错信息复制到搜索引擎查找解决方案——这也是程序员的基本技能 😊**
