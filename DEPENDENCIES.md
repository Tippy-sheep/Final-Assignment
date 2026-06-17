# 运行要求与使用说明

本文档说明项目运行所需的环境、依赖及常用运行流程（数据预处理、模型训练、部署/启动服务器、前端可视化）。建议在虚拟环境或 Conda 环境中执行下面操作。

**最低环境**

- Python 3.8+（推荐 3.8-3.11）
- 推荐内存：至少 8GB，训练时建议有 GPU（NVIDIA + CUDA）

**依赖包（已列入 requirements.txt）**

- torch
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- websockets

安装依赖（在项目目录下）：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

关于 PyTorch：请根据你的平台与 CUDA 版本在官方页面选择安装命令：

https://pytorch.org/

示例（CPU-only）：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

或使用官方给出的带 CUDA 支持的命令（按需替换）。

数据和目录说明

- 原始数据: `data/raw_data/BeijingPM20100101_20151231.csv`（脚本 `data_preprocess.py` 依赖该文件）
- 清洗后数据输出: `data/cleaned_data/BeijingPM25_cleaned.csv`
- 模型权重: `models/` 下的 `.pth` 文件（训练后生成 `models/deploy_<model>_weights.pth` 或 `models/best_<model>.pth`）
- 训练输出/分析图: `output/params/`、`output/training_analysis/`

主要脚本与运行顺序

1. 数据预处理（独立运行）

```bash
cd "Deep Learning Principles and Applications Final Assignment"
python data_preprocess.py
```

运行后会生成 `data/cleaned_data/BeijingPM25_cleaned.csv` 和数据分析图片 `data/data_analysis_report/data_analysis.png`。

2. 模型训练

- 可直接运行训练脚本（训练前请确保已生成清洗数据）：

```bash
python train_model.py
```

- 可通过 `run.py` 一键运行（包含依赖检查、预处理、训练、启动服务器）：

```bash
python run.py --model GRU            # 指定模型: LSTM/GRU/RNN
python run.py --model GRU --skip-preprocess  # 跳过预处理
python run.py --model GRU --skip-train       # 跳过训练，仅启动服务器 (需已存在权重)
```

注意：`train_model.py` 中默认有一些配置变量（如 `MODEL_TYPE`, `SEQ_LENGTH`, `EPOCHS` 等），可直接在文件中调整或使用 `run.py` 来设置 `MODEL_TYPE`。

3. 启动边缘服务器与前端

- 直接启动服务器：

```bash
python edge_server.py
```

- 或使用 `run.py` 在完成训练后自动启动服务器并打开 `index.html`（默认会尝试在浏览器中打开本地页面）：

```bash
python run.py
```

服务器将监听 WebSocket（默认在 `ws://127.0.0.1:8765`），前端页面为 `index.html`，可以直接用浏览器打开查看可视化界面。

调试与常见问题

- 若提示缺少依赖，请先安装 `requirements.txt` 中的包，或手动 `pip install <package>`。
- 如果训练显存不足，降低 `BATCH_SIZE` 或在 `train_model.py` 中切换到 CPU（或在无 CUDA 环境下自动使用 CPU）。
- 若数据文件路径不匹配，请将原始数据放置在 `data/raw_data/`，或修改 `data_preprocess.py` 中的 `DATA_PATH`。

示例完整运行流程（无交互）：

```bash
cd "Deep Learning Principles and Applications Final Assignment"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py --model GRU
```

