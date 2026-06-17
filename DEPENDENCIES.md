# 运行要求与使用说明

本文档说明项目运行所需的环境、依赖及常见使用流程。建议在虚拟环境或 Conda 环境中执行下面操作。

## 环境要求

- Python 3.8+（推荐 3.8-3.11）
- 推荐内存：至少 8GB
- 训练时建议使用 NVIDIA GPU + CUDA（如果具备）

## 依赖包

本项目依赖项已列入 `requirements.txt`：

- torch
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- websockets

> `asyncio` 为 Python 标准库模块，无需额外安装。

## 安装依赖

在项目根目录下执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> 如果需要 GPU 版本的 PyTorch，请前往 https://pytorch.org/ 选择适合当前 CUDA 版本的安装命令。

示例（CPU 版本）：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 数据与目录说明

- 原始数据：`data/raw_data/BeijingPM20100101_20151231.csv`
- 清洗后数据输出：`data/cleaned_data/BeijingPM25_cleaned.csv`
- 数据分析报告：`data/data_analysis_report/`
- 模型权重：`models/` 下的 `.pth` 文件（`best_<model>.pth` / `deploy_<model>_weights.pth`）
- 训练输出与参数：`output/params/`、`output/training_analysis/`

## 主要脚本说明

- `data_preprocess.py`：原始数据清洗、特征工程与分析
- `train_model.py`：模型训练、评估与结果保存
- `edge_server.py`：WebSocket 服务端，用于前端推理交互
- `run.py`：一键运行脚本，包含依赖检查、预处理、训练和服务器启动
- `index.html`：可视化前端页面

## 推荐运行顺序

1. 运行数据预处理：

```bash
python data_preprocess.py
```

2. 训练模型：

```bash
python train_model.py
```

3. 启动服务器：

```bash
python edge_server.py
```

4. 打开前端：

在浏览器中打开 `index.html`，确认页面连接 `ws://127.0.0.1:8765`。

## 一键运行方式

使用 `run.py` 可以自动完成依赖检查、预处理、训练和服务器启动：

```bash
python run.py --model GRU
```

支持选项：

- `--model LSTM|GRU|RNN`：选择模型类型，默认 `GRU`
- `--skip-preprocess`：跳过数据预处理
- `--skip-train`：跳过模型训练，仅启动服务器
- `--no-browser`：启动后不自动打开浏览器

示例：

```bash
python run.py --model GRU --skip-preprocess
python run.py --model GRU --skip-train
python run.py --model GRU --no-browser
```

## 启动服务器与前端

- 服务器默认监听：`ws://127.0.0.1:8765`
- 前端页面：`index.html`
- `run.py` 默认会尝试自动打开浏览器，如果失败可手动打开 `index.html`

## 常见问题

- 若提示缺少依赖，请先安装 `requirements.txt` 中列出的包。
- 如果训练显存不足，可降低 `train_model.py` 中的 `BATCH_SIZE` 或在无 CUDA 环境下使用 CPU。
- 若数据文件路径不匹配，请将原始数据放置到 `data/raw_data/`，或修改 `data_preprocess.py` 中的 `DATA_PATH`。

