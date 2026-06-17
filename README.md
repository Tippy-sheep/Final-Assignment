# Final-Assignment

北京 PM2.5 智能监测与预测系统（课程期末作业）

本项目实现了从原始观测数据到模型训练、推理和前端可视化的完整流程：

- 数据预处理与探索：`data_preprocess.py`
- 序列模型训练：`train_model.py`，支持 `RNN` / `GRU` / `LSTM`
- 边缘 WebSocket 服务：`edge_server.py`
- 前端展示页面：`index.html`
- 一键运行脚本：`run.py`

## 关键特性

- 使用北京 PM2.5 历史数据进行数据清洗、特征工程与可视化分析
- 支持不同循环神经网络模型进行时序预测
- 支持一键执行数据预处理、模型训练、启动服务并打开前端页面

## 快速开始

```bash
# 克隆仓库（如果尚未克隆）
git clone https://github.com/Tippy-sheep/Sheep-owner.git
cd Sheep-owner

# 创建并激活 Python 虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
```

### 运行完整流程

```bash
python run.py --model GRU
```

这会按顺序执行：依赖检查、数据预处理、模型训练、启动 WebSocket 服务器，并自动打开 `index.html` 页面。

### 常用命令

```bash
# 仅运行数据预处理
python data_preprocess.py

# 仅训练模型
python train_model.py

# 跳过预处理，直接训练并启动服务器
python run.py --model GRU --skip-preprocess

# 跳过训练，仅启动服务器（需已有模型权重）
python run.py --model GRU --skip-train

# 启动时不自动打开浏览器
python run.py --model GRU --no-browser
```

### 可选模型类型

- `LSTM`
- `GRU`
- `RNN`

## 项目结构

- `data/`
  - `raw_data/`: 原始 CSV 数据
  - `cleaned_data/`: 清洗后数据输出
  - `data_analysis_report/`: 数据分析结果图表
- `models/`: 训练产生的模型权重文件（`.pth`）
- `output/`: 标准化参数与训练分析结果
- `data_preprocess.py`: 数据清洗、特征工程和分析脚本
- `train_model.py`: 模型训练与评估脚本
- `edge_server.py`: WebSocket 推理服务器
- `index.html`: 前端可视化页面
- `run.py`: 一键运行脚本，支持命令行参数和交互菜单
- `requirements.txt`: Python 依赖包列表
- `DEPENDENCIES.md`: 详细运行说明

## 依赖与环境

- Python 3.8+
- 推荐 8GB 内存，训练时建议有 NVIDIA GPU 与 CUDA
- 依赖包请参见 `requirements.txt`

## 运行提示

- 服务器默认监听：`ws://127.0.0.1:8765`
- 若自动打开浏览器失败，可手动在浏览器中打开 `index.html`
- 若遇到显存不足，可降低 `train_model.py` 中的 `BATCH_SIZE`，或使用 CPU 训练


