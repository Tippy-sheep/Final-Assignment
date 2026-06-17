# Final-Assignment

北京PM2.5智能监测与预测系统（课程期末作业）

本项目实现了从原始观测数据到预测模型和可视化前端的完整流程：

- 数据预处理与探索（`data_preprocess.py`）
- 序列模型训练（`train_model.py`，支持 `RNN` / `GRU` / `LSTM`）
- 边缘 WebSocket 服务（`edge_server.py`）用于模型推理与前端交互
- 简单前端页面（`index.html`）用于展示预测结果与交互

主要特点

- 基于北京及多城市 PM2.5 历史数据进行清洗、特征工程与可视化分析
- 使用循环神经网络系列模型进行时序预测，包含训练/验证/测试流程与可视化结果
- 提供一键运行脚本 `run.py`，支持依次执行预处理、训练并启动服务器

依赖与环境

参见 `requirements.txt` 与 `DEPENDENCIES.md` 获取详细安装说明与运行要求（推荐在虚拟环境中安装）。

快速开始

```bash
# 克隆仓库（如果还未克隆）
git clone https://github.com/Tippy-sheep/Sheep-owner.git
cd Sheep-owner

# 建议创建并激活虚拟环境，然后安装依赖
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 运行完整流程（依赖检查 → 预处理 → 训练 → 启动服务器）
python run.py --model GRU

# 单独运行预处理或训练
python data_preprocess.py
python train_model.py
```

项目结构（重要文件）

- `data/`：原始与清洗后数据、分析报告图片
- `models/`：模型权重文件（`.pth`）
- `output/`：训练分析图与标准化参数
- `data_preprocess.py`：数据清洗与特征工程脚本
- `train_model.py`：模型训练与评估脚本
- `edge_server.py`：WebSocket 服务端脚本
- `index.html`：前端展示页面
- `run.py`：一键运行脚本（带菜单和命令行选项）
- `requirements.txt`：Python 依赖列表
- `DEPENDENCIES.md`：详细运行说明（包含 PyTorch 安装提示）


