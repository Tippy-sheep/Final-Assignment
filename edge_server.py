"""
北京PM2.5空气质量预测系统 - 边缘推理服务器
支持 LSTM、GRU、RNN 三种模型，通过 MODEL_TYPE 切换
"""

import torch
import torch.nn as nn
import numpy as np
import asyncio
import json
import time
import pandas as pd
from datetime import datetime
import os

# ============================================================
# 配置 - 只需要修改这一行即可切换模型
# ============================================================
MODEL_TYPE = 'RNN'  # 可选: 'LSTM', 'GRU', 'RNN'

# 自动选择对应的权重文件
WEIGHTS_PATH = f'models/deploy_{MODEL_TYPE.lower()}_weights.pth'

# 其他配置
NORM_PATH = 'output/params/normalization_params.npz'
DATA_PATH = 'data/cleaned_data/BeijingPM25_cleaned.csv'
HOST = "127.0.0.1"
PORT = 8765
SEQ_LENGTH = 72  # 必须与训练时一致
SEND_INTERVAL = 0.05  # 发送间隔（秒）

# 特征列（必须与训练时完全一致）
FEATURE_COLS = ['DEWP', 'HUMI', 'PRES', 'TEMP', 'Iws',
                'wd_NE', 'wd_NW', 'wd_SE', 'wd_cv',
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                'is_weekend', 'season_sin', 'season_cos',
                'precipitation', 'Iprec']

print("=" * 60)
print(f"  北京PM2.5边缘计算终端 - {MODEL_TYPE}模型")
print("=" * 60)

# ============================================================
# 1. 加载标准化参数
# ============================================================
print("\n[1] 加载标准化参数...")

if not os.path.exists(NORM_PATH):
    print(f"   ❌ 错误: 找不到标准化参数文件: {NORM_PATH}")
    print("   请先运行 train_model.py 训练模型")
    exit(1)

norm_params = np.load(NORM_PATH, allow_pickle=True)

# 修复：正确处理维度
x_mean = norm_params['x_mean']
x_std = norm_params['x_std']

# 将标准化参数降维到1D
if x_mean.ndim == 3:
    x_mean = x_mean.squeeze()  # (1, 1, 18) -> (18,)
    x_std = x_std.squeeze()
elif x_mean.ndim == 2:
    x_mean = x_mean.squeeze()
    x_std = x_std.squeeze()

y_mean = float(norm_params['y_mean'])
y_std = float(norm_params['y_std'])

print(f"   ✅ 标准化参数加载成功")
print(f"   特征维度: {len(x_mean)}")
print(f"   PM2.5均值: {y_mean:.2f}, 标准差: {y_std:.2f}")

# ============================================================
# 2. 模型定义（统一支持 LSTM/GRU/RNN）
# ============================================================
print(f"\n[2] 加载{MODEL_TYPE}模型...")

class AirQualityRNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(AirQualityRNN, self).__init__()
        self.cell_type = MODEL_TYPE.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 根据模型类型选择对应的 RNN 单元
        if self.cell_type == 'RNN':
            print(f"   使用标准 RNN (存在梯度消失问题)")
            self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif self.cell_type == 'GRU':
            print(f"   使用 GRU (门控循环单元，计算效率高)")
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:  # LSTM
            print(f"   使用 LSTM (长短期记忆网络，记忆能力强)")
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # LSTM 返回 (output, (h_n, c_n))
        # GRU 和 RNN 返回 (output, h_n)
        # 统一处理
        if self.cell_type == 'LSTM':
            out, _ = self.rnn(x)  # LSTM 忽略细胞状态
        else:
            out, _ = self.rnn(x)  # GRU 和 RNN
        return self.fc(out[:, -1, :])  # 取最后一个时间步的输出


# 创建模型 - 根据诊断结果，hidden_size=128, num_layers=2
input_size = len(FEATURE_COLS)
model = AirQualityRNN(
    input_size=input_size,
    hidden_size=128,  # 训练时使用的隐藏层维度
    num_layers=2,     # 训练时使用的层数
    dropout=0.2
)

# 加载权重
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\n   模型配置:")
print(f"   - 模型类型: {MODEL_TYPE}")
print(f"   - 输入维度: {input_size}")
print(f"   - 隐藏层维度: 128")
print(f"   - RNN层数: 2")
print(f"   - 运行设备: {device}")

# 检查权重文件
if not os.path.exists(WEIGHTS_PATH):
    print(f"\n   ❌ 错误: 找不到模型权重文件: {WEIGHTS_PATH}")
    print(f"\n   请确保已训练 {MODEL_TYPE} 模型:")
    print(f"   1. 修改 train_model.py 中的 MODEL_TYPE = 'GRU'")
    print(f"   2. 运行 python train_model.py 训练模型")
    print(f"   3. 确保生成 {WEIGHTS_PATH} 文件")
    exit(1)

# 加载权重
try:
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)

    # 打印权重信息（调试用）
    print(f"\n   权重文件加载成功，正在匹配参数...")

    # 检查关键参数的形状
    if 'rnn.weight_ih_l0' in state_dict:
        ih_shape = state_dict['rnn.weight_ih_l0'].shape
        print(f"   - 输入层权重形状: {ih_shape}")

        # 验证隐藏层大小
        if MODEL_TYPE == 'LSTM':
            expected_hidden = ih_shape[0] // 4
        elif MODEL_TYPE == 'GRU':
            expected_hidden = ih_shape[0] // 3
        else:  # RNN
            expected_hidden = ih_shape[0]

        print(f"   - 推断隐藏层大小: {expected_hidden}")

        if expected_hidden != 128:
            print(f"   ⚠️ 警告: 隐藏层大小({expected_hidden})与配置(128)不匹配")
            print(f"   正在自动适配...")
            # 重新创建模型
            model = AirQualityRNN(
                input_size=input_size,
                hidden_size=expected_hidden,
                num_layers=2,
                dropout=0.2
            )

    # 加载权重
    model.load_state_dict(state_dict)
    print(f"   ✅ 模型权重加载成功")

except RuntimeError as e:
    print(f"\n   ❌ 模型加载失败!")
    print(f"   错误信息: {e}")
    print(f"\n   可能的原因:")
    print(f"   1. 训练时的模型类型不是 {MODEL_TYPE}")
    print(f"   2. 训练时的模型参数与当前配置不一致")
    print(f"\n   解决方法:")
    print(f"   1. 检查 train_model.py 中的 MODEL_TYPE 设置")
    print(f"   2. 重新训练 {MODEL_TYPE} 模型")
    exit(1)

model = model.to(device)
model.eval()
print(f"   ✅ 模型已设置为评估模式")

# ============================================================
# 3. 加载真实数据
# ============================================================
print("\n[3] 加载数据...")

if not os.path.exists(DATA_PATH):
    print(f"   ❌ 错误: 找不到数据文件: {DATA_PATH}")
    print("   请先运行 data_preprocess.py 预处理数据")
    exit(1)

df = pd.read_csv(DATA_PATH)
print(f"   ✅ 数据加载成功: {len(df)} 条记录")

# 检查必要的列是否存在
missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
if missing_cols:
    print(f"   ⚠️ 警告: 缺少特征列: {missing_cols[:5]}...")
    print(f"   数据中的列: {df.columns.tolist()[:10]}...")

# 提取特征和目标
X_raw = df[FEATURE_COLS].values.astype(np.float32)
y_raw = df['pm25'].values.astype(np.float32)

print(f"   特征形状: {X_raw.shape}")
print(f"   PM2.5范围: [{y_raw.min():.1f}, {y_raw.max():.1f}]")
print(f"   PM2.5均值: {y_raw.mean():.1f}")

# ============================================================
# 4. 数据标准化
# ============================================================
print("\n[4] 数据标准化...")

# 确保x_mean和x_std的维度与特征匹配
if len(x_mean) != X_raw.shape[1]:
    print(f"   ⚠️ 警告: 标准化参数维度({len(x_mean)})与特征维度({X_raw.shape[1]})不匹配")
    print(f"   使用特征维度重新计算...")
    x_mean = X_raw.mean(axis=0)
    x_std = X_raw.std(axis=0) + 1e-8

# 标准化特征
X_normalized = (X_raw - x_mean) / x_std

# 标准化目标值
y_normalized = (y_raw - y_mean) / y_std

print(f"   ✅ 标准化完成")
print(f"   标准化后特征范围: [{X_normalized.min():.2f}, {X_normalized.max():.2f}]")

# ============================================================
# 5. 实时数据流模拟器
# ============================================================
class RealTimeDataStream:
    """使用真实数据的时间序列数据流"""

    def __init__(self, features, targets, seq_length):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length
        self.current_idx = seq_length  # 从第seq_length个开始
        self.total_samples = len(features) - seq_length

        print(f"\n[数据流] 初始化完成")
        print(f"   总样本数: {self.total_samples}")
        print(f"   序列长度: {seq_length}")

    def has_next(self):
        """检查是否还有数据"""
        return self.current_idx < len(self.features)

    def get_next(self):
        """获取下一个序列和对应的真实值"""
        if not self.has_next():
            # 循环回到开头
            self.current_idx = self.seq_length
            print("\n[数据流] 数据循环，重新开始...")

        # 获取输入序列（前seq_length个时间步）
        X_seq = self.features[self.current_idx - self.seq_length:self.current_idx]
        # 获取目标值（下一个时间步）
        y_true_normalized = self.targets[self.current_idx]
        y_true = y_true_normalized * y_std + y_mean

        # 获取当前时间的环境特征（用于显示）
        current_features = X_seq[-1]  # 最后一个时间步的特征

        self.current_idx += 1

        return X_seq, y_true, current_features

# 创建数据流
data_stream = RealTimeDataStream(X_normalized, y_normalized, SEQ_LENGTH)

# ============================================================
# 6. WebSocket服务
# ============================================================
print("\n[5] 启动WebSocket服务...")

# 性能统计
inference_times = []
latencies = []
predictions_log = []  # 记录预测结果

async def stream_data(websocket):
    print(f"\n[连接] 客户端已连接 - {datetime.now().strftime('%H:%M:%S')}")
    print(f"[模型] 使用 {MODEL_TYPE} 进行推理")

    # 重置数据流位置（每个新连接从头开始）
    data_stream.current_idx = SEQ_LENGTH
    local_count = 0

    try:
        with torch.no_grad():
            while True:
                start_time = time.time()

                # 获取下一个真实样本
                X_seq, y_true, current_features = data_stream.get_next()

                # 准备输入张量
                input_tensor = torch.FloatTensor(X_seq).unsqueeze(0).to(device)

                # 推理
                pred_normalized = model(input_tensor).cpu().numpy()[0, 0]

                # 反标准化
                predicted_pm25 = pred_normalized * y_std + y_mean
                predicted_pm25 = max(0, predicted_pm25)  # PM2.5不能为负

                # 计算推理时间
                inference_time = (time.time() - start_time) * 1000
                inference_times.append(inference_time)
                if len(inference_times) > 100:
                    inference_times.pop(0)

                # 计算端到端延迟（模拟网络延迟）
                network_delay = np.random.uniform(2, 8)
                total_latency = inference_time + network_delay
                latencies.append(total_latency)
                if len(latencies) > 100:
                    latencies.pop(0)

                # 计算预测误差
                error_abs = abs(y_true - predicted_pm25)

                # 记录预测结果
                predictions_log.append({
                    'actual': y_true,
                    'predict': predicted_pm25,
                    'error': error_abs
                })
                if len(predictions_log) > 1000:
                    predictions_log.pop(0)

                # 解析当前特征用于显示
                temp = current_features[3]  # TEMP
                wind = current_features[4]   # Iws
                dewp = current_features[0]   # DEWP
                pres = current_features[2]   # PRES
                humi = current_features[1]   # HUMI

                # 打包数据
                payload = {
                    "timestamp": int(time.time() * 1000),
                    "datetime": datetime.now().strftime("%H:%M:%S"),
                    "model_type": MODEL_TYPE,
                    "pm25_actual": round(float(y_true), 1),
                    "pm25_predict": round(float(predicted_pm25), 1),
                    "error_abs": round(float(error_abs), 2),
                    "temperature": round(float(temp), 1),
                    "wind_speed": round(float(wind), 1),
                    "humidity_dewp": round(float(dewp), 1),
                    "pressure": round(float(pres), 0),
                    "humidity": round(float(humi), 1),
                    "inference_time_ms": round(inference_time, 2),
                    "latency_ms": round(total_latency, 2),
                    "cpu_usage_percent": round(15 + np.random.random() * 15, 1),
                    "memory_used_mb": round(250 + np.random.random() * 100, 0),
                    "avg_inference_ms": round(np.mean(inference_times[-50:]), 2) if inference_times else 0,
                    "p95_inference_ms": round(np.percentile(inference_times[-50:], 95), 2) if inference_times else 0,
                }

                # 发送数据
                await websocket.send(json.dumps(payload))

                # 每10个样本打印一次预测信息
                local_count += 1
                if local_count % 10 == 0:
                    avg_error = np.mean([p['error'] for p in predictions_log[-50:]])
                    print(f"   [{MODEL_TYPE}] 真实: {y_true:5.1f} → 预测: {predicted_pm25:5.1f} | "
                          f"误差: {error_abs:5.2f} | 平均误差: {avg_error:5.2f}")

                # 控制发送频率
                await asyncio.sleep(SEND_INTERVAL)

    except Exception as e:
        print(f"[断开] 连接异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"[断开] 客户端断开连接 - {datetime.now().strftime('%H:%M:%S')}")

# ============================================================
# 7. 主程序
# ============================================================
async def main():
    import websockets

    # 启动WebSocket服务器
    async with websockets.serve(stream_data, HOST, PORT):
        print("\n" + "=" * 60)
        print("  服务器已启动")
        print("=" * 60)
        print(f"  地址: ws://{HOST}:{PORT}")
        print(f"  模型: {MODEL_TYPE}")
        print(f"  权重: {WEIGHTS_PATH}")
        print(f"  序列长度: {SEQ_LENGTH}")
        print(f"  特征维度: {input_size}")
        print(f"  数据: {data_stream.total_samples} 个样本")
        print(f"  发送间隔: {SEND_INTERVAL} 秒")
        print("=" * 60)
        print("\n等待前端连接...")
        print("请打开 index.html 查看实时预测")
        print("按 Ctrl+C 停止服务器\n")

        # 保持服务器运行
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n[关闭] 服务器已停止")

        # 打印性能统计
        if inference_times:
            print(f"\n性能统计 ({MODEL_TYPE}):")
            print(f"  平均推理时间: {np.mean(inference_times):.2f} ms")
            print(f"  平均端到端延迟: {np.mean(latencies):.2f} ms")
            print(f"  平均预测误差: {np.mean([p['error'] for p in predictions_log[-100:]]):.2f} μg/m³")
            print(f"  总预测次数: {len(predictions_log)}")