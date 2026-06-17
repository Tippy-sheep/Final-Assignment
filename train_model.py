"""
北京PM2.5预测模型训练
使用预处理后的数据进行训练
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 配置
# ============================================================
MODEL_TYPE = 'RNN'  # 可选: 'RNN', 'GRU', 'LSTM'
CLEANED_DATA_PATH = 'data/cleaned_data/BeijingPM25_cleaned.csv'  # 预处理后的数据
SEQ_LENGTH = 72  # 使用过去72(x)小时预测未来1小时
BATCH_SIZE = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 2
EPOCHS = 100
LEARNING_RATE = 0.001

CHECKPOINT_PATH = f'checkpoint_{MODEL_TYPE.lower()}.pth'
FINAL_WEIGHTS_PATH = f'models/deploy_{MODEL_TYPE.lower()}_weights.pth'

print("=" * 70)
print(f"  北京PM2.5预测模型训练")
print(f"  模型类型: {MODEL_TYPE}")
print(f"  序列长度: {SEQ_LENGTH}")
print(f"  隐藏层维度: {HIDDEN_SIZE}")
print("=" * 70)

# ============================================================
# 1. 加载预处理后的数据
# ============================================================
print(f"\n[1] 加载预处理数据: {CLEANED_DATA_PATH}")
df = pd.read_csv(CLEANED_DATA_PATH)
print(f"    数据形状: {df.shape}")
print(f"    列名: {df.columns.tolist()[:10]}...")

# 查看数据统计
print(f"\n数据统计:")
print(f"  PM2.5范围: [{df['pm25'].min():.1f}, {df['pm25'].max():.1f}]")
print(f"  PM2.5均值: {df['pm25'].mean():.1f}")

# ============================================================
# 2. 准备特征和目标
# ============================================================
print("\n[2] 准备特征和目标变量")

# 特征列（排除非特征列）
exclude_cols = ['pm25', 'datetime', 'year', 'month', 'day', 'hour', 'aqi_level']
feature_cols = [col for col in df.columns if col not in exclude_cols]
target_col = 'pm25'

print(f"特征数量: {len(feature_cols)}")
print(f"特征列: {feature_cols}")

# 提取特征和目标
X_raw = df[feature_cols].values.astype(np.float32)
y_raw = df[target_col].values.astype(np.float32).reshape(-1, 1)

print(f"特征形状: {X_raw.shape}")
print(f"目标形状: {y_raw.shape}")

# ============================================================
# 3. 创建时间序列样本
# ============================================================
print("\n[3] 创建时间序列样本")


def create_sequences(X, y, seq_length):
    """创建时序样本"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length - 1):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)

    return X_seq, y_seq


X_seq, y_seq = create_sequences(X_raw, y_raw, SEQ_LENGTH)
print(f"序列样本数: {len(X_seq)}")
print(f"X_seq形状: {X_seq.shape}")
print(f"y_seq形状: {y_seq.shape}")

# 检查是否有NaN
if np.isnan(X_seq).any() or np.isnan(y_seq).any():
    print("[警告] 发现NaN，进行清理...")
    valid_idx = ~(np.isnan(X_seq).any(axis=(1, 2)) | np.isnan(y_seq).flatten())
    X_seq = X_seq[valid_idx]
    y_seq = y_seq[valid_idx]
    print(f"清理后样本数: {len(X_seq)}")

# ============================================================
# 4. 处理异常值（关键：处理巨大的数值范围）
# ============================================================
print("\n[4] 处理异常值")

# 查看特征统计
print("特征统计:")
for i, col in enumerate(feature_cols[:5]):
    print(f"  {col}: 均值={X_raw[:, i].mean():.2f}, 标准差={X_raw[:, i].std():.2f}, 最大={X_raw[:, i].max():.2f}")

# 对异常大的值进行裁剪（使用百分位数）
for i in range(X_seq.shape[2]):
    p99 = np.percentile(X_seq[:, :, i], 99)
    p1 = np.percentile(X_seq[:, :, i], 1)
    X_seq[:, :, i] = np.clip(X_seq[:, :, i], p1, p99)

print("异常值裁剪完成")

# ============================================================
# 5. 数据标准化
# ============================================================
print("\n[5] 数据标准化")

# 计算标准化参数
x_mean = X_seq.mean(axis=(0, 1), keepdims=True)
x_std = X_seq.std(axis=(0, 1), keepdims=True) + 1e-8
y_mean = y_seq.mean()
y_std = y_seq.std() + 1e-8

# 标准化
X_seq = (X_seq - x_mean) / x_std
y_seq = (y_seq - y_mean) / y_std

print(f"X均值范围: [{x_mean.min():.3f}, {x_mean.max():.3f}]")
print(f"X标准差范围: [{x_std.min():.3f}, {x_std.max():.3f}]")
print(f"y均值: {y_mean:.3f}, y标准差: {y_std:.3f}")

# 保存标准化参数
norm_params = {
    'x_mean': x_mean, 'x_std': x_std,
    'y_mean': y_mean, 'y_std': y_std,
    'feature_cols': feature_cols
}
np.savez('output/params/normalization_params.npz', **norm_params)

# ============================================================
# 6. 划分数据集
# ============================================================
print("\n[6] 划分训练/验证/测试集")

split1 = int(len(X_seq) * 0.7)
split2 = int(len(X_seq) * 0.85)

X_train, y_train = X_seq[:split1], y_seq[:split1]
X_val, y_val = X_seq[split1:split2], y_seq[split1:split2]
X_test, y_test = X_seq[split2:], y_seq[split2:]

print(f"训练集: {X_train.shape}")
print(f"验证集: {X_val.shape}")
print(f"测试集: {X_test.shape}")

# 创建DataLoader
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 7. 定义模型
# ============================================================
print("\n[7] 定义模型")


class AirQualityRNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(AirQualityRNN, self).__init__()
        self.cell_type = MODEL_TYPE.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if self.cell_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif self.cell_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:  # LSTM
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.gradients = []

    def forward(self, x):
        if self.cell_type == 'LSTM':
            out, _ = self.rnn(x)
        else:
            out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

    def record_gradient(self):
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.gradients.append(total_norm)
        return total_norm


# 创建模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = X_train.shape[2]
model = AirQualityRNN(input_size=input_size, hidden_size=HIDDEN_SIZE,
                      num_layers=NUM_LAYERS, dropout=0.2)
model = model.to(device)

print(f"设备: {device}")
print(f"输入维度: {input_size}")
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# 8. 训练配置（修复verbose参数问题）
# ============================================================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# 移除verbose参数（新版本PyTorch不再支持）
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

# ============================================================
# 9. 训练循环
# ============================================================
print("\n[8] 开始训练")
print("-" * 70)

train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    # 训练
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)

        # 检查loss是否为nan
        if torch.isnan(loss):
            continue

        loss.backward()
        model.record_gradient()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x)
            val_loss += criterion(pred, batch_y).item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    # 学习率调整
    scheduler.step(val_loss)

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f'models/best_{MODEL_TYPE.lower()}.pth')

    # 打印进度
    if (epoch + 1) % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch + 1:3d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")

print("-" * 70)
print(f"训练完成！最佳验证损失: {best_val_loss:.6f}")

# ============================================================
# 10. 保存模型
# ============================================================
print("\n[9] 保存模型")
torch.save(model.state_dict(), FINAL_WEIGHTS_PATH)
print(f"模型已保存: {FINAL_WEIGHTS_PATH}")

# 保存梯度记录
if model.gradients:
    np.save(f'output/params/gradient_{MODEL_TYPE.lower()}.npy', np.array(model.gradients))

# ============================================================
# 11. 测试集评估
# ============================================================
print("\n[10] 测试集评估")

model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        pred = model(batch_x).cpu()
        predictions.extend(pred.numpy().flatten())
        actuals.extend(batch_y.numpy().flatten())

# 反标准化
predictions = np.array(predictions) * y_std + y_mean
actuals = np.array(actuals) * y_std + y_mean

# 计算指标
mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))
r2 = r2_score(actuals, predictions)

print(f"\n测试集结果:")
print(f"  MAE:  {mae:.2f} μg/m³")
print(f"  RMSE: {rmse:.2f} μg/m³")
print(f"  R²:   {r2:.4f}")

# ============================================================
# 12. 可视化结果
# ============================================================
print("\n[11] 生成可视化图表")

fig = plt.figure(figsize=(16, 12))

# 12.1 损失曲线
ax1 = fig.add_subplot(3, 3, 1)
ax1.plot(train_losses, label='训练损失', color='blue', linewidth=1)
ax1.plot(val_losses, label='验证损失', color='red', linewidth=1)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title(f'{MODEL_TYPE} 损失曲线')
ax1.legend()
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# 12.2 梯度变化
ax2 = fig.add_subplot(3, 3, 2)
if model.gradients:
    ax2.plot(model.gradients, color='purple', linewidth=0.5)
    ax2.set_xlabel('训练步数')
    ax2.set_ylabel('梯度范数')
    ax2.set_title(f'{MODEL_TYPE} 梯度变化')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

# 12.3 梯度分布
ax3 = fig.add_subplot(3, 3, 3)
if model.gradients:
    ax3.hist(model.gradients, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('梯度范数')
    ax3.set_ylabel('频次')
    ax3.set_title(f'{MODEL_TYPE} 梯度分布')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)

# 12.4 预测 vs 真实散点图
ax4 = fig.add_subplot(3, 3, 4)
ax4.scatter(actuals, predictions, alpha=0.3, s=5, color='steelblue')
ax4.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', linewidth=2, label='理想预测')
ax4.set_xlabel('真实PM2.5 (μg/m³)')
ax4.set_ylabel('预测PM2.5 (μg/m³)')
ax4.set_title(f'{MODEL_TYPE} 预测散点图 (R²={r2:.3f})')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 12.5 残差分布
ax5 = fig.add_subplot(3, 3, 5)
residuals = actuals - predictions
ax5.hist(residuals, bins=50, color='orange', alpha=0.7, edgecolor='black')
ax5.axvline(0, color='red', linestyle='--', linewidth=2)
ax5.set_xlabel('预测误差 (μg/m³)')
ax5.set_ylabel('频次')
ax5.set_title('残差分布')
ax5.grid(True, alpha=0.3)

# 12.6 时间序列对比
ax6 = fig.add_subplot(3, 3, 6)
sample_size = min(300, len(actuals))
ax6.plot(actuals[:sample_size], label='真实值', color='red', linewidth=1, alpha=0.8)
ax6.plot(predictions[:sample_size], label='预测值', color='green', linewidth=1, linestyle='--', alpha=0.8)
ax6.set_xlabel('时间步')
ax6.set_ylabel('PM2.5 (μg/m³)')
ax6.set_title(f'{MODEL_TYPE} 预测波形对比')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 12.7 MAE随浓度变化
ax7 = fig.add_subplot(3, 3, 7)
bins = np.arange(0, 300, 50)
bin_indices = np.digitize(actuals, bins)
mae_per_bin = []
for i in range(1, len(bins)):
    mask = bin_indices == i
    if mask.any():
        mae_per_bin.append(np.mean(np.abs(actuals[mask] - predictions[mask])))
    else:
        mae_per_bin.append(0)
ax7.bar(bins[:-1], mae_per_bin, width=45, color='coral')
ax7.set_xlabel('PM2.5浓度范围 (μg/m³)')
ax7.set_ylabel('MAE (μg/m³)')
ax7.set_title('不同浓度下的预测误差')
ax7.grid(True, alpha=0.3)

# 12.8 误差统计箱线图
ax8 = fig.add_subplot(3, 3, 8)
# 重新加载数据获取AQI等级
df_full = pd.read_csv(CLEANED_DATA_PATH)
if 'aqi_level' in df_full.columns:
    levels = ['优', '良', '轻度污染', '中度污染', '重度污染', '严重污染']
    error_by_level = []
    for level in levels:
        # 匹配对应长度的数据
        mask = df_full.head(len(actuals))['aqi_level'] == level
        if mask.any():
            error_by_level.append(residuals[mask.values])
        else:
            error_by_level.append([])
    bp = ax8.boxplot(error_by_level, labels=levels, patch_artist=True)
    ax8.set_xlabel('空气质量等级')
    ax8.set_ylabel('预测误差 (μg/m³)')
    ax8.set_title('不同空气质量等级的预测误差')
    ax8.tick_params(axis='x', rotation=45)

# 12.9 指标汇总
ax9 = fig.add_subplot(3, 3, 9)
ax9.axis('off')
metrics_text = f"""
{'='*35}
MODEL PERFORMANCE SUMMARY
{'='*35}
Model Type: {MODEL_TYPE}
Seq Length: {SEQ_LENGTH} hours
Hidden Size: {HIDDEN_SIZE}
Num Layers: {NUM_LAYERS}

TEST RESULTS:
  MAE:  {mae:.2f} μg/m³
  RMSE: {rmse:.2f} μg/m³
  R²:   {r2:.4f}

TRAINING INFO:
  Epochs: {EPOCHS}
  Best Val Loss: {best_val_loss:.6f}
{'='*35}
"""
# metrics_text = f"""
# 模型性能指标汇总
# {'=' * 30}
# 模型类型: {MODEL_TYPE}
# 序列长度: {SEQ_LENGTH}小时
# 隐藏层维度: {HIDDEN_SIZE}
# 层数: {NUM_LAYERS}
#
# 测试集结果:
#   MAE:  {mae:.2f} μg/m³
#   RMSE: {rmse:.2f} μg/m³
#   R²:   {r2:.4f}
#
# 训练信息:
#   训练轮数: {EPOCHS}
#   最佳验证损失: {best_val_loss:.6f}
# """
ax9.text(0.1, 0.5, metrics_text, transform=ax9.transAxes,
         fontsize=10, verticalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle(f'北京PM2.5预测系统 - {MODEL_TYPE}模型分析报告', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'output/training_analysis/{MODEL_TYPE.lower()}.png', dpi=150, facecolor='white')
print(f"图表已保存: output/training_analysis/{MODEL_TYPE.lower()}.png")
plt.show()

print("\n" + "=" * 70)
print("  训练完成！")
print("=" * 70)