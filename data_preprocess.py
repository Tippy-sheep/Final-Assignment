"""
北京PM2.5数据预处理与探索性分析
独立运行，生成清洗后的数据文件
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("  北京PM2.5数据预处理与探索性分析")
print("=" * 70)

# ============================================================
# 1. 加载原始数据
# ============================================================
DATA_PATH = 'data/raw_data/BeijingPM20100101_20151231.csv'
OUTPUT_PATH = 'data/cleaned_data/BeijingPM25_cleaned.csv'

print(f"\n[1] 加载原始数据: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"    原始形状: {df.shape}")
print(f"    列名: {df.columns.tolist()}")

# 显示前5行
print("\n前5行数据预览:")
print(df.head())

# ============================================================
# 2. 数据基本信息
# ============================================================
print("\n" + "=" * 70)
print("[2] 数据基本信息")
print("=" * 70)

print(f"\n数据类型:")
print(df.dtypes)

print(f"\n缺失值统计:")
missing_stats = df.isnull().sum()
missing_pct = (missing_stats / len(df)) * 100
missing_df = pd.DataFrame({'缺失数量': missing_stats, '缺失百分比': missing_pct})
print(missing_df[missing_df['缺失数量'] > 0])

# ============================================================
# 3. PM2.5各站点数据对比
# ============================================================
print("\n" + "=" * 70)
print("[3] PM2.5各监测站点数据对比")
print("=" * 70)

pm25_cols = ['PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan', 'PM_US Post']
for col in pm25_cols:
    if col in df.columns:
        valid_count = df[col].notna().sum()
        valid_pct = valid_count / len(df) * 100
        mean_val = df[col].mean()
        print(f"  {col}: 有效数据 {valid_count}/{len(df)} ({valid_pct:.1f}%), 均值={mean_val:.1f}")

# 选择最优的PM2.5数据源（美国大使馆数据质量最好）
best_pm25_col = 'PM_US Post'
if df[best_pm25_col].isna().sum() > len(df) * 0.5:
    best_pm25_col = 'PM_Dongsi'
print(f"\n选择使用: {best_pm25_col}")

# ============================================================
# 4. 处理缺失值
# ============================================================
print("\n" + "=" * 70)
print("[4] 缺失值处理")
print("=" * 70)

# 创建处理后的DataFrame
df_clean = df.copy()

# 4.1 处理PM2.5目标变量
print(f"\n处理前 PM2.5 ({best_pm25_col}) 缺失数: {df_clean[best_pm25_col].isna().sum()}")

# 方法1: 删除PM2.5为空的记录
df_clean = df_clean[df_clean[best_pm25_col].notna()]
print(f"删除PM2.5空值后: {len(df_clean)} 条记录")

# 4.2 处理特征缺失值（使用插值法）
feature_cols = ['DEWP', 'HUMI', 'PRES', 'TEMP', 'Iws']
for col in feature_cols:
    if col in df_clean.columns:
        before = df_clean[col].isna().sum()
        # 线性插值
        df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
        # 向前填充剩余的空值
        df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
        # 最后填充0
        df_clean[col] = df_clean[col].fillna(0)
        after = df_clean[col].isna().sum()
        print(f"  {col}: 填充前={before}, 填充后={after}")

# 4.3 处理风向（转换为数值编码）
if 'cbwd' in df_clean.columns:
    # 查看风向分布
    print(f"\n风向分布:")
    print(df_clean['cbwd'].value_counts())

    # 独热编码
    dummies = pd.get_dummies(df_clean['cbwd'], prefix='wd')
    df_clean = pd.concat([df_clean, dummies], axis=1)
    print(f"风向独热编码完成，新增列: {dummies.columns.tolist()}")

# 4.4 处理降水数据
if 'precipitation' in df_clean.columns:
    df_clean['precipitation'] = df_clean['precipitation'].fillna(0)
if 'Iprec' in df_clean.columns:
    df_clean['Iprec'] = df_clean['Iprec'].fillna(0)

# ============================================================
# 5. 创建时间特征
# ============================================================
print("\n" + "=" * 70)
print("[5] 创建时间特征")
print("=" * 70)

# 创建datetime列
df_clean['datetime'] = pd.to_datetime(df_clean[['year', 'month', 'day', 'hour']])

# 周期性编码
df_clean['hour_sin'] = np.sin(2 * np.pi * df_clean['hour'] / 24)
df_clean['hour_cos'] = np.cos(2 * np.pi * df_clean['hour'] / 24)
df_clean['month_sin'] = np.sin(2 * np.pi * df_clean['month'] / 12)
df_clean['month_cos'] = np.cos(2 * np.pi * df_clean['month'] / 12)

# 星期几（0=周一, 6=周日）
df_clean['weekday'] = df_clean['datetime'].dt.weekday
df_clean['is_weekend'] = (df_clean['weekday'] >= 5).astype(int)

# 季节（1=春,2=夏,3=秋,4=冬）
df_clean['season_sin'] = np.sin(2 * np.pi * (df_clean['season'] - 1) / 4)
df_clean['season_cos'] = np.cos(2 * np.pi * (df_clean['season'] - 1) / 4)

print(f"新增时间特征: hour_sin/cos, month_sin/cos, weekday, is_weekend, season_sin/cos")

# ============================================================
# 6. 选择最终特征
# ============================================================
print("\n" + "=" * 70)
print("[6] 选择最终特征")
print("=" * 70)

# 基础特征
base_features = ['DEWP', 'HUMI', 'PRES', 'TEMP', 'Iws']

# 风向特征
wind_features = [col for col in df_clean.columns if col.startswith('wd_')]

# 时间特征
time_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'is_weekend', 'season_sin', 'season_cos']

# 降水特征
precip_features = []
if 'precipitation' in df_clean.columns:
    precip_features.append('precipitation')
if 'Iprec' in df_clean.columns:
    precip_features.append('Iprec')

# 合并所有特征
all_features = base_features + wind_features + time_features + precip_features
target_col = best_pm25_col

print(f"\n目标列: {target_col}")
print(f"特征列 ({len(all_features)}个):")
for i, f in enumerate(all_features):
    print(f"  {i+1}. {f}")

# 检查特征是否存在
available_features = [f for f in all_features if f in df_clean.columns]
print(f"\n实际可用特征: {len(available_features)}个")

# ============================================================
# 7. 数据可视化分析
# ============================================================
print("\n" + "=" * 70)
print("[7] 数据可视化分析")
print("=" * 70)

fig = plt.figure(figsize=(16, 12))

# 7.1 PM2.5时间序列
ax1 = fig.add_subplot(3, 2, 1)
sample_data = df_clean.head(1000)
ax1.plot(sample_data['datetime'], sample_data[target_col], linewidth=0.5, color='red')
ax1.set_xlabel('时间')
ax1.set_ylabel('PM2.5 (μg/m³)')
ax1.set_title('PM2.5时间序列（前1000小时）')
ax1.grid(True, alpha=0.3)

# 7.2 PM2.5分布直方图
ax2 = fig.add_subplot(3, 2, 2)
ax2.hist(df_clean[target_col], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax2.set_xlabel('PM2.5 (μg/m³)')
ax2.set_ylabel('频次')
ax2.set_title('PM2.5浓度分布')
ax2.axvline(df_clean[target_col].mean(), color='red', linestyle='--', label=f'均值={df_clean[target_col].mean():.1f}')
ax2.axvline(df_clean[target_col].median(), color='green', linestyle='--', label=f'中位数={df_clean[target_col].median():.1f}')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 7.3 按小时的平均PM2.5
ax3 = fig.add_subplot(3, 2, 3)
hourly_avg = df_clean.groupby('hour')[target_col].mean()
ax3.bar(hourly_avg.index, hourly_avg.values, color='coral')
ax3.set_xlabel('小时')
ax3.set_ylabel('平均PM2.5 (μg/m³)')
ax3.set_title('PM2.5日变化规律')
ax3.grid(True, alpha=0.3)

# 7.4 按月平均PM2.5
ax4 = fig.add_subplot(3, 2, 4)
monthly_avg = df_clean.groupby('month')[target_col].mean()
ax4.plot(monthly_avg.index, monthly_avg.values, marker='o', color='purple', linewidth=2)
ax4.set_xlabel('月份')
ax4.set_ylabel('平均PM2.5 (μg/m³)')
ax4.set_title('PM2.5季节变化规律')
ax4.grid(True, alpha=0.3)

# 7.5 特征相关性热力图
ax5 = fig.add_subplot(3, 2, 5)
corr_features = [target_col] + available_features[:10]  # 限制特征数量
corr_matrix = df_clean[corr_features].corr()
im = ax5.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
ax5.set_xticks(range(len(corr_features)))
ax5.set_xticklabels(corr_features, rotation=45, ha='right', fontsize=8)
ax5.set_yticks(range(len(corr_features)))
ax5.set_yticklabels(corr_features, fontsize=8)
ax5.set_title('特征相关性热力图')
plt.colorbar(im, ax=ax5)

# 7.6 箱线图（按季节）
ax6 = fig.add_subplot(3, 2, 6)
season_names = {1: '春季', 2: '夏季', 3: '秋季', 4: '冬季'}
df_clean['season_name'] = df_clean['season'].map(season_names)
df_clean.boxplot(column=target_col, by='season_name', ax=ax6)
ax6.set_xlabel('季节')
ax6.set_ylabel('PM2.5 (μg/m³)')
ax6.set_title('PM2.5季节分布')
ax6.grid(True, alpha=0.3)

plt.suptitle('北京PM2.5数据分析报告', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('data/data_analysis_report/data_analysis.png', dpi=150, facecolor='white')
print("  已保存: data/data_analysis_report/data_analysis.png")
plt.show()

# ============================================================
# 8. 保存清洗后的数据
# ============================================================
print("\n" + "=" * 70)
print("[8] 保存清洗后的数据")
print("=" * 70)

# 选择最终的数据列
final_columns = [target_col] + available_features + ['datetime', 'year', 'month', 'day', 'hour']
df_final = df_clean[final_columns].copy()

# 重命名目标列
df_final = df_final.rename(columns={target_col: 'pm25'})

print(f"最终数据形状: {df_final.shape}")
print(f"PM2.5范围: [{df_final['pm25'].min():.1f}, {df_final['pm25'].max():.1f}]")
print(f"PM2.5均值: {df_final['pm25'].mean():.1f}, 标准差: {df_final['pm25'].std():.1f}")

# 保存
df_final.to_csv(OUTPUT_PATH, index=False)
print(f"数据已保存: {OUTPUT_PATH}")

# ============================================================
# 9. 数据统计报告
# ============================================================
print("\n" + "=" * 70)
print("[9] 数据统计报告")
print("=" * 70)

print(f"\n数据集时间范围:")
print(f"  开始: {df_final['datetime'].min()}")
print(f"  结束: {df_final['datetime'].max()}")
print(f"  总时长: {len(df_final)} 小时")

print(f"\nPM2.5统计:")
print(f"  均值: {df_final['pm25'].mean():.2f} μg/m³")
print(f"  标准差: {df_final['pm25'].std():.2f}")
print(f"  最小值: {df_final['pm25'].min():.2f}")
print(f"  25%分位: {df_final['pm25'].quantile(0.25):.2f}")
print(f"  中位数: {df_final['pm25'].median():.2f}")
print(f"  75%分位: {df_final['pm25'].quantile(0.75):.2f}")
print(f"  最大值: {df_final['pm25'].max():.2f}")

# AQI等级分布
def get_aqi_level(pm25):
    if pm25 <= 35: return '优'
    elif pm25 <= 75: return '良'
    elif pm25 <= 115: return '轻度污染'
    elif pm25 <= 150: return '中度污染'
    elif pm25 <= 250: return '重度污染'
    else: return '严重污染'

df_final['aqi_level'] = df_final['pm25'].apply(get_aqi_level)
print(f"\n空气质量等级分布:")
level_counts = df_final['aqi_level'].value_counts()
for level, count in level_counts.items():
    pct = count / len(df_final) * 100
    print(f"  {level}: {count} ({pct:.1f}%)")

print("\n" + "=" * 70)
print("  数据预处理完成！")
print("=" * 70)