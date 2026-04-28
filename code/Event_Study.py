"""
升级版事件研究：基于市场模型的异常收益率分析（时区修复版）
- 控制比特币收益率
- 使用区块自助法计算CAR置信区间
- 剔除重叠事件
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ============================================================================
# 1. 加载数据
# ============================================================================
df = pd.read_csv('Master_Analysis_Dataset_Daily_v2_FIXED.csv', index_col=0, parse_dates=True)

# 下载比特币数据作为市场代理
btc = yf.download('BTC-USD', start=df.index.min(), end=df.index.max(), auto_adjust=False)['Close']
btc_ret = btc.pct_change().dropna()

# 【关键修复】统一时区：BTC索引原为无时区，需本地化为UTC以匹配df.index
btc_ret.index = btc_ret.index.tz_localize('UTC')

# 对齐索引并前向填充
btc_ret = btc_ret.reindex(df.index, method='ffill')
df['btc_return'] = btc_ret

# ============================================================================
# 2. 市场模型估计函数
# ============================================================================
def estimate_market_model(df, event_date, est_window=(-200, -11)):
    """为每个事件估计市场模型参数"""
    idx = df.index.get_loc(event_date)
    start = idx + est_window[0]
    end = idx + est_window[1]
    if start < 0 or end < 0:
        return None, None
    est_df = df.iloc[start:end]
    if len(est_df) < 50:
        return None, None
    X = est_df['btc_return'].values
    X = np.column_stack([np.ones(len(X)), X])
    y = est_df['daily_return'].values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta[0], beta[1]  # alpha, beta

# ============================================================================
# 3. 获取事件并去重叠
# ============================================================================
high_impact_dates = df[df['musk_high_impact_tweets'] > 0].index.tolist()
window = (-5, 15)

clean_events = []
for date in sorted(high_impact_dates):
    if not clean_events:
        clean_events.append(date)
    else:
        last_date = clean_events[-1]
        if (date - last_date).days > (window[1] - window[0]):
            clean_events.append(date)

print(f"原始事件数: {len(high_impact_dates)}，去重叠后: {len(clean_events)}")

# ============================================================================
# 4. 计算异常收益率（AR）
# ============================================================================
event_ars = []
for event in clean_events:
    alpha, beta = estimate_market_model(df, event)
    if alpha is None:
        continue

    idx = df.index.get_loc(event)
    start = idx + window[0]
    end = idx + window[1] + 1
    if start < 0 or end > len(df):
        continue

    event_df = df.iloc[start:end].copy()
    expected_ret = alpha + beta * event_df['btc_return'].values
    event_df['AR'] = event_df['daily_return'].values - expected_ret
    event_ars.append(event_df['AR'].values)

if not event_ars:
    raise ValueError("无有效事件，请放宽条件。")

event_ars = np.array(event_ars)
avg_ar = event_ars.mean(axis=0)
car = np.cumsum(avg_ar)

# ============================================================================
# 5. 区块自助法计算CAR置信区间
# ============================================================================
n_events, n_days = event_ars.shape
n_bootstrap = 10000
bootstrap_cars = np.zeros((n_bootstrap, n_days))

np.random.seed(42)
for i in range(n_bootstrap):
    idx = np.random.choice(n_events, size=n_events, replace=True)
    sample_ars = event_ars[idx, :]
    sample_avg_ar = sample_ars.mean(axis=0)
    bootstrap_cars[i, :] = np.cumsum(sample_avg_ar)

ci_lower = np.percentile(bootstrap_cars, 2.5, axis=0)
ci_upper = np.percentile(bootstrap_cars, 97.5, axis=0)

# ============================================================================
# 6. 可视化
# ============================================================================
days = np.arange(window[0], window[1] + 1)
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# 平均异常收益率（AR）
axes[0].bar(days, avg_ar * 100, color='steelblue', alpha=0.7)
axes[0].axhline(0, color='gray', linestyle='-', linewidth=0.5)
axes[0].axvline(0, color='black', linestyle='--', linewidth=1, label='Event Day (t=0)')
axes[0].set_ylabel('Avg AR (%)')
axes[0].set_title('Average Abnormal Return (Market Model Adjusted)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 累计异常收益率（CAR）及自助法置信区间
axes[1].plot(days, car * 100, marker='o', color='green', linewidth=2, label='CAR')
axes[1].fill_between(days, ci_lower * 100, ci_upper * 100,
                     color='green', alpha=0.2, label='95% CI (Bootstrap)')
axes[1].axhline(0, color='gray', linestyle='-', linewidth=0.5)
axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
axes[1].set_xlabel('Days Relative to Event')
axes[1].set_ylabel('Cumulative AR (%)')
axes[1].set_title('Cumulative Abnormal Return (CAR) with Bootstrap CI')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('event_study_market_model_bootstrap.png', dpi=150)
plt.show()

# ============================================================================
# 7. 统计显著性输出
# ============================================================================
t0_idx = abs(window[0])
car_t0 = car[t0_idx]
p_val_t0 = (np.sum(bootstrap_cars[:, t0_idx] <= 0) + 1) / (n_bootstrap + 1)
print(f"事件日(t=0) CAR: {car_t0*100:.2f}%")
print(f"自助法单侧p值 (CAR<=0): {p_val_t0:.4f}")