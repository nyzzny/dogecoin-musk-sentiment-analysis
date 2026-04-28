"""
Dogecoin Sentiment Analysis - Exploratory Data Analysis (EDA)
使用修复后的数据集：Master_Analysis_Dataset_Daily_v2_FIXED.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import stats

# 设置中文显示（可选）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 1. 数据加载与预处理
# ============================================================================
df = pd.read_csv('Master_Analysis_Dataset_Daily_v2_FIXED.csv', index_col=0, parse_dates=True)

# 确保数据类型正确
df['musk_high_impact_tweets'] = df['musk_high_impact_tweets'].astype(int)
df['reddit_post_count'] = df['reddit_post_count'].astype(int)

# 创建收益率百分比形式（便于解释）
df['return_pct'] = df['daily_return'] * 100

print("数据集基本信息：")
print(df.info())
print("\n描述性统计：")
print(df.describe())

# ============================================================================
# 2. 时间序列可视化
# ============================================================================
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# 价格
axes[0].plot(df.index, df['Close'], color='black', linewidth=0.8)
axes[0].set_ylabel('Close Price (USD)')
axes[0].set_title('Dogecoin Price (Log Scale)')
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)

# 收益率
axes[1].plot(df.index, df['return_pct'], color='blue', linewidth=0.5, alpha=0.7)
axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.5)
axes[1].set_ylabel('Daily Return (%)')
axes[1].set_title('Daily Return')
axes[1].grid(True, alpha=0.3)

# Musk 推文数量（条形图）和高影响力事件标记
axes[2].bar(df.index, df['musk_tweet_count'], width=1.0, color='lightblue', label='All Musk Tweets')
high_impact = df[df['musk_high_impact_tweets'] > 0]
axes[2].scatter(high_impact.index, high_impact['musk_tweet_count'], 
                color='red', s=50, marker='v', label='High-Impact Tweets')
axes[2].set_ylabel('Tweet Count')
axes[2].set_title('Elon Musk Daily Tweet Count (with high-impact markers)')
axes[2].legend(loc='upper left')
axes[2].grid(True, alpha=0.3)

# Reddit 帖子数量
axes[3].bar(df.index, df['reddit_post_count'], width=1.0, color='orange', alpha=0.7)
axes[3].set_ylabel('Post Count')
axes[3].set_title('r/dogecoin Daily Post Count')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('time_series_overview.png', dpi=150)
plt.show()

# ============================================================================
# 3. 相关性分析
# ============================================================================
# 选取用于相关性分析的列
corr_cols = ['daily_return', 'musk_tweet_count', 'musk_avg_sentiment', 
             'musk_positive_count', 'musk_negative_count', 
             'reddit_post_count', 'reddit_avg_sentiment']
corr_df = df[corr_cols].copy()
corr_df.rename(columns={'daily_return': 'Return', 
                        'musk_tweet_count': 'Musk_Tweets',
                        'musk_avg_sentiment': 'Musk_Sentiment',
                        'reddit_post_count': 'Reddit_Posts',
                        'reddit_avg_sentiment': 'Reddit_Sentiment'}, inplace=True)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix of Key Variables')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()

# 注意：这里计算的是同期相关系数，对于时间序列需谨慎解读

# ============================================================================
# 4. 事件研究法 (Event Study) - 高影响力推文
# ============================================================================
def event_study(data, event_dates, window=(-5, 10)):
    """
    简单事件研究：计算事件窗口内的平均累计收益率。
    data: DataFrame with 'daily_return' column
    event_dates: list of event dates (as index)
    window: tuple (pre_days, post_days)
    """
    returns = []
    for event in event_dates:
        try:
            loc = data.index.get_loc(event)
            start = loc + window[0]
            end = loc + window[1] + 1
            if start >= 0 and end <= len(data):
                ret_win = data['daily_return'].iloc[start:end].values
                # 对齐窗口长度（如果边缘不足则跳过）
                if len(ret_win) == (window[1] - window[0] + 1):
                    returns.append(ret_win)
        except:
            continue
    
    if not returns:
        return None, None
    
    returns = np.array(returns)
    avg_returns = returns.mean(axis=0)
    # 累计收益率
    car = np.cumprod(1 + avg_returns) - 1
    
    # 计算置信区间 (bootstrap 或 t-test)
    # 简单计算标准误
    se = returns.std(axis=0) / np.sqrt(len(returns))
    ci_lower = avg_returns - 1.96 * se
    ci_upper = avg_returns + 1.96 * se
    
    return car, (avg_returns, ci_lower, ci_upper)

# 获取高影响力事件日期
high_impact_dates = df[df['musk_high_impact_tweets'] > 0].index.tolist()
print(f"\n高影响力推文事件数量: {len(high_impact_dates)}")

if high_impact_dates:
    car, stats_tuple = event_study(df, high_impact_dates, window=(-5, 15))
    if car is not None:
        avg_ret, ci_low, ci_up = stats_tuple
        days = np.arange(-5, 16)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        # 平均收益率
        axes[0].bar(days, avg_ret * 100, color='steelblue', alpha=0.7)
        axes[0].plot(days, ci_low * 100, 'r--', linewidth=1, label='95% CI')
        axes[0].plot(days, ci_up * 100, 'r--', linewidth=1)
        axes[0].axhline(0, color='gray', linestyle='-', linewidth=0.5)
        axes[0].axvline(0, color='black', linestyle='--', linewidth=1, label='Event Day')
        axes[0].set_ylabel('Avg Daily Return (%)')
        axes[0].set_title('Average Return Around High-Impact Musk Tweets')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 累计异常收益率 (CAR)
        axes[1].plot(days, car * 100, marker='o', color='green', linewidth=2)
        axes[1].axhline(0, color='gray', linestyle='-', linewidth=0.5)
        axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
        axes[1].set_xlabel('Days Relative to Event')
        axes[1].set_ylabel('Cumulative Return (%)')
        axes[1].set_title('Cumulative Average Return (CAR)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('event_study_high_impact.png', dpi=150)
        plt.show()

# ============================================================================
# 5. 格兰杰因果检验 (简化版)
# ============================================================================
def granger_test(data, maxlag=5, significance=0.05):
    """对收益率和情绪变量进行格兰杰因果检验"""
    # 只使用完整无NaN的行
    test_df = data[['daily_return', 'musk_avg_sentiment', 'reddit_avg_sentiment']].dropna()
    results = {}
    for col in ['musk_avg_sentiment', 'reddit_avg_sentiment']:
        # 测试 情绪 -> 收益率
        gc_res = grangercausalitytests(test_df[['daily_return', col]], maxlag=maxlag, verbose=False)
        p_values = [gc_res[i+1][0]['ssr_chi2test'][1] for i in range(maxlag)]
        results[f'{col} -> Return'] = p_values
        
        # 测试 收益率 -> 情绪
        gc_res_rev = grangercausalitytests(test_df[[col, 'daily_return']], maxlag=maxlag, verbose=False)
        p_values_rev = [gc_res_rev[i+1][0]['ssr_chi2test'][1] for i in range(maxlag)]
        results[f'Return -> {col}'] = p_values_rev
    
    return results

print("\n格兰杰因果检验 (滞后1-5天) p值:")
granger_results = granger_test(df, maxlag=5)
for direction, pvals in granger_results.items():
    print(f"\n{direction}:")
    for lag, p in enumerate(pvals, 1):
        sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
        print(f"  Lag {lag}: p={p:.4f} {sig}")

print("\n注: *** p<0.01, ** p<0.05, * p<0.1")

# ============================================================================
# 6. 极端收益日分析
# ============================================================================
# 定义极端收益：超过2个标准差
ret_std = df['daily_return'].std()
ret_mean = df['daily_return'].mean()
extreme_up = df['daily_return'] > (ret_mean + 2 * ret_std)
extreme_down = df['daily_return'] < (ret_mean - 2 * ret_std)

print(f"\n极端正收益日数量: {extreme_up.sum()}")
print(f"极端负收益日数量: {extreme_down.sum()}")

# 检查这些极端日在高影响力推文前后的分布
extreme_dates = df[extreme_up | extreme_down].index
high_impact_set = set(high_impact_dates)
near_high_impact = []
for ed in extreme_dates:
    # 检查前后3天是否有高影响力推文
    surrounding = pd.date_range(ed - pd.Timedelta(days=3), ed + pd.Timedelta(days=3))
    if any(d in high_impact_set for d in surrounding):
        near_high_impact.append(ed)

print(f"其中发生在高影响力推文前后3天内的极端收益日: {len(near_high_impact)}")

print("\n分析完成。请查看生成的图片文件。")

# 在事件研究函数末尾添加
from scipy.stats import ttest_1samp

# 提取事件日 (t=0) 的收益率样本
event_day_returns = []
for event in high_impact_dates:
    loc = df.index.get_loc(event)
    if loc < len(df):
        event_day_returns.append(df['daily_return'].iloc[loc])

event_day_returns = np.array(event_day_returns)
t_stat, p_val = ttest_1samp(event_day_returns, 0)
print(f"\n事件日平均收益率: {event_day_returns.mean()*100:.2f}%")
print(f"单样本 t 检验 p 值: {p_val:.4f}")