"""
复合策略：Musk高影响力推文 + Reddit热度异常 + BTC强势
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 1. 数据加载与特征工程
# ============================================================================
df = pd.read_csv('Master_Analysis_Dataset_Daily_v2_FIXED.csv', index_col=0, parse_dates=True)
df['Price'] = df['Close']

# 计算 Reddit 发帖量滚动 Z-Score（30天窗口）
df['reddit_roll_mean'] = df['reddit_post_count'].rolling(30).mean()
df['reddit_roll_std'] = df['reddit_post_count'].rolling(30).std()
df['reddit_zscore'] = (df['reddit_post_count'] - df['reddit_roll_mean']) / df['reddit_roll_std']
df['reddit_zscore'] = df['reddit_zscore'].fillna(0)

# 比特币收益率（假设已存在于数据集中，若无则重新下载）
if 'btc_return' not in df.columns:
    import yfinance as yf
    btc = yf.download('BTC-USD', start=df.index.min(), end=df.index.max(), auto_adjust=False)['Close']
    btc_ret = btc.pct_change().dropna()
    btc_ret.index = btc_ret.index.tz_localize('UTC')
    btc_ret = btc_ret.reindex(df.index, method='ffill')
    df['btc_return'] = btc_ret

# 定义复合信号
df['musk_signal'] = (df['musk_high_impact_tweets'] > 0).astype(int)
df['reddit_signal'] = (df['reddit_zscore'] > 1.5).astype(int)
df['btc_signal'] = (df['btc_return'] > 0).astype(int)
df['composite_signal'] = (
    (df['musk_signal'] == 1) & 
    (df['reddit_signal'] == 1) & 
    (df['btc_signal'] == 1)
).astype(int)

# 事件日期（复合信号为真的日期）
event_dates = df[df['composite_signal'] == 1].index.tolist()
print(f"原始高影响力推文事件数: {df['musk_signal'].sum()}")
print(f"复合信号事件数: {len(event_dates)}")

# ============================================================================
# 2. 策略回测函数（允许重叠持仓）
# ============================================================================
def backtest_composite(df, event_dates, hold_days=5, cost_pct=0.005):
    df = df.copy()
    df['Position'] = 0
    df['Trade_Entry'] = 0
    
    for event in event_dates:
        idx = df.index.get_loc(event)
        end_idx = min(idx + hold_days, len(df))
        for i in range(idx, end_idx):
            df.iloc[i, df.columns.get_loc('Position')] = 1
            if i == idx:
                df.iloc[i, df.columns.get_loc('Trade_Entry')] = 1
    
    df['Strategy_Ret'] = 0.0
    for i in range(1, len(df)):
        if df['Position'].iloc[i] == 1:
            price_ret = df['Price'].iloc[i] / df['Price'].iloc[i-1] - 1
            df.loc[df.index[i], 'Strategy_Ret'] = price_ret
            if df['Trade_Entry'].iloc[i] == 1:
                df.loc[df.index[i], 'Strategy_Ret'] -= cost_pct
    
    df['Strategy_Equity'] = (1 + df['Strategy_Ret']).cumprod()
    df['Benchmark_Equity'] = (1 + df['Price'].pct_change().fillna(0)).cumprod()
    
    trades = []
    for event in event_dates:
        idx = df.index.get_loc(event)
        if idx < len(df) - 1:
            entry_price = df['Price'].iloc[idx]
            exit_idx = min(idx + hold_days, len(df) - 1)
            exit_price = df['Price'].iloc[exit_idx]
            ret = (exit_price / entry_price) - 1 - cost_pct
            trades.append({
                'Entry_Date': event,
                'Entry_Price': entry_price,
                'Exit_Date': df.index[exit_idx],
                'Exit_Price': exit_price,
                'Return': ret
            })
    return df, pd.DataFrame(trades)

# ============================================================================
# 3. 训练期参数优化（选择最优持有期）
# ============================================================================
train_end = pd.Timestamp('2021-12-31', tz='UTC')
df_train = df.loc[:train_end].copy()
event_train = [d for d in event_dates if d <= train_end]

print(f"\n训练期复合信号事件数: {len(event_train)}")

hold_range = range(1, 16)
train_results = []
for hold in hold_range:
    _, trades = backtest_composite(df_train, event_train, hold_days=hold)
    if len(trades) > 0:
        final_eq = (1 + trades['Return']).prod()
    else:
        final_eq = 1.0
    train_results.append({
        'Hold_Days': hold,
        'Total_Return': final_eq - 1,
        'Num_Trades': len(trades),
        'Win_Rate': (trades['Return'] > 0).mean() if len(trades) > 0 else 0,
        'Avg_Return': trades['Return'].mean() if len(trades) > 0 else 0
    })

train_res_df = pd.DataFrame(train_results)
best_idx = train_res_df['Total_Return'].idxmax()
best_hold = train_res_df.loc[best_idx, 'Hold_Days']
print("\n训练期最优参数:")
print(train_res_df.round(4))
print(f"\n选择持有期: {best_hold} 天")

# ============================================================================
# 4. 样本外测试
# ============================================================================
test_start = pd.Timestamp('2022-01-01', tz='UTC')
df_test = df.loc[test_start:].copy()
event_test = [d for d in event_dates if d >= test_start]

print(f"\n样本外复合信号事件数: {len(event_test)}")
df_test_res, trades_test = backtest_composite(df_test, event_test, hold_days=best_hold)

# 绩效计算
def compute_performance(df_res, trades, label='Test'):
    equity = df_res['Strategy_Equity']
    benchmark = df_res['Benchmark_Equity']
    rets = df_res['Strategy_Ret'].replace(0, np.nan).dropna()
    
    total_ret = equity.iloc[-1] - 1
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    ann_ret = (equity.iloc[-1]) ** (1/years) - 1 if years > 0 else 0
    vol = rets.std() * np.sqrt(365) if len(rets) > 1 else 0
    sharpe = ann_ret / vol if vol > 0 else 0
    
    cummax = equity.expanding().max()
    drawdown = (equity - cummax) / cummax
    max_dd = drawdown.min()
    
    bench_total = benchmark.iloc[-1] - 1
    bench_ann = (benchmark.iloc[-1]) ** (1/years) - 1 if years > 0 else 0
    
    win_rate = (trades['Return'] > 0).mean() if len(trades) > 0 else 0
    avg_ret = trades['Return'].mean() if len(trades) > 0 else 0
    profit_factor = (trades[trades['Return'] > 0]['Return'].sum() / 
                     abs(trades[trades['Return'] < 0]['Return'].sum())) if len(trades[trades['Return'] < 0]) > 0 else np.inf
    
    print(f"\n========== {label} 绩效报告 ==========")
    print(f"测试期: {equity.index[0].date()} 至 {equity.index[-1].date()}")
    print(f"交易次数: {len(trades)}")
    print(f"胜率: {win_rate:.2%}")
    print(f"平均单笔收益率: {avg_ret:.2%}")
    print(f"盈亏比: {profit_factor:.2f}")
    print(f"总收益率: {total_ret:.2%}")
    print(f"年化收益率: {ann_ret:.2%}")
    print(f"年化波动率: {vol:.2%}")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"最大回撤: {max_dd:.2%}")
    print(f"基准买入持有总收益率: {bench_total:.2%}")
    print(f"基准年化收益率: {bench_ann:.2%}")

if len(trades_test) > 0:
    compute_performance(df_test_res, trades_test, '样本外 (2022-2024)')
    print("\n样本外交易明细:")
    print(trades_test.round(4))
else:
    print("样本外无符合条件的交易。")

# ============================================================================
# 5. 可视化
# ============================================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

axes[0].plot(df_test_res.index, df_test_res['Strategy_Equity'], label='Composite Strategy', color='blue')
axes[0].plot(df_test_res.index, df_test_res['Benchmark_Equity'], label='Benchmark (Buy & Hold)', color='gray', alpha=0.7)
axes[0].set_ylabel('Equity')
axes[0].set_title(f'Composite Strategy (Hold {best_hold}d) - Out-of-Sample')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

cummax = df_test_res['Strategy_Equity'].expanding().max()
drawdown = (df_test_res['Strategy_Equity'] - cummax) / cummax
axes[1].fill_between(df_test_res.index, 0, drawdown, color='red', alpha=0.3)
axes[1].set_ylabel('Drawdown')
axes[1].set_title('Strategy Drawdown')
axes[1].grid(True, alpha=0.3)

axes[2].plot(df_test_res.index, df_test_res['Price'], color='black', alpha=0.5, label='DOGE Price')
for trade in trades_test.itertuples():
    axes[2].axvline(trade.Entry_Date, color='green', linestyle='--', alpha=0.5, linewidth=0.8)
axes[2].set_ylabel('Price (USD)')
axes[2].set_title('DOGE Price with Trade Entry Points')
axes[2].set_yscale('log')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('composite_strategy_out_of_sample.png', dpi=150)
plt.show()