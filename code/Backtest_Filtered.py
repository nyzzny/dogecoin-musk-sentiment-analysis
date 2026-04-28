"""
改进版高影响力推文策略回测：增加SMA200趋势过滤
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 数据加载
df = pd.read_csv('Master_Analysis_Dataset_Daily_v2_FIXED.csv', index_col=0, parse_dates=True)
df['Price'] = df['Close']

# 计算SMA200
df['SMA200'] = df['Price'].rolling(200).mean()
df['Trend_Up'] = df['Price'] > df['SMA200']

# 事件日期
event_dates = df[df['musk_high_impact_tweets'] > 0].index.tolist()
event_dates = sorted(list(set(event_dates)))
print(f"总事件数: {len(event_dates)}")

def backtest_with_filter(df, event_dates, hold_days, cost_pct=0.005):
    df = df.copy()
    df['Position'] = 0
    df['Trade_Entry'] = 0
    
    for event in event_dates:
        if event in df.index:
            idx = df.index.get_loc(event)
            if df['Trend_Up'].iloc[idx]:  # 仅趋势向上时开仓
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
        if event in df.index:
            idx = df.index.get_loc(event)
            if df['Trend_Up'].iloc[idx]:
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

# 训练期寻找最优持有期（仍基于训练期，但加入过滤）
train_end = pd.Timestamp('2021-12-31', tz='UTC')
df_train = df.loc[:train_end].copy()
event_train = [d for d in event_dates if d <= train_end]

hold_range = range(1, 21)
train_results = []
for hold in hold_range:
    _, trades = backtest_with_filter(df_train, event_train, hold)
    if len(trades) > 0:
        final_eq = (1 + trades['Return']).prod()
    else:
        final_eq = 1.0
    train_results.append({
        'Hold_Days': hold,
        'Total_Return': final_eq - 1,
        'Num_Trades': len(trades),
        'Win_Rate': (trades['Return'] > 0).mean() if len(trades) > 0 else 0
    })

train_res_df = pd.DataFrame(train_results)
best_idx = train_res_df['Total_Return'].idxmax()
best_hold = train_res_df.loc[best_idx, 'Hold_Days']
print(f"\n过滤后训练期最优持有期: {best_hold} 天")

# 样本外测试
test_start = pd.Timestamp('2022-01-01', tz='UTC')
df_test = df.loc[test_start:].copy()
event_test = [d for d in event_dates if d >= test_start]

df_test_res, trades_test = backtest_with_filter(df_test, event_test, best_hold)
# 计算最大回撤
cummax = df_test_res['Strategy_Equity'].cummax()
drawdown = (df_test_res['Strategy_Equity'] - cummax) / cummax
max_dd = drawdown.min()
print(f"最大回撤: {max_dd:.2%}")

print(f"\n过滤后样本外交易次数: {len(trades_test)}")
if len(trades_test) > 0:
    total_ret = df_test_res['Strategy_Equity'].iloc[-1] - 1
    bench_ret = df_test_res['Benchmark_Equity'].iloc[-1] - 1
    print(f"策略总收益率: {total_ret:.2%}")
    print(f"基准总收益率: {bench_ret:.2%}")
    print(f"胜率: {(trades_test['Return'] > 0).mean():.2%}")
    print(f"平均单笔收益: {trades_test['Return'].mean():.2%}")
else:
    print("无符合条件的交易。")

# 可视化
plt.figure(figsize=(12,6))
plt.plot(df_test_res.index, df_test_res['Strategy_Equity'], label='Filtered Strategy')
plt.plot(df_test_res.index, df_test_res['Benchmark_Equity'], label='Benchmark')
plt.legend()
plt.title('Strategy with SMA200 Filter (Out-of-Sample)')
plt.grid(alpha=0.3)
plt.savefig('filtered_strategy.png', dpi=150)
plt.show()