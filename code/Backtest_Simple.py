"""
高影响力推文交易策略回测
- 信号：musk_high_impact_tweets > 0
- 入场：事件日收盘价买入
- 出场：持有N个交易日卖出
- 样本外测试：2022年以后
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================================
# 1. 数据加载与预处理
# ============================================================================
df = pd.read_csv('Master_Analysis_Dataset_Daily_v2_FIXED.csv', index_col=0, parse_dates=True)

# 确保价格列存在
price_col = 'Close'
df['Price'] = df[price_col]

# 计算日收益率（用于基准）
df['Return'] = df['Price'].pct_change()

# 提取事件信号日期
event_dates = df[df['musk_high_impact_tweets'] > 0].index.tolist()
event_dates = sorted(list(set(event_dates)))  # 去重
print(f"总事件数: {len(event_dates)}")

# ============================================================================
# 2. 策略回测函数（允许重叠持仓）
# ============================================================================
def backtest_strategy(df, event_dates, hold_days, cost_pct=0.005):
    """
    回测事件驱动策略
    df: DataFrame with 'Price' column
    event_dates: list of event dates
    hold_days: int, 持有交易日数
    cost_pct: 单边交易成本（买入+卖出合计）
    返回 strategy_equity, benchmark_equity, trades_df
    """
    df = df.copy()
    df['Signal'] = 0
    df['Position'] = 0
    df['Trade_Entry'] = 0.0
    

    df['SMA200'] = df['Price'].rolling(200).mean()
    df['Trend_Up'] = df['Price'] > df['SMA200']

    for event in event_dates:
        if event in df.index:
            idx = df.index.get_loc(event)
            # 仅在趋势向上时开仓
            if df['Trend_Up'].iloc[idx]:
                end_idx = min(idx + hold_days, len(df))
                for i in range(idx, end_idx):
                    df.iloc[i, df.columns.get_loc('Position')] = 1
                    if i == idx:
                        df.iloc[i, df.columns.get_loc('Trade_Entry')] = 1
    
    # 计算策略收益率（考虑交易成本）
    df['Strategy_Ret'] = 0.0
    prev_pos = 0
    for i in range(1, len(df)):
        pos = df['Position'].iloc[i]
        price_ret = df['Price'].iloc[i] / df['Price'].iloc[i-1] - 1
        
        if pos == 1:
            df.loc[df.index[i], 'Strategy_Ret'] = price_ret
        else:
            df.loc[df.index[i], 'Strategy_Ret'] = 0.0
        
        # 交易成本：仅在开仓时扣除
        if df['Trade_Entry'].iloc[i] == 1:
            df.loc[df.index[i], 'Strategy_Ret'] -= cost_pct  # 单边成本（假设买入即扣）
    
    # 卖出时不再额外扣成本（成本已在总cost_pct中）
    # 若需更精确，可将成本拆分，此处简化为买入时扣除全部交易成本
    
    # 计算净值曲线
    df['Strategy_Equity'] = (1 + df['Strategy_Ret']).cumprod()
    df['Benchmark_Equity'] = (1 + df['Return'].fillna(0)).cumprod()
    
    # 收集交易记录
    trades = []
    for event in event_dates:
        if event in df.index:
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
                    'Return': ret,
                    'Hold_Days': hold_days
                })
    trades_df = pd.DataFrame(trades)
    return df, trades_df

# ============================================================================
# 3. 训练期：寻找最优持有期 N（2017-2021）
# ============================================================================
train_end = pd.Timestamp('2021-12-31', tz='UTC')
df_train = df.loc[:train_end].copy()
event_dates_train = [d for d in event_dates if d <= train_end]

hold_range = range(1, 21)
train_results = []
for hold in hold_range:
    df_res, trades = backtest_strategy(df_train, event_dates_train, hold, cost_pct=0.005)
    final_equity = df_res['Strategy_Equity'].iloc[-1]
    total_return = final_equity - 1
    num_trades = len(trades)
    win_rate = (trades['Return'] > 0).mean() if num_trades > 0 else 0
    avg_return = trades['Return'].mean() if num_trades > 0 else 0
    
    # 简单年化（假设约4年）
    years = (df_train.index[-1] - df_train.index[0]).days / 365.25
    ann_return = (final_equity) ** (1/years) - 1
    
    train_results.append({
        'Hold_Days': hold,
        'Total_Return': total_return,
        'Ann_Return': ann_return,
        'Num_Trades': num_trades,
        'Win_Rate': win_rate,
        'Avg_Return': avg_return
    })

train_res_df = pd.DataFrame(train_results)
best_idx = train_res_df['Total_Return'].idxmax()
best_hold = train_res_df.loc[best_idx, 'Hold_Days']
print("\n训练期最优参数（总收益率最高）:")
print(train_res_df.round(4))
print(f"\n选择持有期: {best_hold} 天")

# ============================================================================
# 4. 样本外测试：2022年以后
# ============================================================================
test_start = pd.Timestamp('2022-01-01', tz='UTC')
df_test = df.loc[test_start:].copy()
event_dates_test = [d for d in event_dates if d >= test_start]

print(f"\n样本外事件数: {len(event_dates_test)}")
if len(event_dates_test) == 0:
    print("警告：样本外无事件，无法测试。")
else:
    df_test_res, trades_test = backtest_strategy(df_test, event_dates_test, best_hold, cost_pct=0.005)
    
    # ============================================================================
    # 5. 绩效评估
    # ============================================================================
    def compute_performance(df_res, trades, label='Test'):
        equity = df_res['Strategy_Equity']
        benchmark = df_res['Benchmark_Equity']
        rets = df_res['Strategy_Ret'].replace(0, np.nan).dropna()
        
        total_ret = equity.iloc[-1] - 1
        years = (equity.index[-1] - equity.index[0]).days / 365.25
        ann_ret = (equity.iloc[-1]) ** (1/years) - 1
        vol = rets.std() * np.sqrt(365)  # 年化波动（假设365天）
        sharpe = ann_ret / vol if vol > 0 else 0
        
        # 最大回撤
        cummax = equity.expanding().max()
        drawdown = (equity - cummax) / cummax
        max_dd = drawdown.min()
        
        # 基准
        bench_total = benchmark.iloc[-1] - 1
        bench_ann = (benchmark.iloc[-1]) ** (1/years) - 1
        
        # 交易统计
        win_rate = (trades['Return'] > 0).mean() if len(trades) > 0 else 0
        avg_ret = trades['Return'].mean() if len(trades) > 0 else 0
        profit_factor = trades[trades['Return'] > 0]['Return'].sum() / abs(trades[trades['Return'] < 0]['Return'].sum()) if len(trades[trades['Return'] < 0]) > 0 else np.inf
        
        print(f"\n========== {label} 绩效报告 ==========")
        print(f"测试期: {equity.index[0].date()} 至 {equity.index[-1].date()}")
        print(f"交易次数: {len(trades)}")
        print(f"胜率: {win_rate:.2%}")
        print(f"平均单笔收益率: {avg_ret:.2%}")
        print(f"盈亏比 (Profit Factor): {profit_factor:.2f}")
        print(f"总收益率: {total_ret:.2%}")
        print(f"年化收益率: {ann_ret:.2%}")
        print(f"年化波动率: {vol:.2%}")
        print(f"夏普比率: {sharpe:.2f}")
        print(f"最大回撤: {max_dd:.2%}")
        print(f"基准买入持有总收益率: {bench_total:.2%}")
        print(f"基准年化收益率: {bench_ann:.2%}")
        
        return {
            'Total_Return': total_ret,
            'Ann_Return': ann_ret,
            'Volatility': vol,
            'Sharpe': sharpe,
            'Max_DD': max_dd,
            'Win_Rate': win_rate,
            'Avg_Trade_Ret': avg_ret,
            'Profit_Factor': profit_factor,
            'Bench_Total': bench_total
        }
    
    perf = compute_performance(df_test_res, trades_test, '样本外 (2022-2024)')
    
    # ============================================================================
    # 6. 可视化
    # ============================================================================
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 净值曲线
    axes[0].plot(df_test_res.index, df_test_res['Strategy_Equity'], label='Strategy (Event-Driven)', color='blue')
    axes[0].plot(df_test_res.index, df_test_res['Benchmark_Equity'], label='Benchmark (Buy & Hold)', color='gray', alpha=0.7)
    axes[0].set_ylabel('Equity')
    axes[0].set_title(f'Strategy vs Benchmark (Hold {best_hold} days) - Out-of-Sample')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 回撤
    cummax = df_test_res['Strategy_Equity'].expanding().max()
    drawdown = (df_test_res['Strategy_Equity'] - cummax) / cummax
    axes[1].fill_between(df_test_res.index, 0, drawdown, color='red', alpha=0.3)
    axes[1].set_ylabel('Drawdown')
    axes[1].set_title('Strategy Drawdown')
    axes[1].grid(True, alpha=0.3)
    
    # 交易点标记
    axes[2].plot(df_test_res.index, df_test_res['Price'], color='black', alpha=0.5, label='DOGE Price')
    for trade in trades_test.itertuples():
        axes[2].axvline(trade.Entry_Date, color='green', linestyle='--', alpha=0.5, linewidth=0.8)
    axes[2].set_ylabel('Price (USD)')
    axes[2].set_title('DOGE Price with Trade Entry Points')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('strategy_backtest_out_of_sample.png', dpi=150)
    plt.show()
    
    # 打印交易明细
    print("\n样本外交易明细:")
    print(trades_test.round(4))