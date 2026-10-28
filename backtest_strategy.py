import pandas as pd
import numpy as np

df = pd.read_csv('ibm_daily.csv', parse_dates=['date']).sort_values('date').reset_index(drop=True)

FAST, SLOW = 5, 20
df['sma_fast'] = df['close'].rolling(FAST).mean()
df['sma_slow'] = df['close'].rolling(SLOW).mean()

# Signal: 1 = long, 0 = flat. Long when fast SMA > slow SMA.
df['raw_signal'] = np.where(df['sma_fast'] > df['sma_slow'], 1, 0)
# Trade on next day's open/close to avoid lookahead bias -> shift signal by 1
df['position'] = df['raw_signal'].shift(1).fillna(0)

df['ret'] = df['close'].pct_change()
df['strat_ret'] = df['position'] * df['ret']

# drop the warm-up NaNs (need SLOW=20 days of history before signal is valid)
df_valid = df.dropna(subset=['sma_slow']).reset_index(drop=True)

def perf_stats(returns, label):
    returns = returns.dropna()
    n = len(returns)
    total_ret = (1 + returns).prod() - 1
    ann_factor = 252
    ann_ret = (1 + total_ret) ** (ann_factor / n) - 1 if n > 0 else np.nan
    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = (returns.mean() * ann_factor) / ann_vol if ann_vol > 0 else np.nan
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    max_dd = drawdown.min()
    return {
        'label': label, 'n_days': n, 'total_return': total_ret, 'ann_return': ann_ret,
        'sharpe': sharpe, 'max_drawdown': max_dd
    }

results = []
results.append(perf_stats(df_valid['strat_ret'], 'Strategy - Full Period'))
results.append(perf_stats(df_valid['ret'], 'Buy & Hold - Full Period'))

res_df = pd.DataFrame(results)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')
print(res_df.to_string(index=False))

# Trade count & correlation
df_valid['trade'] = df_valid['position'].diff().fillna(0) != 0
n_trades = df_valid['trade'].sum()
print(f"\nNumber of position changes (entries+exits): {int(n_trades)}")

corr = df_valid[['strat_ret','ret']].dropna().corr().iloc[0,1]
print(f"Correlation of strategy daily returns with buy-and-hold (market direction proxy): {corr:.3f}")

print(f"\nDate range used for signals: {df_valid['date'].min().date()} to {df_valid['date'].max().date()} ({len(df_valid)} days)")

df_valid.to_csv('backtest_detail.csv', index=False)
res_df.to_csv('backtest_results.csv', index=False)
