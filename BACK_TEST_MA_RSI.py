# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:08:09 2026

@author: kkeramati
"""

# ============================================================
#  BTC/USD — MA Crossover + RSI Filter Backtest
#  Improved version with RSI filter
#  Requirements: pip install yfinance pandas numpy matplotlib ta
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import ta

# ─────────────────────────────────────────
#  1. Fetch data from Yahoo Finance
# ─────────────────────────────────────────
print("Fetching data from Yahoo Finance...")

df = yf.download("BTC-USD", period="2y", interval="1d", auto_adjust=True)
df = df[['Close']].copy()
df.columns = ['close']
df.dropna(inplace=True)

print(f"Total days: {len(df)}")
print(f"From {df.index[0].date()} to {df.index[-1].date()}")
print("-" * 40)

# ─────────────────────────────────────────
#  2. Calculate indicators
# ─────────────────────────────────────────
MA_SHORT   = 20    # Short MA period
MA_LONG    = 50    # Long MA period
RSI_PERIOD = 14    # RSI period
RSI_BUY    = 55    # Only buy when RSI is below this (not overbought)
RSI_SELL   = 45    # Only sell when RSI is above this (not oversold)

df['MA_short'] = df['close'].rolling(window=MA_SHORT).mean()
df['MA_long']  = df['close'].rolling(window=MA_LONG).mean()
df['RSI']      = ta.momentum.RSIIndicator(df['close'], window=RSI_PERIOD).rsi()

# ─────────────────────────────────────────
#  3. Generate signals
# ─────────────────────────────────────────

# Without RSI filter (previous version)
df['signal_old'] = 0
df.loc[df['MA_short'] > df['MA_long'], 'signal_old'] = 1
df.loc[df['MA_short'] < df['MA_long'], 'signal_old'] = -1
df['position_old'] = df['signal_old'].diff()

# With RSI filter (new version)
# Buy:  MA crossover up   AND RSI < RSI_BUY  (market not overbought)
# Sell: MA crossover down AND RSI > RSI_SELL (market not oversold)
df['signal_new'] = 0
df.loc[
    (df['MA_short'] > df['MA_long']) & (df['RSI'] < RSI_BUY),
    'signal_new'
] = 1
df.loc[
    (df['MA_short'] < df['MA_long']) & (df['RSI'] > RSI_SELL),
    'signal_new'
] = -1
df['position_new'] = df['signal_new'].diff()

# ─────────────────────────────────────────
#  4. Backtest function
# ─────────────────────────────────────────
def run_backtest(df, position_col, initial_capital=1000):
    capital          = initial_capital
    btc              = 0.0
    portfolio_values = []
    trades           = []

    for date, row in df.iterrows():
        price = float(row['close'])
        pos   = float(row[position_col])

        # Buy signal
        if pos == 2 and capital > 0:
            btc = capital / price
            capital = 0
            trades.append({'date': date, 'type': 'BUY', 'price': price})

        # Sell signal
        elif pos == -2 and btc > 0:
            capital = btc * price
            btc = 0
            trades.append({'date': date, 'type': 'SELL', 'price': price})

        portfolio_values.append(capital + btc * price)

    # Calculate metrics
    final_value  = portfolio_values[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100

    port_series  = pd.Series(portfolio_values, index=df.index)
    rolling_max  = port_series.cummax()
    drawdown     = (port_series - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()

    daily_ret = port_series.pct_change().dropna()
    sharpe    = (daily_ret.mean() / daily_ret.std()) * np.sqrt(365) if daily_ret.std() > 0 else 0

    buys     = [t for t in trades if t['type'] == 'BUY']
    sells    = [t for t in trades if t['type'] == 'SELL']
    pairs    = list(zip(buys, sells))
    wins     = sum(1 for b, s in pairs if s['price'] > b['price'])
    win_rate = (wins / len(pairs) * 100) if pairs else 0

    return {
        'portfolio'   : port_series,
        'trades'      : trades,
        'final_value' : final_value,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe'      : sharpe,
        'n_trades'    : len(trades),
        'win_rate'    : win_rate,
    }

# ─────────────────────────────────────────
#  5. Run both strategies
# ─────────────────────────────────────────
INITIAL_CAPITAL = 1000

result_old   = run_backtest(df, 'position_old', INITIAL_CAPITAL)
result_new   = run_backtest(df, 'position_new', INITIAL_CAPITAL)
buy_hold_ret = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
buy_hold_val = INITIAL_CAPITAL * df['close'] / df['close'].iloc[0]

# ─────────────────────────────────────────
#  6. Print comparison
# ─────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  {'Metric':<25} {'Without RSI':>12} {'With RSI':>12}")
print(f"{'='*55}")
print(f"  {'Initial Capital':<25} {'$1,000':>12} {'$1,000':>12}")
print(f"  {'Final Value':<25} ${result_old['final_value']:>11,.0f} ${result_new['final_value']:>11,.0f}")
print(f"  {'Strategy Return':<25} {result_old['total_return']:>11.1f}% {result_new['total_return']:>11.1f}%")
print(f"  {'Buy & Hold':<25} {buy_hold_ret:>11.1f}% {buy_hold_ret:>11.1f}%")
print(f"  {'Max Drawdown':<25} {result_old['max_drawdown']:>11.1f}% {result_new['max_drawdown']:>11.1f}%")
print(f"  {'Sharpe Ratio':<25} {result_old['sharpe']:>12.2f} {result_new['sharpe']:>12.2f}")
print(f"  {'Total Trades':<25} {result_old['n_trades']:>12} {result_new['n_trades']:>12}")
print(f"  {'Win Rate':<25} {result_old['win_rate']:>11.1f}% {result_new['win_rate']:>11.1f}%")
print(f"{'='*55}\n")

# ─────────────────────────────────────────
#  7. Plot charts
# ─────────────────────────────────────────
fig = plt.figure(figsize=(15, 12), facecolor='#0d1117')
gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.4, height_ratios=[2, 1, 2])

# -- Price & MA chart --
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor('#0d1117')
ax1.plot(df.index, df['close'],    color='#58a6ff', linewidth=1.2, label='BTC/USD', alpha=0.9)
ax1.plot(df.index, df['MA_short'], color='#f0883e', linewidth=1.8, label=f'MA {MA_SHORT}')
ax1.plot(df.index, df['MA_long'],  color='#3fb950', linewidth=1.8, label=f'MA {MA_LONG}')

new_buys  = [t for t in result_new['trades'] if t['type'] == 'BUY']
new_sells = [t for t in result_new['trades'] if t['type'] == 'SELL']
for t in new_buys:
    ax1.scatter(t['date'], t['price'], color='#3fb950', marker='^', s=160, zorder=5)
for t in new_sells:
    ax1.scatter(t['date'], t['price'], color='#f85149', marker='v', s=160, zorder=5)

custom_legend = [
    Line2D([0],[0], color='#58a6ff', linewidth=1.5, label='BTC/USD'),
    Line2D([0],[0], color='#f0883e', linewidth=1.5, label=f'MA {MA_SHORT}'),
    Line2D([0],[0], color='#3fb950', linewidth=1.5, label=f'MA {MA_LONG}'),
    Line2D([0],[0], marker='^', color='w', markerfacecolor='#3fb950', markersize=10, label='Buy (with RSI)'),
    Line2D([0],[0], marker='v', color='w', markerfacecolor='#f85149', markersize=10, label='Sell (with RSI)'),
]
ax1.legend(handles=custom_legend, loc='upper left', facecolor='#161b22', labelcolor='white', fontsize=9)
ax1.set_title('BTC/USD — Price, MA & New Signals', color='white', fontsize=11, pad=8)
ax1.set_ylabel('Price (USD)', color='#8b949e')
ax1.tick_params(colors='#8b949e')
ax1.grid(alpha=0.08, color='white')
for spine in ax1.spines.values(): spine.set_color('#30363d')

# -- RSI chart --
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor('#0d1117')
ax2.plot(df.index, df['RSI'], color='#a371f7', linewidth=1.5, label='RSI')
ax2.axhline(y=70,       color='#f85149', linestyle='--', linewidth=1, alpha=0.7, label='Overbought (70)')
ax2.axhline(y=30,       color='#3fb950', linestyle='--', linewidth=1, alpha=0.7, label='Oversold (30)')
ax2.axhline(y=RSI_BUY,  color='#f0883e', linestyle=':',  linewidth=1, alpha=0.7, label=f'Buy filter ({RSI_BUY})')
ax2.axhline(y=RSI_SELL, color='#58a6ff', linestyle=':',  linewidth=1, alpha=0.7, label=f'Sell filter ({RSI_SELL})')
ax2.fill_between(df.index, df['RSI'], 70, where=(df['RSI'] >= 70), alpha=0.15, color='#f85149')
ax2.fill_between(df.index, df['RSI'], 30, where=(df['RSI'] <= 30), alpha=0.15, color='#3fb950')
ax2.set_ylim(0, 100)
ax2.set_ylabel('RSI', color='#8b949e')
ax2.set_title(f'RSI ({RSI_PERIOD})', color='white', fontsize=11, pad=8)
ax2.tick_params(colors='#8b949e')
ax2.legend(loc='upper left', facecolor='#161b22', labelcolor='white', fontsize=8)
ax2.grid(alpha=0.08, color='white')
for spine in ax2.spines.values(): spine.set_color('#30363d')

# -- Portfolio comparison chart --
ax3 = fig.add_subplot(gs[2])
ax3.set_facecolor('#0d1117')
ax3.plot(df.index, result_old['portfolio'], color='#8b949e', linewidth=1.5, linestyle='--',
         label=f"Without RSI: ${result_old['final_value']:,.0f} ({result_old['total_return']:+.1f}%)")
ax3.plot(df.index, result_new['portfolio'], color='#f0883e', linewidth=2,
         label=f"With RSI: ${result_new['final_value']:,.0f} ({result_new['total_return']:+.1f}%)")
ax3.plot(df.index, buy_hold_val, color='#58a6ff', linewidth=1.5, linestyle=':',
         label=f"Buy & Hold: ${INITIAL_CAPITAL*(1+buy_hold_ret/100):,.0f} ({buy_hold_ret:+.1f}%)")
ax3.axhline(y=INITIAL_CAPITAL, color='#30363d', linestyle=':', linewidth=1)
ax3.fill_between(df.index, result_new['portfolio'], INITIAL_CAPITAL,
                 where=(result_new['portfolio'] >= INITIAL_CAPITAL), alpha=0.12, color='#3fb950')
ax3.fill_between(df.index, result_new['portfolio'], INITIAL_CAPITAL,
                 where=(result_new['portfolio'] < INITIAL_CAPITAL), alpha=0.12, color='#f85149')
ax3.set_title('Portfolio Comparison — Without RSI vs With RSI vs Buy & Hold',
              color='white', fontsize=11, pad=8)
ax3.set_ylabel('Value (USD)', color='#8b949e')
ax3.tick_params(colors='#8b949e')
ax3.legend(loc='upper left', facecolor='#161b22', labelcolor='white', fontsize=9)
ax3.grid(alpha=0.08, color='white')
for spine in ax3.spines.values(): spine.set_color('#30363d')

plt.suptitle(f'MA Crossover + RSI Filter  |  MA{MA_SHORT}/MA{MA_LONG}  |  RSI({RSI_PERIOD})',
             color='white', fontsize=13, fontweight='bold', y=1.01)

plt.savefig('btc_backtest_rsi.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("Chart saved: btc_backtest_rsi.png")
