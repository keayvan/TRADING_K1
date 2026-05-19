# ============================================================
#  BTC/USD -- Moving Average Crossover Backtest
#  Requirements: pip install yfinance pandas numpy matplotlib
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ─────────────────────────────────────────
#  1. Fetch data from Yahoo Finance
# ─────────────────────────────────────────
print("Fetching data from Yahoo Finance...")

ticker = "BTC-USD"
df = yf.download(ticker, period="2y", interval="1d", auto_adjust=True)
df = df[['Close']].copy()
df.columns = ['close']
df.dropna(inplace=True)

print(f"Total days: {len(df)}")
print(f"From {df.index[0].date()} to {df.index[-1].date()}")
print("-" * 40)

# ─────────────────────────────────────────
#  2. Calculate indicators
# ─────────────────────────────────────────
MA_SHORT = 20   # short-term moving average period
MA_LONG  = 50   # long-term moving average period

df['MA_short'] = df['close'].rolling(window=MA_SHORT).mean()
df['MA_long']  = df['close'].rolling(window=MA_LONG).mean()

# ─────────────────────────────────────────
#  3. Generate buy/sell signals
# ─────────────────────────────────────────
df['signal'] = 0
df.loc[df['MA_short'] > df['MA_long'], 'signal'] = 1    # bullish
df.loc[df['MA_short'] < df['MA_long'], 'signal'] = -1   # bearish
df['position'] = df['signal'].diff()  # signal change = trade trigger

# ─────────────────────────────────────────
#  4. Backtesting
# ─────────────────────────────────────────
INITIAL_CAPITAL = 1000  # USD

capital = INITIAL_CAPITAL
btc     = 0.0
portfolio_values = []
trades = []

for date, row in df.iterrows():
    price = float(row['close'])
    pos   = float(row['position'])

    # Buy signal
    if pos == 2 and capital > 0:
        btc = capital / price
        capital = 0
        trades.append({'date': date, 'type': 'BUY', 'price': price, 'btc': btc})

    # Sell signal
    elif pos == -2 and btc > 0:
        capital = btc * price
        btc = 0
        trades.append({'date': date, 'type': 'SELL', 'price': price, 'capital': capital})

    total = capital + btc * price
    portfolio_values.append(total)

df['portfolio'] = portfolio_values

# ─────────────────────────────────────────
#  5. Performance metrics
# ─────────────────────────────────────────
final_value  = df['portfolio'].iloc[-1]
total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
buy_hold_ret = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100

# Max Drawdown
rolling_max  = df['portfolio'].cummax()
drawdown     = (df['portfolio'] - rolling_max) / rolling_max * 100
max_drawdown = drawdown.min()

# Win Rate
buy_trades  = [t for t in trades if t['type'] == 'BUY']
sell_trades = [t for t in trades if t['type'] == 'SELL']
pairs       = list(zip(buy_trades, sell_trades))
wins        = sum(1 for b, s in pairs if s['price'] > b['price'])
total_pairs = len(pairs)
win_rate    = (wins / total_pairs * 100) if total_pairs > 0 else 0

# Sharpe Ratio (annualized approximation)
daily_returns = df['portfolio'].pct_change().dropna()
sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)

print(f"\n{'='*40}")
print(f"  Backtest Results")
print(f"{'='*40}")
print(f"  Initial Capital : ${INITIAL_CAPITAL:,.0f}")
print(f"  Final Value     : ${final_value:,.2f}")
print(f"  Strategy Return : {total_return:+.1f}%")
print(f"  Buy & Hold      : {buy_hold_ret:+.1f}%")
print(f"  Max Drawdown    : {max_drawdown:.1f}%")
print(f"  Sharpe Ratio    : {sharpe:.2f}")
print(f"  Total Trades    : {len(trades)}")
print(f"  Win Rate        : {win_rate:.1f}%")
print(f"{'='*40}\n")

# ─────────────────────────────────────────
#  6. Plot results
# ─────────────────────────────────────────
fig = plt.figure(figsize=(15, 10), facecolor='#0d1117')
gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.35)

# -- Price chart with signals --
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor('#0d1117')
ax1.plot(df.index, df['close'],    color='#58a6ff', linewidth=1.2, label='BTC/USD', alpha=0.9)
ax1.plot(df.index, df['MA_short'], color='#f0883e', linewidth=1.8, label=f'MA {MA_SHORT}', alpha=0.95)
ax1.plot(df.index, df['MA_long'],  color='#3fb950', linewidth=1.8, label=f'MA {MA_LONG}', alpha=0.95)

for t in buy_trades:
    ax1.scatter(t['date'], t['price'], color='#3fb950', marker='^', s=160, zorder=5)
for t in sell_trades:
    ax1.scatter(t['date'], t['price'], color='#f85149', marker='v', s=160, zorder=5)

custom_legend = [
    Line2D([0],[0], color='#58a6ff', linewidth=1.5, label='BTC/USD'),
    Line2D([0],[0], color='#f0883e', linewidth=1.5, label=f'MA {MA_SHORT}'),
    Line2D([0],[0], color='#3fb950', linewidth=1.5, label=f'MA {MA_LONG}'),
    Line2D([0],[0], marker='^', color='w', markerfacecolor='#3fb950', markersize=10, label='Buy'),
    Line2D([0],[0], marker='v', color='w', markerfacecolor='#f85149', markersize=10, label='Sell'),
]
ax1.legend(handles=custom_legend, loc='upper left', facecolor='#161b22', labelcolor='white', fontsize=9)
ax1.set_title('BTC/USD -- Price & Trade Signals', color='white', fontsize=12, pad=10)
ax1.set_ylabel('Price (USD)', color='#8b949e')
ax1.tick_params(colors='#8b949e')
ax1.grid(alpha=0.08, color='white')
for spine in ax1.spines.values():
    spine.set_color('#30363d')

# -- Portfolio value chart --
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor('#0d1117')

bh_values = INITIAL_CAPITAL * df['close'] / df['close'].iloc[0]

ax2.plot(df.index, df['portfolio'], color='#f0883e', linewidth=2,
         label=f'MA Strategy: ${final_value:,.0f}')
ax2.plot(df.index, bh_values, color='#8b949e', linewidth=1.5, linestyle='--',
         label=f'Buy & Hold: ${INITIAL_CAPITAL*(1+buy_hold_ret/100):,.0f}')
ax2.axhline(y=INITIAL_CAPITAL, color='#30363d', linestyle=':', linewidth=1)

ax2.fill_between(df.index, df['portfolio'], INITIAL_CAPITAL,
                 where=(df['portfolio'] >= INITIAL_CAPITAL), alpha=0.12, color='#3fb950')
ax2.fill_between(df.index, df['portfolio'], INITIAL_CAPITAL,
                 where=(df['portfolio'] < INITIAL_CAPITAL), alpha=0.12, color='#f85149')

ax2.set_title('Portfolio Value Over Time', color='white', fontsize=12, pad=10)
ax2.set_ylabel('Value (USD)', color='#8b949e')
ax2.tick_params(colors='#8b949e')
ax2.legend(loc='upper left', facecolor='#161b22', labelcolor='white', fontsize=9)
ax2.grid(alpha=0.08, color='white')
for spine in ax2.spines.values():
    spine.set_color('#30363d')

# -- Results summary bar --
result_str = (
    f"Strategy: {total_return:+.1f}%  |  "
    f"Buy&Hold: {buy_hold_ret:+.1f}%  |  "
    f"Max DD: {max_drawdown:.1f}%  |  "
    f"Sharpe: {sharpe:.2f}  |  "
    f"Win Rate: {win_rate:.0f}%  |  "
    f"Trades: {len(trades)}"
)
fig.text(0.5, 0.01, result_str, ha='center', color='#c9d1d9', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='#161b22', alpha=0.9, edgecolor='#30363d'))

plt.suptitle(f'Moving Average Crossover Backtest  |  MA{MA_SHORT}/MA{MA_LONG}  |  2 Years',
             color='white', fontsize=13, fontweight='bold', y=1.01)

plt.savefig('btc_backtest.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("Chart saved: btc_backtest.png")