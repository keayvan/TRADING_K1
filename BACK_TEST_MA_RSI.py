# ============================================================
#  BTC/USD — Modular Backtest Framework
#  Version 3.1 — Fixed MACD signal logic (state-based)
#  Requirements: pip install yfinance pandas numpy matplotlib ta
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import ta

# ════════════════════════════════════════════════════════════
#  CONFIG — Change your parameters here
# ════════════════════════════════════════════════════════════
CONFIG = {
    'ticker'          : 'BTC-USD',
    'period'          : '2y',
    'interval'        : '1d',
    'initial_capital' : 1000,
    # MA parameters
    'ma_short'        : 20,
    'ma_long'         : 50,
    # RSI parameters
    'rsi_period'      : 14,
    'rsi_buy'         : 70,
    'rsi_sell'        : 30,
    # MACD parameters
    'macd_fast'       : 12,
    'macd_slow'       : 26,
    'macd_signal'     : 9,
}

# ════════════════════════════════════════════════════════════
#  1. DATA
# ════════════════════════════════════════════════════════════

def fetch_data(ticker, period, interval):
    """Fetch OHLCV data from Yahoo Finance. Keeps all columns."""
    print(f"Fetching {ticker} data from Yahoo Finance...")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    df = df.copy()
    df.columns = df.columns.get_level_values(0).str.strip()
    df.dropna(inplace=True)
    print(f"Total days: {len(df)} | From {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Columns: {list(df.columns)}")
    return df

# ════════════════════════════════════════════════════════════
#  2. INDICATORS — one function per indicator
# ════════════════════════════════════════════════════════════

def add_moving_average(df, window, col_name=None):
    """Add a Simple Moving Average column to the dataframe."""
    col_name = col_name or f'MA{window}'
    df[col_name] = df['Close'].rolling(window=window).mean()
    return df


def add_rsi(df, window=14, col_name='RSI'):
    """Add RSI column to the dataframe."""
    df[col_name] = ta.momentum.RSIIndicator(df['Close'], window=window).rsi()
    return df


def add_macd(df, fast=12, slow=26, signal=9):
    """
    Add MACD columns to the dataframe:
      MACD        = EMA(fast) - EMA(slow)
      MACD_signal = EMA(9) of MACD
      MACD_hist   = MACD - MACD_signal
    """
    macd_indicator    = ta.trend.MACD(df['Close'], window_fast=fast,
                                       window_slow=slow, window_sign=signal)
    df['MACD']        = macd_indicator.macd()
    df['MACD_signal'] = macd_indicator.macd_signal()
    df['MACD_hist']   = macd_indicator.macd_diff()
    return df


def add_bollinger_bands(df, window=20, std=2):
    """Add Bollinger Bands (upper, middle, lower) to the dataframe."""
    bb = ta.volatility.BollingerBands(df['Close'], window=window, window_dev=std)
    df['BB_upper']  = bb.bollinger_hband()
    df['BB_middle'] = bb.bollinger_mavg()
    df['BB_lower']  = bb.bollinger_lband()
    return df

# ════════════════════════════════════════════════════════════
#  3. SIGNALS
# ════════════════════════════════════════════════════════════

def generate_signals_ma(df, ma_short_col='MA_short', ma_long_col='MA_long',
                         signal_col='signal_ma'):
    """Generate buy/sell signals based on MA crossover only."""
    df[signal_col] = 0
    df.loc[df[ma_short_col] > df[ma_long_col], signal_col] =  1
    df.loc[df[ma_short_col] < df[ma_long_col], signal_col] = -1
    df[f'position_{signal_col}'] = df[signal_col].diff()
    return df


def generate_signals_ma_rsi(df, ma_short_col='MA_short', ma_long_col='MA_long',
                              rsi_col='RSI', rsi_buy=70, rsi_sell=30,
                              signal_col='signal_ma_rsi'):
    """Generate buy/sell signals based on MA crossover + RSI filter."""
    df[signal_col] = 0
    df.loc[
        (df[ma_short_col] > df[ma_long_col]) & (df[rsi_col] < rsi_buy),
        signal_col
    ] =  1
    df.loc[
        (df[ma_short_col] < df[ma_long_col]) & (df[rsi_col] > rsi_sell),
        signal_col
    ] = -1
    df[f'position_{signal_col}'] = df[signal_col].diff()
    return df


def generate_signals_ma_macd(df, ma_short_col='MA_short', ma_long_col='MA_long',
                               macd_col='MACD', macd_signal_col='MACD_signal',
                               signal_col='signal_ma_macd'):
    """
    Generate buy/sell signals using state-based MA + MACD logic.

    STATE-BASED approach (not crossover-based):
      BUY state  → MA_short > MA_long  AND  MACD > MACD_signal  (both bullish)
      SELL state → MA_short < MA_long  AND  MACD < MACD_signal  (both bearish)

    A trade fires only when the combined state CHANGES — not every day.
    This avoids the crossover timing problem where both signals rarely
    happen on the exact same day.
    """
    # Step 1 — define the combined state each day
    df[signal_col] = 0
    df.loc[
        (df[ma_short_col] > df[ma_long_col]) & (df[macd_col] > df[macd_signal_col]),
        signal_col
    ] =  1   # Both indicators agree: bullish
    df.loc[
        (df[ma_short_col] < df[ma_long_col]) & (df[macd_col] < df[macd_signal_col]),
        signal_col
    ] = -1   # Both indicators agree: bearish

    # Step 2 — fire a trade only when the state changes
    df[f'position_{signal_col}'] = df[signal_col].diff()
    return df

# ════════════════════════════════════════════════════════════
#  4. BACKTEST ENGINE
# ════════════════════════════════════════════════════════════

def run_backtest(df, position_col, initial_capital=1000):
    """
    Run a backtest given a position column.
    Returns a dict with portfolio series, trades list, and all metrics.
    """
    capital          = initial_capital
    btc              = 0.0
    portfolio_values = []
    trades           = []

    for date, row in df.iterrows():
        price = float(row['Close'])
        pos   = float(row[position_col])

        if pos == 2 and capital > 0:           # Buy signal
            btc     = capital / price
            capital = 0
            trades.append({'date': date, 'type': 'BUY', 'price': price})

        elif pos == -2 and btc > 0:            # Sell signal
            capital = btc * price
            btc     = 0
            trades.append({'date': date, 'type': 'SELL', 'price': price})

        portfolio_values.append(capital + btc * price)

    return _calculate_metrics(df, portfolio_values, trades, initial_capital)


def _calculate_metrics(df, portfolio_values, trades, initial_capital):
    """Calculate all backtest performance metrics."""
    port_series  = pd.Series(portfolio_values, index=df.index)
    final_value  = port_series.iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100

    # Max Drawdown
    rolling_max  = port_series.cummax()
    drawdown     = (port_series - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()

    # Sharpe Ratio (annualized)
    daily_ret = port_series.pct_change().dropna()
    sharpe    = (daily_ret.mean() / daily_ret.std()) * np.sqrt(365) \
                if daily_ret.std() > 0 else 0

    # Win Rate
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

# ════════════════════════════════════════════════════════════
#  5. ANALYSIS — print results
# ════════════════════════════════════════════════════════════

def print_single_result(result, label='Strategy', initial_capital=1000):
    """Print backtest results for a single strategy."""
    print(f"\n{'='*40}")
    print(f"  {label}")
    print(f"{'='*40}")
    print(f"  Initial Capital  : ${initial_capital:,}")
    print(f"  Final Value      : ${result['final_value']:,.2f}")
    print(f"  Strategy Return  : {result['total_return']:+.1f}%")
    print(f"  Max Drawdown     : {result['max_drawdown']:.1f}%")
    print(f"  Sharpe Ratio     : {result['sharpe']:.2f}")
    print(f"  Total Trades     : {result['n_trades']}")
    print(f"  Win Rate         : {result['win_rate']:.1f}%")
    print(f"{'='*40}\n")


def print_comparison(results: dict, buy_hold_ret, initial_capital=1000):
    """
    Print a side-by-side comparison of multiple strategies.
    results: dict of {label: result_dict}
    """
    labels = list(results.keys())
    col_w  = 13

    header = f"  {'Metric':<25}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(f"\n{'='*(25 + col_w * len(labels) + 2)}")
    print(header)
    print(f"{'='*(25 + col_w * len(labels) + 2)}")

    rows = [
        ('Initial Capital', lambda r: f"${initial_capital:,}"),
        ('Final Value',     lambda r: f"${r['final_value']:,.0f}"),
        ('Strategy Return', lambda r: f"{r['total_return']:+.1f}%"),
        ('Buy & Hold',      lambda r: f"{buy_hold_ret:+.1f}%"),
        ('Max Drawdown',    lambda r: f"{r['max_drawdown']:.1f}%"),
        ('Sharpe Ratio',    lambda r: f"{r['sharpe']:.2f}"),
        ('Total Trades',    lambda r: f"{r['n_trades']}"),
        ('Win Rate',        lambda r: f"{r['win_rate']:.1f}%"),
    ]

    for metric, fn in rows:
        line = f"  {metric:<25}" + "".join(f"{fn(r):>{col_w}}" for r in results.values())
        print(line)

    print(f"{'='*(25 + col_w * len(labels) + 2)}\n")

# ════════════════════════════════════════════════════════════
#  6. PLOTS
# ════════════════════════════════════════════════════════════

DARK = {
    'bg'    : '#0d1117',
    'panel' : '#161b22',
    'border': '#30363d',
    'text'  : '#c9d1d9',
    'muted' : '#8b949e',
    'blue'  : '#58a6ff',
    'orange': '#f0883e',
    'green' : '#3fb950',
    'red'   : '#f85149',
    'purple': '#a371f7',
    'grey'  : '#8b949e',
}


def _style_axis(ax):
    """Apply consistent dark theme styling to an axis."""
    ax.tick_params(colors=DARK['muted'])
    ax.grid(alpha=0.08, color='white')
    for spine in ax.spines.values():
        spine.set_color(DARK['border'])


def plot_price_and_signals(ax, df, ma_short_col, ma_long_col, trades,
                            ma_short_period, ma_long_period):
    """Plot price, moving averages and buy/sell signals on a given axis."""
    ax.set_facecolor(DARK['bg'])
    ax.plot(df.index, df['Close'],      color=DARK['blue'],   linewidth=1.2, alpha=0.9)
    ax.plot(df.index, df[ma_short_col], color=DARK['orange'], linewidth=1.8)
    ax.plot(df.index, df[ma_long_col],  color=DARK['green'],  linewidth=1.8)

    buys  = [t for t in trades if t['type'] == 'BUY']
    sells = [t for t in trades if t['type'] == 'SELL']
    for t in buys:
        ax.scatter(t['date'], t['price'], color=DARK['green'], marker='^', s=160, zorder=5)
    for t in sells:
        ax.scatter(t['date'], t['price'], color=DARK['red'],   marker='v', s=160, zorder=5)

    legend_items = [
        Line2D([0],[0], color=DARK['blue'],   linewidth=1.5, label='BTC/USD'),
        Line2D([0],[0], color=DARK['orange'], linewidth=1.5, label=f'MA {ma_short_period}'),
        Line2D([0],[0], color=DARK['green'],  linewidth=1.5, label=f'MA {ma_long_period}'),
        Line2D([0],[0], marker='^', color='w', markerfacecolor=DARK['green'], markersize=10, label='Buy'),
        Line2D([0],[0], marker='v', color='w', markerfacecolor=DARK['red'],   markersize=10, label='Sell'),
    ]
    ax.legend(handles=legend_items, loc='upper left',
              facecolor=DARK['panel'], labelcolor='white', fontsize=9)
    ax.set_title('BTC/USD — Price & Signals', color='white', fontsize=11, pad=8)
    ax.set_ylabel('Price (USD)', color=DARK['muted'])
    _style_axis(ax)


def plot_macd(ax, df, macd_col='MACD', signal_col='MACD_signal', hist_col='MACD_hist'):
    """Plot MACD Line, Signal Line and Histogram on a given axis."""
    ax.set_facecolor(DARK['bg'])

    # Histogram — green when positive, red when negative
    colors = [DARK['green'] if v >= 0 else DARK['red'] for v in df[hist_col]]
    ax.bar(df.index, df[hist_col], color=colors, alpha=0.5, width=1, label='Histogram')

    ax.plot(df.index, df[macd_col],   color=DARK['blue'],   linewidth=1.5, label='MACD Line')
    ax.plot(df.index, df[signal_col], color=DARK['orange'], linewidth=1.5, label='Signal Line')
    ax.axhline(y=0, color=DARK['border'], linestyle='--', linewidth=1, alpha=0.7)

    ax.set_ylabel('MACD', color=DARK['muted'])
    ax.set_title('MACD (12, 26, 9)', color='white', fontsize=11, pad=8)
    ax.legend(loc='upper left', facecolor=DARK['panel'], labelcolor='white', fontsize=8)
    _style_axis(ax)


def plot_portfolio_comparison(ax, df, results: dict, buy_hold_val,
                               buy_hold_ret, initial_capital):
    """Plot portfolio value over time for multiple strategies vs Buy & Hold."""
    ax.set_facecolor(DARK['bg'])

    colors = [DARK['orange'], DARK['green'], DARK['purple'], DARK['blue']]
    styles = ['-', '--', '-.', ':']

    for i, (label, result) in enumerate(results.items()):
        color = colors[i % len(colors)]
        style = styles[i % len(styles)]
        ax.plot(df.index, result['portfolio'], color=color, linewidth=2,
                linestyle=style,
                label=f"{label}: ${result['final_value']:,.0f} ({result['total_return']:+.1f}%)")

    ax.plot(df.index, buy_hold_val, color=DARK['grey'], linewidth=1.5,
            linestyle=':', label=f"Buy & Hold: ${initial_capital*(1+buy_hold_ret/100):,.0f} ({buy_hold_ret:+.1f}%)")
    ax.axhline(y=initial_capital, color=DARK['border'], linestyle=':', linewidth=1)

    first_portfolio = list(results.values())[0]['portfolio']
    ax.fill_between(df.index, first_portfolio, initial_capital,
                    where=(first_portfolio >= initial_capital), alpha=0.10, color=DARK['green'])
    ax.fill_between(df.index, first_portfolio, initial_capital,
                    where=(first_portfolio < initial_capital),  alpha=0.10, color=DARK['red'])

    ax.set_title('Portfolio Comparison', color='white', fontsize=11, pad=8)
    ax.set_ylabel('Value (USD)', color=DARK['muted'])
    ax.legend(loc='upper left', facecolor=DARK['panel'], labelcolor='white', fontsize=9)
    _style_axis(ax)


def plot_full_dashboard(df, results, buy_hold_val, buy_hold_ret,
                         initial_capital, ma_short_col, ma_long_col,
                         ma_short_period, ma_long_period,
                         primary_strategy_label, save_path='btc_backtest_v3.png'):
    """
    Render the full 3-panel dashboard:
      Panel 1 — Price + MA + signals
      Panel 2 — MACD
      Panel 3 — Portfolio comparison
    """
    fig = plt.figure(figsize=(15, 12), facecolor=DARK['bg'])
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.4, height_ratios=[2, 1, 2])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    primary_trades = results[primary_strategy_label]['trades']

    plot_price_and_signals(ax1, df, ma_short_col, ma_long_col,
                            primary_trades, ma_short_period, ma_long_period)
    plot_macd(ax2, df)
    plot_portfolio_comparison(ax3, df, results, buy_hold_val,
                               buy_hold_ret, initial_capital)

    plt.suptitle(
        f'MA + MACD Strategy  |  MA{ma_short_period}/MA{ma_long_period}  |  MACD(12,26,9)',
        color='white', fontsize=13, fontweight='bold', y=1.01
    )

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK['bg'])
    plt.show()
    print(f"Chart saved: {save_path}")

# ════════════════════════════════════════════════════════════
#  7. MAIN — run everything
# ════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # -- Load data --
    df = fetch_data(CONFIG['ticker'], CONFIG['period'], CONFIG['interval'])

    # -- Add indicators --
    df = add_moving_average(df, CONFIG['ma_short'], col_name='MA_short')
    df = add_moving_average(df, CONFIG['ma_long'],  col_name='MA_long')
    df = add_rsi(df, CONFIG['rsi_period'])
    df = add_macd(df, CONFIG['macd_fast'], CONFIG['macd_slow'], CONFIG['macd_signal'])
    # df = add_bollinger_bands(df)   # uncomment to add Bollinger Bands

    # -- Generate signals --
    df = generate_signals_ma(df, signal_col='signal_ma')
    df = generate_signals_ma_rsi(df,
                                  rsi_buy=CONFIG['rsi_buy'],
                                  rsi_sell=CONFIG['rsi_sell'],
                                  signal_col='signal_ma_rsi')
    df = generate_signals_ma_macd(df, signal_col='signal_ma_macd')

    # -- Run backtests --
    result_ma      = run_backtest(df, 'position_signal_ma',      CONFIG['initial_capital'])
    result_ma_rsi  = run_backtest(df, 'position_signal_ma_rsi',  CONFIG['initial_capital'])
    result_ma_macd = run_backtest(df, 'position_signal_ma_macd', CONFIG['initial_capital'])

    buy_hold_ret = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
    buy_hold_val = CONFIG['initial_capital'] * df['Close'] / df['Close'].iloc[0]

    # -- Print comparison --
    results = {
        'MA Only'  : result_ma,
        'MA + RSI' : result_ma_rsi,
        'MA + MACD': result_ma_macd,
    }
    print_comparison(results, buy_hold_ret, CONFIG['initial_capital'])

    # -- Plot dashboard --
    plot_full_dashboard(
        df                     = df,
        results                = results,
        buy_hold_val           = buy_hold_val,
        buy_hold_ret           = buy_hold_ret,
        initial_capital        = CONFIG['initial_capital'],
        ma_short_col           = 'MA_short',
        ma_long_col            = 'MA_long',
        ma_short_period        = CONFIG['ma_short'],
        ma_long_period         = CONFIG['ma_long'],
        primary_strategy_label = 'MA + MACD',
        save_path              = 'btc_backtest_v3.png',
    )