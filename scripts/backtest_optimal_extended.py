"""
Optimal config backtest:
  delta=0.30, hold=20, stop=3.0x, no VIX filter

Two windows:
  A) 2019-2026 with real VIX (primary, most reliable)
  B) 2007-2019 with realized-vol proxy for IV (extended, caveat noted)

Author: P. Rameswara Anand
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

NIFTY_PATH = '/root/nifty_combined_19yr.csv'
VIX_PATH = '/root/TOB/data/historical/vix_daily_2019_2026.csv'

LOT_SIZE = 65
RISK_FREE = 0.065
STRIKE_STEP = 50
COST_2_LEG = 130
SLIPPAGE_PCT = 0.005
SPREAD_WIDTH = 500
DAYS_TO_EXPIRY = 30

# OPTIMAL CONFIG (from multisweep)
DELTA = 0.30
HOLD = 20
STOP_MULT = 3.0
VIX_FILTER = None


def bs_price(S, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1


def find_strike_for_delta(S, target_delta, T, r, sigma, option_type):
    strike = round(S / STRIKE_STEP) * STRIKE_STEP
    for _ in range(30):
        d = bs_delta(S, strike, T, r, sigma, option_type)
        if option_type == 'call':
            if d > target_delta:
                strike += STRIKE_STEP
            else:
                break
        else:
            if abs(d) > target_delta:
                strike -= STRIKE_STEP
            else:
                break
    return strike


def load_full_data():
    nifty = pd.read_csv(NIFTY_PATH)
    nifty['datetime'] = pd.to_datetime(nifty['datetime'])
    nifty = nifty.set_index('datetime').sort_index()
    nifty = nifty[~nifty.index.duplicated(keep='first')]
    nifty = nifty.rename(columns={'close': 'nifty_close'})

    vix = pd.read_csv(VIX_PATH)
    vix['datetime'] = pd.to_datetime(vix['datetime'])
    vix = vix.set_index('datetime').sort_index()
    vix = vix[~vix.index.duplicated(keep='first')]
    vix = vix.rename(columns={'close': 'vix_close'})

    merged = nifty.join(vix, how='left')
    merged['vix_close'] = merged['vix_close'].ffill()

    # realized vol for pre-VIX period
    merged['log_ret'] = np.log(merged['nifty_close'] / merged['nifty_close'].shift(1))
    merged['realized_vol'] = merged['log_ret'].rolling(20).std() * np.sqrt(252)

    # IV source: real VIX where available, else realized vol
    merged['iv_real'] = merged['vix_close'] / 100.0
    merged['iv_proxy'] = merged['realized_vol']
    merged['iv_used'] = merged['iv_real'].fillna(merged['iv_proxy']).fillna(0.15)
    merged['iv_used'] = merged['iv_used'].clip(0.08, 0.80)
    merged['iv_is_real'] = merged['iv_real'].notna()

    merged['open'] = merged['nifty_close'].shift(1)
    merged['high'] = merged[['nifty_close', 'open']].max(axis=1)
    merged['low'] = merged[['nifty_close', 'open']].min(axis=1)
    return merged.dropna(subset=['nifty_close'])


def compute_indicators(d):
    d = d.copy()
    d['bb_mid'] = d['nifty_close'].rolling(20).mean()
    d['bb_std'] = d['nifty_close'].rolling(20).std()
    d['bb_upper'] = d['bb_mid'] + 2 * d['bb_std']
    d['bb_lower'] = d['bb_mid'] - 2 * d['bb_std']
    d['bb_width'] = (d['bb_upper'] - d['bb_lower']) / d['bb_mid']
    d['bb_w_min'] = d['bb_width'].rolling(60).min()
    d['bb_w_max'] = d['bb_width'].rolling(60).max()
    d['bb_w_pct'] = (d['bb_width'] - d['bb_w_min']) / (d['bb_w_max'] - d['bb_w_min'] + 1e-10)

    def classify(p):
        if pd.isna(p): return 'NORMAL'
        if p <= 0.20: return 'SQUEEZE'
        if p >= 0.70: return 'WIDE'
        return 'NORMAL'
    d['regime'] = d['bb_w_pct'].apply(classify)

    d['ema5'] = d['nifty_close'].ewm(span=5, adjust=False).mean()
    d['ema13'] = d['nifty_close'].ewm(span=13, adjust=False).mean()
    diff = d['ema5'] - d['ema13']
    d['diff_change'] = diff.diff(3)
    d['mts'] = 'YELLOW'
    d.loc[(d['ema5'] > d['ema13']) & (d['diff_change'] > 0), 'mts'] = 'GREEN'
    d.loc[(d['ema5'] < d['ema13']) & (d['diff_change'] < 0), 'mts'] = 'RED'

    d['kijun'] = (d['low'].rolling(26).min() + d['high'].rolling(26).max()) / 2
    d['tenkan'] = (d['high'].rolling(9).max() + d['low'].rolling(9).min()) / 2
    d['senkouA'] = (d['tenkan'] + d['kijun']) / 2
    d['senkouB'] = (d['high'].rolling(52).max() + d['low'].rolling(52).min()) / 2
    d['cloud_top'] = d[['senkouA', 'senkouB']].max(axis=1)
    d['cloud_bottom'] = d[['senkouA', 'senkouB']].min(axis=1)
    d['above_cloud'] = d['nifty_close'] > d['cloud_top']
    return d


def simulate(S_entry, S_path, iv_entry, iv_path):
    r = RISK_FREE
    T_entry = DAYS_TO_EXPIRY / 365.0
    K_short = find_strike_for_delta(S_entry, DELTA, T_entry, r, iv_entry, 'put')
    K_long = K_short - SPREAD_WIDTH

    short_entry = bs_price(S_entry, K_short, T_entry, r, iv_entry, 'put')
    long_entry = bs_price(S_entry, K_long, T_entry, r, iv_entry, 'put')
    credit = (short_entry - long_entry) * LOT_SIZE
    credit -= COST_2_LEG
    credit -= (short_entry + long_entry) * SLIPPAGE_PCT * LOT_SIZE

    # stop-loss check
    for j, (S_j, iv_j) in enumerate(zip(S_path, iv_path)):
        T_j = max((DAYS_TO_EXPIRY - (j + 1)) / 365.0, 1/365.0)
        short_j = bs_price(S_j, K_short, T_j, r, iv_j, 'put')
        long_j = bs_price(S_j, K_long, T_j, r, iv_j, 'put')
        spread_val = (short_j - long_j) * LOT_SIZE
        if spread_val >= short_entry * STOP_MULT * LOT_SIZE:
            exit_cost = spread_val + COST_2_LEG + (short_j + long_j) * SLIPPAGE_PCT * LOT_SIZE
            return credit - exit_cost

    # time exit
    S_exit = S_path[-1]
    iv_exit = iv_path[-1]
    T_exit = max((DAYS_TO_EXPIRY - HOLD) / 365.0, 1/365.0)
    short_exit = bs_price(S_exit, K_short, T_exit, r, iv_exit, 'put')
    long_exit = bs_price(S_exit, K_long, T_exit, r, iv_exit, 'put')
    exit_cost = (short_exit - long_exit) * LOT_SIZE
    exit_cost += COST_2_LEG
    exit_cost += (short_exit + long_exit) * SLIPPAGE_PCT * LOT_SIZE
    return credit - exit_cost


def run_backtest(daily, start_date, end_date, label):
    d = daily.loc[start_date:end_date]
    trades = []

    for i in range(60, len(d) - HOLD):
        row = d.iloc[i]

        if row['regime'] in ('SQUEEZE', 'WIDE'):
            continue
        if row['mts'] != 'GREEN':
            continue
        if not row['above_cloud']:
            continue

        S_entry = row['nifty_close']
        iv_entry = row['iv_used']
        S_path = d['nifty_close'].iloc[i+1 : i+1+HOLD].values
        iv_path = d['iv_used'].iloc[i+1 : i+1+HOLD].values

        if len(S_path) < HOLD:
            continue

        pnl = simulate(S_entry, S_path, iv_entry, iv_path)
        trades.append({
            'date': d.index[i],
            'year': d.index[i].year,
            'nifty': S_entry,
            'iv': iv_entry,
            'iv_is_real': row['iv_is_real'],
            'pnl': pnl,
        })

    tdf = pd.DataFrame(trades)
    print(f"\n{'=' * 70}")
    print(f"{label}")
    print(f"{'=' * 70}")

    if tdf.empty:
        print("No trades.")
        return tdf

    pnl = tdf['pnl'].values
    n = len(pnl)
    wins = (pnl > 0).sum()

    equity = np.concatenate([[100000], 100000 + np.cumsum(pnl)])
    peak = np.maximum.accumulate(equity)
    dd_series = (equity - peak) / peak
    max_dd = dd_series.min() * 100

    total_return = (equity[-1] - 100000) / 100000 * 100
    years = (tdf['date'].iloc[-1] - tdf['date'].iloc[0]).days / 365.0
    annual_return = ((equity[-1] / 100000) ** (1/years) - 1) * 100 if years > 0 else 0

    trades_per_year = n / years if years > 0 else 0
    sharpe = (pnl.mean() / pnl.std()) * np.sqrt(trades_per_year) if pnl.std() > 0 else 0
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

    print(f"  Period:            {tdf['date'].iloc[0].date()} to {tdf['date'].iloc[-1].date()}")
    print(f"  Years:             {years:.2f}")
    print(f"  Trades:            {n}")
    print(f"  Trades/year:       {trades_per_year:.1f}")
    print(f"  Win rate:          {100*wins/n:.1f}%")
    print(f"  Avg P&L:           Rs {pnl.mean():+.0f}")
    print(f"  Median P&L:        Rs {np.median(pnl):+.0f}")
    print(f"  Total P&L:         Rs {pnl.sum():+.0f}")
    print(f"  Best trade:        Rs {pnl.max():+.0f}")
    print(f"  Worst trade:       Rs {pnl.min():+.0f}")
    print(f"  Total return:      {total_return:+.1f}%")
    print(f"  Annual return:     {annual_return:+.2f}%")
    print(f"  Max drawdown:      {max_dd:.1f}%")
    print(f"  Sharpe ratio:      {sharpe:.3f}")
    print(f"  Calmar ratio:      {calmar:.3f}")

    # Yearly breakdown
    yearly = tdf.groupby('year').agg(
        n_trades=('pnl', 'count'),
        total_pnl=('pnl', 'sum'),
        win_pct=('pnl', lambda x: 100*(x>0).mean()),
    )
    print(f"\n  Yearly breakdown:")
    print(yearly.to_string())

    return tdf


if __name__ == '__main__':
    daily = load_full_data()
    daily = compute_indicators(daily)
    print(f"Data loaded: {len(daily)} daily bars, {daily.index[0].date()} to {daily.index[-1].date()}")

    # Test A: 2019-2026 with real VIX
    trades_A = run_backtest(daily, '2019-01-01', '2026-07-01',
                            'TEST A — 2019-2026 (real VIX)')

    # Test B: 2007-2019 (pre-real-VIX, uses realized-vol proxy)
    trades_B = run_backtest(daily, '2007-09-01', '2018-12-31',
                            'TEST B — 2007-2019 (realized-vol proxy, CAVEAT)')

    # Save
    os.makedirs('results', exist_ok=True)
    if not trades_A.empty:
        trades_A.to_csv('results/trades_optimal_2019.csv', index=False)
    if not trades_B.empty:
        trades_B.to_csv('results/trades_optimal_2007.csv', index=False)
    print(f"\nSaved results/trades_optimal_2019.csv and trades_optimal_2007.csv")
