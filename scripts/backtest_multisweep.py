"""
Multi-dimensional robustness sweep for the bull put credit spread.

Tests across:
  - Stop-loss levels (1.5x, 2.0x, 3.0x credit)
  - Short strike delta (0.20, 0.25, 0.30)
  - Entry regimes (3 variants)
  - Hold periods (10, 15, 20, 30)
  - VIX rank filter (above/below median)

Data: 2019-01 to 2026-07 (real VIX period)

Author: P. Rameswara Anand
"""
import os
import itertools
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


def load_data():
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

    merged = nifty.join(vix, how='inner').dropna()
    merged['open'] = merged['nifty_close'].shift(1)
    merged['high'] = merged[['nifty_close', 'open']].max(axis=1)
    merged['low'] = merged[['nifty_close', 'open']].min(axis=1)
    return merged.dropna()


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

    # VIX rank: percentile of current VIX within last 252 days
    d['vix_rank'] = d['vix_close'].rolling(252, min_periods=60).apply(
        lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 1e-10), raw=True
    )

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


def simulate(S_entry, S_path, iv_entry, iv_path, delta, hold, stop_mult):
    """
    Simulate a bull put credit spread with optional stop-loss.
    Returns net P&L in rupees.
    """
    r = RISK_FREE
    T_entry = DAYS_TO_EXPIRY / 365.0
    width = SPREAD_WIDTH

    K_short = find_strike_for_delta(S_entry, delta, T_entry, r, iv_entry, 'put')
    K_long = K_short - width

    short_entry = bs_price(S_entry, K_short, T_entry, r, iv_entry, 'put')
    long_entry = bs_price(S_entry, K_long, T_entry, r, iv_entry, 'put')

    credit = (short_entry - long_entry) * LOT_SIZE
    credit -= COST_2_LEG
    credit -= (short_entry + long_entry) * SLIPPAGE_PCT * LOT_SIZE

    # Stop-loss check on each intra-hold bar
    if stop_mult is not None:
        for j, (S_j, iv_j) in enumerate(zip(S_path, iv_path)):
            T_j = max((DAYS_TO_EXPIRY - (j + 1)) / 365.0, 1/365.0)
            short_j = bs_price(S_j, K_short, T_j, r, iv_j, 'put')
            long_j = bs_price(S_j, K_long, T_j, r, iv_j, 'put')
            spread_val = (short_j - long_j) * LOT_SIZE
            if spread_val >= short_entry * stop_mult * LOT_SIZE:
                exit_cost = spread_val
                exit_cost += COST_2_LEG
                exit_cost += (short_j + long_j) * SLIPPAGE_PCT * LOT_SIZE
                return credit - exit_cost

    # Time exit
    S_exit = S_path[-1]
    iv_exit = iv_path[-1]
    T_exit = max((DAYS_TO_EXPIRY - hold) / 365.0, 1/365.0)
    short_exit = bs_price(S_exit, K_short, T_exit, r, iv_exit, 'put')
    long_exit = bs_price(S_exit, K_long, T_exit, r, iv_exit, 'put')
    exit_cost = (short_exit - long_exit) * LOT_SIZE
    exit_cost += COST_2_LEG
    exit_cost += (short_exit + long_exit) * SLIPPAGE_PCT * LOT_SIZE
    return credit - exit_cost


def run_single(daily, delta, hold, stop_mult, regime_filter='loose', vix_rank_filter=None):
    """
    Run the backtest once for one parameter set.
    regime_filter: 'loose' (exclude SQUEEZE/WIDE), 'tight' (NORMAL only)
    vix_rank_filter: None, 0.5 (only when VIX rank > 0.5), 0.3
    """
    trades = []
    for i in range(60, len(daily) - hold):
        row = daily.iloc[i]

        if row['regime'] in ('SQUEEZE', 'WIDE'):
            continue
        if row['mts'] != 'GREEN':
            continue
        if not row['above_cloud']:
            continue
        if regime_filter == 'tight' and row['regime'] != 'NORMAL':
            continue
        if vix_rank_filter is not None:
            if pd.isna(row['vix_rank']) or row['vix_rank'] < vix_rank_filter:
                continue

        S_entry = row['nifty_close']
        iv_entry = row['vix_close'] / 100.0
        S_path = daily['nifty_close'].iloc[i+1 : i+1+hold].values
        iv_path = (daily['vix_close'].iloc[i+1 : i+1+hold].values) / 100.0

        if len(S_path) < hold:
            continue

        pnl = simulate(S_entry, S_path, iv_entry, iv_path, delta, hold, stop_mult)
        trades.append({'date': daily.index[i], 'pnl': pnl})

    if not trades:
        return None

    df = pd.DataFrame(trades)
    pnl = df['pnl'].values
    wins = (pnl > 0).sum()

    # Metrics
    equity = np.concatenate([[100000], 100000 + np.cumsum(pnl)])
    peak = np.maximum.accumulate(equity)
    dd = ((equity - peak) / peak).min() * 100
    total_return = (equity[-1] - 100000) / 100000 * 100

    trades_per_year = len(pnl) / 7.5
    sharpe = (pnl.mean() / pnl.std()) * np.sqrt(trades_per_year) if pnl.std() > 0 else 0
    calmar = (total_return / 7.5) / abs(dd) if dd != 0 else 0

    return {
        'n': len(pnl),
        'win_pct': 100 * wins / len(pnl),
        'avg_pnl': pnl.mean(),
        'total_pnl': pnl.sum(),
        'worst': pnl.min(),
        'total_return_pct': total_return,
        'max_dd_pct': dd,
        'sharpe': sharpe,
        'calmar': calmar,
    }


def main():
    daily = load_data()
    daily = compute_indicators(daily)
    print(f"Data: {len(daily)} daily bars, {daily.index[0].date()} to {daily.index[-1].date()}")
    print()

    # Parameter grids
    deltas = [0.20, 0.25, 0.30]
    holds = [10, 15, 20, 30]
    stops = [1.5, 2.0, 3.0, None]
    regimes = ['loose', 'tight']
    vix_filters = [None, 0.5]

    total = len(deltas) * len(holds) * len(stops) * len(regimes) * len(vix_filters)
    print(f"Running {total} combinations...")
    print()

    results = []
    for delta, hold, stop, regime, vix_f in itertools.product(deltas, holds, stops, regimes, vix_filters):
        res = run_single(daily, delta, hold, stop, regime, vix_f)
        if res is None:
            continue
        res['delta'] = delta
        res['hold'] = hold
        res['stop'] = stop if stop else 99
        res['regime'] = regime
        res['vix_filter'] = vix_f if vix_f else 0
        results.append(res)

    rdf = pd.DataFrame(results)
    rdf = rdf.sort_values('sharpe', ascending=False)

    # Top 10 by Sharpe
    print("=" * 100)
    print("TOP 10 BY SHARPE RATIO")
    print("=" * 100)
    top = rdf.head(10)
    for _, row in top.iterrows():
        print(f"delta={row['delta']:.2f}  hold={int(row['hold']):2d}  "
              f"stop={row['stop']:4.1f}  regime={row['regime']:5s}  vix>{row['vix_filter']:.1f}  "
              f"| n={int(row['n']):3d}  win%={row['win_pct']:.1f}  avg=Rs{row['avg_pnl']:+.0f}  "
              f"total=Rs{row['total_pnl']:+.0f}  Sharpe={row['sharpe']:.2f}  DD={row['max_dd_pct']:.1f}%")

    print()
    print("=" * 100)
    print("BOTTOM 10 BY SHARPE RATIO")
    print("=" * 100)
    bottom = rdf.tail(10)
    for _, row in bottom.iterrows():
        print(f"delta={row['delta']:.2f}  hold={int(row['hold']):2d}  "
              f"stop={row['stop']:4.1f}  regime={row['regime']:5s}  vix>{row['vix_filter']:.1f}  "
              f"| n={int(row['n']):3d}  win%={row['win_pct']:.1f}  avg=Rs{row['avg_pnl']:+.0f}  "
              f"total=Rs{row['total_pnl']:+.0f}  Sharpe={row['sharpe']:.2f}  DD={row['max_dd_pct']:.1f}%")

    print()
    print("=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)
    print(f"  Combinations tested:    {len(rdf)}")
    print(f"  Positive avg P&L:       {(rdf['avg_pnl'] > 0).sum()}")
    print(f"  Negative avg P&L:       {(rdf['avg_pnl'] <= 0).sum()}")
    print(f"  Sharpe > 0.5:           {(rdf['sharpe'] > 0.5).sum()}")
    print(f"  Sharpe > 1.0:           {(rdf['sharpe'] > 1.0).sum()}")
    print(f"  Median Sharpe:          {rdf['sharpe'].median():.3f}")
    print(f"  Median avg P&L:         Rs {rdf['avg_pnl'].median():+.0f}")
    print(f"  Best Sharpe:            {rdf['sharpe'].max():.3f}")
    print(f"  Worst Sharpe:           {rdf['sharpe'].min():.3f}")

    # Stop-loss comparison
    print()
    print("STOP-LOSS EFFECT (median across all other params):")
    for stop in sorted(rdf['stop'].unique()):
        sub = rdf[rdf['stop'] == stop]
        print(f"  stop={stop:.1f}x  n_combos={len(sub):3d}  median_Sharpe={sub['sharpe'].median():.3f}  "
              f"median_avg=Rs{sub['avg_pnl'].median():+.0f}  median_worst=Rs{sub['worst'].median():+.0f}")

    # VIX filter effect
    print()
    print("VIX RANK FILTER EFFECT:")
    for vf in sorted(rdf['vix_filter'].unique()):
        sub = rdf[rdf['vix_filter'] == vf]
        print(f"  vix>{vf:.1f}  n_combos={len(sub):3d}  median_Sharpe={sub['sharpe'].median():.3f}  "
              f"median_avg=Rs{sub['avg_pnl'].median():+.0f}")

    # Delta effect
    print()
    print("DELTA EFFECT:")
    for d in sorted(rdf['delta'].unique()):
        sub = rdf[rdf['delta'] == d]
        print(f"  delta={d:.2f}  median_Sharpe={sub['sharpe'].median():.3f}  median_avg=Rs{sub['avg_pnl'].median():+.0f}")

    # Hold effect
    print()
    print("HOLD PERIOD EFFECT:")
    for h in sorted(rdf['hold'].unique()):
        sub = rdf[rdf['hold'] == h]
        print(f"  hold={int(h):2d}  median_Sharpe={sub['sharpe'].median():.3f}  median_avg=Rs{sub['avg_pnl'].median():+.0f}")

    os.makedirs('results', exist_ok=True)
    rdf.to_csv('results/multisweep.csv', index=False)
    print()
    print(f"Saved detailed results to results/multisweep.csv")


if __name__ == '__main__':
    main()
