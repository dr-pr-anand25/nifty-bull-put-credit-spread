"""
Test equity-curve stop: pause strategy after N consecutive losses.

Compares baseline (no pause) against 4 equity-curve stop configs.
Uses optimal parameters: delta=0.30, hold=20, stop=3.0x, no VIX filter.

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

DELTA = 0.30
HOLD = 20
STOP_MULT = 3.0


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

    for j, (S_j, iv_j) in enumerate(zip(S_path, iv_path)):
        T_j = max((DAYS_TO_EXPIRY - (j + 1)) / 365.0, 1/365.0)
        short_j = bs_price(S_j, K_short, T_j, r, iv_j, 'put')
        long_j = bs_price(S_j, K_long, T_j, r, iv_j, 'put')
        spread_val = (short_j - long_j) * LOT_SIZE
        if spread_val >= short_entry * STOP_MULT * LOT_SIZE:
            exit_cost = spread_val + COST_2_LEG + (short_j + long_j) * SLIPPAGE_PCT * LOT_SIZE
            return credit - exit_cost

    S_exit = S_path[-1]
    iv_exit = iv_path[-1]
    T_exit = max((DAYS_TO_EXPIRY - HOLD) / 365.0, 1/365.0)
    short_exit = bs_price(S_exit, K_short, T_exit, r, iv_exit, 'put')
    long_exit = bs_price(S_exit, K_long, T_exit, r, iv_exit, 'put')
    exit_cost = (short_exit - long_exit) * LOT_SIZE
    exit_cost += COST_2_LEG
    exit_cost += (short_exit + long_exit) * SLIPPAGE_PCT * LOT_SIZE
    return credit - exit_cost


def run_backtest(daily, losses_to_pause=None, pause_days=None):
    """Backtest with optional equity-curve stop.

    If losses_to_pause and pause_days are set, the strategy pauses for
    pause_days after losses_to_pause consecutive losing trades.
    """
    trades = []
    consecutive_losses = 0
    pause_until_idx = -1

    for i in range(60, len(daily) - HOLD):
        if i < pause_until_idx:
            continue

        row = daily.iloc[i]

        if row['regime'] in ('SQUEEZE', 'WIDE'):
            continue
        if row['mts'] != 'GREEN':
            continue
        if not row['above_cloud']:
            continue

        S_entry = row['nifty_close']
        iv_entry = row['vix_close'] / 100.0
        S_path = daily['nifty_close'].iloc[i+1 : i+1+HOLD].values
        iv_path = (daily['vix_close'].iloc[i+1 : i+1+HOLD].values) / 100.0

        if len(S_path) < HOLD:
            continue

        pnl = simulate(S_entry, S_path, iv_entry, iv_path)
        trades.append({
            'date': daily.index[i],
            'pnl': pnl,
        })

        # Update equity-curve stop state
        if pnl > 0:
            consecutive_losses = 0
        else:
            consecutive_losses += 1
            if losses_to_pause is not None and consecutive_losses >= losses_to_pause:
                pause_until_idx = i + pause_days
                consecutive_losses = 0

    return pd.DataFrame(trades)


def compute_metrics(trades, label):
    if trades.empty:
        print(f"{label}: no trades")
        return None

    pnl = trades['pnl'].values
    n = len(pnl)
    wins = (pnl > 0).sum()

    equity = np.concatenate([[100000], 100000 + np.cumsum(pnl)])
    peak = np.maximum.accumulate(equity)
    dd = ((equity - peak) / peak).min() * 100
    total_ret = (equity[-1] - 100000) / 100000 * 100

    trades_per_year = n / 7.0  # approx
    sharpe = (pnl.mean() / pnl.std()) * np.sqrt(trades_per_year) if pnl.std() > 0 else 0
    annual_return = total_ret / 7.0
    calmar = annual_return / abs(dd) if dd != 0 else 0

    print(f"\n{label}")
    print(f"  Trades:         {n}")
    print(f"  Win rate:       {100*wins/n:.1f}%")
    print(f"  Avg P&L:        Rs {pnl.mean():+.0f}")
    print(f"  Total P&L:      Rs {pnl.sum():+.0f}")
    print(f"  Total return:   {total_ret:+.1f}%")
    print(f"  Max DD:         {dd:.1f}%")
    print(f"  Sharpe:         {sharpe:.3f}")
    print(f"  Calmar:         {calmar:.3f}")
    print(f"  Worst trade:    Rs {pnl.min():+.0f}")

    return {
        'n': n,
        'win_pct': 100*wins/n,
        'avg': pnl.mean(),
        'total': pnl.sum(),
        'total_ret': total_ret,
        'dd': dd,
        'sharpe': sharpe,
        'calmar': calmar,
    }


if __name__ == '__main__':
    daily = load_data()
    daily = compute_indicators(daily)
    print(f"Data: {len(daily)} daily bars, {daily.index[0].date()} to {daily.index[-1].date()}")

    results = {}

    # Baseline
    print()
    print("=" * 70)
    print("BASELINE (no equity curve stop)")
    print("=" * 70)
    trades = run_backtest(daily, losses_to_pause=None, pause_days=None)
    results['baseline'] = compute_metrics(trades, "Baseline")

    # Test configs
    configs = [
        (3, 20),
        (3, 40),
        (4, 20),
        (5, 30),
    ]

    for losses, pause in configs:
        label = f"Pause after {losses} losses, resume after {pause} days"
        print()
        print("=" * 70)
        print(label)
        print("=" * 70)
        trades = run_backtest(daily, losses_to_pause=losses, pause_days=pause)
        results[f'{losses}_{pause}'] = compute_metrics(trades, label)

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<40} {'Trades':>7} {'Win%':>7} {'Avg':>8} {'Total':>10} {'Sharpe':>8} {'Calmar':>8}")
    for k, v in results.items():
        if v:
            label = 'Baseline' if k == 'baseline' else f'Pause {k}'
            print(f"{label:<40} {v['n']:>7} {v['win_pct']:>7.1f} {v['avg']:>+8.0f} {v['total']:>+10.0f} {v['sharpe']:>8.3f} {v['calmar']:>8.3f}")

    os.makedirs('results', exist_ok=True)
    pd.DataFrame([{'config': k, **v} for k, v in results.items() if v]).to_csv(
        'results/equity_stop_test.csv', index=False)
    print(f"\nSaved results/equity_stop_test.csv")
