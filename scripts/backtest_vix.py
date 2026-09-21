"""
Backtest of the regime-switching framework using REAL India VIX as the
implied volatility input for Black-Scholes option pricing.

Covers 2019-01 to 2026-07 (~7.5 years, ~1,750 trading days).

Author: P. Rameswara Anand
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

NIFTY_PATH = '/root/nifty_combined_19yr.csv'
VIX_PATH = '/root/TOB/data/historical/vix_daily_2019_2026.csv'

LOT_SIZE = 65
HOLD_DAYS = 10
SPREAD_WIDTH = 300
SHORT_DELTA = 0.25
RISK_FREE = 0.065
DAYS_TO_EXPIRY = 30
STRIKE_STEP = 50
COST_2_LEG = 130


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


def load_merged_data():
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

    # Merge on date, keep only dates where both exist
    merged = nifty.join(vix, how='inner')
    merged = merged.dropna()

    # Approximate OHLC
    merged['open'] = merged['nifty_close'].shift(1)
    merged['high'] = merged[['nifty_close', 'open']].max(axis=1)
    merged['low'] = merged[['nifty_close', 'open']].min(axis=1)
    merged = merged.dropna()
    return merged


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

    def classify(p, w, pw):
        if pd.isna(p): return 'NORMAL'
        if p <= 0.20: return 'SQUEEZE'
        if p >= 0.70: return 'WIDE'
        return 'EXPANDING' if w > pw else 'CONTRACTING'

    prev = d['bb_width'].shift(1)
    d['regime'] = [classify(p, w, pw) for p, w, pw in zip(d['bb_w_pct'], d['bb_width'], prev)]

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
    d['below_cloud'] = d['nifty_close'] < d['cloud_bottom']
    return d


def recommend_structure(regime, mts, above, below):
    if regime == 'SQUEEZE':
        if mts == 'GREEN' and above: return 'BULL_CALL_DEBIT'
        if mts == 'RED' and below:   return 'BEAR_PUT_DEBIT'
        return 'WAIT'
    if regime in ('EXPANDING', 'CONTRACTING'):
        if mts == 'GREEN' and above: return 'BULL_PUT_CREDIT'
        if mts == 'RED' and below:   return 'BEAR_CALL_CREDIT'
        return 'WAIT'
    return 'WAIT'


def simulate_spread_bs(structure, S_entry, S_exit, iv_entry, iv_exit):
    r = RISK_FREE
    width = SPREAD_WIDTH
    T_entry = DAYS_TO_EXPIRY / 365.0
    T_exit = max((DAYS_TO_EXPIRY - HOLD_DAYS) / 365.0, 1/365.0)

    if structure == 'BULL_PUT_CREDIT':
        K_short = find_strike_for_delta(S_entry, SHORT_DELTA, T_entry, r, iv_entry, 'put')
        K_long = K_short - width
        credit = (bs_price(S_entry, K_short, T_entry, r, iv_entry, 'put') -
                  bs_price(S_entry, K_long, T_entry, r, iv_entry, 'put')) * LOT_SIZE
        debit = (bs_price(S_exit, K_short, T_exit, r, iv_exit, 'put') -
                 bs_price(S_exit, K_long, T_exit, r, iv_exit, 'put')) * LOT_SIZE
        return credit - debit - COST_2_LEG

    if structure == 'BEAR_CALL_CREDIT':
        K_short = find_strike_for_delta(S_entry, SHORT_DELTA, T_entry, r, iv_entry, 'call')
        K_long = K_short + width
        credit = (bs_price(S_entry, K_short, T_entry, r, iv_entry, 'call') -
                  bs_price(S_entry, K_long, T_entry, r, iv_entry, 'call')) * LOT_SIZE
        debit = (bs_price(S_exit, K_short, T_exit, r, iv_exit, 'call') -
                 bs_price(S_exit, K_long, T_exit, r, iv_exit, 'call')) * LOT_SIZE
        return credit - debit - COST_2_LEG

    if structure == 'BULL_CALL_DEBIT':
        K_long = round(S_entry / STRIKE_STEP) * STRIKE_STEP
        K_short = K_long + width
        debit_in = (bs_price(S_entry, K_long, T_entry, r, iv_entry, 'call') -
                    bs_price(S_entry, K_short, T_entry, r, iv_entry, 'call')) * LOT_SIZE
        credit_out = (bs_price(S_exit, K_long, T_exit, r, iv_exit, 'call') -
                      bs_price(S_exit, K_short, T_exit, r, iv_exit, 'call')) * LOT_SIZE
        return credit_out - debit_in - COST_2_LEG

    if structure == 'BEAR_PUT_DEBIT':
        K_long = round(S_entry / STRIKE_STEP) * STRIKE_STEP
        K_short = K_long - width
        debit_in = (bs_price(S_entry, K_long, T_entry, r, iv_entry, 'put') -
                    bs_price(S_entry, K_short, T_entry, r, iv_entry, 'put')) * LOT_SIZE
        credit_out = (bs_price(S_exit, K_long, T_exit, r, iv_exit, 'put') -
                      bs_price(S_exit, K_short, T_exit, r, iv_exit, 'put')) * LOT_SIZE
        return credit_out - debit_in - COST_2_LEG

    return 0.0


def run_backtest(daily):
    trades = []
    for i in range(60, len(daily) - HOLD_DAYS):
        row = daily.iloc[i]
        rec = recommend_structure(row['regime'], row['mts'], row['above_cloud'], row['below_cloud'])
        if rec == 'WAIT':
            continue

        S_entry = row['nifty_close']
        iv_entry = row['vix_close'] / 100.0   # VIX as IV input

        exit_row = daily.iloc[i + HOLD_DAYS]
        S_exit = exit_row['nifty_close']
        iv_exit = exit_row['vix_close'] / 100.0

        pnl = simulate_spread_bs(rec, S_entry, S_exit, iv_entry, iv_exit)
        trades.append({
            'date': daily.index[i],
            'regime': row['regime'],
            'mts': row['mts'],
            'structure': rec,
            'spot_entry': S_entry,
            'spot_exit': S_exit,
            'vix_entry': row['vix_close'],
            'pnl': pnl,
        })
    return pd.DataFrame(trades)


def summarize(trades):
    if trades.empty:
        print("No trades.")
        return

    print(f"\nTotal trades: {len(trades)}")
    print()

    for struct in sorted(trades['structure'].unique()):
        sub = trades[trades['structure'] == struct]
        wins = (sub['pnl'] > 0).sum()
        print(f"  {struct}")
        print(f"    n={len(sub)},  wins={wins},  win%={100*wins/len(sub):.1f}")
        print(f"    avg PnL = Rs {sub['pnl'].mean():+.0f},  total = Rs {sub['pnl'].sum():+.0f}")
        print(f"    best = Rs {sub['pnl'].max():+.0f},  worst = Rs {sub['pnl'].min():+.0f}")
        print()

    total = trades['pnl'].sum()
    avg = trades['pnl'].mean()
    win_rate = 100 * (trades['pnl'] > 0).sum() / len(trades)
    print("=" * 60)
    print(f"OVERALL: n={len(trades)}, win%={win_rate:.1f}, avg=Rs {avg:+.0f}, total=Rs {total:+.0f}")
    print("=" * 60)

    if avg > 0:
        print("PASS: Framework produces positive expected P&L with real VIX")
    else:
        print("FAIL: Framework produces negative expected P&L even with real VIX")

    trades['year'] = trades['date'].dt.year
    yearly = trades.groupby('year')['pnl'].agg(['count', 'sum', 'mean'])
    print()
    print("Yearly breakdown:")
    print(yearly.to_string())

    os.makedirs('results', exist_ok=True)
    trades.to_csv('results/backtest_vix.csv', index=False)
    print(f"\nSaved to results/backtest_vix.csv")


if __name__ == '__main__':
    daily = load_merged_data()
    print(f"Daily bars:  {len(daily)}")
    print(f"Date range:  {daily.index[0].date()} to {daily.index[-1].date()}")
    print(f"VIX range:   {daily['vix_close'].min():.1f} to {daily['vix_close'].max():.1f}")
    print()

    daily = compute_indicators(daily)
    trades = run_backtest(daily)
    summarize(trades)
