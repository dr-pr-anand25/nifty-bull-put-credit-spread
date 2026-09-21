"""
Corrected backtest of the Signal Convergence Framework v3.2 using
Black-Scholes option pricing on 19 years of NIFTY daily data.

Fixes from previous version:
  - Uses real BS pricing for each option leg (entry and exit)
  - Bounds maximum loss by the spread width
  - Computes P&L from premium changes, not index moves

Author: P. Rameswara Anand
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

DATA_PATH = '/root/nifty_combined_19yr.csv'
LOT_SIZE = 65
HOLD_DAYS = 3

# Assumptions for option pricing
RISK_FREE = 0.065           # India 10-year yield proxy
DEFAULT_IV = 0.15           # fallback IV if VIX unavailable
DAYS_TO_EXPIRY = 7          # weekly option assumption
STRIKE_STEP = 50            # NIFTY strike increment

# Transaction costs
COST_2_LEG = 130            # rupees round trip for 2-leg structure
COST_4_LEG = 220            # rupees round trip for 4-leg structure


def bs_price(S, K, T, r, sigma, option_type):
    """Black-Scholes European option price."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)
        return intrinsic
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, option_type):
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1


def find_strike_for_delta(S, target_delta, T, r, sigma, option_type):
    """Find the strike that produces approximately the target delta."""
    # Start from ATM and step out
    strike = round(S / STRIKE_STEP) * STRIKE_STEP
    for _ in range(20):
        d = bs_delta(S, strike, T, r, sigma, option_type)
        if option_type == 'call':
            if d > target_delta:
                strike += STRIKE_STEP
            else:
                break
        else:  # put, delta is negative
            if abs(d) > target_delta:
                strike -= STRIKE_STEP
            else:
                break
    return strike


def load_daily_data():
    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df['open'] = df['close'].shift(1)
    df['high'] = df[['close', 'open']].max(axis=1)
    df['low'] = df[['close', 'open']].min(axis=1)
    df = df.dropna()

    # Approximate VIX regime using rolling realized volatility
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['realized_vol'] = df['log_ret'].rolling(20).std() * np.sqrt(252)
    df['iv_proxy'] = df['realized_vol'].fillna(DEFAULT_IV).clip(0.08, 0.60)
    return df


def compute_indicators(daily):
    d = daily.copy()

    d['bb_mid'] = d['close'].rolling(20).mean()
    d['bb_std'] = d['close'].rolling(20).std()
    d['bb_upper'] = d['bb_mid'] + 2 * d['bb_std']
    d['bb_lower'] = d['bb_mid'] - 2 * d['bb_std']
    d['bb_width'] = (d['bb_upper'] - d['bb_lower']) / d['bb_mid']

    d['bb_w_min'] = d['bb_width'].rolling(60).min()
    d['bb_w_max'] = d['bb_width'].rolling(60).max()
    d['bb_w_pct'] = (d['bb_width'] - d['bb_w_min']) / (d['bb_w_max'] - d['bb_w_min'] + 1e-10)

    def classify(p, width, prev_width):
        if pd.isna(p):
            return 'NORMAL'
        if p <= 0.20:
            return 'SQUEEZE'
        if p >= 0.70:
            return 'WIDE'
        return 'EXPANDING' if width > prev_width else 'CONTRACTING'

    prev = d['bb_width'].shift(1)
    d['regime'] = [classify(p, w, pw) for p, w, pw in zip(d['bb_w_pct'], d['bb_width'], prev)]

    d['ema5'] = d['close'].ewm(span=5, adjust=False).mean()
    d['ema13'] = d['close'].ewm(span=13, adjust=False).mean()
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

    d['above_cloud'] = d['close'] > d['cloud_top']
    d['below_cloud'] = d['close'] < d['cloud_bottom']

    return d


def recommend_structure(regime, mts, above, below):
    if regime == 'SQUEEZE':
        if mts == 'GREEN' and above:
            return 'BULL_CALL_DEBIT'
        if mts == 'RED' and below:
            return 'BEAR_PUT_DEBIT'
        return 'WAIT'
    if regime in ('EXPANDING', 'CONTRACTING'):
        if mts == 'GREEN' and above:
            return 'BULL_PUT_CREDIT'
        if mts == 'RED' and below:
            return 'BEAR_CALL_CREDIT'
        if mts == 'YELLOW':
            return 'IRON_CONDOR'
        return 'WAIT'
    return 'WAIT'


def simulate_spread_bs(structure, S_entry, S_exit, iv_entry, iv_exit, T):
    """
    Compute P&L of a spread using BS pricing for each leg at entry and exit.
    Returns P&L in rupees per lot, after transaction costs.
    """
    r = RISK_FREE
    width = 150  # spread width in index points

    # Entry premiums
    if structure == 'BULL_PUT_CREDIT':
        # Sell put at delta 0.17, buy put further out
        K_short = find_strike_for_delta(S_entry, 0.17, T, r, iv_entry, 'put')
        K_long = K_short - width
        short_entry = bs_price(S_entry, K_short, T, r, iv_entry, 'put')
        long_entry = bs_price(S_entry, K_long, T, r, iv_entry, 'put')
        net_credit = (short_entry - long_entry) * LOT_SIZE
        # Exit premiums
        short_exit = bs_price(S_exit, K_short, T, r, iv_exit, 'put')
        long_exit = bs_price(S_exit, K_long, T, r, iv_exit, 'put')
        net_debit = (short_exit - long_exit) * LOT_SIZE
        pnl = net_credit - net_debit - COST_2_LEG
        return pnl

    if structure == 'BEAR_CALL_CREDIT':
        K_short = find_strike_for_delta(S_entry, 0.17, T, r, iv_entry, 'call')
        K_long = K_short + width
        short_entry = bs_price(S_entry, K_short, T, r, iv_entry, 'call')
        long_entry = bs_price(S_entry, K_long, T, r, iv_entry, 'call')
        net_credit = (short_entry - long_entry) * LOT_SIZE
        short_exit = bs_price(S_exit, K_short, T, r, iv_exit, 'call')
        long_exit = bs_price(S_exit, K_long, T, r, iv_exit, 'call')
        net_debit = (short_exit - long_exit) * LOT_SIZE
        pnl = net_credit - net_debit - COST_2_LEG
        return pnl

    if structure == 'BULL_CALL_DEBIT':
        # Buy call at ATM, sell call further up
        K_long = round(S_entry / STRIKE_STEP) * STRIKE_STEP
        K_short = K_long + width
        long_entry = bs_price(S_entry, K_long, T, r, iv_entry, 'call')
        short_entry = bs_price(S_entry, K_short, T, r, iv_entry, 'call')
        net_debit = (long_entry - short_entry) * LOT_SIZE
        long_exit = bs_price(S_exit, K_long, T, r, iv_exit, 'call')
        short_exit = bs_price(S_exit, K_short, T, r, iv_exit, 'call')
        exit_credit = (long_exit - short_exit) * LOT_SIZE
        pnl = exit_credit - net_debit - COST_2_LEG
        return pnl

    if structure == 'BEAR_PUT_DEBIT':
        K_long = round(S_entry / STRIKE_STEP) * STRIKE_STEP
        K_short = K_long - width
        long_entry = bs_price(S_entry, K_long, T, r, iv_entry, 'put')
        short_entry = bs_price(S_entry, K_short, T, r, iv_entry, 'put')
        net_debit = (long_entry - short_entry) * LOT_SIZE
        long_exit = bs_price(S_exit, K_long, T, r, iv_exit, 'put')
        short_exit = bs_price(S_exit, K_short, T, r, iv_exit, 'put')
        exit_credit = (long_exit - short_exit) * LOT_SIZE
        pnl = exit_credit - net_debit - COST_2_LEG
        return pnl

    if structure == 'IRON_CONDOR':
        # Sell put at delta 0.17, buy put width below
        K_put_short = find_strike_for_delta(S_entry, 0.17, T, r, iv_entry, 'put')
        K_put_long = K_put_short - width
        # Sell call at delta 0.17, buy call width above
        K_call_short = find_strike_for_delta(S_entry, 0.17, T, r, iv_entry, 'call')
        K_call_long = K_call_short + width

        # Entry premiums
        ps_e = bs_price(S_entry, K_put_short, T, r, iv_entry, 'put')
        pl_e = bs_price(S_entry, K_put_long, T, r, iv_entry, 'put')
        cs_e = bs_price(S_entry, K_call_short, T, r, iv_entry, 'call')
        cl_e = bs_price(S_entry, K_call_long, T, r, iv_entry, 'call')
        net_credit = (ps_e + cs_e - pl_e - cl_e) * LOT_SIZE

        # Exit premiums
        ps_x = bs_price(S_exit, K_put_short, T, r, iv_exit, 'put')
        pl_x = bs_price(S_exit, K_put_long, T, r, iv_exit, 'put')
        cs_x = bs_price(S_exit, K_call_short, T, r, iv_exit, 'call')
        cl_x = bs_price(S_exit, K_call_long, T, r, iv_exit, 'call')
        net_debit = (ps_x + cs_x - pl_x - cl_x) * LOT_SIZE

        pnl = net_credit - net_debit - COST_4_LEG
        return pnl

    return 0.0


def run_backtest(daily):
    trades = []
    T_entry = DAYS_TO_EXPIRY / 365.0

    for i in range(60, len(daily) - HOLD_DAYS):
        row = daily.iloc[i]
        rec = recommend_structure(row['regime'], row['mts'], row['above_cloud'], row['below_cloud'])
        if rec == 'WAIT':
            continue

        S_entry = row['close']
        iv_entry = row['iv_proxy']

        exit_row = daily.iloc[i + HOLD_DAYS]
        S_exit = exit_row['close']
        iv_exit = exit_row['iv_proxy']

        # Time to expiry shrinks by HOLD_DAYS
        T_exit = max((DAYS_TO_EXPIRY - HOLD_DAYS) / 365.0, 1/365.0)

        pnl = simulate_spread_bs(rec, S_entry, S_exit, iv_entry, iv_exit, T_entry)

        trades.append({
            'date': daily.index[i],
            'regime': row['regime'],
            'mts': row['mts'],
            'structure': rec,
            'spot_entry': S_entry,
            'spot_exit': S_exit,
            'iv_entry': iv_entry,
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
        print("PASS: Framework produces positive expected P&L")
    else:
        print("FAIL: Framework produces negative expected P&L")

    trades['year'] = trades['date'].dt.year
    yearly = trades.groupby('year')['pnl'].agg(['count', 'sum', 'mean'])
    print()
    print("Yearly breakdown:")
    print(yearly.to_string())

    os.makedirs('results', exist_ok=True)
    trades.to_csv('results/backtest_19yr_bs.csv', index=False)
    print(f"\nSaved to results/backtest_19yr_bs.csv")


if __name__ == '__main__':
    daily = load_daily_data()
    print(f"Daily bars:  {len(daily)}")
    print(f"Date range:  {daily.index[0].date()} to {daily.index[-1].date()}")
    print()

    daily = compute_indicators(daily)
    trades = run_backtest(daily)
    summarize(trades)
