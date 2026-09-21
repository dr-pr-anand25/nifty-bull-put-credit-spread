"""
Backtest of the Signal Convergence Framework v3.2 decision logic
on NIFTY 50 index data.

Author: P. Rameswara Anand
"""
import os
import numpy as np
import pandas as pd

DATA_PATH = '/root/nifty_5min_extended.csv'
LOT_SIZE = 65
DELTA_ATM = 0.5
DELTA_SHORT_OTM = 0.17
ROUND_TRIP_COST_CREDIT = 190
ROUND_TRIP_COST_DEBIT = 150
HOLD_DAYS = 3


def load_daily_data():
    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    daily = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna()
    return daily


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

    # Regime classification with 5 buckets
    def classify(p, width, prev_width):
        if pd.isna(p):
            return 'NORMAL'
        if p <= 0.20:
            return 'SQUEEZE'
        if p >= 0.70:
            return 'WIDE'
        if width > prev_width:
            return 'EXPANDING'
        return 'CONTRACTING'

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
    if regime == 'WIDE':
        return 'NO_TRADE'
    return 'WAIT'


def simulate_structure(structure, entry_close, path_closes):
    if not path_closes:
        return 0.0

    exit_close = path_closes[-1]
    max_close = max(path_closes)
    min_close = min(path_closes)

    if structure == 'BULL_CALL_DEBIT':
        move = exit_close - entry_close
        return (move * DELTA_ATM * LOT_SIZE) - ROUND_TRIP_COST_DEBIT

    if structure == 'BEAR_PUT_DEBIT':
        move = entry_close - exit_close
        return (move * DELTA_ATM * LOT_SIZE) - ROUND_TRIP_COST_DEBIT

    if structure == 'BULL_PUT_CREDIT':
        threshold = entry_close * 0.985
        premium = entry_close * 0.005 * LOT_SIZE
        if min_close >= threshold:
            return premium - ROUND_TRIP_COST_CREDIT
        loss = (threshold - min_close) * DELTA_SHORT_OTM * LOT_SIZE
        return premium - loss - ROUND_TRIP_COST_CREDIT

    if structure == 'BEAR_CALL_CREDIT':
        threshold = entry_close * 1.015
        premium = entry_close * 0.005 * LOT_SIZE
        if max_close <= threshold:
            return premium - ROUND_TRIP_COST_CREDIT
        loss = (max_close - threshold) * DELTA_SHORT_OTM * LOT_SIZE
        return premium - loss - ROUND_TRIP_COST_CREDIT

    if structure == 'IRON_CONDOR':
        upper = entry_close * 1.015
        lower = entry_close * 0.985
        premium = entry_close * 0.008 * LOT_SIZE
        loss = 0.0
        if max_close > upper:
            loss += (max_close - upper) * DELTA_SHORT_OTM * LOT_SIZE
        if min_close < lower:
            loss += (lower - min_close) * DELTA_SHORT_OTM * LOT_SIZE
        return premium - loss - ROUND_TRIP_COST_CREDIT

    return 0.0


def run_backtest(daily):
    trades = []
    for i in range(60, len(daily) - HOLD_DAYS):
        row = daily.iloc[i]
        rec = recommend_structure(row['regime'], row['mts'], row['above_cloud'], row['below_cloud'])
        if rec in ('WAIT', 'NO_TRADE'):
            continue

        entry_close = row['close']
        path = daily['close'].iloc[i+1 : i+1+HOLD_DAYS].tolist()
        if not path:
            continue

        pnl = simulate_structure(rec, entry_close, path)
        trades.append({
            'date': daily.index[i],
            'regime': row['regime'],
            'mts': row['mts'],
            'structure': rec,
            'entry': entry_close,
            'pnl': pnl,
        })
    return pd.DataFrame(trades)


def summarize(trades):
    if trades.empty:
        print("No trades.")
        return

    print(f"\nTotal trades: {len(trades)}")
    print(f"Unique days:  {trades['date'].nunique()}")
    print()

    for struct in trades['structure'].unique():
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

    os.makedirs('results', exist_ok=True)
    trades.to_csv('results/backtest_switching_framework.csv', index=False)
    print(f"\nSaved to results/backtest_switching_framework.csv")


if __name__ == '__main__':
    daily = load_daily_data()
    print(f"Daily bars:  {len(daily)}")
    print(f"Date range:  {daily.index[0].date()} to {daily.index[-1].date()}")
    print()

    daily = compute_indicators(daily)
    trades = run_backtest(daily)
    summarize(trades)
