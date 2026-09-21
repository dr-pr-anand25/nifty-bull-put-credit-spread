"""
Parameter sweep of the bull put credit spread strategy using real VIX.
Tests robustness across hold, width, delta, and expiry combinations.

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


def simulate_bullput(S_entry, S_exit, iv_entry, iv_exit, width, short_delta, hold, expiry):
    r = RISK_FREE
    T_entry = expiry / 365.0
    T_exit = max((expiry - hold) / 365.0, 1/365.0)

    K_short = find_strike_for_delta(S_entry, short_delta, T_entry, r, iv_entry, 'put')
    K_long = K_short - width

    credit = (bs_price(S_entry, K_short, T_entry, r, iv_entry, 'put') -
              bs_price(S_entry, K_long, T_entry, r, iv_entry, 'put')) * LOT_SIZE
    debit = (bs_price(S_exit, K_short, T_exit, r, iv_exit, 'put') -
             bs_price(S_exit, K_long, T_exit, r, iv_exit, 'put')) * LOT_SIZE

    return credit - debit - COST_2_LEG


def run_one(daily, hold, width, short_delta, expiry):
    trades = []
    for i in range(60, len(daily) - hold):
        row = daily.iloc[i]
        # Bull put credit entry condition: regime in (EXPANDING/CONTRACTING/NORMAL) AND mts GREEN AND above cloud
        if row['regime'] == 'SQUEEZE' or row['regime'] == 'WIDE':
            continue
        if row['mts'] != 'GREEN':
            continue
        if not row['above_cloud']:
            continue

        S_entry = row['nifty_close']
        iv_entry = row['vix_close'] / 100.0
        exit_row = daily.iloc[i + hold]
        S_exit = exit_row['nifty_close']
        iv_exit = exit_row['vix_close'] / 100.0

        pnl = simulate_bullput(S_entry, S_exit, iv_entry, iv_exit, width, short_delta, hold, expiry)
        trades.append({'pnl': pnl})

    if not trades:
        return None

    tdf = pd.DataFrame(trades)
    return {
        'n': len(tdf),
        'win_pct': 100 * (tdf['pnl'] > 0).sum() / len(tdf),
        'avg_pnl': tdf['pnl'].mean(),
        'total_pnl': tdf['pnl'].sum(),
        'best': tdf['pnl'].max(),
        'worst': tdf['pnl'].min(),
    }


def main():
    daily = load_data()
    daily = compute_indicators(daily)

    results = []
    hold_list = [5, 10, 15, 20]
    width_list = [200, 300, 400, 500]
    delta_list = [0.15, 0.20, 0.25, 0.30]
    expiry_list = [15, 30, 45]

    print(f"Running sweep: {len(hold_list) * len(width_list) * len(delta_list) * len(expiry_list)} combinations")
    print(f"Data: {len(daily)} daily bars\n")

    for hold, width, delta, expiry in itertools.product(hold_list, width_list, delta_list, expiry_list):
        res = run_one(daily, hold, width, delta, expiry)
        if res is None:
            continue
        res['hold'] = hold
        res['width'] = width
        res['delta'] = delta
        res['expiry'] = expiry
        results.append(res)

    rdf = pd.DataFrame(results)
    rdf = rdf.sort_values('avg_pnl', ascending=False)

    print("TOP 15 PARAMETER COMBINATIONS (by avg P&L):")
    print(rdf.head(15).to_string(index=False))

    print("\n\nBOTTOM 15 PARAMETER COMBINATIONS:")
    print(rdf.tail(15).to_string(index=False))

    print("\n\nSUMMARY STATISTICS ACROSS ALL COMBINATIONS:")
    print(f"  Combinations tested:    {len(rdf)}")
    print(f"  Combinations positive:  {(rdf['avg_pnl'] > 0).sum()}")
    print(f"  Combinations negative:  {(rdf['avg_pnl'] <= 0).sum()}")
    print(f"  Median avg P&L:         Rs {rdf['avg_pnl'].median():+.0f}")
    print(f"  Mean avg P&L:           Rs {rdf['avg_pnl'].mean():+.0f}")
    print(f"  Best avg P&L:           Rs {rdf['avg_pnl'].max():+.0f}")
    print(f"  Worst avg P&L:          Rs {rdf['avg_pnl'].min():+.0f}")

    os.makedirs('results', exist_ok=True)
    rdf.to_csv('results/sweep_bullput.csv', index=False)
    print(f"\nSaved to results/sweep_bullput.csv")


if __name__ == '__main__':
    main()
