"""
paper5_core.py
Exact reimplementation of Paper 5's bull put credit spread backtest.
Matches /root/paper5/scripts/backtest_equity_stop.py byte-for-byte in
simulation logic and signal generation.

Used as Config A (rule-based baseline) in Paper 6's ablation study.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

# === Exact Paper 5 constants ===
NIFTY_PATH = '/root/nifty_combined_19yr.csv'
VIX_PATH   = '/root/TOB/data/historical/vix_daily_2019_2026.csv'

LOT_SIZE       = 65
RISK_FREE      = 0.065
STRIKE_STEP    = 50
COST_2_LEG     = 130
SLIPPAGE_PCT   = 0.005
SPREAD_WIDTH   = 500
DAYS_TO_EXPIRY = 30

DELTA     = 0.30
HOLD      = 20
STOP_MULT = 3.0

# Equity-curve stop (Paper 5's best config)
EQ_CONSEC_LOSSES = 3
EQ_PAUSE_DAYS    = 40


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
    merged['low']  = merged[['nifty_close', 'open']].min(axis=1)
    return merged.dropna()


def compute_indicators(d):
    d = d.copy()
    d['bb_mid']   = d['nifty_close'].rolling(20).mean()
    d['bb_std']   = d['nifty_close'].rolling(20).std()
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

    d['ema5']  = d['nifty_close'].ewm(span=5,  adjust=False).mean()
    d['ema13'] = d['nifty_close'].ewm(span=13, adjust=False).mean()
    diff = d['ema5'] - d['ema13']
    d['diff_change'] = diff.diff(3)

    d['mts'] = 'YELLOW'
    d.loc[(d['ema5'] > d['ema13']) & (d['diff_change'] > 0), 'mts'] = 'GREEN'
    d.loc[(d['ema5'] < d['ema13']) & (d['diff_change'] < 0), 'mts'] = 'RED'

    d['kijun']   = (d['low'].rolling(26).min()  + d['high'].rolling(26).max()) / 2
    d['tenkan']  = (d['high'].rolling(9).max()  + d['low'].rolling(9).min())  / 2
    d['senkouA'] = (d['tenkan'] + d['kijun']) / 2
    d['senkouB'] = (d['high'].rolling(52).max() + d['low'].rolling(52).min()) / 2
    d['cloud_top']    = d[['senkouA', 'senkouB']].max(axis=1)
    d['cloud_bottom'] = d[['senkouA', 'senkouB']].min(axis=1)
    d['above_cloud']  = d['nifty_close'] > d['cloud_top']

    # Store the exact Paper 5 entry condition
    d['entry_signal'] = (
        (~d['regime'].isin(['SQUEEZE', 'WIDE'])) &
        (d['mts'] == 'GREEN') &
        d['above_cloud']
    )
    return d


def simulate(S_entry, S_path, iv_entry, iv_path):
    """
    Exact copy of Paper 5's simulate() function.
    Returns P&L per lot in rupees.
    """
    r = RISK_FREE
    T_entry = DAYS_TO_EXPIRY / 365.0
    K_short = find_strike_for_delta(S_entry, DELTA, T_entry, r, iv_entry, 'put')
    K_long  = K_short - SPREAD_WIDTH

    short_entry = bs_price(S_entry, K_short, T_entry, r, iv_entry, 'put')
    long_entry  = bs_price(S_entry, K_long,  T_entry, r, iv_entry, 'put')
    credit  = (short_entry - long_entry) * LOT_SIZE
    credit -= COST_2_LEG
    credit -= (short_entry + long_entry) * SLIPPAGE_PCT * LOT_SIZE

    for j, (S_j, iv_j) in enumerate(zip(S_path, iv_path)):
        T_j = max((DAYS_TO_EXPIRY - (j + 1)) / 365.0, 1/365.0)
        short_j = bs_price(S_j, K_short, T_j, r, iv_j, 'put')
        long_j  = bs_price(S_j, K_long,  T_j, r, iv_j, 'put')
        spread_val = (short_j - long_j) * LOT_SIZE
        if spread_val >= short_entry * STOP_MULT * LOT_SIZE:
            exit_cost  = spread_val + COST_2_LEG
            exit_cost += (short_j + long_j) * SLIPPAGE_PCT * LOT_SIZE
            return credit - exit_cost

    S_exit  = S_path[-1]
    iv_exit = iv_path[-1]
    T_exit  = max((DAYS_TO_EXPIRY - HOLD) / 365.0, 1/365.0)
    short_exit = bs_price(S_exit, K_short, T_exit, r, iv_exit, 'put')
    long_exit  = bs_price(S_exit, K_long,  T_exit, r, iv_exit, 'put')
    exit_cost  = (short_exit - long_exit) * LOT_SIZE
    exit_cost += COST_2_LEG
    exit_cost += (short_exit + long_exit) * SLIPPAGE_PCT * LOT_SIZE
    return credit - exit_cost


def run_paper5_backtest(daily=None, use_eq_stop=False):
    """
    Exact copy of Paper 5's run_backtest() function.
    If use_eq_stop=True, applies the 3-loss / 40-day pause.
    Returns a DataFrame with columns: date, pnl, nifty, vix, bb_w_pct, regime, mts.
    """
    if daily is None:
        daily = compute_indicators(load_data())

    trades = []
    consecutive_losses = 0
    pause_until_idx = -1

    losses_to_pause = EQ_CONSEC_LOSSES if use_eq_stop else None
    pause_days      = EQ_PAUSE_DAYS    if use_eq_stop else None

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

        S_entry  = row['nifty_close']
        iv_entry = row['vix_close'] / 100.0
        S_path   = daily['nifty_close'].iloc[i+1 : i+1+HOLD].values
        iv_path  = (daily['vix_close'].iloc[i+1 : i+1+HOLD].values) / 100.0

        if len(S_path) < HOLD:
            continue

        pnl = simulate(S_entry, S_path, iv_entry, iv_path)
        trades.append({
            'date':     daily.index[i],
            'pnl':      pnl,
            'nifty':    S_entry,
            'vix':      row['vix_close'],
            'bb_w_pct': row['bb_w_pct'],
            'regime':   row['regime'],
            'mts':      row['mts'],
        })

        if pnl > 0:
            consecutive_losses = 0
        else:
            consecutive_losses += 1
            if losses_to_pause is not None and consecutive_losses >= losses_to_pause:
                pause_until_idx = i + pause_days
                consecutive_losses = 0

    return pd.DataFrame(trades)
