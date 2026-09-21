"""
Extended analysis of the bull put credit spread strategy.

Adds:
  - 19-year backtest (2007-2026)
  - Realistic slippage (+0.5% per leg)
  - Stop-loss on spread value
  - Drawdown and Sharpe analysis
  - NIFTY buy-and-hold benchmark

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
SLIPPAGE_PCT = 0.005       # 0.5% per leg extra slippage
DEFAULT_IV = 0.15          # fallback IV for pre-VIX years

# Best configuration from sweep
HOLD = 20
WIDTH = 500
DELTA = 0.30
EXPIRY = 30
STOP_MULTIPLE = 2.0        # exit if spread value doubles (loss = 1x credit)


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

    # Fill pre-VIX period with rolling realized vol
    merged['log_ret'] = np.log(merged['nifty_close'] / merged['nifty_close'].shift(1))
    merged['realized_vol'] = merged['log_ret'].rolling(20).std() * np.sqrt(252)
    merged['iv_proxy'] = (merged['vix_close'] / 100.0).fillna(merged['realized_vol']).fillna(DEFAULT_IV)
    merged['iv_proxy'] = merged['iv_proxy'].clip(0.08, 0.80)

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


def simulate_with_stoploss(S_entry, S_exit_path, iv_entry, iv_path, width, delta, hold, expiry, use_stop):
    """
    Simulate a bull put credit spread with optional intra-hold stop-loss.
    S_exit_path is the path of closes during the hold period.
    iv_path is the corresponding IV values.
    """
    r = RISK_FREE
    T_entry = expiry / 365.0
    K_short = find_strike_for_delta(S_entry, delta, T_entry, r, iv_entry, 'put')
    K_long = K_short - width

    short_entry = bs_price(S_entry, K_short, T_entry, r, iv_entry, 'put')
    long_entry = bs_price(S_entry, K_long, T_entry, r, iv_entry, 'put')
    credit = (short_entry - long_entry) * LOT_SIZE
    credit -= COST_2_LEG
    # Slippage on entry
    credit -= (short_entry + long_entry) * SLIPPAGE_PCT * LOT_SIZE

    if use_stop:
        # Check each bar for stop trigger
        for j, (S_j, iv_j) in enumerate(zip(S_exit_path, iv_path)):
            T_j = max((expiry - (j + 1)) / 365.0, 1/365.0)
            short_j = bs_price(S_j, K_short, T_j, r, iv_j, 'put')
            long_j = bs_price(S_j, K_long, T_j, r, iv_j, 'put')
            spread_val = (short_j - long_j) * LOT_SIZE
            # Stop trigger: spread has doubled (loss >= credit collected at entry)
            if spread_val >= short_entry * STOP_MULTIPLE * LOT_SIZE:
                exit_debit = spread_val
                exit_debit += COST_2_LEG
                exit_debit += (short_j + long_j) * SLIPPAGE_PCT * LOT_SIZE
                return credit - exit_debit

    # Time exit at end of hold
    S_exit = S_exit_path[-1]
    iv_exit = iv_path[-1]
    T_exit = max((expiry - hold) / 365.0, 1/365.0)
    short_exit = bs_price(S_exit, K_short, T_exit, r, iv_exit, 'put')
    long_exit = bs_price(S_exit, K_long, T_exit, r, iv_exit, 'put')
    exit_debit = (short_exit - long_exit) * LOT_SIZE
    exit_debit += COST_2_LEG
    exit_debit += (short_exit + long_exit) * SLIPPAGE_PCT * LOT_SIZE
    return credit - exit_debit


def run_backtest(daily, use_stop=False, regime_strict=False):
    trades = []
    for i in range(60, len(daily) - HOLD):
        row = daily.iloc[i]
        # Entry conditions
        if row['regime'] in ('SQUEEZE', 'WIDE'):
            continue
        if row['mts'] != 'GREEN':
            continue
        if not row['above_cloud']:
            continue
        if regime_strict and row['regime'] != 'NORMAL':
            continue

        S_entry = row['nifty_close']
        iv_entry = row['iv_proxy']
        S_path = daily['nifty_close'].iloc[i+1 : i+1+HOLD].values
        iv_path = daily['iv_proxy'].iloc[i+1 : i+1+HOLD].values

        if len(S_path) < HOLD:
            continue

        pnl = simulate_with_stoploss(S_entry, S_path, iv_entry, iv_path, WIDTH, DELTA, HOLD, EXPIRY, use_stop)
        trades.append({
            'date': daily.index[i],
            'year': daily.index[i].year,
            'pnl': pnl,
        })
    return pd.DataFrame(trades)


def compute_metrics(trades, initial_capital=100000):
    """Compute performance metrics on a per-trade P&L series."""
    if trades.empty:
        return None

    pnl = trades['pnl'].values
    n = len(pnl)
    wins = (pnl > 0).sum()

    # Sequential equity curve
    equity = initial_capital + np.cumsum(pnl)
    equity = np.concatenate([[initial_capital], equity])

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min() * 100

    # Returns
    total_return = (equity[-1] - initial_capital) / initial_capital * 100
    avg_return = pnl.mean() / initial_capital * 100

    # Sharpe (assuming 1 trade = 1 period, annualize with trade frequency)
    trades_per_year = n / 7.5  # roughly
    if pnl.std() > 0:
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(trades_per_year)
    else:
        sharpe = 0

    # Calmar ratio
    annual_return = total_return / 7.5
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

    return {
        'n_trades': n,
        'win_pct': 100 * wins / n,
        'avg_pnl': pnl.mean(),
        'total_pnl': pnl.sum(),
        'best': pnl.max(),
        'worst': pnl.min(),
        'total_return_pct': total_return,
        'max_drawdown_pct': max_dd,
        'sharpe': sharpe,
        'calmar': calmar,
    }


def benchmark_buyhold(daily, start_date, end_date):
    """NIFTY buy-and-hold benchmark."""
    d = daily.loc[start_date:end_date]
    if len(d) < 2:
        return None
    start_price = d['nifty_close'].iloc[0]
    end_price = d['nifty_close'].iloc[-1]
    total_return = (end_price - start_price) / start_price * 100

    # Max drawdown on index
    peak = d['nifty_close'].cummax()
    dd = ((d['nifty_close'] - peak) / peak).min() * 100

    # Annualized
    years = (d.index[-1] - d.index[0]).days / 365.0
    annual_return = ((end_price / start_price) ** (1/years) - 1) * 100

    return {
        'start_price': start_price,
        'end_price': end_price,
        'years': years,
        'total_return_pct': total_return,
        'annual_return_pct': annual_return,
        'max_drawdown_pct': dd,
    }


if __name__ == '__main__':
    daily = load_full_data()
    daily = compute_indicators(daily)
    print(f"Daily bars: {len(daily)}")
    print(f"Date range: {daily.index[0].date()} to {daily.index[-1].date()}")
    print()

    # --- Test 1: 19-year backtest with real + proxy VIX ---
    print("=" * 70)
    print("TEST 1: 19-YEAR BACKTEST (2007-2026) — no stop-loss")
    print("=" * 70)
    trades_19yr = run_backtest(daily, use_stop=False)
    m = compute_metrics(trades_19yr)
    if m:
        for k, v in m.items():
            print(f"  {k:25s} {v}")

    # --- Test 2: Same backtest with stop-loss ---
    print()
    print("=" * 70)
    print("TEST 2: 19-YEAR BACKTEST — with stop-loss (spread doubles)")
    print("=" * 70)
    trades_19yr_sl = run_backtest(daily, use_stop=True)
    m_sl = compute_metrics(trades_19yr_sl)
    if m_sl:
        for k, v in m_sl.items():
            print(f"  {k:25s} {v}")

    # --- Test 3: Stricter regime filter ---
    print()
    print("=" * 70)
    print("TEST 3: 19-YEAR BACKTEST — strict regime (NORMAL only) + stop-loss")
    print("=" * 70)
    trades_strict = run_backtest(daily, use_stop=True, regime_strict=True)
    m_strict = compute_metrics(trades_strict)
    if m_strict:
        for k, v in m_strict.items():
            print(f"  {k:25s} {v}")

    # --- Test 4: 2019-2026 window only ---
    print()
    print("=" * 70)
    print("TEST 4: 2019-2026 BACKTEST (with real VIX) + stop-loss")
    print("=" * 70)
    daily_2019 = daily.loc['2019-01-01':]
    trades_2019 = run_backtest(daily_2019, use_stop=True)
    m_2019 = compute_metrics(trades_2019)
    if m_2019:
        for k, v in m_2019.items():
            print(f"  {k:25s} {v}")

    # --- Benchmark ---
    print()
    print("=" * 70)
    print("BENCHMARK: NIFTY BUY-AND-HOLD (2019-2026)")
    print("=" * 70)
    bm = benchmark_buyhold(daily, '2019-01-01', '2026-07-01')
    if bm:
        for k, v in bm.items():
            print(f"  {k:25s} {v}")

    # Save
    os.makedirs('results', exist_ok=True)
    trades_19yr.to_csv('results/trades_19yr.csv', index=False)
    trades_19yr_sl.to_csv('results/trades_19yr_sl.csv', index=False)
    trades_strict.to_csv('results/trades_strict.csv', index=False)
    trades_2019.to_csv('results/trades_2019.csv', index=False)
    print()
    print("Saved CSV outputs to results/")
