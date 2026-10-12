#!/usr/bin/env python3
"""Test: VIX-conditional hedge — activate hedge only when VIX percentile >= threshold."""
import sys
sys.path.insert(0, "/root/paper6/scripts")

import numpy as np
import pandas as pd
from pathlib import Path

from paper5_core import (
    load_data, compute_indicators,
    LOT_SIZE, RISK_FREE, STRIKE_STEP, COST_2_LEG,
    SLIPPAGE_PCT, SPREAD_WIDTH, DAYS_TO_EXPIRY, DELTA, STOP_MULT,
    EQ_CONSEC_LOSSES, EQ_PAUSE_DAYS, bs_price, find_strike_for_delta,
)

COST_3_LEG = 195  # ₹ per 3-leg round trip (Paper 7 config)


RESULTS = Path("/root/paper7/results")
RESULTS.mkdir(parents=True, exist_ok=True)

# ---------- Core simulator (from backtest_hedged.py, unchanged logic) ----------
def simulate_hedged(S_entry, S_path, iv_entry, iv_path, hold_days, hedge_offset):
    """Bull put spread with optional long-put hedge."""
    T_entry = DAYS_TO_EXPIRY / 365.0
    r = RISK_FREE

    K_short = find_strike_for_delta(S_entry, DELTA, T_entry, r, iv_entry, "put")
    K_long  = K_short - SPREAD_WIDTH
    K_hedge = K_long - hedge_offset if hedge_offset > 0 else None

    short_entry = bs_price(S_entry, K_short, T_entry, r, iv_entry, "put")
    long_entry  = bs_price(S_entry, K_long,  T_entry, r, iv_entry, "put")
    hedge_entry = bs_price(S_entry, K_hedge, T_entry, r, iv_entry, "put") if K_hedge else 0.0

    credit  = (short_entry - long_entry) * LOT_SIZE
    credit -= hedge_entry * LOT_SIZE
    cost    = COST_2_LEG if hedge_offset == 0 else COST_3_LEG
    credit -= (short_entry + long_entry + hedge_entry) * SLIPPAGE_PCT * LOT_SIZE
    credit -= cost

    # Walk path
    exit_cost = None
    stop_hit = False
    for j in range(min(hold_days, len(S_path))):
        S_j = S_path[j]
        iv_j = iv_path[j]
        T_j = max(DAYS_TO_EXPIRY - j, 1) / 365.0

        short_j = bs_price(S_j, K_short, T_j, r, iv_j, "put")
        long_j  = bs_price(S_j, K_long,  T_j, r, iv_j, "put")
        hedge_j = bs_price(S_j, K_hedge, T_j, r, iv_j, "put") if K_hedge else 0.0

        spread_val = (short_j - long_j) * LOT_SIZE

        # Stop only if NO hedge is active
        if hedge_offset == 0 and spread_val >= short_entry * STOP_MULT * LOT_SIZE:
            exit_cost = -(short_j - long_j) * LOT_SIZE + hedge_j * LOT_SIZE
            exit_cost -= (short_j + long_j + hedge_j) * SLIPPAGE_PCT * LOT_SIZE
            exit_cost -= cost
            stop_hit = True
            break

    if exit_cost is None:
        S_exit = S_path[hold_days - 1] if hold_days - 1 < len(S_path) else S_path[-1]
        iv_exit = iv_path[hold_days - 1] if hold_days - 1 < len(iv_path) else iv_path[-1]
        T_exit = max(DAYS_TO_EXPIRY - hold_days + 1, 1) / 365.0
        short_exit = bs_price(S_exit, K_short, T_exit, r, iv_exit, "put")
        long_exit  = bs_price(S_exit, K_long,  T_exit, r, iv_exit, "put")
        hedge_exit = bs_price(S_exit, K_hedge, T_exit, r, iv_exit, "put") if K_hedge else 0.0
        exit_cost  = -(short_exit - long_exit) * LOT_SIZE
        exit_cost += hedge_exit * LOT_SIZE
        exit_cost -= (short_exit + long_exit + hedge_exit) * SLIPPAGE_PCT * LOT_SIZE
        exit_cost -= cost

    return credit + exit_cost


# ---------- VIX percentile ----------
def add_vix_percentile(df, window=252, min_periods=60):
    df = df.copy()
    df['vix_rank'] = df['vix_close'].rolling(window, min_periods=min_periods).rank(pct=True)
    return df


# ---------- Runner: dynamic hedge decision per entry ----------
def run_conditional(df, hold_days, vix_threshold):
    """Mirror paper7 run_variant but add VIX-conditional hedge."""
    trades = []
    for i in range(60, len(df) - 30):
        row = df.iloc[i]
        if row["regime"] in ("SQUEEZE", "WIDE"): continue
        if row["mts"] != "GREEN": continue
        if not row["above_cloud"]: continue

        S_entry = row["nifty_close"]
        iv_entry = row["vix_close"] / 100.0
        S_path  = df["nifty_close"].iloc[i+1 : i+1+30].values
        iv_path = (df["vix_close"].iloc[i+1 : i+1+30].values) / 100.0
        if len(S_path) < hold_days: continue

        vr = row.get("vix_rank", float("nan"))
        if pd.isna(vr):
            use_hedge = False
        elif vix_threshold is None:
            use_hedge = False
        elif vix_threshold == -1:
            use_hedge = True
        else:
            use_hedge = (vr >= vix_threshold)

        hedge_offset = 1500 if use_hedge else 0

        pnl = simulate_hedged(S_entry, S_path, iv_entry, iv_path, hold_days, hedge_offset)
        trades.append({
            "date": df.index[i],
            "daily_idx": i,
            "pnl": pnl,
            "vix_rank": vr,
            "hedged": use_hedge,
        })
    return pd.DataFrame(trades)


# ---------- Equity-curve stop (from paper5_core) ----------
def apply_eq_stop(trades_df, consec=EQ_CONSEC_LOSSES, pause=EQ_PAUSE_DAYS):
    if len(trades_df) == 0:
        return trades_df
    trades_df = trades_df.sort_values("daily_idx").reset_index(drop=True)
    cc = 0
    pause_until = -1
    keep = []
    for idx, row in trades_df.iterrows():
        if row["daily_idx"] < pause_until:
            continue
        keep.append(idx)
        if row["pnl"] < 0:
            cc += 1
            if cc >= consec:
                pause_until = row["daily_idx"] + pause
                cc = 0
        else:
            cc = 0
    return trades_df.loc[keep].reset_index(drop=True)


# ---------- Metrics ----------
def metrics(pnl):
    if len(pnl) == 0:
        return {}
    wins = (pnl > 0).sum()
    avg = pnl.mean()
    total = pnl.sum()
    wr = wins / len(pnl) * 100

    import numpy as _np
    eq = _np.cumsum(pnl)
    dd = float(_np.min(eq - _np.maximum.accumulate(eq)))

    weekly_ret = pnl / 100000
    sharpe = (weekly_ret.mean() / weekly_ret.std() * np.sqrt(52)) if weekly_ret.std() > 0 else 0.0
    annual = weekly_ret.mean() * 52
    calmar = annual / abs(dd / 100000) if dd < 0 else 0.0

    return dict(
        n=len(pnl), wr=wr, avg=avg, total=total,
        sharpe=sharpe, calmar=calmar, dd=dd,
        best=pnl.max(), worst=pnl.min(),
    )


# ---------- Main ----------
def main():
    print("Loading data...")
    d = load_data()
    d = compute_indicators(d)
    d = add_vix_percentile(d)
    print(f"Data: {len(d)} rows, {d.index.min().date()} -> {d.index.max().date()}")
    print(f"Entry signals: {d['entry_signal'].sum()}")

    configs = [
        ("A — Paper 5 baseline (no hedge)",       None),   # None = no hedge
        ("B — Paper 7 Config D (always hedge)",   -1),     # -1 = always hedge
        ("C — VIX-conditional >= 0.50",           0.50),
        ("D — VIX-conditional >= 0.40",           0.40),
        ("E — VIX-conditional >= 0.60",           0.60),
    ]

    results = []
    for label, param in configs:
        print(f"\nRunning: {label}")
        if param is None:
            # No hedge at all (baseline). Build a runner with fixed 0.
            trades = run_conditional(d, 20, vix_threshold=1.1)   # 1.1 → never hedges
        elif param == -1:
            # Always hedge.
            trades = run_conditional(d, 20, vix_threshold=-0.1)  # -0.1 → always hedges
        else:
            trades = run_conditional(d, 20, vix_threshold=param)

        trades = apply_eq_stop(trades)
        m = metrics(trades['pnl'].values)
        if not m:
            print(f"  No trades for {label}")
            continue
        m['label'] = label
        m['hedged_trades'] = int(trades['hedged'].sum()) if 'hedged' in trades.columns else 0
        results.append(m)
        print(f"  N={m['n']}  hedged={m['hedged_trades']}  "
              f"WR={m['wr']:.1f}%  Sharpe={m['sharpe']:.3f}  "
              f"Calmar={m['calmar']:.3f}  DD=Rs {m['dd']:,.0f}  "
              f"Total=Rs {m['total']:,.0f}")

    # Save & print summary
    res = pd.DataFrame(results)
    res.to_csv(RESULTS / "vix_conditional_comparison.csv", index=False)
    print(f"\nSaved: {RESULTS / 'vix_conditional_comparison.csv'}")

    print("\n" + "=" * 72)
    print("SUMMARY — VIX-CONDITIONAL HEDGE TEST")
    print("=" * 72)
    print(res[['label','n','hedged_trades','wr','sharpe','calmar','dd','total']].to_string(index=False))

    # Champion identification
    best = res.loc[res['sharpe'].idxmax()]
    print(f"\nHighest Sharpe: {best['label']}  (Sharpe = {best['sharpe']:.3f})")

    baseline = res[res['label'].str.startswith('B')]
    if len(baseline):
        bs = baseline.iloc[0]['sharpe']
        if best['sharpe'] > bs:
            print(f"*** BEATS Paper 7 Config D by {best['sharpe'] - bs:.3f} Sharpe ***")
        else:
            print(f"Does NOT beat Paper 7 Config D ({bs:.3f})")


if __name__ == "__main__":
    main()
