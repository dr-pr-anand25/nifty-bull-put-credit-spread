"""
backtest_hedged.py
Tests whether adding a same-expiry long put hedge improves the Paper 5 strategy.

Baseline:  short put (delta 0.30) + long put (500 below)          = 2 legs
Hedged:    short put (delta 0.30) + long put (500 below)
                                 + hedge put (X below long put)   = 3 legs

Hedge strikes tested: 500 / 1000 / 1500 points below the long put.
Cost per round trip:  2 legs = Rs 130, 3 legs = Rs 195 (approx).
"""
import sys
sys.path.insert(0, "/root/paper6/scripts")
from paper5_core import (
    load_data, compute_indicators, bs_price, find_strike_for_delta,
    RISK_FREE, LOT_SIZE, SLIPPAGE_PCT, SPREAD_WIDTH,
    DAYS_TO_EXPIRY, DELTA, STOP_MULT,
    EQ_CONSEC_LOSSES, EQ_PAUSE_DAYS,
)
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS = Path("/root/paper7/results")
RESULTS.mkdir(parents=True, exist_ok=True)

# Costs: 2-leg = 130, 3-leg = 195 (65 per leg per side × 3)
COST_2_LEG = 130
COST_3_LEG = 195


def simulate_hedged(S_entry, S_path, iv_entry, iv_path, hold_days, hedge_offset):
    """
    Simulate a bull put spread with an optional long-put hedge.

    hedge_offset = 0  → no hedge (baseline)
    hedge_offset > 0  → buy long put at K_long - hedge_offset

    Returns pnl (per lot, after all costs).
    """
    r = RISK_FREE
    T_entry = DAYS_TO_EXPIRY / 365.0
    K_short = find_strike_for_delta(S_entry, DELTA, T_entry, r, iv_entry, "put")
    K_long  = K_short - SPREAD_WIDTH
    K_hedge = K_long - hedge_offset if hedge_offset > 0 else None

    short_entry = bs_price(S_entry, K_short, T_entry, r, iv_entry, "put")
    long_entry  = bs_price(S_entry, K_long,  T_entry, r, iv_entry, "put")
    hedge_entry = bs_price(S_entry, K_hedge, T_entry, r, iv_entry, "put") if K_hedge else 0.0

    # Credit/debit
    credit  = (short_entry - long_entry) * LOT_SIZE
    credit -= hedge_entry * LOT_SIZE                          # hedge cost
    cost = COST_2_LEG if hedge_offset == 0 else COST_3_LEG
    credit -= cost
    credit -= (short_entry + long_entry + hedge_entry) * SLIPPAGE_PCT * LOT_SIZE

    # Walk forward with stop check
    for j, (S_j, iv_j) in enumerate(zip(S_path, iv_path)):
        if j >= hold_days:
            break
        T_j = max((DAYS_TO_EXPIRY - (j + 1)) / 365.0, 1/365.0)

        short_j = bs_price(S_j, K_short, T_j, r, iv_j, "put")
        long_j  = bs_price(S_j, K_long,  T_j, r, iv_j, "put")
        hedge_j = bs_price(S_j, K_hedge, T_j, r, iv_j, "put") if K_hedge else 0.0

        spread_val = (short_j - long_j) * LOT_SIZE

        # Stop-loss check on the bull put only (hedge is separate insurance)
        if spread_val >= short_entry * STOP_MULT * LOT_SIZE:
            exit_cost  = spread_val + cost
            exit_cost += (short_j + long_j + hedge_j) * SLIPPAGE_PCT * LOT_SIZE
            exit_cost -= hedge_j * LOT_SIZE       # hedge sale proceeds
            return credit - exit_cost

    # Time exit
    S_exit  = S_path[hold_days - 1]
    iv_exit = iv_path[hold_days - 1]
    T_exit  = max((DAYS_TO_EXPIRY - hold_days) / 365.0, 1/365.0)

    short_exit = bs_price(S_exit, K_short, T_exit, r, iv_exit, "put")
    long_exit  = bs_price(S_exit, K_long,  T_exit, r, iv_exit, "put")
    hedge_exit = bs_price(S_exit, K_hedge, T_exit, r, iv_exit, "put") if K_hedge else 0.0

    exit_cost  = (short_exit - long_exit) * LOT_SIZE
    exit_cost += cost
    exit_cost += (short_exit + long_exit + hedge_exit) * SLIPPAGE_PCT * LOT_SIZE
    exit_cost -= hedge_exit * LOT_SIZE            # hedge sale proceeds

    return credit - exit_cost


def run_variant(df, hold_days, hedge_offset):
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

        pnl = simulate_hedged(S_entry, S_path, iv_entry, iv_path, hold_days, hedge_offset)
        trades.append({"date": df.index[i], "daily_idx": i, "pnl": pnl})
    return pd.DataFrame(trades)


def apply_eq_stop_paper5(trades_df, consec=EQ_CONSEC_LOSSES, pause=EQ_PAUSE_DAYS):
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


def metrics(pnl):
    pnl = np.array(pnl)
    if len(pnl) == 0: return None
    equity = np.concatenate([[100000], 100000 + np.cumsum(pnl)])
    peak = np.maximum.accumulate(equity)
    dd = ((equity - peak) / peak).min() * 100
    total_ret = (equity[-1] - 100000) / 100000 * 100
    tpy = len(pnl) / 7.0
    sharpe = (pnl.mean() / pnl.std()) * np.sqrt(tpy) if pnl.std() > 0 else 0
    calmar = (total_ret / 7.0) / abs(dd) if dd != 0 else 0
    return {
        "n_trades": len(pnl),
        "win_rate": float((pnl > 0).mean() * 100),
        "avg_pnl": float(pnl.mean()),
        "total_pnl": float(pnl.sum()),
        "total_return_pct": float(total_ret),
        "max_dd_pct": float(dd),
        "sharpe": float(sharpe),
        "calmar": float(calmar),
        "worst_trade": float(pnl.min()),
        "best_trade": float(pnl.max()),
    }


def main():
    print("Loading data...")
    df = compute_indicators(load_data())
    print(f"  {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")

    configs = [
        ("Baseline (no hedge)",           0),
        ("Hedge 500 below long put",     500),
        ("Hedge 1000 below long put",   1000),
        ("Hedge 1500 below long put",   1500),
    ]

    results = []
    for label, hedge in configs:
        print(f"\n=== {label} ===")
        tdf = run_variant(df, 20, hedge)
        m = metrics(tdf["pnl"].values)
        if m:
            m["config"] = label + " [no eq stop]"
            results.append(m)
            print(f"  no eq stop: n={m['n_trades']}  win={m['win_rate']:.1f}%  "
                  f"total={m['total_pnl']:,.0f}  sharpe={m['sharpe']:.3f}  "
                  f"dd={m['max_dd_pct']:.1f}%  worst={m['worst_trade']:,.0f}")

        tdf_eq = apply_eq_stop_paper5(tdf.copy())
        m_eq = metrics(tdf_eq["pnl"].values)
        if m_eq:
            m_eq["config"] = label + " [+eq stop]"
            results.append(m_eq)
            print(f"  + eq stop:  n={m_eq['n_trades']}  win={m_eq['win_rate']:.1f}%  "
                  f"total={m_eq['total_pnl']:,.0f}  sharpe={m_eq['sharpe']:.3f}  "
                  f"dd={m_eq['max_dd_pct']:.1f}%  worst={m_eq['worst_trade']:,.0f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS / "hedge_comparison.csv", index=False)

    print("\n" + "=" * 120)
    print("HEDGE COMPARISON — 20-day hold, 3x stop")
    print("=" * 120)
    print(results_df.to_string(index=False))
    print(f"\nSaved: {RESULTS}/hedge_comparison.csv")


if __name__ == "__main__":
    main()
