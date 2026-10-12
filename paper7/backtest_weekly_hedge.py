"""
backtest_weekly_hedge.py
Compares same-expiry hedge (Config D) vs weekly rolling hedge on NIFTY bull put spreads.

Variants tested:
  D  — Same-expiry hedge at 1000 below long put + eq stop (baseline winner)
  W1 — Weekly rolling: buy 7-DTE put, roll every 5 trading days
  W2 — Ladder: 4 stacked puts at 7, 14, 21, 28 DTE, no roll
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

COST_3_LEG = 195
HEDGE_STRIKE_OFFSET = 1000

# Assumptions for weekly hedge
WEEKLY_HEDGE_STRIKE_OFFSET = 1000    # below long put
ROLL_INTERVAL_DAYS = 5                # roll every 5 trading days


def bs_weekly_put_price(S, K, days_to_expiry, r, sigma):
    """Price a weekly put with given DTE in trading days (5 days = 1 week)."""
    T = max(days_to_expiry / 365.0, 1/365.0)
    return bs_price(S, K, T, r, sigma, "put")


def simulate_same_expiry_hedge(S_entry, S_path, iv_entry, iv_path, hold_days):
    """Config D: one hedge put at same expiry as bull put."""
    r = RISK_FREE
    T_entry = DAYS_TO_EXPIRY / 365.0
    K_short = find_strike_for_delta(S_entry, DELTA, T_entry, r, iv_entry, "put")
    K_long  = K_short - SPREAD_WIDTH
    K_hedge = K_long - HEDGE_STRIKE_OFFSET

    short_entry = bs_price(S_entry, K_short, T_entry, r, iv_entry, "put")
    long_entry  = bs_price(S_entry, K_long,  T_entry, r, iv_entry, "put")
    hedge_entry = bs_price(S_entry, K_hedge, T_entry, r, iv_entry, "put")

    credit  = (short_entry - long_entry) * LOT_SIZE
    credit -= hedge_entry * LOT_SIZE
    credit -= COST_3_LEG
    credit -= (short_entry + long_entry + hedge_entry) * SLIPPAGE_PCT * LOT_SIZE

    S_exit  = S_path[hold_days - 1]
    iv_exit = iv_path[hold_days - 1]
    T_exit  = max((DAYS_TO_EXPIRY - hold_days) / 365.0, 1/365.0)

    short_exit = bs_price(S_exit, K_short, T_exit, r, iv_exit, "put")
    long_exit  = bs_price(S_exit, K_long,  T_exit, r, iv_exit, "put")
    hedge_exit = bs_price(S_exit, K_hedge, T_exit, r, iv_exit, "put")

    exit_cost  = (short_exit - long_exit) * LOT_SIZE
    exit_cost += COST_3_LEG
    exit_cost += (short_exit + long_exit + hedge_exit) * SLIPPAGE_PCT * LOT_SIZE
    exit_cost -= hedge_exit * LOT_SIZE

    return credit - exit_cost


def simulate_weekly_rolling_hedge(S_entry, S_path, iv_entry, iv_path, hold_days):
    """
    W1: Buy a 7-DTE put, roll it every ROLL_INTERVAL_DAYS.

    At each roll, close the old put and buy a new one. Hedge strike is recomputed
    relative to the ORIGINAL K_long (so it doesn't drift).
    """
    r = RISK_FREE
    T_entry = DAYS_TO_EXPIRY / 365.0
    K_short = find_strike_for_delta(S_entry, DELTA, T_entry, r, iv_entry, "put")
    K_long  = K_short - SPREAD_WIDTH
    K_hedge = K_long - WEEKLY_HEDGE_STRIKE_OFFSET

    short_entry = bs_price(S_entry, K_short, T_entry, r, iv_entry, "put")
    long_entry  = bs_price(S_entry, K_long,  T_entry, r, iv_entry, "put")

    credit  = (short_entry - long_entry) * LOT_SIZE
    credit -= COST_3_LEG
    credit -= (short_entry + long_entry) * SLIPPAGE_PCT * LOT_SIZE

    # Initial hedge purchase: 7-day put (equivalent to 7 DTE)
    current_hedge_cost = bs_weekly_put_price(S_entry, K_hedge, 7, r, iv_entry)
    credit -= current_hedge_cost * LOT_SIZE
    credit -= current_hedge_cost * SLIPPAGE_PCT * LOT_SIZE

    total_roll_costs = 0.0
    hedge_value = current_hedge_cost
    days_since_roll = 0

    for j, (S_j, iv_j) in enumerate(zip(S_path, iv_path)):
        if j >= hold_days:
            break

        days_since_roll += 1

        # Mark the current hedge to market (its remaining DTE shrinks)
        current_dte = max(7 - days_since_roll, 0)
        hedge_value = bs_weekly_put_price(S_j, K_hedge, current_dte, r, iv_j) if current_dte > 0 else 0.0

        # Roll when days_since_roll reaches ROLL_INTERVAL_DAYS
        if days_since_roll >= ROLL_INTERVAL_DAYS and j < hold_days - 1:
            # Sell old hedge at mark-to-market, buy new one
            sell_proceeds = hedge_value
            new_hedge_cost = bs_weekly_put_price(S_j, K_hedge, 7, r, iv_j)
            roll_cost = new_hedge_cost - sell_proceeds
            # Add roll brokerage + slippage
            roll_cost += 20  # brokerage on the roll
            roll_cost += new_hedge_cost * SLIPPAGE_PCT
            total_roll_costs += roll_cost

            hedge_value = new_hedge_cost
            days_since_roll = 0

    # At trade exit, sell the current hedge for its remaining value
    S_exit  = S_path[hold_days - 1]
    iv_exit = iv_path[hold_days - 1]
    T_exit  = max((DAYS_TO_EXPIRY - hold_days) / 365.0, 1/365.0)

    current_dte = max(7 - days_since_roll, 0)
    final_hedge_value = bs_weekly_put_price(S_exit, K_hedge, current_dte, r, iv_exit) if current_dte > 0 else 0.0

    short_exit = bs_price(S_exit, K_short, T_exit, r, iv_exit, "put")
    long_exit  = bs_price(S_exit, K_long,  T_exit, r, iv_exit, "put")

    exit_cost  = (short_exit - long_exit) * LOT_SIZE
    exit_cost += COST_3_LEG
    exit_cost += (short_exit + long_exit) * SLIPPAGE_PCT * LOT_SIZE

    pnl = credit - exit_cost + final_hedge_value * LOT_SIZE - total_roll_costs * LOT_SIZE
    return pnl


def simulate_ladder_hedge(S_entry, S_path, iv_entry, iv_path, hold_days):
    """
    W2: Buy 4 puts at once, at 7, 14, 21, 28 DTE. No rolling.
    Each put is at the same strike (K_hedge).
    """
    r = RISK_FREE
    T_entry = DAYS_TO_EXPIRY / 365.0
    K_short = find_strike_for_delta(S_entry, DELTA, T_entry, r, iv_entry, "put")
    K_long  = K_short - SPREAD_WIDTH
    K_hedge = K_long - WEEKLY_HEDGE_STRIKE_OFFSET

    short_entry = bs_price(S_entry, K_short, T_entry, r, iv_entry, "put")
    long_entry  = bs_price(S_entry, K_long,  T_entry, r, iv_entry, "put")

    # Buy 4 laddered puts
    ladder_dtes = [7, 14, 21, 28]
    ladder_costs = [bs_weekly_put_price(S_entry, K_hedge, d, r, iv_entry) for d in ladder_dtes]
    total_ladder_cost = sum(ladder_costs)

    credit  = (short_entry - long_entry) * LOT_SIZE
    credit -= total_ladder_cost * LOT_SIZE
    credit -= COST_3_LEG * 4   # 4 extra legs to buy
    credit -= total_ladder_cost * SLIPPAGE_PCT * LOT_SIZE

    # At exit, value remaining ladder puts
    S_exit  = S_path[hold_days - 1]
    iv_exit = iv_path[hold_days - 1]
    T_exit  = max((DAYS_TO_EXPIRY - hold_days) / 365.0, 1/365.0)

    short_exit = bs_price(S_exit, K_short, T_exit, r, iv_exit, "put")
    long_exit  = bs_price(S_exit, K_long,  T_exit, r, iv_exit, "put")

    # Ladder puts: each has 7, 14, 21, 28 DTE at entry. By hold_days, they have
    # shrunk by hold_days days each.
    remaining_values = []
    for d in ladder_dtes:
        remaining_dte = d - hold_days
        if remaining_dte > 0:
            remaining_values.append(bs_weekly_put_price(S_exit, K_hedge, remaining_dte, r, iv_exit))

    # Peaks: find the max value each put achieved during the hold
    # For simplicity, use final value + assume intermediate peaks are captured
    # We'll use the final value at exit, plus account for puts that hit in-the-money
    ladder_payoff = 0.0
    for d in ladder_dtes:
        max_val = 0.0
        for j, (S_j, iv_j) in enumerate(zip(S_path, iv_path)):
            if j >= hold_days: break
            remaining_dte = max(d - (j + 1), 0)
            if remaining_dte > 0:
                val = bs_weekly_put_price(S_j, K_hedge, remaining_dte, r, iv_j)
                if val > max_val:
                    max_val = val
        ladder_payoff += max_val

    exit_cost  = (short_exit - long_exit) * LOT_SIZE
    exit_cost += COST_3_LEG
    exit_cost += (short_exit + long_exit) * SLIPPAGE_PCT * LOT_SIZE

    pnl = credit - exit_cost + ladder_payoff * LOT_SIZE
    return pnl


def run_variant(df, hold_days, sim_func):
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

        pnl = sim_func(S_entry, S_path, iv_entry, iv_path, hold_days)
        trades.append({"date": df.index[i], "daily_idx": i, "pnl": pnl})
    return pd.DataFrame(trades)


def apply_eq_stop(trades_df, consec=EQ_CONSEC_LOSSES, pause=EQ_PAUSE_DAYS):
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
    }


def main():
    print("Loading data...")
    df = compute_indicators(load_data())
    print(f"  {len(df)} rows")

    configs = [
        ("D — Same-expiry hedge 1000 below + eq stop",      simulate_same_expiry_hedge, True),
        ("W1 — Weekly rolling hedge 1000 below + eq stop",  simulate_weekly_rolling_hedge, True),
        ("W2 — Ladder (4 puts) hedge 1000 below + eq stop", simulate_ladder_hedge, True),
    ]

    results = []
    for label, sim_func, use_eq in configs:
        print(f"\n=== {label} ===")
        tdf = run_variant(df, 20, sim_func)
        if use_eq:
            tdf = apply_eq_stop(tdf.copy())
        m = metrics(tdf["pnl"].values)
        if m:
            m["config"] = label
            results.append(m)
            print(f"  n={m['n_trades']}  win={m['win_rate']:.1f}%  "
                  f"total={m['total_pnl']:,.0f}  sharpe={m['sharpe']:.3f}  "
                  f"dd={m['max_dd_pct']:.1f}%  worst={m['worst_trade']:,.0f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS / "weekly_hedge_comparison.csv", index=False)

    print("\n" + "=" * 130)
    print("SAME-EXPIRY vs WEEKLY ROLLING vs LADDER HEDGE")
    print("=" * 130)
    print(results_df.to_string(index=False))
    print(f"\nSaved: {RESULTS}/weekly_hedge_comparison.csv")


if __name__ == "__main__":
    main()
