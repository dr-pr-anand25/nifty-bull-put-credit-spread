"""
backtest_adaptive_buyer.py
Compares three strategies for option buying:

  A) Static option buying (papers 1-4 baseline):
     Buy every day that MTS is GREEN -> buy call; MTS RED -> buy put.
     No ML.

  B) Adaptive option buying:
     Regime classifier selects sub-model; sub-model decides entry.
     Entry if sub-model probability > threshold.

  C) Rule + ML (hybrid):
     Same as B, but only trade when paper 5's rule also fires.
     (Kept for comparison — it shouldn't fire much, since Paper 5 is selling.)

All use:
  - 0.30-delta option
  - 5-day hold
  - 50% stop-loss
  - ₹130 round-trip cost per lot
"""
import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd

REGIME_DATA = Path("/root/paper8/data/regime_dataset.csv")
LABELS      = Path("/root/paper9/data/option_buy_labels.csv")
MODELS      = Path("/root/paper9/models")
RESULTS     = Path("/root/paper9/results")

ENTRY_THRESHOLD = 0.60

FOLDS = [
    ("2019-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2019-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2019-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2019-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("2019-01-01", "2025-12-31", "2026-01-01", "2026-12-31"),
]


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


def run_config(labels_df, fold_id, tr_s, tr_e, va_s, va_e, config_name):
    tr = labels_df[(labels_df["datetime"] >= tr_s) & (labels_df["datetime"] <= tr_e)]
    va = labels_df[(labels_df["datetime"] >= va_s) & (labels_df["datetime"] <= va_e)]

    if len(tr) < 100 or len(va) < 20:
        return None

    # A) Static: use all validation rows
    if config_name == "A_static":
        return va[["datetime", "pnl"]].reset_index(drop=True)

    # B and C) Adaptive
    keep = []
    for _, row in va.iterrows():
        regime = row["regime"]
        path = MODELS / f"buyer_{regime}_fold{fold_id}.pkl"
        if not path.exists():
            continue
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        model = bundle["model"]
        feats = bundle["features"]
        X = np.array([[row.get(f, np.nan) for f in feats]])
        X = np.nan_to_num(X, nan=0.0)
        prob = float(model.predict(X)[0])
        if prob >= ENTRY_THRESHOLD:
            keep.append({"datetime": row["datetime"], "pnl": row["pnl"]})
    return pd.DataFrame(keep)


def main():
    print("Loading option-buy labels...")
    labels = pd.read_csv(LABELS)
    labels["datetime"] = pd.to_datetime(labels["datetime"])
    print(f"  {len(labels)} rows")

    configs = ["A_static", "B_adaptive"]
    all_results = []
    per_trade_rows = []   # <-- added

    for fold_id, (tr_s, tr_e, va_s, va_e) in enumerate(FOLDS, 1):
        print(f"\n=== Fold {fold_id} ===")
        for config in configs:
            tdf = run_config(labels, fold_id, tr_s, tr_e, va_s, va_e, config)
            if tdf is None or len(tdf) == 0:
                print(f"  {config}: no trades")
                continue
            # --- added: dump per-trade rows ---
            for _, _row in tdf.iterrows():
                per_trade_rows.append({
                    "config": config,
                    "fold": fold_id,
                    "datetime": _row["datetime"],
                    "pnl": _row["pnl"],
                })
            # --- end added ---
            m = metrics(tdf["pnl"].values)
            if m:
                m["config"] = config
                m["fold"] = fold_id
                all_results.append(m)
                print(f"  {config}: n={m['n_trades']} win={m['win_rate']:.1f}% "
                      f"sharpe={m['sharpe']:.3f} total=Rs {m['total_pnl']:,.0f} dd={m['max_dd_pct']:.1f}%")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS / "adaptive_buyer_comparison.csv", index=False)
    # --- added: write per-trade log ---
    if per_trade_rows:
        pd.DataFrame(per_trade_rows).to_csv(
            RESULTS / "config_trades.csv", index=False)
        print(f"  wrote {len(per_trade_rows)} per-trade rows to config_trades.csv")
    # --- end added ---

    if len(results_df) > 0:
        print("\n" + "=" * 120)
        print("ADAPTIVE OPTION BUYING — MAIN COMPARISON")
        print("=" * 120)
        summary = results_df.groupby("config").agg({
            "n_trades": "sum",
            "win_rate": "mean",
            "avg_pnl": "mean",
            "total_pnl": "sum",
            "sharpe": "mean",
            "max_dd_pct": "mean",
        }).round(3)
        print(summary.to_string())


if __name__ == "__main__":
    main()
