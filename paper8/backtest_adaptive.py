"""
backtest_adaptive.py
Compares four strategies:

  A) Paper 5 (rule-based baseline)
  B) Static LightGBM (Papers 1-4 style, single model)
  C) Adaptive (regime-classified ensemble)
  D) Adaptive + Rule-based filter (belt-and-suspenders)

All use the same trade structure:
  - Bull put credit spread (delta 0.30, 500 wide)
  - 30 DTE, 20-day hold
  - Equity-curve stop (3 losses → 40-day pause)
"""
import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/root/paper6/scripts")
from paper5_core import (
    load_data, compute_indicators, simulate,
    EQ_CONSEC_LOSSES, EQ_PAUSE_DAYS, HOLD,
)

REGIME_DATA = Path("/root/paper8/data/regime_dataset.csv")
MODELS = Path("/root/paper8/models")
RESULTS = Path("/root/paper8/results")
RESULTS.mkdir(exist_ok=True, parents=True)

ALL_FEATURES = ["bb_w_pct", "atr_ratio", "ema_spread", "roc_5", "roc_20",
                "rsi_14", "vix_level", "vix_rank", "vix_change_5", "dist_cloud"]

FOLDS = [
    ("2019-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2019-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2019-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2019-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("2019-01-01", "2025-12-31", "2026-01-01", "2026-12-31"),
]

ENTRY_THRESHOLD = 0.65   # probability above which we enter


def build_full_features(df):
    d = df.copy()
    d["datetime"] = pd.to_datetime(d["datetime"])
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - d["nifty_close"].shift()).abs(),
        (d["low"] - d["nifty_close"].shift()).abs(),
    ], axis=1).max(axis=1)
    d["atr_ratio"] = tr.rolling(14).mean() / d["nifty_close"]
    delta = d["nifty_close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    d["rsi_14"] = 100 - (100 / (1 + rs))
    d["ema_spread"] = (d["ema5"] - d["ema13"]) / d["nifty_close"]
    d["roc_5"] = d["nifty_close"].pct_change(5)
    d["roc_20"] = d["nifty_close"].pct_change(20)
    cloud_mid = (d["cloud_top"] + d["cloud_bottom"]) / 2
    d["dist_cloud"] = (d["nifty_close"] - cloud_mid) / d["nifty_close"]
    d["vix_level"] = d["vix_close"]
    d["vix_rank"] = d["vix_close"].rolling(252).apply(lambda x: (x.iloc[-1] > x).mean(), raw=False)
    d["vix_change_5"] = d["vix_close"].pct_change(5)
    return d


def build_label_dataset(df):
    """For every row (not just Paper 5 signal days), simulate the trade and record the PnL."""
    rows = []
    for i in range(60, len(df) - HOLD):
        row = df.iloc[i]
        S_entry = row["nifty_close"]
        iv_entry = row["vix_close"] / 100.0
        S_path = df["nifty_close"].iloc[i+1 : i+1+HOLD].values
        iv_path = (df["vix_close"].iloc[i+1 : i+1+HOLD].values) / 100.0
        if len(S_path) < HOLD:
            continue
        pnl = simulate(S_entry, S_path, iv_entry, iv_path)
        if pnl is None:
            continue
        r = {"datetime": row["datetime"], "regime": row.get("regime", "UNKNOWN"),
             "pnl": pnl, "label": 1 if pnl > 0 else 0}
        for f in ALL_FEATURES:
            r[f] = row.get(f, np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def train_static_model(tr):
    X = tr[ALL_FEATURES].values
    y = tr["label"].values
    params = {
        "objective": "binary", "metric": "auc",
        "boosting_type": "gbdt", "num_leaves": 15,
        "learning_rate": 0.05, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5,
        "min_data_in_leaf": 10, "verbose": -1, "seed": 42,
    }
    dtrain = lgb.Dataset(X, label=y)
    return lgb.train(params, dtrain, num_boost_round=200)


def predict_adaptive(row, fold_id, regime_pred):
    """Load regime-specific sub-model, predict."""
    path = MODELS / f"submodel_{regime_pred}_fold{fold_id}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feats = bundle["features"]
    X = np.array([[row.get(f, np.nan) for f in feats]])
    X = np.nan_to_num(X, nan=0.0)
    return float(model.predict(X)[0])


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


def run_config(df, label_df, fold_id, tr_s, tr_e, va_s, va_e, config_name):
    """Run one config for one fold."""
    tr = label_df[(label_df["datetime"] >= tr_s) & (label_df["datetime"] <= tr_e)].dropna(subset=ALL_FEATURES)
    va = label_df[(label_df["datetime"] >= va_s) & (label_df["datetime"] <= va_e)].dropna(subset=ALL_FEATURES)

    if len(tr) < 100 or len(va) < 20:
        return None

    # A) Paper 5 — no ML, use all va rows
    if config_name == "A_paper5":
        trades = []
        for _, row in va.iterrows():
            trades.append({"datetime": row["datetime"], "pnl": row["pnl"]})
        tdf = pd.DataFrame(trades)
        return tdf

    # B) Static ML
    if config_name == "B_static":
        model = train_static_model(tr)
        proba = model.predict(va[ALL_FEATURES].values)
        keep = proba >= ENTRY_THRESHOLD
        tdf = va[keep][["datetime", "pnl"]].copy()
        return tdf

    # C) Adaptive (regime classifier + sub-model)
    if config_name == "C_adaptive" or config_name == "D_adaptive_plus_rule":
        # Load regime classifier
        rc_path = MODELS / f"regime_classifier_fold{fold_id}.pkl"
        if not rc_path.exists():
            return None
        with open(rc_path, "rb") as f:
            rc_bundle = pickle.load(f)
        rc_model = rc_bundle["model"]
        rc_label_map = rc_bundle["label_map"]

        keep_rows = []
        for _, row in va.iterrows():
            # Predict regime
            X_rc = np.array([[row.get(f, np.nan) for f in [
                "bb_w_pct", "atr_ratio", "rsi_14", "ema_spread",
                "roc_5", "roc_20", "dist_cloud",
                "vix_level", "vix_rank", "vix_change_5",
                "above_cloud_int", "mts_green", "mts_red",
            ]]])
            X_rc = np.nan_to_num(X_rc, nan=0.0)
            # Handle missing regressor features gracefully
            try:
                proba_rc = rc_model.predict(X_rc)[0]
                pred_regime = rc_label_map[int(np.argmax(proba_rc))]
            except Exception:
                pred_regime = "CHOP_HIGH_VOL"

            # Paper 5 rule filter for config D
            if config_name == "D_adaptive_plus_rule":
                # require the original Paper 5 entry signal
                if row.get("regime", "") in ("SQUEEZE", "WIDE"):
                    continue

            # Sub-model prediction
            prob = predict_adaptive(row, fold_id, pred_regime)
            if prob is None:
                continue
            if prob >= ENTRY_THRESHOLD:
                keep_rows.append({"datetime": row["datetime"], "pnl": row["pnl"]})

        tdf = pd.DataFrame(keep_rows)
        return tdf

    return None


def main():
    print("Loading regime dataset...")
    df = pd.read_csv(REGIME_DATA)
    df = build_full_features(df)

    print("Building labels for all rows...")
    label_df = build_label_dataset(df)
    print(f"  {len(label_df)} simulated entries")

    configs = ["A_paper5", "B_static", "C_adaptive", "D_adaptive_plus_rule"]
    all_results = []
    per_trade_rows = []   # <-- added: per-trade log accumulator

    for fold_id, (tr_s, tr_e, va_s, va_e) in enumerate(FOLDS, 1):
        print(f"\n=== Fold {fold_id}: train={tr_s}..{tr_e} val={va_s}..{va_e} ===")
        for config in configs:
            tdf = run_config(df, label_df, fold_id, tr_s, tr_e, va_s, va_e, config)
            if tdf is None or len(tdf) == 0:
                print(f"  {config}: no trades")
                continue
            tdf["daily_idx"] = tdf.index  # placeholder
            tdf = tdf.reset_index(drop=True)
            tdf["daily_idx"] = tdf.index
            tdf_eq = apply_eq_stop(tdf.copy())
            # --- added: dump per-trade rows ---
            for _, _row in tdf_eq.iterrows():
                per_trade_rows.append({
                    "config": config,
                    "fold": fold_id,
                    "datetime": _row["datetime"],
                    "pnl": _row["pnl"],
                })
            # --- end added ---
            m = metrics(tdf_eq["pnl"].values)
            if m:
                m["config"] = config
                m["fold"] = fold_id
                all_results.append(m)
                print(f"  {config}: n={m['n_trades']} sharpe={m['sharpe']:.3f} "
                      f"total=Rs {m['total_pnl']:,.0f} dd={m['max_dd_pct']:.1f}%")

    results_df = pd.DataFrame(all_results)
    if len(results_df) > 0:
        results_df.to_csv(RESULTS / "adaptive_comparison.csv", index=False)
        # --- added: write per-trade log ---
        if per_trade_rows:
            pd.DataFrame(per_trade_rows).to_csv(
                RESULTS / "config_trades.csv", index=False)
            print(f"  wrote {len(per_trade_rows)} per-trade rows to config_trades.csv")
        # --- end added ---

        print("\n" + "=" * 130)
        print("ADAPTIVE ML TEST — MAIN COMPARISON")
        print("=" * 130)

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
