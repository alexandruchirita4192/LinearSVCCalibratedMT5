from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None


SELL_CLASS = -1
FLAT_CLASS = 0
BUY_CLASS = 1

CLASS_ORDER = [SELL_CLASS, FLAT_CLASS, BUY_CLASS]
CLASS_TO_ENC = {SELL_CLASS: 0, FLAT_CLASS: 1, BUY_CLASS: 2}
FEATURE_COLS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "vol_10",
    "vol_20",
    "vol_ratio_10_20",
    "dist_sma_10",
    "dist_sma_20",
    "zscore_20",
    "atr_pct_14",
    "range_pct_1",
    "body_pct_1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LinearSVC + probability calibration classifier for MT5 and export to ONNX."
    )
    parser.add_argument("--symbol", type=str, default="XAGUSD")
    parser.add_argument("--timeframe", type=str, default="M15")
    parser.add_argument("--bars", type=int, default=20000)
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="output_linearsvc_classifier")
    parser.add_argument("--horizon-bars", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--label-quantile", type=float, default=0.67)
    parser.add_argument("--prob-quantile", type=float, default=0.80)
    parser.add_argument("--margin-quantile", type=float, default=0.65)
    parser.add_argument("--walk-forward-splits", type=int, default=5)
    parser.add_argument("--c", type=float, default=1.0, help="LinearSVC regularization parameter.")
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--calibration-cv", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def fetch_rates_from_mt5(symbol: str, timeframe_name: str, bars: int) -> pd.DataFrame:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 Python package is not installed.")

    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    if timeframe_name not in timeframe_map:
        raise ValueError(f"Unsupported timeframe: {timeframe_name}")

    if not mt5.initialize():
        raise RuntimeError(f"initialize() failed: {mt5.last_error()}")

    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe_map[timeframe_name], 0, bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"Could not read data for {symbol} {timeframe_name}. last_error={mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"})
        if "volume" not in df.columns:
            df["volume"] = 0.0
        return df[["time", "open", "high", "low", "close", "volume"]].copy()
    finally:
        mt5.shutdown()


def load_rates_from_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected = {"time", "open", "high", "low", "close"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    return df[["time", "open", "high", "low", "close", "volume"]]


def build_features(df: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    df = df.copy().sort_values("time").reset_index(drop=True)
    eps = 1e-12

    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)

    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()
    df["vol_ratio_10_20"] = (df["vol_10"] / (df["vol_20"] + eps)) - 1.0

    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["dist_sma_10"] = (df["close"] / (df["sma_10"] + eps)) - 1.0
    df["dist_sma_20"] = (df["close"] / (df["sma_20"] + eps)) - 1.0

    roll_mean_20 = df["close"].rolling(20).mean()
    roll_std_20 = df["close"].rolling(20).std()
    df["zscore_20"] = (df["close"] - roll_mean_20) / (roll_std_20 + eps)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = df["tr"].rolling(14).mean()
    df["atr_pct_14"] = df["atr_14"] / (df["close"] + eps)

    df["range_pct_1"] = (df["high"] - df["low"]) / (df["close"] + eps)
    df["body_pct_1"] = (df["close"] - df["open"]) / (df["open"] + eps)

    df["fwd_ret_h"] = df["close"].shift(-horizon_bars) / (df["close"] + eps) - 1.0

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLS + ["fwd_ret_h"]).copy()
    return df


def split_train_test(df: pd.DataFrame, train_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    if len(train_df) < 1500 or len(test_df) < 250:
        raise ValueError("Too few rows after split.")
    return train_df, test_df


def compute_return_barrier(train_df: pd.DataFrame, label_quantile: float) -> float:
    barrier = float(train_df["fwd_ret_h"].abs().quantile(label_quantile))
    return max(barrier, 1e-6)


def label_targets(df: pd.DataFrame, return_barrier: float) -> pd.DataFrame:
    out = df.copy()
    out["target_class"] = FLAT_CLASS
    out.loc[out["fwd_ret_h"] > return_barrier, "target_class"] = BUY_CLASS
    out.loc[out["fwd_ret_h"] < -return_barrier, "target_class"] = SELL_CLASS
    out["target_class"] = out["target_class"].astype(np.int64)
    out["target_class_enc"] = out["target_class"].map(CLASS_TO_ENC).astype(np.int64)
    return out


def make_classifier(c_value: float, max_iter: int, calibration_cv: int, random_state: int) -> Pipeline:
    base = LinearSVC(
        C=c_value,
        max_iter=max_iter,
        random_state=random_state,
        dual="auto",
    )
    calibrated = CalibratedClassifierCV(
        estimator=base,
        method="sigmoid",
        cv=calibration_cv,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("linearsvc_calibrated", calibrated),
    ])


def derive_decision_thresholds(model: Pipeline, train_df: pd.DataFrame, prob_quantile: float, margin_quantile: float) -> Tuple[float, float]:
    X_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    proba = model.predict_proba(X_train)

    p_sell = proba[:, 0]
    p_flat = proba[:, 1]
    p_buy = proba[:, 2]

    best_direction_prob = np.maximum(p_buy, p_sell)
    direction_is_buy = p_buy >= p_sell
    opposite_prob = np.where(direction_is_buy, p_sell, p_buy)
    best_vs_next = best_direction_prob - np.maximum(p_flat, opposite_prob)

    candidates = best_direction_prob > p_flat
    if candidates.any():
        entry_prob_threshold = float(np.quantile(best_direction_prob[candidates], prob_quantile))
        min_prob_gap = float(np.quantile(best_vs_next[candidates], margin_quantile))
    else:
        entry_prob_threshold, min_prob_gap = 0.55, 0.05

    entry_prob_threshold = float(np.clip(entry_prob_threshold, 0.10, 0.95))
    min_prob_gap = float(np.clip(min_prob_gap, 0.00, 0.50))
    return entry_prob_threshold, min_prob_gap


def classify_with_thresholds(model: Pipeline, df: pd.DataFrame, entry_prob_threshold: float, min_prob_gap: float) -> pd.DataFrame:
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    proba = model.predict_proba(X)

    p_sell = proba[:, 0]
    p_flat = proba[:, 1]
    p_buy = proba[:, 2]

    best_direction_prob = np.maximum(p_buy, p_sell)
    direction = np.where(p_buy >= p_sell, BUY_CLASS, SELL_CLASS)
    second_best = np.maximum(p_flat, np.where(direction == BUY_CLASS, p_sell, p_buy))
    prob_gap = best_direction_prob - second_best

    pred = np.full(len(df), FLAT_CLASS, dtype=np.int32)
    take_trade = (
        (best_direction_prob >= entry_prob_threshold)
        & (prob_gap >= min_prob_gap)
        & (best_direction_prob > p_flat)
    )
    pred[take_trade] = direction[take_trade]

    out = df.copy()
    out["p_sell"] = p_sell
    out["p_flat"] = p_flat
    out["p_buy"] = p_buy
    out["best_direction_prob"] = best_direction_prob
    out["prob_gap"] = prob_gap
    out["pred_class"] = pred
    out["trade_taken"] = out["pred_class"] != FLAT_CLASS
    out["direction_correct"] = (
        ((out["pred_class"] == BUY_CLASS) & (out["target_class"] == BUY_CLASS))
        | ((out["pred_class"] == SELL_CLASS) & (out["target_class"] == SELL_CLASS))
    )
    return out


def summarize_predictions(pred_df: pd.DataFrame) -> Dict[str, float]:
    y_true = pred_df["target_class"].to_numpy()
    y_pred = pred_df["pred_class"].to_numpy()
    trade_mask = pred_df["trade_taken"].to_numpy()

    if trade_mask.any():
        directional_precision = float(pred_df.loc[trade_mask, "direction_correct"].mean())
        signed_trade_ret = np.where(
            pred_df.loc[trade_mask, "pred_class"].to_numpy() == BUY_CLASS,
            pred_df.loc[trade_mask, "fwd_ret_h"].to_numpy(),
            -pred_df.loc[trade_mask, "fwd_ret_h"].to_numpy(),
        )
        mean_trade_return = float(signed_trade_ret.mean())
        accepted_trades = int(trade_mask.sum())
        accepted_rate = float(trade_mask.mean())

        gross_profit = float(signed_trade_ret[signed_trade_ret > 0].sum()) if np.any(signed_trade_ret > 0) else 0.0
        gross_loss = float(-signed_trade_ret[signed_trade_ret < 0].sum()) if np.any(signed_trade_ret < 0) else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    else:
        directional_precision = 0.0
        mean_trade_return = 0.0
        accepted_trades = 0
        accepted_rate = 0.0
        gross_profit = 0.0
        gross_loss = 0.0
        profit_factor = 0.0

    return {
        "rows": int(len(pred_df)),
        "accepted_trades": accepted_trades,
        "accepted_rate": accepted_rate,
        "directional_precision_on_trades": directional_precision,
        "mean_signed_fwd_return_on_trades": mean_trade_return,
        "gross_profit_return_units": gross_profit,
        "gross_loss_return_units": gross_loss,
        "profit_factor_return_units": profit_factor,
        "ternary_accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix_sell_flat_buy": confusion_matrix(y_true, y_pred, labels=CLASS_ORDER).tolist(),
    }


def walk_forward_report(
    train_df: pd.DataFrame,
    label_quantile: float,
    prob_quantile: float,
    margin_quantile: float,
    n_splits: int,
    c_value: float,
    max_iter: int,
    calibration_cv: int,
    random_state: int,
) -> Dict[str, float]:
    X = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    accepted_rates: List[float] = []
    directional_precisions: List[float] = []
    mean_trade_returns: List[float] = []
    ternary_accs: List[float] = []
    bal_accs: List[float] = []
    entry_thresholds: List[float] = []
    gap_thresholds: List[float] = []

    for fold, (fit_idx, valid_idx) in enumerate(tscv.split(X), start=1):
        fit_df = train_df.iloc[fit_idx].copy()
        valid_df = train_df.iloc[valid_idx].copy()

        barrier = compute_return_barrier(fit_df, label_quantile)
        fit_df = label_targets(fit_df, barrier)
        valid_df = label_targets(valid_df, barrier)

        model = make_classifier(
            c_value=c_value,
            max_iter=max_iter,
            calibration_cv=calibration_cv,
            random_state=random_state + fold,
        )
        model.fit(
            fit_df[FEATURE_COLS].to_numpy(dtype=np.float32),
            fit_df["target_class_enc"].to_numpy(dtype=np.int64),
        )

        entry_prob_threshold, min_prob_gap = derive_decision_thresholds(
            model=model,
            train_df=fit_df,
            prob_quantile=prob_quantile,
            margin_quantile=margin_quantile,
        )

        pred_df = classify_with_thresholds(
            model=model,
            df=valid_df,
            entry_prob_threshold=entry_prob_threshold,
            min_prob_gap=min_prob_gap,
        )
        summary = summarize_predictions(pred_df)

        accepted_rates.append(summary["accepted_rate"])
        directional_precisions.append(summary["directional_precision_on_trades"])
        mean_trade_returns.append(summary["mean_signed_fwd_return_on_trades"])
        ternary_accs.append(summary["ternary_accuracy"])
        bal_accs.append(summary["balanced_accuracy"])
        entry_thresholds.append(entry_prob_threshold)
        gap_thresholds.append(min_prob_gap)

        print(
            f"Fold {fold}: barrier={barrier:.6f} entry_prob={entry_prob_threshold:.4f} "
            f"gap={min_prob_gap:.4f} accepted_rate={summary['accepted_rate']:.3f} "
            f"precision={summary['directional_precision_on_trades']:.3f} "
            f"mean_trade_ret={summary['mean_signed_fwd_return_on_trades']:.6f} "
            f"bal_acc={summary['balanced_accuracy']:.3f}"
        )

    return {
        "accepted_rate_mean": float(np.mean(accepted_rates)),
        "directional_precision_mean": float(np.mean(directional_precisions)),
        "mean_signed_fwd_return_on_trades_mean": float(np.mean(mean_trade_returns)),
        "ternary_accuracy_mean": float(np.mean(ternary_accs)),
        "balanced_accuracy_mean": float(np.mean(bal_accs)),
        "entry_prob_threshold_mean": float(np.mean(entry_thresholds)),
        "prob_gap_threshold_mean": float(np.mean(gap_thresholds)),
    }


def export_to_onnx(model: Pipeline, output_path: Path) -> None:
    initial_types = [("float_input", FloatTensorType([1, len(FEATURE_COLS)]))]
    options = {id(model): {"zipmap": False}}
    try:
        onx = convert_sklearn(model, initial_types=initial_types, options=options, target_opset=15)
    except Exception as exc:
        raise RuntimeError(
            "ONNX export failed for LinearSVC + probability calibration. "
            "This can depend on sklearn/skl2onnx versions. "
            f"Original error: {exc}"
        ) from exc

    output_path.write_bytes(onx.SerializeToString())


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_rates_from_csv(Path(args.csv)) if args.csv else fetch_rates_from_mt5(args.symbol, args.timeframe, args.bars)
    raw.to_csv(output_dir / "training_rates_snapshot.csv", index=False)

    feat_df = build_features(raw, args.horizon_bars)
    feat_df.to_csv(output_dir / "training_features_snapshot.csv", index=False)

    print(f"Total feature rows: {len(feat_df)}")
    train_df, test_df = split_train_test(feat_df, args.train_ratio)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    print(f"Train window: {train_df['time'].iloc[0]} -> {train_df['time'].iloc[-1]}")
    print(f"Test window : {test_df['time'].iloc[0]} -> {test_df['time'].iloc[-1]}")

    print("\\nWalk-forward on train:")
    walk_forward = walk_forward_report(
        train_df=train_df,
        label_quantile=args.label_quantile,
        prob_quantile=args.prob_quantile,
        margin_quantile=args.margin_quantile,
        n_splits=args.walk_forward_splits,
        c_value=args.c,
        max_iter=args.max_iter,
        calibration_cv=args.calibration_cv,
        random_state=args.random_state,
    )
    print("\\nWalk-forward summary:")
    print(json.dumps(walk_forward, indent=2))

    barrier = compute_return_barrier(train_df, args.label_quantile)
    train_lab = label_targets(train_df, barrier)
    test_lab = label_targets(test_df, barrier)

    model = make_classifier(
        c_value=args.c,
        max_iter=args.max_iter,
        calibration_cv=args.calibration_cv,
        random_state=args.random_state,
    )
    model.fit(
        train_lab[FEATURE_COLS].to_numpy(dtype=np.float32),
        train_lab["target_class_enc"].to_numpy(dtype=np.int64),
    )

    entry_prob_threshold, min_prob_gap = derive_decision_thresholds(
        model=model,
        train_df=train_lab,
        prob_quantile=args.prob_quantile,
        margin_quantile=args.margin_quantile,
    )

    train_pred = classify_with_thresholds(model, train_lab, entry_prob_threshold, min_prob_gap)
    test_pred = classify_with_thresholds(model, test_lab, entry_prob_threshold, min_prob_gap)

    train_summary = summarize_predictions(train_pred)
    test_summary = summarize_predictions(test_pred)

    print(f"\\nLabel barrier abs(fwd_ret_h): {barrier:.8f}")
    print(f"Recommended InpEntryProbThreshold: {entry_prob_threshold:.6f}")
    print(f"Recommended InpMinProbGap:        {min_prob_gap:.6f}")

    print("\\nTrain summary:")
    print(json.dumps(train_summary, indent=2))
    print("\\nTest summary:")
    print(json.dumps(test_summary, indent=2))

    train_pred.to_csv(output_dir / "train_predictions_snapshot.csv", index=False)
    test_pred.to_csv(output_dir / "test_predictions_snapshot.csv", index=False)

    onnx_path = output_dir / "ml_strategy_classifier_linearsvc_calibrated.onnx"
    export_to_onnx(model, onnx_path)

    metadata = {
        "model": "LinearSVC + CalibratedClassifierCV",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "bars": args.bars,
        "horizon_bars": args.horizon_bars,
        "feature_count": len(FEATURE_COLS),
        "features": FEATURE_COLS,
        "train_ratio": args.train_ratio,
        "label_quantile": args.label_quantile,
        "prob_quantile": args.prob_quantile,
        "margin_quantile": args.margin_quantile,
        "c": args.c,
        "max_iter": args.max_iter,
        "calibration_cv": args.calibration_cv,
        "barrier_abs_fwd_ret_h": barrier,
        "entry_prob_threshold": entry_prob_threshold,
        "min_prob_gap": min_prob_gap,
        "walk_forward": walk_forward,
        "train_summary": train_summary,
        "test_summary": test_summary,
    }
    (output_dir / "linearsvc_calibrated_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    run_txt = f"""MODEL: LinearSVC + probability calibration multiclass classifier
SYMBOL: {args.symbol}
TIMEFRAME: {args.timeframe}
HORIZON BARS: {args.horizon_bars}
FEATURE_COUNT: {len(FEATURE_COLS)}

TRAIN UTC:
  start: {train_df["time"].iloc[0]}
  end  : {train_df["time"].iloc[-1]}

TEST UTC:
  start: {test_df["time"].iloc[0]}
  end  : {test_df["time"].iloc[-1]}

FEATURES:
- ret_1
- ret_3
- ret_5
- ret_10
- vol_10
- vol_20
- vol_ratio_10_20
- dist_sma_10
- dist_sma_20
- zscore_20
- atr_pct_14
- range_pct_1
- body_pct_1

TRAINING PARAMS:
  C = {args.c}
  max_iter = {args.max_iter}
  calibration_cv = {args.calibration_cv}
  train_ratio = {args.train_ratio:.4f}

RECOMMENDED EA INPUTS:
  InpEntryProbThreshold = {entry_prob_threshold:.6f}
  InpMinProbGap        = {min_prob_gap:.6f}
  InpMaxBarsInTrade    = {args.horizon_bars}

ONNX FILE:
  - ml_strategy_classifier_linearsvc_calibrated.onnx

IMPORTANT:
- Python and MT5 must use exactly the same feature engineering.
- If ONNX export fails, it is usually due to sklearn/skl2onnx version compatibility for CalibratedClassifierCV.
"""
    (output_dir / "run_in_mt5.txt").write_text(run_txt, encoding="utf-8")

    print(f"\\nModel ONNX saved to: {onnx_path}")
    print("Read: run_in_mt5.txt")


if __name__ == "__main__":
    main()
