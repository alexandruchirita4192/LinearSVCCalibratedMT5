# MT5 LinearSVC + Probability Calibration ONNX Package

This repository contains a baseline package for training a multiclass `LinearSVC` with probability calibration in Python, exporting it to ONNX, and running it inside MetaTrader 5.

## Files

- `train_mt5_linearsvc_calibrated_classifier.py`
- `MT5_LinearSVC_Calibrated_ONNX_Strategy.mq5`
- `README.md`

## Model idea

The model predicts three classes:

- `SELL = -1`
- `FLAT = 0`
- `BUY = 1`

The classifier is:

- `LinearSVC`
- calibrated with `CalibratedClassifierCV(method="sigmoid")`

This is intended to provide usable class probabilities for MT5 threshold-based trading.

## Features

The package uses 13 scale-invariant features:

1. `ret_1`
2. `ret_3`
3. `ret_5`
4. `ret_10`
5. `vol_10`
6. `vol_20`
7. `vol_ratio_10_20`
8. `dist_sma_10`
9. `dist_sma_20`
10. `zscore_20`
11. `atr_pct_14`
12. `range_pct_1`
13. `body_pct_1`

## Python requirements

Typical packages:

```bash
pip install numpy pandas scikit-learn skl2onnx onnx MetaTrader5
```

## Training examples

### Train from MT5 terminal data

```bash
python train_mt5_linearsvc_calibrated_classifier.py ^
  --symbol XAGUSD ^
  --timeframe M15 ^
  --bars 20000 ^
  --horizon-bars 8 ^
  --train-ratio 0.70 ^
  --output-dir output_linearsvc_XAGUSD_M15_h8
```

### Train from CSV

```bash
python train_mt5_linearsvc_calibrated_classifier.py ^
  --csv data.csv ^
  --symbol BTCUSD ^
  --timeframe M15 ^
  --bars 20000 ^
  --horizon-bars 8 ^
  --train-ratio 0.70 ^
  --output-dir output_linearsvc_BTCUSD_M15_h8
```

## Notes

Because this model uses `CalibratedClassifierCV`, ONNX export can depend on sklearn/skl2onnx version compatibility.

If export succeeds, the script writes:

- `ml_strategy_classifier_linearsvc_calibrated.onnx`
- `run_in_mt5.txt`
- metadata and prediction snapshots

## MT5 usage

1. Copy `ml_strategy_classifier_linearsvc_calibrated.onnx` next to the EA source file
2. Compile the EA
3. Read `run_in_mt5.txt`
4. Start backtesting with lower thresholds first

Suggested first search zone:

- `InpEntryProbThreshold`: `0.10 -> 0.40`
- `InpMinProbGap`: `0.00 -> 0.10`

## Why test this model

This model is useful as a middle ground between:

- very simple linear classifiers
- more nonlinear models like MLP / boosting

If it works, it suggests that your features may contain a usable mostly linear directional structure.
