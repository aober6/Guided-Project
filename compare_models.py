"""
Model comparison: LightGBM vs XGBoost (default) vs XGBoost (tuned best params).
Evaluates on held-out test set in log1p space and dollar space.
"""

import json
import sys
import time
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 output so special chars print on Windows cp1252 consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb

# -- Load data -----------------------------------------------------------------
print("Loading data...")
X_train = pd.read_csv("data/X_train.csv")
X_test  = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test  = pd.read_csv("data/y_test.csv").squeeze()

print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"  y dtype: {y_train.dtype}  |  y range: [{y_train.min():.3f}, {y_train.max():.3f}]")

# -- Metric helper -------------------------------------------------------------
def evaluate(name, y_true_log, y_pred_log):
    mae_log  = mean_absolute_error(y_true_log, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    r2_log   = r2_score(y_true_log, y_pred_log)

    # dollar space: invert log1p
    y_true_dollar = np.expm1(y_true_log)
    y_pred_dollar = np.expm1(y_pred_log)
    r2_dollar  = r2_score(y_true_dollar, y_pred_dollar)
    mae_dollar = mean_absolute_error(y_true_dollar, y_pred_dollar)

    return {
        "Model":       name,
        "MAE_log":     round(mae_log,  4),
        "RMSE_log":    round(rmse_log, 4),
        "R2_log":      round(r2_log,   4),
        "R2_dollar":   round(r2_dollar,4),
        "MAE_dollar":  round(mae_dollar, 2),
    }

results = []

# -- 1. LightGBM default -------------------------------------------------------
print("\n[1/3] Training LightGBM (default params)...")
t0 = time.time()
lgbm_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
lgbm_model.fit(X_train, y_train)
print(f"  Done in {time.time()-t0:.1f}s")
results.append(evaluate("LightGBM (default)", y_test, lgbm_model.predict(X_test)))

# -- 2. XGBoost default --------------------------------------------------------
print("\n[2/3] Training XGBoost (default params)...")
t0 = time.time()
xgb_default = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, verbosity=0)
xgb_default.fit(X_train, y_train)
print(f"  Done in {time.time()-t0:.1f}s")
results.append(evaluate("XGBoost  (default)", y_test, xgb_default.predict(X_test)))

# -- 3. XGBoost tuned ----------------------------------------------------------
print("\n[3/3] Training XGBoost (best_params.json)...")
with open("results/best_params.json") as f:
    best_params = json.load(f)
t0 = time.time()
xgb_tuned = xgb.XGBRegressor(**best_params)
xgb_tuned.fit(X_train, y_train)
print(f"  Done in {time.time()-t0:.1f}s")
results.append(evaluate("XGBoost  (tuned)  ", y_test, xgb_tuned.predict(X_test)))

# -- Print results table -------------------------------------------------------
df = pd.DataFrame(results)

SEP = "=" * 74
print(f"\n{SEP}")
print("MODEL COMPARISON  -  Test-set evaluation (y in log1p(totalFare) space)")
print(SEP)
header = f"{'Model':<24}  {'MAE(log)':>9}  {'RMSE(log)':>9}  {'R2(log)':>8}  {'R2($)':>7}  {'MAE($)':>8}"
print(header)
print("-" * 74)
for r in results:
    print(
        f"{r['Model']:<24}  "
        f"{r['MAE_log']:>9.4f}  "
        f"{r['RMSE_log']:>9.4f}  "
        f"{r['R2_log']:>8.4f}  "
        f"{r['R2_dollar']:>7.4f}  "
        f"{r['MAE_dollar']:>8.2f}"
    )
print(SEP)

# -- Best per metric -----------------------------------------------------------
rows = {r["Model"].strip(): r for r in results}
metrics = [
    ("RMSE(log)",  "RMSE_log",   min),
    ("R2(log)",    "R2_log",     max),
    ("R2($)",      "R2_dollar",  max),
    ("MAE($)",     "MAE_dollar", min),
]
print("\nBest per metric:")
for label, key, fn in metrics:
    best_model = fn(rows, key=lambda m: rows[m][key])
    val = rows[best_model][key]
    print(f"  {label:<10} -> {best_model}  ({val})")

# -- Interpretation ------------------------------------------------------------
lgbm_rmse  = rows["LightGBM (default)"]["RMSE_log"]
xgbd_rmse  = rows["XGBoost  (default)"]["RMSE_log"]
xgbt_rmse  = rows["XGBoost  (tuned)"]["RMSE_log"]
tuning_gain        = xgbd_rmse - xgbt_rmse
lgbm_vs_xgb_tuned  = xgbt_rmse - lgbm_rmse   # positive = XGB tuned better

print("\nKey deltas:")
print(f"  Tuning gain  XGB default -> tuned :  {tuning_gain:+.4f} RMSE")
print(f"  LightGBM default vs XGB tuned     :  {lgbm_vs_xgb_tuned:+.4f} RMSE  (positive = XGB tuned better)")

threshold = 0.010
if abs(lgbm_rmse - xgbt_rmse) < threshold:
    print(f"\n  VERDICT: LightGBM (default) and tuned XGBoost are within {threshold} RMSE.")
    print("           Ceiling is likely the DATA, not the model architecture.")
elif lgbm_rmse < xgbt_rmse - threshold:
    gap = xgbt_rmse - lgbm_rmse
    print(f"\n  VERDICT: Default LightGBM beats tuned XGBoost by {gap:.4f} RMSE.")
    print("           Model architecture matters - worth tuning LightGBM.")
else:
    gap = lgbm_rmse - xgbt_rmse
    print(f"\n  VERDICT: Tuned XGBoost is {gap:.4f} RMSE better than default LightGBM.")
    print("           XGBoost tuning is paying off; tune LightGBM before drawing conclusions.")
