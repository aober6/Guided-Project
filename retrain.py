"""
Fast retrain using existing best_params.json — skips Optuna entirely.
Use this to quickly evaluate new features without re-tuning.
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle, os

DATA_DIR    = "data"
RESULTS_DIR = "results"

print("Loading data...")
X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test  = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze()
y_test  = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

with open(f"{RESULTS_DIR}/best_params.json") as f:
    best_params = json.load(f)

# Use last 10% of training data for early stopping — keeps X_test unseen
val_split = int(len(X_train) * 0.9)
X_tr, X_val = X_train.iloc[:val_split], X_train.iloc[val_split:]
y_tr, y_val = y_train.iloc[:val_split], y_train.iloc[val_split:]

print("\nRetraining with best params...")
model = xgb.XGBRegressor(**best_params)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=50)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred_log     = model.predict(X_test)
r2_log         = r2_score(y_test, y_pred_log)
y_pred_dollars = np.expm1(y_pred_log)
y_test_dollars = np.expm1(y_test)
mae  = mean_absolute_error(y_test_dollars, y_pred_dollars)
rmse = np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars))
mape = np.mean(np.abs((y_test_dollars - y_pred_dollars) / np.clip(y_test_dollars, 1, None))) * 100
r2   = r2_score(y_test_dollars, y_pred_dollars)

print(f"\n-- Retrained Model Test Set Metrics ----------------------------")
print(f"  MAE:       ${mae:.2f}")
print(f"  RMSE:      ${rmse:.2f}")
print(f"  R^2 (log): {r2_log:.4f}")
print(f"  R^2 ($):   {r2:.4f}")
print(f"  MAPE:      {mape:.2f}%")

# ── Feature importance ────────────────────────────────────────────────────────
feat_imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(f"\nTop 10 features:\n{feat_imp.head(10).to_string()}")

plt.figure(figsize=(12, 8))
feat_imp.plot(kind="barh")
plt.title("Retrained XGBoost -- Feature Importances")
plt.xlabel("Importance score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/feature_importance_retrained.png", dpi=150)
plt.close()

with open(f"{RESULTS_DIR}/xgboost_retrained.pkl", "wb") as f:
    pickle.dump(model, f)
print(f"\nModel saved to {RESULTS_DIR}/xgboost_retrained.pkl")
