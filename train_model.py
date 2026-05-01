"""
XGBoost model for flight price prediction.
Trains, evaluates, and saves the model + results.
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load preprocessed data ────────────────────────────────────────────────────
print("Loading preprocessed data...")
X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test  = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze()
y_test  = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── Train XGBoost ─────────────────────────────────────────────────────────────
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    verbosity=1,
)

print("\nTraining XGBoost...")
t0 = time.time()
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)
elapsed = time.time() - t0
print(f"Training time: {elapsed:.1f}s")

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred_log = model.predict(X_test)

# R^2 on log scale (what the model optimizes)
r2_log = r2_score(y_test, y_pred_log)

# Dollar-scale metrics (invert log1p for interpretability)
y_pred_dollars = np.expm1(y_pred_log)
y_test_dollars = np.expm1(y_test)

mae  = mean_absolute_error(y_test_dollars, y_pred_dollars)
rmse = np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars))
mape = np.mean(np.abs((y_test_dollars - y_pred_dollars) / np.clip(y_test_dollars, 1, None))) * 100
r2   = r2_score(y_test_dollars, y_pred_dollars)

print(f"\n-- Test Set Metrics ---------------------------------------------")
print(f"  MAE:       ${mae:.2f}")
print(f"  RMSE:      ${rmse:.2f}")
print(f"  R^2 (log):  {r2_log:.4f}")
print(f"  R^2 ($):    {r2:.4f}")
print(f"  MAPE:      {mape:.2f}%")

# ── Feature importance ────────────────────────────────────────────────────────
feat_imp = pd.Series(
    model.feature_importances_, index=X_train.columns
).sort_values(ascending=False).head(20)

plt.figure(figsize=(10, 6))
feat_imp.plot(kind="barh")
plt.title("XGBoost — Top 20 Feature Importances")
plt.xlabel("Importance score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/feature_importance.png", dpi=150)
plt.close()
print(f"\nFeature importance plot saved to {RESULTS_DIR}/feature_importance.png")

# ── Predicted vs Actual scatter ───────────────────────────────────────────────
n = min(5000, len(y_test))
plt.figure(figsize=(8, 8))
plt.scatter(y_test_dollars.values[:n], y_pred_dollars[:n], alpha=0.3, s=5)
mn, mx = float(y_test_dollars.min()), float(y_test_dollars.max())
plt.plot([mn, mx], [mn, mx], "r--", lw=1, label="Perfect prediction")
plt.xlabel("Actual fare ($)")
plt.ylabel("Predicted fare ($)")
plt.title("XGBoost — Predicted vs Actual")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/pred_vs_actual.png", dpi=150)
plt.close()
print(f"Pred vs actual plot saved to {RESULTS_DIR}/pred_vs_actual.png")

# ── Save model ────────────────────────────────────────────────────────────────
with open(f"{RESULTS_DIR}/xgboost_model.pkl", "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to {RESULTS_DIR}/xgboost_model.pkl")

# ── Write results markdown ────────────────────────────────────────────────────
with open("model_results.md", "w") as f:
    f.write(f"# XGBoost Flight Price Prediction — Results\n\n")
    f.write(f"## Metrics\n\n")
    f.write(f"| Metric | Value |\n|--------|-------|\n")
    f.write(f"| MAE    | ${mae:.2f} |\n")
    f.write(f"| RMSE   | ${rmse:.2f} |\n")
    f.write(f"| R^2 (log scale) | {r2_log:.4f} |\n")
    f.write(f"| R^2 ($ scale)   | {r2:.4f} |\n")
    f.write(f"| MAPE   | {mape:.2f}% |\n")
    f.write(f"| Train time | {elapsed:.1f}s |\n\n")
    f.write(f"## Hyperparameters\n\n```\n{model.get_params()}\n```\n\n")
    f.write(f"## Top 20 Features\n\n")
    f.write(feat_imp.reset_index().rename(columns={"index": "Feature", 0: "Importance"}).to_markdown(index=False))
    f.write(f"\n\n## Plots\n\n- `results/feature_importance.png`\n- `results/pred_vs_actual.png`\n")

print("\nResults written to model_results.md")
