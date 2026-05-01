"""
Bayesian hyperparameter tuning for XGBoost flight price model using Optuna.
Optimizes validation RMSE (log scale) with TimeSeriesSplit CV.
Key improvements over v1:
  - early_stopping_rounds=50 inside CV (removes n_estimators as noisy dimension)
  - TimeSeriesSplit instead of KFold (respects temporal order, lower fold variance)
  - multivariate=True + n_startup_trials=20 for better TPE exploration
  - gamma/reg_alpha/reg_lambda on log scale (fixes boundary degeneracy at 0)
  - Full 9M-row training set used for CV (no subsampling)
  - N_ESTIMATORS_MAX=3000 so early stopping finds true tree-count optimum
  - SQLite-backed study — safe to interrupt and resume
"""
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR    = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_TRIALS        = 50
CV_FOLDS        = 3
N_ESTIMATORS_MAX = 3000   # high cap; early stopping picks the true optimum

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test  = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze()
y_test  = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Use the full training set for CV — no subsampling.
# Each trial will be slower but hyperparameters will reflect the actual data distribution.
print(f"Using all {len(X_train):,} rows for CV")
X_arr = X_train.values
y_arr = y_train.values

# ── Optuna objective ──────────────────────────────────────────────────────────
def objective(trial):
    # n_estimators is NOT tuned -- fixed at 1000 with early stopping
    # so effective tree count is derived, not a noisy free parameter
    params = {
        "n_estimators":      N_ESTIMATORS_MAX,
        "early_stopping_rounds": 50,
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth":         trial.suggest_int("max_depth", 4, 9),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
        "gamma":             trial.suggest_float("gamma", 1e-8, 2.0, log=True),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "objective":         "reg:squarederror",
        "tree_method":       "hist",
        "random_state":      42,
        "verbosity":         0,
    }

    # TimeSeriesSplit respects temporal order (expanding window)
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    rmse_scores = []

    for tr_idx, val_idx in tscv.split(X_arr):
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_arr[tr_idx], y_arr[tr_idx],
            eval_set=[(X_arr[val_idx], y_arr[val_idx])],
            verbose=False,
        )
        preds = model.predict(X_arr[val_idx])
        rmse  = np.sqrt(mean_squared_error(y_arr[val_idx], preds))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

# ── Run study ─────────────────────────────────────────────────────────────────
STUDY_DB   = f"sqlite:///{RESULTS_DIR}/optuna_study.db"
STUDY_NAME = "xgboost_flight_price_fullcv"

completed_before = 0
try:
    existing = optuna.load_study(study_name=STUDY_NAME, storage=STUDY_DB)
    completed_before = len([t for t in existing.trials
                            if t.state == optuna.trial.TrialState.COMPLETE])
except Exception:
    pass

remaining = N_TRIALS - completed_before
if remaining <= 0:
    print(f"\nStudy already has {completed_before} completed trials (target: {N_TRIALS}). Nothing to do.")
    study = optuna.load_study(study_name=STUDY_NAME, storage=STUDY_DB)
else:
    if completed_before:
        print(f"\nResuming study — {completed_before} trials done, {remaining} remaining...")
    else:
        print(f"\nRunning {N_TRIALS} Optuna trials ({CV_FOLDS}-fold TimeSeriesSplit CV each)...")
    # n_startup_trials=20: first 20 trials are random; TPE (Bayesian) only kicks
    # in after that, which is why meaningful improvement typically appears around
    # trial 20-25 rather than earlier.
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STUDY_DB,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=20,   # more random exploration before TPE kicks in
            multivariate=True,     # MOTPE: models parameter interactions
        ),
    )
    study.optimize(objective, n_trials=remaining, show_progress_bar=True)

best_params = study.best_params
best_params.update({
    "objective":   "reg:squarederror",
    "tree_method": "hist",
    "random_state": 42,
    "verbosity":   0,
    "n_estimators": N_ESTIMATORS_MAX,
    "early_stopping_rounds": 50,
})

print(f"\nBest CV RMSE (log): {study.best_value:.5f}")
display_params = {k: v for k, v in best_params.items()
                  if k not in ["objective", "tree_method", "random_state", "verbosity",
                               "n_estimators", "early_stopping_rounds"]}
print(f"Best params: {json.dumps(display_params, indent=2)}")

with open(f"{RESULTS_DIR}/best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)
print(f"Saved to {RESULTS_DIR}/best_params.json")

# ── Retrain on full train set with best params ────────────────────────────────
# Use the last 10% of training data as a validation split for early stopping.
# This keeps X_test completely unseen until final evaluation.
print("\nRetraining on full training set with best params...")
val_split = int(len(X_train) * 0.9)
X_tr, X_val = X_train.iloc[:val_split], X_train.iloc[val_split:]
y_tr, y_val = y_train.iloc[:val_split], y_train.iloc[val_split:]

model = xgb.XGBRegressor(**best_params)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred_log    = model.predict(X_test)
r2_log        = r2_score(y_test, y_pred_log)
y_pred_dollars = np.expm1(y_pred_log)
y_test_dollars = np.expm1(y_test)
mae  = mean_absolute_error(y_test_dollars, y_pred_dollars)
rmse = np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars))
mape = np.mean(np.abs((y_test_dollars - y_pred_dollars) / np.clip(y_test_dollars, 1, None))) * 100
r2   = r2_score(y_test_dollars, y_pred_dollars)

print(f"\n-- Tuned Model Test Set Metrics ---------------------------------")
print(f"  MAE:          ${mae:.2f}")
print(f"  RMSE:         ${rmse:.2f}")
print(f"  R^2 (log):    {r2_log:.4f}")
print(f"  R^2 ($):      {r2:.4f}")
print(f"  MAPE:         {mape:.2f}%")

# ── Optuna visualisation — optimization history ───────────────────────────────
trial_nums  = [t.number for t in study.trials]
trial_vals  = [t.value  for t in study.trials]
best_so_far = [min(trial_vals[:i+1]) for i in range(len(trial_vals))]

plt.figure(figsize=(10, 4))
plt.plot(trial_nums, trial_vals,  "o-", alpha=0.4, label="Trial RMSE")
plt.plot(trial_nums, best_so_far, "r-", lw=2,      label="Best so far")
plt.xlabel("Trial")
plt.ylabel("CV RMSE (log scale)")
plt.title("Optuna Optimization History")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/optuna_history.png", dpi=150)
plt.close()
print(f"Optimization history plot saved to {RESULTS_DIR}/optuna_history.png")

# ── Feature importance ────────────────────────────────────────────────────────
feat_imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

plt.figure(figsize=(12, 8))
feat_imp.plot(kind="barh")
plt.title("Tuned XGBoost -- Feature Importances")
plt.xlabel("Importance score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/feature_importance_tuned.png", dpi=150)
plt.close()

# ── Save model ────────────────────────────────────────────────────────────────
with open(f"{RESULTS_DIR}/xgboost_tuned.pkl", "wb") as f:
    pickle.dump(model, f)
print(f"Tuned model saved to {RESULTS_DIR}/xgboost_tuned.pkl")

# ── Update model_results.md ───────────────────────────────────────────────────
with open("model_results.md", "w") as f:
    f.write("# XGBoost Flight Price Prediction -- Results (Chicago ORD/MDW)\n\n")
    f.write("## Tuned Model Metrics\n\n")
    f.write("| Metric | Value |\n|--------|-------|\n")
    f.write(f"| MAE    | ${mae:.2f} |\n")
    f.write(f"| RMSE   | ${rmse:.2f} |\n")
    f.write(f"| R^2 (log) | {r2_log:.4f} |\n")
    f.write(f"| R^2 ($)   | {r2:.4f} |\n")
    f.write(f"| MAPE   | {mape:.2f}% |\n\n")
    f.write("## Best Hyperparameters\n\n```json\n")
    f.write(json.dumps(display_params, indent=2))
    f.write("\n```\n\n")
    f.write("## Top Features\n\n")
    feat_df = feat_imp.reset_index()
    feat_df.columns = ["Feature", "Importance"]
    f.write(feat_df.to_markdown(index=False))
    f.write("\n\n## Plots\n\n")
    f.write("- `results/optuna_history.png`\n")
    f.write("- `results/feature_importance_tuned.png`\n")

print("Results written to model_results.md")
