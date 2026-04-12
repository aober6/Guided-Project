"""
Bayesian hyperparameter tuning for XGBoost flight price model using Optuna.
Optimizes validation RMSE (log scale) with 3-fold cross-validation.
Saves best params to results/best_params.json, then retrains and evaluates.
"""
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR    = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_TRIALS = 50
CV_FOLDS = 3

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test  = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze()
y_test  = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

X_arr = X_train.values
y_arr = y_train.values

# ── Optuna objective ──────────────────────────────────────────────────────────
def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "gamma":             trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 3.0),
        "objective":         "reg:squarederror",
        "tree_method":       "hist",
        "random_state":      42,
        "verbosity":         0,
    }

    kf = KFold(n_splits=CV_FOLDS, shuffle=False)  # no shuffle = respects temporal order
    rmse_scores = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_arr)):
        model = xgb.XGBRegressor(**params)
        model.fit(X_arr[tr_idx], y_arr[tr_idx], verbose=False)
        preds = model.predict(X_arr[val_idx])
        rmse  = np.sqrt(mean_squared_error(y_arr[val_idx], preds))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

# ── Run study ─────────────────────────────────────────────────────────────────
print(f"\nRunning {N_TRIALS} Optuna trials ({CV_FOLDS}-fold CV each)...")
study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_params = study.best_params
best_params.update({"objective": "reg:squarederror", "tree_method": "hist",
                    "random_state": 42, "verbosity": 0})

print(f"\nBest CV RMSE (log): {study.best_value:.5f}")
print(f"Best params: {json.dumps({k: v for k, v in best_params.items() if k not in ['objective','tree_method','random_state','verbosity']}, indent=2)}")

with open(f"{RESULTS_DIR}/best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)
print(f"Saved to {RESULTS_DIR}/best_params.json")

# ── Retrain on full train set with best params ────────────────────────────────
print("\nRetraining on full training set with best params...")
model = xgb.XGBRegressor(**best_params)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=50)

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

plt.figure(figsize=(10, 6))
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
    f.write("# XGBoost Flight Price Prediction -- Results\n\n")
    f.write("## Baseline vs Tuned\n\n")
    f.write("| Metric | Baseline | Tuned |\n|--------|----------|-------|\n")
    f.write(f"| MAE    | $80.72   | ${mae:.2f} |\n")
    f.write(f"| RMSE   | $142.49  | ${rmse:.2f} |\n")
    f.write(f"| R^2 (log) | 0.7519 | {r2_log:.4f} |\n")
    f.write(f"| R^2 ($)   | 0.6603 | {r2:.4f} |\n")
    f.write(f"| MAPE   | 22.03%   | {mape:.2f}% |\n\n")
    f.write("## Best Hyperparameters\n\n```json\n")
    f.write(json.dumps({k: v for k, v in best_params.items()
                        if k not in ["objective","tree_method","random_state","verbosity"]}, indent=2))
    f.write("\n```\n\n")
    f.write("## Top Features\n\n")
    f.write(feat_imp.reset_index().rename(columns={"index":"Feature",0:"Importance"}).to_markdown(index=False))
    f.write("\n\n## Plots\n\n")
    f.write("- `results/optuna_history.png`\n")
    f.write("- `results/feature_importance_tuned.png`\n")
    f.write("- `results/pred_vs_actual.png`\n")

print("Results written to model_results.md")
