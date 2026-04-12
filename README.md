# Flight Price Prediction — SIGAIDA Guided Project

Predict the price of a domestic US flight given how far in advance you search, the route, airline, and other booking characteristics. The model is trained on real Expedia search data scraped April–October 2022.

---

## Problem Statement

**Can we predict what a flight will cost based on when and how you search for it?**

Specifically, the core research question is: *how does the number of days between your search date and the departure date affect the price you see?* Alongside that, we incorporate route, airline, number of stops, and seat scarcity to build a complete pricing model.

---

## Dataset

**Source:** [`dilwong/flightprices`](https://www.kaggle.com/datasets/dilwong/flightprices) on Kaggle (~6M rows, ~5.9 GB)

The dataset was collected by scraping Expedia for US domestic flights across 16 major airports. Each row is one itinerary returned by one search on one day.

**Target variable:** `totalFare` — the all-in price in USD shown to the user.

Download it with:
```bash
uv run python data.py
```

---

## Features

The model uses 14 engineered and raw features:

| Feature | Type | Description |
|---|---|---|
| `days_until_flight` | Engineered | `flightDate - searchDate` — the single most important signal |
| `startingAirport` | Categorical | Origin airport code (label-encoded) |
| `destinationAirport` | Categorical | Destination airport code (label-encoded) |
| `isNonStop` | Binary | 1 if the flight has no layovers |
| `isBasicEconomy` | Binary | 1 if the fare is basic economy (typically cheapest) |
| `isRefundable` | Binary | 1 if the ticket is refundable (typically most expensive) |
| `seatsRemaining` | Numeric | Number of seats left — low seats = scarcity pricing |
| `departure_month` | Engineered | Month of the flight (captures seasonality) |
| `departure_day_of_week` | Engineered | Day of week of departure (Tue/Wed tend to be cheaper) |
| `search_day_of_week` | Engineered | Day of week the search was performed |
| `num_segments` | Engineered | Number of flight legs (from `\|\|` separators in segment data) |
| `trip_duration_minutes` | Engineered | Total travel time parsed from ISO 8601 `travelDuration` |
| `totalTravelDistance` | Numeric | Total route distance in miles (missing values imputed by route median) |
| `primary_airline` | Categorical | First-leg airline; top 10 kept, rest grouped as "Other" (label-encoded) |

**Dropped features:** `baseFare` (direct leakage), `legId` (identifier), `elapsedDays` (no variance), all redundant segment epoch/raw time columns, `segmentsCabinCode`, `segmentsDistance`, `fareBasisCode`, and `segmentsAirlineCode`.

---

## How It Works

### 1. Preprocessing (`preprocess.py`)

1. Loads up to 500k rows from the CSV (set `FULL_DATA=1` to use all ~6M)
2. Drops rows with missing `totalFare`
3. Engineers all temporal features from `searchDate` and `flightDate`
4. Parses `travelDuration` (ISO 8601 format like `PT5H30M`) into total minutes
5. Counts `||`-delimited segments to get `num_segments`
6. Extracts the first-leg airline from `segmentsAirlineName`, keeps top 10, labels the rest "Other"
7. Imputes missing `totalTravelDistance` with the per-route median
8. Label-encodes `startingAirport`, `destinationAirport`, and `primary_airline`
9. **Log1p-transforms `totalFare`** — flight prices are right-skewed; predicting in log space improves model fit and penalizes large errors proportionally
10. **Temporal train/test split (80/20)** — the first 80% of rows (earlier search dates) are training data, the last 20% are the test set. This is intentional: a random split would leak future pricing patterns into training.

Outputs saved to `data/`: `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, `features.txt`

### 2. Model (`train_model.py`)

**Algorithm: XGBoost (Gradient Boosted Decision Trees)**

XGBoost works by sequentially building an ensemble of decision trees where each new tree corrects the residual errors of the previous ones. It's the standard choice for tabular regression problems like this because:
- Handles mixed feature types (numeric, binary, encoded categorical) natively
- Robust to outliers and skewed distributions
- Captures nonlinear interactions (e.g. price jumps when seats drop below 3)
- Fast on large datasets with `tree_method="hist"`

**Hyperparameters used:**
```
n_estimators     = 500      # number of trees
learning_rate    = 0.05     # how much each tree contributes
max_depth        = 7        # maximum tree depth
subsample        = 0.8      # fraction of rows per tree (reduces overfitting)
colsample_bytree = 0.8      # fraction of features per tree (reduces overfitting)
objective        = reg:squarederror  # minimizes MSE in log space
```

The model trains on log1p-transformed fares and predictions are converted back to dollars with `expm1()` for evaluation.

### 3. Evaluation

Metrics are computed on the held-out test set (last 20% of data by date):

| Metric | Current | Target |
|---|---|---|
| MAE | $80.72 | < $20 |
| RMSE | $142.49 | < $30 |
| R² (log scale) | 0.752 | > 0.92 |
| R² (dollar scale) | 0.660 | > 0.92 |
| MAPE | 22.03% | < 10% |

**Why the gap from targets?** The current run uses only 500k of ~6M available rows, and the sample only covers April–June 2022 (limited seasonality signal). Training on the full dataset is expected to significantly close this gap.

---

## Running the Project

### 1. Install dependencies
```bash
uv add kagglehub pandas numpy scikit-learn xgboost matplotlib seaborn tabulate optuna
```

### 2. Preprocess (500k row sample — fast, for iteration)

The dataset is downloaded automatically via `kagglehub` on first run (~5.9 GB).
A Kaggle account is required. Run `kaggle` login or set `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars.

```bash
uv run python preprocess.py
```

### 2a. Preprocess (full ~6M row dataset)

**PowerShell:**
```powershell
$env:FULL_DATA=1; uv run python preprocess.py
```

**bash/zsh (Mac/Linux):**
```bash
FULL_DATA=1 uv run python preprocess.py
```

### 3. Train baseline model
```bash
uv run python train_model.py
```

### 4. Hyperparameter tuning (50 Optuna trials, ~10-15 min)

Finds optimal XGBoost hyperparameters via Bayesian search, then retrains and evaluates the tuned model.

```bash
uv run python tune_model.py
```

---

## Results

After training, outputs are saved to `results/`:
- `xgboost_model.pkl` — baseline model
- `xgboost_tuned.pkl` — hyperparameter-tuned model
- `best_params.json` — best hyperparameters found by Optuna
- `feature_importance.png` / `feature_importance_tuned.png` — feature importance plots
- `pred_vs_actual.png` — predicted vs actual fare scatter plot
- `optuna_history.png` — Optuna optimization history
- `model_results.md` — full metrics comparison (baseline vs tuned)

---

## Next Steps

- Add departure hour as a feature (parse from `segmentsDepartureTimeRaw`)
- Target-encode `startingAirport` + `destinationAirport` as a combined route feature
- Experiment with LightGBM (often faster and slightly better on high-cardinality categoricals)
