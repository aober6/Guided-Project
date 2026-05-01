# FlightLens — SIGAIDA Guided Project

An end-to-end flight price predictor for routes into Chicago O'Hare (ORD). XGBoost
model trained on ~6M Expedia search records, served behind a FastAPI endpoint,
and visualized in a browser dashboard with live prices from Sky Scrapper.

---

## What it does

- **Predicts flight prices** as a function of route, days-until-flight, airline,
  cabin class, departure date, and other booking characteristics.
- **Serves predictions** via a REST API (`POST /predict`, `POST /predict_itineraries`).
- **Visualizes them** in a single-page dashboard: fare-vs-days curve, weekday-by-month
  heatmap, airline breakdown, and per-flight predicted price trajectories.
- **Pulls live fares** from the Sky Scrapper API (RapidAPI) and overlays each live
  itinerary onto its model-predicted curve.

---

## Model

**Algorithm:** XGBoost (gradient-boosted trees) trained on log1p-transformed fares.

**Best model:** `results/xgboost_tuned.pkl` (selected via 50-trial Optuna sweep,
then retrained with early stopping).

| | |
|---|---|
| Total trees | 1143 (`best_iteration` = 1092) |
| `n_estimators` cap | 3000 |
| `early_stopping_rounds` | 50 |
| `learning_rate` | 0.0409 |
| `max_depth` | 9 |
| `subsample` | 0.618 |
| `colsample_bytree` | 0.555 |
| `min_child_weight` | 2 |
| `reg_lambda` | 0.599 |
| `tree_method` | `hist` |

**Test set performance** (last 20% of data, chronologically held out):

| Metric | Value |
|---|---|
| MAE | $41.52 |
| RMSE | $70.27 |
| R² (log space) | 0.8292 |
| R² (dollars) | 0.8111 |
| MAPE | 16.84% |

**Top features** by importance: `isBasicEconomy` (0.21), `be_x_days` (0.19),
`isNonStop` (0.10), `num_segments` (0.09), `route` (0.08). Full list in
`model_results.md`.

22 engineered + raw features total. See `data/features.txt` for the order
the model expects, and `infer.py:130` (`build_features`) for how each is
computed.

---

## Quickstart — run the dashboard

The fastest path: use the pre-trained model that's already in `results/` and
just start the API + serve the HTML.

### 1. Install Python deps

```bash
uv sync
```

(Uses `pyproject.toml`. If you don't have `uv`, install it from https://docs.astral.sh/uv/.)

### 2. Start the model API

```bash
uv run uvicorn model_api:app --host 0.0.0.0 --port 8000 --reload
```

Verify it's up:

```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true}
```

### 3. Get a RapidAPI key for Sky Scrapper (free)

The dashboard's "Live price lookup" panel uses Sky Scrapper. The key in
`flight_dashboard.html:761` is a placeholder — replace it with your own.

1. Sign up at https://rapidapi.com
2. Subscribe to https://rapidapi.com/apiheya/api/sky-scrapper (Basic plan, $0/month, ~20 requests/month)
3. From any endpoint's **Code Snippets** panel, copy the `x-rapidapi-key` value
4. Paste it into `flight_dashboard.html`:
   ```js
   const RAPIDAPI_KEY = 'YOUR_KEY_HERE';
   ```

### 4. Serve the dashboard

`flight_dashboard.html` must be served over HTTP (not opened as a `file://`
URL — browsers block `fetch()` from `file://`). Easiest:

```bash
python -m http.server 5500
```

Then open http://localhost:5500/flight_dashboard.html.

### 5. Use it

- Pick an origin airport from the top bar → click **Analyze route**. The metric
  cards, chart, heatmap, airline list, and insights all populate from the
  trained model.
- In the bottom panel, pick a date and click **Check live prices**. The dashboard
  hits Sky Scrapper for real itineraries, then asks the model for a per-flight
  predicted-price curve anchored to each live price.

If a route returns no results, retry — Sky Scrapper occasionally returns empty
on the free tier. To verify an airport's entityId, open DevTools and run
`forceResolveSkyId('ORD')` (each call uses 1 request, then caches the result).

---

## Retraining from scratch

Skip this section if you only want to use the existing model.

### 1. Install deps

Same as quickstart step 1.

### 2. Download the dataset

The full Expedia dataset is ~5.9 GB (6M rows). You'll need a Kaggle account
and API token (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars).

```bash
uv run python data.py
```

### 3. Filter to Chicago routes

```bash
uv run python filter_chicago.py
```

Produces `data/chicago_itineraries.csv`.

### 4. Preprocess

```bash
# 500k row sample (fast, for iteration)
uv run python preprocess.py

# OR full ~6M dataset (slower, more accurate)
FULL_DATA=1 uv run python preprocess.py     # bash/zsh
$env:FULL_DATA=1; uv run python preprocess.py   # PowerShell
```

Outputs splits to `data/`: `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`,
`features.txt`. Uses an **80/20 temporal split** — first 80% by date for
training, last 20% for test. Target encoding is fit on train only, applied to
test, to avoid leakage.

### 5. Train

```bash
# Baseline (default hyperparameters)
uv run python train_model.py

# OR Optuna-tuned (50 trials Bayesian search, ~10–15 min)
uv run python tune_model.py
```

Both write to `results/`. The tuned run also produces:
- `xgboost_tuned.pkl` — the model the API serves
- `best_params.json` — winning hyperparameter set
- `feature_importance_tuned.png`, `optuna_history.png`, `pred_vs_actual.png` — plots
- `model_results.md` — metrics summary

### 6. Compare baseline vs tuned

```bash
uv run python compare_models.py
```

---

## Architecture

```
                            ┌─────────────────────────┐
                            │    flight_dashboard     │
                            │       (browser)         │
                            └──┬──────────────────┬───┘
                               │                  │
            ┌──────────────────┘                  └────────────────┐
            ▼                                                       ▼
  POST /predict                                          GET /searchFlights
  POST /predict_itineraries                              GET /searchIncomplete
            │                                                       │
            ▼                                                       ▼
  ┌───────────────────────┐                            ┌───────────────────────┐
  │   model_api.py        │                            │   sky-scrapper        │
  │   (FastAPI :8000)     │                            │   (RapidAPI)          │
  └───────────┬───────────┘                            └───────────────────────┘
              │
              ▼
  ┌───────────────────────┐
  │   infer.py            │
  │   build_features()    │
  │   predict_route()     │
  │   predict_itinerary() │
  └───────────┬───────────┘
              │
              ▼
  ┌───────────────────────┐
  │ xgboost_tuned.pkl     │
  └───────────────────────┘
```

| File | Role |
|---|---|
| `data.py` | Download Expedia dataset from Kaggle |
| `filter_chicago.py` | Subset rows to ORD-related itineraries |
| `preprocess.py` | Engineer features, target-encode, 80/20 temporal split |
| `train_model.py` | Train baseline XGBoost |
| `tune_model.py` | Optuna sweep + retrain best |
| `retrain.py` | Retrain with a fixed param set |
| `compare_models.py` | Side-by-side metrics for baseline vs tuned |
| `infer.py` | Feature builder + prediction routines (used by the API) |
| `model_api.py` | FastAPI server exposing `/predict`, `/predict_itineraries`, `/health` |
| `flight_dashboard.html` | Single-page dashboard |

---

## API reference

### `POST /predict`

Route-level summary used to populate the main dashboard.

**Request:**
```json
{ "origin": "LAX", "destination": "ORD" }
```

**Response:** `avgFare`, `window` (e.g. `"21–35 days"`), `cheapDay`, `cheapDayFare`,
`cheapMonth`, `cheapMonthFare`, `fares` (12 values for booking windows
`[1, 3, 7, 14, 21, 28, 35, 42, 56, 70, 90, 120]` days), `airlines`, `heatmap` (7×7).

### `POST /predict_itineraries`

Per-flight curves anchored to live prices. Used by the live-price cards.

**Request:**
```json
{
  "items": [
    {
      "origin": "LAX",
      "destination": "ORD",
      "departure_date": "2026-05-31",
      "departure_hour": 12,
      "is_nonstop": true,
      "num_segments": 1,
      "duration_minutes": 250,
      "is_basic_economy": false,
      "airline_name": "American",
      "current_price": 420
    }
  ]
}
```

**Response:** `results[].rawCurve` (model output, un-anchored),
`results[].anchoredCurve` (multiplicatively scaled so it passes through
`current_price` at today's days-out), plus `anchorScale`, `modelPriceAtNow`,
and `todayDaysOut`.

The *shape* of the curve comes from the model; the *level* is anchored to the
live observed price. This is how the dashboard reconciles a static historical
model with live market data without retraining.

### `GET /health`

```json
{"status":"ok","model_loaded":true}
```

---

## Troubleshooting

- **`ERR_CONNECTION_REFUSED` on :8000** — model API isn't running. Start step 2.
- **`fetch()` fails when opening HTML directly** — open via `http://` not `file://`.
  Run `python -m http.server 5500`.
- **Sky Scrapper returns 403** — you're not subscribed to the API on the account
  whose key you're using. Subscribe to Basic (free) on the API page.
- **Sky Scrapper returns 200 but empty `itineraries`** — either the route+date
  has no flights, or the polling exited too early. Try a different date or click
  again. Free tier (~20 requests/month) is also flaky.
- **"No flights found" but model API works** — usually a Sky Scrapper quota issue.
  Check rate-limit headers in DevTools Network tab.
- **Wrong entityIds in `skyIds`** — the hardcoded table in `flight_dashboard.html`
  is mostly *city*-level entityIds, not airports. To fix one, run
  `forceResolveSkyId('ORD')` in DevTools console (1 request, cached after that).

---

## Notes / known limitations

- The Sky Scrapper key is hardcoded in `flight_dashboard.html`. Anyone with
  the file gets your quota. For anything beyond local demo, move the live-price
  call server-side.
- `seatsRemaining` and `total_travel_distance` aren't returned by Sky Scrapper —
  the per-flight predictor falls back to training-set means for those.
- The model was trained on April–October 2022 fares; predictions for routes
  or seasons outside that range degrade.
