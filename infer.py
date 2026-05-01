"""
infer.py — FlightLens inference script
Mirrors preprocess.py exactly so the model receives features in the same
format it was trained on.

Usage (standalone test):
    python infer.py

Used by model_api.py via:
    from infer import predict_route
"""

import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# ── Load artifacts ────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
DATA_DIR    = Path("data")

def _load_model():
    for name in ["xgboost_tuned.pkl", "xgboost_model.pkl"]:
        p = RESULTS_DIR / name
        if p.exists():
            with open(p, "rb") as f:
                print(f"[infer] Loaded model: {p}")
                return pickle.load(f)
    raise FileNotFoundError(
        "No model found in results/. Run train_model.py or tune_model.py first."
    )

def _load_encoders():
    """
    Load the TargetEncoder fitted on training data.
    preprocess.py saves X_train.csv with route + primary_airline already
    target-encoded, so we need the encoder object if it was saved, OR we
    re-derive the target-encoded means from X_train.csv directly.

    Strategy: read X_train.csv and use the column means as a lookup table.
    This exactly reproduces what TargetEncoder produces at inference time
    (it collapses to the training mean for known categories).
    """
    train_path = DATA_DIR / "X_train.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            "data/X_train.csv not found. Run preprocess.py first."
        )
    X_train = pd.read_csv(train_path)
    return X_train   # we use this to get column means / feature order

MODEL    = _load_model()
X_TRAIN  = _load_encoders()
# Trust the model's own feature_names — X_train.csv on disk may have drifted
# from what the persisted model was actually trained on.
_booster_feats = MODEL.get_booster().feature_names
FEATURES = _booster_feats if _booster_feats else X_TRAIN.columns.tolist()

# ── Precompute target-encoded lookup tables from training data ─────────────────
# preprocess.py target-encodes 'route' and 'primary_airline'.
# At inference time TargetEncoder maps unseen categories to the global mean.
# We replicate this by reading the encoded values from X_train.csv and
# computing per-category means (which IS what TargetEncoder stores internally).
#
# Since X_train already has numeric values for route/primary_airline (post-encoding),
# we need to re-read the RAW training CSV to build the lookup. If chicago_itineraries.csv
# is available we build proper maps; otherwise we fall back to column means.

def _build_te_maps():
    """Build route → encoded_value and primary_airline → encoded_value maps."""
    raw_path = DATA_DIR / "chicago_itineraries.csv"
    te_mean_route    = float(X_TRAIN["route"].mean())     if "route"           in X_TRAIN.columns else 0.0
    te_mean_airline  = float(X_TRAIN["primary_airline"].mean()) if "primary_airline" in X_TRAIN.columns else 0.0

    route_map   = {}
    airline_map = {}

    if raw_path.exists():
        raw = pd.read_csv(raw_path, usecols=["startingAirport", "destinationAirport",
                                              "segmentsAirlineName", "totalFare"],
                          nrows=500_000)
        raw.dropna(subset=["totalFare"], inplace=True)
        raw["log_fare"] = np.log1p(raw["totalFare"])

        # Route map
        raw["route"] = raw["startingAirport"].astype(str) + "_" + raw["destinationAirport"].astype(str)
        route_map = raw.groupby("route")["log_fare"].mean().to_dict()

        # Airline map
        raw["primary_airline"] = raw["segmentsAirlineName"].astype(str).str.split("||", regex=False).str[0].str.strip()
        top10 = raw["primary_airline"].value_counts().nlargest(10).index
        raw["primary_airline"] = raw["primary_airline"].where(raw["primary_airline"].isin(top10), "Other")
        airline_map = raw.groupby("primary_airline")["log_fare"].mean().to_dict()

    return route_map, airline_map, te_mean_route, te_mean_airline

ROUTE_MAP, AIRLINE_MAP, TE_MEAN_ROUTE, TE_MEAN_AIRLINE = _build_te_maps()

# ── Airport label encoding — must match preprocess.py ─────────────────────────
# preprocess.py fits LabelEncoder on all values seen in both train + test splits.
# Since we're working with a fixed set of Chicago routes, we derive the encoding
# from the unique airports seen in X_train (the encoded integers are already there,
# but we need to know which string → int). We reconstruct from the raw CSV.
def _build_airport_encoding():
    raw_path = DATA_DIR / "chicago_itineraries.csv"
    if raw_path.exists():
        raw = pd.read_csv(raw_path, usecols=["startingAirport", "destinationAirport"], nrows=10_000)
        all_airports = sorted(
            pd.concat([raw["startingAirport"], raw["destinationAirport"]]).dropna().astype(str).unique()
        )
        return {ap: i for i, ap in enumerate(all_airports)}
    # Fallback: known Chicago-route airports in alphabetical order
    airports = sorted(["ATL","BOS","CLT","DEN","DFW","DTW","EWR","IAD",
                        "JFK","LAX","LGA","MIA","OAK","ORD","PHL","SFO"])
    return {ap: i for i, ap in enumerate(airports)}

AIRPORT_ENC = _build_airport_encoding()

# ── Column means for filling unknowns ────────────────────────────────────────
COL_MEANS = X_TRAIN.mean(numeric_only=True).to_dict()

# ── Feature builder — mirrors preprocess.py steps 3-15 ───────────────────────
TOP_AIRLINES = [
    "American Airlines", "Delta", "Spirit Airlines", "United",
    "Southwest Airlines", "JetBlue Airways", "Alaska Airlines",
    "Frontier Airlines", "Allegiant Air", "Other",
]

def build_features(
    origin: str,          # e.g. "LAX"
    destination: str,     # e.g. "ORD"
    days_until_flight: int,
    departure_month: int,        # 1-12
    departure_day_of_week: int,  # 0=Mon … 6=Sun
    is_nonstop: int = 1,
    is_basic_economy: int = 0,
    seats_remaining: int = 5,
    num_segments: int = 1,
    trip_duration_minutes: float = None,   # will be imputed if None
    total_travel_distance: float = None,   # will be imputed if None
    primary_airline: str = "United",
    departure_hour: float = 12.0,
    search_day_of_week: int = 2,           # Wednesday baseline
) -> pd.DataFrame:
    """
    Build a single-row DataFrame with features in the exact order FEATURES expects.
    Mirrors every engineering step in preprocess.py.
    """
    # Impute distance/duration from training column means if not provided
    if total_travel_distance is None:
        total_travel_distance = COL_MEANS.get("log_distance", 6.5)
        total_travel_distance = np.expm1(total_travel_distance)
    if trip_duration_minutes is None:
        trip_duration_minutes = np.expm1(COL_MEANS.get("log_duration", 5.0))

    # ── Clip / sanitize ──
    days_until_flight     = max(0, days_until_flight)
    seats_remaining       = max(0, seats_remaining)
    total_travel_distance = max(0, total_travel_distance)
    trip_duration_minutes = max(0, trip_duration_minutes)

    # ── Temporal ──
    log_days = np.log1p(days_until_flight)
    dow_sin  = np.sin(2 * np.pi * departure_day_of_week / 7)
    dow_cos  = np.cos(2 * np.pi * departure_day_of_week / 7)

    # ── Log-transformed inputs ──
    log_distance = np.log1p(total_travel_distance)
    log_duration = np.log1p(trip_duration_minutes)

    # ── Seats ──
    seats_at_max = int(seats_remaining >= 9)

    # ── Interaction features ──
    urgency_scarcity   = np.log1p(days_until_flight) * np.log1p(seats_remaining)
    nonstop_x_distance = float(is_nonstop) * total_travel_distance
    be_x_days          = float(is_basic_economy) * days_until_flight

    # ── Airport label encoding ──
    origin_enc = AIRPORT_ENC.get(origin.upper(), 0)
    dest_enc   = AIRPORT_ENC.get(destination.upper(), 0)

    # ── Route target encoding ──
    route_key = f"{origin.upper()}_{destination.upper()}"
    route_enc = ROUTE_MAP.get(route_key, TE_MEAN_ROUTE)

    # ── Airline normalisation + target encoding ──
    airline_clean = primary_airline.strip()
    if airline_clean not in TOP_AIRLINES:
        airline_clean = "Other"
    airline_enc = AIRLINE_MAP.get(airline_clean, TE_MEAN_AIRLINE)

    # ── Assemble row dict with ALL possible columns ──
    row = {
        "departure_hour":        departure_hour,
        "days_until_flight":     days_until_flight,
        "log_days":              log_days,
        "departure_month":       departure_month,
        "departure_day_of_week": departure_day_of_week,
        "search_day_of_week":    search_day_of_week,
        "dow_sin":               dow_sin,
        "dow_cos":               dow_cos,
        "startingAirport":       origin_enc,
        "destinationAirport":    dest_enc,
        "route":                 route_enc,
        "isNonStop":             int(is_nonstop),
        "isBasicEconomy":        int(is_basic_economy),
        "seatsRemaining":        seats_remaining,
        "seats_at_max":          seats_at_max,
        "num_segments":          num_segments,
        "log_duration":          log_duration,
        "log_distance":          log_distance,
        "primary_airline":       airline_enc,
        "urgency_scarcity":      urgency_scarcity,
        "nonstop_x_distance":    nonstop_x_distance,
        "be_x_days":             be_x_days,
    }

    # Build DataFrame with only the columns the model was trained on, in order
    df = pd.DataFrame([row])
    for col in FEATURES:
        if col not in df.columns:
            df[col] = COL_MEANS.get(col, 0.0)   # fill any missing with training mean
    df = df[FEATURES]
    return df


def _predict_single(df: pd.DataFrame) -> float:
    """Run model on a prepared feature row and return dollar fare."""
    log_pred = MODEL.predict(df)[0]
    return float(np.expm1(log_pred))


# ── Public API ────────────────────────────────────────────────────────────────
BOOKING_WINDOWS = [1, 3, 7, 14, 21, 28, 35, 42, 56, 70, 90, 120]
MONTHS          = [4, 5, 6, 7, 8, 9, 10]   # Apr–Oct
WEEKDAYS        = list(range(7))            # 0=Mon … 6=Sun
DAY_NAMES       = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
MONTH_NAMES     = ["April","May","June","July","August","September","October"]

AIRLINE_DISPLAY = {
    "American Airlines": "American", "Delta": "Delta",
    "Spirit Airlines": "Spirit",     "United": "United",
    "Southwest Airlines": "Southwest", "JetBlue Airways": "JetBlue",
    "Alaska Airlines": "Alaska",     "Frontier Airlines": "Frontier",
    "Allegiant Air": "Allegiant",    "Other": "Other",
}

ROUTE_AIRLINES = {
    "LAX": ["United","American Airlines","Delta","Spirit Airlines","Alaska Airlines","JetBlue Airways"],
    "JFK": ["JetBlue Airways","United","Delta","American Airlines","Spirit Airlines","Alaska Airlines"],
    "MIA": ["American Airlines","Spirit Airlines","United","Delta","Frontier Airlines","JetBlue Airways"],
    "SFO": ["United","Southwest Airlines","Delta","Alaska Airlines","American Airlines","JetBlue Airways"],
    "BOS": ["JetBlue Airways","United","American Airlines","Delta","Spirit Airlines","Alaska Airlines"],
    "ATL": ["Delta","Spirit Airlines","Frontier Airlines","Southwest Airlines","United","American Airlines"],
    "DFW": ["American Airlines","Spirit Airlines","Frontier Airlines","United","Southwest Airlines","Delta"],
    "DEN": ["Frontier Airlines","United","Southwest Airlines","Spirit Airlines","American Airlines","Delta"],
    "EWR": ["United","Spirit Airlines","JetBlue Airways","Delta","American Airlines","Alaska Airlines"],
    "PHL": ["American Airlines","Spirit Airlines","Frontier Airlines","United","Delta","JetBlue Airways"],
    "DTW": ["Delta","Spirit Airlines","Frontier Airlines","United","American Airlines","Southwest Airlines"],
    "LGA": ["Delta","United","American Airlines","Spirit Airlines","JetBlue Airways","Alaska Airlines"],
    "IAD": ["United","Spirit Airlines","Frontier Airlines","Southwest Airlines","American Airlines","Delta"],
    "OAK": ["Southwest Airlines","United","Alaska Airlines","Spirit Airlines","Delta","American Airlines"],
    "CLT": ["American Airlines","Spirit Airlines","Frontier Airlines","United","Delta","Southwest Airlines"],
}


def predict_route(origin: str, destination: str = "ORD") -> dict:
    """
    Run all model queries needed to populate the FlightLens dashboard.
    Returns a dict matching the API response schema.
    """
    origin = origin.upper()
    destination = destination.upper()

    # ── 1. Fare curve — vary days_until_flight ────────────────────────────────
    fares = []
    for days in BOOKING_WINDOWS:
        df = build_features(origin, destination, days_until_flight=days,
                            departure_month=9, departure_day_of_week=1)  # Sep, Tue baseline
        fares.append(round(_predict_single(df)))

    avg_fare = round(sum(fares) / len(fares))

    # Identify sweet spot (minimum region)
    min_idx = fares.index(min(fares))
    window_labels = {
        0:"1–3 days", 1:"1–3 days", 2:"7–14 days", 3:"14–21 days",
        4:"21–35 days", 5:"21–35 days", 6:"28–42 days", 7:"35–56 days",
        8:"56–70 days", 9:"70–90 days", 10:"90–120 days", 11:"90–120 days",
    }
    window = window_labels.get(min_idx, "21–35 days")

    # ── 2. Cheapest day of week ───────────────────────────────────────────────
    day_fares = []
    for dow in WEEKDAYS:
        df = build_features(origin, destination, days_until_flight=28,
                            departure_month=9, departure_day_of_week=dow)
        day_fares.append(_predict_single(df))
    cheap_dow      = int(np.argmin(day_fares))
    cheap_day      = DAY_NAMES[cheap_dow]
    cheap_day_fare = round(day_fares[cheap_dow])

    # ── 3. Cheapest month ─────────────────────────────────────────────────────
    month_fares = []
    for m in MONTHS:
        df = build_features(origin, destination, days_until_flight=28,
                            departure_month=m, departure_day_of_week=cheap_dow)
        month_fares.append(_predict_single(df))
    cheap_month_idx  = int(np.argmin(month_fares))
    cheap_month      = MONTH_NAMES[cheap_month_idx]
    cheap_month_fare = round(month_fares[cheap_month_idx])

    # ── 4. Heatmap: 7 weekdays × 7 months ────────────────────────────────────
    heatmap = []
    for dow in WEEKDAYS:
        row_vals = []
        for m in MONTHS:
            df = build_features(origin, destination, days_until_flight=28,
                                departure_month=m, departure_day_of_week=dow)
            row_vals.append(round(_predict_single(df)))
        heatmap.append(row_vals)

    # ── 5. Airline breakdown ──────────────────────────────────────────────────
    airlines_for_route = ROUTE_AIRLINES.get(origin, list(AIRLINE_MAP.keys())[:6])
    airlines = []
    for airline_full in airlines_for_route:
        df = build_features(origin, destination, days_until_flight=28,
                            departure_month=9, departure_day_of_week=cheap_dow,
                            primary_airline=airline_full)
        fare = round(_predict_single(df))
        if fare < avg_fare - 15:
            cls = "badge-green"
        elif fare > avg_fare + 30:
            cls = "badge-red"
        else:
            cls = "badge-amber"
        airlines.append({
            "name": AIRLINE_DISPLAY.get(airline_full, airline_full),
            "fare": fare,
            "cls":  cls,
        })

    return {
        "avgFare":        avg_fare,
        "window":         window,
        "cheapDay":       cheap_day,
        "cheapDayFare":   cheap_day_fare,
        "cheapMonth":     cheap_month,
        "cheapMonthFare": cheap_month_fare,
        "fares":          fares,
        "airlines":       airlines,
        "heatmap":        heatmap,
    }


def _interp_curve(curve_pts: list, days_out: float) -> float:
    """Linear interp on log-x of a fare curve [{'daysOut': d, 'price': p}, ...]."""
    if not curve_pts:
        return 0.0
    xs = [c["daysOut"] for c in curve_pts]
    ys = [c["price"]   for c in curve_pts]
    if days_out <= xs[0]:  return ys[0]
    if days_out >= xs[-1]: return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= days_out <= xs[i + 1]:
            t = (np.log(days_out) - np.log(xs[i])) / (np.log(xs[i + 1]) - np.log(xs[i]))
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


# Crude name normalization: Sky Scrapper carrier names -> the form the model knows.
_CARRIER_NORMALIZE = {
    "american": "American Airlines", "american airlines": "American Airlines",
    "delta": "Delta", "delta air lines": "Delta",
    "united": "United", "united airlines": "United",
    "spirit": "Spirit Airlines", "spirit airlines": "Spirit Airlines",
    "jetblue": "JetBlue Airways", "jetblue airways": "JetBlue Airways",
    "alaska": "Alaska Airlines", "alaska airlines": "Alaska Airlines",
    "southwest": "Southwest Airlines", "southwest airlines": "Southwest Airlines",
    "frontier": "Frontier Airlines", "frontier airlines": "Frontier Airlines",
    "allegiant": "Allegiant Air", "allegiant air": "Allegiant Air",
}


def predict_itinerary(
    origin: str,
    destination: str,
    departure_date: str,        # ISO 'YYYY-MM-DD'
    departure_hour: float = 12.0,
    is_nonstop: bool = True,
    num_segments: int = 1,
    duration_minutes: float = None,
    is_basic_economy: bool = False,
    airline_name: str = "United",
    current_price: float = None,
) -> dict:
    """
    Per-itinerary fare curve over BOOKING_WINDOWS, optionally anchored
    multiplicatively to the live `current_price` so that
        anchored_curve(today_days_out) == current_price
    while preserving the model's predicted shape.
    """
    from datetime import datetime, date

    dep = datetime.fromisoformat(departure_date).date()
    today = date.today()
    today_days_out = max(0, (dep - today).days)

    # Normalize carrier name to a form the target encoder knows
    key = (airline_name or "").strip().lower()
    primary_airline = _CARRIER_NORMALIZE.get(key, airline_name or "United")

    # Sweep BOOKING_WINDOWS to get the model's raw curve
    raw_curve = []
    for d in BOOKING_WINDOWS:
        df = build_features(
            origin=origin, destination=destination,
            days_until_flight=d,
            departure_month=dep.month,
            departure_day_of_week=dep.weekday(),
            is_nonstop=int(bool(is_nonstop)),
            is_basic_economy=int(bool(is_basic_economy)),
            num_segments=num_segments,
            trip_duration_minutes=duration_minutes,
            primary_airline=primary_airline,
            departure_hour=departure_hour,
            search_day_of_week=today.weekday(),
        )
        raw_curve.append({"daysOut": d, "price": round(_predict_single(df), 2)})

    # Anchor to live price (multiplicative scale on log-x interp)
    anchored = [dict(p) for p in raw_curve]
    scale = 1.0
    if current_price and current_price > 0:
        anchor_price = _interp_curve(raw_curve, max(today_days_out, BOOKING_WINDOWS[0]))
        if anchor_price > 0:
            scale = float(current_price) / anchor_price
            for p in anchored:
                p["price"] = round(p["price"] * scale, 2)

    return {
        "origin":            origin,
        "destination":       destination,
        "todayDaysOut":      today_days_out,
        "departureDate":     departure_date,
        "primaryAirline":    primary_airline,
        "isNonStop":         bool(is_nonstop),
        "currentPrice":      current_price,
        "anchorScale":       round(scale, 4),
        "modelPriceAtNow":   round(_interp_curve(raw_curve, max(today_days_out, BOOKING_WINDOWS[0])), 2),
        "rawCurve":          raw_curve,
        "anchoredCurve":     anchored,
    }


def predict_itineraries(items: list) -> list:
    """Batch wrapper — accepts a list of itinerary dicts, returns a list of curves."""
    out = []
    for it in items:
        out.append(predict_itinerary(**it))
    return out


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    print("\nRunning smoke test: LAX → ORD")
    result = predict_route("LAX", "ORD")
    print(json.dumps(result, indent=2))
    print(f"\nAvg fare: ${result['avgFare']}")
    print(f"Best window: {result['window']}")
    print(f"Cheapest day: {result['cheapDay']} (${result['cheapDayFare']})")
    print(f"Cheapest month: {result['cheapMonth']} (${result['cheapMonthFare']})")
    print(f"Fare curve: {result['fares']}")
