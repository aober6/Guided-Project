"""
Preprocessing pipeline for dilwong/flightprices dataset.
Target variable: totalFare (log1p-transformed)
"""
import os
import re
import numpy as np
import pandas as pd
import kagglehub
from sklearn.preprocessing import LabelEncoder, TargetEncoder

DATA_DIR = "data"
CHICAGO_CSV_PATH = os.path.join(DATA_DIR, "chicago_itineraries.csv")
os.makedirs(DATA_DIR, exist_ok=True)

TARGET = "totalFare"

# ── 1. Load ───────────────────────────────────────────────────────────────────
# Always use the Chicago-filtered dataset. Run filter_chicago.py once to create it.
if not os.path.exists(CHICAGO_CSV_PATH):
    raise FileNotFoundError(
        f"{CHICAGO_CSV_PATH} not found. Run: uv run python filter_chicago.py"
    )

print(f"Loading {CHICAGO_CSV_PATH} ...")
df = pd.read_csv(CHICAGO_CSV_PATH)
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

# ── 2. Drop rows with missing target ─────────────────────────────────────────
df.dropna(subset=[TARGET], inplace=True)


# ── 3. Engineer features from dates ──────────────────────────────────────────
df["searchDate"] = pd.to_datetime(df["searchDate"], errors="coerce")
df["flightDate"]  = pd.to_datetime(df["flightDate"],  errors="coerce")

df["days_until_flight"]     = (df["flightDate"] - df["searchDate"]).dt.days
df["departure_month"]       = df["flightDate"].dt.month
df["departure_day_of_week"] = df["flightDate"].dt.dayofweek  # 0=Mon
df["search_day_of_week"]    = df["searchDate"].dt.dayofweek

# ── 4. num_segments from pipe-delimited segment column ────────────────────────
if "segmentsDepartureAirportCode" in df.columns:
    df["num_segments"] = (
        df["segmentsDepartureAirportCode"]
        .astype(str)
        .str.count(r"\|\|") + 1
    )

# ── 5. trip_duration_minutes from ISO 8601 travelDuration ────────────────────
def parse_duration_minutes(s):
    if pd.isna(s):
        return np.nan
    h = re.search(r"(\d+)H", str(s))
    m = re.search(r"(\d+)M", str(s))
    return int(h.group(1)) * 60 + (int(m.group(1)) if m else 0) if h else (int(m.group(1)) if m else np.nan)

if "travelDuration" in df.columns:
    df["trip_duration_minutes"] = df["travelDuration"].apply(parse_duration_minutes)

# ── 6. totalTravelDistance — impute missing with route median ─────────────────
if "totalTravelDistance" in df.columns:
    route_median = df.groupby(["startingAirport", "destinationAirport"])["totalTravelDistance"].transform("median")
    df["totalTravelDistance"] = df["totalTravelDistance"].fillna(route_median)
    df["totalTravelDistance"] = df["totalTravelDistance"].fillna(df["totalTravelDistance"].median())

# ── 7. primary_airline — keep top 10, group rest as "Other" ──────────────────
if "segmentsAirlineName" in df.columns:
    df["primary_airline"] = df["segmentsAirlineName"].astype(str).str.split("||", regex=False).str[0].str.strip()
    top_airlines = df["primary_airline"].value_counts().nlargest(10).index
    df["primary_airline"] = df["primary_airline"].where(df["primary_airline"].isin(top_airlines), "Other")

# ── 8. Route combined feature ─────────────────────────────────────────────────
df["route"] = df["startingAirport"].astype(str) + "_" + df["destinationAirport"].astype(str)

# ── 9. Departure hour from raw segment departure time ─────────────────────────
def parse_departure_hour(s):
    if pd.isna(s):
        return np.nan
    first_seg = str(s).split("||")[0].strip()
    m = re.search(r"T(\d{2}):", first_seg)
    return int(m.group(1)) if m else np.nan

if "segmentsDepartureTimeRaw" in df.columns:
    df["departure_hour"] = df["segmentsDepartureTimeRaw"].apply(parse_departure_hour)
    # Bucket: 0=red-eye(0-5), 1=morning(6-9), 2=midday(10-15), 3=evening(16-19), 4=night(20-23)
    df["hour_bucket"] = pd.cut(
        df["departure_hour"].fillna(12),
        bins=[-1, 5, 9, 15, 19, 23],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)

# ── 10. Booking window bucket ─────────────────────────────────────────────────
df["booking_window_bucket"] = pd.cut(
    df["days_until_flight"].fillna(0),
    bins=[-1, 3, 7, 14, 21, 45, 90, 120, 9999],
    labels=[0, 1, 2, 3, 4, 5, 6, 7]
).astype(float)

# ── 11. Interaction features ──────────────────────────────────────────────────
df["urgency_scarcity"]  = np.log1p(df["days_until_flight"].clip(lower=0)) * np.log1p(df["seatsRemaining"].clip(lower=0))
df["nonstop_x_distance"] = df["isNonStop"].astype(float) * df["totalTravelDistance"].fillna(0)
df["be_x_days"]          = df["isBasicEconomy"].astype(float) * df["days_until_flight"].clip(lower=0).fillna(0)

# ── 12. Log-transform right-skewed inputs ─────────────────────────────────────
df["log_distance"] = np.log1p(df["totalTravelDistance"].clip(lower=0))
df["log_duration"] = np.log1p(df["trip_duration_minutes"].clip(lower=0))
df["log_days"]     = np.log1p(df["days_until_flight"].clip(lower=0))

# ── 13. Cyclical encoding for day-of-week and month ──────────────────────────
df["dow_sin"]   = np.sin(2 * np.pi * df["departure_day_of_week"] / 7)
df["dow_cos"]   = np.cos(2 * np.pi * df["departure_day_of_week"] / 7)
df["month_sin"] = np.sin(2 * np.pi * (df["departure_month"] - 1) / 12)
df["month_cos"] = np.cos(2 * np.pi * (df["departure_month"] - 1) / 12)

# ── 14. seatsRemaining — clip at 9 (Expedia reports "9+" as 9) ───────────────
df["seats_capped"] = df["seatsRemaining"].clip(upper=9)
df["seats_at_max"] = (df["seatsRemaining"] >= 9).astype(int)

# ── 15. Select final feature set ──────────────────────────────────────────────
# Dropped redundant/noisy features:
#   isRefundable      — 1 positive case in dataset, pure noise
#   seats_capped      — r=0.9998 with seatsRemaining
#   trip_duration_minutes — r=0.9577 with log_duration (keep log version)
#   totalTravelDistance   — r=0.9559 with log_distance (keep log version)
#   booking_window_bucket — r=0.9497 with log_days (keep log version)
#   month_sin, month_cos  — perfectly collinear with departure_month on limited data
KEEP = [
    # Temporal
    "days_until_flight", "log_days",
    "departure_month", "departure_day_of_week", "search_day_of_week",
    "dow_sin", "dow_cos",
    # Route
    "startingAirport", "destinationAirport", "route",
    # Flight characteristics
    "isNonStop", "isBasicEconomy",
    "seatsRemaining", "seats_at_max",
    "num_segments", "log_duration", "log_distance",
    # Airline
    "primary_airline",
    # Interactions
    "urgency_scarcity", "nonstop_x_distance", "be_x_days",
    # Target
    TARGET,
]

# Include departure_hour if successfully parsed
if "departure_hour" in df.columns:
    KEEP = ["departure_hour"] + KEEP

df = df[[c for c in KEEP if c in df.columns]].copy()
print(f"\nUsing {df.shape[1]-1} features + target. Shape: {df.shape}")

# ── 16. Boolean columns -> int ────────────────────────────────────────────────
for col in ["isBasicEconomy", "isRefundable", "isNonStop"]:
    if col in df.columns:
        df[col] = df[col].astype(bool).astype(int)

# ── 17. Fill any remaining nulls with median before split ────────────────────
df = df.fillna(df.median(numeric_only=True))

# ── 18. Log1p-transform target ────────────────────────────────────────────────
df[TARGET] = np.log1p(df[TARGET])

# ── 19. Temporal train/test split (80/20 by row order) ───────────────────────
split_idx = int(len(df) * 0.8)
train_df  = df.iloc[:split_idx].copy()
test_df   = df.iloc[split_idx:].copy()

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test  = test_df.drop(columns=[TARGET])
y_test  = test_df[TARGET]

# ── 20. Label-encode low-cardinality categoricals (startingAirport, destinationAirport) ──
le_cols = ["startingAirport", "destinationAirport"]
le_cols = [c for c in le_cols if c in X_train.columns]
if le_cols:
    print(f"Label-encoding: {le_cols}")
    le = LabelEncoder()
    for col in le_cols:
        all_vals = pd.concat([X_train[col], X_test[col]]).astype(str)
        le.fit(all_vals)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col]  = le.transform(X_test[col].astype(str))

# ── 21. Target-encode high-cardinality categoricals ──────────────────────────
te_cols = [c for c in ["route", "primary_airline"] if c in X_train.columns]
if te_cols:
    print(f"Target-encoding: {te_cols}")
    te = TargetEncoder(target_type="continuous", smooth="auto", cv=5, random_state=42)
    X_train[te_cols] = te.fit_transform(X_train[te_cols], y_train)
    X_test[te_cols]  = te.transform(X_test[te_cols])

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
print(f"Target (log1p) -- mean: {y_train.mean():.3f}, std: {y_train.std():.3f}")
print(f"\nFeature stats:\n{X_train.describe().T[['mean','std','min','max']].to_string()}")

# ── 22. Save ──────────────────────────────────────────────────────────────────
X_train.to_csv(f"{DATA_DIR}/X_train.csv", index=False)
X_test.to_csv(f"{DATA_DIR}/X_test.csv",  index=False)
y_train.to_csv(f"{DATA_DIR}/y_train.csv", index=False)
y_test.to_csv(f"{DATA_DIR}/y_test.csv",  index=False)

with open(f"{DATA_DIR}/features.txt", "w") as f:
    f.write("\n".join(X_train.columns.tolist()))

print(f"\nFeatures used ({len(X_train.columns)}): {X_train.columns.tolist()}")
print(f"Saved splits to {DATA_DIR}/")
print("Preprocessing complete.")
