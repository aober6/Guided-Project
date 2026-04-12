"""
Preprocessing pipeline for dilwong/flightprices dataset.
Target variable: totalFare (log1p-transformed)

Final features (~12):
  days_until_flight, startingAirport, destinationAirport, isNonStop,
  isBasicEconomy, isRefundable, seatsRemaining, departure_month,
  departure_day_of_week, search_day_of_week, num_segments,
  trip_duration_minutes, totalTravelDistance, segmentsAirlineName
"""
import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

CSV_PATH = r"C:\Users\ishur\.cache\kagglehub\datasets\dilwong\flightprices\versions\1\itineraries.csv"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

TARGET = "totalFare"

# ── 1. Load ───────────────────────────────────────────────────────────────────
print(f"Loading {CSV_PATH} ...")
nrows = None if os.getenv("FULL_DATA") else 500_000
df = pd.read_csv(CSV_PATH, nrows=nrows)
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

# ── 2. Drop rows with missing target ─────────────────────────────────────────
df.dropna(subset=[TARGET], inplace=True)

# ── 3. Engineer features from dates ──────────────────────────────────────────
df["searchDate"] = pd.to_datetime(df["searchDate"], errors="coerce")
df["flightDate"]  = pd.to_datetime(df["flightDate"],  errors="coerce")

df["days_until_flight"]    = (df["flightDate"] - df["searchDate"]).dt.days
df["departure_month"]      = df["flightDate"].dt.month
df["departure_day_of_week"] = df["flightDate"].dt.dayofweek  # 0=Mon
df["search_day_of_week"]   = df["searchDate"].dt.dayofweek

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

# ── 7. segmentsAirlineName — keep top 10, group rest as "Other" ──────────────
if "segmentsAirlineName" in df.columns:
    # Pipe-delimited; take first segment's airline as the "primary" airline
    df["primary_airline"] = df["segmentsAirlineName"].astype(str).str.split("||", regex=False).str[0].str.strip()
    top_airlines = df["primary_airline"].value_counts().nlargest(10).index
    df["primary_airline"] = df["primary_airline"].where(df["primary_airline"].isin(top_airlines), "Other")

# ── 8. Select final feature set ───────────────────────────────────────────────
KEEP = [
    # Engineered temporal
    "days_until_flight", "departure_month", "departure_day_of_week", "search_day_of_week",
    # Route
    "startingAirport", "destinationAirport",
    # Flight characteristics
    "isNonStop", "isBasicEconomy", "isRefundable",
    "seatsRemaining", "num_segments", "trip_duration_minutes",
    "totalTravelDistance",
    # Airline
    "primary_airline",
    # Target
    TARGET,
]
df = df[[c for c in KEEP if c in df.columns]].copy()
print(f"\nUsing {df.shape[1]-1} features + target. Shape: {df.shape}")

# ── 9. Boolean columns → int ──────────────────────────────────────────────────
for col in ["isBasicEconomy", "isRefundable", "isNonStop"]:
    if col in df.columns:
        df[col] = df[col].astype(bool).astype(int)

# ── 10. Encode categoricals ───────────────────────────────────────────────────
cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
cat_cols = [c for c in cat_cols if c != TARGET]
print(f"Label-encoding: {cat_cols}")
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# ── 11. Fill any remaining nulls with median ─────────────────────────────────
df.fillna(df.median(numeric_only=True), inplace=True)

print(f"\nFeature stats:\n{df.drop(columns=[TARGET]).describe().T[['mean','std','min','max']].to_string()}")

# ── 12. Log1p-transform target ────────────────────────────────────────────────
df[TARGET] = np.log1p(df[TARGET])

# ── 13. Temporal train/test split (80/20 by row order = earlier/later dates) ──
split_idx = int(len(df) * 0.8)
train_df  = df.iloc[:split_idx]
test_df   = df.iloc[split_idx:]

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test  = test_df.drop(columns=[TARGET])
y_test  = test_df[TARGET]

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
print(f"Target (log1p) — mean: {y_train.mean():.3f}, std: {y_train.std():.3f}")

# ── 14. Save ──────────────────────────────────────────────────────────────────
X_train.to_csv(f"{DATA_DIR}/X_train.csv", index=False)
X_test.to_csv(f"{DATA_DIR}/X_test.csv",  index=False)
y_train.to_csv(f"{DATA_DIR}/y_train.csv", index=False)
y_test.to_csv(f"{DATA_DIR}/y_test.csv",  index=False)

with open(f"{DATA_DIR}/features.txt", "w") as f:
    f.write("\n".join(X_train.columns.tolist()))

print(f"\nFeatures used: {X_train.columns.tolist()}")
print(f"Saved splits to {DATA_DIR}/")
print("Preprocessing complete.")
