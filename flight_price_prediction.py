import re
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


dtypes = {
    "legId": "string",
    "searchDate": "string",
    "flightDate": "string",
    "startingAirport": "category",
    "destinationAirport": "category",
    "fareBasisCode": "string",
    "travelDuration": "string",
    "elapsedDays": "int8",
    "isBasicEconomy": "bool",
    "isRefundable": "bool",
    "isNonStop": "bool",
    "baseFare": "float32",
    "totalFare": "float32",
    "seatsRemaining": "int8",
    "totalTravelDistance": "float32",
    "segmentsDepartureTimeEpochSeconds": "string",
    "segmentsDepartureTimeRaw": "string",
    "segmentsArrivalTimeEpochSeconds": "string",
    "segmentsArrivalTimeRaw": "string",
    "segmentsArrivalAirportCode": "string",
    "segmentsDepartureAirportCode": "string",
    "segmentsAirlineName": "string",
    "segmentsAirlineCode": "string",
    "segmentsEquipmentDescription": "string",
    "segmentsDurationInSeconds": "string",
    "segmentsDistance": "string",
    "segmentsCabinCode": "category",
}

KEEP_COLS = [
    "searchDate", "flightDate", "startingAirport", "destinationAirport",
    "travelDuration", "isBasicEconomy", "isRefundable", "isNonStop",
    "totalFare", "seatsRemaining", "totalTravelDistance", "segmentsAirlineName",
]

chunks = []
for chunk in pd.read_csv("/Users/ljh/Desktop/itineraries.csv", dtype=dtypes, usecols=KEEP_COLS, chunksize=500_000):
    chunks.append(chunk[chunk["destinationAirport"] == "ORD"])

df = pd.concat(chunks, ignore_index=True)

print(f"Shape after load: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

# ── 1. Drop rows with missing totalFare ──────────────────────────────────────
df = df.dropna(subset=["totalFare"])
print(f"Shape after dropping missing totalFare: {df.shape}")

# ── 1b. Validate boolean columns have no nulls ───────────────────────────────
for col in ["isBasicEconomy", "isNonStop", "isRefundable"]:
    assert df[col].isna().sum() == 0, f"{col} has nulls"

# ── 1c. Cap totalFare at 99th percentile before log transform ────────────────
fare_cap = df["totalFare"].quantile(0.99)
before = len(df)
df = df[df["totalFare"] <= fare_cap]
print(f"Shape after fare outlier cap (>{fare_cap:.0f}): {df.shape}  (dropped {before - len(df):,} rows)")

# ── 2. Temporal features from searchDate and flightDate ─────────────────────
df["searchDate"] = pd.to_datetime(df["searchDate"])
df["flightDate"] = pd.to_datetime(df["flightDate"])

df["days_until_flight"]     = (df["flightDate"] - df["searchDate"]).dt.days
df["departure_month"]       = df["flightDate"].dt.month
df["departure_day_of_week"] = df["flightDate"].dt.dayofweek   # 0=Mon … 6=Sun
df["search_day_of_week"]    = df["searchDate"].dt.dayofweek

# Drop rows where search date is after flight date (data errors)
before = len(df)
df = df[df["days_until_flight"] >= 0]
print(f"Shape after dropping negative days_until_flight: {df.shape}  (dropped {before - len(df):,} rows)")

# Clip seatsRemaining to valid range
df["seatsRemaining"] = df["seatsRemaining"].clip(0, 9)

# ── 3. Parse ISO 8601 travelDuration → total minutes ────────────────────────
def parse_iso_duration(s):
    if pd.isna(s):
        return np.nan
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?", str(s))
    if not m:
        return np.nan
    hours   = int(m.group(1)) if m.group(1) else 0
    minutes = int(m.group(2)) if m.group(2) else 0
    return hours * 60 + minutes

df["trip_duration_minutes"] = df["travelDuration"].map(parse_iso_duration)
df["trip_duration_minutes"] = df["trip_duration_minutes"].fillna(df["trip_duration_minutes"].median())

# ── 4. Count ||-delimited segments ──────────────────────────────────────────
df["num_segments"] = (
    df["segmentsAirlineName"]
    .fillna("")
    .str.count(r"\|\|") + 1
)

# ── 5. Extract first-leg airline; keep top 10, label rest "Other" ────────────
df["primary_airline"] = (
    df["segmentsAirlineName"]
    .str.split("||", regex=False)
    .str[0]
    .str.strip()
)

top10_airlines = df["primary_airline"].value_counts().nlargest(10).index
df["primary_airline"] = df["primary_airline"].where(
    df["primary_airline"].isin(top10_airlines), other="Other"
)

# ── 6. Impute missing totalTravelDistance with per-route median ──────────────
route_median = (
    df.groupby(["startingAirport", "destinationAirport"])["totalTravelDistance"]
    .transform("median")
)
df["totalTravelDistance"] = df["totalTravelDistance"].fillna(route_median)
# Fallback for routes where every value is NaN
df["totalTravelDistance"] = df["totalTravelDistance"].fillna(df["totalTravelDistance"].median())

# ── 7. Label-encode categorical columns ─────────────────────────────────────
for col in ["startingAirport", "destinationAirport", "primary_airline"]:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# ── 8. Log1p-transform totalFare ─────────────────────────────────────────────
df["totalFare"] = np.log1p(df["totalFare"])

# ── 9. Temporal train/test split (80 / 20) ───────────────────────────────────
df = df.sort_values("searchDate").reset_index(drop=True)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df  = df.iloc[split_idx:].copy()

FEATURE_COLS = [
    "days_until_flight",
    "startingAirport",
    "destinationAirport",
    "isNonStop",
    "isBasicEconomy",
    "isRefundable",
    "seatsRemaining",
    "departure_month",
    "departure_day_of_week",
    "search_day_of_week",
    "num_segments",
    "trip_duration_minutes",
    "totalTravelDistance",
    "primary_airline",
]
TARGET_COL = "totalFare"

X_train = train_df[FEATURE_COLS]
y_train = train_df[TARGET_COL]
X_test  = test_df[FEATURE_COLS]
y_test  = test_df[TARGET_COL]

print(f"\nTrain size: {len(X_train):,}  |  Test size: {len(X_test):,}")
print(f"Train date range: {train_df['searchDate'].min().date()} → {train_df['searchDate'].max().date()}")
print(f"Test  date range: {test_df['searchDate'].min().date()} → {test_df['searchDate'].max().date()}")
print(f"\nFeature matrix shape — train: {X_train.shape}, test: {X_test.shape}")
print(f"Target (log1p-transformed) — train mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")
print("\nSample of preprocessed training rows:")
print(X_train.head(3))

# Training RandomForest Model
rf = RandomForestRegressor(
    n_estimators = 100,
    max_depth = 10,
    min_samples_split = 5,
    random_state = 42,
    n_jobs = -1
)

rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

mae = mean_absolute_error(np.expm1(y_test), np.expm1(predictions))
r2 = r2_score(y_test, predictions)
print(f"MAE: ${mae:.2f}")
print(f"R2: {r2:.4f}")

importances = rf.feature_importances_
for feature, importance in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
    print(f"{feature}: {importance:.4f}")

y_test_dollars = np.expm1(y_test)
pred_dollars    = np.expm1(predictions)

# ── Plot 1: Actual vs Predicted ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(y_test_dollars, pred_dollars, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
min_val = min(y_test_dollars.min(), pred_dollars.min())
max_val = max(y_test_dollars.max(), pred_dollars.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")

ax.text(0.05, 0.95, f'R² = {r2:.4f}\nMAE = ${mae:.2f}',
        transform=ax.transAxes, fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_xlabel('Actual Price ($)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Price ($)', fontsize=12, fontweight='bold')
ax.set_title('Random Forest: Actual vs Predicted Prices',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'actual_vs_predicted.png'), dpi=300)
plt.close()

# ── Plot 2: Feature Importance ────────────────────────────────────────────────
importances_sorted = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1])
feat_names, feat_vals = zip(*importances_sorted)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(feat_names, feat_vals, color='steelblue', edgecolor='k', linewidth=0.5)
ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_title('Random Forest: Feature Importance', fontsize=14, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=300)
plt.close()

# ── Plot 3: Residuals Analysis ────────────────────────────────────────────────
residuals = pred_dollars - y_test_dollars

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.scatter(y_test_dollars, residuals, alpha=0.4, s=20, edgecolors='k', linewidth=0.3)
ax1.axhline(0, color='r', linestyle='--', lw=2)
ax1.set_xlabel('Actual Price ($)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Residual (Predicted − Actual) ($)', fontsize=12, fontweight='bold')
ax1.set_title('Residuals vs Actual Price', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2.hist(residuals, bins=60, color='steelblue', edgecolor='k', linewidth=0.4)
ax2.axvline(0, color='r', linestyle='--', lw=2, label='Zero error')
ax2.axvline(residuals.mean(), color='orange', linestyle='-', lw=2,
            label=f'Mean = ${residuals.mean():.2f}')
ax2.set_xlabel('Residual ($)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
ax2.set_title('Distribution of Residuals', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.suptitle('Random Forest: Residuals Analysis', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'residuals.png'), dpi=300, bbox_inches='tight')
plt.close()

# ── Plot 4: Absolute Error vs Days Until Flight ───────────────────────────────
days  = test_df["days_until_flight"].values
abs_err = np.abs(residuals)

bucket_size = 10
max_day = int(days.max())
bucket_edges = np.arange(0, max_day + bucket_size, bucket_size)
bucket_centers, bucket_mae = [], []
for lo in bucket_edges[:-1]:
    hi   = lo + bucket_size
    mask = (days >= lo) & (days < hi)
    if mask.sum() >= 5:
        bucket_centers.append(lo + bucket_size / 2)
        bucket_mae.append(abs_err[mask].mean())

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(days, abs_err, alpha=0.2, s=10, color='steelblue', label='Individual error')
ax.plot(bucket_centers, bucket_mae, color='red', lw=2.5, marker='o',
        markersize=5, label=f'Mean MAE per {bucket_size}-day bucket')

ax.set_xlabel('Days Until Flight', fontsize=12, fontweight='bold')
ax.set_ylabel('Absolute Error ($)', fontsize=12, fontweight='bold')
ax.set_title('Prediction Error vs Days Until Flight', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'error_by_days.png'), dpi=300)
plt.close()

print(f"\nSaved all plots to: {OUTPUT_DIR}")