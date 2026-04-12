Here's a breakdown of which features to use and why:

---

## Recommended Features for Your Model

### ✅ Keep & Use Directly

| Feature | Reason |
|---|---|
| `searchDate` | Needed to calculate `days_until_flight` |
| `flightDate` | Needed to calculate `days_until_flight` and extract month/day-of-week |
| `startingAirport` | Defines the route — critical for Chicago-specific filtering |
| `destinationAirport` | Same as above |
| `isBasicEconomy` | Directly affects price — basic economy is almost always cheaper |
| `isRefundable` | Refundable tickets are priced differently |
| `isNonStop` | Non-stop vs. connecting significantly impacts fare |
| `seatsRemaining` | Low seats = higher price (scarcity pricing) |
| `totalFare` | **This is your target variable (what you're predicting)** |

---

### 🔧 Engineer From Existing Columns

These don't exist yet — you create them:

| New Feature | How to Create It | Why It Matters |
|---|---|---|
| `days_until_flight` | `flightDate - searchDate` | **Most important feature** — the core of your research question |
| `departure_month` | Extract month from `flightDate` | Captures seasonality (summer/holidays = expensive) |
| `departure_day_of_week` | Extract day-of-week from `flightDate` | Tuesday/Wednesday flights tend to be cheaper |
| `search_day_of_week` | Extract day-of-week from `searchDate` | Some evidence that searching midweek finds better deals |
| `num_segments` | Count `||` separators in `segmentsDepartureAirportCode` + 1 | Proxy for number of stops — more reliable than `isNonStop` alone |
| `trip_duration_minutes` | Parse `travelDuration` into total minutes | Longer trips with layovers affect price |

---

### ⚠️ Use With Caution

| Feature | Concern | What to Do |
|---|---|---|
| `baseFare` | Highly correlated with `totalFare` — will leak the target | **Drop it** if predicting `totalFare`. Use only for EDA |
| `totalTravelDistance` | Has missing values | Impute with route median, or drop rows if too many missing |
| `segmentsAirlineName` | Categorical with many values | One-hot encode or label encode top airlines, group rest as "Other" |
| `segmentsAirlineCode` | Redundant with `segmentsAirlineName` | Pick one, drop the other |
| `fareBasisCode` | Very high cardinality, hard to interpret | Drop unless you have time to group into fare families |

---

### ❌ Drop These

| Feature | Reason |
|---|---|
| `legId` | Just an identifier — no predictive value |
| `elapsedDays` | Almost always 0, no variance |
| `segmentsDepartureTimeEpochSeconds` | Redundant with the raw time columns |
| `segmentsArrivalTimeEpochSeconds` | Same — use the Raw versions if needed |
| `segmentsDepartureTimeRaw` | Only useful if engineering departure hour (optional) |
| `segmentsArrivalTimeRaw` | Same as above |
| `segmentsArrivalAirportCode` | Redundant with `destinationAirport` for non-stop; complex for multi-leg |
| `segmentsDepartureAirportCode` | Redundant with `startingAirport` |
| `segmentsEquipmentDescription` | Aircraft type rarely drives price meaningfully |
| `segmentsDurationInSeconds` | Captured better by your engineered `trip_duration_minutes` |
| `segmentsDistance` | Redundant with `totalTravelDistance` |
| `segmentsCabinCode` | Mostly "coach" — low variance, not useful |

---

## Final Feature List Summary

Your model should end up with roughly these **10–12 features:**

1. `days_until_flight` ⭐ most important
2. `startingAirport` (encoded)
3. `destinationAirport` (encoded)
4. `isNonStop`
5. `isBasicEconomy`
6. `isRefundable`
7. `seatsRemaining`
8. `departure_month`
9. `departure_day_of_week`
10. `num_segments`
11. `totalTravelDistance` (after imputation)
12. `segmentsAirlineName` (encoded, top airlines only)

**Target:** `totalFare`

---

This keeps your model lean, interpretable, and directly tied to your research question. The feature importance output from Random Forest will then tell you which of these actually move the needle on price.
