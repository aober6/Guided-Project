# XGBoost Flight Price Prediction -- Results (Chicago ORD/MDW)

## Tuned Model Metrics

| Metric | Value |
|--------|-------|
| MAE    | $41.52 |
| RMSE   | $70.27 |
| R^2 (log) | 0.8292 |
| R^2 ($)   | 0.8111 |
| MAPE   | 16.84% |

## Best Hyperparameters

```json
{
  "learning_rate": 0.04085269907038714,
  "max_depth": 9,
  "subsample": 0.6176263978043055,
  "colsample_bytree": 0.5551459173312394,
  "min_child_weight": 2,
  "gamma": 1.85248293973278e-08,
  "reg_alpha": 0.0001995539554706924,
  "reg_lambda": 0.5990863228857825
}
```

## Top Features

| Feature               |   Importance |
|:----------------------|-------------:|
| isBasicEconomy        |   0.210215   |
| be_x_days             |   0.190752   |
| isNonStop             |   0.0988212  |
| num_segments          |   0.0912206  |
| route                 |   0.075921   |
| primary_airline       |   0.0542987  |
| seatsRemaining        |   0.0427033  |
| log_distance          |   0.0406415  |
| nonstop_x_distance    |   0.0268897  |
| departure_month       |   0.0244028  |
| dow_sin               |   0.0216489  |
| log_duration          |   0.0213462  |
| destinationAirport    |   0.016576   |
| seats_at_max          |   0.0164156  |
| startingAirport       |   0.0162504  |
| departure_day_of_week |   0.0132391  |
| urgency_scarcity      |   0.0102637  |
| departure_hour        |   0.00878844 |
| days_until_flight     |   0.00680798 |
| log_days              |   0.00627623 |
| dow_cos               |   0.00561895 |
| search_day_of_week    |   0.00090275 |

## Plots

- `results/optuna_history.png`
- `results/feature_importance_tuned.png`
