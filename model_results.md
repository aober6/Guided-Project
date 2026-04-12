# XGBoost Flight Price Prediction — Results

## Metrics

| Metric | Value |
|--------|-------|
| MAE    | $80.72 |
| RMSE   | $142.49 |
| R^2 (log scale) | 0.7519 |
| R^2 ($ scale)   | 0.6603 |
| MAPE   | 22.03% |
| Train time | 10.9s |

## Hyperparameters

```
{'objective': 'reg:squarederror', 'base_score': None, 'booster': None, 'callbacks': None, 'colsample_bylevel': None, 'colsample_bynode': None, 'colsample_bytree': 0.8, 'device': None, 'early_stopping_rounds': None, 'enable_categorical': False, 'eval_metric': None, 'feature_types': None, 'feature_weights': None, 'gamma': None, 'grow_policy': None, 'importance_type': None, 'interaction_constraints': None, 'learning_rate': 0.05, 'max_bin': None, 'max_cat_threshold': None, 'max_cat_to_onehot': None, 'max_delta_step': None, 'max_depth': 7, 'max_leaves': None, 'min_child_weight': None, 'missing': nan, 'monotone_constraints': None, 'multi_strategy': None, 'n_estimators': 500, 'n_jobs': None, 'num_parallel_tree': None, 'random_state': 42, 'reg_alpha': None, 'reg_lambda': None, 'sampling_method': None, 'scale_pos_weight': None, 'subsample': 0.8, 'tree_method': 'hist', 'validate_parameters': None, 'verbosity': 1}
```

## Top 20 Features

| Feature               |   Importance |
|:----------------------|-------------:|
| isBasicEconomy        |   0.437726   |
| num_segments          |   0.124144   |
| totalTravelDistance   |   0.073358   |
| isNonStop             |   0.0684839  |
| primary_airline       |   0.0549342  |
| departure_month       |   0.0465568  |
| departure_day_of_week |   0.0451073  |
| seatsRemaining        |   0.0410977  |
| trip_duration_minutes |   0.0314077  |
| destinationAirport    |   0.026186   |
| days_until_flight     |   0.0243028  |
| startingAirport       |   0.022696   |
| search_day_of_week    |   0.00399944 |
| isRefundable          |   0          |

## Plots

- `results/feature_importance.png`
- `results/pred_vs_actual.png`
