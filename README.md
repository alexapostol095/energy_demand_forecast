# Energy Demand Forecasting: SARIMA vs. Prophet vs. LightGBM

A rolling-origin backtest comparing three forecasting approaches — a classical statistical model, a decomposition-based model, and a machine-learning model with engineered lag features — on hourly electricity demand. 

## Data

[PJM Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) (Kaggle) — `PJME_hourly.csv`, ~145k hourly observations of PJM East demand (MW), 2002–2018.


## What's in the notebook

1. **Data quality checks** — resolving duplicate timestamps (DST fall-back), filling missing hours (DST spring-forward), reindexing to a continuous hourly grid.
2. **Exploratory analysis** — yearly, weekly, and daily seasonality; STL-style decomposition; weekday vs. weekend daily profiles.
3. **Rolling-origin backtesting protocol** — 6 folds, 7-day (168h) forecast horizon, no random splits (random splits leak the future into training on time series).
4. **Three models:**
   - **SARIMA** on daily aggregates, disaggregated to hourly using each fold's training-period intraday profile.
   - **Prophet**, fit directly on hourly data with US holidays.
   - **LightGBM**, trained as three independent quantile regressors (5th / 50th / 95th percentile) on horizon-safe lag and calendar features.
5. **Evaluation** — MAE, MAPE, and empirical coverage of the 90% prediction interval, compared against the 90% nominal target.

## Results

| Model | MAPE (mean ± std) | Coverage90 (mean ± std) |
|---|---|---|
| **SARIMA (daily→hourly)** | 8.05% ± 2.24 | **95.0% ± 5.8** |
| Prophet | 9.22% ± 2.04 | 93.4% ± 7.5 |
| LightGBM (quantile) | 8.57% ± 3.30 | **70.4% ± 27.2** |
