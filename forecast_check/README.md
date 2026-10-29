# Forecast check, 2019–2026

Do online signals help forecast national wholesale produce prices? We extended the data to January 2019 –
September 2026 and trained forecasting models with and without each online source.

**Result: no.** The best forecasts 1 and 4 weeks ahead use price history alone. Adding Reddit, Google Trends,
news coverage (GDELT), FDA recalls, Wikipedia pageviews or shipments gives no significant gain with any model
(every bootstrap p ≥ 0.07), and adding all online sources together does not help either. For warning of big price
moves, no source raises the AUC by more than 0.014, and every 95% interval includes zero. This matches the main
analysis, where talk and prices correlate weakly at every lag.

## Design

- **Data** (`data/panel_long.csv`): one row per product and week, 4 products × 405 weeks. 2022–2024 is the main
  dataset of this repository; 2019–2021 and 2025–September 2026 were collected with the same pipeline and the
  same classifier (no new labels), see `build_period.sh`. GDELT, FDA and Wikipedia come from `collect_extra.py`.
- **Targets:** change in the national log price index 1 and 4 weeks ahead; and whether the 4-week change is a
  big move (at least the product's 85th percentile of absolute change in the training weeks; 161 of 1,184 test
  product-weeks).
- **Test:** walk-forward from January 2021 to September 2026 in 4-week blocks. Each block uses models trained only
  on weeks whose outcome was known before the block starts. All online features are z-scores against the
  previous 8 weeks or week-to-week changes, so no feature uses later data.
- **Features:** every set contains price history (last 4 weekly changes, distance from the 8-, 26- and 52-week
  means, change over a year, season, product). The other sets add one source: shipments; Reddit (the weekly
  signals of the main analysis and the comment-weighted supply count of section 13); Google Trends; news volume,
  share and tone; FDA recalls; Wikipedia; all online sources; everything.
- **Models:** ridge regression, gradient boosting and random forest for changes; logistic regression and gradient
  boosting for big moves. Significance by moving-block bootstrap over test weeks (8-week blocks).

## Reproduce

```bash
python forecast_check/predict_long.py        # ~8 min; writes outputs/tables/forecast_long_*.csv
```

`bash run_all.sh --analysis` runs it too, and `python scripts/check_reproduction.py` compares the three tables with
the committed ones.

To rebuild the panel from scratch (about a day, mostly downloads; needs `MARS_API_KEY` in `.env`):

```bash
bash forecast_check/build_period.sh 2019-01-01 2022-01-01 forecast_check/periods/2019-2021
bash forecast_check/build_period.sh 2025-01-01 2026-10-01 forecast_check/periods/2025-2026
python forecast_check/collect_extra.py
python forecast_check/merge_long.py
```

The period copies are not committed (they are as large as the main dataset). Arctic Shift and Google Trends are
live services, so a new collection can differ slightly from the committed panel.

## Tables

| File | Content |
|---|---|
| `outputs/tables/forecast_long_regression.csv` | R² against "no change", direction accuracy and p-value against price history alone with the same model, per horizon, feature set and model |
| `outputs/tables/forecast_long_bigmove.csv` | AUC and average precision for big moves, overall and per year |
| `outputs/tables/forecast_long_auc_gain.csv` | AUC gain over price history alone, 95% bootstrap interval |
