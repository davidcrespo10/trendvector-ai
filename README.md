# TrendVector V3: academic prototype

This prototype predicts a simplified Google-search proxy event. It does not directly predict fashion demand or commercial traction.

Commercial TrendVector is the broader system described in the business plan. This app is only a simplified proof of concept on monthly Google search-interest data.

## Run

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Choose **Data available up to** and **Planner market**, then press **Train model & generate forecast**. No date or market is recommended. The date defaults to the latest one with a known outcome, and the market to the first in the list.

To repeat the checks and print the metrics:

```powershell
py verify_v3.py
```

## Target

**Elevated search-interest event:** next month's Google search interest is at least 10% above the average of the current and previous two months.

## Evidence ladder

| Level | Model | Inputs |
|---|---|---|
| Model 0 | Seasonal baseline | Past-only, smoothed concept × calendar-month event rate |
| Model 1 | Local + Global logistic | This market's changes, season, the same concept's average movement in the other markets, concept and market identity |
| Model 2 | Cross-Market logistic | Model 1 plus the same concept's signal from each other market separately |

Both logistic models use L2 regularisation with C=0.5 and have no separate calibration step. Every displayed forecast comes from Model 2, in every market and at every date.

## Validation

- 24-month expanding-window walk-forward across all four markets, refitted each month on earlier dates only.
- Month-block bootstrap 95% interval for the improvement over Model 1.
- Wrong-concept placebo: 20 fixed shuffles that give each row another concept's market-specific signals, leaving Local + Global intact.
- Diagnostics outside the rule: a local-only model to measure the global step, and 20 shuffles that randomise which market produced each signal.

The Cross-Market evidence counts as supportive only if the interval is above zero **and** the placebo shows no average gain. This rule changes what the evidence is called, never which model runs. Source-market effects are pooled across destinations, so no specific lead-market pair is validated, and the results are predictive, not causal.

This is not an untouched test. The rule and model were set after an audit had studied this same data, so only forecasts on new months can confirm the result.

All results are computed at runtime. The app contains no stored metrics.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `trendvector_core.py` | Data, features, models, validation, bands, explanations, decision logic |
| `verify_v3.py` | Reproduces the metrics and runs the pre-submission checks |
| `trendvector_audit.py` | Audit reference only, not used by the app |
| `data/` | The four Google Trends CSV files |
