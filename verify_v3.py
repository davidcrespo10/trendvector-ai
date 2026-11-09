"""Reproduces the V3 metrics and runs the ten pre-submission checks.

Run with: py verify_v3.py
This script is not imported by the app.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import trendvector_core as tv

ROOT = Path(__file__).parent
failures = []


def check(name, passed, detail=""):
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}{' | ' + detail if detail else ''}")
    if not passed:
        failures.append(name)


def corrupt_after(data, dates, last_kept):
    """Replace every value after last_kept with random numbers."""
    rng = np.random.default_rng(99)
    corrupted = {}
    for market, frame in data.items():
        frame = frame.copy()
        mask = frame["Time"] > last_kept
        for concept in tv.CONCEPTS:
            frame[concept] = frame[concept].astype(float)
            frame.loc[mask, concept] = rng.uniform(1, 100, mask.sum())
        corrupted[market] = frame
    return corrupted


data, dates = tv.load_data()
pooled = tv.build_pooled_dataset(data, dates)
cutoffs = [pd.Timestamp(d) for d in sorted(pooled["date"].unique())[-8:]]
latest = cutoffs[-1]
predictions = tv.all_walk_forward_predictions(pooled, cutoffs)
evidence_by_cutoff = {c: tv.compute_evidence(predictions, c) for c in cutoffs}
evidence = evidence_by_cutoff[latest]


def build_app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()


def click_generate(at):
    next(b for b in at.button if b.label.startswith("Train model")).click().run()
    return at


def all_text(at):
    parts = []
    for kind in ("markdown", "caption", "info", "warning", "success", "error", "subheader"):
        parts += [str(e.value) for e in getattr(at, kind)]
    parts += [m.value for m in at.metric]
    return "\n".join(parts)


print("\n1. Run the app headlessly")
at = build_app()
check("app starts without exception", not at.exception)
check("nothing is forecast before the button", "tabs" not in dir(at) or len(at.tabs) == 0)
click_generate(at)
check("app generates without exception", not at.exception)
check("five tabs in the specified order",
      [t.label for t in at.tabs] == ["Opportunity Radar", "Market Detail", "Decision Support", "Evidence", "Prototype Scope"])

print("\n2. Reproduce the Evidence metrics")
s = evidence["scores"]
c = evidence["cross"]
v = evidence["versus_seasonal"]
print(f"  Window {evidence['first_month']:%b %Y} to {evidence['last_month']:%b %Y}, {evidence['forecasts']} forecasts")
print(f"  Seasonal {s['model_0']:.4f} | Local + Global {s['model_1']:.4f} | Cross-Market {s['model_2']:.4f} | Placebo {evidence['placebo_brier']:.4f}")
print(f"  Cross-Market vs Local + Global {c['gain']*100:.1f}% CI [{c['ci_low']*100:.1f}, {c['ci_high']*100:.1f}] months {c['months_better']}/{c['months']}")
print(f"  Placebo mean {evidence['placebo_gain']*100:.1f}%, best {evidence['placebo_best_gain']*100:.1f}%")
g = evidence["global_step"]
print(f"  Local only {evidence['local_only_brier']:.4f}; global step {g['gain']*100:.1f}% CI [{g['ci_low']*100:.1f}, {g['ci_high']*100:.1f}] months {g['months_better']}/{g['months']}")
print(f"  Source-market shuffle {evidence['shuffle_brier']:.4f}, {evidence['shuffle_gain']*100:.1f}% vs Local + Global; beat true model {evidence['shuffles_beating_true']}/20")
for market, row in evidence["by_market"].items():
    x = row["cross"]
    print(f"  {market}: seasonal {row['model_0']:.3f} L+G {row['model_1']:.3f} cross {row['model_2']:.3f} gain {x['gain']*100:+.1f}% [{x['ci_low']*100:+.1f}, {x['ci_high']*100:+.1f}] {x['months_better']}/{x['months']}")
print(f"  Full vs seasonal {v['gain']*100:.1f}% CI [{v['ci_low']*100:.1f}, {v['ci_high']*100:.1f}] months {v['months_better']}/{v['months']}")
check("24 months and 480 forecasts", c["months"] == 24 and evidence["forecasts"] == 480)
manual = float(((pooled[(pooled.date >= evidence["first_month"]) & (pooled.date < latest)].y
                 - pooled[(pooled.date >= evidence["first_month"]) & (pooled.date < latest)].seas_rate) ** 2).mean())
check("seasonal Brier recomputed independently", abs(manual - s["model_0"]) < 1e-12)
month = pd.Timestamp("2025-03-01")
train, test = pooled[pooled.date < month], pooled[pooled.date == month]
direct = tv.predict_logistic(tv.fit_logistic(train, tv.MODEL_2_FEATURES), test, tv.MODEL_2_FEATURES)
check("walk-forward table matches a direct refit",
      np.allclose(direct, predictions[predictions.date == month]["model_2"].to_numpy()))
ladder = [e for e in at.get("plotly_chart")]
metric_values = {m.label: m.value for m in at.metric}
check("Evidence tab shows the computed improvement",
      metric_values.get("Cross-Market vs Local + Global") == f"{c['gain']*100:+.1f}%")
check("Evidence tab shows the computed interval",
      metric_values.get("Bootstrap 95% CI") == f"{c['ci_low']*100:+.1f}% to {c['ci_high']*100:+.1f}%")
check("Evidence tab shows the computed months",
      metric_values.get("Months Cross-Market is better") == f"{c['months_better']}/{c['months']}")

print("\n3. Check for data leakage")
corrupted = corrupt_after(data, dates, latest)
pooled_bad = tv.build_pooled_dataset(corrupted, dates)
_, _, f_true = tv.generate_forecasts(pooled, latest)
_, _, f_bad = tv.generate_forecasts(pooled_bad, latest)
check("final forecast ignores all data after the cutoff",
      np.array_equal(f_true["probability"].to_numpy(), f_bad["probability"].to_numpy()))
for cut in (pd.Timestamp("2025-06-01"), pd.Timestamp("2026-01-01")):
    bad = tv.build_pooled_dataset(corrupt_after(data, dates, cut), dates)
    original = tv.walk_forward_predictions(pooled, [cut])
    altered = tv.walk_forward_predictions(bad, [cut])
    columns = ["model_0", "model_local", "model_1", "model_2"]
    columns += [f"placebo_{k}" for k in range(tv.PLACEBO_SHUFFLES)]
    columns += [f"source_shuffle_{k}" for k in range(tv.PLACEBO_SHUFFLES)]
    check(f"{cut:%b %Y} predictions ignore all later data",
          np.array_equal(original[columns].to_numpy(), altered[columns].to_numpy()))
first = pooled[pooled.date == pooled.date.min()]
check("seasonal rate has no full-sample fallback", np.isclose(first.seas_rate, tv.NEUTRAL_PRIOR).all())
is_zero = all(
    (pooled.loc[pooled.target == m, [f"{m}_foreign_{k}" for k in ("pct1", "pct2", "rel3")]] == 0).all().all()
    for m in tv.MARKETS
)
check("the target market is never its own foreign signal", is_zero)
check("Local + Global holds global movement but no market-specific signal",
      set(tv.GLOBAL_FEATURES) <= set(tv.MODEL_1_FEATURES) and not set(tv.CROSS_FEATURES) & set(tv.MODEL_1_FEATURES))
check("Cross-Market is exactly Local + Global plus market-specific signals",
      tv.MODEL_2_FEATURES == tv.MODEL_1_FEATURES + tv.CROSS_FEATURES)
check("the local-only diagnostic holds no signal from other markets",
      not set(tv.LOCAL_ONLY_FEATURES) & set(tv.GLOBAL_FEATURES + tv.CROSS_FEATURES))
check("redundant local_accel removed", "local_accel" not in pooled.columns)
check("forecast rows passed to the app hold no label",
      not {"y", "persisted_without_growth"} & set(tv.generate_forecasts(pooled, latest)[1].columns))

print("\n4. Check all tabs")
for market in tv.MARKETS:
    at.selectbox(key="market_choice").select(market).run()
    check(f"all tabs render for {market}", not at.exception)
    for concept in tv.CONCEPTS:
        at.session_state["selected_concept"] = concept
        at.run()
        if at.exception:
            check(f"{market}/{concept} renders", False)
for label, value in [("Brand fit (Planner judgement)", "Weak"), ("Matching product?", "No"), ("Can act within horizon?", "No"), ("Stock availability", "Low")]:
    widget = next(w for w in at.selectbox if w.label == label)
    if not widget.disabled:
        widget.select(value).run()
    check(f"Decision input '{label}' = {value}", not at.exception)
at = build_app()
click_generate(at)
explore = [b for b in at.button if b.label == "Explore"]
target = explore[-1].key.split("_")[-1]
explore[-1].click().run()
check("Explore updates the drill-down", at.session_state["selected_concept"] == target and not at.exception)
reveal = at.toggle[0]
reveal.set_value(True).run()
check("historical outcome reveal works", not at.exception and any("event" in str(x.value) for x in list(at.success) + list(at.error)))

print("\n5. One architecture for every probability")
model, rows, forecasts = tv.generate_forecasts(pooled, latest)
fresh = tv.fit_logistic(pooled[pooled.date < latest], tv.MODEL_2_FEATURES)
check("all 20 forecasts come from one fitted model",
      len(forecasts) == 20 and np.allclose(fresh.predict_proba(rows[tv.MODEL_2_FEATURES])[:, 1], forecasts["probability"]))
log_odds = model.decision_function(rows[tv.MODEL_2_FEATURES])
gaps = []
for position, (_, row) in enumerate(rows.iterrows()):
    gaps.append(abs(tv.explain_forecast(model, rows.iloc[[position]])["total_log_odds"] - log_odds[position]))
check("explanation adds exactly to the model's log-odds", max(gaps) < 1e-9, f"largest gap {max(gaps):.1e}")
check("explanation groups cover every non-identity feature once",
      sorted(sum(tv.EXPLANATION_GROUPS.values(), []) + tv.ID_FEATURES) == sorted(tv.MODEL_2_FEATURES))
check("no calibration step in the pipeline", list(model.named_steps) == ["standardscaler", "logisticregression"])
source = (ROOT / "app.py").read_text(encoding="utf-8") + (ROOT / "trendvector_core.py").read_text(encoding="utf-8")
check("no boosting or calibration code remains", not re.search(r"GradientBoostingClassifier|calibrated_predict|MIN_MONTHLY_WINS|local_accel\"", source))

print("\n6. No hard-coded results")
for pattern in (r"0\.15[0-9]", r"0\.14[0-9]", r"0\.16[0-9]", r"0\.17[0-9]", r"11\.2", r"16/24", r"17/24", r"10\.3"):
    check(f"no literal {pattern}", not re.search(pattern, source))
fake = predictions.copy()
fake["y"] = 1 - fake["y"]
check("Evidence recomputes when outcomes change", tv.compute_evidence(fake, latest)["scores"] != s)

print("\n7. No privileged date or market")
app_source = (ROOT / "app.py").read_text(encoding="utf-8")
for phrase in ("Recommended presentation", "recommended demo", "2026-", "August 2026", '.index("GB")', '== "GB"', '"wide leg jeans"'):
    check(f"app.py has no '{phrase}'", phrase not in app_source)
at = build_app()
check("the default market is the list's first entry", at.selectbox(key="market_choice").value == tv.MARKETS[0])
check("the default date is the latest available", at.selectbox(key="cutoff_choice").value == latest)
retired = r"traction onset|trend arrival|sustained traction|Probability effect|significant-growth|Market Radar|Model Validation|Own market \+|own-market"
click_generate(at)
check("retired wording is gone from the app", not re.search(retired, all_text(at), re.I))

print("\n8. Verify the placebo")
frame = tv._with_wrong_concept_signals(pooled, tv._placebo_maps()[0])
mapping = tv._placebo_maps()[0]
sample = frame[(frame.concept == "loafers") & (frame.target == "GB") & (frame.date == latest)].iloc[0]
donor = pooled[(pooled.concept == mapping["loafers"]) & (pooled.target == "GB") & (pooled.date == latest)].iloc[0]
check("market-specific signals come from the mapped concept",
      all(sample[f] == donor[f] for f in tv.CROSS_FEATURES))
own = pooled[(pooled.concept == "loafers") & (pooled.target == "GB") & (pooled.date == latest)].iloc[0]
check("Local + Global inputs, season and label are untouched",
      all(sample[f] == own[f] for f in tv.MODEL_1_FEATURES + ["y"]))
gap = (pooled["global_mean_pct1"] - pooled[[f"{m}_foreign_pct1" for m in tv.MARKETS]].sum(axis=1) / 3).abs().max()
check("global means are exact averages of market-specific columns", gap < 1e-12)
shuffled = tv._with_shuffled_source_markets(pooled, tv.SOURCE_SHUFFLE_SEED)
pct1 = [f"{m}_foreign_pct1" for m in tv.MARKETS]
same_values = np.allclose(np.sort(shuffled[pct1].to_numpy(), axis=1), np.sort(pooled[pct1].to_numpy(), axis=1))
moved = (shuffled[pct1].to_numpy() != pooled[pct1].to_numpy()).any()
check("market-identity shuffle keeps each row's values but moves them", same_values and moved)
check("market-identity shuffle leaves the target's own columns at zero",
      all((shuffled.loc[shuffled.target == m, f"{m}_foreign_pct1"] == 0).all() for m in tv.MARKETS))
check("every shuffle moves every concept",
      all(all(k != w for k, w in m.items()) for m in tv._placebo_maps()) and len(tv._placebo_maps()) == tv.PLACEBO_SHUFFLES)
check("placebo shows no average gain", evidence["placebo_no_gain"], f"{evidence['placebo_gain']*100:.1f}%")

print("\n9. Verify the bootstrap")
window = tv.validation_window(predictions, latest)
check("same seed gives the same interval", tv.compare_models(window, "model_1", "model_2") == c)
errors = window.assign(r=(window.y - window.model_1) ** 2, k=(window.y - window.model_2) ** 2).groupby("date")[["r", "k"]].mean()
gains = []
rng = np.random.default_rng(1234)
for _ in range(2000):
    sample_months = errors.sample(len(errors), replace=True, random_state=int(rng.integers(1_000_000_000)))
    gains.append((sample_months.r.mean() - sample_months.k.mean()) / sample_months.r.mean())
low, high = np.percentile(gains, [2.5, 97.5])
check("independent month-block bootstrap agrees within 1.5 points",
      abs(low - c["ci_low"]) < 0.015 and abs(high - c["ci_high"]) < 0.015, f"[{low*100:.1f}, {high*100:.1f}]")
check("the point estimate lies inside the interval", c["ci_low"] < c["gain"] < c["ci_high"])

print("\n10. Outputs change with the cutoff")
previous = None
for cut in cutoffs:
    _, _, f = tv.generate_forecasts(pooled, cut)
    if previous is not None:
        check(f"forecasts differ at {cut:%b %Y}", not np.allclose(f["probability"], previous))
    previous = f["probability"].to_numpy()
scores_seen = {round(evidence_by_cutoff[cut]["scores"]["model_2"], 6) for cut in cutoffs}
check("Evidence differs across all cutoffs", len(scores_seen) == len(cutoffs))
at = build_app()
click_generate(at)
first_metrics = {m.label: m.value for m in at.metric}
at.selectbox(key="cutoff_choice").select(cutoffs[0]).run()
check("changing the date withdraws old forecasts", len(at.tabs) == 0 and not at.exception)
click_generate(at)
second_metrics = {m.label: m.value for m in at.metric}
check("the app's Evidence changes after retraining",
      first_metrics["Cross-Market vs Local + Global"] != second_metrics["Cross-Market vs Local + Global"])
window_text = all_text(at)
check("the app's Evidence window moves with the date", "Jan 2024 to Dec 2025" in window_text)

print("\nSensitivity reported, not used by the app")
print("  Evidence at every selectable date")
for cut, e in evidence_by_cutoff.items():
    x = e["cross"]
    print(f"    {cut:%b %Y}: seasonal {e['scores']['model_0']:.3f} L+G {e['scores']['model_1']:.3f} cross {e['scores']['model_2']:.3f} "
          f"placebo {e['placebo_brier']:.3f} gain {x['gain']*100:+.1f}% [{x['ci_low']*100:+.1f}, {x['ci_high']*100:+.1f}] "
          f"{x['months_better']}/24 best shuffle {e['placebo_best_gain']*100:+.1f}% rule {'met' if e['supportive'] else 'NOT met'}")
print("  Event threshold, Cross-Market vs Local + Global over the same window")
window_months = pd.date_range(evidence["first_month"], evidence["last_month"], freq="MS")
for threshold in (0.05, 0.15, 0.20):
    alt = tv.build_pooled_dataset(data, dates, threshold=threshold)
    rows_out = []
    for m in window_months:
        tr, te = alt[alt.date < m], alt[alt.date == m]
        out = te[["date", "y"]].copy()
        out["a"] = tv.predict_logistic(tv.fit_logistic(tr, tv.MODEL_1_FEATURES), te, tv.MODEL_1_FEATURES)
        out["b"] = tv.predict_logistic(tv.fit_logistic(tr, tv.MODEL_2_FEATURES), te, tv.MODEL_2_FEATURES)
        rows_out.append(out)
    result = tv.compare_models(pd.concat(rows_out), "a", "b")
    print(f"    {threshold*100:.0f}%: gain {result['gain']*100:+.1f}% [{result['ci_low']*100:+.1f}, {result['ci_high']*100:+.1f}]")
print("  Bands at the latest date")
print(pd.DataFrame(evidence["band_history"]).round(3).to_string(index=False))

print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILED: ' + '; '.join(failures)}")
sys.exit(1 if failures else 0)
