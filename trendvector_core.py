"""TrendVector V3 modelling core (no Streamlit imports).

Academic proof of concept on monthly Google search-interest data.

One row = fashion concept x planner market x origin month t.
Features use months <= t. The label uses month t+1.
A model forecasting from origin t trains only on rows with date < t, whose
labels are all known by month t.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent / "data"

MARKETS = ["US", "GB", "FR", "DE"]
MARKET_NAMES = {
    "US": "United States",
    "GB": "United Kingdom",
    "FR": "France",
    "DE": "Germany",
}
CONCEPTS = ["wide leg jeans", "ballet flats", "cargo pants", "loafers", "puffer jacket"]
CONCEPT_NAMES = {
    "wide leg jeans": "Wide-leg jeans",
    "ballet flats": "Ballet flats",
    "cargo pants": "Cargo pants",
    "loafers": "Loafers",
    "puffer jacket": "Puffer jacket",
}

# Target definition. Kept from the earlier prototype to avoid metric-shopping.
EVENT_THRESHOLD = 0.10

# Validation design. Set after the audit had studied this same data.
VALIDATION_MONTHS = 24
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 0
PLACEBO_SHUFFLES = 20
PLACEBO_SEED = 1
SOURCE_SHUFFLE_SEED = 100
SEASONAL_SMOOTHING = 2.0
NEUTRAL_PRIOR = 0.5
LOGISTIC_C = 0.5

LOCAL_FEATURES = ["local_rel3", "local_pct1", "local_pct2"]
GLOBAL_FEATURES = [
    "global_mean_pct1",
    "global_mean_pct2",
    "global_mean_rel3",
    "global_dispersion_pct1",
]
SEASON_FEATURES = ["month_sin", "month_cos", "seas_logit"]
ID_FEATURES = [f"target_{m}" for m in MARKETS] + [f"concept_{c}" for c in CONCEPTS]
CROSS_FEATURES = [
    f"{m}_foreign_{kind}" for m in MARKETS for kind in ("pct1", "pct2", "rel3")
]

# Ladder: local history -> global movement of the concept -> market-specific signals.
# LOCAL_ONLY_FEATURES is a diagnostic for the global step. It never produces a forecast.
LOCAL_ONLY_FEATURES = LOCAL_FEATURES + SEASON_FEATURES + ID_FEATURES
MODEL_1_FEATURES = LOCAL_FEATURES + GLOBAL_FEATURES + SEASON_FEATURES + ID_FEATURES
MODEL_2_FEATURES = MODEL_1_FEATURES + CROSS_FEATURES

# Global and market-specific signals stay in one explanation group because the
# global means are exact averages of the market-specific columns, so the split
# between them inside Model 2 has no unique meaning.
EXPLANATION_GROUPS = {
    "OWN MARKET": LOCAL_FEATURES,
    "OTHER MARKETS": GLOBAL_FEATURES + CROSS_FEATURES,
    "SEASON": SEASON_FEATURES,
}

# Forecast bands, as declared in the V3 specification. Their hit rates are
# measured from the walk-forward window, never assumed.
BANDS = [
    ("LOW", 0.0, 0.40),
    ("BORDERLINE", 0.40, 0.60),
    ("HIGH", 0.60, 0.85),
    ("VERY HIGH", 0.85, 1.01),
]
BAND_BOUNDARIES = [0.40, 0.60, 0.85]
BORDERLINE_MARGIN = 0.05
TIE_MARGIN = 0.05
DISPLAY_CAP = 0.90
DISPLAY_FLOOR = 0.10


# ------------------------------------------------------------------ data


def load_data(data_dir=DATA_DIR):
    """Read one CSV per market and confirm that they share identical dates."""
    data = {}
    for market in MARKETS:
        frame = pd.read_csv(Path(data_dir) / f"fashion_concepts_{market}.csv")
        frame["Time"] = pd.to_datetime(frame["Time"])
        data[market] = frame.sort_values("Time").reset_index(drop=True)
    dates = [pd.Timestamp(d) for d in data[MARKETS[0]]["Time"]]
    for market in MARKETS:
        if [pd.Timestamp(d) for d in data[market]["Time"]] != dates:
            raise ValueError("Market files do not share identical dates.")
    return data, dates


def _pct(current, previous):
    return 0.0 if previous == 0 or pd.isna(previous) else float(current / previous - 1.0)


def _rel3(series):
    return float(series[0] / np.mean(series[:3]) - 1.0)


def add_seasonal_rate(pooled):
    """Past-only, smoothed concept x calendar-month event rate.

    For each row this uses only rows from strictly earlier dates, whose labels
    were already known. The rate is shrunk towards the concept's own past rate,
    or towards a neutral 0.5 when the concept has no past rows yet.
    """
    pooled = pooled.copy()
    pooled["origin_month"] = pooled["date"].dt.month
    rates = pd.Series(index=pooled.index, dtype=float)
    for (date, concept), group in pooled.groupby(["date", "concept"]):
        past = pooled[(pooled["date"] < date) & (pooled["concept"] == concept)]
        prior = past["y"].mean() if len(past) else NEUTRAL_PRIOR
        same_month = past[past["origin_month"] == date.month]["y"]
        rates[group.index] = (same_month.sum() + SEASONAL_SMOOTHING * prior) / (
            len(same_month) + SEASONAL_SMOOTHING
        )
    pooled["seas_rate"] = rates.clip(0.02, 0.98)
    pooled["seas_logit"] = np.log(pooled["seas_rate"] / (1.0 - pooled["seas_rate"]))
    return pooled


def build_pooled_dataset(data, dates, threshold=EVENT_THRESHOLD):
    """Build scale-invariant features. Raw levels are never compared across markets."""
    rows = []
    for i in range(4, len(dates) - 1):
        date = dates[i]
        for concept in CONCEPTS:
            values = {
                m: [float(data[m].loc[i - k, concept]) for k in range(3)] for m in MARKETS
            }
            for target in MARKETS:
                others = [m for m in MARKETS if m != target]
                local = values[target]
                next_value = float(data[target].loc[i + 1, concept])
                other_pct1 = [_pct(values[m][0], values[m][1]) for m in others]
                other_pct2 = [_pct(values[m][1], values[m][2]) for m in others]
                row = {
                    "date": date,
                    "concept": concept,
                    "target": target,
                    "y": int(next_value >= (1.0 + threshold) * float(np.mean(local))),
                    "persisted_without_growth": int(next_value <= local[0]),
                    "local_rel3": _rel3(local),
                    "local_pct1": _pct(local[0], local[1]),
                    "local_pct2": _pct(local[1], local[2]),
                    "global_mean_pct1": float(np.mean(other_pct1)),
                    "global_mean_pct2": float(np.mean(other_pct2)),
                    "global_mean_rel3": float(np.mean([_rel3(values[m]) for m in others])),
                    "global_dispersion_pct1": float(np.std(other_pct1)),
                    "month_sin": float(np.sin(2 * np.pi * date.month / 12)),
                    "month_cos": float(np.cos(2 * np.pi * date.month / 12)),
                }
                for market in MARKETS:
                    row[f"target_{market}"] = int(target == market)
                for name in CONCEPTS:
                    row[f"concept_{name}"] = int(concept == name)
                # The target market is zeroed: it is never its own foreign signal.
                for source in MARKETS:
                    own = source == target
                    src = values[source]
                    row[f"{source}_foreign_pct1"] = 0.0 if own else _pct(src[0], src[1])
                    row[f"{source}_foreign_pct2"] = 0.0 if own else _pct(src[1], src[2])
                    row[f"{source}_foreign_rel3"] = 0.0 if own else _rel3(src)
                rows.append(row)
    return add_seasonal_rate(pd.DataFrame(rows))


# ---------------------------------------------------------------- models


def new_logistic():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=LOGISTIC_C, solver="liblinear", max_iter=1500),
    )


def fit_logistic(train, features):
    model = new_logistic()
    model.fit(train[features], train["y"])
    return model


def predict_logistic(model, rows, features):
    return model.predict_proba(rows[features])[:, 1]


def fit_final_model(pooled, cutoff):
    """Fit Model 2 on rows dated before the cutoff, for every market alike."""
    return fit_logistic(pooled[pooled["date"] < cutoff], MODEL_2_FEATURES)


def generate_forecasts(pooled, cutoff):
    """Forecast every concept x market from one model. The labels stay hidden."""
    model = fit_final_model(pooled, cutoff)
    rows = pooled[pooled["date"] == cutoff]
    forecasts = rows[["concept", "target"]].copy()
    forecasts["probability"] = predict_logistic(model, rows, MODEL_2_FEATURES)
    return model, rows.drop(columns=["y", "persisted_without_growth"]), forecasts


# ------------------------------------------------------------ validation


def _placebo_maps():
    """Fixed concept reassignments in which no concept keeps its own signals."""
    rng = random.Random(PLACEBO_SEED)
    maps = []
    while len(maps) < PLACEBO_SHUFFLES:
        shuffled = CONCEPTS[:]
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(CONCEPTS, shuffled)):
            maps.append(dict(zip(CONCEPTS, shuffled)))
    return maps


def _with_wrong_concept_signals(pooled, mapping):
    """Give each row the market-specific signals of a different concept.

    The Local + Global inputs are unchanged, so shared seasonality, country-wide
    shocks and the concept's global movement survive. Only the same-concept link
    in the market-specific signals is broken.
    """
    source = pooled.set_index(["date", "target", "concept"])[CROSS_FEATURES]
    keys = list(
        zip(pooled["date"], pooled["target"], [mapping[c] for c in pooled["concept"]])
    )
    scrambled = pooled.copy()
    scrambled[CROSS_FEATURES] = source.loc[keys].values
    return scrambled


def _with_shuffled_source_markets(pooled, seed):
    """Randomise which other market produced which signal, row by row.

    The concept and the three source values stay the same, so global averages
    are preserved. Only the identity of each source market is lost.
    """
    rng = np.random.default_rng(seed)
    kinds = ("pct1", "pct2", "rel3")
    shuffled = pooled.copy()
    for target in MARKETS:
        sources = [m for m in MARKETS if m != target]
        index = shuffled.index[shuffled["target"] == target]
        columns = [[f"{s}_foreign_{k}" for k in kinds] for s in sources]
        block = np.stack([shuffled.loc[index, c].to_numpy() for c in columns], axis=1)
        order = np.argsort(rng.random((len(index), len(sources))), axis=1)
        block = np.take_along_axis(block, order[:, :, None], axis=1)
        for position, c in enumerate(columns):
            shuffled.loc[index, c] = block[:, position, :]
    return shuffled


def walk_forward_predictions(pooled, months):
    """Refit chronologically: each month is predicted only from earlier dates.

    Returns one row per forecast with Models 0 to 2, the local-only diagnostic,
    the placebo and the source-market shuffle. Predictions for a month do not
    depend on the later cutoff, so one table serves every 24-month window.
    """
    frames = {"model_local": (pooled, LOCAL_ONLY_FEATURES)}
    frames["model_1"] = (pooled, MODEL_1_FEATURES)
    frames["model_2"] = (pooled, MODEL_2_FEATURES)
    for k, mapping in enumerate(_placebo_maps()):
        frames[f"placebo_{k}"] = (
            _with_wrong_concept_signals(pooled, mapping),
            MODEL_2_FEATURES,
        )
    for k in range(PLACEBO_SHUFFLES):
        frames[f"source_shuffle_{k}"] = (
            _with_shuffled_source_markets(pooled, SOURCE_SHUFFLE_SEED + k),
            MODEL_2_FEATURES,
        )

    outputs = []
    for month in months:
        test = pooled[pooled["date"] == month]
        if test.empty:
            continue
        out = test[["date", "concept", "target", "y"]].copy()
        out["model_0"] = test["seas_rate"]
        for name, (frame, features) in frames.items():
            model = fit_logistic(frame[frame["date"] < month], features)
            out[name] = predict_logistic(model, frame[frame["date"] == month], features)
        outputs.append(out)
    return pd.concat(outputs, ignore_index=True)


def all_walk_forward_predictions(pooled, cutoffs):
    """Predict every month needed by any selectable cutoff's 24-month window."""
    first = min(cutoffs) - pd.DateOffset(months=VALIDATION_MONTHS)
    last = max(cutoffs) - pd.DateOffset(months=1)
    return walk_forward_predictions(pooled, pd.date_range(first, last, freq="MS"))


def validation_window(predictions, cutoff):
    start = cutoff - pd.DateOffset(months=VALIDATION_MONTHS)
    mask = (predictions["date"] >= start) & (predictions["date"] < cutoff)
    return predictions[mask]


def brier(y, probability):
    return float(np.mean((np.asarray(y) - np.asarray(probability)) ** 2))


def compare_models(window, reference, candidate):
    """Relative Brier gain of candidate over reference, with month-block bootstrap.

    Whole months are resampled because forecasts from one month are correlated.
    """
    errors = window.assign(
        ref=(window["y"] - window[reference]) ** 2,
        cand=(window["y"] - window[candidate]) ** 2,
    ).groupby("date")[["ref", "cand"]].mean()
    ref, cand = errors["ref"].to_numpy(), errors["cand"].to_numpy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(ref), size=(BOOTSTRAP_DRAWS, len(ref)))
    ref_means, cand_means = ref[draws].mean(axis=1), cand[draws].mean(axis=1)
    gains = (ref_means - cand_means) / ref_means
    return {
        "gain": float((ref.mean() - cand.mean()) / ref.mean()),
        "ci_low": float(np.percentile(gains, 2.5)),
        "ci_high": float(np.percentile(gains, 97.5)),
        "months_better": int((cand < ref).sum()),
        "months": len(ref),
    }


def compute_evidence(predictions, cutoff):
    """Evidence ladder and decision rule for one 24-month window."""
    window = validation_window(predictions, cutoff)
    y = window["y"]
    scores = {name: brier(y, window[name]) for name in ("model_0", "model_1", "model_2")}
    placebo_scores = [
        brier(y, window[f"placebo_{k}"]) for k in range(PLACEBO_SHUFFLES)
    ]
    placebo_mean = float(np.mean(placebo_scores))
    placebo_gain = (scores["model_1"] - placebo_mean) / scores["model_1"]
    placebo_best = (scores["model_1"] - min(placebo_scores)) / scores["model_1"]
    shuffle_scores = [
        brier(y, window[f"source_shuffle_{k}"]) for k in range(PLACEBO_SHUFFLES)
    ]
    cross = compare_models(window, "model_1", "model_2")
    versus_seasonal = compare_models(window, "model_0", "model_2")
    ci_above_zero = cross["ci_low"] > 0
    placebo_no_gain = placebo_gain <= 0
    by_market = {}
    for market in MARKETS:
        rows = window[window["target"] == market]
        by_market[market] = {
            name: brier(rows["y"], rows[name]) for name in ("model_0", "model_1", "model_2")
        }
        by_market[market]["cross"] = compare_models(rows, "model_1", "model_2")
    return {
        "first_month": window["date"].min(),
        "last_month": window["date"].max(),
        "forecasts": len(window),
        "scores": scores,
        "placebo_brier": placebo_mean,
        "placebo_gain": float(placebo_gain),
        "placebo_best_gain": float(placebo_best),
        "local_only_brier": brier(y, window["model_local"]),
        "global_step": compare_models(window, "model_local", "model_1"),
        "shuffle_brier": float(np.mean(shuffle_scores)),
        "shuffle_gain": float(
            (scores["model_1"] - np.mean(shuffle_scores)) / scores["model_1"]
        ),
        "shuffles_beating_true": int(sum(s < scores["model_2"] for s in shuffle_scores)),
        "cross": cross,
        "versus_seasonal": versus_seasonal,
        "ci_above_zero": ci_above_zero,
        "placebo_no_gain": placebo_no_gain,
        "supportive": bool(ci_above_zero and placebo_no_gain),
        "by_market": by_market,
        "band_history": band_history(window),
    }


def evidence_takeaway(evidence):
    """State the decision rule's outcome. The forecast model never switches on it."""
    if evidence["supportive"]:
        return (
            "Market-specific signals for the same concept added predictive information "
            "beyond Local + Global in this historical test; wrong-concept signals did not."
        )
    failed = []
    if not evidence["ci_above_zero"]:
        failed.append("the 95% interval versus Local + Global is not above zero")
    if not evidence["placebo_no_gain"]:
        failed.append("wrong-concept placebo signals also produced a gain")
    return (
        "In this window the Cross-Market evidence is not supportive: "
        + " and ".join(failed)
        + "."
    )


def seasonal_statement(evidence):
    """Compare the full model with the seasonal baseline without flattering it."""
    versus = evidence["versus_seasonal"]
    seasonal, full = evidence["scores"]["model_0"], evidence["scores"]["model_2"]
    scores = f"Brier {full:.3f} for the full model, {seasonal:.3f} for the seasonal calendar"
    if versus["ci_low"] > 0:
        return f"The full model beat the seasonal baseline in this window ({scores})."
    if versus["ci_high"] < 0:
        return f"The seasonal baseline beat the full model in this window ({scores})."
    return f"The full model is only similar to the strong seasonal baseline at this data size ({scores})."


# ----------------------------------------------------------------- bands


def band_name(probability):
    for name, low, high in BANDS:
        if low <= probability < high:
            return name
    return BANDS[-1][0]


def nearest_boundary(probability):
    """Return the band boundary within the borderline margin, if any."""
    for boundary in BAND_BOUNDARIES:
        if abs(probability - boundary) <= BORDERLINE_MARGIN + 1e-12:
            return boundary
    return None


def format_estimate(probability):
    """Round to 5 points and cap the tails, where calibration is weakest."""
    if probability >= DISPLAY_CAP:
        return f"~{DISPLAY_CAP * 100:.0f}%+"
    if probability < DISPLAY_FLOOR:
        return f"<{DISPLAY_FLOOR * 100:.0f}%"
    return f"~{5 * round(probability * 100 / 5):.0f}%"


def format_forecast(probability):
    return f"{band_name(probability)} · {format_estimate(probability)}"


def band_history(window):
    """How Model 2's bands behaved in the validation window."""
    labels = window["model_2"].map(band_name)
    rows = []
    for name, low, high in BANDS:
        rows_in_band = window[labels == name]
        rows.append(
            {
                "band": name,
                "low": low,
                "high": high,
                "forecasts": len(rows_in_band),
                "mean_forecast": float(rows_in_band["model_2"].mean())
                if len(rows_in_band)
                else None,
                "hit_rate": float(rows_in_band["y"].mean()) if len(rows_in_band) else None,
            }
        )
    return rows


def rank_forecasts(forecasts, market):
    """Rank one market's concepts and group those within the tie margin.

    Each group is anchored on its highest forecast, so a chain of small steps
    cannot stretch one tie across a wide range.
    """
    ranked = forecasts[forecasts["target"] == market].sort_values(
        ["probability", "concept"], ascending=[False, True]
    ).reset_index(drop=True)
    groups, group, anchor = [], 0, None
    for probability in ranked["probability"]:
        if anchor is None or anchor - probability > TIE_MARGIN:
            group += 1
            anchor = probability
        groups.append(group)
    ranked["tie_group"] = groups
    ranked["tied"] = ranked["tie_group"].map(ranked["tie_group"].value_counts()) > 1
    return ranked


# ---------------------------------------------------------- explanations


def explain_forecast(model, row):
    """Exact additive split of Model 2's log-odds for one standardised row.

    log-odds = starting point + own market + other markets + season

    The starting point is the intercept plus concept and market identity, which
    is what the model expects for this pair when every other signal sits at its
    training average. These are model contributions, not causal effects.
    """
    scaler = model.named_steps["standardscaler"]
    logistic = model.named_steps["logisticregression"]
    z = scaler.transform(row[MODEL_2_FEATURES])[0]
    contributions = pd.Series(logistic.coef_[0] * z, index=MODEL_2_FEATURES)
    groups = {
        group: float(contributions[features].sum())
        for group, features in EXPLANATION_GROUPS.items()
    }
    top_signals = {
        group: contributions[features].abs().idxmax()
        for group, features in EXPLANATION_GROUPS.items()
    }
    start = float(logistic.intercept_[0] + contributions[ID_FEATURES].sum())
    return {
        "start_log_odds": start,
        "groups": groups,
        "top_signals": top_signals,
        "total_log_odds": start + sum(groups.values()),
    }


def describe_signal(feature):
    labels = {
        "local_rel3": "this market vs its own 3-month average",
        "local_pct1": "this market's latest monthly change",
        "local_pct2": "this market's previous monthly change",
        "global_mean_pct1": "other markets' average latest change",
        "global_mean_pct2": "other markets' average previous change",
        "global_mean_rel3": "other markets vs their own 3-month averages",
        "global_dispersion_pct1": "disagreement across other markets",
        "month_sin": "calendar position",
        "month_cos": "calendar position",
        "seas_logit": "past event rate for this concept and calendar month",
    }
    if feature in labels:
        return labels[feature]
    market, _, kind = feature.split("_")
    suffix = {
        "pct1": "latest monthly change",
        "pct2": "previous monthly change",
        "rel3": "level vs its own 3-month average",
    }[kind]
    return f"{MARKET_NAMES[market]} {suffix}"


# ------------------------------------------------------ decision support


def recommend_action(probability, brand_fit, matching_product, can_act, stock):
    """Check feasibility first, then let the forecast band set the action."""
    if brand_fit == "Weak":
        return {
            "action": "Review brand fit, no action",
            "reason": "The planner judged brand fit as weak, so the forecast does not trigger an action.",
            "steps": [],
            "feasible": False,
        }
    if matching_product == "No":
        return {
            "action": "Flag for future assortment review",
            "reason": "No matching product exists. This prototype never recommends automatic production.",
            "steps": [],
            "feasible": False,
        }
    if can_act == "No":
        return {
            "action": "Monitor",
            "reason": "The business cannot act within the one-month forecast horizon.",
            "steps": [],
            "feasible": False,
        }

    band = band_name(probability)
    boundary = nearest_boundary(probability)
    result = {"feasible": True, "steps": []}

    if boundary == 0.85:
        # Near the top boundary the safe reading is the lower tier, not a stock move.
        result["action"] = "Small reversible test, planner decides on any escalation"
        result["reason"] = "The forecast is within 5 points of the 85% boundary, so it is treated as borderline and does not justify a stock review on its own."
        result["steps"] = _test_steps(stock)
    elif boundary is not None or band == "BORDERLINE":
        result["action"] = "Planner judgement, gather more evidence"
        result["reason"] = (
            f"The forecast is within 5 points of the {boundary * 100:.0f}% boundary, so it is treated as borderline."
            if boundary is not None
            else "The forecast sits in the borderline band."
        )
    elif band == "LOW":
        result["action"] = "Monitor"
        result["reason"] = "The forecast is in the low band."
    elif band == "HIGH":
        result["action"] = "Small reversible test"
        result["reason"] = "The forecast is in the high band and the feasibility checks pass."
        result["steps"] = _test_steps(stock)
    elif stock == "Adequate":
        result["action"] = "Stock / replenishment review"
        result["reason"] = "The forecast is in the very high band, with a matching product, time to act and adequate stock."
    else:
        result["action"] = "Check replenishment feasibility first"
        result["reason"] = "The forecast is in the very high band, but low stock makes extra exposure risky."
    return result


def _test_steps(stock):
    if stock == "Low":
        return ["Stock positioning review, before any added exposure"]
    return ["Merchandising visibility", "Limited paid-media test", "Stock positioning review"]


# -------------------------------------------------------- outcome and data


def event_threshold(data, dates, cutoff, concept, market):
    """Index level next month's value must reach, known at the cutoff."""
    index = dates.index(cutoff)
    average = float(np.mean([data[market].loc[index - k, concept] for k in range(3)]))
    return (1.0 + EVENT_THRESHOLD) * average


def historical_outcome(data, dates, cutoff, concept, market):
    """Compute the outcome for the reveal. It never feeds into any model."""
    index = dates.index(cutoff)
    if index + 1 >= len(dates):
        return None
    threshold = event_threshold(data, dates, cutoff, concept, market)
    value = float(data[market].loc[index + 1, concept])
    return {
        "next_date": dates[index + 1],
        "next_value": value,
        "threshold": threshold,
        "event": value >= threshold,
    }


def dataset_summary(data, dates, pooled):
    positives = pooled[pooled["y"] == 1]
    return {
        "concepts": len(CONCEPTS),
        "markets": len(MARKETS),
        "observations": len(dates),
        "forecast_months": int(pooled["date"].nunique()),
        "rows": len(pooled),
        "event_rate": float(pooled["y"].mean()),
        "persisted_share": float(positives["persisted_without_growth"].mean()),
    }
