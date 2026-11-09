"""TrendVector V3: academic proof of concept. Run with: py -m streamlit run app.py"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import trendvector_core as tv

st.set_page_config(page_title="TrendVector V3", page_icon="📈", layout="wide")

PROXY_STATEMENT = (
    "This prototype predicts a simplified Google-search proxy event. "
    "It does not directly predict fashion demand or commercial traction."
)
SCOPE_STATEMENT = (
    "Academic prototype only: a simplified proof of concept on monthly Google "
    "search-interest data. Commercial TrendVector is the broader system in the "
    "business plan, and this app does not test it."
)
EVENT_DEFINITION = (
    "**Elevated search-interest event:** next month's Google search interest is at "
    "least 10% above the average of the current and previous two months."
)
FOOTER = "Academic proxy · Google search interest · monthly · not sales impact"

MARKET_COLORS = {"US": "#2a78d6", "GB": "#eb6834", "FR": "#1baf7a", "DE": "#eda100"}
HIGHLIGHT = "#2a78d6"
NEUTRAL = "#8b8a85"
UP_COLOR = "#2a78d6"
DOWN_COLOR = "#eb6834"
BAND_BADGES = {
    "LOW": ":gray-badge[LOW]",
    "BORDERLINE": ":orange-badge[BORDERLINE]",
    "HIGH": ":blue-badge[HIGH]",
    "VERY HIGH": ":violet-badge[VERY HIGH]",
}
BAND_ACTIONS = {
    "LOW": "Monitor",
    "BORDERLINE": "Planner judgement, gather more evidence",
    "HIGH": "Small reversible test",
    "VERY HIGH": "Stock / replenishment review, if feasible and stock is adequate",
}


# ---------------------------------------------------------------- caching


@st.cache_data(show_spinner=False)
def load_all():
    data, dates = tv.load_data()
    pooled = tv.build_pooled_dataset(data, dates)
    return data, dates, pooled


@st.cache_data(show_spinner=False)
def walk_forward_table(pooled, cutoffs):
    return tv.all_walk_forward_predictions(pooled, cutoffs)


# ---------------------------------------------------------------- helpers


def month_label(date):
    return pd.Timestamp(date).strftime("%B %Y")


def short_month(date):
    return pd.Timestamp(date).strftime("%b %Y")


def footer():
    st.divider()
    st.caption(FOOTER)


def fact(column, label, value):
    """Show a labelled value that wraps, because st.metric truncates long text."""
    with column.container(border=True, height="stretch"):
        st.caption(label)
        st.markdown(f"**{value}**")


def band_row(evidence, band):
    return next(row for row in evidence["band_history"] if row["band"] == band)


def hit_rate_text(evidence, band):
    row = band_row(evidence, band)
    if row["hit_rate"] is None:
        return "No validation forecasts in this band"
    return (
        f"Historical hit rate for this band: {row['hit_rate'] * 100:.0f}% "
        f"({row['forecasts']} forecasts)"
    )


def band_range(low, high):
    if low == 0:
        return f"<{high * 100:.0f}%"
    if high > 1:
        return f"≥{low * 100:.0f}%"
    return f"{low * 100:.0f}–{high * 100:.0f}%"


def band_table(evidence):
    rows = []
    for row in evidence["band_history"]:
        rows.append(
            {
                "Band": row["band"],
                "Forecast range": band_range(row["low"], row["high"]),
                "Forecasts": row["forecasts"],
                "Average forecast": (
                    f"{row['mean_forecast'] * 100:.0f}%"
                    if row["mean_forecast"] is not None
                    else "–"
                ),
                "Event happened": (
                    f"{row['hit_rate'] * 100:.0f}%" if row["hit_rate"] is not None else "–"
                ),
                "Default action": BAND_ACTIONS[row["band"]],
            }
        )
    return pd.DataFrame(rows)


def select_concept(concept):
    st.session_state.selected_concept = concept


# ----------------------------------------------------------------- charts


def history_chart(data, dates, cutoff, concept, target):
    end = dates.index(cutoff) + 1
    fig = go.Figure()
    for market in tv.MARKETS:
        selected = market == target
        fig.add_trace(
            go.Scatter(
                x=dates[:end],
                y=data[market][concept].iloc[:end],
                mode="lines",
                name=tv.MARKET_NAMES[market] + (" (selected)" if selected else ""),
                line=dict(width=3 if selected else 1.5, color=MARKET_COLORS[market]),
                opacity=1 if selected else 0.75,
            )
        )
    threshold = tv.event_threshold(data, dates, cutoff, concept, target)
    fig.add_trace(
        go.Scatter(
            x=[dates[0], dates[end - 1]],
            y=[threshold, threshold],
            mode="lines",
            name="Event threshold (selected)",
            line=dict(width=1, dash="dash", color=MARKET_COLORS[target]),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis_title="Google Trends index (each country on its own scale)",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0),
    )
    return fig


def explanation_chart(explanation):
    groups = list(explanation["groups"])[::-1]
    values = [explanation["groups"][group] for group in groups]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=groups,
            orientation="h",
            marker_color=[UP_COLOR if v >= 0 else DOWN_COLOR for v in values],
            marker_cornerradius=4,
            text=[f"{'↑ higher' if v >= 0 else '↓ lower'} ({v:+.2f})" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:+.2f} log-odds<extra></extra>",
        )
    )
    limit = max(abs(v) for v in values) * 1.6 or 1.0
    fig.update_layout(
        height=210,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            title="Contribution to the model's log-odds",
            range=[-limit, limit],
            zeroline=True,
        ),
        showlegend=False,
    )
    return fig


def ladder_chart(evidence):
    scores = evidence["scores"]
    bars = [
        ("Wrong-concept placebo", evidence["placebo_brier"], "#c9c8c2", "/"),
        ("Cross-Market (+ market-specific signals)", scores["model_2"], HIGHLIGHT, ""),
        ("Local + Global", scores["model_1"], NEUTRAL, ""),
        ("Seasonal calendar", scores["model_0"], NEUTRAL, ""),
    ]
    fig = go.Figure(
        go.Bar(
            x=[bar[1] for bar in bars],
            y=[bar[0] for bar in bars],
            orientation="h",
            marker=dict(
                color=[bar[2] for bar in bars],
                pattern_shape=[bar[3] for bar in bars],
                cornerradius=4,
            ),
            text=[f"{bar[1]:.3f}" for bar in bars],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: Brier %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(title="Brier score (lower is better)", rangemode="tozero"),
        showlegend=False,
    )
    return fig


# -------------------------------------------------------------- tab bodies


def render_radar(payload, market):
    evidence = payload["evidence"]
    st.subheader(f"Opportunity Radar · {tv.MARKET_NAMES[market]}")
    st.write(
        f"Probability of an elevated search-interest event in "
        f"{month_label(payload['next_month'])}, ranked for this planner market."
    )

    ranked = tv.rank_forecasts(payload["forecasts"], market)
    first_position = ranked.reset_index().groupby("tie_group")["index"].min() + 1
    for _, row in ranked.iterrows():
        probability = row["probability"]
        band = tv.band_name(probability)
        with st.container(border=True):
            rank_col, concept_col, band_col, note_col, button_col = st.columns(
                [0.9, 1.8, 2.1, 3.6, 1.1], vertical_alignment="center"
            )
            position = first_position[row["tie_group"]]
            rank_col.markdown(f"**{'Tied ' if row['tied'] else ''}#{position}**")
            concept_col.markdown(f"**{tv.CONCEPT_NAMES[row['concept']]}**")
            band_col.markdown(f"{BAND_BADGES[band]} **{tv.format_estimate(probability)}**")
            notes = [hit_rate_text(evidence, band)]
            if row["tied"]:
                notes.append("Tie / similar evidence")
            boundary = tv.nearest_boundary(probability)
            if boundary is not None:
                notes.append(f"Near the {boundary * 100:.0f}% boundary: treat as borderline")
            note_col.caption(" · ".join(notes))
            button_col.button(
                "Explore",
                key=f"explore_{market}_{row['concept']}",
                width="stretch",
                on_click=select_concept,
                args=(row["concept"],),
            )

    st.caption(
        f"Same model for every market and concept. Forecasts within "
        f"{tv.TIE_MARGIN * 100:.0f} points of a group's highest forecast are shown as tied. "
        "These are the five concepts monitored by this prototype, not the five largest "
        "fashion trends."
    )
    st.info(
        f"Selected for drill-down: **{tv.CONCEPT_NAMES[st.session_state.selected_concept]}**. "
        "Open **Market Detail** or **Decision Support** to continue."
    )
    footer()


def render_detail(payload, market, data, dates):
    evidence = payload["evidence"]
    cutoff = payload["cutoff"]
    st.selectbox(
        "Concept",
        tv.CONCEPTS,
        format_func=lambda concept: tv.CONCEPT_NAMES[concept],
        key="selected_concept",
    )
    concept = st.session_state.selected_concept
    forecasts = payload["forecasts"]
    selected = forecasts[(forecasts["concept"] == concept) & (forecasts["target"] == market)]
    probability = float(selected["probability"].iloc[0])
    band = tv.band_name(probability)

    st.subheader(f"{tv.CONCEPT_NAMES[concept]} · {tv.MARKET_NAMES[market]}")
    planner, forecast, horizon = st.columns(3)
    fact(planner, "Planner market", tv.MARKET_NAMES[market])
    fact(forecast, "Elevated-interest forecast", tv.format_forecast(probability))
    fact(horizon, "Forecast horizon", f"Next month ({short_month(payload['next_month'])})")
    st.caption(hit_rate_text(evidence, band))
    boundary = tv.nearest_boundary(probability)
    if boundary is not None:
        st.caption(
            f"This forecast is within {tv.BORDERLINE_MARGIN * 100:.0f} points of the "
            f"{boundary * 100:.0f}% boundary, so it is treated as borderline."
        )
    st.warning("Google search interest ≠ sales demand.")

    st.markdown("#### Forecast in each market")
    columns = st.columns(len(tv.MARKETS))
    for column, name in zip(columns, tv.MARKETS):
        value = float(forecasts[(forecasts["concept"] == concept) & (forecasts["target"] == name)]["probability"].iloc[0])
        column.markdown(
            f"{tv.MARKET_NAMES[name]}  \n{BAND_BADGES[tv.band_name(value)]} **{tv.format_estimate(value)}**"
        )

    st.markdown("#### Why this forecast?")
    row = payload["rows"][
        (payload["rows"]["concept"] == concept) & (payload["rows"]["target"] == market)
    ]
    explanation = tv.explain_forecast(payload["model"], row)
    st.plotly_chart(
        explanation_chart(explanation), width="stretch", key=f"why_{market}_{concept}"
    )
    for group, feature in explanation["top_signals"].items():
        direction = "↑ higher" if explanation["groups"][group] >= 0 else "↓ lower"
        st.markdown(
            f"**{group}** pushes the forecast {direction}. "
            f"Largest signal: {tv.describe_signal(feature)}."
        )
    st.caption(
        f"The model starts at {explanation['start_log_odds']:+.2f} log-odds for this concept "
        f"and market. The three groups add up exactly to {explanation['total_log_odds']:+.2f}. "
        "These are model contributions on correlated signals. They do not show causes. "
        "Global and market-specific signals are grouped as OTHER MARKETS because the "
        "global averages are built from the market-specific values."
    )

    st.markdown("#### Search-interest history in the four markets")
    st.plotly_chart(
        history_chart(data, dates, cutoff, concept, market),
        width="stretch",
        key=f"history_{market}_{concept}_{cutoff:%Y%m}",
    )
    st.warning(
        "Each country's Google Trends index has its own scale, so do not compare heights "
        "between countries. The chart is for visualization only. The model uses changes "
        "within each market, never raw levels across countries."
    )

    st.markdown("#### Historical outcome check")
    st.caption("The future outcome was hidden from the model when it produced this forecast.")
    outcome = tv.historical_outcome(data, dates, cutoff, concept, market)
    if outcome is None:
        st.write("No outcome is available yet for this forecast date.")
    elif st.toggle(
        "Reveal the next-month outcome", key=f"reveal_{cutoff:%Y%m}_{market}_{concept}"
    ):
        st.write(
            f"Actual {month_label(outcome['next_date'])} index: **{outcome['next_value']:.0f}**. "
            f"Event threshold: **{outcome['threshold']:.1f}**. "
            f"Forecast was **{tv.format_forecast(probability)}**."
        )
        if outcome["event"]:
            st.success("The elevated search-interest event occurred.")
        else:
            st.error("The elevated search-interest event did not occur.")
    footer()


def render_decision(payload, market):
    evidence = payload["evidence"]
    concept = st.session_state.selected_concept
    forecasts = payload["forecasts"]
    selected = forecasts[(forecasts["concept"] == concept) & (forecasts["target"] == market)]
    probability = float(selected["probability"].iloc[0])

    st.subheader(f"Decision Support · {tv.CONCEPT_NAMES[concept]} · {tv.MARKET_NAMES[market]}")

    st.markdown("#### 1. Feasibility check")
    fit_col, product_col, act_col, stock_col = st.columns(4)
    brand_fit = fit_col.selectbox("Brand fit (Planner judgement)", ["Strong", "Weak"])
    matching = product_col.selectbox("Matching product?", ["Yes", "No"])
    can_act = act_col.selectbox("Can act within horizon?", ["Yes", "No"])
    stock = stock_col.selectbox(
        "Stock availability", ["Adequate", "Low"], disabled=matching == "No"
    )

    st.markdown("#### 2. Forecast")
    band = tv.band_name(probability)
    st.markdown(f"{BAND_BADGES[band]} **{tv.format_estimate(probability)}**")
    st.caption(hit_rate_text(evidence, band))
    boundary = tv.nearest_boundary(probability)
    if boundary is not None:
        st.caption(
            f"Within ±{tv.BORDERLINE_MARGIN * 100:.0f} points of the {boundary * 100:.0f}% "
            "band boundary: treated as borderline."
        )

    st.markdown("#### 3. Suggested action")
    result = tv.recommend_action(probability, brand_fit, matching, can_act, stock)
    with st.container(border=True):
        st.markdown(f"**{result['action']}**")
        st.write(result["reason"])
        for step in result["steps"]:
            st.markdown(f"- {step}")

    st.caption(
        "A human planner reviews this and can accept, reject or change it. "
        "The forecast is never converted into units."
    )
    with st.expander("How bands map to actions"):
        st.dataframe(
            band_table(evidence)[["Band", "Forecast range", "Default action"]],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"Within ±{tv.BORDERLINE_MARGIN * 100:.0f} percentage points of a band boundary, "
            "the forecast is treated as borderline. Near the 85% boundary, any stock move "
            "stays a planner decision."
        )
    footer()


def render_evidence(payload):
    evidence = payload["evidence"]
    cross = evidence["cross"]
    versus = evidence["versus_seasonal"]
    st.subheader("Evidence")
    st.write(
        f"{cross['months']}-month expanding-window walk-forward across all four markets: "
        f"{short_month(evidence['first_month'])} to {short_month(evidence['last_month'])}, "
        f"{evidence['forecasts']} forecasts. Each month is predicted only from earlier dates."
    )

    if evidence["supportive"]:
        st.success(tv.evidence_takeaway(evidence))
    else:
        st.warning(tv.evidence_takeaway(evidence))
    st.info(tv.seasonal_statement(evidence))

    st.plotly_chart(
        ladder_chart(evidence), width="stretch", key=f"ladder_{payload['cutoff']:%Y%m}"
    )

    improvement, interval = st.columns(2)
    months, placebo = st.columns(2)
    improvement.metric(
        "Cross-Market vs Local + Global",
        f"{cross['gain'] * 100:+.1f}%",
        help="Relative Brier improvement. Positive means the Cross-Market model is better.",
        border=True,
    )
    interval.metric(
        "Bootstrap 95% CI",
        f"{cross['ci_low'] * 100:+.1f}% to {cross['ci_high'] * 100:+.1f}%",
        help=f"Month-block bootstrap with {tv.BOOTSTRAP_DRAWS} draws and a fixed seed.",
        border=True,
    )
    months.metric(
        "Months Cross-Market is better",
        f"{cross['months_better']}/{cross['months']}",
        border=True,
    )
    placebo.metric(
        "Placebo vs Local + Global",
        f"{evidence['placebo_gain'] * 100:+.1f}%",
        help=f"Average of {tv.PLACEBO_SHUFFLES} wrong-concept shuffles. Best single shuffle: "
        f"{evidence['placebo_best_gain'] * 100:+.1f}%.",
        border=True,
    )

    st.markdown("#### Decision rule")
    mark = {True: "✅ Met", False: "❌ Not met"}
    st.markdown(
        f"- 95% interval versus Local + Global is above zero: {mark[evidence['ci_above_zero']]}\n"
        f"- Wrong-concept placebo signals produce no average gain: {mark[evidence['placebo_no_gain']]}"
    )
    all_evidence = payload["evidence_by_cutoff"].values()
    dates_met = sum(item["supportive"] for item in all_evidence)
    st.warning(
        f"This rule is met at {dates_met} of {len(all_evidence)} selectable dates. "
        "The result depends on the window, and these windows share most of their months."
    )
    st.caption(
        f"Best single placebo shuffle: {evidence['placebo_best_gain'] * 100:+.1f}%. Forecasts "
        "always come from the Cross-Market model. This rule changes what the evidence is "
        "called, not which model runs."
    )
    st.caption(
        "This is not an untouched test: the rule and model were set after an audit had "
        "already studied this same data. Only a forecast on new months can confirm it."
    )
    st.markdown(
        f"Full model vs seasonal calendar: **{versus['gain'] * 100:+.1f}%**, "
        f"95% CI {versus['ci_low'] * 100:+.1f}% to {versus['ci_high'] * 100:+.1f}%, "
        f"better in {versus['months_better']}/{versus['months']} months."
    )

    st.markdown("#### What each step of the ladder adds")
    step = evidence["global_step"]
    st.markdown(
        f"- **Global step:** local history alone scores {evidence['local_only_brier']:.3f}. "
        f"Adding global movement changes that by **{step['gain'] * 100:+.1f}%** "
        f"(95% CI {step['ci_low'] * 100:+.1f}% to {step['ci_high'] * 100:+.1f}%, "
        f"better in {step['months_better']}/{step['months']} months).\n"
        f"- **Market-identity check:** keeping the same values but randomising which market "
        f"produced them scores {evidence['shuffle_brier']:.3f}, or "
        f"**{evidence['shuffle_gain'] * 100:+.1f}%** versus Local + Global. "
        f"{evidence['shuffles_beating_true']} of {tv.PLACEBO_SHUFFLES} shuffles beat the "
        "true Cross-Market model."
    )
    st.caption(
        "The identity check is a diagnostic, not part of the decision rule. Source-market "
        "effects are pooled across destinations, so no specific pair, such as one market "
        "leading another, has been validated. These results are predictive, not causal."
    )

    st.markdown("#### How the forecast bands behaved")
    st.dataframe(
        band_table(evidence).drop(columns=["Default action"]),
        hide_index=True,
        width="stretch",
    )

    with st.expander("The same test at every selectable date"):
        st.dataframe(stability_table(payload), hide_index=True, width="stretch")
        st.caption(
            "These windows share most of their months, so they are not independent "
            "replications. All dates are listed so that none is privileged."
        )
    with st.expander("Results by market (descriptive only)"):
        table = pd.DataFrame(
            [
                {
                    "Destination market": tv.MARKET_NAMES[market],
                    "Seasonal calendar": round(scores["model_0"], 3),
                    "Local + Global": round(scores["model_1"], 3),
                    "Cross-Market": round(scores["model_2"], 3),
                    "Improvement": f"{scores['cross']['gain'] * 100:+.1f}%",
                    "95% CI": f"{scores['cross']['ci_low'] * 100:+.1f}% to {scores['cross']['ci_high'] * 100:+.1f}%",
                    "Months better": f"{scores['cross']['months_better']}/{scores['cross']['months']}",
                }
                for market, scores in evidence["by_market"].items()
            ]
        )
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("The decision rule is pooled. Each market alone has too few months to judge.")
    footer()


def stability_table(payload):
    rows = []
    for cutoff, evidence in payload["evidence_by_cutoff"].items():
        cross = evidence["cross"]
        scores = evidence["scores"]
        rows.append(
            {
                "Data up to": short_month(cutoff),
                "Seasonal": round(scores["model_0"], 3),
                "Local + Global": round(scores["model_1"], 3),
                "Cross-Market": round(scores["model_2"], 3),
                "Placebo": round(evidence["placebo_brier"], 3),
                "Improvement": f"{cross['gain'] * 100:+.1f}%",
                "95% CI": f"{cross['ci_low'] * 100:+.1f}% to {cross['ci_high'] * 100:+.1f}%",
                "Months better": f"{cross['months_better']}/{cross['months']}",
                "Placebo best shuffle": f"{evidence['placebo_best_gain'] * 100:+.1f}%",
                "Rule met": "Yes" if evidence["supportive"] else "No",
            }
        )
    return pd.DataFrame(rows)


def render_scope(summary):
    st.subheader("Prototype Scope")
    st.write(PROXY_STATEMENT)
    st.markdown(EVENT_DEFINITION)
    left, right = st.columns(2)
    with left:
        st.markdown("#### What this academic prototype is")
        st.markdown(
            f"""
- {summary['concepts']} fashion concepts
- {summary['markets']} countries
- {summary['observations']} monthly observations per series
- Approximately {summary['forecast_months']} usable forecast months
- {summary['rows']} pooled concept–market rows, from only ~{summary['forecast_months']} distinct forecast months
- Monthly Google Trends proxy data
- One consistent regularised logistic architecture
- Evidence ladder: seasonal calendar → Local + Global → Cross-Market
- Chronological walk-forward validation
- Placebo test
- Bootstrap uncertainty
"""
        )
    with right:
        st.markdown("#### What commercial TrendVector would add")
        st.markdown(
            """
- Licensed visual, social and retail-intelligence data
- Multilingual concept mapping across many more concepts and countries
- Multiple operational forecast horizons
- Catalogue, stock and operational integrations
- Larger samples for stronger probability calibration
- Prospective customer pilots before claiming business impact
"""
        )
    st.markdown("#### Limitations")
    st.markdown(
        f"""
- Small number of independent months: {summary['rows']} rows do not mean {summary['rows']} independent observations
- Google Trends sampling and normalisation: each download is a sample, and each country has its own scale
- Low-resolution terms: small series move in a few integer values, so one point can cross the threshold
- Strong fashion seasonality: a plain seasonal calendar is a hard baseline
- English terms across all countries, which reach different audiences in each market
- Search interest is not demand
- Source-market effects are pooled across destinations, so no specific lead-market pair is validated, and the results are predictive, not causal
- The commercial system requires richer data
- The event rewards staying high as well as growing: in {summary['persisted_share'] * 100:.0f}% of events, next month's interest was not above the current month's
"""
    )
    st.caption(
        "The 10% threshold is a transparent academic choice, not an industry standard. "
        "Gradient boosting was tested during the audit and showed no gain at this sample size."
    )
    footer()


# -------------------------------------------------------------------- app

data, dates, pooled = load_all()
available_cutoffs = sorted(pooled["date"].unique())[-8:]
available_cutoffs = [pd.Timestamp(date) for date in available_cutoffs]

st.title("TrendVector V3")
st.caption("Academic proof of concept · not the commercial product")
st.info(f"{PROXY_STATEMENT}\n\n{SCOPE_STATEMENT}")
st.markdown(EVENT_DEFINITION)

with st.sidebar:
    st.header("Prototype controls")
    cutoff = st.selectbox(
        "Data available up to",
        available_cutoffs,
        index=len(available_cutoffs) - 1,
        format_func=month_label,
        key="cutoff_choice",
    )
    market = st.selectbox(
        "Planner market",
        tv.MARKETS,
        format_func=lambda code: tv.MARKET_NAMES[code],
        key="market_choice",
    )

cutoff = pd.Timestamp(cutoff)
next_month = cutoff + pd.DateOffset(months=1)
st.markdown(f"### Data up to **{month_label(cutoff)}** → forecasting **{month_label(next_month)}**")

if st.button("Train model & generate forecast", type="primary", width="stretch"):
    with st.spinner("Training the model and running the 24-month walk-forward validation..."):
        predictions = walk_forward_table(pooled, available_cutoffs)
        model, rows, forecasts = tv.generate_forecasts(pooled, cutoff)
        evidence_by_cutoff = {
            date: tv.compute_evidence(predictions, date) for date in available_cutoffs
        }
    st.session_state.payload = {
        "cutoff": cutoff,
        "next_month": next_month,
        "model": model,
        "rows": rows,
        "forecasts": forecasts,
        "evidence": evidence_by_cutoff[cutoff],
        "evidence_by_cutoff": evidence_by_cutoff,
        "train_rows": int((pooled["date"] < cutoff).sum()),
    }

payload = st.session_state.get("payload")
if payload is None or payload["cutoff"] != cutoff:
    st.warning("Press the button to train the model and generate forecasts for this date.")
    st.stop()

st.caption(
    f"Cross-Market logistic model trained on {payload['train_rows']} rows dated before "
    f"{month_label(cutoff)}, and used for all markets and concepts."
)

if "selected_concept" not in st.session_state:
    st.session_state.selected_concept = tv.rank_forecasts(payload["forecasts"], market)[
        "concept"
    ].iloc[0]

radar_tab, detail_tab, decision_tab, evidence_tab, scope_tab = st.tabs(
    ["Opportunity Radar", "Market Detail", "Decision Support", "Evidence", "Prototype Scope"]
)
with radar_tab:
    render_radar(payload, market)
with detail_tab:
    render_detail(payload, market, data, dates)
with decision_tab:
    render_decision(payload, market)
with evidence_tab:
    render_evidence(payload)
with scope_tab:
    render_scope(tv.dataset_summary(data, dates, pooled))
