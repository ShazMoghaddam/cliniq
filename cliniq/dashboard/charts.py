"""
ClinIQ Dashboard — Chart builders
All charts use the dark theme palette. Each function is pure (no DB access).
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import plotly.graph_objects as go

from cliniq.dashboard.design import COLOURS, risk_colour

# Shared transparent base — no axis keys, add per-chart to avoid duplicate kwargs
_BASE = {
    "template":      "none",
    "paper_bgcolor": "#1a1d2e",
    "plot_bgcolor":  "#1a1d2e",
    "font":          {"color": COLOURS["text_primary"],
                      "family": "Inter, system-ui, sans-serif"},
    "margin":        {"t": 30, "b": 40, "l": 50, "r": 20},
}


def enrolment_curve(
    dates: list, enrolled_cumulative: list[int], target: int,
    projected_completion: Optional[date] = None,
    title: str = "Enrolment Progress",
) -> go.Figure:
    fig = go.Figure()
    if dates and enrolled_cumulative:
        fig.add_trace(go.Scatter(
            x=dates, y=enrolled_cumulative,
            mode="lines+markers", name="Enrolled",
            line={"color": COLOURS["accent"], "width": 2},
            marker={"size": 4},
        ))
    if dates:
        fig.add_hline(y=target, line_dash="dash", line_color=COLOURS["green"],
                      annotation_text=f"Target: {target}",
                      annotation_font_color=COLOURS["green"])
    if projected_completion:
        fig.add_vline(x=str(projected_completion), line_dash="dot",
                      line_color=COLOURS["amber"],
                      annotation_text="Projected",
                      annotation_font_color=COLOURS["amber"])
    fig.update_layout(
        **_BASE,
        title={"text": title, "x": 0},
        legend={"font": {"color": COLOURS["text_secondary"]}},
        xaxis={"gridcolor": COLOURS["border"], "zerolinecolor": COLOURS["border"], "color": COLOURS["text_secondary"]},
        yaxis={"title": "Patients enrolled", "gridcolor": COLOURS["border"]},
    )
    return fig


def deviation_timeline(
    months: list[str], minor_counts: list[int],
    major_counts: list[int], critical_counts: list[int],
    title: str = "Deviation Timeline",
) -> go.Figure:
    fig = go.Figure()
    for label, counts, colour in [
        ("Minor",    minor_counts,    COLOURS["green"]),
        ("Major",    major_counts,    COLOURS["amber"]),
        ("Critical", critical_counts, COLOURS["red"]),
    ]:
        fig.add_trace(go.Bar(x=months, y=counts, name=label, marker_color=colour))
    fig.update_layout(
        **_BASE,
        title={"text": title, "x": 0},
        barmode="stack",
        legend={"font": {"color": COLOURS["text_secondary"]}},
        xaxis={"gridcolor": COLOURS["border"]},
        yaxis={"gridcolor": COLOURS["border"]},
    )
    return fig


def lag_trend(
    dates: list, lag_mean: list[float], lag_p90: list[float],
    title: str = "Data Lag Trend",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=lag_mean, mode="lines", name="Mean lag",
                             line={"color": COLOURS["accent"], "width": 2}))
    fig.add_trace(go.Scatter(x=dates, y=lag_p90, mode="lines", name="P90 lag",
                             line={"color": COLOURS["amber"], "width": 2, "dash": "dash"}))
    fig.add_hline(y=7, line_dash="dot", line_color=COLOURS["green"],
                  annotation_text="7-day target",
                  annotation_font_color=COLOURS["green"])
    fig.update_layout(
        **_BASE,
        title={"text": title, "x": 0},
        legend={"font": {"color": COLOURS["text_secondary"]}},
        xaxis={"gridcolor": COLOURS["border"]},
        yaxis={"title": "Days", "gridcolor": COLOURS["border"]},
    )
    return fig


def screening_funnel(
    screened: int, enrolled: int, withdrawn: int, completed: int,
    title: str = "Screening Funnel",
) -> go.Figure:
    stages  = ["Screened", "Enrolled", "Withdrawn", "Completed"]
    values  = [screened, enrolled, withdrawn, completed]
    colours = [COLOURS["blue"], COLOURS["accent"],
               COLOURS["amber"], COLOURS["green"]]
    fig = go.Figure(go.Bar(
        x=values, y=stages, orientation="h",
        marker_color=colours, text=values, textposition="auto",
    ))
    fig.update_layout(
        **_BASE,
        title={"text": title, "x": 0},
        showlegend=False,
        xaxis={"title": "Patients", "gridcolor": COLOURS["border"]},
        yaxis={"gridcolor": "rgba(0,0,0,0)"},
    )
    return fig


def portfolio_heatmap(
    site_ids: list[str], risk_scores: list[float],
    enrolment_pcts: list[float], velocities: list[float],
    countries: list[str], title: str = "Portfolio Risk Heatmap",
) -> go.Figure:
    colours = [risk_colour(s) for s in risk_scores]
    sizes   = [max(8, min(30, v * 60 + 8)) for v in velocities]
    fig = go.Figure(go.Scatter(
        x=enrolment_pcts, y=risk_scores,
        mode="markers",
        text=site_ids,
        marker={"size": sizes, "color": colours,
                "line": {"width": 1, "color": COLOURS["border"]}},
        customdata=list(zip(site_ids, countries, velocities)),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Risk: %{y:.0f}/100<br>"
            "Enrolled: %{x:.0%}<br>"
            "Country: %{customdata[1]}<br>"
            "Velocity: %{customdata[2]:.3f} pts/day<br>"
            "<extra></extra>"
        ),
    ))
    fig.add_hline(y=70, line_dash="dash", line_color=COLOURS["red"],   line_width=1)
    fig.add_hline(y=40, line_dash="dash", line_color=COLOURS["amber"],  line_width=1)
    fig.add_vline(x=0.5, line_dash="dash", line_color=COLOURS["border"], line_width=1)
    fig.update_layout(
        **_BASE,
        title={"text": title, "x": 0},
        xaxis={"title": "Enrolment progress (%)", "tickformat": ".0%",
               "gridcolor": COLOURS["border"]},
        yaxis={"title": "Composite risk score", "gridcolor": COLOURS["border"],
               "range": [-5, 105]},
    )
    return fig


def risk_breakdown(
    enrolment_c: float, deviation_c: float, lag_c: float,
    dropout_c: float, monitoring_c: float, weights: dict,
    title: str = "Risk Score Breakdown",
) -> go.Figure:
    components = {
        "Enrolment shortfall":  enrolment_c  * weights["enrolment_shortfall"],
        "Deviation rate":       deviation_c  * weights["deviation_rate"],
        "Data lag":             lag_c        * weights["data_lag"],
        "Dropout probability":  dropout_c    * weights["dropout_probability"],
        "Monitoring recency":   monitoring_c * weights["monitoring_recency"],
    }
    component_colours = [COLOURS["amber"], COLOURS["red"], COLOURS["blue"],
                         COLOURS["accent"], COLOURS["text_muted"]]
    fig = go.Figure(go.Bar(
        x=list(components.values()), y=list(components.keys()),
        orientation="h", marker_color=component_colours,
        text=[f"{v:.1f}" for v in components.values()],
        textposition="auto",
    ))
    base = dict(_BASE)
    base["margin"] = {"t": 30, "b": 40, "l": 160, "r": 20}
    fig.update_layout(
        **base,
        title={"text": title, "x": 0},
        showlegend=False,
        xaxis={"title": "Score contribution", "range": [0, 40],
               "gridcolor": COLOURS["border"], "color": COLOURS["text_secondary"]},
        yaxis={"gridcolor": "rgba(0,0,0,0)", "color": COLOURS["text_secondary"]},
    )
    return fig
