"""
ClinIQ Dashboard — Design System
Colour tokens, typography, reusable component factories.
Dark mode by default; matches VoltEdge pattern for consistency.
"""
from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

COLOURS = {
    # Backgrounds
    "bg_dark":      "#0f1117",
    "bg_card":      "#1a1d2e",
    "bg_surface":   "#242638",

    # Brand
    "accent":       "#6366f1",   # Indigo — clinical, trustworthy
    "accent_light": "#818cf8",

    # Status
    "green":        "#22c55e",
    "amber":        "#f59e0b",
    "red":          "#ef4444",
    "blue":         "#3b82f6",

    # Text
    "text_primary":   "#f1f5f9",
    "text_secondary": "#94a3b8",
    "text_muted":     "#64748b",

    # Borders
    "border":       "#2d3148",
}

RISK_THRESHOLDS = {"low": 40, "medium": 70}  # < 40 = green, 40-70 = amber, > 70 = red


def risk_colour(score: float) -> str:
    if score < RISK_THRESHOLDS["low"]:
        return COLOURS["green"]
    if score < RISK_THRESHOLDS["medium"]:
        return COLOURS["amber"]
    return COLOURS["red"]


def risk_badge_colour(score: float) -> str:
    if score < RISK_THRESHOLDS["low"]:
        return "success"
    if score < RISK_THRESHOLDS["medium"]:
        return "warning"
    return "danger"


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

SIDEBAR_WIDTH   = "240px"

CARD_STYLE = {
    "backgroundColor": COLOURS["bg_card"],
    "border":          f"1px solid {COLOURS['border']}",
    "borderRadius":    "12px",
    "padding":         "1.25rem",
    "marginBottom":    "1rem",
}

CARD_HEADER_STYLE = {
    "color":        COLOURS["text_secondary"],
    "fontSize":     "0.75rem",
    "fontWeight":   "600",
    "textTransform": "uppercase",
    "letterSpacing": "0.08em",
    "marginBottom": "0.5rem",
}

# ---------------------------------------------------------------------------
# Reusable component factories
# ---------------------------------------------------------------------------

def metric_card(title: str, value: str, subtitle: str = "", colour: str = None) -> html.Div:
    return html.Div([
        html.P(title, style=CARD_HEADER_STYLE),
        html.H3(
            value,
            style={"color": colour or COLOURS["text_primary"], "margin": "0",
                   "fontSize": "1.75rem", "fontWeight": "700"},
        ),
        html.P(subtitle, style={"color": COLOURS["text_muted"], "margin": "0",
                                "fontSize": "0.8rem"}) if subtitle else None,
    ], style=CARD_STYLE)


def alert_badge(label: str) -> dbc.Badge:
    colour_map = {
        "HIGH_RISK":       "danger",
        "DROPOUT_RISK":    "warning",
        "DATA_LAG":        "warning",
        "BEHIND_SCHEDULE": "danger",
        "OFF_TRACK":       "secondary",
    }
    return dbc.Badge(
        label.replace("_", " "),
        color=colour_map.get(label, "secondary"),
        className="me-1",
        style={"fontSize": "0.7rem"},
    )


def section_header(title: str, subtitle: str = "") -> html.Div:
    return html.Div([
        html.H4(title, style={"color": COLOURS["text_primary"], "fontWeight": "700",
                               "margin": "0 0 0.25rem"}),
        html.P(subtitle, style={"color": COLOURS["text_muted"], "margin": "0",
                                 "fontSize": "0.85rem"}) if subtitle else None,
    ], style={"marginBottom": "1.25rem"})

