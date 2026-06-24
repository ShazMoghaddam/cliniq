"""ClinIQ Dashboard — Plotly Dash application."""
from __future__ import annotations

import os
from datetime import date
from collections import defaultdict
from typing import Optional

from dash import Dash, Input, Output, State, dcc, html, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from cliniq.dashboard.design import (
    alert_badge, risk_badge_colour,
)
from cliniq.dashboard.charts import (
    portfolio_heatmap, risk_breakdown,
)
from cliniq.db.database import SessionLocal, init_db
from cliniq.db.models import (
    DeviationSeverity, EnrolmentStatus, PatientEnrolment,
    ProtocolDeviation, Trial, TrialSite,
)
from cliniq.ml.risk_model import compute_risk_score, WEIGHTS
from cliniq.analytics.velocity import compute_velocity
from cliniq.analytics.lag import compute_lag

# ---------------------------------------------------------------------------
# Icon helper — Bootstrap Icons via CSS class (bi bi-*)
# dash.html.Svg was removed in Dash 4; BI classes are the reliable fallback
# ---------------------------------------------------------------------------

def bi(name: str, colour: str = None) -> html.I:
    style = {"verticalAlign": "middle", "marginRight": "0.1rem"}
    if colour:
        style["color"] = colour
    return html.I(className=f"bi bi-{name}", style=style)

ICON_PORTFOLIO = "grid-3x3-gap-fill"
ICON_WATCHLIST = "exclamation-triangle-fill"
ICON_ASSISTANT = "chat-dots-fill"
ICON_RISK      = "shield-exclamation"
ICON_ENROLLED  = "person-plus-fill"
ICON_LAG       = "clock-history"
ICON_DROPOUT   = "graph-down-arrow"

# ---------------------------------------------------------------------------
# DB helper
# ---------------------------------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise

# ---------------------------------------------------------------------------
# Shared CSS injected into <head>
# ---------------------------------------------------------------------------

GLOBAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, html {
    margin: 0; padding: 0;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background: #0f1117;
    color: #f1f5f9;
    font-size: 15px;
    line-height: 1.6;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #1a1d2e; }
::-webkit-scrollbar-thumb { background: #2d3148; border-radius: 3px; }

/* ── Metric cards ──────────────────────────────────────────────────────── */
.metric-card {
    background: #1a1d2e;
    border: 1px solid #2d3148;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.metric-card:hover { border-color: #6366f1; box-shadow: 0 0 0 1px #6366f122; }
.metric-label {
    font-size: 0.7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; margin: 0;
}
.metric-value { font-size: 1.8rem; font-weight: 700; margin: 0; line-height: 1.2; }
.metric-sub  { font-size: 0.75rem; color: #64748b; margin: 0; }

/* ── Sidebar nav ───────────────────────────────────────────────────────── */
.nav-link-item {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.55rem 0.85rem; border-radius: 8px; margin-bottom: 4px;
    font-size: 0.875rem; font-weight: 500; color: #94a3b8;
    text-decoration: none; transition: all 0.15s; cursor: pointer;
}
.nav-link-item:hover { background: #242638; color: #f1f5f9; text-decoration: none; }
.nav-link-item.active { background: #242638; color: #f1f5f9; font-weight: 600; }

/* ── Page cards ─────────────────────────────────────────────────────────── */
.page-card {
    background: #1a1d2e; border: 1px solid #2d3148;
    border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
}

/* ── AI example questions ──────────────────────────────────────────────── */
.example-question {
    background: #242638; border: 1px solid #2d3148; border-radius: 8px;
    padding: 0.55rem 0.9rem; font-size: 0.82rem; color: #94a3b8;
    cursor: pointer; transition: all 0.15s; display: inline-block; margin: 0.25rem;
}
.example-question:hover { border-color: #6366f1; color: #f1f5f9; }

/* ── Chat bubbles ──────────────────────────────────────────────────────── */
.chat-bubble-user {
    background: #6366f1; color: #fff;
    padding: 0.6rem 1rem; border-radius: 12px 12px 2px 12px;
    margin-bottom: 0.5rem; max-width: 80%; margin-left: auto;
    font-size: 0.875rem; line-height: 1.5;
}
.chat-bubble-ai {
    background: #242638; color: #f1f5f9;
    padding: 0.6rem 1rem; border-radius: 12px 12px 12px 2px;
    margin-bottom: 0.5rem; max-width: 80%;
    font-size: 0.875rem; line-height: 1.5; border: 1px solid #2d3148;
}

/* ── Footer ─────────────────────────────────────────────────────────────── */
.footer-bar {
    position: fixed; bottom: 0; left: 240px; right: 0;
    background: #0f1117; border-top: 1px solid #2d3148;
    padding: 0.5rem 2rem; display: flex; align-items: center;
    justify-content: center; gap: 0.5rem;
    font-size: 0.75rem; color: #64748b; z-index: 50;
}
.footer-bar a { color: #94a3b8; text-decoration: none; transition: color 0.15s; }
.footer-bar a:hover { color: #6366f1; }

/* ── Mobile top bar ─────────────────────────────────────────────────────── */
.mobile-topbar {
    display: none; position: fixed; top: 0; left: 0; right: 0;
    height: 52px; background: #1a1d2e; border-bottom: 1px solid #2d3148;
    align-items: center; justify-content: space-between;
    padding: 0 1rem; z-index: 200;
}
.mobile-logo { font-size: 1.15rem; font-weight: 800; letter-spacing: -0.3px; }
.mobile-nav-drawer {
    display: none; position: fixed; top: 52px; left: 0; right: 0;
    background: #1a1d2e; border-bottom: 1px solid #2d3148;
    padding: 0.5rem 1rem 0.75rem; z-index: 199;
    flex-direction: column; gap: 0.25rem;
}
.mobile-nav-drawer.open { display: flex; }
.mobile-nav-link {
    padding: 0.65rem 0.85rem; border-radius: 8px; color: #94a3b8;
    font-size: 0.9rem; font-weight: 500; text-decoration: none;
    display: flex; align-items: center; gap: 0.5rem;
}
.mobile-nav-link:hover { background: #242638; color: #f1f5f9; text-decoration: none; }
.mobile-hamburger {
    background: none; border: none; color: #94a3b8;
    font-size: 1.4rem; cursor: pointer; padding: 0.25rem; line-height: 1;
}

/* ── Responsive ─────────────────────────────────────────────────────────── */
@media (max-width: 768px) {
    .sidebar-wrap { display: none; }
    .main-content { margin-left: 0 !important; padding: 4.5rem 1rem 5rem !important; }
    .footer-bar { left: 0; }
    .mobile-topbar { display: flex; }
}

/* ── Dash 4 dropdown dark theme ─────────────────────────────────────────── */
/* Class names from async-dropdown.js in Dash 4.3.0                         */
.dash-dropdown-wrapper {
    background-color: #1e2235 !important;
    border: 1px solid #2d3148 !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
    overflow: hidden !important;
}
.dash-dropdown-wrapper:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 1px #6366f133 !important;
}
.dash-dropdown-trigger,
.dash-dropdown-grid-container {
    background-color: #1e2235 !important;
    color: #f1f5f9 !important;
    min-height: 38px !important;
    border-radius: 0 !important;
    cursor: pointer !important;
}
.dash-dropdown-value,
.dash-dropdown-value-item {
    color: #f1f5f9 !important;
    background-color: transparent !important;
    font-size: 0.875rem !important;
}
.dash-dropdown-placeholder { color: #64748b !important; font-size: 0.875rem !important; }
.dash-dropdown-trigger-icon,
.dash-dropdown-action-button { color: #64748b !important; }
.dash-dropdown-content {
    background-color: #1e2235 !important;
    border: 1px solid #2d3148 !important;
    border-radius: 8px !important;
    margin-top: 2px !important;
    z-index: 9999 !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4) !important;
}
.dash-dropdown-search-container {
    background-color: #242638 !important;
    border-bottom: 1px solid #2d3148 !important;
    padding: 6px 8px !important;
}
.dash-dropdown-search {
    background-color: #1a1d2e !important;
    border: 1px solid #2d3148 !important;
    border-radius: 6px !important;
    color: #f1f5f9 !important;
    font-size: 0.85rem !important;
    padding: 4px 8px !important;
    width: 100% !important;
    outline: none !important;
}
.dash-dropdown-search::placeholder { color: #64748b !important; }
.dash-dropdown-search-icon { color: #64748b !important; }
.dash-dropdown-options {
    background-color: #1e2235 !important;
    padding: 4px 0 !important;
}
.dash-dropdown-option {
    background-color: #1e2235 !important;
    color: #f1f5f9 !important;
    font-size: 0.875rem !important;
    padding: 8px 12px !important;
    cursor: pointer !important;
}
.dash-dropdown-option:hover { background-color: #2d3148 !important; color: #fff !important; }
.dash-dropdown-option[aria-selected="true"] {
    background-color: #6366f1 !important;
    color: #ffffff !important;
}
.dash-dropdown-clear { color: #64748b !important; }
.dash-dropdown-clear:hover { color: #f1f5f9 !important; }
"""

FAVICON_SVG = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 120 40'>
  <rect width='120' height='40' fill='#0f1117' rx='6'/>
  <text x='8' y='28' font-family='Inter,system-ui,sans-serif' font-size='20'
        font-weight='800' fill='#ffffff' letter-spacing='-0.5'>Clin</text>
  <text x='52' y='28' font-family='Inter,system-ui,sans-serif' font-size='20'
        font-weight='800' fill='#6366f1' letter-spacing='-0.5'>IQ</text>
</svg>"""


def _auto_seed():
    """Seed the database automatically if it contains no trial data.
    Called on every startup — safe to call repeatedly (idempotent)."""
    from cliniq.db.models import Trial
    from cliniq.db.seeder import seed, _wipe
    db = SessionLocal()
    try:
        if db.query(Trial).count() == 0:
            print("ClinIQ: database empty — seeding demo data...")
            _wipe(db)           # clear any partial data first
            result = seed(db)
            print(f"ClinIQ: seeded {result.get('sites', 0)} sites, "
                  f"{result.get('patient_enrolments', 0)} enrolments")
    except Exception as e:
        print(f"ClinIQ: auto-seed warning: {e}")
        db.rollback()
    finally:
        db.close()


def create_dashboard(server=None) -> Dash:
    init_db()
    _auto_seed()

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP],
        suppress_callback_exceptions=True,
        title="ClinIQ",
        use_pages=False,
        meta_tags=[{"name": "viewport",
                    "content": "width=device-width, initial-scale=1.0"}],
    )

    # Inject global CSS + favicon
    app.index_string = f"""<!DOCTYPE html>
<html>
  <head>
    {{%metas%}}
    <title>{{%title%}}</title>
    <link rel="icon" type="image/svg+xml"
          href="data:image/svg+xml,{FAVICON_SVG.replace('#', '%23')}">
    {{%css%}}
    <style>{GLOBAL_CSS}</style>
  </head>
  <body>
    {{%app_entry%}}
    <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
  </body>
</html>"""

    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------

    def make_sidebar(active_path: str = "/") -> html.Div:
        nav_items = [
            ("/",          ICON_PORTFOLIO,  "Portfolio"),
            ("/watchlist", ICON_WATCHLIST,  "Watchlist"),
            ("/assistant", ICON_ASSISTANT,  "AI Assistant"),
        ]
        links = []
        for path, icon_d, label in nav_items:
            is_active = active_path == path
            links.append(
                html.A(
                    [bi(icon_d), html.Span(label)],
                    href=path,
                    className=f"nav-link-item{'  active' if is_active else ''}",
                )
            )

        return html.Div([
            # Logo
            html.Div([
                html.Div([
                    html.Span("Clin", style={"color": "#ffffff", "fontWeight": "800",
                                              "fontSize": "1.5rem", "letterSpacing": "-0.5px"}),
                    html.Span("IQ",   style={"color": "#6366f1", "fontWeight": "800",
                                              "fontSize": "1.5rem", "letterSpacing": "-0.5px"}),
                ], style={"display": "flex", "alignItems": "baseline"}),
                html.P("Clinical Intelligence",
                       style={"color": "#64748b", "fontSize": "0.7rem",
                               "margin": "0", "letterSpacing": "0.06em",
                               "textTransform": "uppercase"}),
            ], style={"padding": "1.5rem 1.25rem 1.25rem"}),

            # Nav
            html.Div(links, style={"padding": "0 0.75rem"}),
        ],
        className="sidebar-wrap",
        style={
            "width": "240px",
            "minHeight": "100vh",
            "backgroundColor": "#1a1d2e",
            "borderRight": "1px solid #2d3148",
            "position": "fixed",
            "top": 0, "left": 0,
            "zIndex": 100,
            "paddingBottom": "4rem",
        })

    # -----------------------------------------------------------------------
    # Header bar (time + date)
    # -----------------------------------------------------------------------

    def make_header(title: str, subtitle: str = "") -> html.Div:
        return html.Div([
            html.Div([
                html.H2(title, style={
                    "margin": "0", "fontWeight": "800", "fontSize": "1.6rem",
                    "color": "#f1f5f9", "letterSpacing": "-0.3px",
                }),
                html.P(subtitle, style={
                    "margin": "0.1rem 0 0", "color": "#64748b",
                    "fontSize": "0.82rem",
                }) if subtitle else None,
            ]),
            html.Div([
                html.Div(id="live-clock", style={
                    "textAlign": "right", "color": "#94a3b8",
                    "fontSize": "0.82rem", "lineHeight": "1.5",
                }),
            ]),
        ], style={
            "display": "flex", "justifyContent": "space-between",
            "alignItems": "flex-start", "marginBottom": "1.75rem",
            "paddingBottom": "1.25rem",
            "borderBottom": "1px solid #2d3148",
        })

    # -----------------------------------------------------------------------
    # Uniform metric card
    # -----------------------------------------------------------------------

    def uniform_metric(icon_d: str, label: str, value: str,
                       sub: str = "", colour: str = "#f1f5f9") -> html.Div:
        return html.Div([
            html.P(label, className="metric-label"),
            html.Div([
                bi(icon_d, colour),
                html.Span(value, className="metric-value",
                          style={"color": colour, "marginLeft": "0.5rem"}),
            ], style={"display": "flex", "alignItems": "center"}),
            html.P(sub, className="metric-sub"),
        ], className="metric-card")

    # -----------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------

    def make_footer() -> html.Div:
        return html.Div([
            html.Span("Developed by"),
            html.A("Shaz Moghaddam",
                   href="https://shazmoghaddam.github.io/",
                   target="_blank"),
            html.Span("|"),
            html.A("LinkedIn",
                   href="https://www.linkedin.com/in/shazmoghaddam/",
                   target="_blank"),
            html.Span("| London"),
        ], className="footer-bar")

    # -----------------------------------------------------------------------
    # Page layouts
    # -----------------------------------------------------------------------

    def portfolio_layout(trial_id=None) -> html.Div:
        """Build portfolio page inline — no async callback needed."""
        fig_content = html.Div(
            html.P("Select a trial to view the portfolio",
                   style={"color": "#64748b", "textAlign": "center",
                          "paddingTop": "3rem"}),
        )
        site_cards = []

        if trial_id is not None:
            db = get_db()
            try:
                from cliniq.dashboard.data import get_portfolio_risk
                summaries = get_portfolio_risk(db, trial_id)
                if summaries:
                    live_pcts, live_velocities = [], []
                    for s in summaries:
                        v = compute_velocity(db, s.trial_site_id)
                        pct = min(1.0, (v.enrolled_to_date / v.enrolment_target)) if v.enrolment_target else 0.0
                        live_pcts.append(round(pct, 4))
                        live_velocities.append(v.velocity_28d)

                    fig = portfolio_heatmap(
                        [s.site_id for s in summaries],
                        [s.composite_score for s in summaries],
                        live_pcts, live_velocities,
                        [s.country for s in summaries],
                    )
                    fig_content = dcc.Graph(figure=fig, style={"height": "420px"},
                                            config={"displayModeBar": False})

                    site_cards = dbc.Row([
                        dbc.Col(html.Div([
                            html.Div([
                                html.Span(s.site_id,
                                          style={"fontWeight": "700", "color": "#f1f5f9",
                                                  "fontSize": "0.9rem"}),
                                dbc.Badge(f"{s.composite_score:.0f}",
                                          color=risk_badge_colour(s.composite_score),
                                          className="ms-2", style={"fontSize": "0.75rem"}),
                            ], style={"display": "flex", "alignItems": "center",
                                       "marginBottom": "0.25rem"}),
                            html.P(s.site_name,
                                   style={"color": "#64748b", "fontSize": "0.78rem",
                                           "margin": "0 0 0.4rem"}),
                            html.Div([alert_badge(f) for f in s.alert_flags],
                                     style={"marginBottom": "0.5rem"}),
                            dcc.Link("→ Drill-down", href=f"/site/{s.trial_site_id}",
                                    style={"color": "#6366f1", "fontSize": "0.8rem",
                                           "textDecoration": "none", "fontWeight": "500"}),
                        ], style={"background": "#1a1d2e", "border": "1px solid #2d3148",
                                   "borderRadius": "10px", "padding": "1rem 1.1rem",
                                   "height": "100%"}),
                        md=3, className="mb-3")
                        for s in summaries
                    ], className="g-3")
            finally:
                db.close()

        return html.Div([
            make_header("Portfolio Overview", "Risk heatmap across all active sites"),
            html.Div(className="page-card", children=[fig_content]),
            html.Div(site_cards, style={"marginTop": "0.5rem"}),
        ])

    def watchlist_layout(trial_id=None) -> html.Div:
        """Build watchlist page inline — no async callback needed."""
        table_content = html.P("Select a trial to view the watchlist.",
                               style={"color": "#64748b"})

        if trial_id is not None:
            db = get_db()
            try:
                from cliniq.dashboard.data import get_portfolio_risk
                summaries = get_portfolio_risk(db, trial_id)
            finally:
                db.close()

            if summaries:
                rows = []
                for s in summaries:
                    enr_pct  = f"{s.enrolment_pct*100:.0f}%" if s.enrolment_pct else "—"
                    on_track = html.Td(bi("check-lg", "#22c55e")) if s.is_on_track else (
                               html.Td(bi("x-lg",    "#ef4444")) if s.is_on_track is False
                               else html.Td("—"))
                    rows.append(html.Tr([
                        html.Td(dcc.Link(s.site_id, href=f"/site/{s.trial_site_id}",
                                        style={"color": "#6366f1", "fontWeight": "600",
                                               "textDecoration": "none",
                                               "fontSize": "0.875rem"})),
                        html.Td(s.country, style={"color": "#94a3b8", "fontSize": "0.82rem"}),
                        html.Td(dbc.Badge(f"{s.composite_score:.0f}",
                                          color=risk_badge_colour(s.composite_score))),
                        html.Td(enr_pct, style={"color": "#94a3b8", "fontSize": "0.82rem"}),
                        on_track,
                        html.Td(html.Div([alert_badge(f) for f in s.alert_flags])),
                        html.Td(dcc.Link("Deviations",
                                        href=f"/deviations/{s.trial_site_id}",
                                        style={"color": "#64748b", "fontSize": "0.78rem",
                                               "textDecoration": "none"})),
                    ], style={"borderBottom": "1px solid #2d3148"}))

                table_content = dbc.Table([
                    html.Thead(html.Tr([
                        html.Th(h) for h in
                        ["Site", "Country", "Risk", "Enrolled", "On track", "Alerts", ""]
                    ], style={"color": "#64748b", "fontSize": "0.72rem",
                               "textTransform": "uppercase", "letterSpacing": "0.06em",
                               "borderBottom": "2px solid #2d3148"})),
                    html.Tbody(rows),
                ], bordered=False, hover=True, responsive=True,
                   style={"color": "#f1f5f9", "margin": "0"})

        return html.Div([
            make_header("Watchlist", "Sites ranked by composite risk score"),
            html.Div(className="page-card", children=[table_content]),
        ])

    def site_layout(ts_id: Optional[int] = None) -> html.Div:
        """Build the site drill-down page. Charts are built here directly."""
        import plotly.graph_objects as _go
        from collections import defaultdict as _dd
        from cliniq.db.models import (
            DataEntryEvent as _DEE, PatientEnrolment as _PE,
            ProtocolDeviation as _PD, EnrolmentStatus as _ES,
        )

        _dark = dict(template="none", paper_bgcolor="#1a1d2e",
                     plot_bgcolor="#1a1d2e",
                     font={"color": "#f1f5f9",
                           "family": "Inter, system-ui, sans-serif"},
                     margin={"t": 35, "b": 40, "l": 55, "r": 20},
                     legend={"font": {"color": "#94a3b8"},
                             "bgcolor": "rgba(0,0,0,0)"})

        def _empty(msg=""):
            fig = _go.Figure()
            fig.update_layout(**_dark)
            if msg:
                fig.add_annotation(text=msg, showarrow=False,
                                   font={"color": "#64748b", "size": 13},
                                   xref="paper", yref="paper", x=0.5, y=0.5)
            return fig

        metrics_div = html.P("No site selected.",
                             style={"color": "#64748b"})
        enr_fig = lag_fig = dev_fig = risk_fig = _empty()

        if ts_id is not None:
            db = get_db()
            try:
                ts = db.get(TrialSite, ts_id)
                if ts:
                    # ── Velocity / risk / lag ───────────────────────────────
                    vel  = compute_velocity(db, ts_id)
                    risk = compute_risk_score(db, ts_id)
                    lag  = compute_lag(db, ts_id)

                    # ── Enrolment rows ──────────────────────────────────────
                    enr_rows = (
                        db.query(_PE)
                        .filter(
                            _PE.trial_site_id == ts_id,
                            _PE.enrolled_date.isnot(None),
                            _PE.status.in_([_ES.ENROLLED, _ES.COMPLETED,
                                            _ES.WITHDRAWN]),
                        )
                        .order_by(_PE.enrolled_date.asc())
                        .all()
                    )
                    enr_dates = [str(e.enrolled_date) for e in enr_rows]
                    enr_cum   = list(range(1, len(enr_dates) + 1))

                    # ── Deviation rows ──────────────────────────────────────
                    dev_rows = (
                        db.query(_PD)
                        .filter(_PD.trial_site_id == ts_id)
                        .all()
                    )
                    md = _dd(lambda: {"minor": 0, "major": 0, "critical": 0})
                    for d in dev_rows:
                        md[d.deviation_date.strftime("%Y-%m")][d.severity.value] += 1
                    months = sorted(md.keys())

                    # ── Lag rows ────────────────────────────────────────────
                    lag_raw = (
                        db.query(_DEE)
                        .filter(_DEE.trial_site_id == ts_id,
                                _DEE.lag_days.isnot(None))
                        .order_by(_DEE.visit_date.asc())
                        .all()
                    )
                    lag_dates = [str(r.visit_date) for r in lag_raw]
                    lag_vals  = [float(r.lag_days)  for r in lag_raw]

                    # ── Colours ─────────────────────────────────────────────
                    rs = risk.composite_score
                    rc = "#ef4444" if rs >= 70 else "#f59e0b" if rs >= 40 else "#22c55e"
                    lc = "#ef4444" if (lag.lag_mean or 0) > 14 else (
                         "#f59e0b" if (lag.lag_mean or 0) > 7 else "#22c55e")

                    # ── Metric cards ────────────────────────────────────────
                    metrics_div = dbc.Row([
                        dbc.Col(uniform_metric(ICON_RISK, "Composite Risk",
                            f"{rs:.0f}/100", "Risk model score", rc), md=3),
                        dbc.Col(uniform_metric(ICON_ENROLLED, "Enrolled / Target",
                            f"{min(vel.enrolled_to_date, ts.enrolment_target)} / {ts.enrolment_target}",
                            f"Velocity: {vel.velocity_28d:.3f} pts/day", "#6366f1"), md=3),
                        dbc.Col(uniform_metric(ICON_LAG, "Mean Data Lag",
                            f"{lag.lag_mean:.1f}d" if lag.lag_mean else "—",
                            f"P90: {lag.lag_p90:.1f}d" if lag.lag_p90 else "", lc), md=3),
                        dbc.Col(uniform_metric(ICON_DROPOUT, "Dropout Probability",
                            f"{risk.dropout_probability:.0%}",
                            "Logistic regression model", "#94a3b8"), md=3),
                    ], className="g-3 mb-3")

                    # ── Enrolment figure ────────────────────────────────────
                    enr_fig = _go.Figure()
                    if enr_dates:
                        enr_fig.add_trace(_go.Scatter(
                            x=enr_dates, y=enr_cum,
                            mode="lines+markers", name="Enrolled",
                            line={"color": "#6366f1", "width": 2},
                            marker={"size": 4},
                        ))
                        enr_fig.add_trace(_go.Scatter(
                            x=[enr_dates[0], enr_dates[-1]],
                            y=[ts.enrolment_target, ts.enrolment_target],
                            mode="lines",
                            name=f"Target ({ts.enrolment_target})",
                            line={"color": "#22c55e", "width": 1, "dash": "dash"},
                        ))
                    enr_fig.update_layout(**_dark,
                        title={"text": f"Enrolment — {ts.site.site_id}", "x": 0,
                               "font": {"color": "#f1f5f9"}},
                        xaxis={"gridcolor": "#2d3148", "type": "date",
                               "color": "#94a3b8"},
                        yaxis={"gridcolor": "#2d3148", "title": "Patients enrolled",
                               "color": "#94a3b8"},
                    )

                    # ── Lag figure ──────────────────────────────────────────
                    lag_fig = _go.Figure()
                    if lag_dates:
                        lag_fig.add_trace(_go.Scatter(
                            x=lag_dates, y=lag_vals,
                            mode="markers", name="Lag (days)",
                            marker={"color": "#6366f1", "size": 4, "opacity": 0.6},
                        ))
                        lag_fig.add_trace(_go.Scatter(
                            x=[lag_dates[0], lag_dates[-1]], y=[7, 7],
                            mode="lines", name="7-day target",
                            line={"color": "#22c55e", "width": 1, "dash": "dot"},
                        ))
                    lag_fig.update_layout(**_dark,
                        title={"text": "Data Lag (days)", "x": 0,
                               "font": {"color": "#f1f5f9"}},
                        xaxis={"gridcolor": "#2d3148", "type": "date",
                               "color": "#94a3b8"},
                        yaxis={"gridcolor": "#2d3148", "title": "Days",
                               "color": "#94a3b8"},
                    )

                    # ── Deviation figure ────────────────────────────────────
                    dev_fig = _go.Figure()
                    for label, key, colour in [
                        ("Minor", "minor", "#22c55e"),
                        ("Major", "major", "#f59e0b"),
                        ("Critical", "critical", "#ef4444"),
                    ]:
                        dev_fig.add_trace(_go.Bar(
                            x=months,
                            y=[md[m][key] for m in months],
                            name=label, marker_color=colour,
                        ))
                    dev_fig.update_layout(**_dark,
                        title={"text": "Deviation Timeline", "x": 0,
                               "font": {"color": "#f1f5f9"}},
                        barmode="stack",
                        xaxis={"gridcolor": "#2d3148", "color": "#94a3b8"},
                        yaxis={"gridcolor": "#2d3148", "color": "#94a3b8"},
                    )

                    # ── Risk breakdown figure ───────────────────────────────
                    risk_fig = risk_breakdown(
                        risk.enrolment_component, risk.deviation_component,
                        risk.data_lag_component, risk.dropout_component,
                        risk.monitoring_component, WEIGHTS,
                    )

            finally:
                db.close()

        # Breadcrumb
        site_label = f"{ts.site.site_id} — {ts.site.name}" if ts_id and ts else "Site"
        breadcrumb = html.Div([
            dcc.Link("← Watchlist", href="/watchlist",
                     style={"color": "#6366f1", "fontSize": "0.82rem",
                            "textDecoration": "none", "fontWeight": "500"}),
            html.Span(" · ", style={"color": "#2d3148", "margin": "0 0.3rem"}),
            html.Span(site_label, style={"color": "#64748b", "fontSize": "0.82rem"}),
        ], style={"marginBottom": "1rem"})

        return html.Div([
            make_header("Site Drill-Down",
                        "Enrolment, lag, deviations, risk breakdown"),
            breadcrumb,
            metrics_div,
            html.Div(style={"marginTop": "1rem"}, children=[
                dbc.Row([
                    dbc.Col(html.Div(className="page-card", children=[
                        dcc.Graph(figure=enr_fig,
                                  config={"displayModeBar": False})
                    ]), md=6),
                    dbc.Col(html.Div(className="page-card", children=[
                        dcc.Graph(figure=lag_fig,
                                  config={"displayModeBar": False})
                    ]), md=6),
                ], className="g-3"),
                dbc.Row([
                    dbc.Col(html.Div(className="page-card", children=[
                        dcc.Graph(figure=dev_fig,
                                  config={"displayModeBar": False}),
                        html.Div(
                            dcc.Link(
                                [bi("list-ul"), " View full deviation log"],
                                href=f"/deviations/{ts_id}" if ts_id else "/",
                                style={"color": "#6366f1", "fontSize": "0.8rem",
                                       "textDecoration": "none", "fontWeight": "500",
                                       "display": "inline-flex", "alignItems": "center",
                                       "gap": "0.35rem", "marginTop": "0.5rem"},
                            ),
                            style={"textAlign": "right", "paddingTop": "0.25rem"},
                        ),
                    ]), md=6),
                    dbc.Col(html.Div(className="page-card", children=[
                        dcc.Graph(figure=risk_fig,
                                  config={"displayModeBar": False})
                    ]), md=6),
                ], className="g-3 mt-0"),
            ]),
        ])

    def deviation_log_layout(ts_id: Optional[int] = None) -> html.Div:
        return html.Div([
            make_header("Deviation Log",
                        "Filterable deviation log with NLP category tags"),
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id="dev-severity-filter",
                    options=[
                        {"label": "All severities", "value": "all"},
                        {"label": "Minor",    "value": "minor"},
                        {"label": "Major",    "value": "major"},
                        {"label": "Critical", "value": "critical"},
                    ],
                    value="all", clearable=False,
                    className="dark-dropdown",
                ), md=3),
                dbc.Col(dcc.Dropdown(
                    id="dev-category-filter",
                    options=[{"label": "All categories", "value": "all"}] + [
                        {"label": c.title(), "value": c}
                        for c in ["consent", "dosing", "eligibility",
                                  "documentation", "safety"]
                    ],
                    value="all", clearable=False,
                    className="dark-dropdown",
                ), md=3),
            ], className="mb-3 g-2"),
            html.Div(className="page-card", children=[
                dcc.Loading(html.Div(id="deviation-log-table"), color="#6366f1"),
            ]),
        ])

    def assistant_layout() -> html.Div:
        example_questions = [
            "Which site is at risk of falling behind?",
            "Where are deviations most concentrated?",
            "Which sites have stale monitoring?",
            "Compare enrolment velocity across sites",
            "What are the top data quality concerns?",
        ]
        return html.Div([
            make_header("AI Assistant",
                        "Ask questions about your trial portfolio"),
            # Token notice
            html.Div([
                html.Div([
                    bi(ICON_RISK, "#f59e0b"),
                    html.Span("API key required",
                              style={"fontWeight": "600", "color": "#f59e0b",
                                     "marginLeft": "0.5rem", "fontSize": "0.85rem"}),
                ], style={"display": "flex", "alignItems": "center",
                           "marginBottom": "0.4rem"}),
                html.P(
                    "Set the ANTHROPIC_API_KEY environment variable to enable the AI assistant. "
                    "Restart the server after setting it.",
                    style={"color": "#94a3b8", "fontSize": "0.82rem",
                            "margin": "0"},
                ),
            ], style={
                "background": "#1e1a0e", "border": "1px solid #3d2f0a",
                "borderRadius": "10px", "padding": "0.9rem 1.1rem",
                "marginBottom": "1.25rem",
            }) if not os.getenv("ANTHROPIC_API_KEY") else None,

            # Trial selector
            dcc.Dropdown(
                id="assistant-trial-select",
                placeholder="Select trial for context…",
                className="dark-dropdown",
                style={"marginBottom": "1rem"},
            ),

            # Example questions
            html.Div([
                html.P("Try asking:",
                       style={"color": "#64748b", "fontSize": "0.75rem",
                               "marginBottom": "0.4rem", "fontWeight": "600",
                               "textTransform": "uppercase",
                               "letterSpacing": "0.06em"}),
                html.Div([
                    html.Span(q, id={"type": "example-q", "index": i},
                              className="example-question",
                              n_clicks=0)
                    for i, q in enumerate(example_questions)
                ]),
            ], style={"marginBottom": "1.25rem"}),

            # Chat history
            html.Div(
                id="assistant-chat-history",
                style={
                    "height": "380px", "overflowY": "auto",
                    "backgroundColor": "#1a1d2e",
                    "border": "1px solid #2d3148",
                    "borderRadius": "10px", "padding": "1rem",
                    "marginBottom": "1rem",
                    "display": "flex", "flexDirection": "column",
                }
            ),

            # Input
            dbc.InputGroup([
                dbc.Input(
                    id="assistant-input",
                    placeholder="Ask about enrolment velocity, site risk, deviations…",
                    type="text",
                    style={
                        "backgroundColor": "#242638",
                        "color": "#f1f5f9",
                        "border": "1px solid #2d3148",
                        "borderRadius": "8px 0 0 8px",
                        "fontSize": "0.875rem",
                    },
                ),
                dbc.Button(
                    [bi("send-fill"), " Ask"],
                    id="assistant-submit",
                    color="primary",
                    style={"borderRadius": "0 8px 8px 0", "fontWeight": "600"},
                ),
            ]),
            dcc.Store(id="assistant-history", data=[]),
        ])

    # -----------------------------------------------------------------------
    # Root layout
    # -----------------------------------------------------------------------

    mobile_topbar = html.Div([
        html.Div([
            html.Span("Clin", style={"color": "#fff"}),
            html.Span("IQ",   style={"color": "#6366f1"}),
        ], className="mobile-logo"),
        html.Button("☰", id="mobile-menu-btn", className="mobile-hamburger",
                    n_clicks=0),
    ], className="mobile-topbar")

    mobile_nav = html.Div([
        html.A([html.I(className="bi bi-grid-3x3-gap-fill"), " Portfolio"],
               href="/", className="mobile-nav-link"),
        html.A([html.I(className="bi bi-exclamation-triangle-fill"), " Watchlist"],
               href="/watchlist", className="mobile-nav-link"),
        html.A([html.I(className="bi bi-chat-dots-fill"), " AI Assistant"],
               href="/assistant", className="mobile-nav-link"),
    ], className="mobile-nav-drawer", id="mobile-nav-drawer")

    app.layout = html.Div([
        dcc.Location(id="url", refresh=False),
        dcc.Interval(id="clock-interval", interval=60000, n_intervals=0),
        dcc.Store(id="trial-options-store", data=[]),
        mobile_topbar,
        mobile_nav,
        html.Div(id="sidebar-container"),
        html.Div(
            id="page-content",
            className="main-content",
            style={
                "marginLeft": "240px",
                "padding": "2rem 2.5rem 4.5rem",
                "backgroundColor": "#0f1117",
                "minHeight": "100vh",
            },
        ),
        make_footer(),
    ])

    # -----------------------------------------------------------------------
    # Clock callback
    # -----------------------------------------------------------------------

    @app.callback(
        Output("live-clock", "children"),
        Input("clock-interval", "n_intervals"),
    )
    def update_clock(_):
        from datetime import datetime
        now = datetime.now()
        return [
            html.Div(now.strftime("%H:%M"),
                     style={"fontWeight": "600", "fontSize": "1rem",
                             "color": "#f1f5f9"}),
            html.Div(now.strftime("%A, %d %B %Y"),
                     style={"color": "#64748b", "fontSize": "0.75rem"}),
        ]

    # -----------------------------------------------------------------------
    # URL router
    # -----------------------------------------------------------------------

    @app.callback(
        Output("sidebar-container", "children"),
        Output("page-content", "children"),
        Output("trial-options-store", "data"),
        Input("url", "pathname"),
    )
    def route_page(pathname: str):
        pathname = pathname or "/"
        db = get_db()
        try:
            trials = db.query(Trial).all()
            trial_options = [{"label": t.title[:60], "value": t.id} for t in trials]
            auto_trial_id = trial_options[0]["value"] if trial_options else None
        finally:
            db.close()

        sb = make_sidebar(pathname)

        if pathname in ("/", "/portfolio"):
            layout = portfolio_layout(trial_id=auto_trial_id)
        elif pathname == "/watchlist":
            layout = watchlist_layout(trial_id=auto_trial_id)
        elif pathname == "/assistant":
            layout = assistant_layout()
        elif pathname.startswith("/site/"):
            try:
                ts_id = int(pathname.split("/")[-1])
            except ValueError:
                ts_id = None
            layout = site_layout(ts_id)
        elif pathname.startswith("/deviations/"):
            try:
                ts_id = int(pathname.split("/")[-1])
            except ValueError:
                ts_id = None
            layout = deviation_log_layout(ts_id)
        else:
            layout = html.Div([
                make_header("Page not found"),
                html.P("The page you requested does not exist.",
                       style={"color": "#64748b"}),
            ])

        return sb, layout, trial_options

    # -----------------------------------------------------------------------
    # Dropdown population
    # -----------------------------------------------------------------------

    # Assistant dropdown only — portfolio and watchlist no longer use dropdowns
    @app.callback(
        Output("assistant-trial-select", "options"),
        Input("trial-options-store", "data"),
    )
    def populate_assistant_dropdown(options):
        return options or []

    # -----------------------------------------------------------------------
    # Portfolio heatmap
    # -----------------------------------------------------------------------


    # Portfolio content built inline in route_page — no separate callback needed

    # -----------------------------------------------------------------------
    # Watchlist table
    # -----------------------------------------------------------------------


    # Watchlist content built inline in route_page — no separate callback needed

    # (site drill-down charts now built inline in site_layout)


    # -----------------------------------------------------------------------
    # Deviation log
    # -----------------------------------------------------------------------

    @app.callback(
        Output("deviation-log-table", "children"),
        Input("url", "pathname"),
        Input("dev-severity-filter", "value"),
        Input("dev-category-filter", "value"),
    )
    def update_deviation_log(pathname, severity_filter, category_filter):
        ts_id = None
        if pathname and pathname.startswith("/deviations/"):
            try:
                ts_id = int(pathname.split("/")[-1])
            except ValueError:
                ts_id = None
        if ts_id is None:
            return html.P("No site selected.", style={"color": "#64748b"})
        db = get_db()
        try:
            q = db.query(ProtocolDeviation).filter(
                ProtocolDeviation.trial_site_id == ts_id
            )
            if severity_filter and severity_filter != "all":
                sev = DeviationSeverity(severity_filter)
                q = q.filter(ProtocolDeviation.severity == sev)
            if category_filter and category_filter != "all":
                q = q.filter(ProtocolDeviation.category == category_filter)
            devs = (q.order_by(ProtocolDeviation.deviation_date.desc())
                     .limit(100).all())
        finally:
            db.close()

        if not devs:
            return html.P("No deviations match the filter.",
                          style={"color": "#64748b"})

        sev_colour = {"minor": "#22c55e", "major": "#f59e0b",
                      "critical": "#ef4444"}
        rows = [
            html.Tr([
                html.Td(str(d.deviation_date),
                        style={"color": "#94a3b8", "fontSize": "0.8rem",
                               "whiteSpace": "nowrap"}),
                html.Td(
                    html.Span(d.severity.value,
                              style={
                                  "background": sev_colour.get(d.severity.value, "#64748b") + "22",
                                  "color": sev_colour.get(d.severity.value, "#64748b"),
                                  "padding": "2px 8px",
                                  "borderRadius": "4px",
                                  "fontSize": "0.75rem",
                                  "fontWeight": "600",
                                  "textTransform": "uppercase",
                              }),
                ),
                html.Td(
                    html.Span(d.category or "—",
                              style={
                                  "background": "#242638",
                                  "color": "#94a3b8",
                                  "padding": "2px 8px",
                                  "borderRadius": "4px",
                                  "fontSize": "0.75rem",
                              }) if d.category else "—",
                ),
                html.Td(
                    (d.free_text[:80] + "…")
                    if d.free_text and len(d.free_text) > 80
                    else (d.free_text or "—"),
                    style={"color": "#94a3b8", "fontSize": "0.82rem",
                           "maxWidth": "400px"},
                ),
                html.Td(
                    html.Div(
                        bi("check-lg", "#22c55e")
                        if d.is_resolved
                        else html.Span("Open", style={
                            "color": "#f59e0b", "fontSize": "0.75rem",
                            "fontWeight": "600",
                        })
                    )
                ),
            ], style={"borderBottom": "1px solid #1e2235"})
            for d in devs
        ]

        return dbc.Table(
            [
                html.Thead(html.Tr([
                    html.Th(h) for h in
                    ["Date", "Severity", "Category", "Description", "Status"]
                ], style={"color": "#64748b", "fontSize": "0.72rem",
                           "textTransform": "uppercase",
                           "letterSpacing": "0.06em",
                           "borderBottom": "2px solid #2d3148"})),
                html.Tbody(rows),
            ],
            bordered=False, hover=True, responsive=True,
            style={"color": "#f1f5f9", "margin": "0"},
        )

    # -----------------------------------------------------------------------
    # AI assistant — example questions fill input
    # -----------------------------------------------------------------------

    @app.callback(
        Output("assistant-input", "value"),
        Input({"type": "example-q", "index": 0}, "n_clicks"),
        Input({"type": "example-q", "index": 1}, "n_clicks"),
        Input({"type": "example-q", "index": 2}, "n_clicks"),
        Input({"type": "example-q", "index": 3}, "n_clicks"),
        Input({"type": "example-q", "index": 4}, "n_clicks"),
        prevent_initial_call=True,
    )
    def fill_example_question(*_):
        from dash import ctx
        if not ctx.triggered:
            return no_update
        questions = [
            "Which site is at risk of falling behind?",
            "Where are deviations most concentrated?",
            "Which sites have stale monitoring?",
            "Compare enrolment velocity across sites",
            "What are the top data quality concerns?",
        ]
        triggered_id = ctx.triggered_id
        if isinstance(triggered_id, dict):
            idx = triggered_id.get("index", 0)
            return questions[idx]
        return no_update

    # -----------------------------------------------------------------------
    # AI assistant — send query
    # -----------------------------------------------------------------------

    @app.callback(
        Output("assistant-chat-history", "children"),
        Output("assistant-history", "data"),
        Output("assistant-input", "value", allow_duplicate=True),
        Input("assistant-submit", "n_clicks"),
        Input("assistant-input", "n_submit"),
        State("assistant-input", "value"),
        State("assistant-trial-select", "value"),
        State("assistant-history", "data"),
        prevent_initial_call=True,
    )
    def handle_assistant_query(n_clicks, n_submit, question, trial_id, history):
        if not question or not question.strip():
            return no_update, no_update, no_update

        from cliniq.ai.assistant import build_context, query_assistant

        context = ""
        if trial_id:
            db = get_db()
            try:
                context = build_context(db, trial_id)
            finally:
                db.close()

        answer = query_assistant(question.strip(), context, history)

        history = history or []
        history.append({"role": "user",    "content": question.strip()})
        history.append({"role": "assistant", "content": answer})

        bubbles = []
        for msg in history:
            is_user = msg["role"] == "user"
            bubbles.append(
                html.Div(
                    msg["content"],
                    className="chat-bubble-user" if is_user else "chat-bubble-ai",
                )
            )

        return bubbles, history, ""

    # Mobile hamburger toggle (clientside — no server round-trip)
    app.clientside_callback(
        """
        function(n) {
            var d = document.getElementById('mobile-nav-drawer');
            if (d) { d.className = n % 2 === 1
                ? 'mobile-nav-drawer open'
                : 'mobile-nav-drawer'; }
            return window.dash_clientside.no_update;
        }
        """,
        Output("mobile-nav-drawer", "className"),
        Input("mobile-menu-btn", "n_clicks"),
        prevent_initial_call=True,
    )

    return app


if __name__ == "__main__":
    app = create_dashboard()
    app.run(debug=False, port=8050)
