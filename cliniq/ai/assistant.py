"""
ClinIQ AI — Context builder and query handler for the Claude assistant.
Context is limited to aggregated KPIs and site-level summaries.
No patient-level records are ever passed to the Claude API.
"""
from __future__ import annotations

import os
from datetime import date
from typing import Optional

from sqlalchemy.orm import Session

from cliniq.dashboard.data import get_portfolio_risk, get_site_kpi_timeseries
from cliniq.db.models import Trial, TrialSite


def build_context(
    db: Session,
    trial_id: int,
    trial_site_id: Optional[int] = None,
) -> str:
    """
    Build a plain-text context block summarising trial KPIs.
    Deliberately limited to aggregated metrics — no patient identifiers.
    """
    trial: Trial = db.get(Trial, trial_id)
    if trial is None:
        return "No trial data available."

    lines = [
        f"TRIAL: {trial.title}",
        f"Sponsor: {trial.sponsor} | Phase: {trial.phase.value} | Status: {trial.status.value}",
        f"EudraCT: {trial.eudract_number or 'N/A'} | ISRCTN: {trial.isrctn_number or 'N/A'}",
        f"Start: {trial.start_date} | Planned end: {trial.planned_end_date}",
        "",
        "SITE RISK SUMMARY (as of today):",
    ]

    summaries = get_portfolio_risk(db, trial_id)
    for s in summaries:
        enr_pct = f"{s.enrolment_pct*100:.0f}%" if s.enrolment_pct is not None else "N/A"
        flags   = ", ".join(s.alert_flags) if s.alert_flags else "none"
        lines.append(
            f"  {s.site_id} ({s.country}): risk={s.composite_score:.0f}/100  "
            f"enrolled={enr_pct}  vel={s.velocity_28d:.3f}pts/day  alerts=[{flags}]"
        )

    if trial_site_id is not None:
        ts: TrialSite = db.get(TrialSite, trial_site_id)
        if ts:
            lines += ["", f"FOCUSED SITE: {ts.site.name} ({ts.site.site_id})"]
            snaps = get_site_kpi_timeseries(db, trial_site_id, days=30)
            if snaps:
                latest = snaps[-1]
                lines += [
                    f"  Latest KPI snapshot ({latest.snapshot_date}):",
                    f"    Enrolment rate 28d: {latest.enrolment_rate_28d:.3f} pts/day",
                    f"    Data lag (mean/p90): {latest.data_lag_mean}/{latest.data_lag_p90} days",
                    f"    Open queries: {latest.open_queries}",
                    f"    Days since monitoring: {latest.days_since_monitoring}",
                ]

    return "\n".join(lines)


def query_assistant(
    question: str,
    context: str,
    conversation_history: Optional[list[dict]] = None,
) -> str:
    """
    Send a clinical context + user question to Claude.
    Returns the assistant's plain-text response.
    Falls back to a stub if the API key is not set.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return (
            "[AI assistant unavailable — ANTHROPIC_API_KEY not set]\n\n"
            "To enable: set the ANTHROPIC_API_KEY environment variable and restart."
        )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        system_prompt = (
            "You are ClinIQ, a clinical trial operations assistant. "
            "You help CRO staff interpret site performance data, identify risks, "
            "and take action. You are concise, evidence-based, and avoid speculation. "
            "Never reference individual patient data — work only with aggregated KPIs. "
            "Always ground your response in the context data provided.\n\n"
            f"CURRENT TRIAL CONTEXT:\n{context}"
        )

        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": question})

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text

    except Exception as e:
        return f"[AI assistant error: {e}]"
