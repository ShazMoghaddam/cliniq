"""
ClinIQ Analytics — Screening Funnel Calculator
Screen failure rate, consent rate, enrolment conversion by site and country.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from cliniq.db.models import EnrolmentStatus, PatientEnrolment, Site, Trial, TrialSite


@dataclass
class FunnelResult:
    trial_site_id: int
    site_id: str
    country: str
    screened: int
    screen_fails: int
    enrolled: int
    withdrawn: int
    completed: int
    screen_fail_rate: float   # screen_fails / screened
    enrolment_conversion: float  # enrolled / screened
    withdrawal_rate: float    # withdrawn / enrolled (0 if none enrolled)


def compute_funnel(
    session: Session,
    trial_site_id: int,
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
) -> FunnelResult:
    """Screening funnel breakdown for a single trial-site."""
    ts: TrialSite = session.get(TrialSite, trial_site_id)
    if ts is None:
        raise ValueError(f"TrialSite {trial_site_id} not found")

    q = session.query(
        PatientEnrolment.status,
        func.count(PatientEnrolment.id).label("n"),
    ).filter(PatientEnrolment.trial_site_id == trial_site_id)

    if from_date:
        q = q.filter(PatientEnrolment.screened_date >= from_date)
    if to_date:
        q = q.filter(PatientEnrolment.screened_date <= to_date)

    rows = q.group_by(PatientEnrolment.status).all()
    counts = {r.status: r.n for r in rows}

    screened    = sum(counts.values())
    screen_fail = counts.get(EnrolmentStatus.SCREEN_FAIL, 0)
    enrolled    = counts.get(EnrolmentStatus.ENROLLED, 0)
    withdrawn   = counts.get(EnrolmentStatus.WITHDRAWN, 0)
    completed   = counts.get(EnrolmentStatus.COMPLETED, 0)

    # screened = 0 edge case
    sfr  = screen_fail / screened if screened else 0.0
    conv = enrolled / screened if screened else 0.0
    wdr  = withdrawn / enrolled if enrolled else 0.0

    return FunnelResult(
        trial_site_id=trial_site_id,
        site_id=ts.site.site_id,
        country=ts.site.country,
        screened=screened,
        screen_fails=screen_fail,
        enrolled=enrolled,
        withdrawn=withdrawn,
        completed=completed,
        screen_fail_rate=round(sfr, 4),
        enrolment_conversion=round(conv, 4),
        withdrawal_rate=round(wdr, 4),
    )


def compute_funnel_by_country(
    session: Session,
    trial_id: int,
) -> dict[str, dict]:
    """Aggregate funnel metrics grouped by country for a trial."""
    trial_sites = (
        session.query(TrialSite)
        .filter(TrialSite.trial_id == trial_id, TrialSite.is_active.is_(True))
        .all()
    )
    results: dict[str, dict] = {}
    for ts in trial_sites:
        f = compute_funnel(session, ts.id)
        country = f.country
        if country not in results:
            results[country] = {
                "screened": 0, "screen_fails": 0,
                "enrolled": 0, "withdrawn": 0, "completed": 0,
            }
        results[country]["screened"]     += f.screened
        results[country]["screen_fails"] += f.screen_fails
        results[country]["enrolled"]     += f.enrolled
        results[country]["withdrawn"]    += f.withdrawn
        results[country]["completed"]    += f.completed

    # Compute rates for each country aggregate
    for country, d in results.items():
        s = d["screened"]
        e = d["enrolled"]
        d["screen_fail_rate"]      = round(d["screen_fails"] / s, 4) if s else 0.0
        d["enrolment_conversion"]  = round(e / s, 4) if s else 0.0
        d["withdrawal_rate"]       = round(d["withdrawn"] / e, 4) if e else 0.0

    return results
