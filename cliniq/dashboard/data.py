"""
ClinIQ Dashboard — Data layer
get_portfolio_risk: ranked site risk summaries with alert flags and caching.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

from sqlalchemy.orm import Session

from cliniq.analytics.velocity import compute_velocity
from cliniq.dashboard.cache import cache_get, cache_set
from cliniq.db.models import RiskScore, SiteKPISnapshot, TrialSite
from cliniq.ml.risk_model import compute_risk_score


@dataclass
class SiteRiskSummary:
    trial_site_id: int
    site_id: str
    site_name: str
    country: str
    composite_score: float
    dropout_probability: float
    enrolment_pct: Optional[float]
    velocity_28d: Optional[float]
    is_on_track: Optional[bool]
    alert_flags: list[str]


def get_portfolio_risk(db: Session, trial_id: int,
                       as_of: Optional[date] = None) -> list[SiteRiskSummary]:
    """Ranked list of site risk summaries for a trial, with caching."""
    today = as_of or date.today()
    cache_key = f"portfolio_risk:{trial_id}:{today}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    trial_sites = (
        db.query(TrialSite)
        .filter(TrialSite.trial_id == trial_id, TrialSite.is_active.is_(True))
        .all()
    )

    summaries: list[SiteRiskSummary] = []
    for ts in trial_sites:
        risk_row = (
            db.query(RiskScore)
            .filter_by(trial_site_id=ts.id, score_date=today)
            .first()
        )
        if risk_row:
            composite = risk_row.composite_score
            dropout   = risk_row.dropout_probability or 0.0
        else:
            r = compute_risk_score(db, ts.id, today)
            composite = r.composite_score
            dropout   = r.dropout_probability

        snap = (
            db.query(SiteKPISnapshot)
            .filter_by(trial_site_id=ts.id)
            .order_by(SiteKPISnapshot.snapshot_date.desc())
            .first()
        )
        enrolment_pct = snap.enrolment_pct if snap else None
        velocity_result = compute_velocity(db, ts.id, today)
        velocity_28d    = velocity_result.velocity_28d
        is_on_track     = velocity_result.is_on_track

        flags = []
        if composite >= 70:   flags.append("HIGH_RISK")
        if dropout >= 0.65:   flags.append("DROPOUT_RISK")
        if snap and snap.data_lag_mean and snap.data_lag_mean >= 14:
            flags.append("DATA_LAG")
        if not is_on_track:   flags.append("OFF_TRACK")

        summaries.append(SiteRiskSummary(
            trial_site_id=ts.id,
            site_id=ts.site.site_id,
            site_name=ts.site.name,
            country=ts.site.country,
            composite_score=composite,
            dropout_probability=dropout,
            enrolment_pct=enrolment_pct,
            velocity_28d=velocity_28d,
            is_on_track=is_on_track,
            alert_flags=flags,
        ))

    summaries.sort(key=lambda s: s.composite_score, reverse=True)
    cache_set(cache_key, summaries)
    return summaries
