"""
ClinIQ API — Portfolio watchlist
GET /portfolio/watchlist  Sites ranked by composite risk score with alert flags
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from cliniq.api.schemas import WatchlistEntry, WatchlistResponse
from cliniq.db.database import get_db
from cliniq.db.models import EnrolmentForecast, RiskScore, SiteKPISnapshot, Trial, TrialSite
from cliniq.ml.risk_model import compute_risk_score
from cliniq.rbac.auth import TokenData, get_current_user, log_action, require_trial_access

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

# Alert thresholds
RISK_HIGH_THRESHOLD      = 70.0
LAG_ALERT_DAYS           = 14
DROPOUT_PROB_ALERT       = 0.65
ENROLMENT_BEHIND_DAYS    = -30   # days_ahead_behind below this = alert


@router.get("/watchlist", response_model=WatchlistResponse)
def get_watchlist(
    request: Request,
    trial_id: Optional[int] = Query(None, description="Filter to a specific trial"),
    db: Session = Depends(get_db),
    token: TokenData = Depends(get_current_user),
):
    """
    Ranked list of sites by composite risk score (highest first).
    Includes alert flags for critical conditions.
    """
    today = date.today()

    # Resolve trial filter
    if token.role.value == "sponsor_view":
        if token.trial_id is None:
            return WatchlistResponse(
                trial_id="N/A", as_of_date=today, entries=[]
            )
        trial = db.get(Trial, token.trial_id)
        trial_sites = trial.trial_sites if trial else []
    elif trial_id is not None:
        trial = db.get(Trial, trial_id)
        trial_sites = trial.trial_sites if trial else []
    else:
        # All active sites across all trials
        trial_sites = db.query(TrialSite).filter(TrialSite.is_active.is_(True)).all()

    entries: list[WatchlistEntry] = []

    for ts in trial_sites:
        # Risk score — try stored, else compute live
        stored_risk = (
            db.query(RiskScore)
            .filter_by(trial_site_id=ts.id, score_date=today)
            .first()
        )
        if stored_risk:
            composite   = stored_risk.composite_score
            dropout_prob = stored_risk.dropout_probability or 0.0
        else:
            risk = compute_risk_score(db, ts.id, today)
            composite    = risk.composite_score
            dropout_prob = risk.dropout_probability

        # Latest KPI snapshot for enrolment pct and lag
        snap = (
            db.query(SiteKPISnapshot)
            .filter_by(trial_site_id=ts.id)
            .order_by(SiteKPISnapshot.snapshot_date.desc())
            .first()
        )
        enrolment_pct   = snap.enrolment_pct if snap else None
        lag_mean        = snap.data_lag_mean if snap else None

        # Latest forecast for on-track
        forecast = (
            db.query(EnrolmentForecast)
            .filter_by(trial_site_id=ts.id)
            .order_by(EnrolmentForecast.forecast_date.desc())
            .first()
        )
        is_on_track          = forecast.is_on_track if forecast else None
        days_ahead_behind    = forecast.days_ahead_behind if forecast else None

        # Alert flags
        flags: list[str] = []
        if composite >= RISK_HIGH_THRESHOLD:
            flags.append("HIGH_RISK")
        if dropout_prob >= DROPOUT_PROB_ALERT:
            flags.append("DROPOUT_RISK")
        if lag_mean and lag_mean >= LAG_ALERT_DAYS:
            flags.append("DATA_LAG")
        if days_ahead_behind is not None and days_ahead_behind <= ENROLMENT_BEHIND_DAYS:
            flags.append("BEHIND_SCHEDULE")
        if is_on_track is False:
            flags.append("OFF_TRACK")

        entries.append(WatchlistEntry(
            trial_site_id=ts.id,
            site_id=ts.site.site_id,
            site_name=ts.site.name,
            country=ts.site.country,
            composite_score=composite,
            dropout_probability=dropout_prob,
            enrolment_pct=enrolment_pct,
            is_on_track=is_on_track,
            alert_flags=flags,
        ))

    entries.sort(key=lambda e: e.composite_score, reverse=True)

    trial_label = trial_id or (token.trial_id if token.role.value == "sponsor_view" else "all")
    log_action(db, token, "READ_WATCHLIST", "/portfolio/watchlist", request)
    db.commit()

    return WatchlistResponse(
        trial_id=str(trial_label),
        as_of_date=today,
        entries=entries,
    )
