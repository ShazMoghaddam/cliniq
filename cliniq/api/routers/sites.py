"""
ClinIQ API — Site endpoints
GET /sites/{ts_id}/kpis        Time-series KPI snapshots with date range filter
GET /sites/{ts_id}/risk        Current risk score with component breakdown
GET /sites/{ts_id}/forecast    Enrolment projection and projected completion date
GET /sites/{ts_id}/deviations  Paginated deviation log with NLP category tags
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from cliniq.api.schemas import (
    DeviationListResponse, DeviationSchema, ForecastSchema,
    KPIResponse, KPISnapshotSchema, PaginationMeta, RiskScoreSchema,
)
from cliniq.analytics.velocity import compute_velocity
from cliniq.analytics.snapshot import write_snapshot_for_site
from cliniq.db.database import get_db
from cliniq.db.models import (
    EnrolmentForecast, ProtocolDeviation, RiskScore,
    SiteKPISnapshot, TrialSite,
)
from cliniq.ml.risk_model import compute_risk_score
from cliniq.rbac.auth import TokenData, get_current_user, log_action, require_trial_access

router = APIRouter(prefix="/sites", tags=["sites"])


def _get_trial_site(db: Session, ts_id: int) -> TrialSite:
    ts = db.get(TrialSite, ts_id)
    if ts is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"TrialSite {ts_id} not found")
    return ts


@router.get("/{ts_id}/kpis", response_model=KPIResponse)
def get_site_kpis(
    ts_id: int,
    request: Request,
    from_date: Optional[date] = Query(None, description="Start of date range (inclusive)"),
    to_date: Optional[date] = Query(None, description="End of date range (inclusive)"),
    db: Session = Depends(get_db),
    token: TokenData = Depends(get_current_user),
):
    """Time-series KPI snapshots for a site. Generates today's snapshot on demand."""
    ts = _get_trial_site(db, ts_id)
    require_trial_access(ts.trial_id, token)

    # Generate today's snapshot on demand if not yet written
    try:
        write_snapshot_for_site(db, ts_id, date.today())
        db.commit()
    except Exception:
        db.rollback()

    q = db.query(SiteKPISnapshot).filter(SiteKPISnapshot.trial_site_id == ts_id)
    if from_date:
        q = q.filter(SiteKPISnapshot.snapshot_date >= from_date)
    if to_date:
        q = q.filter(SiteKPISnapshot.snapshot_date <= to_date)
    snapshots = q.order_by(SiteKPISnapshot.snapshot_date.asc()).all()

    log_action(db, token, "READ_SITE_KPIS", f"/sites/{ts_id}/kpis", request)
    db.commit()

    return KPIResponse(
        trial_site_id=ts_id,
        site_id=ts.site.site_id,
        trial_id=ts.trial.trial_id,
        snapshots=[KPISnapshotSchema.model_validate(s) for s in snapshots],
    )


@router.get("/{ts_id}/risk", response_model=RiskScoreSchema)
def get_site_risk(
    ts_id: int,
    request: Request,
    as_of: Optional[date] = Query(None, description="Score date (defaults to today)"),
    db: Session = Depends(get_db),
    token: TokenData = Depends(get_current_user),
):
    """Current composite risk score with component breakdown."""
    ts = _get_trial_site(db, ts_id)
    require_trial_access(ts.trial_id, token)

    target_date = as_of or date.today()

    # Try stored score first
    stored = (
        db.query(RiskScore)
        .filter_by(trial_site_id=ts_id, score_date=target_date)
        .first()
    )

    if stored:
        result_data = {
            "composite_score":      stored.composite_score,
            "dropout_probability":  stored.dropout_probability or 0.0,
            "enrolment_component":  stored.enrolment_component or 0.0,
            "deviation_component":  stored.deviation_component or 0.0,
            "data_lag_component":   stored.data_lag_component or 0.0,
            "dropout_component":    (stored.dropout_probability or 0.0) * 100,
            "monitoring_component": stored.monitoring_component or 0.0,
        }
    else:
        # Compute live
        result = compute_risk_score(db, ts_id, target_date)
        result_data = {
            "composite_score":      result.composite_score,
            "dropout_probability":  result.dropout_probability,
            "enrolment_component":  result.enrolment_component,
            "deviation_component":  result.deviation_component,
            "data_lag_component":   result.data_lag_component,
            "dropout_component":    result.dropout_component,
            "monitoring_component": result.monitoring_component,
        }

    log_action(db, token, "READ_SITE_RISK", f"/sites/{ts_id}/risk", request)
    db.commit()

    return RiskScoreSchema(
        trial_site_id=ts_id,
        site_id=ts.site.site_id,
        as_of_date=target_date,
        **result_data,
    )


@router.get("/{ts_id}/forecast", response_model=ForecastSchema)
def get_site_forecast(
    ts_id: int,
    request: Request,
    db: Session = Depends(get_db),
    token: TokenData = Depends(get_current_user),
):
    """Live enrolment forecast — projected completion date and velocity."""
    ts = _get_trial_site(db, ts_id)
    require_trial_access(ts.trial_id, token)

    velocity = compute_velocity(db, ts_id)

    log_action(db, token, "READ_SITE_FORECAST", f"/sites/{ts_id}/forecast", request)
    db.commit()

    return ForecastSchema(
        trial_site_id=ts_id,
        site_id=ts.site.site_id,
        forecast_date=velocity.as_of_date,
        projected_completion=velocity.projected_completion,
        velocity_28d=velocity.velocity_28d,
        enrolled_to_date=velocity.enrolled_to_date,
        remaining_to_target=velocity.remaining_to_target,
        is_on_track=velocity.is_on_track,
        days_ahead_behind=velocity.days_ahead_behind,
    )


@router.get("/{ts_id}/deviations", response_model=DeviationListResponse)
def get_site_deviations(
    ts_id: int,
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    severity: Optional[str] = Query(None, description="Filter by severity: minor|major|critical"),
    category: Optional[str] = Query(None, description="Filter by NLP category"),
    db: Session = Depends(get_db),
    token: TokenData = Depends(get_current_user),
):
    """Paginated deviation log with NLP category tags and severity colour coding."""
    ts = _get_trial_site(db, ts_id)
    require_trial_access(ts.trial_id, token)

    q = db.query(ProtocolDeviation).filter(ProtocolDeviation.trial_site_id == ts_id)

    if severity:
        from cliniq.db.models import DeviationSeverity
        try:
            sev = DeviationSeverity(severity.lower())
            q = q.filter(ProtocolDeviation.severity == sev)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid severity: {severity}",
            )
    if category:
        q = q.filter(ProtocolDeviation.category == category.lower())

    total = q.count()
    offset = (page - 1) * page_size
    deviations = (
        q.order_by(ProtocolDeviation.deviation_date.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )

    log_action(db, token, "READ_SITE_DEVIATIONS", f"/sites/{ts_id}/deviations", request)
    db.commit()

    return DeviationListResponse(
        trial_site_id=ts_id,
        site_id=ts.site.site_id,
        deviations=[DeviationSchema.model_validate(d) for d in deviations],
        meta=PaginationMeta(total=total, page=page, page_size=page_size),
    )
