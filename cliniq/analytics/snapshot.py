"""
ClinIQ Analytics — KPI Snapshot Writer
Scheduled job: writes daily KPI rollup to site_kpi_snapshots and risk_scores tables.
Idempotent: skips sites already snapshotted for the target date.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from cliniq.db.models import (
    DataEntryEvent, EnrolmentForecast, EnrolmentStatus,
    MonitoringVisit, PatientEnrolment, QueryLog,
    RiskScore, SiteKPISnapshot, Trial, TrialSite,
)
from cliniq.analytics.velocity import compute_velocity
from cliniq.analytics.lag import compute_lag
from cliniq.ml.risk_model import compute_risk_score


@dataclass
class SnapshotSummary:
    target_date: date
    sites_processed: int
    sites_skipped: int
    errors: list[str]


def write_snapshot_for_site(
    session: Session,
    trial_site_id: int,
    target_date: Optional[date] = None,
) -> bool:
    """
    Write KPI snapshot for a single trial-site on target_date.
    Returns True if written, False if already exists (idempotent).
    """
    if target_date is None:
        target_date = date.today()

    existing = (
        session.query(SiteKPISnapshot)
        .filter_by(trial_site_id=trial_site_id, snapshot_date=target_date)
        .first()
    )
    if existing:
        return False

    ts: TrialSite = session.get(TrialSite, trial_site_id)
    if ts is None:
        raise ValueError(f"TrialSite {trial_site_id} not found")

    # ---- Enrolment ----
    velocity = compute_velocity(session, trial_site_id, target_date)
    enrolment_pct = (
        velocity.enrolled_to_date / ts.enrolment_target
        if ts.enrolment_target else 0.0
    )

    # ---- Deviations ----
    from cliniq.db.models import ProtocolDeviation
    total_devs = (
        session.query(ProtocolDeviation)
        .filter(
            ProtocolDeviation.trial_site_id == trial_site_id,
            ProtocolDeviation.deviation_date <= target_date,
        ).count()
    )
    dev_rate = (
        total_devs / velocity.enrolled_to_date
        if velocity.enrolled_to_date else 0.0
    )

    # ---- Data lag ----
    lag = compute_lag(session, trial_site_id, target_date)

    # ---- Queries ----
    open_queries = (
        session.query(QueryLog)
        .filter(
            QueryLog.trial_site_id == trial_site_id,
            QueryLog.is_resolved.is_(False),
        ).count()
    )
    query_ages = (
        session.query(QueryLog.age_days)
        .filter(
            QueryLog.trial_site_id == trial_site_id,
            QueryLog.is_resolved.is_(False),
            QueryLog.age_days.isnot(None),
        ).all()
    )
    query_age_mean = float(np.mean([r.age_days for r in query_ages])) if query_ages else None

    # ---- Monitoring recency ----
    last_visit = (
        session.query(MonitoringVisit.visit_date)
        .filter(
            MonitoringVisit.trial_site_id == trial_site_id,
            MonitoringVisit.visit_date <= target_date,
        )
        .order_by(MonitoringVisit.visit_date.desc())
        .first()
    )
    days_since_monitoring = (
        (target_date - last_visit.visit_date).days if last_visit else None
    )

    # ---- Write snapshot ----
    snapshot = SiteKPISnapshot(
        trial_site_id=trial_site_id,
        snapshot_date=target_date,
        enrolment_rate_28d=velocity.velocity_28d,
        enrolment_pct=round(enrolment_pct, 4),
        deviation_rate=round(dev_rate, 4),
        data_lag_mean=lag.lag_mean,
        data_lag_p90=lag.lag_p90,
        open_queries=open_queries,
        query_age_mean=round(query_age_mean, 2) if query_age_mean else None,
        days_since_monitoring=days_since_monitoring,
    )
    session.add(snapshot)

    # ---- Write risk score ----
    existing_risk = (
        session.query(RiskScore)
        .filter_by(trial_site_id=trial_site_id, score_date=target_date)
        .first()
    )
    if not existing_risk:
        risk = compute_risk_score(session, trial_site_id, target_date)
        session.add(RiskScore(
            trial_site_id=trial_site_id,
            score_date=target_date,
            composite_score=risk.composite_score,
            enrolment_component=risk.enrolment_component,
            deviation_component=risk.deviation_component,
            data_lag_component=risk.data_lag_component,
            dropout_probability=risk.dropout_probability,
            monitoring_component=risk.monitoring_component,
        ))

    # ---- Write enrolment forecast ----
    existing_forecast = (
        session.query(EnrolmentForecast)
        .filter_by(trial_site_id=trial_site_id, forecast_date=target_date)
        .first()
    )
    if not existing_forecast and velocity.projected_completion:
        session.add(EnrolmentForecast(
            trial_site_id=trial_site_id,
            forecast_date=target_date,
            projected_completion=velocity.projected_completion,
            velocity_28d=velocity.velocity_28d,
            enrolled_to_date=velocity.enrolled_to_date,
            remaining_to_target=velocity.remaining_to_target,
            is_on_track=velocity.is_on_track,
            days_ahead_behind=velocity.days_ahead_behind,
        ))

    return True


def run_daily_snapshot(
    session: Session,
    trial_id: int,
    target_date: Optional[date] = None,
) -> SnapshotSummary:
    """
    Run the daily KPI snapshot job for all active sites in a trial.
    Commits after each site to limit transaction scope.
    """
    if target_date is None:
        target_date = date.today()

    trial_sites = (
        session.query(TrialSite)
        .filter(TrialSite.trial_id == trial_id, TrialSite.is_active.is_(True))
        .all()
    )

    processed = 0
    skipped = 0
    errors: list[str] = []

    for ts in trial_sites:
        try:
            written = write_snapshot_for_site(session, ts.id, target_date)
            session.commit()
            if written:
                processed += 1
            else:
                skipped += 1
        except Exception as e:
            session.rollback()
            errors.append(f"site {ts.id}: {e}")

    return SnapshotSummary(
        target_date=target_date,
        sites_processed=processed,
        sites_skipped=skipped,
        errors=errors,
    )
