"""
ClinIQ Analytics — Enrolment Velocity Engine
Rolling 28-day enrolment rate, linear projection to target, completion date forecast.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from cliniq.db.models import EnrolmentStatus, PatientEnrolment, TrialSite


@dataclass
class VelocityResult:
    trial_site_id: int
    as_of_date: date
    enrolled_to_date: int
    enrolment_target: int
    velocity_28d: float              # patients enrolled per day (rolling 28-day)
    projected_completion: Optional[date]
    days_ahead_behind: Optional[int] # positive = ahead, negative = behind
    is_on_track: Optional[bool]
    remaining_to_target: int
    protocol_end_date: Optional[date]


def compute_velocity(
    session: Session,
    trial_site_id: int,
    as_of_date: Optional[date] = None,
) -> VelocityResult:
    """
    Compute enrolment velocity for a single trial-site as of a given date.
    Uses a 28-day rolling window ending on as_of_date.
    """
    if as_of_date is None:
        as_of_date = date.today()

    window_start = as_of_date - timedelta(days=28)

    ts: TrialSite = session.get(TrialSite, trial_site_id)
    if ts is None:
        raise ValueError(f"TrialSite {trial_site_id} not found")

    # Total enrolled to date
    enrolled_to_date = (
        session.query(PatientEnrolment)
        .filter(
            PatientEnrolment.trial_site_id == trial_site_id,
            PatientEnrolment.status.in_([
                EnrolmentStatus.ENROLLED,
                EnrolmentStatus.COMPLETED,
                EnrolmentStatus.WITHDRAWN,
            ]),
            PatientEnrolment.enrolled_date <= as_of_date,
        )
        .count()
    )

    # 28-day window enrolments
    window_enrolments = (
        session.query(PatientEnrolment)
        .filter(
            PatientEnrolment.trial_site_id == trial_site_id,
            PatientEnrolment.status.in_([
                EnrolmentStatus.ENROLLED,
                EnrolmentStatus.COMPLETED,
                EnrolmentStatus.WITHDRAWN,
            ]),
            PatientEnrolment.enrolled_date > window_start,
            PatientEnrolment.enrolled_date <= as_of_date,
        )
        .count()
    )

    velocity_28d = window_enrolments / 28.0
    remaining = max(0, ts.enrolment_target - enrolled_to_date)

    # Projection
    if remaining == 0:
        projected_completion = as_of_date
        is_on_track = True
        days_ahead_behind = None
    elif velocity_28d <= 0:
        projected_completion = None
        is_on_track = False
        days_ahead_behind = None
    else:
        days_needed = int(np.ceil(remaining / velocity_28d))
        projected_completion = as_of_date + timedelta(days=days_needed)

        protocol_end = ts.trial.planned_end_date if ts.trial else None
        if protocol_end:
            days_ahead_behind = (protocol_end - projected_completion).days
            is_on_track = days_ahead_behind >= 0
        else:
            days_ahead_behind = None
            is_on_track = None

    return VelocityResult(
        trial_site_id=trial_site_id,
        as_of_date=as_of_date,
        enrolled_to_date=enrolled_to_date,
        enrolment_target=ts.enrolment_target,
        velocity_28d=round(velocity_28d, 4),
        projected_completion=projected_completion,
        days_ahead_behind=days_ahead_behind,
        is_on_track=is_on_track,
        remaining_to_target=remaining,
        protocol_end_date=ts.trial.planned_end_date if ts.trial else None,
    )


def compute_velocity_all_sites(
    session: Session,
    trial_id: int,
    as_of_date: Optional[date] = None,
) -> list[VelocityResult]:
    """Compute velocity for every active site in a trial."""
    trial_sites = (
        session.query(TrialSite)
        .filter(TrialSite.trial_id == trial_id, TrialSite.is_active.is_(True))
        .all()
    )
    return [compute_velocity(session, ts.id, as_of_date) for ts in trial_sites]
