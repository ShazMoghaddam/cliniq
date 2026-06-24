"""
ClinIQ Analytics — Data Lag Aggregator
Mean and 90th-percentile visit-to-entry days per site, 7-day trend.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from cliniq.db.models import DataEntryEvent, TrialSite


@dataclass
class LagResult:
    trial_site_id: int
    as_of_date: date
    n_events: int
    lag_mean: Optional[float]    # mean visit-to-entry days
    lag_p90: Optional[float]     # 90th percentile
    lag_max: Optional[float]
    trend_7d: Optional[float]    # mean lag this 7 days vs prev 7 days (negative = improving)


def compute_lag(
    session: Session,
    trial_site_id: int,
    as_of_date: Optional[date] = None,
    lookback_days: int = 90,
) -> LagResult:
    """
    Compute data lag metrics for a trial-site over a lookback window.
    Trend compares the most recent 7 days vs the prior 7 days.
    """
    if as_of_date is None:
        as_of_date = date.today()

    window_start = as_of_date - timedelta(days=lookback_days)

    rows = (
        session.query(DataEntryEvent.lag_days, DataEntryEvent.visit_date)
        .filter(
            DataEntryEvent.trial_site_id == trial_site_id,
            DataEntryEvent.visit_date >= window_start,
            DataEntryEvent.visit_date <= as_of_date,
            DataEntryEvent.lag_days.isnot(None),
        )
        .all()
    )

    if not rows:
        return LagResult(
            trial_site_id=trial_site_id,
            as_of_date=as_of_date,
            n_events=0,
            lag_mean=None, lag_p90=None, lag_max=None, trend_7d=None,
        )

    lags = np.array([r.lag_days for r in rows], dtype=float)

    # 7-day trend
    recent_cutoff  = as_of_date - timedelta(days=7)
    prior_cutoff   = as_of_date - timedelta(days=14)

    recent_lags = [r.lag_days for r in rows if r.visit_date > recent_cutoff]
    prior_lags  = [r.lag_days for r in rows
                   if prior_cutoff < r.visit_date <= recent_cutoff]

    trend = None
    if recent_lags and prior_lags:
        trend = round(float(np.mean(recent_lags)) - float(np.mean(prior_lags)), 2)
    elif recent_lags:
        trend = 0.0

    return LagResult(
        trial_site_id=trial_site_id,
        as_of_date=as_of_date,
        n_events=len(lags),
        lag_mean=round(float(np.mean(lags)), 2),
        lag_p90=round(float(np.percentile(lags, 90)), 2),
        lag_max=round(float(np.max(lags)), 2),
        trend_7d=trend,
    )
