"""
ClinIQ ML — Dropout Risk Model & Composite Site Risk Score
Logistic regression dropout probability + weighted composite 0-100 risk score.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy.orm import Session

from cliniq.db.models import (
    DataEntryEvent, DeviationSeverity, EnrolmentStatus,
    MonitoringVisit, PatientEnrolment, ProtocolDeviation, TrialSite,
)

# ---------------------------------------------------------------------------
# Component weights for composite risk score (must sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHTS = {
    "enrolment_shortfall": 0.35,
    "deviation_rate":      0.25,
    "data_lag":            0.20,
    "dropout_probability": 0.10,
    "monitoring_recency":  0.10,
}


@dataclass
class RiskFeatures:
    """Raw feature values extracted for a trial-site."""
    trial_site_id: int
    enrolment_pct: float          # 0–1, % of target enrolled
    velocity_28d: float           # patients/day in last 28 days
    deviation_rate: float         # deviations per enrolled patient
    critical_deviation_rate: float
    data_lag_mean: float          # mean visit-to-entry days
    data_lag_p90: float
    days_since_monitoring: int    # days since last CRA visit
    open_query_rate: float        # open queries / enrolled patients


@dataclass
class RiskResult:
    trial_site_id: int
    as_of_date: date
    composite_score: float        # 0–100 (higher = more at risk)
    dropout_probability: float    # 0–1 from logistic regression
    enrolment_component: float
    deviation_component: float
    data_lag_component: float
    dropout_component: float
    monitoring_component: float
    features: RiskFeatures


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(session: Session, trial_site_id: int,
                     as_of_date: Optional[date] = None) -> RiskFeatures:
    if as_of_date is None:
        as_of_date = date.today()

    ts: TrialSite = session.get(TrialSite, trial_site_id)
    if ts is None:
        raise ValueError(f"TrialSite {trial_site_id} not found")

    window_28 = as_of_date - timedelta(days=28)

    # Enrolment counts
    enrolled_total = (
        session.query(PatientEnrolment)
        .filter(
            PatientEnrolment.trial_site_id == trial_site_id,
            PatientEnrolment.status.in_([
                EnrolmentStatus.ENROLLED,
                EnrolmentStatus.COMPLETED,
                EnrolmentStatus.WITHDRAWN,
            ]),
            PatientEnrolment.enrolled_date <= as_of_date,
        ).count()
    )
    enrolled_28d = (
        session.query(PatientEnrolment)
        .filter(
            PatientEnrolment.trial_site_id == trial_site_id,
            PatientEnrolment.status.in_([
                EnrolmentStatus.ENROLLED,
                EnrolmentStatus.COMPLETED,
                EnrolmentStatus.WITHDRAWN,
            ]),
            PatientEnrolment.enrolled_date > window_28,
            PatientEnrolment.enrolled_date <= as_of_date,
        ).count()
    )

    enrolment_pct = enrolled_total / ts.enrolment_target if ts.enrolment_target else 0.0
    velocity_28d  = enrolled_28d / 28.0

    # Deviations
    total_devs = (
        session.query(ProtocolDeviation)
        .filter(
            ProtocolDeviation.trial_site_id == trial_site_id,
            ProtocolDeviation.deviation_date <= as_of_date,
        ).count()
    )
    critical_devs = (
        session.query(ProtocolDeviation)
        .filter(
            ProtocolDeviation.trial_site_id == trial_site_id,
            ProtocolDeviation.severity == DeviationSeverity.CRITICAL,
            ProtocolDeviation.deviation_date <= as_of_date,
        ).count()
    )
    deviation_rate = total_devs / enrolled_total if enrolled_total else 0.0
    critical_dev_rate = critical_devs / enrolled_total if enrolled_total else 0.0

    # Data lag
    lag_rows = (
        session.query(DataEntryEvent.lag_days)
        .filter(
            DataEntryEvent.trial_site_id == trial_site_id,
            DataEntryEvent.visit_date <= as_of_date,
            DataEntryEvent.lag_days.isnot(None),
        ).all()
    )
    if lag_rows:
        lags = np.array([r.lag_days for r in lag_rows], dtype=float)
        lag_mean = float(np.mean(lags))
        lag_p90  = float(np.percentile(lags, 90))
    else:
        lag_mean = 0.0
        lag_p90  = 0.0

    # Monitoring recency
    last_visit = (
        session.query(MonitoringVisit.visit_date)
        .filter(
            MonitoringVisit.trial_site_id == trial_site_id,
            MonitoringVisit.visit_date <= as_of_date,
        )
        .order_by(MonitoringVisit.visit_date.desc())
        .first()
    )
    days_since_monitoring = (
        (as_of_date - last_visit.visit_date).days if last_visit else 999
    )

    # Open queries
    from cliniq.db.models import QueryLog
    open_queries = (
        session.query(QueryLog)
        .filter(
            QueryLog.trial_site_id == trial_site_id,
            QueryLog.is_resolved.is_(False),
        ).count()
    )
    open_query_rate = open_queries / enrolled_total if enrolled_total else 0.0

    return RiskFeatures(
        trial_site_id=trial_site_id,
        enrolment_pct=min(1.0, enrolment_pct),
        velocity_28d=velocity_28d,
        deviation_rate=deviation_rate,
        critical_deviation_rate=critical_dev_rate,
        data_lag_mean=lag_mean,
        data_lag_p90=lag_p90,
        days_since_monitoring=days_since_monitoring,
        open_query_rate=open_query_rate,
    )


# ---------------------------------------------------------------------------
# Logistic regression dropout probability
# Uses a simple heuristic-trained model for v1 — extensible to real labels.
# ---------------------------------------------------------------------------

def _build_dropout_model() -> LogisticRegression:
    """
    Bootstrap a logistic regression on synthetic training data.
    Feature vector: [enrolment_pct, deviation_rate, lag_mean_normalised, days_since_monitoring_normalised]
    In production: replace with real labelled site outcomes.
    """
    rng = np.random.default_rng(42)

    n = 400
    # Low-risk sites: high enrolment, low deviations, low lag, recent monitoring
    X_low = np.column_stack([
        rng.uniform(0.6, 1.0, n // 2),
        rng.uniform(0.0, 0.15, n // 2),
        rng.uniform(0.0, 0.3, n // 2),
        rng.uniform(0.0, 0.3, n // 2),
    ])
    # High-risk sites: low enrolment, high deviations, high lag, stale monitoring
    X_high = np.column_stack([
        rng.uniform(0.0, 0.4, n // 2),
        rng.uniform(0.2, 1.0, n // 2),
        rng.uniform(0.5, 1.0, n // 2),
        rng.uniform(0.5, 1.0, n // 2),
    ])
    X = np.vstack([X_low, X_high])
    y = np.array([0] * (n // 2) + [1] * (n // 2))

    # Add realistic noise
    noise_idx = rng.choice(n, size=n // 10, replace=False)
    y[noise_idx] = 1 - y[noise_idx]

    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X, y)
    return model


_DROPOUT_MODEL: Optional[LogisticRegression] = None
_SCALER = MinMaxScaler(feature_range=(0, 1))
_SCALER_FIT_BOUNDS = np.array([
    [0.0, 0.0, 0.0, 0.0],     # min bounds
    [1.0, 2.0, 60.0, 365.0],  # max bounds: enrolment_pct, dev_rate, lag_mean, days_since_monitoring
])
_SCALER.fit(_SCALER_FIT_BOUNDS)


def get_dropout_model() -> LogisticRegression:
    global _DROPOUT_MODEL
    if _DROPOUT_MODEL is None:
        _DROPOUT_MODEL = _build_dropout_model()
    return _DROPOUT_MODEL


def _scale_features(f: RiskFeatures) -> np.ndarray:
    raw = np.array([[
        f.enrolment_pct,
        f.deviation_rate,
        f.data_lag_mean,
        min(f.days_since_monitoring, 365),
    ]])
    return _SCALER.transform(raw)


def compute_dropout_probability(features: RiskFeatures) -> float:
    model = get_dropout_model()
    X = _scale_features(features)
    prob = model.predict_proba(X)[0][1]  # P(class=1 = high risk)
    return round(float(prob), 4)


# ---------------------------------------------------------------------------
# Composite risk score
# ---------------------------------------------------------------------------

def _enrolment_component(f: RiskFeatures) -> float:
    """Higher shortfall → higher risk. Returns 0–100."""
    shortfall = max(0.0, 1.0 - f.enrolment_pct)
    # Also penalise zero velocity
    velocity_penalty = 1.0 if f.velocity_28d == 0 else 0.0
    raw = (shortfall * 0.7) + (velocity_penalty * 0.3)
    return round(min(100.0, raw * 100), 2)


def _deviation_component(f: RiskFeatures) -> float:
    """Deviations per enrolled patient, scaled 0–100. Cap at rate=1.0."""
    rate = min(f.deviation_rate, 1.0)
    critical_boost = min(f.critical_deviation_rate * 2, 0.3)
    return round(min(100.0, (rate + critical_boost) * 100), 2)


def _lag_component(f: RiskFeatures) -> float:
    """Data lag mean, scaled against 30-day ceiling."""
    return round(min(100.0, (f.data_lag_mean / 30.0) * 100), 2)


def _monitoring_component(f: RiskFeatures) -> float:
    """Days since last monitoring visit, scaled against 90-day expected cadence."""
    return round(min(100.0, (f.days_since_monitoring / 90.0) * 100), 2)


def compute_risk_score(
    session: Session,
    trial_site_id: int,
    as_of_date: Optional[date] = None,
) -> RiskResult:
    """Compute composite risk score (0–100) for a trial-site."""
    if as_of_date is None:
        as_of_date = date.today()

    features = extract_features(session, trial_site_id, as_of_date)
    dropout_prob = compute_dropout_probability(features)

    enr_c   = _enrolment_component(features)
    dev_c   = _deviation_component(features)
    lag_c   = _lag_component(features)
    mon_c   = _monitoring_component(features)
    drop_c  = round(dropout_prob * 100, 2)

    composite = (
        enr_c  * WEIGHTS["enrolment_shortfall"] +
        dev_c  * WEIGHTS["deviation_rate"] +
        lag_c  * WEIGHTS["data_lag"] +
        drop_c * WEIGHTS["dropout_probability"] +
        mon_c  * WEIGHTS["monitoring_recency"]
    )
    composite = round(min(100.0, max(0.0, composite)), 2)

    return RiskResult(
        trial_site_id=trial_site_id,
        as_of_date=as_of_date,
        composite_score=composite,
        dropout_probability=dropout_prob,
        enrolment_component=enr_c,
        deviation_component=dev_c,
        data_lag_component=lag_c,
        dropout_component=drop_c,
        monitoring_component=mon_c,
        features=features,
    )
