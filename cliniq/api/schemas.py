"""
ClinIQ — Pydantic response schemas (v2).
All API responses are typed; no raw ORM objects returned to clients.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class PaginationMeta(BaseModel):
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    user_id: str


class LoginRequest(BaseModel):
    user_id: str
    password: str
    role: str   # In demo mode: role is trusted. Production: validate against user store.
    trial_id: Optional[int] = None  # Required for sponsor_view


# ---------------------------------------------------------------------------
# Trial
# ---------------------------------------------------------------------------

class TrialSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    trial_id: str
    sponsor: str
    phase: str
    status: str
    title: str
    therapeutic_area: Optional[str]
    eudract_number: Optional[str]
    isrctn_number: Optional[str]
    start_date: Optional[date]
    planned_end_date: Optional[date]
    site_count: int = 0


class TrialDetail(TrialSummary):
    protocol_versions: list[ProtocolVersionSchema] = []


# ---------------------------------------------------------------------------
# Site
# ---------------------------------------------------------------------------

class SiteSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    site_id: str
    name: str
    country: str
    city: Optional[str]
    pi_name: str
    site_type: str
    is_active: bool


# ---------------------------------------------------------------------------
# Protocol Version
# ---------------------------------------------------------------------------

class ProtocolVersionSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    version: str
    amendment_date: date
    changes: Optional[str]


# ---------------------------------------------------------------------------
# KPI Snapshot
# ---------------------------------------------------------------------------

class KPISnapshotSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    snapshot_date: date
    enrolment_rate_28d: Optional[float]
    enrolment_pct: Optional[float]
    deviation_rate: Optional[float]
    data_lag_mean: Optional[float]
    data_lag_p90: Optional[float]
    open_queries: Optional[int]
    query_age_mean: Optional[float]
    days_since_monitoring: Optional[int]


class KPIResponse(BaseModel):
    trial_site_id: int
    site_id: str
    trial_id: str
    snapshots: list[KPISnapshotSchema]


# ---------------------------------------------------------------------------
# Risk Score
# ---------------------------------------------------------------------------

class RiskScoreSchema(BaseModel):
    trial_site_id: int
    site_id: str
    as_of_date: date
    composite_score: float
    dropout_probability: float
    enrolment_component: float
    deviation_component: float
    data_lag_component: float
    dropout_component: float
    monitoring_component: float


# ---------------------------------------------------------------------------
# Enrolment Forecast
# ---------------------------------------------------------------------------

class ForecastSchema(BaseModel):
    trial_site_id: int
    site_id: str
    forecast_date: date
    projected_completion: Optional[date]
    velocity_28d: Optional[float]
    enrolled_to_date: Optional[int]
    remaining_to_target: Optional[int]
    is_on_track: Optional[bool]
    days_ahead_behind: Optional[int]


# ---------------------------------------------------------------------------
# Deviation
# ---------------------------------------------------------------------------

class DeviationSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    deviation_id: str
    category: Optional[str]
    severity: str
    deviation_date: date
    reported_date: Optional[date]
    free_text: Optional[str]
    is_resolved: bool


class DeviationListResponse(BaseModel):
    trial_site_id: int
    site_id: str
    deviations: list[DeviationSchema]
    meta: PaginationMeta


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------

class WatchlistEntry(BaseModel):
    trial_site_id: int
    site_id: str
    site_name: str
    country: str
    composite_score: float
    dropout_probability: float
    enrolment_pct: Optional[float]
    is_on_track: Optional[bool]
    alert_flags: list[str]


class WatchlistResponse(BaseModel):
    trial_id: str
    as_of_date: date
    entries: list[WatchlistEntry]


# ---------------------------------------------------------------------------
# AI query
# ---------------------------------------------------------------------------

class AIQueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    trial_site_id: Optional[int] = None


class AIQueryResponse(BaseModel):
    question: str
    answer: str
    context_used: str


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class ReportResponse(BaseModel):
    site_id: str
    report_type: str
    generated_at: datetime
    download_url: str
