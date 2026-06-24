"""
ClinIQ — SQLAlchemy Models
Three-schema architecture: Protocol Config / Operational Time-Series / Derived Analytics
"""
from __future__ import annotations

import hashlib
from datetime import datetime, date
from enum import Enum as PyEnum

from sqlalchemy import (
    BigInteger, Boolean, Column, Date, DateTime, Enum, Float,
    ForeignKey, Index, Integer, String, Text, UniqueConstraint,
    event,
)
from sqlalchemy.orm import DeclarativeBase, relationship, validates


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TrialStatus(str, PyEnum):
    PLANNED    = "planned"
    ACTIVE     = "active"
    SUSPENDED  = "suspended"
    COMPLETED  = "completed"
    TERMINATED = "terminated"


class TrialPhase(str, PyEnum):
    PHASE_I    = "I"
    PHASE_II   = "II"
    PHASE_III  = "III"
    PHASE_IV   = "IV"


class SiteType(str, PyEnum):
    ACADEMIC   = "academic"
    HOSPITAL   = "hospital"
    COMMUNITY  = "community"
    PRIVATE    = "private"


class EnrolmentStatus(str, PyEnum):
    SCREENED   = "screened"
    ENROLLED   = "enrolled"
    WITHDRAWN  = "withdrawn"
    COMPLETED  = "completed"
    SCREEN_FAIL = "screen_fail"


class DeviationSeverity(str, PyEnum):
    MINOR    = "minor"
    MAJOR    = "major"
    CRITICAL = "critical"


class VisitType(str, PyEnum):
    ONSITE       = "onsite"
    REMOTE       = "remote"
    CENTRALISED  = "centralised"


class UserRole(str, PyEnum):
    ADMIN          = "admin"
    CLINICAL_LEAD  = "clinical_lead"
    CRA            = "cra"
    SPONSOR_VIEW   = "sponsor_view"


# ---------------------------------------------------------------------------
# SCHEMA LAYER 1 — Protocol Configuration
# ---------------------------------------------------------------------------

class Trial(Base):
    """Master trial registry with protocol version and regulatory IDs."""
    __tablename__ = "trials"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    trial_id       = Column(String(64), nullable=False, unique=True, index=True)
    sponsor        = Column(String(200), nullable=False)
    phase          = Column(Enum(TrialPhase), nullable=False)
    status         = Column(Enum(TrialStatus), nullable=False, default=TrialStatus.PLANNED)
    title          = Column(String(400), nullable=False)
    therapeutic_area = Column(String(100))
    eudract_number = Column(String(50))          # EU regulatory ID
    isrctn_number  = Column(String(50))          # UK registry ID
    start_date     = Column(Date)
    planned_end_date = Column(Date)
    created_at     = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at     = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    trial_sites        = relationship("TrialSite", back_populates="trial", cascade="all, delete-orphan")
    protocol_versions  = relationship("ProtocolVersion", back_populates="trial", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Trial {self.trial_id} phase={self.phase} status={self.status}>"


class Site(Base):
    """Investigator site registry; supports multi-country EU trials."""
    __tablename__ = "sites"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    site_id     = Column(String(64), nullable=False, unique=True, index=True)
    name        = Column(String(300), nullable=False)
    country     = Column(String(3), nullable=False)       # ISO 3166-1 alpha-3
    city        = Column(String(100))
    pi_name     = Column(String(200), nullable=False)     # Principal Investigator
    pi_email    = Column(String(254))
    site_type   = Column(Enum(SiteType), nullable=False)
    is_active   = Column(Boolean, nullable=False, default=True)
    created_at  = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at  = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    trial_sites = relationship("TrialSite", back_populates="site", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Site {self.site_id} {self.name} ({self.country})>"


class TrialSite(Base):
    """Junction table: links trials to sites with per-site enrolment quota."""
    __tablename__ = "trial_sites"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    trial_id          = Column(Integer, ForeignKey("trials.id", ondelete="CASCADE"), nullable=False)
    site_id           = Column(Integer, ForeignKey("sites.id", ondelete="CASCADE"), nullable=False)
    enrolment_target  = Column(Integer, nullable=False)
    activation_date   = Column(Date)
    close_date        = Column(Date)
    is_active         = Column(Boolean, nullable=False, default=True)
    created_at        = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    trial              = relationship("Trial", back_populates="trial_sites")
    site               = relationship("Site", back_populates="trial_sites")
    patient_enrolments = relationship("PatientEnrolment", back_populates="trial_site", cascade="all, delete-orphan")
    protocol_deviations = relationship("ProtocolDeviation", back_populates="trial_site", cascade="all, delete-orphan")
    data_entry_events  = relationship("DataEntryEvent", back_populates="trial_site", cascade="all, delete-orphan")
    monitoring_visits  = relationship("MonitoringVisit", back_populates="trial_site", cascade="all, delete-orphan")
    query_logs         = relationship("QueryLog", back_populates="trial_site", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("trial_id", "site_id", name="uq_trial_site"),
    )

    @validates("enrolment_target")
    def validate_enrolment_target(self, key, value):
        if value < 1:
            raise ValueError("enrolment_target must be >= 1")
        return value

    def __repr__(self) -> str:
        return f"<TrialSite trial={self.trial_id} site={self.site_id} target={self.enrolment_target}>"


class ProtocolVersion(Base):
    """Protocol amendment log; contextualises deviation spikes."""
    __tablename__ = "protocol_versions"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    trial_id        = Column(Integer, ForeignKey("trials.id", ondelete="CASCADE"), nullable=False)
    version         = Column(String(20), nullable=False)
    amendment_date  = Column(Date, nullable=False)
    changes         = Column(Text)
    created_at      = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    trial = relationship("Trial", back_populates="protocol_versions")

    __table_args__ = (
        UniqueConstraint("trial_id", "version", name="uq_trial_version"),
    )

    def __repr__(self) -> str:
        return f"<ProtocolVersion trial={self.trial_id} v{self.version}>"


# ---------------------------------------------------------------------------
# SCHEMA LAYER 2 — Operational Time-Series
# ---------------------------------------------------------------------------

class PatientEnrolment(Base):
    """Per-patient enrolment funnel: screening → consent → enrolment → withdrawal."""
    __tablename__ = "patient_enrolments"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    trial_site_id   = Column(Integer, ForeignKey("trial_sites.id", ondelete="CASCADE"), nullable=False)
    patient_id      = Column(String(64), nullable=False)   # SHA-256 hash — no PII
    screened_date   = Column(Date, nullable=False)
    enrolled_date   = Column(Date)
    completion_date = Column(Date)
    withdrawal_date = Column(Date)
    status          = Column(Enum(EnrolmentStatus), nullable=False, default=EnrolmentStatus.SCREENED)
    created_at      = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    trial_site = relationship("TrialSite", back_populates="patient_enrolments")

    __table_args__ = (
        UniqueConstraint("trial_site_id", "patient_id", name="uq_patient_per_site"),
        Index("ix_enrolments_screened_date", "screened_date"),
        Index("ix_enrolments_trial_site_status", "trial_site_id", "status"),
    )

    @staticmethod
    def hash_patient_code(raw_code: str) -> str:
        """Pseudonymise a sponsor patient code to a SHA-256 hash."""
        return hashlib.sha256(raw_code.encode()).hexdigest()

    def __repr__(self) -> str:
        return f"<PatientEnrolment {self.patient_id} status={self.status}>"


class ProtocolDeviation(Base):
    """Deviation log with severity and free-text for NLP categorisation."""
    __tablename__ = "protocol_deviations"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    trial_site_id   = Column(Integer, ForeignKey("trial_sites.id", ondelete="CASCADE"), nullable=False)
    deviation_id    = Column(String(64), nullable=False, unique=True)
    category        = Column(String(100))           # Populated by NLP classifier post-insert
    severity        = Column(Enum(DeviationSeverity), nullable=False)
    deviation_date  = Column(Date, nullable=False)
    reported_date   = Column(Date)
    free_text       = Column(Text)
    is_resolved     = Column(Boolean, nullable=False, default=False)
    created_at      = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    trial_site = relationship("TrialSite", back_populates="protocol_deviations")

    __table_args__ = (
        Index("ix_deviations_trial_site_severity", "trial_site_id", "severity"),
        Index("ix_deviations_date", "deviation_date"),
    )

    def __repr__(self) -> str:
        return f"<ProtocolDeviation {self.deviation_id} severity={self.severity}>"


class DataEntryEvent(Base):
    """EDC entry timestamps; computes data lag KPI (visit-to-entry days)."""
    __tablename__ = "data_entry_events"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    trial_site_id   = Column(Integer, ForeignKey("trial_sites.id", ondelete="CASCADE"), nullable=False)
    visit_date      = Column(Date, nullable=False)
    entry_date      = Column(Date, nullable=False)
    ecrf_page       = Column(String(100))
    lag_days        = Column(Integer)               # Computed: entry_date - visit_date
    created_at      = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    trial_site = relationship("TrialSite", back_populates="data_entry_events")

    __table_args__ = (
        Index("ix_data_entry_trial_site_visit", "trial_site_id", "visit_date"),
    )

    @validates("lag_days")
    def validate_lag(self, key, value):
        if value is not None and value < 0:
            raise ValueError("lag_days cannot be negative")
        return value

    def __repr__(self) -> str:
        return f"<DataEntryEvent site={self.trial_site_id} lag={self.lag_days}d>"


class MonitoringVisit(Base):
    """CRA visit log; tracks SDV completion and open findings."""
    __tablename__ = "monitoring_visits"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    trial_site_id   = Column(Integer, ForeignKey("trial_sites.id", ondelete="CASCADE"), nullable=False)
    visit_type      = Column(Enum(VisitType), nullable=False)
    visit_date      = Column(Date, nullable=False)
    findings        = Column(Text)
    sdv_complete    = Column(Boolean, default=False)
    created_at      = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    trial_site = relationship("TrialSite", back_populates="monitoring_visits")

    __table_args__ = (
        Index("ix_monitoring_trial_site_date", "trial_site_id", "visit_date"),
    )

    def __repr__(self) -> str:
        return f"<MonitoringVisit site={self.trial_site_id} type={self.visit_type} date={self.visit_date}>"


class QueryLog(Base):
    """Data query lifecycle; drives query resolution time KPI."""
    __tablename__ = "query_log"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    trial_site_id   = Column(Integer, ForeignKey("trial_sites.id", ondelete="CASCADE"), nullable=False)
    query_id        = Column(String(64), nullable=False, unique=True)
    opened_date     = Column(Date, nullable=False)
    resolved_date   = Column(Date)
    category        = Column(String(100))
    is_resolved     = Column(Boolean, nullable=False, default=False)
    age_days        = Column(Integer)               # Snapshot: days open at last refresh
    created_at      = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    trial_site = relationship("TrialSite", back_populates="query_logs")

    __table_args__ = (
        Index("ix_query_log_trial_site_resolved", "trial_site_id", "is_resolved"),
    )

    def __repr__(self) -> str:
        return f"<QueryLog {self.query_id} resolved={self.is_resolved}>"


# ---------------------------------------------------------------------------
# SCHEMA LAYER 3 — Derived Analytics
# ---------------------------------------------------------------------------

class SiteKPISnapshot(Base):
    """Daily rollup of enrolment rate, deviation rate, data lag, query age."""
    __tablename__ = "site_kpi_snapshots"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    trial_site_id       = Column(Integer, ForeignKey("trial_sites.id", ondelete="CASCADE"), nullable=False)
    snapshot_date       = Column(Date, nullable=False)
    enrolment_rate_28d  = Column(Float)   # Patients enrolled per day (28-day rolling)
    enrolment_pct       = Column(Float)   # % of target enrolled
    deviation_rate      = Column(Float)   # Deviations per enrolled patient
    data_lag_mean       = Column(Float)   # Mean visit-to-entry days
    data_lag_p90        = Column(Float)   # 90th percentile lag
    open_queries        = Column(Integer)
    query_age_mean      = Column(Float)   # Mean age of open queries (days)
    days_since_monitoring = Column(Integer)
    created_at          = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("trial_site_id", "snapshot_date", name="uq_kpi_snapshot_date"),
        Index("ix_kpi_snapshots_date", "snapshot_date"),
    )

    def __repr__(self) -> str:
        return f"<SiteKPISnapshot site={self.trial_site_id} date={self.snapshot_date}>"


class RiskScore(Base):
    """Composite site risk score (0–100) with component weights."""
    __tablename__ = "risk_scores"

    id                      = Column(Integer, primary_key=True, autoincrement=True)
    trial_site_id           = Column(Integer, ForeignKey("trial_sites.id", ondelete="CASCADE"), nullable=False)
    score_date              = Column(Date, nullable=False)
    composite_score         = Column(Float, nullable=False)   # 0–100
    enrolment_component     = Column(Float)   # Contribution from enrolment shortfall
    deviation_component     = Column(Float)   # Contribution from deviation rate
    data_lag_component      = Column(Float)   # Contribution from data lag
    dropout_probability     = Column(Float)   # Output of logistic regression model (0–1)
    monitoring_component    = Column(Float)   # Days since last monitoring visit
    created_at              = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("trial_site_id", "score_date", name="uq_risk_score_date"),
        Index("ix_risk_scores_date_score", "score_date", "composite_score"),
    )

    @validates("composite_score")
    def validate_score(self, key, value):
        if not (0.0 <= value <= 100.0):
            raise ValueError(f"composite_score must be in [0, 100], got {value}")
        return value

    def __repr__(self) -> str:
        return f"<RiskScore site={self.trial_site_id} date={self.score_date} score={self.composite_score:.1f}>"


class EnrolmentForecast(Base):
    """Per-site projected completion date from linear regression."""
    __tablename__ = "enrolment_forecasts"

    id                      = Column(Integer, primary_key=True, autoincrement=True)
    trial_site_id           = Column(Integer, ForeignKey("trial_sites.id", ondelete="CASCADE"), nullable=False)
    forecast_date           = Column(Date, nullable=False)     # When forecast was computed
    projected_completion    = Column(Date)                     # Projected enrolment close date
    velocity_28d            = Column(Float)                    # Rolling 28-day daily rate
    enrolled_to_date        = Column(Integer)
    remaining_to_target     = Column(Integer)
    is_on_track             = Column(Boolean)
    days_ahead_behind       = Column(Integer)                  # Positive = ahead, negative = behind
    created_at              = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("trial_site_id", "forecast_date", name="uq_forecast_date"),
    )

    def __repr__(self) -> str:
        return f"<EnrolmentForecast site={self.trial_site_id} projected={self.projected_completion}>"


class DeviationCluster(Base):
    """NLP-tagged deviation categories and per-site cluster profiles."""
    __tablename__ = "deviation_clusters"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    trial_site_id   = Column(Integer, ForeignKey("trial_sites.id", ondelete="CASCADE"), nullable=False)
    cluster_date    = Column(Date, nullable=False)
    category        = Column(String(100), nullable=False)   # e.g. consent, dosing, eligibility
    count           = Column(Integer, nullable=False, default=0)
    pct_of_total    = Column(Float)
    created_at      = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("trial_site_id", "cluster_date", "category", name="uq_cluster_cat_date"),
    )

    def __repr__(self) -> str:
        return f"<DeviationCluster site={self.trial_site_id} cat={self.category} n={self.count}>"


class AuditLog(Base):
    """
    Immutable event log — GCP audit trail (ICH E6 / 21 CFR Part 11).
    NO UPDATE or DELETE ever issued against this table.
    """
    __tablename__ = "audit_log"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    timestamp   = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    user_id     = Column(String(64))
    role        = Column(Enum(UserRole))
    action      = Column(String(100), nullable=False)   # e.g. READ_SITE, EXPORT_PDF
    resource    = Column(String(200))                   # e.g. /sites/SITE_001
    detail      = Column(Text)
    ip_address  = Column(String(45))                    # IPv4 or IPv6
    # No foreign keys — audit log must survive cascade deletes of operational data

    def __repr__(self) -> str:
        return f"<AuditLog {self.timestamp} {self.action} by {self.user_id}>"
