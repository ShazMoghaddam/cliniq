"""
ClinIQ — shared pytest fixtures.
Uses an in-memory SQLite engine scoped to each test function for isolation.
"""
import hashlib
import uuid
from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

from cliniq.db.models import (
    Base, Trial, Site, TrialSite, ProtocolVersion, PatientEnrolment,
    ProtocolDeviation, DataEntryEvent, MonitoringVisit, QueryLog,
    TrialPhase, TrialStatus, SiteType, EnrolmentStatus, DeviationSeverity,
    VisitType,
)


# ---------------------------------------------------------------------------
# Engine & Session fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def engine():
    """In-memory SQLite engine, foreign keys enabled."""
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})

    @event.listens_for(eng, "connect")
    def set_pragma(conn, _):
        conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(bind=eng)
    yield eng
    Base.metadata.drop_all(bind=eng)


@pytest.fixture(scope="function")
def session(engine) -> Session:
    """Transactional session that rolls back after each test."""
    connection = engine.connect()
    transaction = connection.begin()
    SessionLocal = sessionmaker(bind=connection, autocommit=False, autoflush=False)
    db = SessionLocal()
    yield db
    db.close()
    transaction.rollback()
    connection.close()


# ---------------------------------------------------------------------------
# Minimal fixture data (3 sites, 30 days of activity)
# ---------------------------------------------------------------------------

TODAY = date.today()
START = TODAY - timedelta(days=30)


@pytest.fixture
def trial(session) -> Trial:
    t = Trial(
        trial_id="TEST-001",
        sponsor="Test Sponsor Ltd",
        phase=TrialPhase.PHASE_II,
        status=TrialStatus.ACTIVE,
        title="Test Trial",
        start_date=START,
        planned_end_date=TODAY + timedelta(days=365),
    )
    session.add(t)
    session.flush()
    return t


@pytest.fixture
def sites(session) -> list[Site]:
    data = [
        ("SITE-A", "Alpha Hospital",    "GBR", "London",    "Dr. A",  SiteType.HOSPITAL),
        ("SITE-B", "Beta Academy",      "DEU", "Berlin",    "Dr. B",  SiteType.ACADEMIC),
        ("SITE-C", "Gamma Community",   "POL", "Warsaw",    "Dr. C",  SiteType.COMMUNITY),
    ]
    objs = []
    for sid, name, country, city, pi, stype in data:
        s = Site(site_id=sid, name=name, country=country, city=city,
                 pi_name=pi, site_type=stype)
        session.add(s)
        session.flush()
        objs.append(s)
    return objs


@pytest.fixture
def trial_sites(session, trial, sites) -> list[TrialSite]:
    targets = [10, 15, 8]
    objs = []
    for site, target in zip(sites, targets):
        ts = TrialSite(
            trial_id=trial.id,
            site_id=site.id,
            enrolment_target=target,
            activation_date=START,
        )
        session.add(ts)
        session.flush()
        objs.append(ts)
    return objs


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


@pytest.fixture
def patient_enrolments(session, trial_sites) -> list[PatientEnrolment]:
    """5 enrolled + 2 screen failures for trial_site[0]."""
    ts = trial_sites[0]
    objs = []
    for i in range(5):
        enr_date = START + timedelta(days=i * 4)
        pe = PatientEnrolment(
            trial_site_id=ts.id,
            patient_id=_hash(f"TS0-P{i}"),
            screened_date=enr_date - timedelta(days=5),
            enrolled_date=enr_date,
            status=EnrolmentStatus.ENROLLED,
        )
        session.add(pe)
        objs.append(pe)
    for j in range(2):
        pe = PatientEnrolment(
            trial_site_id=ts.id,
            patient_id=_hash(f"TS0-SF{j}"),
            screened_date=START + timedelta(days=j),
            status=EnrolmentStatus.SCREEN_FAIL,
        )
        session.add(pe)
        objs.append(pe)
    session.flush()
    return objs


@pytest.fixture
def deviations(session, trial_sites) -> list[ProtocolDeviation]:
    ts = trial_sites[0]
    severities = [DeviationSeverity.MINOR] * 4 + [DeviationSeverity.MAJOR] * 2 + [DeviationSeverity.CRITICAL]
    objs = []
    for i, sev in enumerate(severities):
        d = ProtocolDeviation(
            trial_site_id=ts.id,
            deviation_id=str(uuid.uuid4()),
            severity=sev,
            deviation_date=START + timedelta(days=i * 3),
            free_text=f"Deviation {i}: test free text for {sev}",
        )
        session.add(d)
        objs.append(d)
    session.flush()
    return objs


@pytest.fixture
def data_entries(session, trial_sites) -> list[DataEntryEvent]:
    ts = trial_sites[0]
    objs = []
    for i in range(10):
        visit = START + timedelta(days=i * 2)
        lag = i % 8
        de = DataEntryEvent(
            trial_site_id=ts.id,
            visit_date=visit,
            entry_date=visit + timedelta(days=lag),
            ecrf_page=f"CRF-{i:02d}",
            lag_days=lag,
        )
        session.add(de)
        objs.append(de)
    session.flush()
    return objs


@pytest.fixture
def monitoring_visits(session, trial_sites) -> list[MonitoringVisit]:
    ts = trial_sites[0]
    objs = []
    for i, vtype in enumerate([VisitType.ONSITE, VisitType.REMOTE, VisitType.CENTRALISED]):
        mv = MonitoringVisit(
            trial_site_id=ts.id,
            visit_type=vtype,
            visit_date=START + timedelta(days=i * 10),
            sdv_complete=i % 2 == 0,
        )
        session.add(mv)
        objs.append(mv)
    session.flush()
    return objs


@pytest.fixture
def query_logs(session, trial_sites) -> list[QueryLog]:
    ts = trial_sites[0]
    objs = []
    # 3 open + 2 resolved
    for i in range(5):
        is_resolved = i >= 3
        ql = QueryLog(
            trial_site_id=ts.id,
            query_id=str(uuid.uuid4()),
            opened_date=START + timedelta(days=i * 5),
            resolved_date=(START + timedelta(days=i * 5 + 7)) if is_resolved else None,
            category="data_entry",
            is_resolved=is_resolved,
            age_days=None if is_resolved else (TODAY - (START + timedelta(days=i * 5))).days,
        )
        session.add(ql)
        objs.append(ql)
    session.flush()
    return objs
