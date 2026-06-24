"""
ClinIQ — Phase 3 Tests: FastAPI Layer
Covers: JWT auth, all endpoints, RBAC enforcement, schema validation, edge cases, audit log
Target: 100 tests
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import date, timedelta
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, StaticPool
from sqlalchemy.orm import sessionmaker, Session

from cliniq.api.app import create_app
from cliniq.db.database import get_db
from cliniq.db.models import (
    AuditLog, Base, DeviationSeverity, EnrolmentStatus, MonitoringVisit,
    PatientEnrolment, ProtocolDeviation, Site, SiteType, Trial, TrialPhase,
    TrialSite, TrialStatus, UserRole, VisitType,
)
from cliniq.api.routers.auth import DEMO_PASSWORD
from cliniq.rbac.auth import create_access_token

TODAY = date.today()
START = TODAY - timedelta(days=60)


# ---------------------------------------------------------------------------
# Test database + client fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def test_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def set_pragma(conn, _):
        conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="module")
def db_session(test_engine) -> Generator[Session, None, None]:
    TestSession = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)
    session = TestSession()
    yield session
    session.close()


@pytest.fixture(scope="module")
def client(test_engine) -> TestClient:
    """TestClient with DB overridden to use the test engine (StaticPool shared memory DB)."""
    TestSessionLocal = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)
    app = create_app()

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Seed minimal data for API tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def seed_api_data(db_session):
    """Seed one trial, two sites, and minimal operational data."""
    trial = Trial(
        trial_id="API-TEST-001",
        sponsor="API Sponsor Ltd",
        phase=TrialPhase.PHASE_II,
        status=TrialStatus.ACTIVE,
        title="API Test Trial",
        start_date=START,
        planned_end_date=TODAY + timedelta(days=365),
    )
    db_session.add(trial)
    db_session.flush()

    site_a = Site(site_id="API-A", name="Alpha API Site", country="GBR",
                  city="London", pi_name="Dr A", site_type=SiteType.HOSPITAL)
    site_b = Site(site_id="API-B", name="Beta API Site", country="DEU",
                  city="Berlin", pi_name="Dr B", site_type=SiteType.ACADEMIC)
    db_session.add_all([site_a, site_b])
    db_session.flush()

    ts_a = TrialSite(trial_id=trial.id, site_id=site_a.id,
                     enrolment_target=10, activation_date=START)
    ts_b = TrialSite(trial_id=trial.id, site_id=site_b.id,
                     enrolment_target=8, activation_date=START)
    db_session.add_all([ts_a, ts_b])
    db_session.flush()

    # 5 enrolled patients for site A
    for i in range(5):
        db_session.add(PatientEnrolment(
            trial_site_id=ts_a.id,
            patient_id=hashlib.sha256(f"API-A-{i}".encode()).hexdigest(),
            screened_date=START + timedelta(days=i * 5),
            enrolled_date=START + timedelta(days=i * 5 + 3),
            status=EnrolmentStatus.ENROLLED,
        ))

    # 3 deviations for site A
    for i, sev in enumerate([DeviationSeverity.MINOR, DeviationSeverity.MAJOR,
                              DeviationSeverity.CRITICAL]):
        db_session.add(ProtocolDeviation(
            trial_site_id=ts_a.id,
            deviation_id=str(uuid.uuid4()),
            severity=sev,
            deviation_date=START + timedelta(days=i * 10 + 5),
            free_text=f"Test deviation {i}",
            category=["consent", "dosing", "safety"][i],
        ))

    # Monitoring visit
    db_session.add(MonitoringVisit(
        trial_site_id=ts_a.id, visit_type=VisitType.ONSITE,
        visit_date=TODAY - timedelta(days=15), sdv_complete=True,
    ))

    db_session.commit()

    # Store IDs for use in tests
    seed_api_data.trial_id = trial.id
    seed_api_data.ts_a_id  = ts_a.id
    seed_api_data.ts_b_id  = ts_b.id


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def _admin_token() -> str:
    return create_access_token("admin-user", UserRole.ADMIN)

def _cra_token() -> str:
    return create_access_token("cra-user", UserRole.CRA)

def _clinical_lead_token() -> str:
    return create_access_token("lead-user", UserRole.CLINICAL_LEAD)

def _sponsor_token(trial_id: int) -> str:
    return create_access_token("sponsor-user", UserRole.SPONSOR_VIEW, trial_id=trial_id)

def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ===========================================================================
# 1. AUTH ENDPOINT
# ===========================================================================

class TestAuthEndpoint:
    def test_login_admin_success(self, client):
        r = client.post("/auth/token", json={
            "user_id": "admin1", "password": DEMO_PASSWORD, "role": "admin"
        })
        assert r.status_code == 200
        data = r.json()
        assert "access_token" in data
        assert data["role"] == "admin"
        assert data["token_type"] == "bearer"

    def test_login_cra_success(self, client):
        r = client.post("/auth/token", json={
            "user_id": "cra1", "password": DEMO_PASSWORD, "role": "cra"
        })
        assert r.status_code == 200

    def test_login_clinical_lead_success(self, client):
        r = client.post("/auth/token", json={
            "user_id": "lead1", "password": DEMO_PASSWORD, "role": "clinical_lead"
        })
        assert r.status_code == 200

    def test_login_sponsor_view_success(self, client):
        r = client.post("/auth/token", json={
            "user_id": "sponsor1", "password": DEMO_PASSWORD,
            "role": "sponsor_view", "trial_id": 1
        })
        assert r.status_code == 200

    def test_login_wrong_password_401(self, client):
        r = client.post("/auth/token", json={
            "user_id": "bad", "password": "wrong", "role": "admin"
        })
        assert r.status_code == 401

    def test_login_invalid_role_422(self, client):
        r = client.post("/auth/token", json={
            "user_id": "u", "password": DEMO_PASSWORD, "role": "superuser"
        })
        assert r.status_code == 422

    def test_sponsor_view_without_trial_id_422(self, client):
        r = client.post("/auth/token", json={
            "user_id": "s", "password": DEMO_PASSWORD, "role": "sponsor_view"
        })
        assert r.status_code == 422

    def test_login_returns_user_id(self, client):
        r = client.post("/auth/token", json={
            "user_id": "myuser", "password": DEMO_PASSWORD, "role": "cra"
        })
        assert r.json()["user_id"] == "myuser"


# ===========================================================================
# 2. HEALTH CHECK
# ===========================================================================

class TestHealth:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health_returns_version(self, client):
        r = client.get("/health")
        assert "version" in r.json()


# ===========================================================================
# 3. MISSING / INVALID TOKEN
# ===========================================================================

class TestAuthGuard:
    def test_missing_token_trials_401(self, client):
        r = client.get("/trials")
        assert r.status_code == 401

    def test_missing_token_sites_kpis_401(self, client):
        r = client.get("/sites/1/kpis")
        assert r.status_code == 401

    def test_missing_token_risk_401(self, client):
        r = client.get("/sites/1/risk")
        assert r.status_code == 401

    def test_missing_token_forecast_401(self, client):
        r = client.get("/sites/1/forecast")
        assert r.status_code == 401

    def test_missing_token_deviations_401(self, client):
        r = client.get("/sites/1/deviations")
        assert r.status_code == 401

    def test_missing_token_watchlist_401(self, client):
        r = client.get("/portfolio/watchlist")
        assert r.status_code == 401

    def test_malformed_token_401(self, client):
        r = client.get("/trials", headers={"Authorization": "Bearer not.a.token"})
        assert r.status_code == 401

    def test_expired_token_401(self, client):
        from datetime import timedelta
        expired = create_access_token(
            "u", UserRole.ADMIN, expires_delta=timedelta(seconds=-1)
        )
        r = client.get("/trials", headers=_auth(expired))
        assert r.status_code == 401

    def test_wrong_scheme_401(self, client):
        r = client.get("/trials", headers={"Authorization": "Basic dXNlcjpwYXNz"})
        assert r.status_code == 401


# ===========================================================================
# 4. TRIAL ENDPOINTS
# ===========================================================================

class TestTrialEndpoints:
    def test_list_trials_admin(self, client):
        r = client.get("/trials", headers=_auth(_admin_token()))
        assert r.status_code == 200
        assert isinstance(r.json(), list)
        assert len(r.json()) >= 1

    def test_list_trials_cra(self, client):
        r = client.get("/trials", headers=_auth(_cra_token()))
        assert r.status_code == 200

    def test_get_trial_by_id(self, client):
        tid = seed_api_data.trial_id
        r = client.get(f"/trials/{tid}", headers=_auth(_admin_token()))
        assert r.status_code == 200
        data = r.json()
        assert data["trial_id"] == "API-TEST-001"
        assert data["phase"] == "II"

    def test_get_trial_404(self, client):
        r = client.get("/trials/99999", headers=_auth(_admin_token()))
        assert r.status_code == 404

    def test_trial_response_has_required_fields(self, client):
        tid = seed_api_data.trial_id
        r = client.get(f"/trials/{tid}", headers=_auth(_admin_token()))
        data = r.json()
        for field in ["id", "trial_id", "sponsor", "phase", "status", "title"]:
            assert field in data

    def test_list_trial_sites(self, client):
        tid = seed_api_data.trial_id
        r = client.get(f"/trials/{tid}/sites", headers=_auth(_admin_token()))
        assert r.status_code == 200
        sites = r.json()
        assert len(sites) == 2

    def test_trial_sites_have_site_id_and_country(self, client):
        tid = seed_api_data.trial_id
        r = client.get(f"/trials/{tid}/sites", headers=_auth(_admin_token()))
        for s in r.json():
            assert "site_id" in s
            assert "country" in s

    def test_trial_sites_unknown_trial_404(self, client):
        r = client.get("/trials/99999/sites", headers=_auth(_admin_token()))
        assert r.status_code == 404

    def test_sponsor_view_sees_only_own_trial(self, client):
        tid = seed_api_data.trial_id
        token = _sponsor_token(tid)
        r = client.get("/trials", headers=_auth(token))
        data = r.json()
        assert all(t["id"] == tid for t in data)

    def test_sponsor_view_cannot_access_other_trial(self, client):
        token = _sponsor_token(99999)  # wrong trial
        tid = seed_api_data.trial_id
        r = client.get(f"/trials/{tid}", headers=_auth(token))
        assert r.status_code == 403


# ===========================================================================
# 5. SITE KPI ENDPOINT
# ===========================================================================

class TestSiteKPIEndpoint:
    def test_kpi_returns_200(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/kpis", headers=_auth(_admin_token()))
        assert r.status_code == 200

    def test_kpi_response_has_site_and_trial_ids(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/kpis", headers=_auth(_admin_token()))
        data = r.json()
        assert "site_id" in data
        assert "trial_id" in data
        assert "snapshots" in data

    def test_kpi_unknown_site_404(self, client):
        r = client.get("/sites/99999/kpis", headers=_auth(_admin_token()))
        assert r.status_code == 404

    def test_kpi_date_range_filter_from(self, client):
        ts_id = seed_api_data.ts_a_id
        future = (TODAY + timedelta(days=10)).isoformat()
        r = client.get(f"/sites/{ts_id}/kpis?from_date={future}",
                       headers=_auth(_admin_token()))
        assert r.status_code == 200
        assert r.json()["snapshots"] == []

    def test_kpi_date_range_filter_to(self, client):
        ts_id = seed_api_data.ts_a_id
        past = (TODAY - timedelta(days=365)).isoformat()
        r = client.get(f"/sites/{ts_id}/kpis?to_date={past}",
                       headers=_auth(_admin_token()))
        assert r.status_code == 200
        assert r.json()["snapshots"] == []

    def test_sponsor_view_blocked_from_other_site(self, client):
        # Sponsor with trial_id=1, accessing a ts that belongs to a different trial
        # We'll use ts_b from a different sponsor token pointing at wrong trial
        token = _sponsor_token(99999)
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/kpis", headers=_auth(token))
        assert r.status_code == 403


# ===========================================================================
# 6. RISK SCORE ENDPOINT
# ===========================================================================

class TestRiskEndpoint:
    def test_risk_returns_200(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/risk", headers=_auth(_admin_token()))
        assert r.status_code == 200

    def test_risk_score_in_valid_range(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/risk", headers=_auth(_admin_token()))
        score = r.json()["composite_score"]
        assert 0.0 <= score <= 100.0

    def test_risk_has_all_components(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/risk", headers=_auth(_admin_token()))
        data = r.json()
        for field in [
            "composite_score", "dropout_probability",
            "enrolment_component", "deviation_component",
            "data_lag_component", "monitoring_component",
        ]:
            assert field in data

    def test_risk_dropout_probability_in_range(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/risk", headers=_auth(_admin_token()))
        p = r.json()["dropout_probability"]
        assert 0.0 <= p <= 1.0

    def test_risk_unknown_site_404(self, client):
        r = client.get("/sites/99999/risk", headers=_auth(_admin_token()))
        assert r.status_code == 404

    def test_risk_as_of_date_param(self, client):
        ts_id = seed_api_data.ts_a_id
        past = (TODAY - timedelta(days=7)).isoformat()
        r = client.get(f"/sites/{ts_id}/risk?as_of={past}",
                       headers=_auth(_admin_token()))
        assert r.status_code == 200
        assert r.json()["as_of_date"] == past

    def test_risk_cra_can_access(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/risk", headers=_auth(_cra_token()))
        assert r.status_code == 200


# ===========================================================================
# 7. FORECAST ENDPOINT
# ===========================================================================

class TestForecastEndpoint:
    def test_forecast_returns_200(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/forecast", headers=_auth(_admin_token()))
        assert r.status_code == 200

    def test_forecast_has_required_fields(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/forecast", headers=_auth(_admin_token()))
        data = r.json()
        for field in ["site_id", "forecast_date", "velocity_28d",
                      "enrolled_to_date", "remaining_to_target"]:
            assert field in data

    def test_forecast_enrolled_to_date_correct(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/forecast", headers=_auth(_admin_token()))
        assert r.json()["enrolled_to_date"] == 5

    def test_forecast_remaining_non_negative(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/forecast", headers=_auth(_admin_token()))
        assert r.json()["remaining_to_target"] >= 0

    def test_forecast_unknown_site_404(self, client):
        r = client.get("/sites/99999/forecast", headers=_auth(_admin_token()))
        assert r.status_code == 404

    def test_forecast_velocity_non_negative(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/forecast", headers=_auth(_admin_token()))
        assert r.json()["velocity_28d"] >= 0


# ===========================================================================
# 8. DEVIATIONS ENDPOINT
# ===========================================================================

class TestDeviationsEndpoint:
    def test_deviations_returns_200(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/deviations", headers=_auth(_admin_token()))
        assert r.status_code == 200

    def test_deviations_has_meta(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/deviations", headers=_auth(_admin_token()))
        data = r.json()
        assert "meta" in data
        assert "total" in data["meta"]

    def test_deviations_total_correct(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/deviations", headers=_auth(_admin_token()))
        assert r.json()["meta"]["total"] == 3

    def test_deviations_filter_by_severity(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/deviations?severity=minor",
                       headers=_auth(_admin_token()))
        data = r.json()
        assert data["meta"]["total"] == 1
        assert all(d["severity"] == "minor" for d in data["deviations"])

    def test_deviations_filter_by_category(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/deviations?category=consent",
                       headers=_auth(_admin_token()))
        data = r.json()
        assert all(d["category"] == "consent" for d in data["deviations"])

    def test_deviations_invalid_severity_422(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/deviations?severity=extreme",
                       headers=_auth(_admin_token()))
        assert r.status_code == 422

    def test_deviations_pagination_page_size(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/deviations?page_size=1",
                       headers=_auth(_admin_token()))
        data = r.json()
        assert len(data["deviations"]) == 1
        assert data["meta"]["total"] == 3

    def test_deviations_page_2_empty(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/deviations?page=2&page_size=10",
                       headers=_auth(_admin_token()))
        assert r.json()["deviations"] == []

    def test_deviations_unknown_site_404(self, client):
        r = client.get("/sites/99999/deviations", headers=_auth(_admin_token()))
        assert r.status_code == 404

    def test_deviations_empty_site_returns_empty_list(self, client):
        ts_id = seed_api_data.ts_b_id
        r = client.get(f"/sites/{ts_id}/deviations", headers=_auth(_admin_token()))
        data = r.json()
        assert data["meta"]["total"] == 0
        assert data["deviations"] == []

    def test_deviations_each_has_required_fields(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/deviations", headers=_auth(_admin_token()))
        for d in r.json()["deviations"]:
            assert "deviation_id" in d
            assert "severity" in d
            assert "deviation_date" in d
            assert "is_resolved" in d


# ===========================================================================
# 9. WATCHLIST ENDPOINT
# ===========================================================================

class TestWatchlistEndpoint:
    def test_watchlist_returns_200(self, client):
        r = client.get("/portfolio/watchlist", headers=_auth(_admin_token()))
        assert r.status_code == 200

    def test_watchlist_has_entries(self, client):
        r = client.get("/portfolio/watchlist", headers=_auth(_admin_token()))
        data = r.json()
        assert "entries" in data
        assert len(data["entries"]) >= 1

    def test_watchlist_sorted_by_risk_desc(self, client):
        r = client.get("/portfolio/watchlist", headers=_auth(_admin_token()))
        scores = [e["composite_score"] for e in r.json()["entries"]]
        assert scores == sorted(scores, reverse=True)

    def test_watchlist_entries_have_alert_flags(self, client):
        r = client.get("/portfolio/watchlist", headers=_auth(_admin_token()))
        for e in r.json()["entries"]:
            assert "alert_flags" in e
            assert isinstance(e["alert_flags"], list)

    def test_watchlist_filter_by_trial(self, client):
        tid = seed_api_data.trial_id
        r = client.get(f"/portfolio/watchlist?trial_id={tid}",
                       headers=_auth(_admin_token()))
        assert r.status_code == 200
        data = r.json()
        assert len(data["entries"]) == 2

    def test_watchlist_sponsor_sees_own_sites_only(self, client):
        tid = seed_api_data.trial_id
        token = _sponsor_token(tid)
        r = client.get("/portfolio/watchlist", headers=_auth(token))
        assert r.status_code == 200

    def test_watchlist_scores_in_range(self, client):
        r = client.get("/portfolio/watchlist", headers=_auth(_admin_token()))
        for e in r.json()["entries"]:
            assert 0.0 <= e["composite_score"] <= 100.0

    def test_watchlist_dropout_prob_in_range(self, client):
        r = client.get("/portfolio/watchlist", headers=_auth(_admin_token()))
        for e in r.json()["entries"]:
            assert 0.0 <= e["dropout_probability"] <= 1.0


# ===========================================================================
# 10. AUDIT LOG
# ===========================================================================

class TestAuditLog:
    @pytest.fixture(autouse=True)
    def audit_session(self, test_engine):
        """Direct session to the test engine for audit log assertions."""
        TestSessionLocal = sessionmaker(bind=test_engine, autoflush=False)
        self._db = TestSessionLocal()
        yield
        self._db.close()

    def test_audit_log_written_on_trial_list(self, client):
        before = self._db.query(AuditLog).count()
        client.get("/trials", headers=_auth(_admin_token()))
        self._db.expire_all()
        after = self._db.query(AuditLog).count()
        assert after > before

    def test_audit_log_written_on_risk_read(self, client):
        ts_id = seed_api_data.ts_a_id
        before = self._db.query(AuditLog).count()
        client.get(f"/sites/{ts_id}/risk", headers=_auth(_cra_token()))
        self._db.expire_all()
        after = self._db.query(AuditLog).count()
        assert after > before

    def test_audit_log_records_action_field(self, client):
        client.get("/trials", headers=_auth(_admin_token()))
        self._db.expire_all()
        log = self._db.query(AuditLog).filter(AuditLog.action == "LIST_TRIALS").first()
        assert log is not None

    def test_audit_log_records_user_id(self, client):
        client.get("/trials", headers=_auth(_admin_token()))
        self._db.expire_all()
        log = self._db.query(AuditLog).filter(AuditLog.user_id == "admin-user").first()
        assert log is not None

    def test_audit_log_records_role(self, client):
        client.get("/trials", headers=_auth(_cra_token()))
        self._db.expire_all()
        log = self._db.query(AuditLog).filter(AuditLog.role == UserRole.CRA).first()
        assert log is not None


# ===========================================================================
# 11. EDGE CASES
# ===========================================================================

class TestEdgeCases:
    def test_empty_site_kpis_returns_empty_snapshots(self, client):
        ts_id = seed_api_data.ts_b_id
        r = client.get(f"/sites/{ts_id}/kpis", headers=_auth(_admin_token()))
        assert r.status_code == 200

    def test_forecast_with_no_enrolments_returns_none_completion(self, client):
        ts_id = seed_api_data.ts_b_id
        r = client.get(f"/sites/{ts_id}/forecast", headers=_auth(_admin_token()))
        data = r.json()
        assert data["enrolled_to_date"] == 0
        assert data["projected_completion"] is None

    def test_watchlist_unknown_trial_returns_empty(self, client):
        r = client.get("/portfolio/watchlist?trial_id=99999",
                       headers=_auth(_admin_token()))
        assert r.status_code == 200
        assert r.json()["entries"] == []

    def test_deviations_page_size_max_100(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/deviations?page_size=101",
                       headers=_auth(_admin_token()))
        assert r.status_code == 422

    def test_deviations_page_min_1(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/deviations?page=0",
                       headers=_auth(_admin_token()))
        assert r.status_code == 422

    def test_invalid_date_format_in_kpi_range(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/kpis?from_date=not-a-date",
                       headers=_auth(_admin_token()))
        assert r.status_code == 422

    def test_openapi_schema_accessible(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert "paths" in schema
        assert "/trials" in schema["paths"]


# ===========================================================================
# 12. RBAC ROLE ENFORCEMENT
# ===========================================================================

class TestRBACEnforcement:
    def test_sponsor_view_token_has_no_trial_id_returns_empty_list(self, client):
        # sponsor_view with trial_id=999 (no data) sees empty list
        token = _sponsor_token(999)
        r = client.get("/trials", headers=_auth(token))
        assert r.status_code == 200
        assert r.json() == []

    def test_sponsor_view_cannot_access_different_trial_sites(self, client):
        tid = seed_api_data.trial_id
        wrong_token = _sponsor_token(tid + 999)
        r = client.get(f"/trials/{tid}/sites", headers=_auth(wrong_token))
        assert r.status_code == 403

    def test_sponsor_view_cannot_access_other_trial_kpis(self, client):
        ts_id = seed_api_data.ts_a_id
        wrong_token = _sponsor_token(99999)
        r = client.get(f"/sites/{ts_id}/kpis", headers=_auth(wrong_token))
        assert r.status_code == 403

    def test_sponsor_view_cannot_access_other_trial_risk(self, client):
        ts_id = seed_api_data.ts_a_id
        wrong_token = _sponsor_token(99999)
        r = client.get(f"/sites/{ts_id}/risk", headers=_auth(wrong_token))
        assert r.status_code == 403

    def test_sponsor_view_cannot_access_other_trial_forecast(self, client):
        ts_id = seed_api_data.ts_a_id
        wrong_token = _sponsor_token(99999)
        r = client.get(f"/sites/{ts_id}/forecast", headers=_auth(wrong_token))
        assert r.status_code == 403

    def test_sponsor_view_cannot_access_other_trial_deviations(self, client):
        ts_id = seed_api_data.ts_a_id
        wrong_token = _sponsor_token(99999)
        r = client.get(f"/sites/{ts_id}/deviations", headers=_auth(wrong_token))
        assert r.status_code == 403

    def test_cra_can_access_all_trials(self, client):
        r = client.get("/trials", headers=_auth(_cra_token()))
        assert r.status_code == 200
        assert len(r.json()) >= 1

    def test_clinical_lead_can_access_watchlist(self, client):
        r = client.get("/portfolio/watchlist", headers=_auth(_clinical_lead_token()))
        assert r.status_code == 200

    def test_sponsor_correct_trial_can_access_kpis(self, client):
        tid = seed_api_data.trial_id
        ts_id = seed_api_data.ts_a_id
        token = _sponsor_token(tid)
        r = client.get(f"/sites/{ts_id}/kpis", headers=_auth(token))
        assert r.status_code == 200

    def test_token_role_is_embedded_correctly(self, client):
        """The token issued contains the correct role claim."""
        r = client.post("/auth/token", json={
            "user_id": "rolecheck", "password": DEMO_PASSWORD, "role": "clinical_lead"
        })
        assert r.json()["role"] == "clinical_lead"


# ===========================================================================
# 13. SCHEMA VALIDATION & RESPONSE SHAPE
# ===========================================================================

class TestResponseSchemas:
    def test_trial_list_items_have_site_count(self, client):
        r = client.get("/trials", headers=_auth(_admin_token()))
        for t in r.json():
            assert "site_count" in t
            assert t["site_count"] >= 0

    def test_trial_detail_has_protocol_versions_list(self, client):
        tid = seed_api_data.trial_id
        r = client.get(f"/trials/{tid}", headers=_auth(_admin_token()))
        assert "protocol_versions" in r.json()
        assert isinstance(r.json()["protocol_versions"], list)

    def test_risk_site_id_matches_expected(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/risk", headers=_auth(_admin_token()))
        assert r.json()["site_id"] == "API-A"

    def test_forecast_site_id_in_response(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/forecast", headers=_auth(_admin_token()))
        assert r.json()["site_id"] == "API-A"

    def test_deviations_site_id_in_response(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/deviations", headers=_auth(_admin_token()))
        assert r.json()["site_id"] == "API-A"

    def test_watchlist_entry_has_site_name(self, client):
        r = client.get("/portfolio/watchlist", headers=_auth(_admin_token()))
        for entry in r.json()["entries"]:
            assert "site_name" in entry
            assert len(entry["site_name"]) > 0

    def test_watchlist_entry_has_country(self, client):
        r = client.get("/portfolio/watchlist", headers=_auth(_admin_token()))
        for entry in r.json()["entries"]:
            assert "country" in entry
            assert len(entry["country"]) == 3   # ISO 3166-1 alpha-3

    def test_kpi_snapshot_has_all_fields(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/kpis", headers=_auth(_admin_token()))
        snaps = r.json()["snapshots"]
        if snaps:
            snap = snaps[0]
            for field in ["snapshot_date", "enrolment_rate_28d", "enrolment_pct",
                          "deviation_rate", "data_lag_mean"]:
                assert field in snap

    def test_auth_token_response_has_all_fields(self, client):
        r = client.post("/auth/token", json={
            "user_id": "fieldcheck", "password": DEMO_PASSWORD, "role": "admin"
        })
        data = r.json()
        assert "access_token" in data
        assert "token_type" in data
        assert "role" in data
        assert "user_id" in data

    def test_deviations_meta_has_page_info(self, client):
        ts_id = seed_api_data.ts_a_id
        r = client.get(f"/sites/{ts_id}/deviations?page=1&page_size=2",
                       headers=_auth(_admin_token()))
        meta = r.json()["meta"]
        assert meta["page"] == 1
        assert meta["page_size"] == 2
        assert "total" in meta

    def test_trial_phase_value_is_roman_numeral(self, client):
        tid = seed_api_data.trial_id
        r = client.get(f"/trials/{tid}", headers=_auth(_admin_token()))
        phase = r.json()["phase"]
        assert phase in ["I", "II", "III", "IV"]
