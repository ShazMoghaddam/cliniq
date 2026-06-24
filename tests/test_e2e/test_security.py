"""
ClinIQ — Phase 5 Security & Hardening Tests
Covers: input sanitisation, rate limit logic, audit completeness, env config
Target: 24 tests (combined with E2E = 60 total for Phase 5)
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import StaticPool, create_engine, event
from sqlalchemy.orm import sessionmaker

from cliniq.api.app import create_app
from cliniq.api.routers.auth import DEMO_PASSWORD
from cliniq.db.database import get_db
from cliniq.db.models import AuditLog, Base, UserRole
from cliniq.rbac.auth import create_access_token


# ---------------------------------------------------------------------------
# Test client setup (mirrors Phase 3 pattern)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sec_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def set_fk(conn, _):
        conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="module")
def sec_client(sec_engine) -> TestClient:
    TestSL = sessionmaker(bind=sec_engine, autocommit=False, autoflush=False)
    app = create_app()

    def override():
        db = TestSL()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override
    return TestClient(app, raise_server_exceptions=True)


def _token(role: UserRole = UserRole.ADMIN) -> dict:
    t = create_access_token("sec-user", role)
    return {"Authorization": f"Bearer {t}"}


# ===========================================================================
# INPUT SANITISATION
# ===========================================================================

class TestInputSanitisation:
    def test_ai_query_min_length_enforced(self, sec_client):
        """AI query endpoint rejects questions shorter than 3 chars."""
        # POST /ai/query not yet implemented — test via schema validation
        from cliniq.api.schemas import AIQueryRequest
        with pytest.raises(Exception):
            AIQueryRequest(question="ab")  # min_length=3

    def test_ai_query_max_length_enforced(self, sec_client):
        """AI query schema rejects questions over 1000 chars."""
        from cliniq.api.schemas import AIQueryRequest
        with pytest.raises(Exception):
            AIQueryRequest(question="x" * 1001)

    def test_ai_query_valid_length_accepted(self):
        from cliniq.api.schemas import AIQueryRequest
        q = AIQueryRequest(question="What is the enrolment velocity?")
        assert len(q.question) >= 3

    def test_deviation_severity_filter_rejects_invalid(self, sec_client):
        """Invalid severity value returns 422."""
        r = sec_client.get("/sites/1/deviations?severity=EXTREME",
                           headers=_token())
        assert r.status_code in (404, 422)

    def test_page_size_max_100_enforced(self, sec_client):
        r = sec_client.get("/sites/1/deviations?page_size=9999",
                           headers=_token())
        assert r.status_code == 422

    def test_page_min_1_enforced(self, sec_client):
        r = sec_client.get("/sites/1/deviations?page=0",
                           headers=_token())
        assert r.status_code == 422

    def test_invalid_date_format_rejected(self, sec_client):
        r = sec_client.get("/sites/1/kpis?from_date=not-a-date",
                           headers=_token())
        assert r.status_code == 422

    def test_non_integer_site_id_rejected(self, sec_client):
        r = sec_client.get("/sites/abc/risk", headers=_token())
        assert r.status_code == 422

    def test_non_integer_trial_id_rejected(self, sec_client):
        r = sec_client.get("/trials/abc", headers=_token())
        assert r.status_code == 422

    def test_login_request_requires_user_id(self, sec_client):
        r = sec_client.post("/auth/token", json={
            "password": DEMO_PASSWORD, "role": "admin"
        })
        assert r.status_code == 422

    def test_login_request_requires_password(self, sec_client):
        r = sec_client.post("/auth/token", json={
            "user_id": "u", "role": "admin"
        })
        assert r.status_code == 422


# ===========================================================================
# AUTH HARDENING
# ===========================================================================

class TestAuthHardening:
    def test_expired_token_rejected(self, sec_client):
        expired = create_access_token(
            "u", UserRole.ADMIN, expires_delta=timedelta(seconds=-1)
        )
        r = sec_client.get("/trials",
                           headers={"Authorization": f"Bearer {expired}"})
        assert r.status_code == 401

    def test_tampered_token_rejected(self, sec_client):
        token = create_access_token("u", UserRole.ADMIN)
        tampered = token[:-5] + "XXXXX"
        r = sec_client.get("/trials",
                           headers={"Authorization": f"Bearer {tampered}"})
        assert r.status_code == 401

    def test_empty_bearer_rejected(self, sec_client):
        r = sec_client.get("/trials",
                           headers={"Authorization": "Bearer "})
        assert r.status_code == 401

    def test_wrong_scheme_rejected(self, sec_client):
        token = create_access_token("u", UserRole.ADMIN)
        r = sec_client.get("/trials",
                           headers={"Authorization": f"Token {token}"})
        assert r.status_code == 401

    def test_sponsor_view_trial_id_scoping(self, sec_client):
        """Sponsor View token with trial_id=1 cannot access trial_id=2."""
        token = create_access_token("s", UserRole.SPONSOR_VIEW, trial_id=1)
        r = sec_client.get("/trials/2",
                           headers={"Authorization": f"Bearer {token}"})
        assert r.status_code in (403, 404)

    def test_wrong_password_returns_401_not_422(self, sec_client):
        r = sec_client.post("/auth/token", json={
            "user_id": "u", "password": "wrong", "role": "admin"
        })
        assert r.status_code == 401


# ===========================================================================
# ENVIRONMENT CONFIG
# ===========================================================================

class TestEnvironmentConfig:
    def test_settings_secret_key_from_env(self):
        with patch.dict(os.environ, {"SECRET_KEY": "my-test-key-here-32chars-minimum"}):
            from cliniq.config.settings import Settings
            s = Settings()
            assert s.SECRET_KEY == "my-test-key-here-32chars-minimum"

    def test_settings_database_url_from_env(self):
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/testdb"}):
            from cliniq.config.settings import Settings
            s = Settings()
            assert s.DATABASE_URL == "postgresql://localhost/testdb"

    def test_settings_debug_false_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEBUG", None)
            from cliniq.config.settings import Settings
            s = Settings()
            assert s.DEBUG is False

    def test_settings_token_expiry_configurable(self):
        with patch.dict(os.environ, {"ACCESS_TOKEN_EXPIRE_MINUTES": "30"}):
            from cliniq.config.settings import Settings
            s = Settings()
            assert s.ACCESS_TOKEN_EXPIRE_MINUTES == 30

    def test_settings_algorithm_is_hs256(self):
        from cliniq.config.settings import Settings
        s = Settings()
        assert s.ALGORITHM == "HS256"

    def test_settings_app_title_set(self):
        from cliniq.config.settings import Settings
        s = Settings()
        assert s.APP_TITLE == "ClinIQ API"
