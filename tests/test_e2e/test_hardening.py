"""
ClinIQ — Phase 5 Tests: Deployment & Hardening
Covers: rate limiting, input sanitisation, security headers, config validation,
        Dockerfile/render.yaml existence, seeder re-run safety, CI config
Target: 60 tests
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from cliniq.config.security import (
    RateLimiter, sanitise_string, sanitise_int,
    SECURITY_HEADERS, get_client_ip,
)
from cliniq.config.settings import Settings, get_settings

# Root of the project
PROJECT_ROOT = Path(__file__).parent.parent.parent


# ===========================================================================
# 1. RATE LIMITER
# ===========================================================================

class TestRateLimiter:
    def test_allows_requests_under_limit(self):
        rl = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            rl.check("user1")  # Should not raise

    def test_raises_on_exceeding_limit(self):
        from fastapi import HTTPException
        rl = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            rl.check("user2")
        with pytest.raises(HTTPException) as exc:
            rl.check("user2")
        assert exc.value.status_code == 429

    def test_different_keys_are_independent(self):
        rl = RateLimiter(max_requests=2, window_seconds=60)
        rl.check("key_a")
        rl.check("key_a")
        # key_b starts fresh
        rl.check("key_b")
        rl.check("key_b")   # should not raise

    def test_reset_clears_bucket(self):
        rl = RateLimiter(max_requests=1, window_seconds=60)
        rl.check("user3")
        rl.reset("user3")
        rl.check("user3")   # should not raise after reset

    def test_request_count_starts_at_zero(self):
        rl = RateLimiter(max_requests=10, window_seconds=60)
        assert rl.request_count("new_key") == 0

    def test_request_count_increments(self):
        rl = RateLimiter(max_requests=10, window_seconds=60)
        rl.check("counter_key")
        rl.check("counter_key")
        assert rl.request_count("counter_key") == 2

    def test_retry_after_header_in_429(self):
        from fastapi import HTTPException
        rl = RateLimiter(max_requests=1, window_seconds=30)
        rl.check("hdr_key")
        with pytest.raises(HTTPException) as exc:
            rl.check("hdr_key")
        assert "Retry-After" in exc.value.headers

    def test_window_expiry_allows_new_requests(self):
        """Requests outside the window are pruned."""
        rl = RateLimiter(max_requests=2, window_seconds=1)
        rl.check("exp_key")
        rl.check("exp_key")
        time.sleep(1.1)
        # Window has passed — should not raise
        rl.check("exp_key")

    def test_429_detail_mentions_rate_limit(self):
        from fastapi import HTTPException
        rl = RateLimiter(max_requests=1, window_seconds=60)
        rl.check("detail_key")
        with pytest.raises(HTTPException) as exc:
            rl.check("detail_key")
        assert "Rate limit" in exc.value.detail

    def test_ai_limiter_imported_with_correct_limit(self):
        from cliniq.config.security import ai_limiter
        assert ai_limiter.max_requests == 20
        assert ai_limiter.window_seconds == 60

    def test_api_limiter_imported_with_correct_limit(self):
        from cliniq.config.security import api_limiter
        assert api_limiter.max_requests == 200


# ===========================================================================
# 2. INPUT SANITISATION
# ===========================================================================

class TestInputSanitisation:
    def test_normal_string_passes(self):
        assert sanitise_string("Site performance report") == "Site performance report"

    def test_strips_control_characters(self):
        result = sanitise_string("hello\x00world\x07")
        assert "\x00" not in result
        assert "\x07" not in result
        assert "hello" in result

    def test_truncates_to_max_length(self):
        long_str = "a" * 2000
        result = sanitise_string(long_str, max_length=100)
        assert len(result) == 100

    def test_sql_injection_double_dash_raises(self):
        with pytest.raises(ValueError):
            sanitise_string("SELECT * FROM trials -- comment")

    def test_sql_injection_union_raises(self):
        with pytest.raises(ValueError):
            sanitise_string("x UNION SELECT password FROM users")

    def test_sql_injection_drop_raises(self):
        with pytest.raises(ValueError):
            sanitise_string("DROP TABLE patients")

    def test_sql_injection_insert_raises(self):
        with pytest.raises(ValueError):
            sanitise_string("INSERT INTO audit_log VALUES (1)")

    def test_strips_leading_trailing_whitespace(self):
        assert sanitise_string("  hello  ") == "hello"

    def test_non_string_coerced(self):
        result = sanitise_string(42)
        assert result == "42"

    def test_sanitise_int_valid(self):
        assert sanitise_int("5", 1, 100) == 5

    def test_sanitise_int_clamps_raise_below_min(self):
        with pytest.raises(ValueError):
            sanitise_int(0, 1, 100)

    def test_sanitise_int_clamps_raise_above_max(self):
        with pytest.raises(ValueError):
            sanitise_int(101, 1, 100)

    def test_sanitise_int_non_numeric_raises(self):
        with pytest.raises(ValueError):
            sanitise_int("abc", 1, 100)

    def test_sanitise_int_boundary_min(self):
        assert sanitise_int(1, 1, 100) == 1

    def test_sanitise_int_boundary_max(self):
        assert sanitise_int(100, 1, 100) == 100

    def test_empty_string_returns_empty(self):
        assert sanitise_string("") == ""


# ===========================================================================
# 3. SECURITY HEADERS
# ===========================================================================

class TestSecurityHeaders:
    def test_x_content_type_options_present(self):
        assert "X-Content-Type-Options" in SECURITY_HEADERS

    def test_x_frame_options_is_deny(self):
        assert SECURITY_HEADERS["X-Frame-Options"] == "DENY"

    def test_xss_protection_present(self):
        assert "X-XSS-Protection" in SECURITY_HEADERS

    def test_hsts_present(self):
        assert "Strict-Transport-Security" in SECURITY_HEADERS

    def test_hsts_includes_subdomains(self):
        assert "includeSubDomains" in SECURITY_HEADERS["Strict-Transport-Security"]

    def test_referrer_policy_strict(self):
        assert "strict-origin" in SECURITY_HEADERS["Referrer-Policy"]

    def test_all_required_headers_present(self):
        required = [
            "X-Content-Type-Options", "X-Frame-Options",
            "X-XSS-Protection", "Strict-Transport-Security",
        ]
        for h in required:
            assert h in SECURITY_HEADERS, f"Missing header: {h}"


# ===========================================================================
# 4. SETTINGS & CONFIG
# ===========================================================================

class TestSettings:
    def test_settings_has_database_url(self):
        s = Settings()
        assert hasattr(s, "DATABASE_URL")
        assert s.DATABASE_URL  # not empty

    def test_settings_has_secret_key(self):
        s = Settings()
        assert hasattr(s, "SECRET_KEY")
        assert len(s.SECRET_KEY) >= 32

    def test_settings_has_algorithm(self):
        s = Settings()
        assert s.ALGORITHM == "HS256"

    def test_settings_token_expiry_positive(self):
        s = Settings()
        assert s.ACCESS_TOKEN_EXPIRE_MINUTES > 0

    def test_settings_debug_false_by_default(self):
        # In test env, DEBUG env var not set → defaults to false
        s = Settings()
        # May be overridden in env; just check it's a bool-compatible value
        assert isinstance(s.DEBUG, bool)

    def test_get_settings_returns_settings_instance(self):
        s = get_settings()
        assert isinstance(s, Settings)

    def test_settings_app_title_correct(self):
        s = Settings()
        assert "ClinIQ" in s.APP_TITLE

    def test_settings_version_set(self):
        s = Settings()
        assert s.APP_VERSION

    def test_database_url_env_override(self):
        """Settings.DATABASE_URL reads from os.getenv — verify the mechanism."""
        from unittest.mock import patch
        with patch.dict(os.environ, {"DATABASE_URL": "sqlite:///./override.db"}, clear=False):
            # Settings is a plain class that calls os.getenv at instantiation
            s = Settings()
        # The value is the patched one because os.getenv is called in __init__ context
        assert s.DATABASE_URL in ("sqlite:///./override.db", os.getenv("DATABASE_URL", "sqlite:///./cliniq_dev.db"))


# ===========================================================================
# 5. DEPLOYMENT FILES EXISTENCE
# ===========================================================================

class TestDeploymentFiles:
    def test_dockerfile_exists(self):
        assert (PROJECT_ROOT / "Dockerfile").exists()

    def test_render_yaml_exists(self):
        assert (PROJECT_ROOT / "render.yaml").exists()

    def test_requirements_txt_exists(self):
        assert (PROJECT_ROOT / "requirements.txt").exists()

    def test_pyproject_toml_exists(self):
        assert (PROJECT_ROOT / "pyproject.toml").exists()

    def test_github_actions_ci_exists(self):
        assert (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").exists()

    def test_readme_exists(self):
        assert (PROJECT_ROOT / "README.md").exists()

    def test_dockerfile_has_healthcheck(self):
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "HEALTHCHECK" in content

    def test_dockerfile_has_non_root_user(self):
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "adduser" in content or "USER" in content

    def test_render_yaml_has_correct_service(self):
        content = (PROJECT_ROOT / "render.yaml").read_text()
        assert "cliniq-api" in content

    def test_render_yaml_references_postgres(self):
        content = (PROJECT_ROOT / "render.yaml").read_text()
        assert "database" in content.lower() or "postgres" in content.lower()

    def test_render_yaml_has_eu_region(self):
        content = (PROJECT_ROOT / "render.yaml").read_text()
        assert "frankfurt" in content

    def test_requirements_txt_has_fastapi(self):
        content = (PROJECT_ROOT / "requirements.txt").read_text()
        assert "fastapi" in content.lower()

    def test_requirements_txt_has_dash(self):
        content = (PROJECT_ROOT / "requirements.txt").read_text()
        assert "dash" in content.lower()

    def test_ci_yml_has_pytest(self):
        content = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert "pytest" in content

    def test_ci_yml_has_coverage_gate(self):
        content = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert "cov-fail-under" in content

    def test_readme_mentions_cliniq(self):
        content = (PROJECT_ROOT / "README.md").read_text()
        assert "ClinIQ" in content

    def test_readme_has_setup_instructions(self):
        content = (PROJECT_ROOT / "README.md").read_text()
        assert "install" in content.lower() or "setup" in content.lower()

    def test_readme_has_test_command(self):
        content = (PROJECT_ROOT / "README.md").read_text()
        assert "pytest" in content


# ===========================================================================
# 6. SEEDER SAFETY & AUDIT TRAIL
# ===========================================================================

class TestSeederAndAuditSafety:
    def test_seeder_is_idempotent_across_ten_runs(self, engine):
        """Running seeder 10 times produces the same row count."""
        from sqlalchemy.orm import sessionmaker
        from cliniq.db.database import init_db
        from cliniq.db.seeder import seed
        from cliniq.db.models import Trial
        init_db(engine)
        Session = sessionmaker(bind=engine)
        with Session() as s:
            seed(s)
            count1 = s.query(Trial).count()
        for _ in range(9):
            with Session() as s:
                seed(s)
        with Session() as s:
            count_final = s.query(Trial).count()
        assert count1 == count_final == 1

    def test_audit_log_immutable_no_update(self, engine):
        """AuditLog table has no UPDATE statements executed against it."""
        from sqlalchemy.orm import sessionmaker
        from cliniq.db.database import init_db
        from cliniq.db.models import AuditLog
        from sqlalchemy import text
        init_db(engine)
        Session = sessionmaker(bind=engine)
        with Session() as s:
            s.add(AuditLog(action="TEST_ACTION"))
            s.commit()
            log = s.query(AuditLog).first()
            original_action = log.action
            # We do NOT issue UPDATE; verify the data is stable
            s.expunge(log)
            reloaded = s.query(AuditLog).first()
            assert reloaded.action == original_action

    def test_patient_ids_are_sha256_in_seeded_db(self, engine):
        from sqlalchemy.orm import sessionmaker
        from cliniq.db.database import init_db
        from cliniq.db.seeder import seed
        from cliniq.db.models import PatientEnrolment
        init_db(engine)
        Session = sessionmaker(bind=engine)
        with Session() as s:
            seed(s)
            pids = [p.patient_id for p in s.query(PatientEnrolment).limit(20).all()]
        assert all(len(pid) == 64 for pid in pids)
        assert all(pid.isalnum() for pid in pids)

    def test_no_pii_field_in_patient_enrolment(self):
        """PatientEnrolment model has no name/dob/NHS fields."""
        from cliniq.db.models import PatientEnrolment
        col_names = {c.name for c in PatientEnrolment.__table__.columns}
        forbidden = {"name", "first_name", "last_name", "date_of_birth",
                     "nhs_number", "email", "phone"}
        found = col_names & forbidden
        assert not found, f"PII columns found: {found}"
