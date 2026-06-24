"""
ClinIQ — Phase 5 E2E Smoke Tests
Browser-level tests via Playwright against a live in-process Dash server.
"""
from __future__ import annotations

import hashlib
import threading
import time
import uuid
from datetime import date, timedelta

import pytest
from playwright.sync_api import Page, sync_playwright

SERVER_PORT = 18050
SERVER_URL  = f"http://127.0.0.1:{SERVER_PORT}"
_server_started = False


def _seed_test_db():
    from cliniq.db.database import SessionLocal, init_db
    from cliniq.db.models import (
        DeviationSeverity, EnrolmentStatus, MonitoringVisit,
        PatientEnrolment, ProtocolDeviation, Site, SiteType,
        Trial, TrialPhase, TrialSite, TrialStatus, VisitType,
    )
    init_db()
    with SessionLocal() as db:
        if db.query(Trial).first():
            return
        t = Trial(
            trial_id="E2E-001", sponsor="E2E Sponsor",
            phase=TrialPhase.PHASE_II, status=TrialStatus.ACTIVE,
            title="E2E Smoke Test Trial",
            start_date=date.today() - timedelta(days=90),
            planned_end_date=date.today() + timedelta(days=365),
        )
        db.add(t); db.flush()
        site = Site(site_id="E2E-A", name="E2E Hospital", country="GBR",
                    city="London", pi_name="Dr E2E", site_type=SiteType.HOSPITAL)
        db.add(site); db.flush()
        ts = TrialSite(trial_id=t.id, site_id=site.id, enrolment_target=10,
                       activation_date=date.today() - timedelta(days=60))
        db.add(ts); db.flush()
        for i in range(5):
            db.add(PatientEnrolment(
                trial_site_id=ts.id,
                patient_id=hashlib.sha256(f"e2e-{i}".encode()).hexdigest(),
                screened_date=date.today() - timedelta(days=40 - i*5),
                enrolled_date=date.today() - timedelta(days=35 - i*5),
                status=EnrolmentStatus.ENROLLED,
            ))
        for i, sev in enumerate([DeviationSeverity.MINOR, DeviationSeverity.MAJOR,
                                  DeviationSeverity.CRITICAL]):
            db.add(ProtocolDeviation(
                trial_site_id=ts.id, deviation_id=str(uuid.uuid4()),
                severity=sev, deviation_date=date.today() - timedelta(days=10+i*5),
                free_text=f"E2E deviation {i}", category=["consent","dosing","safety"][i],
            ))
        db.add(MonitoringVisit(
            trial_site_id=ts.id, visit_type=VisitType.ONSITE,
            visit_date=date.today()-timedelta(days=20), sdv_complete=True,
        ))
        db.commit()


def _get_ts_id():
    from cliniq.db.database import SessionLocal
    from cliniq.db.models import Trial
    with SessionLocal() as db:
        t = db.query(Trial).filter_by(trial_id="E2E-001").first()
        return t.trial_sites[0].id if t and t.trial_sites else 1


@pytest.fixture(scope="module")
def live_server():
    global _server_started
    _seed_test_db()
    if not _server_started:
        from cliniq.dashboard.app import create_dashboard
        dash_app = create_dashboard()
        flask_srv = dash_app.server

        def run():
            flask_srv.run(host="127.0.0.1", port=SERVER_PORT,
                          debug=False, use_reloader=False)

        threading.Thread(target=run, daemon=True).start()
        import urllib.request
        for _ in range(50):
            try:
                urllib.request.urlopen(f"{SERVER_URL}/", timeout=0.5)
                break
            except Exception:
                time.sleep(0.2)
        _server_started = True
    yield SERVER_URL


@pytest.fixture(scope="module")
def browser_page(live_server):
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx  = browser.new_context()
        page = ctx.new_page()
        yield page
        ctx.close()
        browser.close()


def _wait(page: Page):
    page.wait_for_load_state("networkidle", timeout=10000)
    page.wait_for_timeout(700)


# ===========================================================================
# SERVER HEALTH
# ===========================================================================

class TestServerHealth:
    def test_server_responds_200(self, live_server):
        import urllib.request
        r = urllib.request.urlopen(f"{live_server}/")
        assert r.status == 200

    def test_server_returns_html(self, live_server):
        import urllib.request
        content = urllib.request.urlopen(f"{live_server}/").read().decode()
        assert "<html" in content or "<!DOCTYPE" in content

    def test_server_url_has_port(self, live_server):
        assert str(SERVER_PORT) in live_server


# ===========================================================================
# PORTFOLIO VIEW
# ===========================================================================

class TestPortfolioView:
    def test_portfolio_loads(self, browser_page):
        browser_page.goto(SERVER_URL + "/")
        _wait(browser_page)
        assert "ClinIQ" in browser_page.content() or "Portfolio" in browser_page.content()

    def test_portfolio_sidebar_present(self, browser_page):
        browser_page.goto(SERVER_URL + "/")
        _wait(browser_page)
        assert "Watchlist" in browser_page.content()

    def test_portfolio_dropdown_in_dom(self, browser_page):
        browser_page.goto(SERVER_URL + "/")
        _wait(browser_page)
        assert browser_page.locator("#portfolio-trial-select").count() > 0

    def test_portfolio_heatmap_in_dom(self, browser_page):
        browser_page.goto(SERVER_URL + "/")
        _wait(browser_page)
        assert browser_page.locator("#portfolio-heatmap-chart").count() > 0

    def test_portfolio_no_traceback(self, browser_page):
        browser_page.goto(SERVER_URL + "/")
        _wait(browser_page)
        assert "Traceback" not in browser_page.content()

    def test_portfolio_title_tag(self, browser_page):
        browser_page.goto(SERVER_URL + "/")
        _wait(browser_page)
        assert "ClinIQ" in browser_page.title()


# ===========================================================================
# WATCHLIST VIEW
# ===========================================================================

class TestWatchlistView:
    def test_watchlist_loads(self, browser_page):
        browser_page.goto(SERVER_URL + "/watchlist")
        _wait(browser_page)
        assert "Watchlist" in browser_page.content()

    def test_watchlist_dropdown_present(self, browser_page):
        browser_page.goto(SERVER_URL + "/watchlist")
        _wait(browser_page)
        assert browser_page.locator("#watchlist-trial-select").count() > 0

    def test_watchlist_url_correct(self, browser_page):
        browser_page.goto(SERVER_URL + "/watchlist")
        _wait(browser_page)
        assert "/watchlist" in browser_page.url

    def test_watchlist_no_traceback(self, browser_page):
        browser_page.goto(SERVER_URL + "/watchlist")
        _wait(browser_page)
        assert "Traceback" not in browser_page.content()

    def test_watchlist_sidebar_ai_link_visible(self, browser_page):
        browser_page.goto(SERVER_URL + "/watchlist")
        _wait(browser_page)
        assert "AI Assistant" in browser_page.content()


# ===========================================================================
# SITE DRILL-DOWN
# ===========================================================================

class TestSiteDrillDown:
    def test_site_drilldown_loads(self, browser_page):
        ts_id = _get_ts_id()
        browser_page.goto(f"{SERVER_URL}/site/{ts_id}")
        _wait(browser_page)
        content = browser_page.content()
        assert "Drill" in content or "Site" in content or "Risk" in content

    def test_site_enrolment_chart_in_dom(self, browser_page):
        ts_id = _get_ts_id()
        browser_page.goto(f"{SERVER_URL}/site/{ts_id}")
        _wait(browser_page)
        assert browser_page.locator("#site-enrolment-chart").count() > 0

    def test_site_lag_chart_in_dom(self, browser_page):
        ts_id = _get_ts_id()
        browser_page.goto(f"{SERVER_URL}/site/{ts_id}")
        _wait(browser_page)
        assert browser_page.locator("#site-lag-chart").count() > 0

    def test_site_deviation_chart_in_dom(self, browser_page):
        ts_id = _get_ts_id()
        browser_page.goto(f"{SERVER_URL}/site/{ts_id}")
        _wait(browser_page)
        assert browser_page.locator("#site-deviation-chart").count() > 0

    def test_site_risk_breakdown_chart_in_dom(self, browser_page):
        ts_id = _get_ts_id()
        browser_page.goto(f"{SERVER_URL}/site/{ts_id}")
        _wait(browser_page)
        assert browser_page.locator("#site-risk-breakdown-chart").count() > 0

    def test_site_invalid_id_no_crash(self, browser_page):
        browser_page.goto(f"{SERVER_URL}/site/99999")
        _wait(browser_page)
        assert "Traceback" not in browser_page.content()
        assert "Internal Server Error" not in browser_page.content()


# ===========================================================================
# DEVIATION LOG
# ===========================================================================

class TestDeviationLogView:
    def test_deviation_log_loads(self, browser_page):
        ts_id = _get_ts_id()
        browser_page.goto(f"{SERVER_URL}/deviations/{ts_id}")
        _wait(browser_page)
        assert "Deviation" in browser_page.content()

    def test_deviation_severity_filter_in_dom(self, browser_page):
        ts_id = _get_ts_id()
        browser_page.goto(f"{SERVER_URL}/deviations/{ts_id}")
        _wait(browser_page)
        assert browser_page.locator("#dev-severity-filter").count() > 0

    def test_deviation_category_filter_in_dom(self, browser_page):
        ts_id = _get_ts_id()
        browser_page.goto(f"{SERVER_URL}/deviations/{ts_id}")
        _wait(browser_page)
        assert browser_page.locator("#dev-category-filter").count() > 0

    def test_deviation_log_no_traceback(self, browser_page):
        ts_id = _get_ts_id()
        browser_page.goto(f"{SERVER_URL}/deviations/{ts_id}")
        _wait(browser_page)
        assert "Traceback" not in browser_page.content()


# ===========================================================================
# AI ASSISTANT
# ===========================================================================

class TestAssistantView:
    def test_assistant_loads(self, browser_page):
        browser_page.goto(SERVER_URL + "/assistant")
        _wait(browser_page)
        assert "Assistant" in browser_page.content() or "assistant" in browser_page.content()

    def test_assistant_input_in_dom(self, browser_page):
        browser_page.goto(SERVER_URL + "/assistant")
        _wait(browser_page)
        assert browser_page.locator("#assistant-input").count() > 0

    def test_assistant_submit_button_in_dom(self, browser_page):
        browser_page.goto(SERVER_URL + "/assistant")
        _wait(browser_page)
        assert browser_page.locator("#assistant-submit").count() > 0

    def test_assistant_trial_dropdown_in_dom(self, browser_page):
        browser_page.goto(SERVER_URL + "/assistant")
        _wait(browser_page)
        assert browser_page.locator("#assistant-trial-select").count() > 0

    def test_assistant_no_traceback(self, browser_page):
        browser_page.goto(SERVER_URL + "/assistant")
        _wait(browser_page)
        assert "Traceback" not in browser_page.content()


# ===========================================================================
# URL ROUTING
# ===========================================================================

class TestURLRouting:
    def test_root_shows_portfolio(self, browser_page):
        browser_page.goto(SERVER_URL + "/")
        _wait(browser_page)
        assert browser_page.locator("#portfolio-heatmap-chart").count() > 0

    def test_unknown_path_no_crash(self, browser_page):
        browser_page.goto(SERVER_URL + "/this-does-not-exist")
        _wait(browser_page)
        assert "Traceback" not in browser_page.content()

    def test_watchlist_path_sets_url(self, browser_page):
        browser_page.goto(SERVER_URL + "/watchlist")
        _wait(browser_page)
        assert "/watchlist" in browser_page.url

    def test_sidebar_watchlist_link_navigates(self, browser_page):
        browser_page.goto(SERVER_URL + "/")
        _wait(browser_page)
        link = browser_page.locator("a[href='/watchlist']").first
        link.click()
        _wait(browser_page)
        assert "/watchlist" in browser_page.url


# ===========================================================================
# SECURITY HARDENING
# ===========================================================================

class TestSecurity:
    def test_no_secret_key_in_source(self, browser_page):
        browser_page.goto(SERVER_URL + "/")
        _wait(browser_page)
        assert "dev-secret-key" not in browser_page.content()

    def test_no_stack_traces_on_main_views(self, browser_page):
        for path in ["/", "/watchlist", "/assistant"]:
            browser_page.goto(SERVER_URL + path)
            _wait(browser_page)
            assert "Traceback" not in browser_page.content(), f"Traceback found at {path}"

    def test_page_title_set(self, browser_page):
        browser_page.goto(SERVER_URL + "/")
        _wait(browser_page)
        assert "ClinIQ" in browser_page.title()
