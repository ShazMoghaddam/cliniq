"""
ClinIQ — Phase 2 Tests: Analytics Engine
Covers: enrolment velocity, screening funnel, data lag
"""
import hashlib
from datetime import date, timedelta

import pytest

from cliniq.db.models import (
    DataEntryEvent, EnrolmentStatus, MonitoringVisit, PatientEnrolment,
    QueryLog, TrialSite, VisitType,
)
from cliniq.analytics.velocity import compute_velocity, compute_velocity_all_sites
from cliniq.analytics.funnel import compute_funnel, compute_funnel_by_country
from cliniq.analytics.lag import compute_lag

TODAY = date.today()
START = TODAY - timedelta(days=30)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_enrolment(session, ts_id, days_ago_screened, days_ago_enrolled=None,
                   status=EnrolmentStatus.ENROLLED, suffix=""):
    screened = TODAY - timedelta(days=days_ago_screened)
    enrolled = TODAY - timedelta(days=days_ago_enrolled) if days_ago_enrolled is not None else None
    raw = f"{ts_id}-{days_ago_screened}-{suffix}"
    pe = PatientEnrolment(
        trial_site_id=ts_id,
        patient_id=hashlib.sha256(raw.encode()).hexdigest(),
        screened_date=screened,
        enrolled_date=enrolled,
        status=status,
    )
    session.add(pe)
    session.flush()
    return pe


def _add_lag(session, ts_id, visit_days_ago, lag):
    visit = TODAY - timedelta(days=visit_days_ago)
    de = DataEntryEvent(
        trial_site_id=ts_id,
        visit_date=visit,
        entry_date=visit + timedelta(days=lag),
        lag_days=lag,
    )
    session.add(de)
    session.flush()
    return de


# ===========================================================================
# VELOCITY TESTS
# ===========================================================================

class TestEnrolmentVelocity:
    def test_zero_enrolments_gives_zero_velocity(self, session, trial_sites):
        result = compute_velocity(session, trial_sites[0].id)
        assert result.velocity_28d == 0.0
        assert result.enrolled_to_date == 0

    def test_single_enrolment_in_window_counts(self, session, trial_sites):
        _add_enrolment(session, trial_sites[0].id, days_ago_screened=10, days_ago_enrolled=5)
        result = compute_velocity(session, trial_sites[0].id)
        assert result.enrolled_to_date == 1
        assert result.velocity_28d == round(1 / 28.0, 4)

    def test_enrolment_outside_28d_window_not_in_velocity(self, session, trial_sites):
        # Enrolled 30 days ago — outside the 28-day window but still in enrolled_to_date
        _add_enrolment(session, trial_sites[0].id, days_ago_screened=35, days_ago_enrolled=30)
        result = compute_velocity(session, trial_sites[0].id)
        assert result.enrolled_to_date == 1
        assert result.velocity_28d == 0.0   # not in 28d window

    def test_multiple_enrolments_sum_correctly(self, session, trial_sites):
        for i in range(5):
            _add_enrolment(session, trial_sites[0].id,
                           days_ago_screened=i + 10, days_ago_enrolled=i + 2,
                           suffix=str(i))
        result = compute_velocity(session, trial_sites[0].id)
        assert result.enrolled_to_date == 5
        assert result.velocity_28d == round(5 / 28.0, 4)

    def test_screen_fails_excluded_from_velocity(self, session, trial_sites):
        _add_enrolment(session, trial_sites[0].id, days_ago_screened=5,
                       status=EnrolmentStatus.SCREEN_FAIL, suffix="sf")
        result = compute_velocity(session, trial_sites[0].id)
        assert result.enrolled_to_date == 0

    def test_withdrawn_patients_counted_in_enrolled(self, session, trial_sites):
        _add_enrolment(session, trial_sites[0].id, days_ago_screened=10,
                       days_ago_enrolled=5,
                       status=EnrolmentStatus.WITHDRAWN, suffix="w")
        result = compute_velocity(session, trial_sites[0].id)
        assert result.enrolled_to_date == 1

    def test_target_met_gives_zero_remaining(self, session, trial_sites):
        ts = trial_sites[0]  # target = 10
        for i in range(10):
            _add_enrolment(session, ts.id, days_ago_screened=i + 10,
                           days_ago_enrolled=i + 2, suffix=str(i))
        result = compute_velocity(session, ts.id)
        assert result.remaining_to_target == 0
        assert result.projected_completion == TODAY

    def test_no_velocity_gives_no_projected_completion(self, session, trial_sites):
        # No enrolments at all
        result = compute_velocity(session, trial_sites[0].id)
        assert result.projected_completion is None

    def test_invalid_trial_site_raises(self, session):
        with pytest.raises(ValueError):
            compute_velocity(session, 99999)

    def test_compute_all_sites_returns_correct_count(self, session, trial, trial_sites):
        results = compute_velocity_all_sites(session, trial.id)
        assert len(results) == len(trial_sites)

    def test_as_of_date_respected(self, session, trial_sites):
        # Enrolment 15 days ago; if we ask as_of 20 days ago it shouldn't count
        _add_enrolment(session, trial_sites[0].id,
                       days_ago_screened=20, days_ago_enrolled=15, suffix="past")
        past_date = TODAY - timedelta(days=20)
        result = compute_velocity(session, trial_sites[0].id, as_of_date=past_date)
        assert result.enrolled_to_date == 0

    def test_velocity_28d_is_non_negative(self, session, trial_sites):
        result = compute_velocity(session, trial_sites[0].id)
        assert result.velocity_28d >= 0

    def test_remaining_never_negative(self, session, trial_sites):
        ts = trial_sites[0]
        # Enrol more than target
        for i in range(ts.enrolment_target + 3):
            _add_enrolment(session, ts.id, days_ago_screened=i + 5,
                           days_ago_enrolled=i + 1, suffix=str(i))
        result = compute_velocity(session, ts.id)
        assert result.remaining_to_target == 0


# ===========================================================================
# SCREENING FUNNEL TESTS
# ===========================================================================

class TestScreeningFunnel:
    def test_empty_site_all_zeros(self, session, trial_sites):
        result = compute_funnel(session, trial_sites[1].id)
        assert result.screened == 0
        assert result.screen_fail_rate == 0.0
        assert result.enrolment_conversion == 0.0

    def test_screen_fail_rate_correct(self, session, trial_sites):
        ts = trial_sites[0]
        for i in range(7):
            _add_enrolment(session, ts.id, days_ago_screened=i + 5,
                           days_ago_enrolled=i + 1, suffix=f"e{i}")
        for j in range(3):
            _add_enrolment(session, ts.id, days_ago_screened=j + 1,
                           status=EnrolmentStatus.SCREEN_FAIL, suffix=f"sf{j}")
        result = compute_funnel(session, ts.id)
        assert result.screened == 10
        assert result.screen_fail_rate == pytest.approx(0.3, rel=1e-3)
        assert result.enrolment_conversion == pytest.approx(0.7, rel=1e-3)

    def test_all_screen_failures(self, session, trial_sites):
        ts = trial_sites[0]
        for i in range(5):
            _add_enrolment(session, ts.id, days_ago_screened=i + 1,
                           status=EnrolmentStatus.SCREEN_FAIL, suffix=str(i))
        result = compute_funnel(session, ts.id)
        assert result.screen_fail_rate == 1.0
        assert result.enrolment_conversion == 0.0
        assert result.withdrawal_rate == 0.0

    def test_withdrawal_rate_correct(self, session, trial_sites):
        ts = trial_sites[0]
        for i in range(8):
            _add_enrolment(session, ts.id, days_ago_screened=i + 5,
                           days_ago_enrolled=i + 1, suffix=f"e{i}")
        for j in range(2):
            _add_enrolment(session, ts.id, days_ago_screened=j + 20,
                           days_ago_enrolled=j + 15,
                           status=EnrolmentStatus.WITHDRAWN, suffix=f"w{j}")
        result = compute_funnel(session, ts.id)
        assert result.enrolled == 8
        assert result.withdrawn == 2
        assert result.withdrawal_rate == pytest.approx(2 / 8, rel=1e-3)

    def test_invalid_site_raises(self, session):
        with pytest.raises(ValueError):
            compute_funnel(session, 99999)

    def test_date_filter_from_date(self, session, trial_sites):
        ts = trial_sites[0]
        _add_enrolment(session, ts.id, days_ago_screened=20,
                       days_ago_enrolled=15, suffix="old")
        _add_enrolment(session, ts.id, days_ago_screened=5,
                       days_ago_enrolled=2, suffix="new")
        cutoff = TODAY - timedelta(days=10)
        result = compute_funnel(session, ts.id, from_date=cutoff)
        assert result.screened == 1

    def test_country_aggregation(self, session, trial, trial_sites):
        ts = trial_sites[0]  # GBR
        _add_enrolment(session, ts.id, days_ago_screened=5,
                       days_ago_enrolled=2, suffix="g1")
        result = compute_funnel_by_country(session, trial.id)
        assert "GBR" in result
        assert result["GBR"]["screened"] >= 1

    def test_site_id_in_result(self, session, trial_sites, sites):
        result = compute_funnel(session, trial_sites[0].id)
        assert result.site_id == sites[0].site_id

    def test_zero_enrolled_withdrawal_rate_is_zero(self, session, trial_sites):
        result = compute_funnel(session, trial_sites[1].id)
        assert result.withdrawal_rate == 0.0


# ===========================================================================
# DATA LAG TESTS
# ===========================================================================

class TestDataLag:
    def test_no_events_returns_none_metrics(self, session, trial_sites):
        result = compute_lag(session, trial_sites[1].id)
        assert result.n_events == 0
        assert result.lag_mean is None
        assert result.lag_p90 is None

    def test_single_event_mean_equals_lag(self, session, trial_sites):
        _add_lag(session, trial_sites[0].id, visit_days_ago=5, lag=7)
        result = compute_lag(session, trial_sites[0].id)
        assert result.n_events == 1
        assert result.lag_mean == 7.0
        assert result.lag_p90 == 7.0

    def test_mean_computed_correctly(self, session, trial_sites):
        for lag in [2, 4, 6, 8, 10]:
            _add_lag(session, trial_sites[0].id, visit_days_ago=lag + 5, lag=lag)
        result = compute_lag(session, trial_sites[0].id)
        assert result.lag_mean == pytest.approx(6.0, rel=1e-3)

    def test_p90_above_mean(self, session, trial_sites):
        for i, lag in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 20]):
            _add_lag(session, trial_sites[0].id, visit_days_ago=i + 5, lag=lag)
        result = compute_lag(session, trial_sites[0].id)
        assert result.lag_p90 >= result.lag_mean

    def test_zero_lag_events_mean_zero(self, session, trial_sites):
        for i in range(5):
            _add_lag(session, trial_sites[0].id, visit_days_ago=i + 5, lag=0)
        result = compute_lag(session, trial_sites[0].id)
        assert result.lag_mean == 0.0
        assert result.lag_p90 == 0.0

    def test_trend_negative_when_improving(self, session, trial_sites):
        ts = trial_sites[0]
        # Prior 7 days (8–15 days ago): high lag
        for i in range(7):
            _add_lag(session, ts.id, visit_days_ago=8 + i, lag=15)
        # Recent 7 days (1–7 days ago): low lag
        for i in range(7):
            _add_lag(session, ts.id, visit_days_ago=1 + i, lag=3)
        result = compute_lag(session, ts.id)
        assert result.trend_7d is not None
        assert result.trend_7d < 0   # improving

    def test_trend_positive_when_worsening(self, session, trial_sites):
        ts = trial_sites[0]
        for i in range(7):
            _add_lag(session, ts.id, visit_days_ago=8 + i, lag=2)
        for i in range(7):
            _add_lag(session, ts.id, visit_days_ago=1 + i, lag=14)
        result = compute_lag(session, ts.id)
        assert result.trend_7d > 0   # worsening

    def test_lookback_window_respected(self, session, trial_sites):
        ts = trial_sites[0]
        _add_lag(session, ts.id, visit_days_ago=200, lag=10)  # outside 90-day window
        result = compute_lag(session, ts.id, lookback_days=90)
        assert result.n_events == 0

    def test_lag_max_gte_p90(self, session, trial_sites):
        for i, lag in enumerate([1, 5, 10, 25]):
            _add_lag(session, trial_sites[0].id, visit_days_ago=i + 5, lag=lag)
        result = compute_lag(session, trial_sites[0].id)
        assert result.lag_max >= result.lag_p90
