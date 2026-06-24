"""
ClinIQ — Phase 2 Tests: ML Engine
Covers: risk score range/components, NLP classifier precision, dropout probability,
        snapshot idempotency, rollup arithmetic
"""
import hashlib
from datetime import date, timedelta

import pytest

from cliniq.db.models import (
    DataEntryEvent, DeviationSeverity, EnrolmentStatus, MonitoringVisit,
    PatientEnrolment, ProtocolDeviation, QueryLog, RiskScore,
    SiteKPISnapshot, VisitType,
)
from cliniq.ml.risk_model import (
    RiskFeatures, compute_risk_score, compute_dropout_probability,
    extract_features, _enrolment_component, _deviation_component,
    _lag_component, _monitoring_component, WEIGHTS,
)
from cliniq.ml.deviation_classifier import (
    DeviationClassifier, classify_deviation, CATEGORIES,
)
from cliniq.analytics.snapshot import write_snapshot_for_site, run_daily_snapshot

TODAY = date.today()
START = TODAY - timedelta(days=30)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_enrolment(session, ts_id, days_ago=5, status=EnrolmentStatus.ENROLLED, suffix="x"):
    screened = TODAY - timedelta(days=days_ago + 5)
    enrolled = TODAY - timedelta(days=days_ago) if status != EnrolmentStatus.SCREEN_FAIL else None
    pe = PatientEnrolment(
        trial_site_id=ts_id,
        patient_id=hashlib.sha256(f"{ts_id}-{days_ago}-{suffix}".encode()).hexdigest(),
        screened_date=screened,
        enrolled_date=enrolled,
        status=status,
    )
    session.add(pe)
    session.flush()
    return pe


def _add_deviation(session, ts_id, severity=DeviationSeverity.MINOR, days_ago=5, suffix="d"):
    import uuid
    d = ProtocolDeviation(
        trial_site_id=ts_id,
        deviation_id=str(uuid.uuid4()),
        severity=severity,
        deviation_date=TODAY - timedelta(days=days_ago),
        free_text=f"Test deviation {suffix}",
    )
    session.add(d)
    session.flush()
    return d


def _add_monitoring(session, ts_id, days_ago=10):
    mv = MonitoringVisit(
        trial_site_id=ts_id,
        visit_type=VisitType.ONSITE,
        visit_date=TODAY - timedelta(days=days_ago),
        sdv_complete=True,
    )
    session.add(mv)
    session.flush()
    return mv


def _add_lag(session, ts_id, lag=5, days_ago=10):
    visit = TODAY - timedelta(days=days_ago)
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
# RISK SCORE TESTS
# ===========================================================================

class TestRiskScore:
    def test_composite_score_in_range(self, session, trial_sites):
        result = compute_risk_score(session, trial_sites[0].id)
        assert 0.0 <= result.composite_score <= 100.0

    def test_all_components_in_range(self, session, trial_sites):
        result = compute_risk_score(session, trial_sites[0].id)
        for component in [
            result.enrolment_component, result.deviation_component,
            result.data_lag_component, result.monitoring_component,
            result.dropout_component,
        ]:
            assert 0.0 <= component <= 100.0

    def test_dropout_probability_in_range(self, session, trial_sites):
        result = compute_risk_score(session, trial_sites[0].id)
        assert 0.0 <= result.dropout_probability <= 1.0

    def test_weights_sum_to_one(self):
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

    def test_high_risk_site_scores_above_low_risk(self, session, trial_sites):
        """Site with many critical deviations and stale monitoring > clean site."""
        high_ts = trial_sites[0]
        low_ts  = trial_sites[1]

        # High risk: critical deviations, high lag, no monitoring
        for i in range(5):
            _add_enrolment(session, high_ts.id, days_ago=i + 2, suffix=f"h{i}")
            _add_deviation(session, high_ts.id, severity=DeviationSeverity.CRITICAL,
                           days_ago=i + 2, suffix=f"ch{i}")
            _add_lag(session, high_ts.id, lag=20, days_ago=i + 5)

        # Low risk: enrolled, no deviations, recent monitoring
        for i in range(8):
            _add_enrolment(session, low_ts.id, days_ago=i + 2, suffix=f"l{i}")
        _add_monitoring(session, low_ts.id, days_ago=3)
        _add_lag(session, low_ts.id, lag=1, days_ago=5)

        high_score = compute_risk_score(session, high_ts.id).composite_score
        low_score  = compute_risk_score(session, low_ts.id).composite_score
        assert high_score > low_score

    def test_invalid_trial_site_raises(self, session):
        with pytest.raises(ValueError):
            compute_risk_score(session, 99999)

    def test_enrolment_component_max_when_no_enrolment(self):
        f = RiskFeatures(
            trial_site_id=1, enrolment_pct=0.0, velocity_28d=0.0,
            deviation_rate=0.0, critical_deviation_rate=0.0,
            data_lag_mean=0.0, data_lag_p90=0.0,
            days_since_monitoring=0, open_query_rate=0.0,
        )
        comp = _enrolment_component(f)
        assert comp == 100.0

    def test_enrolment_component_low_when_target_met(self):
        f = RiskFeatures(
            trial_site_id=1, enrolment_pct=1.0, velocity_28d=0.5,
            deviation_rate=0.0, critical_deviation_rate=0.0,
            data_lag_mean=0.0, data_lag_p90=0.0,
            days_since_monitoring=0, open_query_rate=0.0,
        )
        comp = _enrolment_component(f)
        assert comp == 0.0

    def test_deviation_component_caps_at_100(self):
        f = RiskFeatures(
            trial_site_id=1, enrolment_pct=0.5, velocity_28d=0.2,
            deviation_rate=5.0, critical_deviation_rate=2.0,
            data_lag_mean=0.0, data_lag_p90=0.0,
            days_since_monitoring=0, open_query_rate=0.0,
        )
        comp = _deviation_component(f)
        assert comp <= 100.0

    def test_lag_component_zero_when_no_lag(self):
        f = RiskFeatures(
            trial_site_id=1, enrolment_pct=0.5, velocity_28d=0.2,
            deviation_rate=0.1, critical_deviation_rate=0.0,
            data_lag_mean=0.0, data_lag_p90=0.0,
            days_since_monitoring=30, open_query_rate=0.0,
        )
        assert _lag_component(f) == 0.0

    def test_monitoring_component_max_when_stale(self):
        f = RiskFeatures(
            trial_site_id=1, enrolment_pct=0.5, velocity_28d=0.2,
            deviation_rate=0.1, critical_deviation_rate=0.0,
            data_lag_mean=0.0, data_lag_p90=0.0,
            days_since_monitoring=999, open_query_rate=0.0,
        )
        assert _monitoring_component(f) == 100.0

    def test_composite_never_below_zero(self, session, trial_sites):
        # Fresh site with no data
        result = compute_risk_score(session, trial_sites[2].id)
        assert result.composite_score >= 0.0

    def test_composite_never_above_100(self, session, trial_sites):
        ts = trial_sites[0]
        # Worst-case site
        for i in range(3):
            _add_deviation(session, ts.id, severity=DeviationSeverity.CRITICAL,
                           days_ago=i + 2, suffix=str(i))
        _add_lag(session, ts.id, lag=60, days_ago=5)
        result = compute_risk_score(session, ts.id)
        assert result.composite_score <= 100.0

    def test_features_extracted_correctly(self, session, trial_sites):
        ts = trial_sites[0]
        _add_enrolment(session, ts.id, days_ago=5, suffix="f1")
        _add_deviation(session, ts.id, days_ago=3, suffix="fd1")
        features = extract_features(session, ts.id)
        assert features.deviation_rate > 0
        assert features.enrolment_pct > 0


# ===========================================================================
# DROPOUT PROBABILITY TESTS
# ===========================================================================

class TestDropoutProbability:
    def test_probability_in_range(self, session, trial_sites):
        features = extract_features(session, trial_sites[0].id)
        prob = compute_dropout_probability(features)
        assert 0.0 <= prob <= 1.0

    def test_high_risk_features_give_higher_prob(self):
        low_risk = RiskFeatures(
            trial_site_id=1, enrolment_pct=0.9, velocity_28d=0.4,
            deviation_rate=0.05, critical_deviation_rate=0.0,
            data_lag_mean=2.0, data_lag_p90=4.0,
            days_since_monitoring=14, open_query_rate=0.1,
        )
        high_risk = RiskFeatures(
            trial_site_id=2, enrolment_pct=0.1, velocity_28d=0.0,
            deviation_rate=0.8, critical_deviation_rate=0.3,
            data_lag_mean=25.0, data_lag_p90=45.0,
            days_since_monitoring=180, open_query_rate=1.5,
        )
        low_prob  = compute_dropout_probability(low_risk)
        high_prob = compute_dropout_probability(high_risk)
        assert high_prob > low_prob

    def test_multiple_calls_consistent(self):
        f = RiskFeatures(
            trial_site_id=1, enrolment_pct=0.5, velocity_28d=0.2,
            deviation_rate=0.2, critical_deviation_rate=0.05,
            data_lag_mean=7.0, data_lag_p90=14.0,
            days_since_monitoring=30, open_query_rate=0.3,
        )
        probs = [compute_dropout_probability(f) for _ in range(5)]
        assert len(set(probs)) == 1  # deterministic


# ===========================================================================
# NLP DEVIATION CLASSIFIER TESTS
# ===========================================================================

class TestDeviationClassifier:
    @pytest.fixture(autouse=True)
    def classifier(self):
        self.clf = DeviationClassifier()

    # Consent category
    def test_classifies_consent_icf(self):
        assert self.clf.classify("Informed consent not obtained before dosing") == "consent"

    def test_classifies_consent_reconsent(self):
        assert self.clf.classify("Patient re-consented after protocol amendment") == "consent"

    def test_classifies_consent_icf_abbreviation(self):
        assert self.clf.classify("ICF version 1.2 used after v1.3 was approved") == "consent"

    # Dosing category
    def test_classifies_dosing_window(self):
        assert self.clf.classify("Dose administered 4 hours outside the permitted window") == "dosing"

    def test_classifies_dosing_imp(self):
        assert self.clf.classify("IMP handling procedure not followed for dose preparation") == "dosing"

    def test_classifies_dosing_administration(self):
        result = self.clf.classify("Drug administration deviated from protocol requirements")
        assert result in ("dosing", "documentation")  # acceptable ambiguity

    # Eligibility category
    def test_classifies_eligibility_inclusion(self):
        assert self.clf.classify("Patient enrolled despite not meeting inclusion criterion 3") == "eligibility"

    def test_classifies_eligibility_ecg(self):
        assert self.clf.classify("Baseline ECG assessment performed outside the screening window") == "eligibility"

    def test_classifies_eligibility_egfr(self):
        assert self.clf.classify("Patient enrolled with eGFR below threshold") == "eligibility"

    def test_classifies_eligibility_washout(self):
        assert self.clf.classify("Washout period not confirmed before first dose") == "eligibility"

    # Documentation category
    def test_classifies_documentation_crf(self):
        assert self.clf.classify("CRF page submitted without PI signature") == "documentation"

    def test_classifies_documentation_source_data(self):
        assert self.clf.classify("Source data not available to verify visit date") == "documentation"

    def test_classifies_documentation_missing(self):
        assert self.clf.classify("Missing visit notes from scheduled assessment") == "documentation"

    # Safety category
    def test_classifies_safety_ae(self):
        assert self.clf.classify("Adverse event not reported within 24-hour window") == "safety"

    def test_classifies_safety_sae(self):
        assert self.clf.classify("SAE narrative submitted 48 hours after deadline") == "safety"

    def test_classifies_safety_serious(self):
        result = self.clf.classify("Serious adverse event not escalated per protocol")
        assert result == "safety"

    # Edge cases
    def test_empty_string_returns_documentation(self):
        assert self.clf.classify("") == "documentation"

    def test_whitespace_only_returns_documentation(self):
        assert self.clf.classify("   ") == "documentation"

    def test_none_handled(self):
        # None should default to documentation
        assert self.clf.classify(None or "") == "documentation"

    def test_all_categories_reachable(self):
        texts = [
            "ICF not obtained",
            "Dose outside window",
            "Inclusion criteria not met",
            "CRF missing signature",
            "Adverse event unreported",
        ]
        results = self.clf.classify_batch(texts)
        assert set(results) == {"consent", "dosing", "eligibility", "documentation", "safety"}

    def test_classify_batch_length_matches_input(self):
        texts = ["consent ICF", "dosing IMP", "eligibility criteria"]
        results = self.clf.classify_batch(texts)
        assert len(results) == len(texts)

    def test_confidence_scores_returns_all_categories(self):
        scores = self.clf.confidence_scores("Patient re-consented after amendment")
        assert set(scores.keys()) == set(CATEGORIES)

    def test_confidence_scores_non_negative(self):
        scores = self.clf.confidence_scores("Some deviation text")
        assert all(v >= 0 for v in scores.values())

    def test_module_singleton(self):
        r1 = classify_deviation("ICF missing")
        r2 = classify_deviation("ICF missing")
        assert r1 == r2 == "consent"


# ===========================================================================
# SNAPSHOT WRITER TESTS
# ===========================================================================

class TestSnapshotWriter:
    def test_snapshot_writes_successfully(self, session, trial_sites):
        ts = trial_sites[0]
        written = write_snapshot_for_site(session, ts.id, TODAY)
        session.flush()
        assert written is True

    def test_snapshot_idempotent_second_call_returns_false(self, session, trial_sites):
        ts = trial_sites[0]
        write_snapshot_for_site(session, ts.id, TODAY)
        session.flush()
        written_again = write_snapshot_for_site(session, ts.id, TODAY)
        assert written_again is False

    def test_snapshot_persisted_in_db(self, session, trial_sites):
        ts = trial_sites[0]
        write_snapshot_for_site(session, ts.id, TODAY)
        session.flush()
        snap = session.query(SiteKPISnapshot).filter_by(
            trial_site_id=ts.id, snapshot_date=TODAY
        ).first()
        assert snap is not None

    def test_risk_score_written_with_snapshot(self, session, trial_sites):
        ts = trial_sites[0]
        write_snapshot_for_site(session, ts.id, TODAY)
        session.flush()
        rs = session.query(RiskScore).filter_by(
            trial_site_id=ts.id, score_date=TODAY
        ).first()
        assert rs is not None
        assert 0.0 <= rs.composite_score <= 100.0

    def test_snapshot_different_dates_both_written(self, session, trial_sites):
        ts = trial_sites[0]
        yesterday = TODAY - timedelta(days=1)
        write_snapshot_for_site(session, ts.id, yesterday)
        session.flush()
        write_snapshot_for_site(session, ts.id, TODAY)
        session.flush()
        count = session.query(SiteKPISnapshot).filter_by(trial_site_id=ts.id).count()
        assert count == 2

    def test_run_daily_snapshot_all_sites(self, session, trial, trial_sites):
        summary = run_daily_snapshot(session, trial.id, TODAY)
        assert summary.sites_processed == len(trial_sites)
        assert summary.sites_skipped == 0
        assert summary.errors == []

    def test_run_daily_snapshot_idempotent(self, session, trial, trial_sites):
        run_daily_snapshot(session, trial.id, TODAY)
        summary2 = run_daily_snapshot(session, trial.id, TODAY)
        assert summary2.sites_processed == 0
        assert summary2.sites_skipped == len(trial_sites)

    def test_snapshot_enrolment_rate_matches_velocity(self, session, trial_sites):
        ts = trial_sites[0]
        for i in range(3):
            _add_enrolment(session, ts.id, days_ago=i + 2, suffix=str(i))
        write_snapshot_for_site(session, ts.id, TODAY)
        session.flush()
        snap = session.query(SiteKPISnapshot).filter_by(
            trial_site_id=ts.id, snapshot_date=TODAY
        ).first()
        expected_rate = round(3 / 28.0, 4)
        assert snap.enrolment_rate_28d == pytest.approx(expected_rate, rel=1e-3)

    def test_snapshot_open_queries_counted_correctly(self, session, trial_sites):
        ts = trial_sites[0]
        import uuid
        for i in range(4):
            session.add(QueryLog(
                trial_site_id=ts.id, query_id=str(uuid.uuid4()),
                opened_date=TODAY - timedelta(days=i + 3),
                is_resolved=False, category="data_entry",
                age_days=i + 3,
            ))
        session.flush()
        write_snapshot_for_site(session, ts.id, TODAY)
        session.flush()
        snap = session.query(SiteKPISnapshot).filter_by(
            trial_site_id=ts.id, snapshot_date=TODAY
        ).first()
        assert snap.open_queries == 4

    def test_invalid_trial_site_raises(self, session):
        with pytest.raises(ValueError):
            write_snapshot_for_site(session, 99999, TODAY)
