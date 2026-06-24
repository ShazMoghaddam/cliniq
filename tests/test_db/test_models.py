"""
ClinIQ — Phase 1 Tests: Data Foundation
Tests: model constraints, FK integrity, seeder idempotency, enum validation
Target: 80 tests
"""
import hashlib
import uuid
from datetime import date, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from cliniq.db.models import (
    AuditLog, DataEntryEvent, DeviationCluster, DeviationSeverity,
    EnrolmentForecast, EnrolmentStatus, MonitoringVisit, PatientEnrolment,
    ProtocolDeviation, ProtocolVersion, QueryLog, RiskScore, Site,
    SiteKPISnapshot, SiteType, Trial, TrialPhase, TrialSite, TrialStatus,
    UserRole, VisitType,
)
from cliniq.db.seeder import seed, _wipe, TRIAL_SEED

TODAY = date.today()
START = TODAY - timedelta(days=30)


# ===========================================================================
# 1. Trial model
# ===========================================================================

class TestTrialModel:
    def test_create_trial_minimal(self, session, trial):
        """Trial persists with required fields."""
        fetched = session.query(Trial).filter_by(trial_id="TEST-001").first()
        assert fetched is not None
        assert fetched.sponsor == "Test Sponsor Ltd"

    def test_trial_id_unique(self, session, trial):
        """Duplicate trial_id raises IntegrityError."""
        dupe = Trial(
            trial_id="TEST-001",
            sponsor="Other", phase=TrialPhase.PHASE_I,
            status=TrialStatus.PLANNED, title="Duplicate",
        )
        session.add(dupe)
        with pytest.raises(IntegrityError):
            session.flush()

    def test_trial_status_enum_values(self, session, trial):
        for status in TrialStatus:
            trial.status = status
            session.flush()
            assert session.query(Trial).get(trial.id).status == status

    def test_trial_phase_enum_values(self, session, trial):
        for phase in TrialPhase:
            trial.phase = phase
            session.flush()
            assert session.query(Trial).get(trial.id).phase == phase

    def test_trial_optional_fields_nullable(self, session):
        t = Trial(
            trial_id="TEST-MINIMAL",
            sponsor="S", phase=TrialPhase.PHASE_I,
            status=TrialStatus.PLANNED, title="Minimal",
        )
        session.add(t)
        session.flush()
        assert t.eudract_number is None
        assert t.isrctn_number is None
        assert t.therapeutic_area is None

    def test_trial_repr(self, trial):
        assert "TEST-001" in repr(trial)
        assert "phase" in repr(trial).lower()

    def test_trial_cascade_delete_protocol_versions(self, session, trial):
        pv = ProtocolVersion(trial_id=trial.id, version="1.0",
                             amendment_date=START)
        session.add(pv)
        session.flush()
        session.delete(trial)
        session.flush()
        assert session.query(ProtocolVersion).filter_by(trial_id=trial.id).count() == 0

    def test_trial_relationships_not_none(self, session, trial):
        assert trial.trial_sites is not None
        assert trial.protocol_versions is not None


# ===========================================================================
# 2. Site model
# ===========================================================================

class TestSiteModel:
    def test_site_id_unique(self, session, sites):
        dupe = Site(site_id="SITE-A", name="Duplicate", country="GBR",
                    pi_name="Dr X", site_type=SiteType.HOSPITAL)
        session.add(dupe)
        with pytest.raises(IntegrityError):
            session.flush()

    def test_site_is_active_default(self, session, sites):
        assert all(s.is_active for s in sites)

    def test_site_type_all_values(self, session):
        for i, stype in enumerate(SiteType):
            s = Site(site_id=f"ST-{i}", name=f"Site {i}", country="GBR",
                     pi_name=f"Dr {i}", site_type=stype)
            session.add(s)
        session.flush()
        assert session.query(Site).count() >= len(SiteType)

    def test_site_country_stores_iso3(self, session, sites):
        assert all(len(s.country) == 3 for s in sites)

    def test_site_repr(self, sites):
        for s in sites:
            assert s.site_id in repr(s)

    def test_site_pi_name_required(self, session):
        s = Site(site_id="X001", name="No PI", country="GBR",
                 site_type=SiteType.HOSPITAL)
        session.add(s)
        with pytest.raises(IntegrityError):
            session.flush()


# ===========================================================================
# 3. TrialSite (junction)
# ===========================================================================

class TestTrialSite:
    def test_unique_trial_site_pair(self, session, trial_sites, trial, sites):
        dupe = TrialSite(
            trial_id=trial.id, site_id=sites[0].id,
            enrolment_target=5
        )
        session.add(dupe)
        with pytest.raises(IntegrityError):
            session.flush()

    def test_enrolment_target_minimum_one(self, trial_sites):
        ts = trial_sites[0]
        with pytest.raises(ValueError):
            ts.enrolment_target = 0

    def test_enrolment_target_persisted(self, session, trial_sites):
        targets = [ts.enrolment_target for ts in trial_sites]
        assert all(t >= 1 for t in targets)

    def test_trial_site_repr(self, trial_sites):
        for ts in trial_sites:
            assert "trial=" in repr(ts) or "TrialSite" in repr(ts)

    def test_fk_cascade_delete_from_trial(self, session, trial, trial_sites):
        session.delete(trial)
        session.flush()
        assert session.query(TrialSite).count() == 0

    def test_activation_date_nullable(self, session, trial, sites):
        ts = TrialSite(trial_id=trial.id, site_id=sites[0].id,
                       enrolment_target=5, activation_date=None)
        session.add(ts)
        session.flush()
        assert ts.activation_date is None


# ===========================================================================
# 4. PatientEnrolment
# ===========================================================================

class TestPatientEnrolment:
    def test_unique_patient_per_site(self, session, trial_sites):
        pid = hashlib.sha256(b"duplicate").hexdigest()
        ts = trial_sites[0]
        session.add(PatientEnrolment(
            trial_site_id=ts.id, patient_id=pid,
            screened_date=TODAY, status=EnrolmentStatus.SCREENED,
        ))
        session.flush()
        session.add(PatientEnrolment(
            trial_site_id=ts.id, patient_id=pid,
            screened_date=TODAY, status=EnrolmentStatus.ENROLLED,
        ))
        with pytest.raises(IntegrityError):
            session.flush()

    def test_same_patient_different_sites_allowed(self, session, trial_sites):
        pid = hashlib.sha256(b"multi-site").hexdigest()
        for ts in trial_sites[:2]:
            session.add(PatientEnrolment(
                trial_site_id=ts.id, patient_id=pid,
                screened_date=TODAY, status=EnrolmentStatus.SCREENED,
            ))
        session.flush()  # should not raise

    def test_all_enrolment_statuses(self, session, trial_sites):
        ts = trial_sites[1]
        for i, status in enumerate(EnrolmentStatus):
            pid = hashlib.sha256(f"status-{i}".encode()).hexdigest()
            session.add(PatientEnrolment(
                trial_site_id=ts.id, patient_id=pid,
                screened_date=TODAY, status=status,
            ))
        session.flush()

    def test_hash_patient_code(self):
        h = PatientEnrolment.hash_patient_code("SPONSOR-P001")
        assert len(h) == 64
        assert PatientEnrolment.hash_patient_code("SPONSOR-P001") == h

    def test_patient_enrolments_in_fixture(self, session, patient_enrolments):
        assert len(patient_enrolments) == 7
        enrolled = [p for p in patient_enrolments if p.status == EnrolmentStatus.ENROLLED]
        assert len(enrolled) == 5

    def test_screened_date_required(self, session, trial_sites):
        ts = trial_sites[0]
        pe = PatientEnrolment(
            trial_site_id=ts.id,
            patient_id=hashlib.sha256(b"no-date").hexdigest(),
            status=EnrolmentStatus.SCREENED,
        )
        session.add(pe)
        with pytest.raises(IntegrityError):
            session.flush()

    def test_enrolled_date_nullable_for_screen_fails(self, session, patient_enrolments):
        failures = [p for p in patient_enrolments if p.status == EnrolmentStatus.SCREEN_FAIL]
        assert all(p.enrolled_date is None for p in failures)


# ===========================================================================
# 5. ProtocolDeviation
# ===========================================================================

class TestProtocolDeviation:
    def test_deviation_severity_distribution(self, deviations):
        minors = sum(1 for d in deviations if d.severity == DeviationSeverity.MINOR)
        assert minors == 4

    def test_deviation_id_unique(self, session, trial_sites, deviations):
        ts = trial_sites[0]
        existing_id = deviations[0].deviation_id
        dup = ProtocolDeviation(
            trial_site_id=ts.id, deviation_id=existing_id,
            severity=DeviationSeverity.MINOR, deviation_date=TODAY,
        )
        session.add(dup)
        with pytest.raises(IntegrityError):
            session.flush()

    def test_free_text_nullable(self, session, trial_sites):
        ts = trial_sites[0]
        d = ProtocolDeviation(
            trial_site_id=ts.id, deviation_id=str(uuid.uuid4()),
            severity=DeviationSeverity.MAJOR, deviation_date=TODAY,
        )
        session.add(d)
        session.flush()
        assert d.free_text is None

    def test_all_severity_levels_stored(self, deviations):
        stored = {d.severity for d in deviations}
        assert DeviationSeverity.MINOR in stored
        assert DeviationSeverity.MAJOR in stored
        assert DeviationSeverity.CRITICAL in stored

    def test_deviation_is_resolved_default_false(self, session, trial_sites):
        ts = trial_sites[0]
        d = ProtocolDeviation(
            trial_site_id=ts.id, deviation_id=str(uuid.uuid4()),
            severity=DeviationSeverity.MINOR, deviation_date=TODAY,
        )
        session.add(d)
        session.flush()
        assert d.is_resolved is False


# ===========================================================================
# 6. DataEntryEvent
# ===========================================================================

class TestDataEntryEvent:
    def test_lag_days_stored_correctly(self, data_entries):
        for de in data_entries:
            assert de.lag_days >= 0

    def test_lag_days_validation_negative_raises(self, session, trial_sites):
        ts = trial_sites[0]
        de = DataEntryEvent(
            trial_site_id=ts.id, visit_date=TODAY,
            entry_date=TODAY, lag_days=0,
        )
        session.add(de)
        session.flush()
        with pytest.raises(ValueError):
            de.lag_days = -1

    def test_ecrf_page_nullable(self, session, trial_sites):
        ts = trial_sites[0]
        de = DataEntryEvent(
            trial_site_id=ts.id, visit_date=TODAY,
            entry_date=TODAY, lag_days=0,
        )
        session.add(de)
        session.flush()
        assert de.ecrf_page is None

    def test_ten_entries_created_in_fixture(self, data_entries):
        assert len(data_entries) == 10


# ===========================================================================
# 7. MonitoringVisit
# ===========================================================================

class TestMonitoringVisit:
    def test_all_visit_types_persist(self, session, trial_sites):
        ts = trial_sites[0]
        for i, vt in enumerate(VisitType):
            session.add(MonitoringVisit(
                trial_site_id=ts.id, visit_type=vt,
                visit_date=TODAY + timedelta(days=i),
            ))
        session.flush()

    def test_sdv_complete_default_false(self, session, trial_sites):
        ts = trial_sites[0]
        mv = MonitoringVisit(
            trial_site_id=ts.id, visit_type=VisitType.REMOTE, visit_date=TODAY,
        )
        session.add(mv)
        session.flush()
        assert mv.sdv_complete is False

    def test_monitoring_fixture_has_three_visits(self, monitoring_visits):
        assert len(monitoring_visits) == 3


# ===========================================================================
# 8. QueryLog
# ===========================================================================

class TestQueryLog:
    def test_open_queries_have_no_resolved_date(self, query_logs):
        open_qs = [q for q in query_logs if not q.is_resolved]
        assert all(q.resolved_date is None for q in open_qs)

    def test_resolved_queries_have_resolved_date(self, query_logs):
        resolved = [q for q in query_logs if q.is_resolved]
        assert all(q.resolved_date is not None for q in resolved)

    def test_query_id_unique(self, session, trial_sites, query_logs):
        ts = trial_sites[0]
        dupe = QueryLog(
            trial_site_id=ts.id, query_id=query_logs[0].query_id,
            opened_date=TODAY, is_resolved=False,
        )
        session.add(dupe)
        with pytest.raises(IntegrityError):
            session.flush()

    def test_five_queries_in_fixture(self, query_logs):
        assert len(query_logs) == 5
        assert sum(1 for q in query_logs if not q.is_resolved) == 3


# ===========================================================================
# 9. Derived Analytics models
# ===========================================================================

class TestDerivedAnalyticsModels:
    def test_risk_score_range_valid(self, session, trial_sites):
        ts = trial_sites[0]
        for i, score in enumerate([0.0, 50.0, 100.0]):
            rs = RiskScore(
                trial_site_id=ts.id,
                score_date=TODAY + timedelta(days=i + 10),
                composite_score=score,
            )
            session.add(rs)
        session.flush()

    def test_risk_score_out_of_range_raises(self, session, trial_sites):
        ts = trial_sites[0]
        rs = RiskScore(trial_site_id=ts.id, score_date=TODAY, composite_score=50.0)
        with pytest.raises(ValueError):
            rs.composite_score = 101.0

    def test_risk_score_unique_per_date(self, session, trial_sites):
        ts = trial_sites[0]
        session.add(RiskScore(trial_site_id=ts.id, score_date=TODAY, composite_score=40.0))
        session.flush()
        session.add(RiskScore(trial_site_id=ts.id, score_date=TODAY, composite_score=60.0))
        with pytest.raises(IntegrityError):
            session.flush()

    def test_kpi_snapshot_unique_per_date(self, session, trial_sites):
        ts = trial_sites[0]
        session.add(SiteKPISnapshot(trial_site_id=ts.id, snapshot_date=TODAY))
        session.flush()
        session.add(SiteKPISnapshot(trial_site_id=ts.id, snapshot_date=TODAY))
        with pytest.raises(IntegrityError):
            session.flush()

    def test_enrolment_forecast_stores_projected_date(self, session, trial_sites):
        ts = trial_sites[0]
        proj = TODAY + timedelta(days=90)
        ef = EnrolmentForecast(
            trial_site_id=ts.id, forecast_date=TODAY,
            projected_completion=proj, velocity_28d=0.3,
            enrolled_to_date=5, remaining_to_target=5, is_on_track=True,
        )
        session.add(ef)
        session.flush()
        assert ef.projected_completion == proj

    def test_deviation_cluster_unique_per_cat_date(self, session, trial_sites):
        ts = trial_sites[0]
        session.add(DeviationCluster(trial_site_id=ts.id, cluster_date=TODAY,
                                     category="consent", count=3))
        session.flush()
        session.add(DeviationCluster(trial_site_id=ts.id, cluster_date=TODAY,
                                     category="consent", count=5))
        with pytest.raises(IntegrityError):
            session.flush()


# ===========================================================================
# 10. AuditLog (immutability contract)
# ===========================================================================

class TestAuditLog:
    def test_audit_log_no_foreign_keys(self):
        """AuditLog must not have FK constraints (survives cascade deletes)."""
        cols = {c.name: c for c in AuditLog.__table__.columns}
        fk_cols = [c for c in AuditLog.__table__.foreign_keys]
        assert len(fk_cols) == 0, "AuditLog must have no FK constraints"

    def test_audit_log_insert(self, session):
        al = AuditLog(
            user_id="user-123",
            role=UserRole.CRA,
            action="READ_SITE",
            resource="/sites/SITE-A",
            ip_address="10.0.0.1",
        )
        session.add(al)
        session.flush()
        assert al.id is not None
        assert al.timestamp is not None

    def test_audit_log_survives_trial_deletion(self, session, trial, sites, trial_sites):
        al = AuditLog(action="READ_TRIAL", resource=f"/trials/{trial.trial_id}")
        session.add(al)
        session.flush()
        log_id = al.id
        session.delete(trial)
        session.flush()
        assert session.query(AuditLog).get(log_id) is not None

    def test_all_user_roles(self, session):
        for role in UserRole:
            al = AuditLog(action="LOGIN", role=role)
            session.add(al)
        session.flush()


# ===========================================================================
# 11. Seeder tests
# ===========================================================================

class TestSeeder:
    def test_seeder_populates_all_tables(self, engine):
        from sqlalchemy.orm import sessionmaker
        from cliniq.db.database import init_db
        from cliniq.db.seeder import seed
        init_db(engine)
        Session = sessionmaker(bind=engine)
        with Session() as s:
            result = seed(s)
        assert result["status"] == "seeded"
        assert result["trials"] == 1
        assert result["sites"] == 8
        assert result["trial_sites"] == 8

    def test_seeder_idempotent_no_duplicates(self, engine):
        from sqlalchemy.orm import sessionmaker
        from cliniq.db.database import init_db
        init_db(engine)
        Session = sessionmaker(bind=engine)
        with Session() as s:
            r1 = seed(s)
        with Session() as s:
            r2 = seed(s)
        assert r1["status"] == "seeded"
        assert r2["status"] == "already_seeded"

    def test_seeder_trial_id_matches_config(self, engine):
        from sqlalchemy.orm import sessionmaker
        from cliniq.db.database import init_db
        init_db(engine)
        Session = sessionmaker(bind=engine)
        with Session() as s:
            seed(s)
            t = s.query(Trial).first()
        assert t.trial_id == TRIAL_SEED["trial_id"]

    def test_seeder_all_sites_have_activation_dates(self, engine):
        from sqlalchemy.orm import sessionmaker
        from cliniq.db.database import init_db
        init_db(engine)
        Session = sessionmaker(bind=engine)
        with Session() as s:
            seed(s)
            tss = s.query(TrialSite).all()
        assert all(ts.activation_date is not None for ts in tss)

    def test_seeder_deviations_severity_distribution(self, engine):
        from sqlalchemy.orm import sessionmaker
        from cliniq.db.database import init_db
        init_db(engine)
        Session = sessionmaker(bind=engine)
        with Session() as s:
            seed(s)
            total = s.query(ProtocolDeviation).count()
            minors = s.query(ProtocolDeviation).filter_by(severity=DeviationSeverity.MINOR).count()
            criticals = s.query(ProtocolDeviation).filter_by(severity=DeviationSeverity.CRITICAL).count()
        # Minor should be majority; critical should be < minor
        assert minors > criticals
        assert total > 0

    def test_seeder_patient_ids_are_hashed(self, engine):
        from sqlalchemy.orm import sessionmaker
        from cliniq.db.database import init_db
        init_db(engine)
        Session = sessionmaker(bind=engine)
        with Session() as s:
            seed(s)
            patients = s.query(PatientEnrolment).limit(10).all()
        for p in patients:
            assert len(p.patient_id) == 64, "Patient ID must be SHA-256 hex (64 chars)"
            assert p.patient_id.isalnum(), "Patient ID must be hex-encoded"

    def test_seeder_protocol_versions_created(self, engine):
        from sqlalchemy.orm import sessionmaker
        from cliniq.db.database import init_db
        init_db(engine)
        Session = sessionmaker(bind=engine)
        with Session() as s:
            seed(s)
            pvs = s.query(ProtocolVersion).all()
        assert len(pvs) == 3

    def test_seeder_wipe_and_reseed(self, engine):
        from sqlalchemy.orm import sessionmaker
        from cliniq.db.database import init_db
        init_db(engine)
        Session = sessionmaker(bind=engine)
        with Session() as s:
            seed(s)
            count_before = s.query(Trial).count()
        with Session() as s:
            result = seed(s, reset=True)
            count_after = s.query(Trial).count()
        assert result["status"] == "seeded"
        assert count_after == count_before == 1

    def test_seeder_enrolments_all_hashed_no_pii(self, engine):
        """Verify no raw patient identifiers stored — only hashes."""
        from sqlalchemy.orm import sessionmaker
        from cliniq.db.database import init_db
        init_db(engine)
        Session = sessionmaker(bind=engine)
        with Session() as s:
            seed(s)
            pids = [p.patient_id for p in s.query(PatientEnrolment).all()]
        # All patient IDs should be 64-char hex strings
        assert all(len(pid) == 64 for pid in pids)
        # Spot-check: no common PII patterns
        assert not any("patient" in pid.lower() for pid in pids)


# ===========================================================================
# 12. Foreign key integrity
# ===========================================================================

class TestForeignKeyIntegrity:
    def test_trial_site_requires_valid_trial(self, session, sites):
        ts = TrialSite(trial_id=99999, site_id=sites[0].id, enrolment_target=5)
        session.add(ts)
        with pytest.raises(IntegrityError):
            session.flush()

    def test_trial_site_requires_valid_site(self, session, trial):
        ts = TrialSite(trial_id=trial.id, site_id=99999, enrolment_target=5)
        session.add(ts)
        with pytest.raises(IntegrityError):
            session.flush()

    def test_patient_enrolment_requires_valid_trial_site(self, session):
        pe = PatientEnrolment(
            trial_site_id=99999,
            patient_id=hashlib.sha256(b"x").hexdigest(),
            screened_date=TODAY,
            status=EnrolmentStatus.SCREENED,
        )
        session.add(pe)
        with pytest.raises(IntegrityError):
            session.flush()

    def test_deviation_requires_valid_trial_site(self, session):
        d = ProtocolDeviation(
            trial_site_id=99999,
            deviation_id=str(uuid.uuid4()),
            severity=DeviationSeverity.MINOR,
            deviation_date=TODAY,
        )
        session.add(d)
        with pytest.raises(IntegrityError):
            session.flush()

    def test_monitoring_visit_requires_valid_trial_site(self, session):
        mv = MonitoringVisit(
            trial_site_id=99999, visit_type=VisitType.REMOTE, visit_date=TODAY,
        )
        session.add(mv)
        with pytest.raises(IntegrityError):
            session.flush()


# ===========================================================================
# 13. ProtocolVersion
# ===========================================================================

class TestProtocolVersion:
    def test_protocol_version_unique_per_trial(self, session, trial):
        session.add(ProtocolVersion(trial_id=trial.id, version="2.0", amendment_date=TODAY))
        session.flush()
        session.add(ProtocolVersion(trial_id=trial.id, version="2.0", amendment_date=TODAY))
        with pytest.raises(IntegrityError):
            session.flush()

    def test_protocol_version_repr(self, session, trial):
        pv = ProtocolVersion(trial_id=trial.id, version="3.0", amendment_date=TODAY)
        session.add(pv)
        session.flush()
        assert "3.0" in repr(pv)

    def test_changes_field_nullable(self, session, trial):
        pv = ProtocolVersion(trial_id=trial.id, version="4.0", amendment_date=TODAY, changes=None)
        session.add(pv)
        session.flush()
        assert pv.changes is None
