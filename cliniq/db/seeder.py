"""
ClinIQ — Synthetic data seeder.
Produces 8 fictional sites across UK, Germany, Poland, Netherlands
with deliberately varied profiles for demo richness.
Run:  python -m cliniq.db.seeder [--reset]
"""
from __future__ import annotations

import hashlib
import random
import uuid
from datetime import date, timedelta
from typing import List

from faker import Faker
from sqlalchemy.orm import Session

from cliniq.db.models import (
    AuditLog, DataEntryEvent, DeviationCluster, DeviationSeverity,
    EnrolmentForecast, EnrolmentStatus, MonitoringVisit, PatientEnrolment,
    ProtocolDeviation, ProtocolVersion, QueryLog, RiskScore, Site,
    SiteKPISnapshot, SiteType, Trial, TrialPhase, TrialSite, TrialStatus,
    VisitType,
)

fake = Faker("en_GB")
random.seed(42)

# ---------------------------------------------------------------------------
# Site profiles — deliberately varied for demo richness
# ---------------------------------------------------------------------------

SITE_PROFILES = [
    # (site_id_suffix, name, country, city, pi, site_type, profile_tag)
    ("UK001", "St. Mary's Clinical Research Unit",   "GBR", "London",     "Dr. Sarah Hutchins",    SiteType.HOSPITAL,   "high_enroller"),
    ("UK002", "Cambridge Biomedical Research Centre", "GBR", "Cambridge",  "Prof. James Whitfield", SiteType.ACADEMIC,   "chronic_deviator"),
    ("UK003", "Leeds Community Trials Network",       "GBR", "Leeds",      "Dr. Priya Nair",        SiteType.COMMUNITY,  "slow_starter"),
    ("DE001", "Universitätsklinikum Frankfurt",       "DEU", "Frankfurt",  "Prof. Klaus Bauer",     SiteType.ACADEMIC,   "balanced"),
    ("DE002", "Charité Forschungszentrum Berlin",     "DEU", "Berlin",     "Dr. Annika Vogel",      SiteType.HOSPITAL,   "high_enroller"),
    ("PL001", "Kraków Clinical Research Institute",   "POL", "Kraków",     "Dr. Marek Wiśniewski",  SiteType.PRIVATE,    "slow_starter"),
    ("NL001", "Amsterdam UMC Trials Centre",          "NLD", "Amsterdam",  "Prof. Hanna de Vries",  SiteType.ACADEMIC,   "balanced"),
    ("NL002", "Erasmus MC Research Division",         "NLD", "Rotterdam",  "Dr. Sander Visser",     SiteType.HOSPITAL,   "chronic_deviator"),
]

# Profile → (enrolment_rate_per_day, screen_fail_pct, deviation_rate_per_pt, lag_mean_days)
PROFILE_PARAMS = {
    "high_enroller":    (0.35, 0.20, 0.08, 4.0),
    "chronic_deviator": (0.18, 0.30, 0.35, 12.0),
    "slow_starter":     (0.08, 0.45, 0.10, 9.0),
    "balanced":         (0.20, 0.28, 0.12, 6.5),
}

DEVIATION_FREE_TEXTS = {
    "consent": [
        "Patient re-consented after protocol amendment; original ICF retained",
        "Informed consent obtained by study coordinator instead of PI",
        "Re-consent documentation missing from patient file",
        "Consent form version 1.2 used after v1.3 was approved",
    ],
    "dosing": [
        "Dose administered 4 hours outside the permitted window",
        "Study drug given at incorrect dose level due to transcription error",
        "Concomitant medication not withheld prior to dosing as per protocol",
        "Dose preparation deviated from IMP handling procedure",
    ],
    "eligibility": [
        "Patient enrolled with eGFR marginally below inclusion threshold",
        "Baseline ECG assessment performed 3 days outside the screening window",
        "Inclusion criterion 7 (prior therapy washout) not confirmed before enrolment",
        "Laboratory result outside normal range not reviewed by PI before dosing",
    ],
    "documentation": [
        "Source data not available to verify visit assessment date",
        "CRF page submitted without PI signature",
        "Protocol deviation not reported to sponsor within 5 working days",
        "Visit procedure performed out of sequence; not documented in visit notes",
    ],
    "safety": [
        "Adverse event not reported within 24-hour window as per protocol",
        "SAE narrative submitted to sponsor 48 hours after the regulatory deadline",
        "Unscheduled safety visit not documented in investigator site file",
    ],
}

TRIAL_SEED = {
    "trial_id": "CLQ-2024-001",
    "sponsor": "Meridian Therapeutics Ltd",
    "phase": TrialPhase.PHASE_II,
    "title": "A Phase II Randomised Study of MT-7741 in Adults with Moderate-to-Severe Plaque Psoriasis",
    "therapeutic_area": "Dermatology",
    "eudract_number": "2024-001234-12",
    "isrctn_number": "ISRCTN87654321",
}

SEED_START_DATE = date.today() - timedelta(days=365)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_patient(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def _date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _random_deviation_text(category: str) -> str:
    return random.choice(DEVIATION_FREE_TEXTS.get(category, ["Deviation noted; details under review"]))


def _severity_draw() -> DeviationSeverity:
    r = random.random()
    if r < 0.70:
        return DeviationSeverity.MINOR
    elif r < 0.95:
        return DeviationSeverity.MAJOR
    return DeviationSeverity.CRITICAL


# ---------------------------------------------------------------------------
# Seeder main
# ---------------------------------------------------------------------------

def seed(session: Session, reset: bool = False) -> dict:
    """
    Idempotent seeder. Returns a summary dict of inserted counts.
    Pass reset=True to wipe and re-seed (dev only).
    """
    if reset:
        _wipe(session)

    # Skip if already seeded
    if session.query(Trial).filter_by(trial_id=TRIAL_SEED["trial_id"]).first():
        return {"status": "already_seeded"}

    counts: dict = {}

    # ---- Trial ----
    trial = Trial(
        trial_id=TRIAL_SEED["trial_id"],
        sponsor=TRIAL_SEED["sponsor"],
        phase=TRIAL_SEED["phase"],
        status=TrialStatus.ACTIVE,
        title=TRIAL_SEED["title"],
        therapeutic_area=TRIAL_SEED["therapeutic_area"],
        eudract_number=TRIAL_SEED["eudract_number"],
        isrctn_number=TRIAL_SEED["isrctn_number"],
        start_date=SEED_START_DATE,
        planned_end_date=SEED_START_DATE + timedelta(days=540),
    )
    session.add(trial)
    session.flush()
    counts["trials"] = 1

    # ---- Protocol versions ----
    for i, (ver, delta) in enumerate([("1.0", 0), ("1.1", 90), ("1.2", 210)]):
        session.add(ProtocolVersion(
            trial_id=trial.id,
            version=ver,
            amendment_date=SEED_START_DATE + timedelta(days=delta),
            changes=f"Amendment {i}: {'Initial version' if i == 0 else fake.sentence(nb_words=12)}",
        ))
    session.flush()
    counts["protocol_versions"] = 3

    # ---- Sites + TrialSites ----
    site_objs: list[tuple[Site, TrialSite, str]] = []
    for suf, name, country, city, pi, stype, profile in SITE_PROFILES:
        site = Site(
            site_id=suf,
            name=name,
            country=country,
            city=city,
            pi_name=pi,
            pi_email=f"{suf.lower()}@clinresearch.example.com",
            site_type=stype,
        )
        session.add(site)
        session.flush()

        target = random.randint(12, 30)
        activation_offset = random.randint(0, 60)
        ts = TrialSite(
            trial_id=trial.id,
            site_id=site.id,
            enrolment_target=target,
            activation_date=SEED_START_DATE + timedelta(days=activation_offset),
        )
        session.add(ts)
        session.flush()
        site_objs.append((site, ts, profile))

    counts["sites"] = len(site_objs)
    counts["trial_sites"] = len(site_objs)

    # ---- Operational time-series per site ----
    enrolment_count = deviation_count = entry_count = visit_count = query_count = 0

    for site, ts, profile in site_objs:
        rate, sfail_pct, dev_rate, lag_mean = PROFILE_PARAMS[profile]
        activation = ts.activation_date or SEED_START_DATE
        today = date.today()

        patient_counter = 0
        enrolled_patients: list[date] = []

        # Patient enrolments
        for d in _date_range(activation, today):
            # Slow-starters ramp up after 60 days
            effective_rate = rate
            if profile == "slow_starter" and (d - activation).days < 60:
                effective_rate = rate * 0.3

            expected = effective_rate
            if random.random() < expected:
                patient_counter += 1
                raw_id = f"{ts.site_id}-P{patient_counter:04d}"
                screened_date = d
                is_fail = random.random() < sfail_pct

                if is_fail:
                    status = EnrolmentStatus.SCREEN_FAIL
                    enrolled_date = None
                else:
                    enrolment_lag = random.randint(1, 14)
                    enrolled_date = screened_date + timedelta(days=enrolment_lag)
                    if enrolled_date > today:
                        enrolled_date = None
                        status = EnrolmentStatus.SCREENED
                    else:
                        is_withdrawn = random.random() < 0.05
                        status = EnrolmentStatus.WITHDRAWN if is_withdrawn else EnrolmentStatus.ENROLLED
                        if enrolled_date:
                            enrolled_patients.append(enrolled_date)

                session.add(PatientEnrolment(
                    trial_site_id=ts.id,
                    patient_id=_hash_patient(raw_id),
                    screened_date=screened_date,
                    enrolled_date=enrolled_date,
                    status=status,
                ))
                enrolment_count += 1

        # Protocol deviations — rate per enrolled patient
        n_deviations = max(1, int(len(enrolled_patients) * dev_rate))
        for _ in range(n_deviations):
            cat = random.choice(list(DEVIATION_FREE_TEXTS.keys()))
            dev_date = activation + timedelta(days=random.randint(10, (today - activation).days))
            session.add(ProtocolDeviation(
                trial_site_id=ts.id,
                deviation_id=str(uuid.uuid4()),
                category=cat,
                severity=_severity_draw(),
                deviation_date=dev_date,
                reported_date=dev_date + timedelta(days=random.randint(1, 5)),
                free_text=_random_deviation_text(cat),
                is_resolved=random.random() < 0.7,
            ))
            deviation_count += 1

        # Data entry events
        for ep_date in enrolled_patients:
            n_visits = random.randint(2, 6)
            for v in range(n_visits):
                visit_date = ep_date + timedelta(days=v * 28 + random.randint(-3, 3))
                if visit_date > today:
                    break
                lag = max(0, int(random.gauss(lag_mean, lag_mean * 0.4)))
                entry_date = visit_date + timedelta(days=lag)
                if entry_date > today:
                    entry_date = today
                session.add(DataEntryEvent(
                    trial_site_id=ts.id,
                    visit_date=visit_date,
                    entry_date=entry_date,
                    ecrf_page=f"CRF-{random.randint(1, 20):02d}",
                    lag_days=lag,
                ))
                entry_count += 1

        # Monitoring visits — every 8–12 weeks
        mv_date = activation + timedelta(days=random.randint(20, 40))
        while mv_date <= today:
            session.add(MonitoringVisit(
                trial_site_id=ts.id,
                visit_type=random.choice(list(VisitType)),
                visit_date=mv_date,
                findings=fake.sentence(nb_words=10) if random.random() < 0.4 else None,
                sdv_complete=random.random() < 0.8,
            ))
            visit_count += 1
            mv_date += timedelta(weeks=random.randint(8, 12))

        # Query log
        n_queries = random.randint(3, 15)
        for _ in range(n_queries):
            open_date = activation + timedelta(days=random.randint(10, (today - activation).days))
            is_resolved = random.random() < 0.65
            resolved_date = open_date + timedelta(days=random.randint(3, 30)) if is_resolved else None
            if resolved_date and resolved_date > today:
                resolved_date = None
                is_resolved = False
            session.add(QueryLog(
                trial_site_id=ts.id,
                query_id=str(uuid.uuid4()),
                opened_date=open_date,
                resolved_date=resolved_date,
                category=random.choice(["data_entry", "missing_value", "eligibility", "safety", "dosing"]),
                is_resolved=is_resolved,
                age_days=(today - open_date).days if not is_resolved else None,
            ))
            query_count += 1

    session.commit()

    counts.update({
        "patient_enrolments": enrolment_count,
        "protocol_deviations": deviation_count,
        "data_entry_events": entry_count,
        "monitoring_visits": visit_count,
        "query_logs": query_count,
        "status": "seeded",
    })
    return counts


def _wipe(session: Session):
    """Delete all data (dev/test only)."""
    from sqlalchemy import text
    tables = [
        "audit_log", "deviation_clusters", "enrolment_forecasts",
        "risk_scores", "site_kpi_snapshots", "query_log",
        "monitoring_visits", "data_entry_events", "protocol_deviations",
        "patient_enrolments", "trial_sites", "protocol_versions",
        "sites", "trials",
    ]
    for t in tables:
        session.execute(text(f"DELETE FROM {t}"))
    session.commit()


if __name__ == "__main__":
    import argparse
    from cliniq.db.database import SessionLocal, init_db
    parser = argparse.ArgumentParser(description="ClinIQ seeder")
    parser.add_argument("--reset", action="store_true", help="Wipe and re-seed")
    args = parser.parse_args()
    init_db()
    with SessionLocal() as s:
        result = seed(s, reset=args.reset)
        print(result)
