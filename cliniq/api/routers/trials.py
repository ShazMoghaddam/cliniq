"""
ClinIQ API — Trial endpoints
GET /trials                   List all accessible trials
GET /trials/{trial_id}        Trial detail with protocol versions
GET /trials/{trial_id}/sites  Sites enrolled in a trial
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from cliniq.api.schemas import TrialDetail, TrialSummary, SiteSummary
from cliniq.db.database import get_db
from cliniq.db.models import Trial, TrialSite
from cliniq.rbac.auth import (
    TokenData, get_current_user, log_action, require_trial_access,
)

router = APIRouter(prefix="/trials", tags=["trials"])


def _trial_summary(trial: Trial) -> TrialSummary:
    return TrialSummary(
        id=trial.id,
        trial_id=trial.trial_id,
        sponsor=trial.sponsor,
        phase=trial.phase.value,
        status=trial.status.value,
        title=trial.title,
        therapeutic_area=trial.therapeutic_area,
        eudract_number=trial.eudract_number,
        isrctn_number=trial.isrctn_number,
        start_date=trial.start_date,
        planned_end_date=trial.planned_end_date,
        site_count=len(trial.trial_sites),
    )


@router.get("", response_model=list[TrialSummary])
def list_trials(
    request: Request,
    db: Session = Depends(get_db),
    token: TokenData = Depends(get_current_user),
):
    """List trials. SPONSOR_VIEW sees only their assigned trial."""
    q = db.query(Trial)
    if token.role.value == "sponsor_view":
        if token.trial_id is None:
            return []
        q = q.filter(Trial.id == token.trial_id)

    trials = q.all()
    log_action(db, token, "LIST_TRIALS", "/trials", request)
    db.commit()
    return [_trial_summary(t) for t in trials]


@router.get("/{trial_id}", response_model=TrialDetail)
def get_trial(
    trial_id: int,
    request: Request,
    db: Session = Depends(get_db),
    token: TokenData = Depends(get_current_user),
):
    """Trial detail with protocol versions."""
    require_trial_access(trial_id, token)
    trial = db.get(Trial, trial_id)
    if trial is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trial not found")

    from cliniq.api.schemas import ProtocolVersionSchema
    log_action(db, token, "READ_TRIAL", f"/trials/{trial_id}", request)
    db.commit()

    summary = _trial_summary(trial)
    return TrialDetail(
        **summary.model_dump(),
        protocol_versions=[
            ProtocolVersionSchema.model_validate(pv) for pv in trial.protocol_versions
        ],
    )


@router.get("/{trial_id}/sites", response_model=list[SiteSummary])
def list_trial_sites(
    trial_id: int,
    request: Request,
    db: Session = Depends(get_db),
    token: TokenData = Depends(get_current_user),
):
    """Sites enrolled in a trial."""
    require_trial_access(trial_id, token)
    trial = db.get(Trial, trial_id)
    if trial is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trial not found")

    log_action(db, token, "LIST_TRIAL_SITES", f"/trials/{trial_id}/sites", request)
    db.commit()

    return [
        SiteSummary(
            id=ts.site.id,
            site_id=ts.site.site_id,
            name=ts.site.name,
            country=ts.site.country,
            city=ts.site.city,
            pi_name=ts.site.pi_name,
            site_type=ts.site.site_type.value,
            is_active=ts.site.is_active,
        )
        for ts in trial.trial_sites
    ]
