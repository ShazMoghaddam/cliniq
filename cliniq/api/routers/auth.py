"""
ClinIQ API — Auth endpoints
POST /auth/token  Issue a JWT for a given role (demo mode: password not validated)
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from cliniq.api.schemas import LoginRequest, TokenResponse
from cliniq.db.models import UserRole
from cliniq.rbac.auth import create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])

# Demo mode: fixed password. Replace with proper user store in production.
DEMO_PASSWORD = "cliniq-demo-2024"


@router.post("/token", response_model=TokenResponse)
def login(body: LoginRequest):
    """
    Issue a JWT. In demo mode, any user_id with the correct demo password works.
    role must be one of: admin, clinical_lead, cra, sponsor_view
    sponsor_view requires trial_id in the request body.
    """
    if body.password != DEMO_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    try:
        role = UserRole(body.role)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid role '{body.role}'. Valid: admin, clinical_lead, cra, sponsor_view",
        )

    if role == UserRole.SPONSOR_VIEW and body.trial_id is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="sponsor_view role requires trial_id",
        )

    token = create_access_token(
        user_id=body.user_id,
        role=role,
        trial_id=body.trial_id,
    )
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        role=role.value,
        user_id=body.user_id,
    )
