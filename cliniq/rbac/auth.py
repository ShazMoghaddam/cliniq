"""
ClinIQ — JWT Authentication & Role-Based Access Control
Roles: admin > clinical_lead > cra > sponsor_view
Sponsor View is read-only and restricted to their assigned trial only.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from cliniq.config.settings import get_settings
from cliniq.db.database import get_db
from cliniq.db.models import AuditLog, UserRole

settings = get_settings()

# ---------------------------------------------------------------------------
# Role hierarchy — higher index = more access
# ---------------------------------------------------------------------------
ROLE_HIERARCHY = [
    UserRole.SPONSOR_VIEW,
    UserRole.CRA,
    UserRole.CLINICAL_LEAD,
    UserRole.ADMIN,
]

bearer_scheme = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Token creation
# ---------------------------------------------------------------------------

def create_access_token(
    user_id: str,
    role: UserRole,
    trial_id: Optional[int] = None,   # required for sponsor_view
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a signed JWT. Embeds user_id, role, and optional trial_id claim.
    trial_id is enforced at the ORM query level for SPONSOR_VIEW.
    """
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload = {
        "sub": user_id,
        "role": role.value,
        "exp": expire,
    }
    if trial_id is not None:
        payload["trial_id"] = trial_id

    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


# ---------------------------------------------------------------------------
# Token verification
# ---------------------------------------------------------------------------

class TokenData:
    def __init__(self, user_id: str, role: UserRole, trial_id: Optional[int] = None):
        self.user_id = user_id
        self.role = role
        self.trial_id = trial_id  # scoped access for sponsor_view


def _decode_token(token: str) -> TokenData:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        role_str: str = payload.get("role")
        trial_id: Optional[int] = payload.get("trial_id")

        if user_id is None or role_str is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token claims",
            )
        try:
            role = UserRole(role_str)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Unknown role: {role_str}",
            )
        return TokenData(user_id=user_id, role=role, trial_id=trial_id)

    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token validation failed: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> TokenData:
    """Dependency: extract and validate the Bearer token."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _decode_token(credentials.credentials)


def require_role(minimum_role: UserRole):
    """
    Dependency factory: require at least `minimum_role` in the hierarchy.
    Usage: Depends(require_role(UserRole.CLINICAL_LEAD))
    """
    def _check(token: TokenData = Depends(get_current_user)) -> TokenData:
        caller_level  = ROLE_HIERARCHY.index(token.role)
        required_level = ROLE_HIERARCHY.index(minimum_role)
        if caller_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{token.role.value}' insufficient; need '{minimum_role.value}'",
            )
        return token
    return _check


def require_trial_access(trial_id: int, token: TokenData) -> None:
    """
    For SPONSOR_VIEW: enforce that the trial_id in the token matches the
    requested trial. All other roles have unrestricted trial access.
    Enforced at the ORM query level — not just UI.
    """
    if token.role == UserRole.SPONSOR_VIEW:
        if token.trial_id is None or token.trial_id != trial_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Sponsor View access is restricted to your assigned trial",
            )


# ---------------------------------------------------------------------------
# Audit logging helper
# ---------------------------------------------------------------------------

def log_action(
    db: Session,
    token: TokenData,
    action: str,
    resource: str,
    request: Optional[Request] = None,
    detail: Optional[str] = None,
) -> None:
    """Write an immutable audit log entry. Caller must commit."""
    ip = None
    if request:
        forwarded = request.headers.get("X-Forwarded-For")
        ip = forwarded.split(",")[0].strip() if forwarded else (
            request.client.host if request.client else None
        )
    db.add(AuditLog(
        user_id=token.user_id,
        role=token.role,
        action=action,
        resource=resource,
        detail=detail,
        ip_address=ip,
    ))
