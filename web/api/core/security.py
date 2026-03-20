"""
core/security.py
----------------
Password hashing and JWT creation / verification.
Nothing in here touches the database — pure crypto utilities.
"""


from datetime import datetime, timedelta, timezone
from typing import Optional
import hashlib
from jose import JWTError, jwt
from passlib.context import CryptContext

from core.config import settings

# bcrypt context — auto-upgrades cost factor when loaded
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── Passwords ─────────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    """Return a bcrypt hash of *plain*."""
    # Truncate the string to 72 chars to ensure it fits in bcrypt bytes
    # Most users won't have 72+ char passwords, but this prevents the crash
    truncated_plain = plain[:72] 
    return _pwd_context.hash(truncated_plain)

def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plain string against a bcrypt hash."""
    return _pwd_context.verify(plain[:72], hashed)


# ── JWT ───────────────────────────────────────────────────────────────────────

def create_access_token(subject: str) -> str:
    """
    Create a short-lived JWT access token.

    Parameters
    ----------
    subject : str
        The user's UUID (stored in the ``sub`` claim).
    """
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes
    )
    payload = {"sub": subject, "exp": expire, "type": "access"}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def create_refresh_token(subject: str) -> str:
    """Create a long-lived JWT refresh token."""
    expire = datetime.now(timezone.utc) + timedelta(
        days=settings.refresh_token_expire_days
    )
    payload = {"sub": subject, "exp": expire, "type": "refresh"}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_token(token: str) -> Optional[str]:
    """
    Decode a JWT and return its ``sub`` claim.

    Returns None if the token is invalid or expired rather than raising,
    so callers can decide how to handle it.
    """
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.algorithm]
        )
        return payload.get("sub")
    except JWTError:
        return None


def decode_refresh_token(token: str) -> Optional[str]:
    """Decode a refresh token; returns sub only if type claim matches."""
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.algorithm]
        )
        if payload.get("type") != "refresh":
            return None
        return payload.get("sub")
    except JWTError:
        return None
