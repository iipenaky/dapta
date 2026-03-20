"""
core/dependencies.py
--------------------
FastAPI dependency injection providers.

Import these with Depends() in route handlers.
They handle auth, DB sessions, and the DAPTA system singleton.
"""


from typing import AsyncGenerator

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from core.security import decode_token
from database import AsyncSessionLocal
from models.user import User
from services.user_service import UserService

_bearer = HTTPBearer(auto_error=False)


# ── Database session ──────────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async SQLAlchemy session; always closes on exit."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── Current user ──────────────────────────────────────────────────────────────

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Resolve the JWT bearer token in the Authorization header to a User row.
    Raises 401 if the token is missing, invalid, or the user no longer exists.
    """
    _401 = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if credentials is None:
        raise _401

    user_id = decode_token(credentials.credentials)
    if user_id is None:
        raise _401

    user = await UserService(db).get_by_id(user_id)
    if user is None:
        raise _401

    return user


async def get_current_active_user(
    user: User = Depends(get_current_user),
) -> User:
    """Like get_current_user but also enforces the is_active flag."""
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated.",
        )
    return user
