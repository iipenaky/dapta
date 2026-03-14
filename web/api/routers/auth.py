"""
routers/auth.py
---------------
Authentication endpoints: signup, login, refresh, me, update profile.
"""



from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.dependencies import get_current_active_user, get_db
from core.security import create_access_token, create_refresh_token, decode_refresh_token
from models.user import User
from schemas.auth import (
    LoginRequest,
    RefreshRequest,
    SignupRequest,
    TokenResponse,
    UpdateProfileRequest,
    UserResponse,
)
from services.user_service import UserService

router = APIRouter()


@router.post(
    "/signup",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new account",
)
async def signup(
    body: SignupRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    svc = UserService(db)

    if await svc.email_exists(body.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    user = await svc.create(body)

    return TokenResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Log in and receive tokens",
)
async def login(
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    svc = UserService(db)
    user = await svc.authenticate(body.email, body.password)

    if user is None:
        # Same error for wrong email OR wrong password — don't leak which
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    return TokenResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Exchange a refresh token for a new token pair",
)
async def refresh(
    body: RefreshRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    user_id = decode_refresh_token(body.refresh_token)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token.",
        )

    user = await UserService(db).get_by_id(user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated.",
        )

    return TokenResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get the authenticated user's profile",
)
async def me(
    user: User = Depends(get_current_active_user),
) -> User:
    return user


@router.patch(
    "/me",
    response_model=UserResponse,
    summary="Update clinical profile",
)
async def update_profile(
    body: UpdateProfileRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> User:
    return await UserService(db).update_profile(user, body)


@router.post(
    "/logout",
    status_code=status.HTTP_200_OK,
    summary="Log out (client should discard tokens)",
)
async def logout(
    _: User = Depends(get_current_active_user),
) -> dict:
    # JWT is stateless — true revocation requires a token denylist (Redis).
    # For now, the client discards the token. Add Redis denylist for production.
    return {"detail": "Logged out successfully."}