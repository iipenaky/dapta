"""
routers/sessions.py
-------------------
Session CRUD endpoints.
"""



from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.dependencies import get_current_active_user, get_db
from models.user import User
from schemas.session import SessionListItem, SessionResponse
from services.session_service import SessionService

router = APIRouter()


@router.get(
    "/",
    response_model=list[SessionListItem],
    summary="List all sessions for the current user",
)
async def list_sessions(
    limit: int = 20,
    offset: int = 0,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list:
    sessions = await SessionService(db).list_for_user(user.id, limit, offset)
    return [
        SessionListItem(
            id=s.id,
            created_at=s.created_at,
            exercise_completed=s.exercise_completed,
            recommended_exercise_name=s.recommended_exercise_name,
            confidence=s.confidence,
            overall_rating=(
                s.exercise_feedback.get("feedback", {}).get("overall")
                if s.exercise_feedback else None
            ),
        )
        for s in sessions
    ]


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Get a single session by ID",
)
async def get_session(
    session_id: str,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    session = await SessionService(db).get_by_id(session_id, user.id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    return session


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a session",
)
async def delete_session(
    session_id: str,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    svc = SessionService(db)
    session = await svc.get_by_id(session_id, user.id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    await db.delete(session)
    await db.commit()
    return {"detail": "Session deleted."}