"""
services/session_service.py
----------------------------
All database operations for therapy sessions.
"""


from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from models.session import Session


class SessionService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_by_id(self, session_id: str, user_id: str) -> Optional[Session]:
        """Get a session, verifying it belongs to this user."""
        result = await self.db.execute(
            select(Session).where(
                Session.id == session_id,
                Session.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_for_user(
        self, user_id: str, limit: int = 20, offset: int = 0
    ) -> list[Session]:
        result = await self.db.execute(
            select(Session)
            .where(Session.user_id == user_id)
            .order_by(desc(Session.created_at))
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def create(self, user_id: str) -> Session:
        session = Session(user_id=user_id)
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        return session

    async def save_assessment(
        self,
        session: Session,
        metrics: dict,
        transcript: Optional[str] = None,
    ) -> Session:
        session.metrics = metrics
        session.transcript = transcript
        await self.db.commit()
        await self.db.refresh(session)
        return session

    async def save_recommendation(
        self,
        session: Session,
        exercise_id: str,
        exercise_name: str,
        explanation: dict,
        confidence: str,
    ) -> Session:
        session.recommended_exercise_id = exercise_id
        session.recommended_exercise_name = exercise_name
        session.explanation = explanation
        session.confidence = confidence
        await self.db.commit()
        await self.db.refresh(session)
        return session

    async def save_feedback(
        self,
        session: Session,
        feedback: dict,
        metrics_after: Optional[dict] = None,
        transcript: Optional[str] = None,
    ) -> Session:
        session.exercise_completed = True
        session.exercise_feedback = feedback
        session.metrics_after = metrics_after
        if transcript:
            session.transcript = transcript
        session.completed_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(session)
        return session
