"""
models/session.py
-----------------
Therapy session ORM model.
"""

from __future__ import annotations
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Assessment
    transcript: Mapped[str | None] = mapped_column(String(8000), nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Recommendation
    recommended_exercise_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    recommended_exercise_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    explanation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    confidence: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # Exercise outcome
    exercise_completed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    exercise_feedback: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metrics_after: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="sessions")  # noqa: F821

    def __repr__(self) -> str:
        return f"<Session id={self.id} user_id={self.user_id}>"
