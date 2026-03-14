"""
routers/recommendations.py
---------------------------
Recommendation and exercise submission endpoints.
"""


import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.dependencies import get_current_active_user, get_db
from models.user import User
from schemas.session import ExerciseFeedback, FeedbackSummary, RecommendationResponse, SpeechMetrics
from services.dapta_service import dapta_service
from services.session_service import SessionService

router = APIRouter()


@router.get(
    "/{session_id}",
    response_model=RecommendationResponse,
    summary="Get an exercise recommendation for a session",
)
async def get_recommendation(
    session_id: str,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> RecommendationResponse:
    svc = SessionService(db)
    session = await svc.get_by_id(session_id, user.id)

    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    if session.metrics is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session has no assessment yet. Run assessment first.",
        )

    user_profile = {
        "aphasia_subtype": user.aphasia_subtype or "Other",
        "wab_aq": user.wab_aq or 50.0,
        "months_post_onset": user.months_post_onset or 12.0,
    }

    rec = dapta_service.get_recommendation(session.metrics, user_profile)

    await svc.save_recommendation(
        session,
        exercise_id=rec["exercise"]["id"],
        exercise_name=rec["exercise"]["name"],
        explanation=rec["explanation"],
        confidence=rec["confidence"],
    )

    return RecommendationResponse(
        exercise=rec["exercise"],
        explanation=rec["explanation"],
        confidence=rec["confidence"],
        plain_explanation=rec["plain_explanation"],
    )


@router.post(
    "/{session_id}/submit-audio",
    response_model=ExerciseFeedback,
    summary="Submit exercise audio response and receive feedback",
)
async def submit_audio(
    session_id: str,
    audio: UploadFile = File(...),
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ExerciseFeedback:
    svc = SessionService(db)
    session = await svc.get_by_id(session_id, user.id)

    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    if session.metrics is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No assessment found.")

    # Save audio to temp file, transcribe, re-assess
    suffix = Path(audio.filename or "exercise.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        # audio -> Whisper -> .cha -> DAE (consistent with training pipeline)
        metrics_after_raw, transcript = dapta_service.extract_metrics_from_audio(tmp_path)
        if not transcript:
            metrics_after_raw = session.metrics
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    feedback_raw = dapta_service.compute_feedback(session.metrics, metrics_after_raw)
    await svc.save_feedback(session, feedback_raw, metrics_after_raw, transcript)

    metrics_before = SpeechMetrics(**{k: v for k, v in session.metrics.items() if k in SpeechMetrics.model_fields})
    metrics_after  = SpeechMetrics(**{k: v for k, v in metrics_after_raw.items() if k in SpeechMetrics.model_fields})

    return ExerciseFeedback(
        feedback=FeedbackSummary(**feedback_raw["feedback"]),
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        transcript=transcript or None,
    )
