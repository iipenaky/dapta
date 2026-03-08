"""
routers/assessment.py
----------------------
Assessment endpoints: text, file upload (.cha or audio), audio blob.
All three input modes produce the same AssessmentResult shape.
"""

from __future__ import annotations
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.dependencies import get_current_active_user, get_db
from models.user import User
from schemas.session import AssessmentResult, SpeechMetrics
from services.dapta_service import dapta_service
from services.session_service import SessionService

router = APIRouter()

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".mp4", ".m4a", ".webm", ".ogg"}
_CHA_EXTENSION = ".cha"


@router.post(
    "/text",
    response_model=AssessmentResult,
    status_code=status.HTTP_201_CREATED,
    summary="Assess from typed or pasted text",
)
async def assess_from_text(
    text: str = Form(..., min_length=10),
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> AssessmentResult:
    metrics_raw = dapta_service.extract_metrics_from_text(text)
    session = await SessionService(db).create(user.id)
    await SessionService(db).save_assessment(session, metrics_raw, transcript=text)

    return AssessmentResult(
        session_id=session.id,
        metrics=SpeechMetrics(**{k: v for k, v in metrics_raw.items() if k in SpeechMetrics.model_fields}),
        transcript=text,
    )


@router.post(
    "/upload",
    response_model=AssessmentResult,
    status_code=status.HTTP_201_CREATED,
    summary="Assess from uploaded .cha transcript or audio file",
)
async def assess_from_upload(
    file: UploadFile = File(...),
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> AssessmentResult:
    suffix = Path(file.filename or "").suffix.lower()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if suffix == _CHA_EXTENSION:
            metrics_raw, transcript = dapta_service.extract_metrics_from_cha(tmp_path)
        elif suffix in _AUDIO_EXTENSIONS:
            # audio -> Whisper -> .cha -> DAE (same pipeline as training data)
            metrics_raw, transcript = dapta_service.extract_metrics_from_audio(tmp_path)
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unsupported file type '{suffix}'. Use .cha or audio files.",
            )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    session = await SessionService(db).create(user.id)
    await SessionService(db).save_assessment(session, metrics_raw, transcript=transcript)

    return AssessmentResult(
        session_id=session.id,
        metrics=SpeechMetrics(**{k: v for k, v in metrics_raw.items() if k in SpeechMetrics.model_fields}),
        transcript=transcript or None,
    )
