"""
schemas/session.py
------------------
Pydantic shapes for session, assessment, and recommendation endpoints.
"""


from datetime import datetime
from typing import Any

from pydantic import BaseModel


#  Metrics 

class SpeechMetrics(BaseModel):
    ciu_rate: float = 0.0
    mc_score: float = 0.0
    mlu_morphemes: float = 0.0
    ttr: float = 0.0
    syntactic_complexity: float = 0.0
    mean_surprisal: float | None = None


#  Assessment 

class AssessmentResult(BaseModel):
    session_id: str
    metrics: SpeechMetrics
    transcript: str | None = None


#  Recommendation 

class Exercise(BaseModel):
    id: str
    name: str
    description: str | None = None
    context: str | None = None


class Explanation(BaseModel):
    plain_explanation: str
    primary_reason: str | None = None
    metric_analysis: dict[str, Any] | None = None
    confidence_rationale: str | None = None


class RecommendationResponse(BaseModel):
    exercise: Exercise
    explanation: Explanation
    confidence: str  # "High" | "Moderate" | "Low"
    plain_explanation: str  # convenience alias


#  Feedback 

class FeedbackSummary(BaseModel):
    overall: str  # "excellent" | "good" | "stable" | "needs_practice"
    emoji: str
    message: str
    encouragement: str
    improved: list[str] = []


class ExerciseFeedback(BaseModel):
    feedback: FeedbackSummary
    metrics_before: SpeechMetrics | None = None
    metrics_after: SpeechMetrics | None = None
    transcript: str | None = None


#  Session 

class SessionResponse(BaseModel):
    id: str
    transcript: str | None
    metrics: SpeechMetrics | None
    recommended_exercise_name: str | None
    recommended_exercise_id: str | None
    explanation: Explanation | None
    confidence: str | None
    exercise_completed: bool
    exercise_feedback: ExerciseFeedback | None
    created_at: datetime

    model_config = {"from_attributes": True}


class SessionListItem(BaseModel):
    id: str
    created_at: datetime
    exercise_completed: bool
    recommended_exercise_name: str | None
    confidence: str | None
    overall_rating: str | None = None

    model_config = {"from_attributes": True}
