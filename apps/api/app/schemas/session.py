from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class SessionStartRequest(BaseModel):
    learner_id: str
    domain: str = "general"
    limit: int = 20

class AnswerRequest(BaseModel):
    session_id: str
    card_id: str
    rating: str  # "again", "hard", "good", "easy" or simplified frontend rating
    dwell_time_ms: int
    hesitation_count: int = 0

class LearningCardResponse(BaseModel):
    card_id: str
    type: str # "concept", "question", "problem"
    title: str
    content: str
    options: Optional[List[Dict[str, Any]]] = None
    correct_option_id: Optional[str] = None
    difficulty: float = 5.0
    context: Optional[Dict[str, Any]] = None

class SessionState(BaseModel):
    session_id: str
    learner_id: str
    current_card: Optional[LearningCardResponse] = None
    cards_reviewed: int = 0
    cards_correct: int = 0
    total_xp_earned: int = 0
    current_streak: int = 0
    zpd_zone: str = "optimal"
    scaffolding_active: List[str] = Field(default_factory=list)
    started_at: datetime
    achievements_unlocked: List[str] = Field(default_factory=list)

class AnswerResponse(BaseModel):
    correct: bool
    xp_earned: int
    new_total_xp: int
    level: int
    level_progress: float
    next_card: Optional[LearningCardResponse] = None
    zpd_zone: str
    zpd_message: str
    scaffolding: Optional[Dict[str, Any]] = None
    achievement_unlocked: Optional[Dict[str, Any]] = None
