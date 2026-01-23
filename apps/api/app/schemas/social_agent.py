from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

# ==================== Coding Challenges ====================

class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class TestCase(BaseModel):
    input: Any
    expected: Any
    description: str

class CodingChallenge(BaseModel):
    challenge_id: str
    title: str
    description: str
    difficulty: DifficultyLevel
    category: str
    concepts: List[str]
    function_name: str
    parameters: List[Dict[str, str]]  # name, type
    return_type: str
    test_cases: List[TestCase]
    estimated_minutes: int
    language: str = "python"

class EvaluationRequest(BaseModel):
    challenge_id: str
    user_id: str
    code: str

class DimensionScore(BaseModel):
    score: float
    feedback: str

class FeedbackItem(BaseModel):
    type: str # praise, issue, suggestion
    message: str
    line_number: Optional[int] = None
    suggestion: Optional[str] = None

class EvaluationResult(BaseModel):
    passed: bool
    overall_score: float
    tests_passed: int
    tests_total: int
    execution_time_ms: float
    dimension_scores: Dict[str, DimensionScore]
    feedback: List[FeedbackItem]
    concepts_demonstrated: List[str]
    concepts_to_review: List[str]
    runtime_errors: List[str] = []

class HintRequest(BaseModel):
    challenge_id: str
    user_id: str
    code: str
    hint_level: str # nudge, guidance, explanation, partial, solution

class HintResponse(BaseModel):
    hint_level: str
    hint: str
    cost: int

# ==================== Debates ====================

class DebateFormat(str, Enum):
    OXFORD = "oxford"
    SOCRATIC = "socratic"
    ROUNDTABLE = "roundtable"
    DEVILS_ADVOCATE = "devils_advocate"
    SYNTHESIS = "synthesis"

class PanelPreset(str, Enum):
    TECHNICAL_PROS_CONS = "technical_pros_cons"
    PHILOSOPHICAL = "philosophical"
    PRACTICAL_APPLICATION = "practical_application"

class DebateArgument(BaseModel):
    speaker: str
    role: str
    content: str
    argument_type: str
    key_points: List[str]
    timestamp: datetime

class DebateSessionResponse(BaseModel):
    session_id: str
    topic: str
    format: DebateFormat
    participants: List[Dict[str, str]] # name, role
    current_round: int
    max_rounds: int
    opening_statements: List[DebateArgument]

class AdvanceDebateRequest(BaseModel):
    session_id: str
    learner_contribution: Optional[str] = None

class DebateRoundResponse(BaseModel):
    session_id: str
    current_round: int
    arguments: List[DebateArgument]
    completed: bool

class DebateSummary(BaseModel):
    topic: str
    executive_summary: str
    total_rounds: int
    total_arguments: int
    participants: List[str]
    key_insights: List[str]
    consensus_points: List[str]
    disagreement_points: List[str]
    learning_takeaways: List[str]

class StartDebateRequest(BaseModel):
    topic: str
    format: DebateFormat = DebateFormat.ROUNDTABLE
    panel_preset: PanelPreset = PanelPreset.TECHNICAL_PROS_CONS
    learner_id: Optional[str] = None
    max_rounds: int = 5

# ==================== Teaching ====================

class TeachingSessionStartRequest(BaseModel):
    user_id: str
    concept_id: str
    concept_name: str
    persona: str
    concept_description: Optional[str] = None

class TeachingSessionResponse(BaseModel):
    session_id: str
    persona: str
    opening_question: str
    comprehension: float
    comprehension_level: str

class ExplanationRequest(BaseModel):
    session_id: str
    explanation: str
    concept_description: Optional[str] = None

class TeachingResponse(BaseModel):
    response: str
    question_type: Optional[str] = None
    comprehension: float
    comprehension_level: str
    knowledge_gaps: List[str]
    concepts_understood: List[str]

class TeachingSessionSummary(BaseModel):
    concept: str
    teaching_effectiveness: float
    final_comprehension: float
    total_exchanges: int
    duration_minutes: float
    recommendations: List[str]
    knowledge_gaps_identified: List[str]
    strong_explanations: List[str]
