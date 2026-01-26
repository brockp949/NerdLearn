"""
Pydantic schemas for Counterfactual Explanations API

Request and response models for counterfactual analysis,
recourse planning, SHAP explanations, and Socratic dialogue.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


# Enums

class DialogueModeEnum(str, Enum):
    """Types of Socratic dialogue modes."""
    RETROSPECTIVE = "retrospective"
    PROSPECTIVE = "prospective"
    EXPLORATORY = "exploratory"
    CHALLENGE = "challenge"


class FeatureConstraintType(str, Enum):
    """Types of feature constraints for recourse."""
    IMMUTABLE = "immutable"
    ACTIONABLE_LOW_EFFORT = "actionable_low_effort"
    ACTIONABLE_HIGH_EFFORT = "actionable_high_effort"


# Base Models

class AttemptRecord(BaseModel):
    """Record of a single learning attempt."""
    question_id: Optional[str] = None
    concept_id: str
    correct: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    difficulty: Optional[float] = Field(None, ge=0, le=1)
    response_time_seconds: Optional[float] = None
    question_type: Optional[str] = None


class StudySession(BaseModel):
    """Record of a study session."""
    session_id: Optional[str] = None
    concept_id: str
    duration_minutes: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    gap_hours: Optional[float] = None
    activity_type: Optional[str] = None


class BeliefStateData(BaseModel):
    """Serialized belief state data."""
    mastery: float = Field(..., ge=0, le=1)
    uncertainty: Optional[float] = Field(None, ge=0)
    half_life_hours: Optional[float] = None
    last_update: Optional[datetime] = None


# Counterfactual Requests/Responses

class CounterfactualExplanationRequest(BaseModel):
    """Request for counterfactual explanation."""
    student_id: str
    concept_id: str
    concept_name: str
    intervention: Dict[str, Any] = Field(
        ...,
        description="The intervention to analyze, e.g., {'study_time': 30}",
        examples=[{"study_time": 30}, {"practice_sessions": 5}]
    )
    question: Optional[str] = Field(
        None,
        description="Natural language question, e.g., 'What if I had studied longer?'"
    )
    belief_state: BeliefStateData
    recent_attempts: List[AttemptRecord] = []
    study_sessions: List[StudySession] = []
    target_mastery: float = Field(0.85, ge=0, le=1)


class FeatureContributionResponse(BaseModel):
    """SHAP feature contribution."""
    feature: str
    value: float
    contribution: float
    interpretation: Optional[str] = None


class CounterfactualResultResponse(BaseModel):
    """Counterfactual computation result."""
    original_probability: float = Field(..., ge=0, le=1)
    counterfactual_probability: float = Field(..., ge=0, le=1)
    probability_change: float
    inferred_exogenous: Optional[Dict[str, float]] = None
    explanation: Optional[str] = None


class RecourseActionResponse(BaseModel):
    """Single recourse action."""
    action: str
    feature: Optional[str] = None
    original_value: Optional[float] = None
    target_value: Optional[float] = None
    effort: float
    expected_impact: float


class RecoursePlanResponse(BaseModel):
    """Complete recourse plan."""
    actions: List[RecourseActionResponse]
    total_effort: float
    time_estimate_minutes: float
    expected_probability: float = Field(..., ge=0, le=1)


class CounterfactualExplanationResponse(BaseModel):
    """Full counterfactual explanation response."""
    query: str
    counterfactual_result: CounterfactualResultResponse
    shap_explanation: Optional[Dict[str, Any]] = None
    recourse_plan: Optional[RecoursePlanResponse] = None
    natural_language_explanation: str


# Retrospective Analysis

class CriticalDecisionResponse(BaseModel):
    """Critical decision point identified in retrospective analysis."""
    timestamp: Optional[str] = None
    description: str
    impact_score: float
    alternative_outcome: Optional[str] = None


class RetrospectiveAnalysisRequest(BaseModel):
    """Request for retrospective analysis."""
    student_id: str
    concept_id: str
    concept_name: str
    belief_state: BeliefStateData
    recent_attempts: List[AttemptRecord]
    study_sessions: List[StudySession] = []
    target_mastery: float = Field(0.85, ge=0, le=1)


class SuggestedCounterfactual(BaseModel):
    """Suggested counterfactual query."""
    intervention: Dict[str, Any]
    question: str


class RetrospectiveAnalysisResponse(BaseModel):
    """Retrospective analysis response."""
    status: str
    concept_id: str
    concept_name: str
    current_mastery: float
    num_failures: Optional[int] = None
    critical_decisions: List[CriticalDecisionResponse] = []
    shap_explanation: Optional[Dict[str, Any]] = None
    suggested_counterfactuals: List[SuggestedCounterfactual] = []
    message: Optional[str] = None


# Prospective Planning

class ProspectivePlanRequest(BaseModel):
    """Request for prospective planning."""
    student_id: str
    concept_id: str
    concept_name: str
    belief_state: BeliefStateData
    recent_attempts: List[AttemptRecord] = []
    study_sessions: List[StudySession] = []
    time_budget_minutes: Optional[float] = None
    target_probability: float = Field(0.85, ge=0, le=1)


class SHAPFactorResponse(BaseModel):
    """SHAP factor summary."""
    feature: str
    contribution: float


class ProspectivePlanResponse(BaseModel):
    """Prospective plan response."""
    status: str
    concept_id: str
    concept_name: str
    current_mastery: float
    target_probability: float
    time_budget: Optional[float] = None
    recourse_plan: RecoursePlanResponse
    shap_factors: Dict[str, List[SHAPFactorResponse]]


# Recommendation Explanation

class ExplainRecommendationRequest(BaseModel):
    """Request to explain a recommendation."""
    student_id: str
    concept_id: str
    concept_name: str
    recommended_arm: str
    arm_values: Dict[str, float]
    belief_state: BeliefStateData
    recent_attempts: List[AttemptRecord] = []
    study_sessions: List[StudySession] = []


class AlternativeArmResponse(BaseModel):
    """Alternative arm option."""
    arm: str
    value: float


class ExplainRecommendationResponse(BaseModel):
    """Recommendation explanation response."""
    recommended_arm: str
    explanation: Dict[str, Any]
    natural_language: str
    alternative_arms: List[AlternativeArmResponse]


# Path Comparison

class ComparePathsRequest(BaseModel):
    """Request to compare learning paths."""
    student_id: str
    concept_id: str
    concept_name: str
    path_a: List[str] = Field(..., description="First learning path (concept IDs)")
    path_b: List[str] = Field(..., description="Second learning path (concept IDs)")
    belief_state: BeliefStateData
    recent_attempts: List[AttemptRecord] = []
    study_sessions: List[StudySession] = []
    num_simulations: int = Field(100, ge=10, le=1000)


class PathOutcomeResponse(BaseModel):
    """Outcome for a single path."""
    sequence: List[str]
    expected_mastery: float
    success_probability: float


class SequenceContributionResponse(BaseModel):
    """Contribution analysis for sequence item."""
    item: str
    value_contribution: float
    position_contribution: float


class ComparePathsResponse(BaseModel):
    """Path comparison response."""
    path_a: PathOutcomeResponse
    path_b: PathOutcomeResponse
    difference: Dict[str, float]
    sequence_analysis: Dict[str, List[SequenceContributionResponse]]
    recommendation: str


# Socratic Dialogue

class StartDialogueRequest(BaseModel):
    """Request to start Socratic dialogue."""
    student_id: str
    concept_id: str
    concept_name: str
    mode: DialogueModeEnum = DialogueModeEnum.EXPLORATORY
    belief_state: BeliefStateData
    recent_attempts: List[AttemptRecord] = []
    study_sessions: List[StudySession] = []
    target_mastery: float = Field(0.85, ge=0, le=1)


class DialogueTurnResponse(BaseModel):
    """Single dialogue turn."""
    role: str
    content: str
    timestamp: str


class StartDialogueResponse(BaseModel):
    """Dialogue session response."""
    session_id: str
    mode: str
    concept_id: str
    concept_name: str
    current_mastery: float
    target_mastery: float
    turns: List[DialogueTurnResponse]


class ContinueDialogueRequest(BaseModel):
    """Request to continue dialogue."""
    session_id: str
    student_message: str


class ContinueDialogueResponse(BaseModel):
    """Continued dialogue response."""
    session_id: str
    turns: List[DialogueTurnResponse]


class DialogueSummaryResponse(BaseModel):
    """Dialogue summary response."""
    session_id: str
    summary: str


# What-If Scenario

class WhatIfRequest(BaseModel):
    """Request for what-if scenario analysis."""
    student_id: str
    concept_id: str
    concept_name: str
    scenarios: List[Dict[str, Any]] = Field(
        ...,
        description="List of intervention scenarios to evaluate",
        examples=[[{"study_time": 30}, {"study_time": 60}]]
    )
    belief_state: BeliefStateData
    recent_attempts: List[AttemptRecord] = []


class ScenarioOutcomeResponse(BaseModel):
    """Outcome for a single what-if scenario."""
    intervention: Dict[str, Any]
    original_probability: float
    counterfactual_probability: float
    probability_change: float
    feasibility_score: Optional[float] = None


class WhatIfResponse(BaseModel):
    """What-if analysis response."""
    concept_id: str
    concept_name: str
    current_mastery: float
    scenarios: List[ScenarioOutcomeResponse]
    best_scenario: Optional[Dict[str, Any]] = None
    recommendation: Optional[str] = None


# Time-Constrained Recourse

class TimeConstrainedRecourseRequest(BaseModel):
    """Request for time-constrained recourse planning."""
    student_id: str
    concept_id: str
    concept_name: str
    belief_state: BeliefStateData
    time_budget_minutes: float = Field(..., gt=0)
    target_probability: float = Field(0.85, ge=0, le=1)
    recent_attempts: List[AttemptRecord] = []
    study_sessions: List[StudySession] = []


class TimeConstrainedRecourseResponse(BaseModel):
    """Time-constrained recourse response."""
    concept_id: str
    time_budget_minutes: float
    target_probability: float
    achievable: bool
    recourse_plan: Optional[RecoursePlanResponse] = None
    max_achievable_probability: float
    time_to_target: Optional[float] = None
    message: str
