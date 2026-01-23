"""
Cognitive Layer API Endpoints - Phase 3

Research alignment:
- Affective Computing: Frustration detection and response
- Metacognition: Self-monitoring and reflection
- Just-in-Time Interventions: Adaptive support

Endpoints:
- /frustration: Detect and respond to learner frustration
- /metacognition: Confidence ratings, self-explanation analysis
- /interventions: Get and manage learning interventions
- /calibration: Track accuracy of self-assessment
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import logging

from app.adaptive.cognitive import (
    # Frustration
    FrustrationDetector,
    FrustrationLevel,
    StruggleType,
    InteractionEvent,
    get_frustration_detector,
    # Metacognition
    MetacognitionPrompter,
    CalibrationTracker,
    SelfExplanationAnalyzer,
    ConfidenceRating,
    CalibrationLevel,
    get_metacognition_prompter,
    get_calibration_tracker,
    get_explanation_analyzer,
    # Interventions
    InterventionEngine,
    LearnerState,
    FrustrationEstimate,
    FrustrationIndicators,
    get_intervention_engine,
)
from app.schemas.cognitive import ObservationRequest, ObservationResponse
from app.services.cognitive.observer_agent import MetacognitiveObserver, get_observer_agent

logger = logging.getLogger(__name__)

router = APIRouter()


# ============== Request/Response Models ==============

class InteractionEventInput(BaseModel):
    """Input for a single interaction event"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: str = Field(..., description="Event type: answer, click, navigation, hint, pause")
    correct: Optional[bool] = Field(default=None, description="Whether answer was correct")
    response_time_ms: Optional[int] = Field(default=None, ge=0)
    content_id: Optional[str] = None
    hint_used: bool = False
    attempts: int = Field(default=1, ge=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FrustrationDetectionRequest(BaseModel):
    """Request for frustration detection"""
    user_id: str = Field(..., description="User identifier")
    events: List[InteractionEventInput] = Field(..., min_length=1)
    context: Optional[Dict[str, Any]] = None


class FrustrationResponse(BaseModel):
    """Response with frustration detection results"""
    level: str
    score: float
    struggle_type: str
    active_signals: List[str]
    recommended_action: str
    confidence: float
    indicators: Dict[str, Any]


class ConfidenceRatingInput(BaseModel):
    """Input for recording a confidence rating"""
    user_id: str
    concept_id: str
    content_id: str
    confidence: float = Field(..., ge=0, le=1, description="Confidence level 0-1")
    context: str = Field(default="during_practice", description="pre_test, post_study, during_practice")


class CalibrationRequest(BaseModel):
    """Request for calibration data"""
    user_id: str
    concept_id: Optional[str] = None
    time_window_hours: Optional[int] = Field(default=None, ge=1)


class SelfExplanationInput(BaseModel):
    """Input for self-explanation analysis"""
    explanation_text: str = Field(..., min_length=10)
    concept_name: str
    expected_concepts: List[str] = Field(default_factory=list)
    common_misconceptions: Optional[List[str]] = None


class MetacognitionPromptRequest(BaseModel):
    """Request for metacognition prompt"""
    user_id: str
    concept_name: str
    timing: str = Field(..., description="before, during, or after")
    context: Optional[Dict[str, Any]] = None
    force: bool = False


class LearnerStateInput(BaseModel):
    """Input for learner state"""
    user_id: str
    frustration_score: float = Field(default=0.0, ge=0, le=1)
    frustration_level: str = Field(default="none")
    cognitive_load_score: float = Field(default=0.5, ge=0, le=1)
    cognitive_load_level: str = Field(default="optimal")
    calibration_level: str = Field(default="unknown")
    consecutive_errors: int = Field(default=0, ge=0)
    time_on_task_minutes: float = Field(default=0, ge=0)
    session_duration_minutes: float = Field(default=0, ge=0)
    concepts_mastered_today: int = Field(default=0, ge=0)


class InterventionRequest(BaseModel):
    """Request for intervention decision"""
    learner_state: LearnerStateInput
    events: List[InteractionEventInput] = Field(default_factory=list)
    context: Optional[Dict[str, Any]] = None


# ============== Frustration Detection Endpoints ==============

@router.post("/frustration/detect", response_model=FrustrationResponse)
async def detect_frustration(
    request: FrustrationDetectionRequest,
    detector: FrustrationDetector = Depends(get_frustration_detector)
):
    """
    Detect frustration level from recent interaction events

    Analyzes:
    - Response time patterns
    - Error sequences
    - Help-seeking behavior
    - Navigation patterns

    Returns frustration level, signals, and recommended action.
    """
    try:
        # Convert input to InteractionEvent objects
        events = [
            InteractionEvent(
                timestamp=e.timestamp,
                event_type=e.event_type,
                correct=e.correct,
                response_time_ms=e.response_time_ms,
                content_id=e.content_id,
                hint_used=e.hint_used,
                attempts=e.attempts,
                metadata=e.metadata,
            )
            for e in request.events
        ]

        # Detect frustration
        estimate = detector.detect_frustration(
            user_id=request.user_id,
            events=events,
            context=request.context
        )

        return FrustrationResponse(
            level=estimate.level.value,
            score=estimate.score,
            struggle_type=estimate.struggle_type.value,
            active_signals=[s.value for s in estimate.active_signals],
            recommended_action=estimate.recommended_action,
            confidence=estimate.confidence,
            indicators={
                "response_time_variance": estimate.indicators.response_time_variance,
                "rapid_response_ratio": estimate.indicators.rapid_response_ratio,
                "extended_pause_ratio": estimate.indicators.extended_pause_ratio,
                "consecutive_error_count": estimate.indicators.consecutive_error_count,
                "hint_usage_rate": estimate.indicators.hint_usage_rate,
                "navigation_entropy": estimate.indicators.navigation_entropy,
            }
        )

    except Exception as e:
        logger.error(f"Error detecting frustration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/frustration/update-baseline")
async def update_user_baseline(
    user_id: str,
    events: List[InteractionEventInput],
    detector: FrustrationDetector = Depends(get_frustration_detector)
):
    """
    Update user's baseline metrics for more accurate frustration detection

    Should be called with historical data to calibrate the detector.
    """
    try:
        interaction_events = [
            InteractionEvent(
                timestamp=e.timestamp,
                event_type=e.event_type,
                correct=e.correct,
                response_time_ms=e.response_time_ms,
                content_id=e.content_id,
                hint_used=e.hint_used,
                attempts=e.attempts,
                metadata=e.metadata,
            )
            for e in events
        ]

        detector.update_user_baseline(user_id, interaction_events)

        baseline = detector._get_user_baseline(user_id)
        return {
            "user_id": user_id,
            "baseline_updated": True,
            "baseline": {
                "avg_response_time_ms": baseline.get("avg_rt"),
                "response_time_std": baseline.get("std_rt"),
                "baseline_accuracy": baseline.get("accuracy"),
            }
        }

    except Exception as e:
        logger.error(f"Error updating baseline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Metacognition Endpoints ==============

@router.post("/metacognition/prompt")
async def get_metacognition_prompt(
    request: MetacognitionPromptRequest,
    prompter: MetacognitionPrompter = Depends(get_metacognition_prompter)
):
    """
    Get a metacognitive prompt for the learner

    Types:
    - confidence_rating: Rate understanding
    - self_explanation: Explain in own words
    - prediction: Predict performance
    - reflection: Reflect on learning
    - strategy_selection: Choose approach
    - error_analysis: Analyze mistakes
    """
    try:
        prompt = prompter.generate_prompt(
            user_id=request.user_id,
            concept_name=request.concept_name,
            timing=request.timing,
            context=request.context,
            force=request.force
        )

        if not prompt:
            return {"prompt": None, "reason": "not_scheduled"}

        return {
            "prompt_id": prompt.prompt_id,
            "prompt_type": prompt.prompt_type.value,
            "prompt_text": prompt.prompt_text,
            "timing": prompt.timing,
            "required": prompt.required,
            "follow_up_prompts": prompt.follow_up_prompts,
        }

    except Exception as e:
        logger.error(f"Error generating prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metacognition/confidence-scale")
async def get_confidence_scale(
    concept_name: str,
    scale_type: str = "numeric",  # numeric, verbal, emoji
    prompter: MetacognitionPrompter = Depends(get_metacognition_prompter)
):
    """
    Get a confidence rating scale for a concept

    Scale types:
    - numeric: 1-5 scale with labels
    - verbal: Text descriptions
    - emoji: Emoji-based scale
    """
    return prompter.generate_confidence_scale(concept_name, scale_type)


@router.post("/metacognition/record-confidence")
async def record_confidence_rating(
    rating: ConfidenceRatingInput,
    tracker: CalibrationTracker = Depends(get_calibration_tracker)
):
    """
    Record a learner's confidence rating for calibration tracking
    """
    try:
        confidence_rating = ConfidenceRating(
            user_id=rating.user_id,
            concept_id=rating.concept_id,
            content_id=rating.content_id,
            confidence=rating.confidence,
            timestamp=datetime.utcnow(),
            context=rating.context,
        )

        tracker.record_rating(confidence_rating)

        return {
            "recorded": True,
            "user_id": rating.user_id,
            "concept_id": rating.concept_id,
            "confidence": rating.confidence,
        }

    except Exception as e:
        logger.error(f"Error recording confidence: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metacognition/update-performance")
async def update_actual_performance(
    user_id: str,
    concept_id: str,
    actual_performance: float,
    tracker: CalibrationTracker = Depends(get_calibration_tracker)
):
    """
    Update actual performance for a confidence rating (for calibration)

    Call this after assessment to link confidence to actual performance.
    """
    try:
        tracker.update_performance(user_id, concept_id, actual_performance)

        return {
            "updated": True,
            "user_id": user_id,
            "concept_id": concept_id,
            "actual_performance": actual_performance,
        }

    except Exception as e:
        logger.error(f"Error updating performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metacognition/analyze-explanation")
async def analyze_self_explanation(
    explanation: SelfExplanationInput,
    analyzer: SelfExplanationAnalyzer = Depends(get_explanation_analyzer)
):
    """
    Analyze quality of a self-explanation

    Returns:
    - Quality score
    - Concepts mentioned
    - Misconceptions detected
    - Feedback for improvement
    """
    try:
        analysis = analyzer.analyze_explanation(
            explanation=explanation.explanation_text,
            concept_name=explanation.concept_name,
            expected_concepts=explanation.expected_concepts,
            common_misconceptions=explanation.common_misconceptions,
        )

        return analysis

    except Exception as e:
        logger.error(f"Error analyzing explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Calibration Endpoints ==============

@router.post("/calibration/calculate")
async def calculate_calibration(
    request: CalibrationRequest,
    tracker: CalibrationTracker = Depends(get_calibration_tracker)
):
    """
    Calculate learner's metacognitive calibration

    Calibration = alignment between confidence and actual performance
    - Well-calibrated: Confidence ≈ Performance
    - Overconfident: Confidence > Performance
    - Underconfident: Confidence < Performance
    """
    try:
        from datetime import timedelta

        time_window = None
        if request.time_window_hours:
            time_window = timedelta(hours=request.time_window_hours)

        calibration = tracker.calculate_calibration(
            user_id=request.user_id,
            concept_id=request.concept_id,
            time_window=time_window,
        )

        return {
            "user_id": request.user_id,
            "calibration_level": calibration.calibration_level.value,
            "mean_confidence": calibration.mean_confidence,
            "mean_performance": calibration.mean_performance,
            "calibration_error": calibration.calibration_error,
            "overconfidence_rate": calibration.overconfidence_rate,
            "underconfidence_rate": calibration.underconfidence_rate,
            "data_points": len(calibration.confidence_ratings),
        }

    except Exception as e:
        logger.error(f"Error calculating calibration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calibration/feedback")
async def get_calibration_feedback(
    request: CalibrationRequest,
    tracker: CalibrationTracker = Depends(get_calibration_tracker)
):
    """
    Get personalized feedback to improve calibration
    """
    try:
        from datetime import timedelta

        time_window = None
        if request.time_window_hours:
            time_window = timedelta(hours=request.time_window_hours)

        calibration = tracker.calculate_calibration(
            user_id=request.user_id,
            concept_id=request.concept_id,
            time_window=time_window,
        )

        feedback = tracker.generate_calibration_feedback(calibration)

        return feedback

    except Exception as e:
        logger.error(f"Error generating feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Intervention Endpoints ==============

@router.post("/intervention/decide")
async def decide_intervention(
    request: InterventionRequest,
    engine: InterventionEngine = Depends(get_intervention_engine)
):
    """
    Decide whether to show an intervention based on learner state

    Returns intervention if appropriate, or null if none needed.
    """
    try:
        # Convert events
        events = [
            InteractionEvent(
                timestamp=e.timestamp,
                event_type=e.event_type,
                correct=e.correct,
                response_time_ms=e.response_time_ms,
                content_id=e.content_id,
                hint_used=e.hint_used,
                attempts=e.attempts,
                metadata=e.metadata,
            )
            for e in request.events
        ]

        # Build frustration estimate from input
        frustration = FrustrationEstimate(
            level=FrustrationLevel(request.learner_state.frustration_level),
            score=request.learner_state.frustration_score,
            struggle_type=StruggleType.NONE,
            indicators=FrustrationIndicators(),
            active_signals=[],
            confidence=0.7,
            recommended_action="continue",
        )

        # Build learner state
        learner_state = LearnerState(
            user_id=request.learner_state.user_id,
            frustration=frustration,
            cognitive_load_score=request.learner_state.cognitive_load_score,
            cognitive_load_level=request.learner_state.cognitive_load_level,
            calibration_level=CalibrationLevel(request.learner_state.calibration_level),
            consecutive_errors=request.learner_state.consecutive_errors,
            time_on_task_minutes=request.learner_state.time_on_task_minutes,
            session_duration_minutes=request.learner_state.session_duration_minutes,
            concepts_mastered_today=request.learner_state.concepts_mastered_today,
        )

        # Get decision
        decision = engine.decide_intervention(
            learner_state=learner_state,
            events=events,
            context=request.context,
        )

        if decision.should_intervene and decision.intervention:
            intervention = decision.intervention
            return {
                "should_intervene": True,
                "intervention": {
                    "intervention_id": intervention.intervention_id,
                    "type": intervention.intervention_type.value,
                    "priority": intervention.priority.value,
                    "title": intervention.title,
                    "message": intervention.message,
                    "action": intervention.action,
                    "action_data": intervention.action_data,
                    "display_duration_seconds": intervention.display_duration_seconds,
                    "dismissible": intervention.dismissible,
                    "follow_ups": intervention.follow_ups,
                },
                "reason": decision.reason,
                "cooldown_seconds": decision.cooldown_seconds,
            }
        else:
            return {
                "should_intervene": False,
                "intervention": None,
                "reason": decision.reason,
                "cooldown_seconds": decision.cooldown_seconds,
            }

    except Exception as e:
        logger.error(f"Error deciding intervention: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/intervention/history/{user_id}")
async def get_intervention_history(
    user_id: str,
    engine: InterventionEngine = Depends(get_intervention_engine)
):
    """
    Get intervention history and effectiveness for a user
    """
    try:
        effectiveness = engine.get_intervention_effectiveness(user_id)
        return effectiveness

    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Combined Cognitive Profile ==============

@router.get("/profile/{user_id}")
async def get_cognitive_profile(
    user_id: str,
    detector: FrustrationDetector = Depends(get_frustration_detector),
    tracker: CalibrationTracker = Depends(get_calibration_tracker),
    engine: InterventionEngine = Depends(get_intervention_engine),
):
    """
    Get comprehensive cognitive profile for a user

    Combines:
    - Frustration patterns
    - Metacognitive calibration
    - Intervention history
    """
    try:
        # Get baseline
        baseline = detector._get_user_baseline(user_id)

        # Get calibration
        from datetime import timedelta
        calibration = tracker.calculate_calibration(
            user_id=user_id,
            time_window=timedelta(days=7)
        )

        # Get intervention stats
        intervention_stats = engine.get_intervention_effectiveness(user_id)

        return {
            "user_id": user_id,
            "baseline": {
                "avg_response_time_ms": baseline.get("avg_rt"),
                "response_time_std": baseline.get("std_rt"),
                "baseline_accuracy": baseline.get("accuracy"),
            },
            "calibration": {
                "level": calibration.calibration_level.value,
                "mean_confidence": calibration.mean_confidence,
                "mean_performance": calibration.mean_performance,
                "error": calibration.calibration_error,
            },
            "interventions": intervention_stats,
        }

    except Exception as e:
        logger.error(f"Error getting profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== Observer Agent Endpoints ==============

@router.post("/observe", response_model=ObservationResponse)
async def observe_behavior(
    request: ObservationRequest,
    observer: MetacognitiveObserver = Depends(get_observer_agent)
):
    """
    Observer Agent: Analyze recent events for unproductive patterns.
    Detects:
    - Gaming the system
    - Wheel spinning
    """
    try:
        response = await observer.observe(request)
        return response
    except Exception as e:
        logger.error(f"Error in observer agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

