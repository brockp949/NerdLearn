"""
Adaptive Learning API Endpoints
Spaced repetition, content recommendations, mastery tracking, cognitive load estimation,
and interleaved practice scheduling

Research alignment:
- FSRS spaced repetition algorithm
- Bayesian Knowledge Tracing
- Zone of Proximal Development regulation
- Cognitive Load Theory with expertise reversal effect
- Interleaved Practice (g=0.42 effect size, hybrid scheduling)
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from app.core.database import get_db
from app.models.spaced_repetition import SpacedRepetitionCard, ReviewLog, Concept
from app.models.assessment import UserConceptMastery
from app.models.course import Course, Module
from app.adaptive.fsrs import FSRSAlgorithm, FSRSCard, Rating
from app.adaptive.bkt import BayesianKnowledgeTracer
from app.adaptive.zpd import ZPDRegulator
from app.adaptive.cognitive_load import (
    CognitiveLoadEstimator,
    CognitiveLoadLevel,
    ExpertiseLevel,
    ResponseMetrics,
)
from app.adaptive.interleaved import (
    InterleavedScheduler,
    PracticeMode,
    PracticeItem,
    ConceptProficiency,
)
from app.gamification.variable_rewards import VariableRewardEngine
from app.services.scaffolding_service import ScaffoldingService, get_scaffolding_service, HintRequest, HintResponse
from pydantic import BaseModel, Field
from enum import Enum

router = APIRouter()

# Initialize adaptive algorithms
fsrs = FSRSAlgorithm()
bkt = BayesianKnowledgeTracer()
zpd = ZPDRegulator()
cognitive_load_estimator = CognitiveLoadEstimator()
interleaved_scheduler = InterleavedScheduler()
vr_engine = VariableRewardEngine()


class ReviewRating(str, Enum):
    """Review rating options"""
    AGAIN = "again"
    HARD = "hard"
    GOOD = "good"
    EASY = "easy"


class ReviewRequest(BaseModel):
    """Request model for submitting a review"""
    card_id: int
    rating: ReviewRating
    review_duration_ms: int


class MasteryUpdateRequest(BaseModel):
    """Request for updating mastery from stealth assessment"""
    user_id: int
    concept_id: int
    evidence_score: float


@router.get("/reviews/due")
async def get_due_reviews(
    user_id: int,
    course_id: int,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """
    Get spaced repetition cards due for review

    Args:
        user_id: User ID
        course_id: Course ID
        limit: Maximum number of cards to return
    """
    # Query due cards
    now = datetime.now()

    result = await db.execute(
        select(SpacedRepetitionCard)
        .where(
            and_(
                SpacedRepetitionCard.user_id == user_id,
                SpacedRepetitionCard.course_id == course_id,
                SpacedRepetitionCard.due <= now
            )
        )
        .order_by(SpacedRepetitionCard.due)
        .limit(limit)
    )

    cards = result.scalars().all()

    # Format response
    due_cards = []
    for card in cards:
        # Get concept name
        concept_result = await db.execute(
            select(Concept).where(Concept.id == card.concept_id)
        )
        concept = concept_result.scalar_one_or_none()

        # Get next intervals prediction
        fsrs_card = FSRSCard(
            concept_id=card.concept_id,
            user_id=card.user_id,
            stability=card.stability,
            difficulty=card.difficulty,
            elapsed_days=card.elapsed_days,
            scheduled_days=card.scheduled_days,
            reps=card.reps,
            lapses=card.lapses,
            state=card.state,
            last_review=card.last_review,
            due=card.due,
        )

        predictions = fsrs.get_next_states(fsrs_card)

        due_cards.append({
            "card_id": card.id,
            "concept_id": card.concept_id,
            "concept_name": concept.name if concept else "Unknown",
            "state": card.state,
            "stability": card.stability,
            "difficulty": card.difficulty,
            "reps": card.reps,
            "lapses": card.lapses,
            "due": card.due.isoformat(),
            "predictions": {
                "again": predictions[Rating.AGAIN],
                "hard": predictions[Rating.HARD],
                "good": predictions[Rating.GOOD],
                "easy": predictions[Rating.EASY],
            }
        })

    return {
        "due_count": len(due_cards),
        "cards": due_cards
    }


@router.post("/reviews/submit")
async def submit_review(
    review: ReviewRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Submit a spaced repetition review

    Updates FSRS parameters and schedules next review
    """
    # Get card
    result = await db.execute(
        select(SpacedRepetitionCard).where(SpacedRepetitionCard.id == review.card_id)
    )
    card = result.scalar_one_or_none()

    if not card:
        raise HTTPException(status_code=404, detail="Card not found")

    # Convert to FSRS card
    fsrs_card = FSRSCard(
        concept_id=card.concept_id,
        user_id=card.user_id,
        stability=card.stability,
        difficulty=card.difficulty,
        elapsed_days=card.elapsed_days,
        scheduled_days=card.scheduled_days,
        reps=card.reps,
        lapses=card.lapses,
        state=card.state,
        last_review=card.last_review,
        due=card.due,
    )

    # Map rating
    rating_map = {
        ReviewRating.AGAIN: Rating.AGAIN,
        ReviewRating.HARD: Rating.HARD,
        ReviewRating.GOOD: Rating.GOOD,
        ReviewRating.EASY: Rating.EASY,
    }
    rating = rating_map[review.rating]

    # Process review
    updated_card, review_log = fsrs.review_card(fsrs_card, rating)

    # Update database card
    card.stability = updated_card.stability
    card.difficulty = updated_card.difficulty
    card.elapsed_days = updated_card.elapsed_days
    card.scheduled_days = updated_card.scheduled_days
    card.reps = updated_card.reps
    card.lapses = updated_card.lapses
    card.state = updated_card.state
    card.last_review = updated_card.last_review
    card.due = updated_card.due

    # Create review log
    log = ReviewLog(
        card_id=card.id,
        user_id=card.user_id,
        rating=rating.value,
        review_time=datetime.now(),
        elapsed_days=updated_card.elapsed_days,
        scheduled_days=updated_card.scheduled_days,
        stability=updated_card.stability,
        difficulty=updated_card.difficulty,
        state=updated_card.state,
        review_duration_ms=review.review_duration_ms,
        source="manual",
    )

    db.add(log)
    await db.commit()
    await db.refresh(card)

    return {
        "card_id": card.id,
        "rating": review.rating,
        "next_review": card.due.isoformat(),
        "interval_days": card.scheduled_days,
        "stability": card.stability,
        "difficulty": card.difficulty,
        "state": card.state,
    }


@router.post("/mastery/update")
async def update_mastery_from_evidence(
    request: MasteryUpdateRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Update concept mastery using Bayesian Knowledge Tracing from stealth assessment evidence

    Args:
        request: Mastery update request with evidence score
    """
    # Get or create mastery record
    result = await db.execute(
        select(UserConceptMastery).where(
            and_(
                UserConceptMastery.user_id == request.user_id,
                UserConceptMastery.concept_id == request.concept_id
            )
        )
    )
    mastery_record = result.scalar_one_or_none()

    if not mastery_record:
        # Create new mastery record
        mastery_record = UserConceptMastery(
            user_id=request.user_id,
            concept_id=request.concept_id,
            mastery_level=bkt.params["p_l0"],  # Initial prior
            practice_count=0,
        )
        db.add(mastery_record)

    # Update mastery using BKT
    current_mastery = mastery_record.mastery_level
    new_mastery, update_details = bkt.update_from_evidence(
        current_mastery,
        request.evidence_score
    )

    mastery_record.mastery_level = new_mastery
    mastery_record.practice_count += 1
    mastery_record.last_practiced = datetime.now()

    await db.commit()
    await db.refresh(mastery_record)

    return {
        "concept_id": request.concept_id,
        "user_id": request.user_id,
        "prior_mastery": current_mastery,
        "evidence_score": request.evidence_score,
        "new_mastery": new_mastery,
        "mastery_change": update_details["mastery_change"],
        "is_mastered": new_mastery >= 0.95,
    }


@router.get("/recommendations")
async def get_content_recommendations(
    user_id: int,
    course_id: int,
    top_n: int = 5,
    db: AsyncSession = Depends(get_db)
):
    """
    Get ZPD-based content recommendations

    Recommends modules that are optimally challenging for the user
    """
    # Get user's concept masteries
    mastery_result = await db.execute(
        select(UserConceptMastery).where(UserConceptMastery.user_id == user_id)
    )
    masteries = mastery_result.scalars().all()

    user_concept_masteries = {
        m.concept_id: m.mastery_level for m in masteries
    }

    # Get available modules
    modules_result = await db.execute(
        select(Module).where(Module.course_id == course_id)
    )
    modules = modules_result.scalars().all()

    # Format modules for ZPD
    available_modules = []
    for module in modules:
        # Get concepts for module (simplified - in production, query from knowledge graph)
        available_modules.append({
            "id": module.id,
            "title": module.title,
            "concepts": [],  # TODO: Get from knowledge graph
            "difficulty": module.duration_minutes / 10 if module.duration_minutes else 5,  # Proxy
        })

    # Get concept prerequisites (simplified)
    concept_prerequisites = {}  # TODO: Query from Neo4j

    # Get recommendations
    recommendations = zpd.recommend_content(
        user_concept_masteries,
        available_modules,
        concept_prerequisites,
        top_n
    )

    return {
        "user_id": user_id,
        "course_id": course_id,
        "recommendations": [
            {
                "module_id": rec.module_id,
                "zpd_score": rec.zpd_score,
                "difficulty": rec.difficulty,
                "estimated_success_rate": rec.estimated_success_rate,
                "rationale": rec.rationale,
            }
            for rec in recommendations
        ]
    }


@router.get("/mastery/{user_id}/course/{course_id}")
async def get_user_mastery(
    user_id: int,
    course_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get user's mastery levels for all concepts in a course
    """
    # Get concepts for course
    concepts_result = await db.execute(
        select(Concept).where(Concept.course_id == course_id)
    )
    concepts = concepts_result.scalars().all()

    # Get user masteries
    mastery_result = await db.execute(
        select(UserConceptMastery).where(
            and_(
                UserConceptMastery.user_id == user_id,
                UserConceptMastery.concept_id.in_([c.id for c in concepts])
            )
        )
    )
    masteries = mastery_result.scalars().all()

    mastery_map = {m.concept_id: m for m in masteries}

    # Format response
    concept_masteries = []
    for concept in concepts:
        mastery = mastery_map.get(concept.id)

        if mastery:
            concept_masteries.append({
                "concept_id": concept.id,
                "concept_name": concept.name,
                "mastery_level": mastery.mastery_level,
                "practice_count": mastery.practice_count,
                "last_practiced": mastery.last_practiced.isoformat() if mastery.last_practiced else None,
                "is_mastered": mastery.mastery_level >= 0.95,
            })
        else:
            concept_masteries.append({
                "concept_id": concept.id,
                "concept_name": concept.name,
                "mastery_level": 0.0,
                "practice_count": 0,
                "last_practiced": None,
                "is_mastered": False,
            })

    return {
        "user_id": user_id,
        "course_id": course_id,
        "concept_count": len(concept_masteries),
        "mastered_count": sum(1 for c in concept_masteries if c["is_mastered"]),
        "avg_mastery": sum(c["mastery_level"] for c in concept_masteries) / len(concept_masteries) if concept_masteries else 0,
        "concepts": concept_masteries,
    }


# ============== Cognitive Load Estimation ==============

class ResponseMetricInput(BaseModel):
    """Input model for a single response metric"""
    response_time_ms: int = Field(..., gt=0, description="Response time in milliseconds")
    correct: bool = Field(..., description="Whether response was correct")
    confidence: Optional[float] = Field(default=None, ge=0, le=1, description="Self-reported confidence")
    hint_used: bool = Field(default=False, description="Whether hints were used")
    attempts: int = Field(default=1, ge=1, description="Number of attempts")
    content_difficulty: float = Field(default=5.0, ge=1, le=10, description="Content difficulty 1-10")


class CognitiveLoadRequest(BaseModel):
    """Request model for cognitive load estimation"""
    recent_metrics: List[ResponseMetricInput] = Field(..., min_length=1, description="Recent response metrics")
    content_difficulty: float = Field(default=5.0, ge=1, le=10, description="Current content difficulty")


class ExpertiseDetectionRequest(BaseModel):
    """Request model for expertise detection"""
    all_metrics: List[ResponseMetricInput] = Field(..., min_length=1, description="Performance history")
    concept_id: Optional[int] = Field(default=None, description="Optional concept filter")


class ScaffoldingRequest(BaseModel):
    """Request model for scaffolding recommendation"""
    expertise_level: str = Field(..., description="Detected expertise level")
    cognitive_load_score: float = Field(..., ge=0, le=1, description="Cognitive load score")
    cognitive_load_level: str = Field(..., description="Cognitive load level")
    content_type: str = Field(default="problem", description="Content type: problem, reading, video")


class InterventionCheckRequest(BaseModel):
    """Request model for intervention check"""
    cognitive_load_score: float = Field(..., ge=0, le=1, description="Current cognitive load score")
    cognitive_load_level: str = Field(..., description="Current cognitive load level")
    consecutive_errors: int = Field(default=0, ge=0, description="Consecutive wrong answers")
    time_on_task_minutes: float = Field(default=0, ge=0, description="Time on current task")
    expertise_level: str = Field(default="intermediate", description="User expertise level")


@router.post("/cognitive-load/estimate")
async def estimate_cognitive_load(request: CognitiveLoadRequest):
    """
    Estimate current cognitive load from recent behavioral metrics

    Research basis: Cognitive Load Theory
    - Response time patterns (primary indicator)
    - Error rates and patterns
    - Hint usage frequency
    - Answer changes

    Returns cognitive load level and breakdown into:
    - Intrinsic load (content complexity)
    - Extraneous load (interface/instruction issues)
    - Germane load (active schema construction)
    """
    # Convert input to ResponseMetrics
    metrics = [
        ResponseMetrics(
            response_time_ms=m.response_time_ms,
            correct=m.correct,
            confidence=m.confidence,
            hint_used=m.hint_used,
            attempts=m.attempts,
            content_difficulty=m.content_difficulty,
        )
        for m in request.recent_metrics
    ]

    # Estimate cognitive load
    estimate = cognitive_load_estimator.estimate_cognitive_load(
        recent_metrics=metrics,
        content_difficulty=request.content_difficulty,
    )

    return {
        "level": estimate.level.value,
        "score": round(estimate.score, 3),
        "load_breakdown": {
            "intrinsic": round(estimate.intrinsic_load, 3),
            "extraneous": round(estimate.extraneous_load, 3),
            "germane": round(estimate.germane_load, 3),
        },
        "confidence": round(estimate.confidence, 3),
        "indicators": {k: round(v, 3) for k, v in estimate.indicators.items()},
        "interpretation": _get_load_interpretation(estimate.level),
    }


@router.post("/cognitive-load/detect-expertise")
async def detect_expertise_level(request: ExpertiseDetectionRequest):
    """
    Detect user's expertise level from performance history

    Research basis: Expertise Reversal Effect (d=0.45-2.99)
    - Scaffolding helps novices
    - Scaffolding can hinder experts

    Uses:
    - Accuracy patterns
    - Response time consistency
    - Learning trajectory
    """
    # Convert input to ResponseMetrics
    metrics = [
        ResponseMetrics(
            response_time_ms=m.response_time_ms,
            correct=m.correct,
            confidence=m.confidence,
            hint_used=m.hint_used,
            attempts=m.attempts,
            content_difficulty=m.content_difficulty,
        )
        for m in request.all_metrics
    ]

    # Detect expertise
    expertise_level, confidence = cognitive_load_estimator.detect_expertise(
        all_metrics=metrics,
        concept_id=request.concept_id,
    )

    return {
        "expertise_level": expertise_level.value,
        "confidence": round(confidence, 3),
        "sample_size": len(metrics),
        "description": _get_expertise_description(expertise_level),
        "scaffolding_recommendation": _get_scaffolding_level(expertise_level),
    }


@router.post("/cognitive-load/scaffolding")
async def get_scaffolding_recommendation(request: ScaffoldingRequest):
    """
    Get adaptive scaffolding recommendation

    Research basis:
    - Expertise reversal effect
    - Backward fading strategy: remove scaffolds from end of problem first
    - Adaptive fading based on expertise detection
    """
    # Convert string to enums
    try:
        expertise = ExpertiseLevel(request.expertise_level)
    except ValueError:
        expertise = ExpertiseLevel.INTERMEDIATE

    try:
        load_level = CognitiveLoadLevel(request.cognitive_load_level)
    except ValueError:
        load_level = CognitiveLoadLevel.OPTIMAL

    # Create estimate object
    from app.adaptive.cognitive_load import CognitiveLoadEstimate
    estimate = CognitiveLoadEstimate(
        level=load_level,
        score=request.cognitive_load_score,
        intrinsic_load=request.cognitive_load_score * 0.5,
        extraneous_load=0.2,
        germane_load=0.3,
        confidence=0.8,
        indicators={},
    )

    # Get recommendation
    recommendation = cognitive_load_estimator.recommend_scaffolding(
        expertise_level=expertise,
        cognitive_load=estimate,
        content_type=request.content_type,
    )

    return {
        "expertise_level": recommendation.expertise_level.value,
        "scaffolding_level": round(recommendation.scaffolding_level, 2),
        "fade_strategy": recommendation.fade_strategy,
        "specific_recommendations": recommendation.specific_recommendations,
        "rationale": recommendation.rationale,
    }


@router.post("/cognitive-load/check-intervention")
async def check_intervention_needed(request: InterventionCheckRequest):
    """
    Check if learning intervention is needed

    Triggers:
    - Cognitive overload detected
    - Multiple consecutive errors
    - Extended time without progress
    """
    # Convert string to enum
    try:
        load_level = CognitiveLoadLevel(request.cognitive_load_level)
    except ValueError:
        load_level = CognitiveLoadLevel.OPTIMAL

    try:
        expertise = ExpertiseLevel(request.expertise_level)
    except ValueError:
        expertise = ExpertiseLevel.INTERMEDIATE

    # Create estimate object
    from app.adaptive.cognitive_load import CognitiveLoadEstimate
    estimate = CognitiveLoadEstimate(
        level=load_level,
        score=request.cognitive_load_score,
        intrinsic_load=request.cognitive_load_score * 0.5,
        extraneous_load=0.2,
        germane_load=0.3,
        confidence=0.8,
        indicators={},
    )

    # Check if intervention needed
    should_intervene, intervention_type = cognitive_load_estimator.should_intervene(
        cognitive_load=estimate,
        consecutive_errors=request.consecutive_errors,
        time_on_task_minutes=request.time_on_task_minutes,
    )

    # Get intervention details
    intervention = cognitive_load_estimator.get_intervention_recommendation(
        intervention_type=intervention_type,
        expertise_level=expertise,
    )

    return {
        "should_intervene": should_intervene,
        "intervention_type": intervention_type,
        "action": intervention["action"],
        "message": intervention["message"],
        "recommendations": intervention["recommendations"],
    }


@router.post("/cognitive-load/optimal-difficulty")
async def get_optimal_difficulty(
    expertise_level: str,
    current_mastery: float,
    cognitive_load_score: float,
    cognitive_load_level: str,
):
    """
    Calculate optimal content difficulty range

    Based on:
    - Expertise level
    - Current mastery
    - Cognitive load state
    - ZPD principles
    """
    # Convert strings to enums
    try:
        expertise = ExpertiseLevel(expertise_level)
    except ValueError:
        expertise = ExpertiseLevel.INTERMEDIATE

    try:
        load_level = CognitiveLoadLevel(cognitive_load_level)
    except ValueError:
        load_level = CognitiveLoadLevel.OPTIMAL

    # Create estimate
    from app.adaptive.cognitive_load import CognitiveLoadEstimate
    estimate = CognitiveLoadEstimate(
        level=load_level,
        score=cognitive_load_score,
        intrinsic_load=cognitive_load_score * 0.5,
        extraneous_load=0.2,
        germane_load=0.3,
        confidence=0.8,
        indicators={},
    )

    # Calculate optimal difficulty
    min_diff, max_diff = cognitive_load_estimator.calculate_optimal_difficulty(
        expertise_level=expertise,
        current_mastery=current_mastery,
        cognitive_load=estimate,
    )

    return {
        "min_difficulty": round(min_diff, 1),
        "max_difficulty": round(max_diff, 1),
        "optimal_difficulty": round((min_diff + max_diff) / 2, 1),
        "rationale": f"Based on {expertise.value} expertise, {current_mastery:.0%} mastery, {load_level.value} cognitive load",
    }


def _get_load_interpretation(level: CognitiveLoadLevel) -> str:
    """Get human-readable interpretation of cognitive load level"""
    interpretations = {
        CognitiveLoadLevel.LOW: "Under-challenged. Consider increasing difficulty or reducing scaffolding.",
        CognitiveLoadLevel.OPTIMAL: "In optimal learning zone. Current challenge level is appropriate.",
        CognitiveLoadLevel.HIGH: "Challenged but managing. Monitor for signs of overload.",
        CognitiveLoadLevel.OVERLOAD: "Cognitive overload detected. Reduce complexity or provide support.",
    }
    return interpretations.get(level, "Unknown")


def _get_expertise_description(level: ExpertiseLevel) -> str:
    """Get description of expertise level"""
    descriptions = {
        ExpertiseLevel.NOVICE: "New to the topic. Benefits from full scaffolding and worked examples.",
        ExpertiseLevel.BEGINNER: "Basic understanding. Benefits from partial scaffolding.",
        ExpertiseLevel.INTERMEDIATE: "Solid grasp of fundamentals. Minimal scaffolding recommended.",
        ExpertiseLevel.ADVANCED: "Strong command of material. Scaffolding may slow learning.",
        ExpertiseLevel.EXPERT: "Expert level. Scaffolding can be counterproductive (expertise reversal).",
    }
    return descriptions.get(level, "Unknown")


def _get_scaffolding_level(level: ExpertiseLevel) -> str:
    """Get recommended scaffolding level for expertise"""
    levels = {
        ExpertiseLevel.NOVICE: "full",
        ExpertiseLevel.BEGINNER: "high",
        ExpertiseLevel.INTERMEDIATE: "moderate",
        ExpertiseLevel.ADVANCED: "minimal",
        ExpertiseLevel.EXPERT: "none",
    }
    return levels.get(level, "moderate")


# ============== Interleaved Practice ==============

class ConceptProficiencyInput(BaseModel):
    """Input model for concept proficiency"""
    concept_id: int = Field(..., description="Concept ID")
    concept_name: str = Field(..., description="Concept name")
    mastery_level: float = Field(..., ge=0, le=1, description="Mastery level 0-1")
    practice_count: int = Field(default=0, ge=0, description="Total practice count")
    recent_accuracy: float = Field(default=0.5, ge=0, le=1, description="Recent accuracy")
    last_practiced: Optional[datetime] = Field(default=None, description="Last practice time")
    stability: float = Field(default=0.0, ge=0, description="FSRS stability")


class PracticeItemInput(BaseModel):
    """Input model for practice item"""
    item_id: str = Field(..., description="Item ID")
    concept_id: int = Field(..., description="Concept ID")
    concept_name: str = Field(..., description="Concept name")
    difficulty: float = Field(default=5.0, ge=1, le=10, description="Difficulty 1-10")
    item_type: str = Field(default="problem", description="Item type")
    content: Optional[Dict] = Field(default=None, description="Item content")


class PracticeSequenceRequest(BaseModel):
    """Request for generating practice sequence"""
    concept_proficiencies: List[ConceptProficiencyInput] = Field(
        ..., min_length=1, description="User's concept proficiencies"
    )
    available_items: Dict[str, List[PracticeItemInput]] = Field(
        ..., description="Available items per concept (concept_id as string key)"
    )
    target_items: int = Field(default=10, ge=1, le=50, description="Target items")
    target_duration_minutes: int = Field(default=15, ge=1, le=60, description="Target duration")
    target_concept_id: Optional[int] = Field(default=None, description="Focus concept")


class PracticeModeRequest(BaseModel):
    """Request for determining practice mode"""
    concept_proficiencies: List[ConceptProficiencyInput] = Field(
        ..., min_length=1, description="User's concept proficiencies"
    )
    target_concept_id: Optional[int] = Field(default=None, description="Target concept")


@router.post("/interleaved/practice-mode")
async def determine_practice_mode(request: PracticeModeRequest):
    """
    Determine optimal practice mode based on proficiencies

    Research basis: Hybrid scheduling
    - Blocked practice for concepts below 75% proficiency
    - Interleaved practice for concepts at/above 75%

    Returns recommended practice mode with rationale.
    """
    # Convert input to ConceptProficiency objects
    proficiencies = [
        ConceptProficiency(
            concept_id=p.concept_id,
            concept_name=p.concept_name,
            mastery_level=p.mastery_level,
            practice_count=p.practice_count,
            recent_accuracy=p.recent_accuracy,
            last_practiced=p.last_practiced,
            stability=p.stability,
        )
        for p in request.concept_proficiencies
    ]

    mode, rationale = interleaved_scheduler.determine_practice_mode(
        concept_proficiencies=proficiencies,
        target_concept_id=request.target_concept_id,
    )

    # Count concepts in each category
    ready_count = sum(1 for p in proficiencies if p.mastery_level >= 0.75)
    learning_count = sum(1 for p in proficiencies if p.mastery_level < 0.75)

    return {
        "mode": mode.value,
        "rationale": rationale,
        "statistics": {
            "total_concepts": len(proficiencies),
            "ready_for_interleaving": ready_count,
            "still_learning": learning_count,
            "interleave_threshold": 0.75,
        }
    }


@router.post("/interleaved/generate-sequence")
async def generate_practice_sequence(request: PracticeSequenceRequest):
    """
    Generate optimal practice sequence

    Research basis: Interleaved Practice (g=0.42 effect size)
    - Hybrid scheduling: blocked for acquisition, interleaved for retention
    - Optimal spacing between same-concept items
    - Contextual interference for discrimination learning
    """
    # Convert proficiencies
    proficiencies = [
        ConceptProficiency(
            concept_id=p.concept_id,
            concept_name=p.concept_name,
            mastery_level=p.mastery_level,
            practice_count=p.practice_count,
            recent_accuracy=p.recent_accuracy,
            last_practiced=p.last_practiced,
            stability=p.stability,
        )
        for p in request.concept_proficiencies
    ]

    # Convert available items (handle string keys from JSON)
    available_items: Dict[int, List[PracticeItem]] = {}
    for concept_id_str, items in request.available_items.items():
        concept_id = int(concept_id_str)
        available_items[concept_id] = [
            PracticeItem(
                item_id=item.item_id,
                concept_id=item.concept_id,
                concept_name=item.concept_name,
                difficulty=item.difficulty,
                item_type=item.item_type,
                content=item.content or {},
            )
            for item in items
        ]

    # Generate sequence
    sequence = interleaved_scheduler.generate_practice_sequence(
        concept_proficiencies=proficiencies,
        available_items=available_items,
        target_items=request.target_items,
        target_duration_minutes=request.target_duration_minutes,
        target_concept_id=request.target_concept_id,
    )

    return {
        "mode": sequence.mode.value,
        "items": [
            {
                "item_id": item.item_id,
                "concept_id": item.concept_id,
                "concept_name": item.concept_name,
                "difficulty": item.difficulty,
                "item_type": item.item_type,
            }
            for item in sequence.items
        ],
        "concepts_included": sequence.concepts_included,
        "estimated_duration_minutes": sequence.estimated_duration_minutes,
        "interleaving_ratio": round(sequence.interleaving_ratio, 3),
        "rationale": sequence.rationale,
        "total_items": len(sequence.items),
    }


@router.post("/interleaved/prioritize")
async def prioritize_concepts_for_practice(
    concept_proficiencies: List[ConceptProficiencyInput],
):
    """
    Prioritize concepts for interleaved practice

    Considers:
    - Spacing benefit (FSRS integration)
    - Proficiency level
    - Time since last practice
    """
    # Convert proficiencies
    proficiencies = [
        ConceptProficiency(
            concept_id=p.concept_id,
            concept_name=p.concept_name,
            mastery_level=p.mastery_level,
            practice_count=p.practice_count,
            recent_accuracy=p.recent_accuracy,
            last_practiced=p.last_practiced,
            stability=p.stability,
        )
        for p in concept_proficiencies
    ]

    # Get prioritized list
    prioritized = interleaved_scheduler.prioritize_for_interleaving(proficiencies)

    return {
        "prioritized_concepts": [
            {
                "concept_id": prof.concept_id,
                "concept_name": prof.concept_name,
                "mastery_level": round(prof.mastery_level, 3),
                "priority_score": round(score, 3),
                "last_practiced": prof.last_practiced.isoformat() if prof.last_practiced else None,
            }
            for prof, score in prioritized
        ],
        "total_concepts": len(prioritized),
    }


@router.get("/interleaved/spacing-benefit")
async def calculate_spacing_benefit(
    last_practiced: Optional[datetime] = None,
    stability: float = 1.0,
):
    """
    Calculate benefit of practicing now based on spacing

    Integrates with FSRS to determine optimal practice timing.

    Returns higher score when:
    - More time has passed since last practice
    - Retrievability is declining
    """
    benefit = interleaved_scheduler.calculate_spacing_benefit(
        last_practiced=last_practiced,
        stability=stability,
    )

    return {
        "spacing_benefit": round(benefit, 3),
        "last_practiced": last_practiced.isoformat() if last_practiced else None,
        "stability": stability,
        "recommendation": _get_spacing_recommendation(benefit),
    }


@router.post("/interleaved/statistics")
async def get_interleaving_statistics(
    session_history: List[Dict],
):
    """
    Get statistics about interleaving patterns from practice history

    Useful for analyzing effectiveness and tuning parameters.
    """
    stats = interleaved_scheduler.get_interleaving_statistics(session_history)

    return {
        "statistics": stats,
        "recommendations": _get_interleaving_recommendations(stats),
    }


def _get_spacing_recommendation(benefit: float) -> str:
    """Get recommendation based on spacing benefit"""
    if benefit > 0.8:
        return "High benefit - optimal time to practice this concept"
    elif benefit > 0.5:
        return "Moderate benefit - good time for practice"
    elif benefit > 0.2:
        return "Low benefit - concept still fresh, consider other priorities"
    else:
        return "Very low benefit - recently practiced, focus on other concepts"


def _get_interleaving_recommendations(stats: Dict) -> List[str]:
    """Get recommendations based on interleaving statistics"""
    recommendations = []

    avg_ratio = stats.get("avg_interleaving_ratio", 0)
    mode_dist = stats.get("mode_distribution", {})
    avg_concepts = stats.get("avg_concepts_per_session", 0)

    if avg_ratio < 0.3:
        recommendations.append(
            "Low interleaving ratio. Consider increasing variety across concepts."
        )
    elif avg_ratio > 0.9:
        recommendations.append(
            "Very high interleaving. Some blocked practice may help initial acquisition."
        )

    if avg_concepts < 2:
        recommendations.append(
            "Few concepts per session. Include more concepts when proficiency allows."
        )
    elif avg_concepts > 5:
        recommendations.append(
            "Many concepts per session. Consider focusing on fewer for deeper practice."
        )

    blocked_pct = mode_dist.get("blocked", 0)
    if blocked_pct > 0.7:
        recommendations.append(
            "Mostly blocked practice. Work on building proficiency to enable interleaving."
        )

    if not recommendations:
        recommendations.append("Good interleaving balance. Continue current approach.")

    return recommendations

# ============== ZPD and Scaffolding Endpoints ==============

class ZPDAnalysisRequest(BaseModel):
    user_id: str
    content_difficulty: float = Field(..., ge=0, le=1)
    user_mastery: float = Field(..., ge=0, le=1)

@router.post("/zpd/analyze")
async def analyze_zpd_fit(
    request: ZPDAnalysisRequest,
    service: ScaffoldingService = Depends(get_scaffolding_service)
):
    """
    Analyze if content fits within the user's Zone of Proximal Development
    """
    try:
        analysis = await service.analyze_zpd_fit(
            user_id=request.user_id,
            content_difficulty=request.content_difficulty,
            user_mastery=request.user_mastery
        )
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scaffolding/hint", response_model=HintResponse)
async def get_adaptive_hint(
    request: HintRequest,
    service: ScaffoldingService = Depends(get_scaffolding_service)
):
    """
    Get an adaptive hint based on context and history
    """
    try:
        return await service.get_adaptive_hint(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Curriculum Reinforcement Learning ==============

from app.adaptive.crl import CurriculumRLPolicy, PolicyConfig, PolicyType
from app.adaptive.td_bkt import TemporalDifferenceBKT, TDBKTConfig
from app.adaptive.hlr import HLRModel, RewardCalculator, HLRConfig
from app.adaptive.kg_mask import ActionMasker, ActionMaskerConfig

# Initialize CRL components (lazy loaded on first use)
_crl_policy: Optional[CurriculumRLPolicy] = None
_td_bkt: Optional[TemporalDifferenceBKT] = None
_hlr_model: Optional[HLRModel] = None
_action_masker: Optional[ActionMasker] = None


from app.core.config import settings

def get_crl_policy() -> CurriculumRLPolicy:
    """Get or create CRL policy instance"""
    global _crl_policy
    if _crl_policy is None:
        config = PolicyConfig(
            policy_type=PolicyType.DT_LITE,  # Use lightweight inference by default
            fallback_to_zpd=True,
            dt_weights_path=settings.CRL_DT_WEIGHTS_PATH,
            dt_config_path=settings.CRL_DT_CONFIG_PATH,
        )
        _crl_policy = CurriculumRLPolicy(config)
    return _crl_policy


def get_td_bkt() -> TemporalDifferenceBKT:
    """Get or create TD-BKT instance"""
    global _td_bkt
    if _td_bkt is None:
        config = TDBKTConfig()
        _td_bkt = TemporalDifferenceBKT(config)
    return _td_bkt


def get_hlr_model() -> HLRModel:
    """Get or create HLR model instance"""
    global _hlr_model
    if _hlr_model is None:
        config = HLRConfig()
        _hlr_model = HLRModel(config)
    return _hlr_model


def get_action_masker() -> ActionMasker:
    """Get or create action masker instance"""
    global _action_masker
    if _action_masker is None:
        config = ActionMaskerConfig()
        _action_masker = ActionMasker(config)
    return _action_masker


class CRLConceptSelectionRequest(BaseModel):
    """Request for CRL concept selection"""
    user_id: int = Field(..., description="User ID")
    course_id: int = Field(..., description="Course ID")
    session_history: Optional[List[Dict]] = Field(default=None, description="Current session history")
    use_ab_test: bool = Field(default=True, description="Use A/B testing if configured")


class CRLUpdateRequest(BaseModel):
    """Request for updating CRL state after interaction"""
    user_id: int = Field(..., description="User ID")
    concept_id: int = Field(..., description="Concept practiced")
    correct: bool = Field(..., description="Whether response was correct")
    response_time_ms: int = Field(..., description="Response time in milliseconds")
    timestamp: Optional[datetime] = Field(default=None, description="Interaction timestamp")


class CRLBeliefStateRequest(BaseModel):
    """Request for getting belief state"""
    user_id: int = Field(..., description="User ID")
    concept_ids: List[int] = Field(..., description="Concept IDs to include")


class CRLBenchmarkRequest(BaseModel):
    """Request for running CRL benchmark"""
    num_students: int = Field(default=100, ge=10, le=1000, description="Number of simulated students")
    num_sessions: int = Field(default=30, ge=5, le=100, description="Sessions per student")
    items_per_session: int = Field(default=10, ge=5, le=50, description="Items per session")


@router.post("/crl/select-concept")
async def crl_select_next_concept(
    request: CRLConceptSelectionRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Select next concept using Curriculum RL policy

    Uses offline RL (Decision Transformer or CQL) trained on
    historical student data to optimize for long-term retention.

    Research basis:
    - Curriculum Reinforcement Learning for ITS
    - Offline RL: Conservative Q-Learning, Decision Transformer
    - TD-BKT for state estimation with forgetting
    """
    try:
        # Get user's concept masteries for this course
        concepts_result = await db.execute(
            select(Concept).where(Concept.course_id == request.course_id)
        )
        concepts = concepts_result.scalars().all()
        concept_ids = [str(c.id) for c in concepts]

        if not concept_ids:
            raise HTTPException(status_code=404, detail="No concepts found for course")

        # Get user masteries
        mastery_result = await db.execute(
            select(UserConceptMastery).where(
                and_(
                    UserConceptMastery.user_id == request.user_id,
                    UserConceptMastery.concept_id.in_([c.id for c in concepts])
                )
            )
        )
        masteries = mastery_result.scalars().all()
        mastery_map = {str(m.concept_id): m.mastery_level for m in masteries}

        # Initialize TD-BKT for state estimation
        td_bkt = get_td_bkt()
        td_bkt.initialize(concept_ids)

        # Update TD-BKT with current masteries
        for cid, mastery in mastery_map.items():
            if cid in td_bkt.belief_state.concepts:
                td_bkt.belief_state.concepts[cid].mastery = mastery

        # Get belief state vector
        belief_state = td_bkt.get_belief_state()
        belief_vector = belief_state.to_vector()

        # Get action mask from knowledge graph (simplified - in production, query KG)
        action_masker = get_action_masker()
        action_masker.initialize(concept_ids)
        action_mask = action_masker.get_action_mask(belief_state.concepts)

        # Get CRL policy
        policy = get_crl_policy()

        # Select concept
        selected_concept, selection_info = policy.select_next_concept(
            belief_state=belief_vector,
            valid_actions=action_mask,
            concept_ids=concept_ids,
        )

        # Map back to integer ID
        selected_concept_id = int(selected_concept)

        # Get concept details
        concept_result = await db.execute(
            select(Concept).where(Concept.id == selected_concept_id)
        )
        concept = concept_result.scalar_one_or_none()

        return {
            "selected_concept_id": selected_concept_id,
            "concept_name": concept.name if concept else "Unknown",
            "policy_type": selection_info.get("policy_type", "unknown"),
            "confidence": selection_info.get("confidence", 0.0),
            "action_values": selection_info.get("action_values", {}),
            "alternative_concepts": selection_info.get("alternatives", [])[:3],
            "belief_state_summary": {
                "mean_mastery": float(belief_vector[:len(concept_ids)].mean()) if len(belief_vector) > 0 else 0,
                "concepts_mastered": sum(1 for m in mastery_map.values() if m >= 0.95),
                "concepts_total": len(concept_ids),
            }
        }

    except Exception as e:
        # Fallback to ZPD-based selection on error
        import logging
        logging.error(f"CRL selection error: {e}")
        return await get_content_recommendations(
            user_id=request.user_id,
            course_id=request.course_id,
            top_n=1,
            db=db
        )


@router.post("/crl/update")
async def crl_update_state(
    request: CRLUpdateRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Update CRL state after a learning interaction

    Updates:
    - TD-BKT belief state (with forgetting)
    - HLR half-life estimates
    - Records for offline RL training
    """
    try:
        td_bkt = get_td_bkt()
        hlr = get_hlr_model()

        concept_id = str(request.concept_id)
        timestamp = request.timestamp or datetime.now()

        # Update TD-BKT
        if concept_id in td_bkt.belief_state.concepts:
            td_bkt.update(
                concept_id=concept_id,
                correct=request.correct,
                timestamp=timestamp,
            )

        # Update HLR
        hlr.update(
            concept_id=concept_id,
            correct=request.correct,
            timestamp=timestamp,
        )

        # Get updated state
        belief_state = td_bkt.get_belief_state()
        concept_state = belief_state.concepts.get(concept_id)
        mastery = concept_state.mastery if concept_state else 0.0

        # Variable Reward Trigger (Research-backed VR schedule)
        # Higher mastery = lower extrinsic reward probability (Fading)
        reward_item = None
        if request.correct:
            reward_item = vr_engine.trigger_reward(mastery)

        # Calculate reward (for logging/analysis)
        reward_calc = RewardCalculator(hlr)
        reward = reward_calc.compute_reward(
            concept_id=concept_id,
            correct=request.correct,
            reward_type="delta_retention",
        )

        # Update database mastery record
        result = await db.execute(
            select(UserConceptMastery).where(
                and_(
                    UserConceptMastery.user_id == request.user_id,
                    UserConceptMastery.concept_id == request.concept_id
                )
            )
        )
        mastery_record = result.scalar_one_or_none()

        if mastery_record:
            mastery_record.mastery_level = mastery
            mastery_record.practice_count += 1
            mastery_record.last_practiced = timestamp
            await db.commit()

        return {
            "concept_id": request.concept_id,
            "updated_mastery": mastery,
            "half_life_hours": concept_state.half_life_hours if concept_state else 24.0,
            "reward": reward,
            "recall_probability": concept_state.compute_recall_probability() if concept_state else 0.0,
            "variable_reward": reward_item.dict() if reward_item else None,
        }

    except Exception as e:
        import logging
        logging.error(f"CRL update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/crl/belief-state/{user_id}")
async def get_crl_belief_state(
    user_id: int,
    course_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get current TD-BKT belief state for user

    Returns mastery, recency, and half-life estimates
    for all concepts in the course.
    """
    try:
        # Get concepts
        concepts_result = await db.execute(
            select(Concept).where(Concept.course_id == course_id)
        )
        concepts = concepts_result.scalars().all()
        concept_ids = [str(c.id) for c in concepts]

        # Get TD-BKT state
        td_bkt = get_td_bkt()
        if not td_bkt.belief_state.concepts:
            td_bkt.initialize(concept_ids)

        belief_state = td_bkt.get_belief_state()

        # Format response
        concept_states = []
        for concept in concepts:
            cid = str(concept.id)
            state = belief_state.concepts.get(cid)

            if state:
                concept_states.append({
                    "concept_id": concept.id,
                    "concept_name": concept.name,
                    "mastery": state.mastery,
                    "half_life_hours": state.half_life_hours,
                    "recall_probability": state.compute_recall_probability(),
                    "practice_count": state.practice_count,
                    "last_practiced": state.last_practiced.isoformat() if state.last_practiced else None,
                })
            else:
                concept_states.append({
                    "concept_id": concept.id,
                    "concept_name": concept.name,
                    "mastery": 0.0,
                    "half_life_hours": 24.0,
                    "recall_probability": 0.0,
                    "practice_count": 0,
                    "last_practiced": None,
                })

        # Compute summary statistics
        masteries = [s["mastery"] for s in concept_states]
        recalls = [s["recall_probability"] for s in concept_states]

        return {
            "user_id": user_id,
            "course_id": course_id,
            "concepts": concept_states,
            "summary": {
                "mean_mastery": sum(masteries) / len(masteries) if masteries else 0,
                "mean_recall_probability": sum(recalls) / len(recalls) if recalls else 0,
                "concepts_mastered": sum(1 for m in masteries if m >= 0.95),
                "total_concepts": len(concept_states),
            }
        }

    except Exception as e:
        import logging
        logging.error(f"CRL belief state error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/crl/benchmark")
async def run_crl_benchmark(request: CRLBenchmarkRequest):
    """
    Run CRL policy benchmark against baselines

    Evaluates:
    - Day-30 retention (primary metric)
    - Learning efficiency
    - Interleaving score

    Baselines:
    - Random selection
    - Round-robin
    - Mastery threshold (0.75)
    - Spaced-only
    """
    try:
        from app.adaptive.simulator import (
            Benchmark,
            BenchmarkConfig,
            StudentSimulatorConfig,
        )

        # Configure benchmark
        config = BenchmarkConfig(
            num_simulated_students=request.num_students,
            num_sessions_per_student=request.num_sessions,
            items_per_session=request.items_per_session,
            num_workers=4,
        )

        simulator_config = StudentSimulatorConfig(
            num_concepts=20,
            seed=42,
        )

        benchmark = Benchmark(config, simulator_config)

        # Get CRL policy function
        policy = get_crl_policy()

        def crl_policy_fn(simulator, belief_state=None):
            # Get belief vector from simulator
            mastery_vector = simulator.get_true_mastery_vector()

            # Create simple belief state
            import numpy as np
            belief = np.concatenate([
                mastery_vector,
                np.ones_like(mastery_vector),  # recency
                np.ones_like(mastery_vector) * 24,  # half-lives
            ])

            # Get action mask (all valid for simulation)
            action_mask = np.ones(len(simulator.concept_ids))

            # Select concept
            concept, _ = policy.select_next_concept(
                belief_state=belief,
                valid_actions=action_mask,
                concept_ids=simulator.concept_ids,
            )

            return concept

        # Run benchmark
        results = benchmark.compare_policies(
            policies=[("crl_policy", crl_policy_fn)],
            include_baselines=True,
        )

        return {
            "summary": results.summary(),
            "ranking_by_retention": results.ranking_by_retention,
            "ranking_by_efficiency": results.ranking_by_efficiency,
            "policy_results": {
                name: {
                    "day30_retention_mean": r.day30_retention_mean,
                    "day30_retention_std": r.day30_retention_std,
                    "learning_efficiency_mean": r.learning_efficiency_mean,
                    "interleaving_score_mean": r.interleaving_score_mean,
                }
                for name, r in results.policy_results.items()
            },
            "improvement_over_baseline": results.get_improvement_over_baseline(
                "crl_policy", "mastery_threshold"
            ) if "crl_policy" in results.policy_results else {},
        }

    except Exception as e:
        import logging
        logging.error(f"CRL benchmark error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/crl/policy-info")
async def get_crl_policy_info():
    """
    Get information about the current CRL policy configuration
    """
    policy = get_crl_policy()

    return {
        "policy_type": policy.config.policy_type.value,
        "num_concepts": policy.config.num_concepts,
        "state_dim": policy.config.state_dim,
        "context_length": policy.config.context_length,
        "fallback_to_zpd": policy.config.fallback_to_zpd,
        "model_loaded": policy._policy is not None,
        "description": "Curriculum RL policy using offline RL for optimal concept sequencing",
        "research_basis": [
            "Conservative Q-Learning (Kumar et al., 2020)",
            "Decision Transformer (Chen et al., 2021)",
            "TD-BKT for state estimation with forgetting",
            "Half-Life Regression for reward shaping",
        ],
    }


# ============== Causal Discovery Endpoints ==============
# Based on "Causal Discovery for Educational Graphs" PDF specification

from app.adaptive.causal_discovery.manager import causal_manager
from app.services.graph_service import AsyncGraphService

import time
import logging

logger = logging.getLogger(__name__)


class CausalDiscoveryRequest(BaseModel):
    """Request model for triggering causal discovery pipeline"""
    course_id: int = Field(..., description="Course ID to analyze")
    min_users: int = Field(default=50, ge=10, description="Minimum users required for reliable discovery")
    notears_lambda: float = Field(default=0.1, ge=0.01, le=1.0, description="NOTEARS L1 sparsity penalty")
    fci_alpha: float = Field(default=0.05, ge=0.01, le=0.1, description="FCI significance level")
    edge_threshold: float = Field(default=0.3, ge=0.1, le=0.9, description="Minimum edge weight threshold")


class CausalDiscoveryResponse(BaseModel):
    """Response model for causal discovery results"""
    status: str
    course_id: int
    edges_discovered: int
    edges_persisted: int
    edges_skipped: int
    communities_detected: int
    execution_time_seconds: float
    message: str


class CausalEdgeStatusResponse(BaseModel):
    """Response model for causal edge status"""
    course_id: Optional[int]
    total_edges: int
    by_status: Dict[str, int]
    by_method: Dict[str, int]


@router.post("/causal-discovery/run", response_model=CausalDiscoveryResponse)
async def run_causal_discovery(
    request: CausalDiscoveryRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Trigger the causal discovery pipeline for a course.

    Pipeline (per PDF Sections 3-7):
    1. Data preprocessing: Convert mastery logs to User x Concept matrix
    2. Global discovery: NOTEARS for DAG skeleton with L1 sparsity
    3. Community detection: Leiden algorithm for clustering
    4. Local refinement: FCI on dense subcommunities to detect confounders
    5. Persistence: MERGE edges to Apache AGE with confidence scoring

    Research basis:
    - NOTEARS: Continuous optimization for acyclic structure learning (Zheng et al., 2018)
    - FCI: Fast Causal Inference for latent confounders (Spirtes et al., 2000)
    - Leiden: Guaranteed well-connected communities (Traag et al., 2019)

    Bootstrap stability scoring (PDF Section 6):
    - >0.85 confidence = 'verified' status
    - 0.5-0.85 confidence = 'hypothetical' status
    - <0.5 confidence = discarded as noise
    """
    start_time = time.time()

    try:
        # 1. Fetch concepts for course
        concepts_result = await db.execute(
            select(Concept).where(Concept.course_id == request.course_id)
        )
        concepts = concepts_result.scalars().all()
        concept_ids = [c.id for c in concepts]

        if not concept_ids:
            raise HTTPException(
                status_code=404,
                detail=f"No concepts found for course {request.course_id}"
            )

        # 2. Fetch mastery data
        mastery_result = await db.execute(
            select(UserConceptMastery).where(
                UserConceptMastery.concept_id.in_(concept_ids)
            )
        )
        masteries = mastery_result.scalars().all()

        # Convert to format expected by manager
        mastery_data = [
            {
                "user_id": m.user_id,
                "concept_id": m.concept_id,
                "mastery": m.mastery_level
            }
            for m in masteries
        ]

        # Check minimum user threshold
        unique_users = len(set(m["user_id"] for m in mastery_data))
        if unique_users < request.min_users:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient users for reliable discovery: {unique_users} < {request.min_users} minimum"
            )

        # 3. Run discovery pipeline
        graph_service = AsyncGraphService(db)
        await causal_manager.run_discovery_pipeline(mastery_data, graph_service)

        # 4. Get results
        edges_discovered = len(causal_manager._last_edges) if hasattr(causal_manager, '_last_edges') else 0
        persist_result = causal_manager._last_persist_result if hasattr(causal_manager, '_last_persist_result') else {"persisted": 0, "skipped": 0}
        communities = len(causal_manager._last_communities) if hasattr(causal_manager, '_last_communities') else 0

        execution_time = time.time() - start_time

        return CausalDiscoveryResponse(
            status="completed",
            course_id=request.course_id,
            edges_discovered=edges_discovered,
            edges_persisted=persist_result.get("persisted", 0),
            edges_skipped=persist_result.get("skipped", 0),
            communities_detected=communities,
            execution_time_seconds=round(execution_time, 2),
            message=f"Discovered {edges_discovered} edges from {unique_users} users across {len(concept_ids)} concepts"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Causal discovery failed for course {request.course_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/causal-discovery/status/{course_id}", response_model=CausalEdgeStatusResponse)
async def get_causal_discovery_status(
    course_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get status of causally discovered edges for a course.

    Returns counts of verified vs hypothetical edges and breakdown by discovery method.
    """
    try:
        graph_service = AsyncGraphService(db)
        stats = await graph_service.get_causal_edge_statistics(course_id)

        return CausalEdgeStatusResponse(
            course_id=course_id,
            total_edges=stats.get("total", 0),
            by_status=stats.get("by_status", {}),
            by_method=stats.get("by_method", {})
        )

    except Exception as e:
        logger.error(f"Failed to get causal discovery status for course {course_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/causal-discovery/edges/{course_id}")
async def get_causal_edges(
    course_id: int,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieve causally discovered edges for a course.

    Args:
        course_id: Course ID to filter by
        status: Optional filter by status ('verified' or 'hypothetical')

    Returns:
        List of causal edges with source, target, weight, confidence, method, and status
    """
    if status and status not in ["verified", "hypothetical"]:
        raise HTTPException(
            status_code=400,
            detail="Status must be 'verified' or 'hypothetical'"
        )

    try:
        graph_service = AsyncGraphService(db)
        edges = await graph_service.get_causal_edges(course_id, status)

        return {
            "course_id": course_id,
            "filter_status": status,
            "count": len(edges),
            "edges": edges
        }

    except Exception as e:
        logger.error(f"Failed to get causal edges for course {course_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Neural ODE Scheduling ==============
# Hybrid FSRS/Neural ODE scheduling with uncertainty quantification

from app.models.neural_ode import ODECardState, MemoryStateTrajectory


class NeuralODEScheduleRequest(BaseModel):
    """Request for Neural ODE scheduling"""
    user_id: int = Field(..., description="User ID")
    concept_id: int = Field(..., description="Concept ID")
    current_fsrs_interval_days: Optional[float] = Field(default=None, description="Current FSRS interval")


class NeuralODEStateResponse(BaseModel):
    """Response model for Neural ODE state"""
    user_id: int
    concept_id: int
    control_mode: str
    ode_confidence: float
    review_count: int
    current_retrievability: Optional[float]
    uncertainty_epistemic: Optional[float]
    uncertainty_aleatoric: Optional[float]
    recommended_interval_days: float


class NeuralODEReviewRequest(BaseModel):
    """Request for updating Neural ODE state after review"""
    user_id: int = Field(..., description="User ID")
    concept_id: int = Field(..., description="Concept ID")
    grade: int = Field(..., ge=1, le=4, description="Grade 1-4 (Again, Hard, Good, Easy)")
    response_time_ms: int = Field(..., description="Response time in milliseconds")
    hesitation_count: Optional[int] = Field(default=0, description="Number of hesitation pauses")
    cursor_tortuosity: Optional[float] = Field(default=1.0, description="Cursor path tortuosity")


@router.post("/neural-ode/schedule")
async def schedule_neural_ode_review(
    request: NeuralODEScheduleRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Schedule next review using Hybrid FSRS/Neural ODE system.

    Control modes (from CT-MCN paper):
    - SHADOW: Neural ODE runs but FSRS decides (< 20 reviews or < 0.5 confidence)
    - HYBRID: Confidence-weighted blend (20-50 reviews, >= 0.5 confidence)
    - ACTIVE: Neural ODE fully controls (>= 50 reviews, >= 0.8 confidence)

    Returns recommended interval with uncertainty estimates.
    """
    try:
        from app.adaptive.neural_ode.hybrid_scheduler import (
            HybridFSRSScheduler,
            HybridConfig,
            ODEState,
            ControlMode,
        )

        # Get or create ODE state
        result = await db.execute(
            select(ODECardState).where(
                and_(
                    ODECardState.user_id == request.user_id,
                    ODECardState.concept_id == request.concept_id,
                )
            )
        )
        ode_card_state = result.scalar_one_or_none()

        if not ode_card_state:
            # Create new state with default values
            ode_card_state = ODECardState(
                user_id=request.user_id,
                concept_id=request.concept_id,
                current_latent_state=[0.0] * 32,
                last_state_time=datetime.utcnow(),
                control_mode="shadow",
                ode_confidence=0.0,
                review_count=0,
            )
            db.add(ode_card_state)
            await db.flush()

        # Convert to ODEState for scheduler
        import torch
        ode_state = ODEState(
            latent_state=torch.tensor(ode_card_state.current_latent_state or [0.0] * 32),
            last_update_time=ode_card_state.last_state_time or datetime.utcnow(),
            review_count=ode_card_state.review_count or 0,
            ode_confidence=ode_card_state.ode_confidence or 0.0,
        )

        # Initialize scheduler
        config = HybridConfig()
        scheduler = HybridFSRSScheduler(config=config)

        # Determine control mode
        control_mode = scheduler.determine_control_mode(ode_state)

        # Get FSRS interval
        fsrs_interval = request.current_fsrs_interval_days or 1.0

        # Get scheduling result
        card_features = torch.randn(64)  # Placeholder - in production, extract from concept
        scheduling_result = scheduler.schedule_review(
            ode_state=ode_state,
            fsrs_interval_days=fsrs_interval,
            card_features=card_features,
        )

        # Update database state
        ode_card_state.control_mode = control_mode.value
        await db.commit()

        return {
            "user_id": request.user_id,
            "concept_id": request.concept_id,
            "control_mode": scheduling_result.control_mode.value,
            "recommended_interval_days": scheduling_result.recommended_interval_days,
            "fsrs_interval_days": scheduling_result.fsrs_interval_days,
            "ode_interval_days": scheduling_result.ode_interval_days,
            "blend_weight": scheduling_result.blend_weight,
            "ode_confidence": scheduling_result.ode_confidence,
            "predicted_retention": scheduling_result.predicted_retention,
            "uncertainty_epistemic": scheduling_result.uncertainty_epistemic,
            "uncertainty_aleatoric": scheduling_result.uncertainty_aleatoric,
            "rationale": scheduling_result.rationale,
        }

    except ImportError as e:
        logger.warning(f"Neural ODE module not available: {e}")
        return {
            "user_id": request.user_id,
            "concept_id": request.concept_id,
            "control_mode": "shadow",
            "recommended_interval_days": request.current_fsrs_interval_days or 1.0,
            "fsrs_interval_days": request.current_fsrs_interval_days or 1.0,
            "ode_interval_days": None,
            "blend_weight": 0.0,
            "ode_confidence": 0.0,
            "predicted_retention": None,
            "uncertainty_epistemic": None,
            "uncertainty_aleatoric": None,
            "rationale": "Neural ODE not available, using FSRS only",
        }
    except Exception as e:
        logger.error(f"Neural ODE scheduling error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/neural-ode/state/{user_id}/{concept_id}")
async def get_neural_ode_state(
    user_id: int,
    concept_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get current Neural ODE state for a user-concept pair.

    Returns control mode, confidence, uncertainty estimates,
    and current retrievability prediction.
    """
    try:
        # Get ODE state
        result = await db.execute(
            select(ODECardState).where(
                and_(
                    ODECardState.user_id == user_id,
                    ODECardState.concept_id == concept_id,
                )
            )
        )
        ode_card_state = result.scalar_one_or_none()

        if not ode_card_state:
            return {
                "user_id": user_id,
                "concept_id": concept_id,
                "control_mode": "shadow",
                "ode_confidence": 0.0,
                "review_count": 0,
                "current_retrievability": None,
                "uncertainty_epistemic": None,
                "uncertainty_aleatoric": None,
                "recommended_interval_days": 1.0,
                "message": "No ODE state exists yet for this user-concept pair",
            }

        # Get latest trajectory point for uncertainty
        trajectory_result = await db.execute(
            select(MemoryStateTrajectory)
            .where(
                and_(
                    MemoryStateTrajectory.user_id == user_id,
                    MemoryStateTrajectory.concept_id == concept_id,
                )
            )
            .order_by(MemoryStateTrajectory.timestamp.desc())
            .limit(1)
        )
        latest_trajectory = trajectory_result.scalar_one_or_none()

        return {
            "user_id": user_id,
            "concept_id": concept_id,
            "control_mode": ode_card_state.control_mode or "shadow",
            "ode_confidence": ode_card_state.ode_confidence or 0.0,
            "review_count": ode_card_state.review_count or 0,
            "current_retrievability": latest_trajectory.predicted_retrievability if latest_trajectory else None,
            "uncertainty_epistemic": latest_trajectory.uncertainty_epistemic if latest_trajectory else None,
            "uncertainty_aleatoric": latest_trajectory.uncertainty_aleatoric if latest_trajectory else None,
            "last_state_time": ode_card_state.last_state_time.isoformat() if ode_card_state.last_state_time else None,
            "model_version": ode_card_state.model_version or 0,
        }

    except Exception as e:
        logger.error(f"Error getting Neural ODE state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/neural-ode/review")
async def process_neural_ode_review(
    request: NeuralODEReviewRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Process a review outcome and update Neural ODE state.

    Updates:
    - Latent state via jump network
    - ODE confidence based on prediction accuracy
    - Review count and control mode

    Returns updated state and next review recommendation.
    """
    try:
        from app.adaptive.neural_ode.hybrid_scheduler import (
            HybridFSRSScheduler,
            HybridConfig,
            ODEState,
        )

        # Get ODE state
        result = await db.execute(
            select(ODECardState).where(
                and_(
                    ODECardState.user_id == request.user_id,
                    ODECardState.concept_id == request.concept_id,
                )
            )
        )
        ode_card_state = result.scalar_one_or_none()

        if not ode_card_state:
            ode_card_state = ODECardState(
                user_id=request.user_id,
                concept_id=request.concept_id,
                current_latent_state=[0.0] * 32,
                last_state_time=datetime.utcnow(),
                control_mode="shadow",
                ode_confidence=0.0,
                review_count=0,
            )
            db.add(ode_card_state)

        # Convert to ODEState
        import torch
        ode_state = ODEState(
            latent_state=torch.tensor(ode_card_state.current_latent_state or [0.0] * 32),
            last_update_time=ode_card_state.last_state_time or datetime.utcnow(),
            review_count=ode_card_state.review_count or 0,
            ode_confidence=ode_card_state.ode_confidence or 0.0,
        )

        # Initialize scheduler
        config = HybridConfig()
        scheduler = HybridFSRSScheduler(config=config)

        # Create telemetry tensor
        telemetry = torch.tensor([
            min(request.response_time_ms / 30000.0, 1.0),  # Normalized RT
            min((request.hesitation_count or 0) / 5.0, 1.0),  # Normalized hesitation
            request.cursor_tortuosity or 1.0,  # Tortuosity
            1.0 if request.grade >= 3 else 0.5,  # Fluency proxy
        ])

        # Process review
        updated_state = scheduler.process_review_outcome(
            ode_state=ode_state,
            grade=request.grade,
            telemetry=telemetry,
        )

        # Update database
        ode_card_state.current_latent_state = updated_state.latent_state.tolist()
        ode_card_state.last_state_time = updated_state.last_update_time
        ode_card_state.review_count = updated_state.review_count
        ode_card_state.ode_confidence = updated_state.ode_confidence
        ode_card_state.control_mode = scheduler.determine_control_mode(updated_state).value

        await db.commit()
        await db.refresh(ode_card_state)

        return {
            "user_id": request.user_id,
            "concept_id": request.concept_id,
            "updated_review_count": ode_card_state.review_count,
            "updated_confidence": ode_card_state.ode_confidence,
            "control_mode": ode_card_state.control_mode,
            "message": f"Processed grade {request.grade} review",
        }

    except ImportError as e:
        logger.warning(f"Neural ODE module not available: {e}")
        # Update review count anyway
        result = await db.execute(
            select(ODECardState).where(
                and_(
                    ODECardState.user_id == request.user_id,
                    ODECardState.concept_id == request.concept_id,
                )
            )
        )
        ode_card_state = result.scalar_one_or_none()
        if ode_card_state:
            ode_card_state.review_count = (ode_card_state.review_count or 0) + 1
            await db.commit()

        return {
            "user_id": request.user_id,
            "concept_id": request.concept_id,
            "updated_review_count": ode_card_state.review_count if ode_card_state else 1,
            "updated_confidence": 0.0,
            "control_mode": "shadow",
            "message": "Neural ODE not available, review count updated only",
        }
    except Exception as e:
        logger.error(f"Neural ODE review processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== JCI (Joint Causal Inference) Endpoints ==============
# Combine observational + experimental evidence for causal graph updates

from app.models.jci import ExperimentEdge, ExperimentEdgeStatus, EdgeValidationQueue


class JCIExperimentLinkRequest(BaseModel):
    """Request to link an experiment to a causal edge"""
    experiment_id: str = Field(..., description="A/B experiment ID")
    source_concept: str = Field(..., description="Source concept name")
    target_concept: str = Field(..., description="Target concept name")
    prior_confidence: float = Field(..., ge=0, le=1, description="Prior confidence from observational data")
    course_id: Optional[int] = Field(default=None, description="Course ID")
    treatment_description: Optional[str] = Field(default=None, description="Description of the intervention")
    control_description: Optional[str] = Field(default=None, description="Description of control condition")


class JCIExperimentCompleteRequest(BaseModel):
    """Request to process completed experiment"""
    experiment_id: str = Field(..., description="Completed A/B experiment ID")


class JCIRecommendExperimentsRequest(BaseModel):
    """Request for experiment recommendations"""
    course_id: int = Field(..., description="Course ID")
    max_recommendations: int = Field(default=5, ge=1, le=20, description="Maximum recommendations")


@router.post("/causal/link-experiment")
async def link_experiment_to_edge(
    request: JCIExperimentLinkRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Link an A/B experiment to a causal edge for validation.

    When you want to validate a hypothesized causal relationship A→B
    through experimentation, create this link before running the experiment.

    The JCI framework will then:
    1. Track the experiment results
    2. Perform Bayesian update on edge confidence
    3. Sync updated confidence to the knowledge graph
    """
    try:
        from app.adaptive.causal_discovery.jci_updater import JCIConfidenceUpdater
        from app.services.graph_service import AsyncGraphService

        graph_service = AsyncGraphService(db)
        updater = JCIConfidenceUpdater(db, graph_service)

        experiment_edge = await updater.create_experiment_edge_link(
            experiment_id=request.experiment_id,
            source_concept=request.source_concept,
            target_concept=request.target_concept,
            prior_confidence=request.prior_confidence,
            course_id=request.course_id,
            treatment_description=request.treatment_description,
            control_description=request.control_description,
        )

        return {
            "experiment_edge_id": experiment_edge.id,
            "experiment_id": experiment_edge.experiment_id,
            "source_concept": experiment_edge.source_concept,
            "target_concept": experiment_edge.target_concept,
            "prior_confidence": experiment_edge.prior_confidence,
            "status": experiment_edge.status.value,
            "message": "Experiment-edge link created successfully",
        }

    except Exception as e:
        logger.error(f"Failed to link experiment to edge: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/causal/experiment-completed")
async def process_experiment_completion(
    request: JCIExperimentCompleteRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Process a completed A/B experiment and update causal edge confidence.

    This endpoint:
    1. Fetches experiment results from A/B testing framework
    2. Performs Bayesian update on all linked edges
    3. Syncs updated confidence to the knowledge graph

    Call this when an experiment reaches statistical significance
    or is terminated.
    """
    try:
        from app.adaptive.causal_discovery.jci_updater import JCIConfidenceUpdater
        from app.services.graph_service import AsyncGraphService

        graph_service = AsyncGraphService(db)
        updater = JCIConfidenceUpdater(db, graph_service)

        updates = await updater.process_experiment_completion(request.experiment_id)

        return {
            "experiment_id": request.experiment_id,
            "edges_updated": len(updates),
            "updates": [
                {
                    "prior": u.prior,
                    "posterior": u.posterior,
                    "likelihood_ratio": u.likelihood_ratio,
                    "evidence_strength": u.evidence_strength,
                    "update_reason": u.update_reason,
                }
                for u in updates
            ],
            "message": f"Processed {len(updates)} edge updates from experiment",
        }

    except Exception as e:
        logger.error(f"Failed to process experiment completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/causal/recommend-experiments")
async def recommend_causal_experiments(
    request: JCIRecommendExperimentsRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Recommend experiments to validate uncertain causal edges.

    Uses active learning to identify edges with:
    - High uncertainty (confidence near 0.5)
    - High information gain potential
    - Feasible sample size requirements

    These are edges where an A/B test would most reduce
    our uncertainty about the causal relationship.
    """
    try:
        from app.adaptive.causal_discovery.jci_updater import JCIConfidenceUpdater
        from app.services.graph_service import AsyncGraphService

        graph_service = AsyncGraphService(db)
        updater = JCIConfidenceUpdater(db, graph_service)

        recommendations = await updater.recommend_experiments(
            course_id=request.course_id,
            max_recommendations=request.max_recommendations,
        )

        return {
            "course_id": request.course_id,
            "recommendation_count": len(recommendations),
            "recommendations": [
                {
                    "source_concept": r["source"],
                    "target_concept": r["target"],
                    "current_confidence": r["current_confidence"],
                    "entropy": r["entropy"],
                    "priority_score": r["priority"],
                    "rationale": f"High uncertainty (conf={r['current_confidence']:.2f}), information gain={r['entropy']:.3f}",
                }
                for r in recommendations
            ],
        }

    except Exception as e:
        logger.error(f"Failed to recommend experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/causal/experiment-edges/{course_id}")
async def get_experiment_edges(
    course_id: int,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Get experiment-edge links for a course.

    Returns all A/B experiments linked to causal edges,
    with their status and confidence updates.
    """
    try:
        query = select(ExperimentEdge).where(ExperimentEdge.course_id == course_id)

        if status:
            try:
                status_enum = ExperimentEdgeStatus(status)
                query = query.where(ExperimentEdge.status == status_enum)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status: {status}. Valid values: pending, running, completed, validated, invalidated"
                )

        result = await db.execute(query)
        edges = result.scalars().all()

        return {
            "course_id": course_id,
            "filter_status": status,
            "count": len(edges),
            "experiment_edges": [
                {
                    "id": e.id,
                    "experiment_id": e.experiment_id,
                    "source_concept": e.source_concept,
                    "target_concept": e.target_concept,
                    "prior_confidence": e.prior_confidence,
                    "posterior_confidence": e.posterior_confidence,
                    "experiment_effect_size": e.experiment_effect_size,
                    "experiment_p_value": e.experiment_p_value,
                    "status": e.status.value,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                    "experiment_completed_at": e.experiment_completed_at.isoformat() if e.experiment_completed_at else None,
                }
                for e in edges
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get experiment edges: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/causal/validation-queue/{course_id}")
async def get_validation_queue(
    course_id: int,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """
    Get the edge validation queue for a course.

    Returns edges prioritized for experimental validation,
    sorted by priority score (entropy * feasibility).
    """
    try:
        result = await db.execute(
            select(EdgeValidationQueue)
            .where(
                and_(
                    EdgeValidationQueue.course_id == course_id,
                    EdgeValidationQueue.status == "pending",
                )
            )
            .order_by(EdgeValidationQueue.priority_score.desc())
            .limit(limit)
        )
        queue_items = result.scalars().all()

        return {
            "course_id": course_id,
            "queue_length": len(queue_items),
            "items": [
                {
                    "id": item.id,
                    "source_concept": item.source_concept,
                    "target_concept": item.target_concept,
                    "priority_score": item.priority_score,
                    "information_gain": item.information_gain,
                    "uncertainty": item.uncertainty,
                    "current_confidence": item.current_confidence,
                    "estimated_sample_size": item.estimated_sample_size,
                    "feasibility_score": item.feasibility_score,
                    "queued_at": item.queued_at.isoformat() if item.queued_at else None,
                }
                for item in queue_items
            ],
        }

    except Exception as e:
        logger.error(f"Failed to get validation queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))
