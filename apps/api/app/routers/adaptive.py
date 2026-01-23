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
