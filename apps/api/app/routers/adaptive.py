"""
Adaptive Learning API Endpoints
Spaced repetition, content recommendations, and mastery tracking
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from datetime import datetime, timedelta
from typing import List, Dict, Any
from app.core.database import get_db
from app.models.spaced_repetition import SpacedRepetitionCard, ReviewLog, Concept
from app.models.assessment import UserConceptMastery
from app.models.course import Course, Module
from app.adaptive.fsrs import FSRSAlgorithm, FSRSCard, Rating
from app.adaptive.bkt import BayesianKnowledgeTracer
from app.adaptive.zpd import ZPDRegulator
from pydantic import BaseModel
from enum import Enum

router = APIRouter()

# Initialize adaptive algorithms
fsrs = FSRSAlgorithm()
bkt = BayesianKnowledgeTracer()
zpd = ZPDRegulator()


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
