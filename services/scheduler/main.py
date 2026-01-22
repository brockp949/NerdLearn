"""
FSRS Scheduler Microservice
FastAPI service for spaced repetition scheduling
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime
from enum import Enum
import redis.asyncio as redis
import json

from scheduler import FSRSScheduler, ReviewCard, ReviewRating, FSRSParameters


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class RatingEnum(str, Enum):
    AGAIN = "again"
    HARD = "hard"
    GOOD = "good"
    EASY = "easy"


class ReviewRequest(BaseModel):
    card_id: str = Field(..., description="Unique identifier for the learning item")
    rating: RatingEnum = Field(..., description="User's performance rating")
    review_time: Optional[datetime] = None
    learner_id: str = Field(..., description="Learner's unique ID")


class CardStateResponse(BaseModel):
    card_id: str
    stability: float
    difficulty: float
    scheduled_days: int
    due_date: datetime
    state: str
    review_count: int


class IntervalsPreviewResponse(BaseModel):
    card_id: str
    intervals: Dict[str, int]


class SchedulerConfig(BaseModel):
    request_retention: float = Field(0.9, ge=0.7, le=0.98)
    maximum_interval: int = Field(365, ge=1, le=3650)
    minimum_interval: int = Field(1, ge=1, le=1440)


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="NerdLearn FSRS Scheduler",
    description="Microservice for optimal spaced repetition scheduling",
    version="0.1.0"
)

# Global scheduler instance
scheduler = FSRSScheduler()

# Redis connection for card state persistence
redis_client: Optional[redis.Redis] = None


# ============================================================================
# DEPENDENCIES
# ============================================================================

async def get_redis() -> redis.Redis:
    """Get Redis connection"""
    global redis_client
    if redis_client is None:
        redis_client = await redis.from_url(
            "redis://localhost:6379",
            encoding="utf-8",
            decode_responses=True
        )
    return redis_client


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def rating_enum_to_review_rating(rating: RatingEnum) -> ReviewRating:
    """Convert API enum to scheduler enum"""
    mapping = {
        RatingEnum.AGAIN: ReviewRating.AGAIN,
        RatingEnum.HARD: ReviewRating.HARD,
        RatingEnum.GOOD: ReviewRating.GOOD,
        RatingEnum.EASY: ReviewRating.EASY,
    }
    return mapping[rating]


async def load_card_state(
    learner_id: str,
    card_id: str,
    redis_conn: redis.Redis
) -> ReviewCard:
    """Load card state from Redis or create new card"""
    key = f"card:{learner_id}:{card_id}"
    data = await redis_conn.get(key)

    if data:
        state = json.loads(data)
        return ReviewCard(
            id=card_id,
            stability=state.get('stability', 2.5),
            difficulty=state.get('difficulty', 5.0),
            elapsed_days=state.get('elapsed_days', 0.0),
            scheduled_days=state.get('scheduled_days', 0.0),
            review_count=state.get('review_count', 0),
            last_review=datetime.fromisoformat(state['last_review']) if state.get('last_review') else None,
            due_date=datetime.fromisoformat(state['due_date']) if state.get('due_date') else None,
            state=state.get('state', 'new'),
        )
    else:
        # New card
        return ReviewCard(id=card_id)


async def save_card_state(
    learner_id: str,
    card: ReviewCard,
    redis_conn: redis.Redis
):
    """Persist card state to Redis"""
    key = f"card:{learner_id}:{card.id}"
    state = scheduler.export_card_state(card)
    await redis_conn.set(key, json.dumps(state))
    # Set expiry for 1 year (cards not reviewed in a year are considered abandoned)
    await redis_conn.expire(key, 31536000)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "NerdLearn FSRS Scheduler",
        "status": "operational",
        "version": "0.1.0"
    }


@app.post("/review", response_model=CardStateResponse)
async def process_review(
    request: ReviewRequest,
    redis_conn: redis.Redis = Depends(get_redis)
):
    """
    Process a review and return updated schedule

    This is the main endpoint for the FSRS algorithm. It:
    1. Loads current card state
    2. Applies FSRS scheduling algorithm
    3. Returns new schedule and persists state
    """
    try:
        # Load current card state
        card = await load_card_state(request.learner_id, request.card_id, redis_conn)

        # Convert rating
        rating = rating_enum_to_review_rating(request.rating)

        # Schedule the card
        review_time = request.review_time or datetime.now()
        updated_card = scheduler.schedule_card(card, rating, review_time)

        # Save state
        await save_card_state(request.learner_id, updated_card, redis_conn)

        # Return response
        return CardStateResponse(
            card_id=updated_card.id,
            stability=updated_card.stability,
            difficulty=updated_card.difficulty,
            scheduled_days=updated_card.scheduled_days,
            due_date=updated_card.due_date,
            state=updated_card.state,
            review_count=updated_card.review_count,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scheduling error: {str(e)}")


@app.get("/preview/{learner_id}/{card_id}", response_model=IntervalsPreviewResponse)
async def preview_intervals(
    learner_id: str,
    card_id: str,
    redis_conn: redis.Redis = Depends(get_redis)
):
    """
    Preview intervals for all possible ratings without updating state

    Useful for showing learners how their rating will affect next review
    """
    try:
        # Load current card state
        card = await load_card_state(learner_id, card_id, redis_conn)

        # Get interval previews
        intervals = scheduler.get_optimal_intervals(card)

        # Convert to string keys for JSON
        interval_dict = {
            rating.name.lower(): days
            for rating, days in intervals.items()
        }

        return IntervalsPreviewResponse(
            card_id=card_id,
            intervals=interval_dict
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview error: {str(e)}")


@app.get("/due/{learner_id}")
async def get_due_cards(
    learner_id: str,
    limit: int = 20,
    redis_conn: redis.Redis = Depends(get_redis)
):
    """
    Get cards due for review for a specific learner

    Returns up to 'limit' cards sorted by priority
    """
    try:
        # Scan for learner's cards
        pattern = f"card:{learner_id}:*"
        due_cards = []
        now = datetime.now()

        async for key in redis_conn.scan_iter(match=pattern):
            data = await redis_conn.get(key)
            if data:
                state = json.loads(data)
                if state.get('due_date'):
                    due_date = datetime.fromisoformat(state['due_date'])
                    if due_date <= now:
                        card_id = key.split(':')[-1]
                        due_cards.append({
                            'card_id': card_id,
                            'due_date': state['due_date'],
                            'difficulty': state.get('difficulty', 5.0),
                            'state': state.get('state', 'new')
                        })

        # Sort by due date (oldest first) and limit
        due_cards.sort(key=lambda x: x['due_date'])
        return due_cards[:limit]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@app.post("/config")
async def update_scheduler_config(config: SchedulerConfig):
    """
    Update global scheduler configuration

    Allows adjusting retention targets and intervals
    """
    try:
        scheduler.adjust_retention_target(config.request_retention)
        scheduler.params.maximum_interval = config.maximum_interval
        scheduler.params.minimum_interval = config.minimum_interval

        return {
            "status": "updated",
            "config": {
                "request_retention": scheduler.params.request_retention,
                "maximum_interval": scheduler.params.maximum_interval,
                "minimum_interval": scheduler.params.minimum_interval,
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Config error: {str(e)}")


@app.delete("/card/{learner_id}/{card_id}")
async def reset_card(
    learner_id: str,
    card_id: str,
    redis_conn: redis.Redis = Depends(get_redis)
):
    """Reset a card to initial state (delete from Redis)"""
    try:
        key = f"card:{learner_id}:{card_id}"
        deleted = await redis_conn.delete(key)

        if deleted:
            return {"status": "reset", "card_id": card_id}
        else:
            raise HTTPException(status_code=404, detail="Card not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset error: {str(e)}")


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    global redis_client
    redis_client = await redis.from_url(
        "redis://localhost:6379",
        encoding="utf-8",
        decode_responses=True
    )
    print("✅ FSRS Scheduler service started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if redis_client:
        await redis_client.close()
    print("🛑 FSRS Scheduler service stopped")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
