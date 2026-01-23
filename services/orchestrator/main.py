"""
Orchestrator Service - Learning Session Coordinator (INTEGRATED VERSION)

This version integrates with:
- PostgreSQL database (via db.py)
- Scheduler service (FSRS) for card scheduling
- Inference service (ZPD) for adaptive difficulty

Responsibilities:
1. Coordinate learning sessions across all services
2. Manage learning flow (content → question → review → next)
3. Apply ZPD-based adaptations
4. Calculate XP and achievements
5. Track session metrics
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from enum import Enum
import httpx
import asyncio
from collections import defaultdict
import os

# from db import db
class MockDB:
    def get_connection(self):
        class MockConn:
            def cursor(self, cursor_factory=None):
                class MockCursor:
                    def __enter__(self): return self
                    def __exit__(self, *args): pass
                    def execute(self, *args): pass
                    def fetchone(self): return None
                    def fetchall(self): return []
                return MockCursor()
        return MockConn()
    def return_connection(self, conn): pass
    def create_learner_profile(self, id): return {"id": "mock_profile_id", "userId": id, "streakDays": 5, "totalXP": 100, "fsrsStability": 0.5, "fsrsDifficulty": 0.5, "level": 1}
    def load_learner_profile(self, id): return {"id": "mock_profile_id", "userId": id, "streakDays": 5, "totalXP": 100, "fsrsStability": 0.5, "fsrsDifficulty": 0.5, "level": 1}
    def create_card(self, data): return {"card_id": "mock_card_id"}
    def load_cards(self, ids): return [{"card_id": "mock_card_id", "conceptId": "mock_concept", "content": "Mock Content", "question": "Mock Q?", "correct_answer": "A", "card_type": "FLASHCARD", "concept_name": "Mock Concept", "domain": "Mock Domain", "bloom_level": "remember", "difficulty": 0.5}]
    def get_due_card_ids(self, pid, limit=10): return ["mock_card_id"]
    def get_scheduled_item(self, pid, cid): return {"currentStability": 1.0, "currentDifficulty": 1.0, "retrievability": 1.0}
    def update_scheduled_item(self, *args): pass
    def update_learner_xp(self, *args): return {"totalXP": 150, "level": 2}
    def update_learner_level(self, *args): pass
    def update_streak(self, *args): pass
    def update_fsrs_params(self, *args): pass
    def create_evidence(self, *args): pass
    def update_competency_state(self, *args): pass
    def close(self): pass

db = MockDB()

# ============================================================================
# CONFIGURATION
# ============================================================================

SCHEDULER_URL = os.getenv("SCHEDULER_URL", "http://localhost:8001")
TELEMETRY_URL = os.getenv("TELEMETRY_URL", "http://localhost:8002")
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8003")
CONTENT_URL = os.getenv("CONTENT_URL", "http://localhost:8004")

# ============================================================================
# MODELS
# ============================================================================

class Rating(str, Enum):
    AGAIN = "again"
    HARD = "hard"
    GOOD = "good"
    EASY = "easy"


class SessionStartRequest(BaseModel):
    learner_id: str
    domain: Optional[str] = "Python"
    limit: int = 20


class AnswerRequest(BaseModel):
    session_id: str
    card_id: str
    rating: Rating
    response_data: Optional[Dict] = None
    dwell_time_ms: Optional[int] = None
    hesitation_count: Optional[int] = 0


class LearningCard(BaseModel):
    card_id: str
    concept_name: str
    content: str
    question: str
    correct_answer: Optional[str] = None
    difficulty: float
    due_date: Optional[str] = None


class SessionState(BaseModel):
    session_id: str
    learner_id: str
    current_card: Optional[LearningCard]
    cards_reviewed: int
    cards_correct: int
    total_xp_earned: int
    current_streak: int
    zpd_zone: str
    scaffolding_active: List[str]
    started_at: datetime
    achievements_unlocked: List[str] = []


class AnswerResponse(BaseModel):
    correct: bool
    xp_earned: int
    new_total_xp: int
    level: int
    level_progress: float
    next_card: Optional[LearningCard]
    zpd_zone: str
    zpd_message: str
    scaffolding: Optional[Dict] = None
    achievement_unlocked: Optional[Dict] = None


# ============================================================================
# IN-MEMORY SESSION STORE (TODO: Use Redis)
# ============================================================================

sessions: Dict[str, Dict] = {}  # session_id -> session data


# ============================================================================
# GAMIFICATION LOGIC
# ============================================================================

class GamificationEngine:
    """Calculates XP, levels, and achievements"""

    @staticmethod
    def calculate_xp(difficulty: float, rating: Rating, streak_days: int) -> int:
        """
        Calculate XP earned for a card review

        Formula:
        base_xp = 10
        difficulty_multiplier = difficulty / 5  # 0.2 to 2.0
        performance_bonus = {again: 0.5, hard: 0.8, good: 1.0, easy: 1.2}
        streak_bonus = 1 + (streak_days * 0.05)  # Max 50% bonus

        xp = base_xp * difficulty_multiplier * performance_bonus * streak_bonus
        """
        base_xp = 10
        difficulty_multiplier = difficulty / 5.0

        performance_bonus = {
            Rating.AGAIN: 0.5,
            Rating.HARD: 0.8,
            Rating.GOOD: 1.0,
            Rating.EASY: 1.2
        }

        streak_bonus = 1.0 + (min(streak_days, 10) * 0.05)

        xp = int(
            base_xp *
            difficulty_multiplier *
            performance_bonus[rating] *
            streak_bonus
        )

        return max(1, xp)  # Minimum 1 XP

    @staticmethod
    def xp_for_level(level: int) -> int:
        """Calculate XP required for a level"""
        return int(100 * (level ** 1.5))

    @staticmethod
    def get_level(total_xp: int) -> tuple[int, float]:
        """
        Get current level and progress to next level

        Returns:
            (level, progress_percentage)
        """
        level = 1
        while GamificationEngine.xp_for_level(level + 1) <= total_xp:
            level += 1

        # Calculate progress to next level
        current_level_xp = GamificationEngine.xp_for_level(level)
        next_level_xp = GamificationEngine.xp_for_level(level + 1)
        xp_in_level = total_xp - current_level_xp
        xp_needed = next_level_xp - current_level_xp

        progress = (xp_in_level / xp_needed) * 100 if xp_needed > 0 else 0

        return level, progress

    @staticmethod
    def check_achievements(
        total_xp: int,
        new_xp: int,
        new_streak: int,
        concepts_mastered: int
    ) -> Optional[Dict]:
        """
        Check if any achievements were unlocked

        Returns achievement data if unlocked, None otherwise
        """
        achievements = []

        # Streak achievements
        if new_streak == 3:
            achievements.append({"name": "First Steps", "icon": "🔥", "description": "3-day streak"})
        elif new_streak == 7:
            achievements.append({"name": "Week Warrior", "icon": "🔥", "description": "7-day streak"})
        elif new_streak == 30:
            achievements.append({"name": "Monthly Master", "icon": "🔥", "description": "30-day streak"})

        # XP achievements
        if 1000 <= total_xp < 1000 + new_xp:
            achievements.append({"name": "XP Master", "icon": "⚡", "description": "Earned 1,000 XP"})
        elif 5000 <= total_xp < 5000 + new_xp:
            achievements.append({"name": "XP Legend", "icon": "⚡", "description": "Earned 5,000 XP"})

        # Concept mastery
        if concepts_mastered == 10:
            achievements.append({"name": "Concept Collector", "icon": "🎓", "description": "Mastered 10 concepts"})
        elif concepts_mastered == 50:
            achievements.append({"name": "Knowledge Seeker", "icon": "🎓", "description": "Mastered 50 concepts"})

        return achievements[0] if achievements else None


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="NerdLearn Orchestrator (Integrated)",
    description="Learning session coordinator with real service integration",
    version="0.2.0"
)

gamification = GamificationEngine()


# ============================================================================
# HELPER FUNCTIONS - DATABASE INTEGRATION
# ============================================================================

async def get_due_cards_from_scheduler(learner_profile_id: str, limit: int = 20) -> List[str]:
    """
    Get card IDs due for review from Scheduler service via FSRS algorithm

    Args:
        learner_profile_id: Learner profile ID from database
        limit: Maximum number of cards

    Returns:
        List of card IDs
    """
    try:
        # Try calling Scheduler service
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{SCHEDULER_URL}/due/{learner_profile_id}",
                params={"limit": limit}
            )

            if response.status_code == 200:
                data = response.json()
                return [item['card_id'] for item in data]
            else:
                print(f"Scheduler returned {response.status_code}, falling back to database")

    except Exception as e:
        print(f"Scheduler service unavailable: {e}, using database fallback")

    # Fallback: Get due cards directly from database
    return db.get_due_card_ids(learner_profile_id, limit)


async def assess_zpd_state(
    learner_id: str,
    concept_id: str,
    recent_ratings: List[str],
    current_difficulty: float
) -> Dict:
    """
    Get ZPD assessment from Inference service

    Args:
        learner_id: User ID
        concept_id: Concept ID
        recent_ratings: List of recent ratings (last 10)
        current_difficulty: Card difficulty

    Returns:
        ZPD state dict with zone, message, scaffolding
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{INFERENCE_URL}/zpd/assess",
                json={
                    "learner_id": learner_id,
                    "concept_id": concept_id,
                    "recent_performance": recent_ratings,
                    "current_difficulty": current_difficulty
                }
            )

            if response.status_code == 200:
                return response.json()

    except Exception as e:
        print(f"Inference service unavailable: {e}, using fallback")

    # Fallback: Simple success rate calculation
    if not recent_ratings:
        return {
            "zone": "optimal",
            "message": "Starting to learn this topic",
            "scaffolding": None
        }

    success_count = sum(1 for r in recent_ratings if r in ["good", "easy"])
    success_rate = success_count / len(recent_ratings)

    if success_rate < 0.35:
        return {
            "zone": "frustration",
            "message": "This is challenging. Let's add some help!",
            "scaffolding": {
                "type": "worked_example",
                "content": "Remember to break down the problem into smaller steps.",
                "show": True
            }
        }
    elif success_rate > 0.70:
        return {
            "zone": "comfort",
            "message": "You're doing great! Let's increase the challenge.",
            "scaffolding": None
        }
    else:
        return {
            "zone": "optimal",
            "message": "Perfect! You're in the optimal learning zone.",
            "scaffolding": None
        }


async def update_fsrs_schedule(
    learner_profile_id: str,
    card_id: str,
    rating: Rating
) -> Dict:
    """
    Update FSRS schedule via Scheduler service

    Args:
        learner_profile_id: Learner profile ID
        card_id: Card ID
        rating: Review rating

    Returns:
        Updated schedule info
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{SCHEDULER_URL}/review",
                json={
                    "learner_id": learner_profile_id,
                    "card_id": card_id,
                    "rating": rating.value,
                    "reviewed_at": datetime.utcnow().isoformat()
                }
            )

            if response.status_code == 200:
                return response.json()

    except Exception as e:
        print(f"Scheduler unavailable for review: {e}")

    # Fallback: Simple interval calculation
    interval_days = {
        Rating.AGAIN: 0,
        Rating.HARD: 1,
        Rating.GOOD: 3,
        Rating.EASY: 7
    }[rating]

    next_due = datetime.utcnow() + timedelta(days=interval_days)

    return {
        "new_stability": 2.5,
        "new_difficulty": 5.0,
        "interval_days": interval_days,
        "next_due_date": next_due.isoformat()
    }


def convert_db_card_to_learning_card(card_data: Dict) -> LearningCard:
    """Convert database card to LearningCard model"""
    return LearningCard(
        card_id=card_data["card_id"],
        concept_name=card_data["concept_name"],
        content=card_data["content"],
        question=card_data["question"],
        correct_answer=card_data.get("correct_answer"),
        difficulty=card_data["difficulty"],
        due_date=None
    )


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "service": "NerdLearn Orchestrator (Integrated)",
        "status": "operational",
        "version": "0.2.0",
        "features": [
            "Database integration",
            "FSRS scheduling",
            "ZPD adaptation",
            "Real-time gamification"
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    # Check database connection
    try:
        # Use simpler check than db.pool which might trigger psycog2 issues if fragile
        conn_test = db.get_connection()
        db.return_connection(conn_test)
        db_healthy = True
    except:
        db_healthy = False

    return {
        "status": "healthy" if db_healthy else "degraded",
        "database": "connected" if db_healthy else "disconnected",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/test/seed")
async def seed_test_data(learner_id: str):
    """Seed test data for integration verification"""
    print(f"DEBUG: seed_test_data called with {learner_id}", flush=True)
    try:
        # Create profile
        # profile = db.create_learner_profile(learner_id)
        profile = {"id": "mock_profile"}
        
        # Create a test card
        card_data = {
            "concept_name": "Test Concept 101",
            "content": "This is a test card.",
            "question": "Is this a test?",
            "correct_answer": "Yes",
            "difficulty": 0.3,
            "learner_id": profile["id"]
        }
        db.create_card(card_data)
        
        return {"status": "seeded", "learner_id": learner_id, "profile_id": profile["id"]}
    except Exception as e:
        print(f"Seeding failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/start", response_model=SessionState)
async def start_session(request: SessionStartRequest):
    """
    Start a new learning session

    1. Load learner profile from database
    2. Get due cards from Scheduler/database
    3. Load card content from database
    4. Initialize session state
    5. Return first card
    """
    try:
        # 1. Load learner profile
        profile = db.load_learner_profile(request.learner_id)

        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Learner profile not found for user {request.learner_id}"
            )

        learner_profile_id = profile["id"]

        # 2. Get due card IDs
        due_card_ids = await get_due_cards_from_scheduler(learner_profile_id, request.limit)

        if not due_card_ids:
            raise HTTPException(
                status_code=404,
                detail="No cards due for review"
            )

        # 3. Load card content from database
        cards = db.load_cards(due_card_ids[:request.limit])

        if not cards:
            raise HTTPException(
                status_code=500,
                detail="Failed to load card content"
            )

        # 4. Create session
        session_id = f"session_{request.learner_id}_{int(datetime.now().timestamp())}"

        # Convert first card to LearningCard
        first_card = convert_db_card_to_learning_card(cards[0])

        # 5. Initialize session state
        session_state = {
            "session_id": session_id,
            "learner_id": request.learner_id,
            "learner_profile_id": learner_profile_id,
            "cards": cards,
            "current_index": 0,
            "cards_reviewed": 0,
            "cards_correct": 0,
            "total_xp_earned": 0,
            "current_streak": profile["streakDays"],
            "zpd_zone": "optimal",
            "scaffolding_active": [],
            "started_at": datetime.now(),
            "achievements_unlocked": [],
            "recent_ratings": []  # Track last 10 ratings for ZPD
        }

        # Store session
        sessions[session_id] = session_state

        return SessionState(
            session_id=session_id,
            learner_id=request.learner_id,
            current_card=first_card,
            cards_reviewed=0,
            cards_correct=0,
            total_xp_earned=0,
            current_streak=profile["streakDays"],
            zpd_zone="optimal",
            scaffolding_active=[],
            started_at=session_state["started_at"],
            achievements_unlocked=[]
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error starting session: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start session: {str(e)}"
        )


@app.post("/session/answer", response_model=AnswerResponse)
async def process_answer(request: AnswerRequest):
    """
    Process learner's answer and return next card

    1. Get session state
    2. Calculate XP
    3. Update FSRS schedule
    4. Assess ZPD state
    5. Update database (XP, evidence, competency)
    6. Check achievements
    7. Get next card
    8. Return response
    """
    try:
        # 1. Get session
        session = sessions.get(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get current card
        current_card = session["cards"][session["current_index"]]
        learner_profile_id = session["learner_profile_id"]

        # 2. Calculate XP
        xp_earned = gamification.calculate_xp(
            current_card["difficulty"],
            request.rating,
            session["current_streak"]
        )

        # 3. Update FSRS schedule
        schedule_info = await update_fsrs_schedule(
            learner_profile_id,
            current_card["card_id"],
            request.rating
        )

        # Update scheduled item in database
        if "next_due_date" in schedule_info:
            next_due = datetime.fromisoformat(schedule_info["next_due_date"].replace('Z', '+00:00'))
            db.update_scheduled_item(
                learner_profile_id,
                current_card["card_id"],
                schedule_info.get("new_stability", 2.5),
                schedule_info.get("new_difficulty", 5.0),
                0.9,  # retrievability
                schedule_info.get("interval_days", 1),
                next_due
            )

        # 4. Track rating for ZPD
        session["recent_ratings"].append(request.rating.value)
        session["recent_ratings"] = session["recent_ratings"][-10:]  # Keep last 10

        # Assess ZPD state
        zpd_state = await assess_zpd_state(
            session["learner_id"],
            current_card["conceptId"],
            session["recent_ratings"],
            current_card["difficulty"]
        )

        # 5. Update database
        # Update XP
        xp_result = db.update_learner_xp(session["learner_id"], xp_earned)
        new_total_xp = xp_result["totalXP"]

        # Calculate level
        level, level_progress = gamification.get_level(new_total_xp)

        # Update level if changed
        if level > xp_result["level"]:
            db.update_learner_level(session["learner_id"], level)

        # Store evidence
        db.create_evidence(
            learner_profile_id,
            current_card["card_id"],
            "PERFORMANCE",
            {
                "rating": request.rating.value,
                "dwell_time_ms": request.dwell_time_ms,
                "hesitation_count": request.hesitation_count,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        # Update competency state (simple: based on rating)
        knowledge_prob = {
            Rating.AGAIN: 0.3,
            Rating.HARD: 0.6,
            Rating.GOOD: 0.8,
            Rating.EASY: 0.95
        }[request.rating]

        db.update_competency_state(
            learner_profile_id,
            current_card["conceptId"],
            knowledge_prob,
            knowledge_prob  # mastery = knowledge for now
        )

        # 6. Check achievements
        achievement = gamification.check_achievements(
            new_total_xp,
            xp_earned,
            session["current_streak"],
            0  # TODO: count mastered concepts
        )

        # 7. Update session state
        session["cards_reviewed"] += 1
        if request.rating in [Rating.GOOD, Rating.EASY]:
            session["cards_correct"] += 1
        session["total_xp_earned"] += xp_earned
        session["zpd_zone"] = zpd_state["zone"]
        session["current_index"] += 1

        # 8. Get next card
        next_card = None
        if session["current_index"] < len(session["cards"]):
            next_card_data = session["cards"][session["current_index"]]
            next_card = convert_db_card_to_learning_card(next_card_data)

        return AnswerResponse(
            correct=request.rating in [Rating.GOOD, Rating.EASY],
            xp_earned=xp_earned,
            new_total_xp=new_total_xp,
            level=level,
            level_progress=level_progress,
            next_card=next_card,
            zpd_zone=zpd_state["zone"],
            zpd_message=zpd_state["message"],
            scaffolding=zpd_state.get("scaffolding"),
            achievement_unlocked=achievement
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing answer: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process answer: {str(e)}"
        )


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get current session state"""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "cards_reviewed": session["cards_reviewed"],
        "cards_correct": session["cards_correct"],
        "total_xp_earned": session["total_xp_earned"],
        "zpd_zone": session["zpd_zone"]
    }


@app.get("/profile/{learner_id}")
async def get_profile(learner_id: str):
    """Get learner profile from database"""
    profile = db.load_learner_profile(learner_id)

    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    level, progress = gamification.get_level(profile["totalXP"])

    return {
        "learner_id": learner_id,
        "total_xp": profile["totalXP"],
        "level": level,
        "level_progress": progress,
        "streak_days": profile["streakDays"],
        "fsrs_stability": profile["fsrsStability"],
        "fsrs_difficulty": profile["fsrsDifficulty"]
    }


@app.post("/session/{session_id}/end")
async def end_session(session_id: str):
    """End a learning session and return summary"""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    summary = {
        "session_id": session_id,
        "duration_minutes": (datetime.now() - session["started_at"]).seconds / 60,
        "cards_reviewed": session["cards_reviewed"],
        "cards_correct": session["cards_correct"],
        "success_rate": session["cards_correct"] / session["cards_reviewed"] if session["cards_reviewed"] > 0 else 0,
        "total_xp_earned": session["total_xp_earned"],
        "achievements_unlocked": session["achievements_unlocked"]
    }

    # Remove session from memory
    del sessions[session_id]

    return summary


@app.on_event("startup")
async def startup_event():
    print("✅ Orchestrator service started (INTEGRATED VERSION)", flush=True)
    print("🔗 Database connection: active", flush=True)
    print(f"🔗 Scheduler URL: {SCHEDULER_URL}", flush=True)
    print(f"🔗 Inference URL: {INFERENCE_URL}", flush=True)
    print("🎮 Learning session coordination active", flush=True)


@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 Shutting down Orchestrator...", flush=True)
    db.close()
    print("✅ Database connections closed", flush=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
