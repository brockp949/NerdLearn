"""
Orchestrator Service - Learning Session Coordinator

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
from datetime import datetime
from enum import Enum
import httpx
import asyncio
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

SCHEDULER_URL = "http://localhost:8001"
TELEMETRY_URL = "http://localhost:8002"
INFERENCE_URL = "http://localhost:8003"
CONTENT_URL = "http://localhost:8004"

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

sessions: Dict[str, SessionState] = {}
learner_profiles: Dict[str, Dict] = {}  # learner_id -> {xp, level, streak, etc.}


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
        learner_profile: Dict,
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
        total_xp = learner_profile.get("total_xp", 0) + new_xp
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
    title="NerdLearn Orchestrator",
    description="Learning session coordinator",
    version="0.1.0"
)

http_client = httpx.AsyncClient()
gamification = GamificationEngine()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_learner_profile(learner_id: str) -> Dict:
    """Get or create learner profile"""
    if learner_id not in learner_profiles:
        learner_profiles[learner_id] = {
            "total_xp": 0,
            "level": 1,
            "streak_days": 0,
            "last_activity": None,
            "concepts_mastered": 0
        }
    return learner_profiles[learner_id]


async def get_due_cards(learner_id: str, limit: int = 20) -> List[Dict]:
    """Get cards due for review from Scheduler"""
    try:
        # For demo, return mock cards
        # TODO: Call actual scheduler service
        return [
            {
                "card_id": f"card_{i}",
                "concept_id": f"concept_{i}",
                "difficulty": 5.0 + (i % 5),
                "due_date": datetime.now().isoformat()
            }
            for i in range(min(limit, 5))
        ]
    except Exception as e:
        print(f"Error getting due cards: {e}")
        return []


def create_learning_card(card_data: Dict, concept_name: str = "Python Functions") -> LearningCard:
    """Convert card data to LearningCard"""

    # Demo content (TODO: Load from database)
    demo_content = {
        "card_0": {
            "content": "A **function** is a reusable block of code that performs a specific task. Functions help organize code and avoid repetition.",
            "question": "What keyword is used to define a function in Python?",
            "correct_answer": "def"
        },
        "card_1": {
            "content": "**Parameters** are variables that you pass to a function. They allow functions to work with different inputs.",
            "question": "How do you pass multiple parameters to a function?",
            "correct_answer": "Separate them with commas"
        },
        "card_2": {
            "content": "The **return** statement sends a value back from a function to where it was called.",
            "question": "What happens if a function doesn't have a return statement?",
            "correct_answer": "Returns None"
        },
        "card_3": {
            "content": "**Recursion** is when a function calls itself. It's useful for problems that can be broken into smaller similar problems.",
            "question": "What is the essential component of a recursive function?",
            "correct_answer": "Base case"
        },
        "card_4": {
            "content": "**Lambda functions** are small anonymous functions defined with the lambda keyword. They can have any number of parameters but only one expression.",
            "question": "When would you use a lambda function instead of a regular function?",
            "correct_answer": "For simple, one-line operations"
        },
    }

    card_id = card_data["card_id"]
    demo = demo_content.get(card_id, demo_content["card_0"])

    return LearningCard(
        card_id=card_id,
        concept_name=concept_name,
        content=demo["content"],
        question=demo["question"],
        correct_answer=demo["correct_answer"],
        difficulty=card_data.get("difficulty", 5.0),
        due_date=card_data.get("due_date")
    )


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "service": "NerdLearn Orchestrator",
        "status": "operational",
        "version": "0.1.0"
    }


@app.post("/session/start", response_model=SessionState)
async def start_session(request: SessionStartRequest):
    """
    Start a new learning session

    1. Get due cards from Scheduler
    2. Initialize session state
    3. Return first card
    """
    try:
        # Get due cards
        due_cards = await get_due_cards(request.learner_id, request.limit)

        if not due_cards:
            raise HTTPException(
                status_code=404,
                detail="No cards due for review"
            )

        # Create session
        session_id = f"session_{request.learner_id}_{int(datetime.now().timestamp())}"

        # Get learner profile
        profile = get_learner_profile(request.learner_id)

        # Get first card
        first_card = create_learning_card(due_cards[0])

        # Create session state
        session_state = SessionState(
            session_id=session_id,
            learner_id=request.learner_id,
            current_card=first_card,
            cards_reviewed=0,
            cards_correct=0,
            total_xp_earned=0,
            current_streak=profile["streak_days"],
            zpd_zone="optimal",
            scaffolding_active=[],
            started_at=datetime.now(),
            achievements_unlocked=[]
        )

        # Store session
        sessions[session_id] = session_state

        return session_state

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error starting session: {str(e)}"
        )


@app.post("/session/answer", response_model=AnswerResponse)
async def process_answer(request: AnswerRequest):
    """
    Process a learner's answer

    Flow:
    1. Update Scheduler (FSRS)
    2. Update Inference (ZPD)
    3. Send to Telemetry
    4. Calculate XP
    5. Check achievements
    6. Get next card
    7. Apply scaffolding if needed
    """
    try:
        # Get session
        if request.session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = sessions[request.session_id]

        # Get learner profile
        profile = get_learner_profile(session.learner_id)

        # Determine if answer is correct (simplified for demo)
        is_correct = request.rating in [Rating.GOOD, Rating.EASY]

        # Update session stats
        session.cards_reviewed += 1
        if is_correct:
            session.cards_correct += 1

        # Calculate XP
        current_card = session.current_card
        xp_earned = gamification.calculate_xp(
            current_card.difficulty if current_card else 5.0,
            request.rating,
            profile["streak_days"]
        )

        session.total_xp_earned += xp_earned
        profile["total_xp"] += xp_earned

        # Calculate level
        level, level_progress = gamification.get_level(profile["total_xp"])
        old_level = profile["level"]
        profile["level"] = level

        # Check for achievements
        achievement = gamification.check_achievements(
            profile,
            xp_earned,
            profile["streak_days"],
            profile["concepts_mastered"]
        )

        if achievement:
            session.achievements_unlocked.append(achievement["name"])

        # Calculate success rate for ZPD
        success_rate = session.cards_correct / session.cards_reviewed if session.cards_reviewed > 0 else 0.5

        # Determine ZPD zone
        if success_rate < 0.35:
            zpd_zone = "frustration"
            zpd_message = "You're struggling with this topic. Let's add some help!"
            scaffolding = {
                "type": "worked_example",
                "content": "Here's a complete example to help you understand...",
                "show": True
            }
        elif success_rate > 0.70:
            zpd_zone = "comfort"
            zpd_message = "You're doing great! Let's increase the challenge."
            scaffolding = None
        else:
            zpd_zone = "optimal"
            zpd_message = "Perfect! You're in the optimal learning zone."
            scaffolding = None

        session.zpd_zone = zpd_zone

        # Get next card (for demo, cycle through available cards)
        # TODO: Get from scheduler based on ZPD state
        due_cards = await get_due_cards(session.learner_id)
        next_card = None
        if session.cards_reviewed < len(due_cards):
            next_card_data = due_cards[session.cards_reviewed]
            next_card = create_learning_card(next_card_data)
            session.current_card = next_card

        # Prepare response
        response = AnswerResponse(
            correct=is_correct,
            xp_earned=xp_earned,
            new_total_xp=profile["total_xp"],
            level=level,
            level_progress=level_progress,
            next_card=next_card,
            zpd_zone=zpd_zone,
            zpd_message=zpd_message,
            scaffolding=scaffolding,
            achievement_unlocked=achievement
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing answer: {str(e)}"
        )


@app.get("/session/{session_id}", response_model=SessionState)
async def get_session(session_id: str):
    """Get current session state"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return sessions[session_id]


@app.get("/profile/{learner_id}")
async def get_profile(learner_id: str):
    """Get learner profile"""
    profile = get_learner_profile(learner_id)

    level, level_progress = gamification.get_level(profile["total_xp"])

    return {
        **profile,
        "level": level,
        "level_progress": level_progress,
        "xp_to_next_level": gamification.xp_for_level(level + 1) - profile["total_xp"]
    }


@app.post("/session/{session_id}/end")
async def end_session(session_id: str):
    """End a learning session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    # Calculate session summary
    summary = {
        "session_id": session_id,
        "cards_reviewed": session.cards_reviewed,
        "cards_correct": session.cards_correct,
        "success_rate": session.cards_correct / session.cards_reviewed if session.cards_reviewed > 0 else 0,
        "xp_earned": session.total_xp_earned,
        "achievements_unlocked": session.achievements_unlocked,
        "duration_minutes": (datetime.now() - session.started_at).total_seconds() / 60
    }

    # Clean up session
    del sessions[session_id]

    return summary


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    print("✅ Orchestrator service started")
    print("🎮 Learning session coordination active")


@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()
    print("🛑 Orchestrator service stopped")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
