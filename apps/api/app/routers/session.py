from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any
import uuid
import random
from datetime import datetime
from app.schemas.session import SessionStartRequest, SessionState, AnswerRequest, AnswerResponse, LearningCardResponse
from app.adaptive.fsrs import FSRSAlgorithm
from app.adaptive.bkt import BayesianKnowledgeTracer

router = APIRouter()

# In-memory storage for MVP sessions
# In production, use Redis
_sessions: Dict[str, SessionState] = {}

# Initialize algorithms
fsrs = FSRSAlgorithm()
bkt = BayesianKnowledgeTracer()

# Mock Content Database
MOCK_CARDS = [
    {
        "id": "c1",
        "type": "concept",
        "title": "Python Variables",
        "content": "Variables are containers for storing data values. In Python, you don't need to declare variables before using them.",
        "difficulty": 1.0,
    },
    {
        "id": "q1",
        "type": "question",
        "title": "Variable Assignment",
        "content": "Which of the following is a valid variable assignment in Python?",
        "options": [
            {"id": "a", "text": "x = 5"},
            {"id": "b", "text": "int x = 5"},
            {"id": "c", "text": "5 = x"},
            {"id": "d", "text": "var x = 5"}
        ],
        "correct_option_id": "a",
        "difficulty": 2.0,
    },
    {
        "id": "c2",
        "type": "concept",
        "title": "Python Lists",
        "content": "Lists are used to store multiple items in a single variable. Lists are created using square brackets.",
        "difficulty": 2.5,
    },
    {
        "id": "q2",
        "type": "question",
        "title": "List Indexing",
        "content": "What is the index of the first item in a list?",
        "options": [
            {"id": "a", "text": "1"},
            {"id": "b", "text": "0"},
            {"id": "c", "text": "-1"},
            {"id": "d", "text": "None"}
        ],
        "correct_option_id": "b",
        "difficulty": 3.0,
    }
]

@router.post("/start", response_model=SessionState)
async def start_session(request: SessionStartRequest):
    session_id = str(uuid.uuid4())
    
    # Select first card (Mock logic for now)
    # TODO: Connect to AdaptiveService for real recommendations
    first_card_data = MOCK_CARDS[0]
    
    current_card = LearningCardResponse(
        card_id=first_card_data["id"],
        type=first_card_data["type"],
        title=first_card_data["title"],
        content=first_card_data["content"],
        options=first_card_data.get("options"),
        correct_option_id=first_card_data.get("correct_option_id"),
        difficulty=first_card_data["difficulty"],
        context={"domain": request.domain}
    )

    session = SessionState(
        session_id=session_id,
        learner_id=request.learner_id,
        current_card=current_card,
        cards_reviewed=0,
        cards_correct=0,
        total_xp_earned=0,
        started_at=datetime.utcnow()
    )
    
    _sessions[session_id] = session
    return session

@router.post("/answer", response_model=AnswerResponse)
async def submit_answer(request: AnswerRequest):
    if request.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = _sessions[request.session_id]
    
    # Validate Answer
    # For MVP, we assume "correct" if rating is "good" or "easy" OR check mock data correctness
    # The frontend sends "rating" (again, hard, good, easy) which implies the user's self-eval or result.
    # But for questions, the frontend might compute correctness locally or we do it here.
    # LearnPage.tsx handleAnswer sends `rating` which is expected to be `Rating` enum?
    # Actually LearnPage calls it with `rating`. Let's assume 'good'/'easy' means correct.
    
    is_correct = request.rating in ["good", "easy"]
    
    # Verify against mock data if possible
    card_data = next((c for c in MOCK_CARDS if c["id"] == request.card_id), None)
    if card_data and card_data.get("type") == "question":
        # In a real app, the request should include the selected_option_id if it's a question
        # But the current schema only has rating.
        # We'll trust the rating for this mock implementation as "Perceived Mastery" or "Correctness"
        pass

    # Update Session Stats
    session.cards_reviewed += 1
    xp = 10 if is_correct else 2
    if is_correct:
         session.cards_correct += 1
         session.total_xp_earned += xp
         session.current_streak += 1
    else:
         session.current_streak = 0
         
    # Pick Next Card (Simple sequential or random)
    # In real app: adaptive.get_recommendations()
    next_idx = session.cards_reviewed % len(MOCK_CARDS)
    next_card_data = MOCK_CARDS[next_idx]
    
    next_card = LearningCardResponse(
        card_id=next_card_data["id"],
        type=next_card_data["type"],
        title=next_card_data["title"],
        content=next_card_data["content"],
        options=next_card_data.get("options"),
        correct_option_id=next_card_data.get("correct_option_id"),
        difficulty=next_card_data["difficulty"]
    )
    
    session.current_card = next_card
    
    return AnswerResponse(
        correct=is_correct,
        xp_earned=xp,
        new_total_xp=session.total_xp_earned,
        level=1 + (session.total_xp_earned // 100),
        level_progress=(session.total_xp_earned % 100) / 100.0,
        next_card=next_card,
        zpd_zone="optimal",
        zpd_message="Keeping you in the flow!",
        scaffolding={}
    )
