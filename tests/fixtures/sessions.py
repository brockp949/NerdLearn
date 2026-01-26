"""
Session fixtures and factories for testing.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid

from .cards import create_mock_card


@dataclass
class SessionFactory:
    """Factory for creating test session data."""

    default_limit: int = 10

    def create(
        self,
        learner_id: Optional[str] = None,
        card_count: int = 10
    ) -> Dict:
        """Create a test session data dict."""
        return create_mock_session(
            learner_id=learner_id,
            card_count=card_count
        )


def create_mock_session(
    session_id: Optional[str] = None,
    learner_id: Optional[str] = None,
    card_count: int = 10,
    include_cards: bool = True
) -> Dict:
    """
    Generate mock session data.

    Args:
        session_id: Session identifier (auto-generated if None)
        learner_id: Learner identifier (auto-generated if None)
        card_count: Number of cards in session
        include_cards: Whether to include card data

    Returns:
        Dict with session data
    """
    session = {
        "session_id": session_id or f"session_{uuid.uuid4().hex[:8]}",
        "learner_id": learner_id or f"learner_{uuid.uuid4().hex[:8]}",
        "started_at": datetime.utcnow().isoformat(),
        "total_cards": card_count,
        "cards_remaining": card_count,
        "cards_completed": 0,
        "total_xp": 0
    }

    if include_cards and card_count > 0:
        session["current_card"] = create_mock_card()
    else:
        session["current_card"] = None

    return session


def create_mock_answer(
    session_id: str,
    card_id: str,
    rating: str = "good",
    dwell_time_ms: int = 10000,
    hesitation_count: int = 1
) -> Dict:
    """
    Generate mock answer data for a card.

    Args:
        session_id: Session identifier
        card_id: Card identifier
        rating: Answer rating (again, hard, good, easy)
        dwell_time_ms: Time spent on card in ms
        hesitation_count: Number of hesitations

    Returns:
        Dict with answer data
    """
    return {
        "session_id": session_id,
        "card_id": card_id,
        "rating": rating,
        "dwell_time_ms": dwell_time_ms,
        "hesitation_count": hesitation_count,
        "answered_at": datetime.utcnow().isoformat()
    }


def create_mock_answer_response(
    rating: str = "good",
    xp_earned: int = 15,
    zpd_zone: str = "optimal",
    next_card: Optional[Dict] = None
) -> Dict:
    """
    Generate mock answer response data.

    Args:
        rating: The rating given
        xp_earned: XP earned from the answer
        zpd_zone: Current ZPD zone
        next_card: Next card data (None if session complete)

    Returns:
        Dict with answer response data
    """
    xp_map = {"again": 5, "hard": 10, "good": 15, "easy": 25}

    return {
        "xp_earned": xp_earned or xp_map.get(rating, 15),
        "zpd_zone": zpd_zone,
        "next_card": next_card,
        "scheduling_info": {
            "interval_days": 1 if rating == "again" else 3,
            "next_review": datetime.utcnow().isoformat()
        }
    }
