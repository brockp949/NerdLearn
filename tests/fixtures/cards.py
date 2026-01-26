"""
Card fixtures and factories for testing.
"""

from typing import Dict, Optional
from dataclasses import dataclass
import uuid


@dataclass
class CardFactory:
    """Factory for creating test card data."""

    default_difficulty: float = 3.5
    default_card_type: str = "BASIC"

    def create(
        self,
        card_id: Optional[str] = None,
        concept_id: Optional[str] = None,
        difficulty: Optional[float] = None
    ) -> Dict:
        """Create a test card data dict."""
        return create_mock_card(
            card_id=card_id,
            concept_id=concept_id,
            difficulty=difficulty or self.default_difficulty,
            card_type=self.default_card_type
        )


def create_mock_card(
    card_id: Optional[str] = None,
    concept_id: Optional[str] = None,
    concept_name: str = "Python Variables",
    content: str = "**Variables** are containers for storing data values.",
    question: str = "What keyword is used to assign a value to a variable?",
    correct_answer: str = "=",
    difficulty: float = 3.5,
    card_type: str = "BASIC"
) -> Dict:
    """
    Generate mock card data.

    Args:
        card_id: Card identifier (auto-generated if None)
        concept_id: Concept identifier (auto-generated if None)
        concept_name: Name of the concept
        content: Card content/explanation
        question: Card question
        correct_answer: Expected answer
        difficulty: Card difficulty (1-10)
        card_type: Type of card (BASIC, CLOZE, etc.)

    Returns:
        Dict with card data
    """
    return {
        "card_id": card_id or f"card_{uuid.uuid4().hex[:8]}",
        "concept_id": concept_id or f"concept_{uuid.uuid4().hex[:8]}",
        "concept_name": concept_name,
        "content": content,
        "question": question,
        "correct_answer": correct_answer,
        "difficulty": difficulty,
        "card_type": card_type
    }


def create_mock_card_batch(count: int, base_difficulty: float = 3.0) -> list:
    """
    Generate a batch of mock cards with varying difficulties.

    Args:
        count: Number of cards to generate
        base_difficulty: Starting difficulty level

    Returns:
        List of card dicts
    """
    cards = []
    for i in range(count):
        difficulty = base_difficulty + (i * 0.5)  # Increase difficulty
        cards.append(create_mock_card(
            card_id=f"card_batch_{i}",
            concept_id=f"concept_batch_{i}",
            concept_name=f"Concept {i+1}",
            difficulty=min(difficulty, 10.0)  # Cap at 10
        ))
    return cards
