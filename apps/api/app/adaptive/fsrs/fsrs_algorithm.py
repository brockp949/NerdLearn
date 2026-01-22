"""
FSRS (Free Spaced Repetition Scheduler) Algorithm
Based on the paper: "A Stochastic Shortest Path Algorithm for Optimizing Spaced Repetition Scheduling"

FSRS is a modern spaced repetition algorithm that optimizes review scheduling
using stability, difficulty, and retrievability calculations.
"""
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
from enum import Enum
import math


class Rating(Enum):
    """Review rating options"""
    AGAIN = 1  # Completely forgot
    HARD = 2   # Difficult to recall
    GOOD = 3   # Recalled correctly
    EASY = 4   # Recalled easily


class FSRSCard:
    """
    Represents a flashcard with FSRS parameters
    """

    def __init__(
        self,
        concept_id: int,
        user_id: int,
        stability: float = 0.0,
        difficulty: float = 0.0,
        elapsed_days: int = 0,
        scheduled_days: int = 0,
        reps: int = 0,
        lapses: int = 0,
        state: str = "new",  # new, learning, review, relearning
        last_review: Optional[datetime] = None,
        due: Optional[datetime] = None,
    ):
        self.concept_id = concept_id
        self.user_id = user_id
        self.stability = stability
        self.difficulty = difficulty
        self.elapsed_days = elapsed_days
        self.scheduled_days = scheduled_days
        self.reps = reps
        self.lapses = lapses
        self.state = state
        self.last_review = last_review
        self.due = due or datetime.now()

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "concept_id": self.concept_id,
            "user_id": self.user_id,
            "stability": self.stability,
            "difficulty": self.difficulty,
            "elapsed_days": self.elapsed_days,
            "scheduled_days": self.scheduled_days,
            "reps": self.reps,
            "lapses": self.lapses,
            "state": self.state,
            "last_review": self.last_review.isoformat() if self.last_review else None,
            "due": self.due.isoformat() if self.due else None,
        }


class FSRSAlgorithm:
    """
    FSRS Algorithm Implementation

    Core formulas:
    - Retrievability R(t,S) = (1 + t/(9*S))^(-1)
    - Next stability S' depends on rating and current parameters
    - Difficulty D updates based on performance
    """

    # Default FSRS parameters (optimized for general learning)
    DEFAULT_PARAMS = {
        "w": [
            0.4,    # w[0]: Initial stability for AGAIN
            0.6,    # w[1]: Initial stability for HARD
            2.4,    # w[2]: Initial stability for GOOD
            5.8,    # w[3]: Initial stability for EASY
            4.93,   # w[4]: Stability multiplier base
            0.94,   # w[5]: Stability multiplier factor
            0.86,   # w[6]: Difficulty weight
            0.01,   # w[7]: Difficulty decay
            1.49,   # w[8]: Stability increase for successful recall
            0.14,   # w[9]: Stability factor for difficulty
            0.94,   # w[10]: Difficulty penalty for AGAIN
            2.18,   # w[11]: Difficulty reward for EASY
            0.05,   # w[12]: Retrievability threshold
            0.34,   # w[13]: Memory decay exponent
            1.26,   # w[14]: Forgetting curve shape
            0.29,   # w[15]: Difficulty scaling
            2.61,   # w[16]: Stability growth rate
        ],
        "request_retention": 0.9,  # Target retention rate (90%)
        "maximum_interval": 36500,  # Max interval in days (100 years)
        "easy_bonus": 1.3,  # Bonus multiplier for EASY rating
        "hard_penalty": 1.2,  # Penalty for HARD rating
    }

    def __init__(self, params: Optional[Dict] = None):
        """Initialize with optional custom parameters"""
        self.params = params or self.DEFAULT_PARAMS
        self.w = self.params["w"]

    def calculate_retrievability(self, elapsed_days: int, stability: float) -> float:
        """
        Calculate retrievability (probability of recall)

        Formula: R(t,S) = 0.9^(t/S)
        Based on最新的 FSRS research for exponential memory decay.

        Args:
            elapsed_days: Days since last review
            stability: Memory stability

        Returns:
            Retrievability between 0 and 1
        """
        if stability == 0:
            return 0.0
        return math.pow(0.9, elapsed_days / stability)

    def init_stability(self, rating: Rating) -> float:
        """
        Calculate initial stability for a new card

        Args:
            rating: First review rating

        Returns:
            Initial stability value
        """
        return self.w[rating.value - 1]

    def init_difficulty(self, rating: Rating) -> float:
        """
        Calculate initial difficulty for a new card

        Args:
            rating: First review rating

        Returns:
            Initial difficulty (1-10 scale)
        """
        # Difficulty increases for harder ratings
        difficulty = self.w[4] - self.w[5] * (rating.value - 3)
        return max(1, min(10, difficulty))

    def next_stability(
        self,
        current_stability: float,
        difficulty: float,
        retrievability: float,
        rating: Rating,
    ) -> float:
        """
        Calculate next stability after a review

        Args:
            current_stability: Current memory stability
            difficulty: Card difficulty
            retrievability: Current retrievability
            rating: Review rating

        Returns:
            New stability value
        """
        if rating == Rating.AGAIN:
            # Failed recall - reset stability
            new_stability = (
                self.w[11]
                * pow(difficulty, -self.w[12])
                * (pow(current_stability + 1, self.w[13]) - 1)
                * math.exp((1 - retrievability) * self.w[14])
            )
        else:
            # Successful recall - increase stability
            hard_penalty = self.w[15] if rating == Rating.HARD else 1
            easy_bonus = self.w[16] if rating == Rating.EASY else 1

            new_stability = (
                current_stability
                * (
                    1
                    + math.exp(self.w[8])
                    * (11 - difficulty)
                    * pow(current_stability, -self.w[9])
                    * (math.exp((1 - retrievability) * self.w[10]) - 1)
                    * hard_penalty
                    * easy_bonus
                )
            )

        return max(0.1, new_stability)

    def next_difficulty(self, current_difficulty: float, rating: Rating) -> float:
        """
        Calculate next difficulty after a review

        Args:
            current_difficulty: Current difficulty
            rating: Review rating

        Returns:
            New difficulty (1-10 scale)
        """
        # Difficulty adjustment based on rating
        delta = rating.value - 3  # -2, -1, 0, +1

        new_difficulty = current_difficulty - self.w[6] * delta

        # Mean reversion to prevent extreme values
        new_difficulty = new_difficulty * (1 - self.w[7]) + self.w[4] * self.w[7]

        return max(1, min(10, new_difficulty))

    def next_interval(self, stability: float) -> int:
        """
        Calculate optimal next review interval

        Formula: I = S * ln(R_req) / ln(0.9)
        Derived from R = 0.9^(I/S)

        Args:
            stability: Memory stability

        Returns:
            Interval in days
        """
        # Calculate interval that maintains target retention
        request_retention = self.params["request_retention"]
        
        # Logarithmic derivation for the next interval
        interval = stability * (math.log(request_retention) / math.log(0.9))

        # Apply constraints
        interval = max(1, min(self.params["maximum_interval"], int(interval)))

        return interval

    def review_card(
        self, card: FSRSCard, rating: Rating, review_time: Optional[datetime] = None
    ) -> Tuple[FSRSCard, Dict]:
        """
        Process a card review and update its state

        Args:
            card: The card being reviewed
            rating: User's rating
            review_time: Time of review (default: now)

        Returns:
            (Updated card, Review log)
        """
        review_time = review_time or datetime.now()

        # Calculate elapsed time
        if card.last_review:
            elapsed_days = (review_time - card.last_review).days
        else:
            elapsed_days = 0

        # Handle new cards
        if card.state == "new":
            card.stability = self.init_stability(rating)
            card.difficulty = self.init_difficulty(rating)
            card.state = "learning" if rating == Rating.AGAIN else "review"
        else:
            # Calculate retrievability
            retrievability = self.calculate_retrievability(elapsed_days, card.stability)

            # Update stability and difficulty
            card.stability = self.next_stability(
                card.stability, card.difficulty, retrievability, rating
            )
            card.difficulty = self.next_difficulty(card.difficulty, rating)

            # Update state
            if rating == Rating.AGAIN:
                card.lapses += 1
                card.state = "relearning"
            else:
                card.state = "review"

        # Calculate next interval
        interval = self.next_interval(card.stability)

        # Update card metadata
        card.reps += 1
        card.elapsed_days = elapsed_days
        card.scheduled_days = interval
        card.last_review = review_time
        card.due = review_time + timedelta(days=interval)

        # Create review log
        log = {
            "card_id": f"{card.user_id}_{card.concept_id}",
            "rating": rating.value,
            "review_time": review_time.isoformat(),
            "elapsed_days": elapsed_days,
            "scheduled_days": interval,
            "stability": card.stability,
            "difficulty": card.difficulty,
            "state": card.state,
            "reps": card.reps,
        }

        return card, log

    def get_next_states(self, card: FSRSCard) -> Dict[Rating, Dict]:
        """
        Preview what would happen for each rating
        Useful for showing users predicted intervals

        Args:
            card: Current card state

        Returns:
            Dictionary mapping ratings to predicted outcomes
        """
        predictions = {}

        for rating in Rating:
            # Create temporary card copy
            temp_card = FSRSCard(
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

            # Calculate what would happen
            updated_card, log = self.review_card(temp_card, rating)

            predictions[rating] = {
                "interval": updated_card.scheduled_days,
                "due": updated_card.due.isoformat() if updated_card.due else None,
                "stability": updated_card.stability,
                "difficulty": updated_card.difficulty,
            }

        return predictions
