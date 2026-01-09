"""
FSRS Spaced Repetition System

Includes:
- FSRSAlgorithm: Core FSRS-4.5 scheduling algorithm
- FSRSCard: Card model with stability, difficulty, retrievability
- Rating: Review rating enum (AGAIN, HARD, GOOD, EASY)
- Parameter Learning: Per-user optimization for 5-10% improvement
"""
from .fsrs_algorithm import FSRSAlgorithm, FSRSCard, Rating

from .parameter_learning import (
    # Data models
    ReviewRecord,
    OptimizationResult,
    UserLearningProfile,
    # Loss functions
    LossFunction,
    # Core learner
    FSRSParameterLearner,
    # Adaptive wrapper
    AdaptiveFSRS,
)

__all__ = [
    # Core algorithm
    "FSRSAlgorithm",
    "FSRSCard",
    "Rating",
    # Parameter learning
    "ReviewRecord",
    "OptimizationResult",
    "UserLearningProfile",
    "LossFunction",
    "FSRSParameterLearner",
    "AdaptiveFSRS",
]
