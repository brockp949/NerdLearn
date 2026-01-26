"""
Student Simulator for Curriculum RL Evaluation

Simulates realistic student learning dynamics:
1. Hidden mastery state (ground truth)
2. Learning: Mastery increases with practice
3. Forgetting: Mastery decays over time
4. Slip/Guess: Noisy observations

This allows safe evaluation of RL policies without
experimenting on real students.

Based on cognitive science models:
- Ebbinghaus forgetting curve
- Spacing effect
- Slip and guess probabilities from BKT
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class StudentSimulatorConfig:
    """Configuration for student simulator"""

    # Number of concepts
    num_concepts: int = 20

    # Learning parameters (per practice opportunity)
    learning_rate_mean: float = 0.15
    learning_rate_std: float = 0.05
    max_mastery: float = 1.0

    # Forgetting parameters
    initial_half_life_hours: float = 24.0    # 1 day
    max_half_life_hours: float = 720.0       # 30 days
    half_life_growth_factor: float = 1.5     # Growth on successful recall
    half_life_decay_factor: float = 0.7      # Decay on failed recall

    # Slip/Guess probabilities
    slip_probability: float = 0.1            # P(incorrect | know)
    guess_probability: float = 0.2           # P(correct | don't know)

    # Prior knowledge
    initial_mastery_mean: float = 0.1
    initial_mastery_std: float = 0.05

    # Heterogeneity (student-level variation)
    heterogeneous: bool = True
    skill_variance: float = 0.2              # Variance in learning ability

    # Spacing effect
    enable_spacing_effect: bool = True
    spacing_bonus_factor: float = 0.1        # Extra learning for spaced practice

    # Cognitive load
    enable_cognitive_load: bool = True
    max_daily_learning: float = 0.5          # Max total learning per day

    # Random seed
    seed: Optional[int] = None

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict) -> "StudentSimulatorConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SimulatedInteraction:
    """Record of a simulated interaction"""
    concept_id: str
    correct: bool
    timestamp: datetime
    true_mastery_before: float
    true_mastery_after: float
    recall_probability: float
    elapsed_hours: float


class StudentSimulator:
    """
    Simulates a student learning through a curriculum.

    Maintains hidden ground-truth state and generates
    noisy observations that mimic real student behavior.

    Usage:
        simulator = StudentSimulator(config)
        simulator.reset()

        # Policy selects concept
        concept_id = "concept_5"

        # Simulator generates response
        correct, info = simulator.respond(concept_id, current_time)

        # Optionally advance time
        simulator.advance_time(hours=24)
    """

    def __init__(self, config: StudentSimulatorConfig):
        """
        Initialize simulator.

        Args:
            config: Simulator configuration
        """
        self.config = config

        if config.seed is not None:
            np.random.seed(config.seed)

        # Student-level parameters (heterogeneity)
        if config.heterogeneous:
            self.learning_rate_multiplier = np.random.lognormal(0, config.skill_variance)
            self.forgetting_rate_multiplier = np.random.lognormal(0, config.skill_variance)
        else:
            self.learning_rate_multiplier = 1.0
            self.forgetting_rate_multiplier = 1.0

        # State variables (initialized in reset())
        self.concept_ids: List[str] = []
        self.true_mastery: Dict[str, float] = {}
        self.half_lives: Dict[str, float] = {}
        self.last_practice: Dict[str, datetime] = {}
        self.practice_counts: Dict[str, int] = {}

        self.current_time: datetime = datetime.now()
        self.daily_learning: float = 0.0
        self.last_learning_day: Optional[datetime] = None

        # History
        self.interaction_history: List[SimulatedInteraction] = []

    def reset(
        self,
        concept_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None
    ):
        """
        Reset simulator to initial state.

        Args:
            concept_ids: List of concept IDs (auto-generates if None)
            start_time: Starting timestamp
        """
        # Set concept IDs
        if concept_ids is not None:
            self.concept_ids = concept_ids
        else:
            self.concept_ids = [f"concept_{i}" for i in range(self.config.num_concepts)]

        # Initialize state for each concept
        self.true_mastery = {}
        self.half_lives = {}
        self.last_practice = {}
        self.practice_counts = {}

        for cid in self.concept_ids:
            # Initial mastery with some variation
            initial = np.random.normal(
                self.config.initial_mastery_mean,
                self.config.initial_mastery_std
            )
            self.true_mastery[cid] = np.clip(initial, 0.0, 0.3)

            self.half_lives[cid] = self.config.initial_half_life_hours
            self.practice_counts[cid] = 0

        # Time
        self.current_time = start_time or datetime.now()
        self.daily_learning = 0.0
        self.last_learning_day = self.current_time.date()

        # Clear history
        self.interaction_history = []

        logger.debug(f"Reset simulator with {len(self.concept_ids)} concepts")

    def _apply_forgetting(self, concept_id: str) -> float:
        """
        Apply forgetting to a concept based on time elapsed.

        Uses exponential decay: m(t) = m(0) * 2^(-t/h)

        Args:
            concept_id: Concept to apply forgetting to

        Returns:
            New mastery after forgetting
        """
        if concept_id not in self.last_practice:
            return self.true_mastery[concept_id]

        elapsed = (self.current_time - self.last_practice[concept_id]).total_seconds()
        elapsed_hours = elapsed / 3600

        if elapsed_hours <= 0:
            return self.true_mastery[concept_id]

        half_life = self.half_lives[concept_id]

        # Apply decay
        decay_factor = np.power(2, -elapsed_hours * self.forgetting_rate_multiplier / half_life)
        new_mastery = self.true_mastery[concept_id] * decay_factor

        return max(0.0, new_mastery)

    def _compute_recall_probability(self, concept_id: str) -> float:
        """
        Compute probability of correct recall.

        P(correct) = mastery * (1 - slip) + (1 - mastery) * guess

        Args:
            concept_id: Concept to compute for

        Returns:
            Recall probability
        """
        # Apply forgetting first
        current_mastery = self._apply_forgetting(concept_id)

        # BKT-style observation probability
        p_correct = (
            current_mastery * (1 - self.config.slip_probability) +
            (1 - current_mastery) * self.config.guess_probability
        )

        return p_correct

    def _apply_learning(
        self,
        concept_id: str,
        correct: bool,
        elapsed_hours: float
    ) -> float:
        """
        Apply learning effect after practice.

        Args:
            concept_id: Concept practiced
            correct: Whether response was correct
            elapsed_hours: Time since last practice

        Returns:
            Learning gain
        """
        # Check daily learning limit
        current_day = self.current_time.date()
        if self.last_learning_day != current_day:
            self.daily_learning = 0.0
            self.last_learning_day = current_day

        if self.config.enable_cognitive_load:
            remaining_capacity = self.config.max_daily_learning - self.daily_learning
            if remaining_capacity <= 0:
                return 0.0  # Saturated for today

        # Base learning rate
        lr = self.config.learning_rate_mean * self.learning_rate_multiplier

        # Add noise
        lr += np.random.normal(0, self.config.learning_rate_std)
        lr = max(0.01, lr)

        # Spacing effect bonus
        if self.config.enable_spacing_effect and elapsed_hours > 0:
            # Optimal spacing around 1 day
            spacing_bonus = self.config.spacing_bonus_factor * min(elapsed_hours / 24, 2.0)
            lr *= (1 + spacing_bonus)

        # Apply learning based on outcome
        old_mastery = self.true_mastery[concept_id]

        if correct:
            # Successful practice: increase mastery
            gain = lr * (self.config.max_mastery - old_mastery)
            new_mastery = old_mastery + gain

            # Update half-life (spacing effect on memory)
            self.half_lives[concept_id] = min(
                self.config.max_half_life_hours,
                self.half_lives[concept_id] * self.config.half_life_growth_factor
            )
        else:
            # Failed practice: slight decrease or learning from error
            gain = lr * 0.3 * (1 - old_mastery)  # Still learn a bit from errors
            new_mastery = old_mastery + gain * 0.5

            # Decrease half-life
            self.half_lives[concept_id] = max(
                1.0,
                self.half_lives[concept_id] * self.config.half_life_decay_factor
            )

        # Clamp mastery
        new_mastery = np.clip(new_mastery, 0.0, self.config.max_mastery)
        actual_gain = new_mastery - old_mastery

        # Update daily learning
        self.daily_learning += abs(actual_gain)

        # Limit by remaining capacity
        if self.config.enable_cognitive_load:
            if actual_gain > remaining_capacity:
                new_mastery = old_mastery + remaining_capacity * np.sign(actual_gain)
                actual_gain = remaining_capacity * np.sign(actual_gain)

        self.true_mastery[concept_id] = new_mastery
        return actual_gain

    def respond(
        self,
        concept_id: str,
        timestamp: Optional[datetime] = None
    ) -> Tuple[bool, Dict]:
        """
        Generate simulated response to a concept.

        Args:
            concept_id: Concept being practiced
            timestamp: Time of practice (uses current_time if None)

        Returns:
            (correct, response_info)
        """
        if concept_id not in self.concept_ids:
            raise ValueError(f"Unknown concept: {concept_id}")

        if timestamp is not None:
            self.current_time = timestamp

        # Get elapsed time
        if concept_id in self.last_practice:
            elapsed = (self.current_time - self.last_practice[concept_id]).total_seconds()
            elapsed_hours = elapsed / 3600
        else:
            elapsed_hours = 0.0

        # Store mastery before
        mastery_before = self._apply_forgetting(concept_id)
        self.true_mastery[concept_id] = mastery_before  # Update with decayed value

        # Compute recall probability
        p_correct = self._compute_recall_probability(concept_id)

        # Generate response
        correct = np.random.random() < p_correct

        # Apply learning
        learning_gain = self._apply_learning(concept_id, correct, elapsed_hours)

        # Update tracking
        self.last_practice[concept_id] = self.current_time
        self.practice_counts[concept_id] += 1

        # Record interaction
        interaction = SimulatedInteraction(
            concept_id=concept_id,
            correct=correct,
            timestamp=self.current_time,
            true_mastery_before=mastery_before,
            true_mastery_after=self.true_mastery[concept_id],
            recall_probability=p_correct,
            elapsed_hours=elapsed_hours,
        )
        self.interaction_history.append(interaction)

        # Response info
        info = {
            "concept_id": concept_id,
            "correct": correct,
            "p_correct": p_correct,
            "mastery_before": mastery_before,
            "mastery_after": self.true_mastery[concept_id],
            "learning_gain": learning_gain,
            "elapsed_hours": elapsed_hours,
            "half_life_hours": self.half_lives[concept_id],
            "practice_count": self.practice_counts[concept_id],
        }

        return correct, info

    def advance_time(self, hours: float = 0, days: float = 0, minutes: float = 0):
        """
        Advance simulation time.

        Args:
            hours: Hours to advance
            days: Days to advance
            minutes: Minutes to advance
        """
        total_hours = hours + days * 24 + minutes / 60
        self.current_time += timedelta(hours=total_hours)

    def get_true_mastery_vector(self) -> np.ndarray:
        """Get ground-truth mastery as array"""
        return np.array([self.true_mastery[c] for c in self.concept_ids])

    def get_retention_after_delay(self, delay_days: float = 30) -> Dict[str, float]:
        """
        Calculate retention probability after a delay.

        Args:
            delay_days: Delay in days

        Returns:
            Dict mapping concept_id to retention probability
        """
        delay_hours = delay_days * 24
        retentions = {}

        for cid in self.concept_ids:
            if cid in self.last_practice:
                # Time since last practice plus delay
                already_elapsed = (self.current_time - self.last_practice[cid]).total_seconds() / 3600
                total_elapsed = already_elapsed + delay_hours

                # Apply forgetting curve
                half_life = self.half_lives[cid]
                decay = np.power(2, -total_elapsed / half_life)
                retentions[cid] = self.true_mastery[cid] * decay
            else:
                retentions[cid] = self.true_mastery[cid]

        return retentions

    def compute_day30_retention(self) -> float:
        """
        Compute average retention probability at 30 days.

        This is the primary evaluation metric.
        """
        retentions = self.get_retention_after_delay(30)
        return float(np.mean(list(retentions.values())))

    def get_statistics(self) -> Dict:
        """Get simulator statistics"""
        retentions = self.get_retention_after_delay(30)

        return {
            "num_concepts": len(self.concept_ids),
            "num_interactions": len(self.interaction_history),
            "mean_mastery": float(np.mean(list(self.true_mastery.values()))),
            "std_mastery": float(np.std(list(self.true_mastery.values()))),
            "max_mastery": float(np.max(list(self.true_mastery.values()))),
            "min_mastery": float(np.min(list(self.true_mastery.values()))),
            "mean_half_life_hours": float(np.mean(list(self.half_lives.values()))),
            "day30_retention": float(np.mean(list(retentions.values()))),
            "accuracy": float(np.mean([i.correct for i in self.interaction_history])) if self.interaction_history else 0,
            "concepts_practiced": sum(1 for c in self.practice_counts.values() if c > 0),
        }

    def to_dict(self) -> Dict:
        """Serialize simulator state"""
        return {
            "config": self.config.to_dict(),
            "concept_ids": self.concept_ids,
            "true_mastery": self.true_mastery,
            "half_lives": self.half_lives,
            "last_practice": {k: v.isoformat() for k, v in self.last_practice.items()},
            "practice_counts": self.practice_counts,
            "current_time": self.current_time.isoformat(),
        }
