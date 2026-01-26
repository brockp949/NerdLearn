"""
Half-Life Regression (HLR) Model for Reward Shaping

Provides the reward proxy for Curriculum Reinforcement Learning.
Optimizes for long-term memory retention rather than immediate correctness.

Key Formulas:
- Recall probability: p = 2^(-Δ/h) where Δ=elapsed time, h=half-life
- Half-life update: h_new = h_prev * 2^(θ·x) where x=features, θ=weights
- Memory strength: S = log_2(h)
- Reward: r_t = Σ_k(S_t^k - S_{t-1}^k)

The reward function captures the "whack-a-mole" dynamics of spaced repetition:
- Active practice increases memory strength (positive reward)
- Neglected concepts decay (negative reward through missed opportunities)

References:
- Settles & Meeder (2016): A Trainable Spaced Repetition Model for Language Learning
- Duolingo Research: Half-Life Regression
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import math
import numpy as np
from enum import Enum


class RewardType(str, Enum):
    """Types of reward formulations"""
    DELTA_STRENGTH = "delta_strength"       # Change in total memory strength
    DELTA_RETENTION = "delta_retention"     # Change in sum of recall probabilities
    RETENTION_WEIGHTED = "retention_weighted"  # Weighted by concept importance
    BINARY_CORRECT = "binary_correct"       # Simple correctness (baseline)


@dataclass
class HLRConfig:
    """Configuration for Half-Life Regression model"""

    # Initial half-life parameters
    initial_half_life_hours: float = 24.0    # 1 day initial half-life
    min_half_life_hours: float = 1.0         # Minimum 1 hour
    max_half_life_hours: float = 8760.0      # Maximum 1 year (365 days)

    # Half-life update parameters (θ vector)
    # h_new = h_prev * 2^(θ·x)
    theta_correct: float = 1.0               # Weight for correct response
    theta_incorrect: float = -0.5            # Weight for incorrect response
    theta_spacing: float = 0.1               # Weight for spacing benefit
    theta_difficulty: float = -0.2           # Weight for item difficulty
    theta_streak: float = 0.05               # Weight for consecutive correct

    # Recall probability threshold
    target_retention: float = 0.9            # Target 90% retention
    review_threshold: float = 0.85           # Review when retention drops below

    # Reward calculation parameters
    reward_type: RewardType = RewardType.DELTA_STRENGTH
    decay_penalty_weight: float = 0.1        # Weight for decay penalty in reward
    learning_bonus_weight: float = 1.0       # Weight for learning bonus

    # Normalization
    normalize_reward: bool = True
    max_reward_magnitude: float = 10.0

    def to_dict(self) -> Dict:
        return {
            "initial_half_life_hours": self.initial_half_life_hours,
            "min_half_life_hours": self.min_half_life_hours,
            "max_half_life_hours": self.max_half_life_hours,
            "theta_correct": self.theta_correct,
            "theta_incorrect": self.theta_incorrect,
            "theta_spacing": self.theta_spacing,
            "theta_difficulty": self.theta_difficulty,
            "theta_streak": self.theta_streak,
            "target_retention": self.target_retention,
            "review_threshold": self.review_threshold,
            "reward_type": self.reward_type.value,
            "decay_penalty_weight": self.decay_penalty_weight,
            "learning_bonus_weight": self.learning_bonus_weight,
            "normalize_reward": self.normalize_reward,
            "max_reward_magnitude": self.max_reward_magnitude,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "HLRConfig":
        d = d.copy()
        if "reward_type" in d:
            d["reward_type"] = RewardType(d["reward_type"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ConceptMemory:
    """
    Memory state for a single concept.

    Tracks the half-life (memory strength) and provides
    recall probability calculations.
    """
    concept_id: str
    half_life_hours: float = 24.0            # Current half-life
    last_review_at: Optional[datetime] = None
    review_count: int = 0
    correct_count: int = 0
    consecutive_correct: int = 0
    difficulty: float = 0.5                   # Item difficulty [0, 1]
    importance: float = 1.0                   # Concept importance weight

    def get_elapsed_hours(self, current_time: Optional[datetime] = None) -> float:
        """Get hours since last review"""
        if self.last_review_at is None:
            return 0.0
        current_time = current_time or datetime.now()
        elapsed = (current_time - self.last_review_at).total_seconds()
        return elapsed / 3600  # Convert to hours

    def get_recall_probability(self, current_time: Optional[datetime] = None) -> float:
        """
        Calculate recall probability using HLR formula.

        p = 2^(-Δ/h)

        Args:
            current_time: Time to calculate probability for

        Returns:
            Recall probability in [0, 1]
        """
        elapsed_hours = self.get_elapsed_hours(current_time)
        if elapsed_hours == 0 or self.half_life_hours <= 0:
            return 1.0  # Just reviewed or new item

        # HLR formula
        p = math.pow(2, -elapsed_hours / self.half_life_hours)
        return max(0.0, min(1.0, p))

    def get_memory_strength(self) -> float:
        """
        Get memory strength as log_2(half_life).

        Higher half-life = stronger memory = higher strength.
        This is the quantity we want to maximize.

        Returns:
            Memory strength (log scale)
        """
        return math.log2(max(1.0, self.half_life_hours))

    def get_optimal_review_time(self, target_retention: float = 0.9) -> Optional[datetime]:
        """
        Calculate optimal next review time to maintain target retention.

        Derived from: p = 2^(-Δ/h)
        => Δ = -h * log_2(p)

        Args:
            target_retention: Desired recall probability at review time

        Returns:
            Optimal review datetime
        """
        if self.last_review_at is None:
            return None

        # Solve for Δ
        optimal_hours = -self.half_life_hours * math.log2(target_retention)

        return self.last_review_at + timedelta(hours=optimal_hours)

    def to_dict(self) -> Dict:
        return {
            "concept_id": self.concept_id,
            "half_life_hours": self.half_life_hours,
            "last_review_at": self.last_review_at.isoformat() if self.last_review_at else None,
            "review_count": self.review_count,
            "correct_count": self.correct_count,
            "consecutive_correct": self.consecutive_correct,
            "difficulty": self.difficulty,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ConceptMemory":
        d = d.copy()
        if d.get("last_review_at"):
            d["last_review_at"] = datetime.fromisoformat(d["last_review_at"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class HLRModel:
    """
    Half-Life Regression Model.

    Provides:
    1. Recall probability estimation
    2. Half-life updates after reviews
    3. Optimal review scheduling
    4. Memory strength tracking for rewards
    """

    def __init__(self, config: Optional[HLRConfig] = None):
        """
        Initialize HLR model.

        Args:
            config: Optional configuration (uses defaults if not provided)
        """
        self.config = config or HLRConfig()
        self.concept_memories: Dict[str, ConceptMemory] = {}

    def get_or_create_memory(
        self,
        concept_id: str,
        difficulty: float = 0.5,
        importance: float = 1.0
    ) -> ConceptMemory:
        """
        Get existing memory or create new one for a concept.

        Args:
            concept_id: Concept identifier
            difficulty: Item difficulty [0, 1]
            importance: Concept importance weight

        Returns:
            ConceptMemory instance
        """
        if concept_id not in self.concept_memories:
            self.concept_memories[concept_id] = ConceptMemory(
                concept_id=concept_id,
                half_life_hours=self.config.initial_half_life_hours,
                difficulty=difficulty,
                importance=importance,
            )
        return self.concept_memories[concept_id]

    def _compute_feature_vector(
        self,
        memory: ConceptMemory,
        correct: bool,
        elapsed_hours: float
    ) -> np.ndarray:
        """
        Compute feature vector for half-life update.

        Features:
        - x[0]: Correct response indicator
        - x[1]: Incorrect response indicator
        - x[2]: Spacing benefit (normalized elapsed time)
        - x[3]: Item difficulty
        - x[4]: Consecutive correct streak

        Args:
            memory: Concept memory state
            correct: Whether response was correct
            elapsed_hours: Hours since last review

        Returns:
            Feature vector
        """
        # Spacing benefit: normalized by current half-life
        # More benefit for recalling after longer delay
        if memory.half_life_hours > 0:
            spacing_benefit = min(elapsed_hours / memory.half_life_hours, 3.0)
        else:
            spacing_benefit = 0.0

        features = np.array([
            1.0 if correct else 0.0,                    # x[0]: correct
            1.0 if not correct else 0.0,                # x[1]: incorrect
            spacing_benefit,                             # x[2]: spacing
            memory.difficulty,                           # x[3]: difficulty
            min(memory.consecutive_correct / 5.0, 1.0), # x[4]: streak (normalized)
        ])

        return features

    def _compute_theta_dot_x(self, features: np.ndarray) -> float:
        """
        Compute dot product of theta (weights) and features.

        θ·x = θ_correct*x[0] + θ_incorrect*x[1] + θ_spacing*x[2] +
              θ_difficulty*x[3] + θ_streak*x[4]

        Args:
            features: Feature vector

        Returns:
            Scalar weight for half-life update
        """
        theta = np.array([
            self.config.theta_correct,
            self.config.theta_incorrect,
            self.config.theta_spacing,
            self.config.theta_difficulty,
            self.config.theta_streak,
        ])

        return float(np.dot(theta, features))

    def update_after_review(
        self,
        concept_id: str,
        correct: bool,
        review_time: Optional[datetime] = None,
        difficulty: Optional[float] = None
    ) -> Tuple[ConceptMemory, Dict]:
        """
        Update memory after a review.

        Updates half-life using: h_new = h_prev * 2^(θ·x)

        Args:
            concept_id: Concept identifier
            correct: Whether response was correct
            review_time: Time of review
            difficulty: Optional difficulty override

        Returns:
            (updated_memory, update_details)
        """
        review_time = review_time or datetime.now()
        memory = self.get_or_create_memory(concept_id)

        if difficulty is not None:
            memory.difficulty = difficulty

        # Get elapsed time
        elapsed_hours = memory.get_elapsed_hours(review_time)

        # Store old values for reward calculation
        old_half_life = memory.half_life_hours
        old_strength = memory.get_memory_strength()
        old_retention = memory.get_recall_probability(review_time)

        # Compute features
        features = self._compute_feature_vector(memory, correct, elapsed_hours)

        # Compute new half-life: h_new = h_prev * 2^(θ·x)
        theta_dot_x = self._compute_theta_dot_x(features)
        new_half_life = old_half_life * math.pow(2, theta_dot_x)

        # Clamp to valid range
        new_half_life = max(
            self.config.min_half_life_hours,
            min(self.config.max_half_life_hours, new_half_life)
        )

        # Update memory state
        memory.half_life_hours = new_half_life
        memory.last_review_at = review_time
        memory.review_count += 1

        if correct:
            memory.correct_count += 1
            memory.consecutive_correct += 1
        else:
            memory.consecutive_correct = 0

        # Calculate new values
        new_strength = memory.get_memory_strength()

        # Compile details
        details = {
            "concept_id": concept_id,
            "correct": correct,
            "review_time": review_time.isoformat(),
            "elapsed_hours": elapsed_hours,
            "old_half_life_hours": old_half_life,
            "new_half_life_hours": new_half_life,
            "half_life_change": new_half_life - old_half_life,
            "old_strength": old_strength,
            "new_strength": new_strength,
            "strength_change": new_strength - old_strength,
            "old_retention": old_retention,
            "features": features.tolist(),
            "theta_dot_x": theta_dot_x,
        }

        return memory, details

    def predict_recall(
        self,
        concept_id: str,
        prediction_time: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Predict recall probability for a concept.

        Args:
            concept_id: Concept identifier
            prediction_time: Time to predict for

        Returns:
            Dictionary with prediction details
        """
        prediction_time = prediction_time or datetime.now()

        if concept_id not in self.concept_memories:
            return {
                "concept_id": concept_id,
                "recall_probability": 0.0,
                "memory_strength": 0.0,
                "half_life_hours": 0.0,
                "elapsed_hours": float('inf'),
                "is_new": True,
            }

        memory = self.concept_memories[concept_id]
        elapsed = memory.get_elapsed_hours(prediction_time)
        recall_prob = memory.get_recall_probability(prediction_time)

        return {
            "concept_id": concept_id,
            "recall_probability": recall_prob,
            "memory_strength": memory.get_memory_strength(),
            "half_life_hours": memory.half_life_hours,
            "elapsed_hours": elapsed,
            "optimal_review_time": memory.get_optimal_review_time(
                self.config.target_retention
            ).isoformat() if memory.last_review_at else None,
            "is_new": False,
        }

    def get_total_memory_strength(
        self,
        concept_ids: Optional[List[str]] = None
    ) -> float:
        """
        Get total memory strength across concepts.

        S_total = Σ_k log_2(h_k)

        Args:
            concept_ids: Optional list to filter (all if None)

        Returns:
            Total memory strength
        """
        if concept_ids is None:
            memories = self.concept_memories.values()
        else:
            memories = [
                self.concept_memories.get(cid)
                for cid in concept_ids
                if cid in self.concept_memories
            ]

        return sum(m.get_memory_strength() for m in memories if m)

    def get_total_retention(
        self,
        concept_ids: Optional[List[str]] = None,
        current_time: Optional[datetime] = None
    ) -> float:
        """
        Get sum of recall probabilities across concepts.

        P_total = Σ_k p_k(t)

        Args:
            concept_ids: Optional list to filter (all if None)
            current_time: Time to calculate retention for

        Returns:
            Sum of recall probabilities
        """
        current_time = current_time or datetime.now()

        if concept_ids is None:
            memories = self.concept_memories.values()
        else:
            memories = [
                self.concept_memories.get(cid)
                for cid in concept_ids
                if cid in self.concept_memories
            ]

        return sum(m.get_recall_probability(current_time) for m in memories if m)

    def serialize(self) -> Dict:
        """Serialize model state for persistence"""
        return {
            "config": self.config.to_dict(),
            "concept_memories": {
                k: v.to_dict() for k, v in self.concept_memories.items()
            },
        }

    @classmethod
    def deserialize(cls, data: Dict) -> "HLRModel":
        """Deserialize model from saved state"""
        config = HLRConfig.from_dict(data["config"])
        model = cls(config)
        model.concept_memories = {
            k: ConceptMemory.from_dict(v)
            for k, v in data.get("concept_memories", {}).items()
        }
        return model


class RewardCalculator:
    """
    Calculates rewards for Curriculum RL based on memory strength changes.

    The reward captures the "whack-a-mole" dynamics:
    - Positive reward for increasing memory strength (successful reviews)
    - Negative reward for allowing decay (neglected concepts)

    This encourages the RL agent to:
    1. Schedule reviews just before forgetting (optimal spacing)
    2. Interleave concepts to maintain all memories
    3. Prioritize decaying concepts
    """

    def __init__(
        self,
        config: Optional[HLRConfig] = None,
        hlr_model: Optional[HLRModel] = None
    ):
        """
        Initialize reward calculator.

        Args:
            config: HLR configuration
            hlr_model: Optional shared HLR model instance
        """
        self.config = config or HLRConfig()
        self.hlr_model = hlr_model or HLRModel(self.config)

        # Cache for previous state (for delta calculations)
        self._prev_strengths: Dict[str, float] = {}
        self._prev_retentions: Dict[str, float] = {}
        self._prev_time: Optional[datetime] = None

    def initialize_state(
        self,
        concept_ids: List[str],
        current_time: Optional[datetime] = None
    ):
        """
        Initialize reward state for a learning session.

        Args:
            concept_ids: List of concepts in the curriculum
            current_time: Session start time
        """
        current_time = current_time or datetime.now()
        self._prev_time = current_time

        # Initialize or get memories for all concepts
        for cid in concept_ids:
            memory = self.hlr_model.get_or_create_memory(cid)
            self._prev_strengths[cid] = memory.get_memory_strength()
            self._prev_retentions[cid] = memory.get_recall_probability(current_time)

    def calculate_reward(
        self,
        concept_id: str,
        correct: bool,
        current_time: Optional[datetime] = None,
        all_concept_ids: Optional[List[str]] = None
    ) -> Tuple[float, Dict]:
        """
        Calculate reward after an action (reviewing a concept).

        The reward is the change in total system memory strength.

        Args:
            concept_id: Concept that was reviewed
            correct: Whether response was correct
            current_time: Time of review
            all_concept_ids: All concepts to track (for decay penalty)

        Returns:
            (reward, reward_details)
        """
        current_time = current_time or datetime.now()

        # Update HLR model
        memory, update_details = self.hlr_model.update_after_review(
            concept_id, correct, current_time
        )

        # Calculate reward based on configured type
        if self.config.reward_type == RewardType.DELTA_STRENGTH:
            reward, details = self._calc_delta_strength_reward(
                concept_id, memory, all_concept_ids, current_time
            )
        elif self.config.reward_type == RewardType.DELTA_RETENTION:
            reward, details = self._calc_delta_retention_reward(
                concept_id, memory, all_concept_ids, current_time
            )
        elif self.config.reward_type == RewardType.RETENTION_WEIGHTED:
            reward, details = self._calc_weighted_retention_reward(
                concept_id, memory, all_concept_ids, current_time
            )
        else:  # BINARY_CORRECT
            reward = 1.0 if correct else 0.0
            details = {"type": "binary_correct", "correct": correct}

        # Normalize reward if configured
        if self.config.normalize_reward:
            reward = max(
                -self.config.max_reward_magnitude,
                min(self.config.max_reward_magnitude, reward)
            )

        # Update cached state
        self._update_cached_state(all_concept_ids, current_time)

        # Compile full details
        full_details = {
            "reward": reward,
            "reward_type": self.config.reward_type.value,
            "concept_id": concept_id,
            "correct": correct,
            "hlr_update": update_details,
            "reward_components": details,
        }

        return reward, full_details

    def _calc_delta_strength_reward(
        self,
        concept_id: str,
        updated_memory: ConceptMemory,
        all_concept_ids: Optional[List[str]],
        current_time: datetime
    ) -> Tuple[float, Dict]:
        """
        Calculate reward as change in total memory strength.

        r_t = Σ_k(S_t^k - S_{t-1}^k)

        Args:
            concept_id: Concept that was reviewed
            updated_memory: Updated memory state
            all_concept_ids: All concepts to track
            current_time: Current time

        Returns:
            (reward, details)
        """
        # Active gain from reviewed concept
        old_strength = self._prev_strengths.get(concept_id, 0.0)
        new_strength = updated_memory.get_memory_strength()
        active_gain = new_strength - old_strength

        # Passive decay from other concepts (penalty for time passing)
        passive_decay = 0.0
        if all_concept_ids:
            for cid in all_concept_ids:
                if cid != concept_id and cid in self.hlr_model.concept_memories:
                    memory = self.hlr_model.concept_memories[cid]
                    # Strength doesn't decay, but retention does
                    # Penalize for low retention on other concepts
                    retention = memory.get_recall_probability(current_time)
                    if retention < self.config.review_threshold:
                        # Penalty proportional to how much below threshold
                        penalty = (self.config.review_threshold - retention) * memory.importance
                        passive_decay += penalty

        # Combined reward
        reward = (
            self.config.learning_bonus_weight * active_gain -
            self.config.decay_penalty_weight * passive_decay
        )

        details = {
            "type": "delta_strength",
            "active_gain": active_gain,
            "passive_decay_penalty": passive_decay,
            "old_strength": old_strength,
            "new_strength": new_strength,
        }

        return reward, details

    def _calc_delta_retention_reward(
        self,
        concept_id: str,
        updated_memory: ConceptMemory,
        all_concept_ids: Optional[List[str]],
        current_time: datetime
    ) -> Tuple[float, Dict]:
        """
        Calculate reward as change in sum of recall probabilities.

        r_t = Σ_k(p_t^k - p_{t-1}^k)

        Args:
            concept_id: Concept that was reviewed
            updated_memory: Updated memory state
            all_concept_ids: All concepts to track
            current_time: Current time

        Returns:
            (reward, details)
        """
        total_delta = 0.0

        # Reviewed concept goes to full retention
        old_retention = self._prev_retentions.get(concept_id, 0.0)
        new_retention = 1.0  # Just reviewed = full retention
        total_delta += (new_retention - old_retention)

        # Other concepts decay
        if all_concept_ids:
            for cid in all_concept_ids:
                if cid != concept_id and cid in self.hlr_model.concept_memories:
                    memory = self.hlr_model.concept_memories[cid]
                    prev_ret = self._prev_retentions.get(cid, 1.0)
                    curr_ret = memory.get_recall_probability(current_time)
                    total_delta += (curr_ret - prev_ret)

        details = {
            "type": "delta_retention",
            "reviewed_delta": new_retention - old_retention,
            "total_delta": total_delta,
        }

        return total_delta, details

    def _calc_weighted_retention_reward(
        self,
        concept_id: str,
        updated_memory: ConceptMemory,
        all_concept_ids: Optional[List[str]],
        current_time: datetime
    ) -> Tuple[float, Dict]:
        """
        Calculate reward weighted by concept importance.

        r_t = Σ_k w_k * (p_t^k - p_{t-1}^k)

        Args:
            concept_id: Concept that was reviewed
            updated_memory: Updated memory state
            all_concept_ids: All concepts to track
            current_time: Current time

        Returns:
            (reward, details)
        """
        total_weighted_delta = 0.0

        # Reviewed concept
        old_retention = self._prev_retentions.get(concept_id, 0.0)
        new_retention = 1.0
        total_weighted_delta += updated_memory.importance * (new_retention - old_retention)

        # Other concepts
        if all_concept_ids:
            for cid in all_concept_ids:
                if cid != concept_id and cid in self.hlr_model.concept_memories:
                    memory = self.hlr_model.concept_memories[cid]
                    prev_ret = self._prev_retentions.get(cid, 1.0)
                    curr_ret = memory.get_recall_probability(current_time)
                    total_weighted_delta += memory.importance * (curr_ret - prev_ret)

        details = {
            "type": "retention_weighted",
            "total_weighted_delta": total_weighted_delta,
        }

        return total_weighted_delta, details

    def _update_cached_state(
        self,
        concept_ids: Optional[List[str]],
        current_time: datetime
    ):
        """Update cached state for next reward calculation"""
        self._prev_time = current_time

        concepts_to_update = concept_ids or list(self.hlr_model.concept_memories.keys())

        for cid in concepts_to_update:
            if cid in self.hlr_model.concept_memories:
                memory = self.hlr_model.concept_memories[cid]
                self._prev_strengths[cid] = memory.get_memory_strength()
                self._prev_retentions[cid] = memory.get_recall_probability(current_time)

    def get_urgency_scores(
        self,
        concept_ids: List[str],
        current_time: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Get urgency scores for each concept.

        Higher urgency = more beneficial to review now.
        Based on how much retention has dropped and importance.

        Args:
            concept_ids: Concepts to score
            current_time: Current time

        Returns:
            Dictionary mapping concept_id to urgency score
        """
        current_time = current_time or datetime.now()
        urgencies = {}

        for cid in concept_ids:
            if cid in self.hlr_model.concept_memories:
                memory = self.hlr_model.concept_memories[cid]
                retention = memory.get_recall_probability(current_time)

                # Urgency based on:
                # 1. How much below target retention
                # 2. Concept importance
                # 3. Memory strength (prefer consolidating weak memories)

                if retention < self.config.target_retention:
                    retention_deficit = self.config.target_retention - retention
                    strength_penalty = 1.0 / (1.0 + memory.get_memory_strength())
                    urgency = retention_deficit * memory.importance * (1 + strength_penalty)
                else:
                    urgency = 0.0

                urgencies[cid] = urgency
            else:
                # New concept - moderate urgency to introduce
                urgencies[cid] = 0.3

        return urgencies

    def get_state_snapshot(
        self,
        concept_ids: List[str],
        current_time: Optional[datetime] = None
    ) -> Dict:
        """
        Get snapshot of current memory state for logging.

        Args:
            concept_ids: Concepts to include
            current_time: Current time

        Returns:
            State snapshot dictionary
        """
        current_time = current_time or datetime.now()

        return {
            "timestamp": current_time.isoformat(),
            "total_strength": self.hlr_model.get_total_memory_strength(concept_ids),
            "total_retention": self.hlr_model.get_total_retention(concept_ids, current_time),
            "concept_states": {
                cid: self.hlr_model.predict_recall(cid, current_time)
                for cid in concept_ids
                if cid in self.hlr_model.concept_memories
            },
        }
