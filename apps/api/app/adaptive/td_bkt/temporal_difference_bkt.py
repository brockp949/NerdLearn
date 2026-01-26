"""
Temporal Difference Bayesian Knowledge Tracing (TD-BKT)

Extends standard BKT with temporal dynamics for spaced repetition optimization.
Implements a two-step filter: Prediction (time decay) and Correction (observation update).

Key Features:
1. Forgetting curve modeling via Half-Life decay: L_{t|t-1} = L_{t-1} * 2^(-τ/h)
2. Time-varying slip/guess probabilities based on recency
3. Belief state vector for POMDP integration: [mastery_vector, recency_vector]
4. Cognitive inflection point detection for adaptive learning rate

Mathematical Foundation:
- Prediction Step: P(L_t | no observation) = P(L_{t-1}) * decay_factor(time_elapsed, half_life)
- Correction Step: Bayesian update using decayed prior and observation likelihood

References:
- Integrating Temporal Information Into Knowledge Tracing (IEEE, 2018)
- TD-BKT for Curriculum Reinforcement Learning
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import math
import numpy as np
from enum import Enum


class DecayModel(str, Enum):
    """Memory decay models"""
    EXPONENTIAL = "exponential"  # Standard exponential decay
    HALF_LIFE = "half_life"      # Half-life regression style
    POWER_LAW = "power_law"      # Power law forgetting


@dataclass
class TDBKTConfig:
    """Configuration for Temporal Difference BKT"""

    # Standard BKT parameters
    p_l0: float = 0.1       # Prior probability of knowing (10%)
    p_t: float = 0.15       # Probability of learning per opportunity (15%)
    p_g: float = 0.2        # Probability of guessing correctly (20%)
    p_s: float = 0.1        # Probability of slip/error (10%)

    # Temporal decay parameters
    decay_model: DecayModel = DecayModel.HALF_LIFE
    initial_half_life_days: float = 1.0    # Initial memory half-life (1 day)
    max_half_life_days: float = 365.0      # Maximum half-life (1 year)
    half_life_growth_factor: float = 2.0   # How much half-life grows on success
    half_life_decay_factor: float = 0.5    # How much half-life shrinks on failure

    # Time-dependent slip/guess adjustments
    recency_slip_bonus: float = 0.05       # Reduced slip probability for recent items
    recency_guess_penalty: float = 0.05    # Reduced guess probability for old items
    recency_threshold_days: float = 7.0    # Threshold for "recent" items

    # Belief state normalization
    max_recency_days: float = 90.0         # Max days for recency normalization

    # Cognitive inflection point detection
    detect_inflection_points: bool = True
    inflection_threshold: float = 0.15     # Mastery change threshold for inflection

    # Mastery thresholds
    mastery_threshold: float = 0.85        # Consider mastered above this
    prerequisite_threshold: float = 0.85   # Prerequisite satisfaction threshold

    def to_dict(self) -> Dict:
        return {
            "p_l0": self.p_l0,
            "p_t": self.p_t,
            "p_g": self.p_g,
            "p_s": self.p_s,
            "decay_model": self.decay_model.value,
            "initial_half_life_days": self.initial_half_life_days,
            "max_half_life_days": self.max_half_life_days,
            "half_life_growth_factor": self.half_life_growth_factor,
            "half_life_decay_factor": self.half_life_decay_factor,
            "recency_slip_bonus": self.recency_slip_bonus,
            "recency_guess_penalty": self.recency_guess_penalty,
            "recency_threshold_days": self.recency_threshold_days,
            "max_recency_days": self.max_recency_days,
            "detect_inflection_points": self.detect_inflection_points,
            "inflection_threshold": self.inflection_threshold,
            "mastery_threshold": self.mastery_threshold,
            "prerequisite_threshold": self.prerequisite_threshold,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TDBKTConfig":
        d = d.copy()
        if "decay_model" in d:
            d["decay_model"] = DecayModel(d["decay_model"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ConceptState:
    """
    State tracking for a single concept.

    Tracks both mastery probability and memory strength (half-life).
    """
    concept_id: str
    mastery_probability: float = 0.1     # P(L) - probability of knowing
    half_life_days: float = 1.0          # Memory strength in days
    last_interaction_at: Optional[datetime] = None
    total_attempts: int = 0
    correct_attempts: int = 0
    consecutive_correct: int = 0
    consecutive_incorrect: int = 0

    # Cognitive inflection tracking
    is_at_inflection: bool = False
    mastery_velocity: float = 0.0        # Rate of mastery change

    def get_elapsed_days(self, current_time: Optional[datetime] = None) -> float:
        """Get days since last interaction"""
        if self.last_interaction_at is None:
            return 0.0
        current_time = current_time or datetime.now()
        elapsed = (current_time - self.last_interaction_at).total_seconds()
        return elapsed / (24 * 3600)  # Convert to days

    def get_retention_probability(self, current_time: Optional[datetime] = None) -> float:
        """
        Calculate current retention probability accounting for decay.

        Uses Half-Life Regression formula: p = 2^(-Δ/h)
        """
        elapsed_days = self.get_elapsed_days(current_time)
        if elapsed_days == 0 or self.half_life_days == 0:
            return self.mastery_probability

        # Apply decay to mastery
        decay_factor = math.pow(2, -elapsed_days / self.half_life_days)
        return self.mastery_probability * decay_factor

    def to_dict(self) -> Dict:
        return {
            "concept_id": self.concept_id,
            "mastery_probability": self.mastery_probability,
            "half_life_days": self.half_life_days,
            "last_interaction_at": self.last_interaction_at.isoformat() if self.last_interaction_at else None,
            "total_attempts": self.total_attempts,
            "correct_attempts": self.correct_attempts,
            "consecutive_correct": self.consecutive_correct,
            "consecutive_incorrect": self.consecutive_incorrect,
            "is_at_inflection": self.is_at_inflection,
            "mastery_velocity": self.mastery_velocity,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ConceptState":
        d = d.copy()
        if d.get("last_interaction_at"):
            d["last_interaction_at"] = datetime.fromisoformat(d["last_interaction_at"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class BeliefState:
    """
    Complete belief state for the POMDP.

    This is the input to the RL policy. Contains:
    - Mastery vector: [p̂(L_1), ..., p̂(L_K)] - probability of mastery for each concept
    - Recency vector: [Δ_1, ..., Δ_K] - normalized time since last interaction
    - Half-life vector: [h_1, ..., h_K] - memory strength for each concept

    Total dimension: 3K for K concepts
    """
    user_id: str
    concept_states: Dict[str, ConceptState] = field(default_factory=dict)
    last_updated: Optional[datetime] = None

    # Ordered list of concept IDs for consistent vector representation
    concept_order: List[str] = field(default_factory=list)

    def get_mastery_vector(self, current_time: Optional[datetime] = None) -> np.ndarray:
        """
        Get mastery probabilities as a vector.

        Returns decayed mastery accounting for time elapsed.
        """
        if not self.concept_order:
            return np.array([])

        current_time = current_time or datetime.now()
        masteries = []

        for concept_id in self.concept_order:
            if concept_id in self.concept_states:
                state = self.concept_states[concept_id]
                # Get retention probability (mastery with decay)
                mastery = state.get_retention_probability(current_time)
            else:
                mastery = 0.1  # Default prior
            masteries.append(mastery)

        return np.array(masteries, dtype=np.float32)

    def get_recency_vector(
        self,
        current_time: Optional[datetime] = None,
        max_days: float = 90.0
    ) -> np.ndarray:
        """
        Get normalized recency values as a vector.

        Recency is normalized to [0, 1] where:
        - 0 = just practiced
        - 1 = not practiced for max_days or more
        """
        if not self.concept_order:
            return np.array([])

        current_time = current_time or datetime.now()
        recencies = []

        for concept_id in self.concept_order:
            if concept_id in self.concept_states:
                state = self.concept_states[concept_id]
                elapsed = state.get_elapsed_days(current_time)
                # Normalize to [0, 1]
                normalized = min(elapsed / max_days, 1.0)
            else:
                normalized = 1.0  # Never practiced = max recency
            recencies.append(normalized)

        return np.array(recencies, dtype=np.float32)

    def get_half_life_vector(self, max_days: float = 365.0) -> np.ndarray:
        """
        Get normalized half-life values as a vector.

        Normalized to [0, 1] where higher = stronger memory.
        """
        if not self.concept_order:
            return np.array([])

        half_lives = []
        for concept_id in self.concept_order:
            if concept_id in self.concept_states:
                state = self.concept_states[concept_id]
                # Normalize using log scale
                normalized = math.log(state.half_life_days + 1) / math.log(max_days + 1)
            else:
                normalized = 0.0  # No memory
            half_lives.append(min(normalized, 1.0))

        return np.array(half_lives, dtype=np.float32)

    def to_vector(self, current_time: Optional[datetime] = None) -> np.ndarray:
        """
        Convert belief state to a single vector for RL policy input.

        Format: [mastery_vector, recency_vector, half_life_vector]
        Dimension: 3K for K concepts
        """
        current_time = current_time or datetime.now()

        mastery = self.get_mastery_vector(current_time)
        recency = self.get_recency_vector(current_time)
        half_life = self.get_half_life_vector()

        return np.concatenate([mastery, recency, half_life])

    def get_concept_mastery(self, concept_id: str) -> float:
        """Get current mastery for a specific concept"""
        if concept_id in self.concept_states:
            return self.concept_states[concept_id].get_retention_probability()
        return 0.1  # Default prior

    def is_concept_mastered(self, concept_id: str, threshold: float = 0.85) -> bool:
        """Check if a concept is mastered"""
        return self.get_concept_mastery(concept_id) >= threshold

    def get_weakest_concepts(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get the n concepts with lowest mastery"""
        current_time = datetime.now()
        masteries = [
            (cid, state.get_retention_probability(current_time))
            for cid, state in self.concept_states.items()
        ]
        masteries.sort(key=lambda x: x[1])
        return masteries[:n]

    def get_most_urgent_reviews(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Get concepts most urgently needing review.

        Urgency = mastery * (1 - retention_probability)
        High urgency = knew it well but decayed significantly
        """
        current_time = datetime.now()
        urgencies = []

        for cid, state in self.concept_states.items():
            if state.total_attempts > 0:
                raw_mastery = state.mastery_probability
                current_retention = state.get_retention_probability(current_time)
                # Urgency: how much have we forgotten from peak mastery?
                urgency = raw_mastery * (1 - current_retention / (raw_mastery + 1e-6))
                urgencies.append((cid, urgency))

        urgencies.sort(key=lambda x: x[1], reverse=True)
        return urgencies[:n]

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "concept_states": {
                k: v.to_dict() for k, v in self.concept_states.items()
            },
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "concept_order": self.concept_order,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "BeliefState":
        concept_states = {
            k: ConceptState.from_dict(v)
            for k, v in d.get("concept_states", {}).items()
        }
        last_updated = None
        if d.get("last_updated"):
            last_updated = datetime.fromisoformat(d["last_updated"])

        return cls(
            user_id=d["user_id"],
            concept_states=concept_states,
            last_updated=last_updated,
            concept_order=d.get("concept_order", []),
        )


class TemporalDifferenceBKT:
    """
    Temporal Difference Bayesian Knowledge Tracing.

    Extends BKT with:
    1. Time-based decay of mastery (forgetting curves)
    2. Dynamic half-life tracking for memory strength
    3. Time-varying slip/guess probabilities
    4. Belief state output for POMDP integration

    Usage:
        td_bkt = TemporalDifferenceBKT(config)

        # Initialize belief state for a user
        belief = td_bkt.create_belief_state(user_id, concept_ids)

        # Update after observation
        belief, details = td_bkt.update(belief, concept_id, correct, timestamp)

        # Get belief vector for RL policy
        state_vector = belief.to_vector()
    """

    def __init__(self, config: Optional[TDBKTConfig] = None):
        """
        Initialize TD-BKT with configuration.

        Args:
            config: Optional configuration (uses defaults if not provided)
        """
        self.config = config or TDBKTConfig()

    def create_belief_state(
        self,
        user_id: str,
        concept_ids: List[str],
        initial_masteries: Optional[Dict[str, float]] = None
    ) -> BeliefState:
        """
        Create a new belief state for a user.

        Args:
            user_id: User identifier
            concept_ids: List of concept IDs in the curriculum
            initial_masteries: Optional initial mastery values

        Returns:
            New BeliefState initialized with priors
        """
        belief = BeliefState(
            user_id=user_id,
            concept_order=concept_ids.copy(),
            last_updated=datetime.now(),
        )

        # Initialize concept states with priors
        for concept_id in concept_ids:
            initial_mastery = self.config.p_l0
            if initial_masteries and concept_id in initial_masteries:
                initial_mastery = initial_masteries[concept_id]

            belief.concept_states[concept_id] = ConceptState(
                concept_id=concept_id,
                mastery_probability=initial_mastery,
                half_life_days=self.config.initial_half_life_days,
            )

        return belief

    def _compute_decay_factor(
        self,
        elapsed_days: float,
        half_life_days: float
    ) -> float:
        """
        Compute memory decay factor.

        Formula: F(τ) = 2^(-τ/h)

        Args:
            elapsed_days: Time since last interaction
            half_life_days: Memory half-life

        Returns:
            Decay factor in [0, 1]
        """
        if elapsed_days <= 0 or half_life_days <= 0:
            return 1.0

        if self.config.decay_model == DecayModel.HALF_LIFE:
            return math.pow(2, -elapsed_days / half_life_days)
        elif self.config.decay_model == DecayModel.EXPONENTIAL:
            return math.exp(-elapsed_days / half_life_days)
        elif self.config.decay_model == DecayModel.POWER_LAW:
            return math.pow(1 + elapsed_days, -1 / half_life_days)
        else:
            return math.pow(2, -elapsed_days / half_life_days)

    def _get_time_adjusted_params(
        self,
        elapsed_days: float
    ) -> Tuple[float, float]:
        """
        Get slip and guess probabilities adjusted for recency.

        Recent items: lower slip (less likely to make errors on familiar items)
        Old items: lower guess (harder to guess without recent practice)

        Args:
            elapsed_days: Time since last interaction

        Returns:
            (adjusted_p_s, adjusted_p_g)
        """
        p_s = self.config.p_s
        p_g = self.config.p_g

        if elapsed_days < self.config.recency_threshold_days:
            # Recent item - reduce slip probability
            recency_factor = 1 - (elapsed_days / self.config.recency_threshold_days)
            p_s = max(0.01, p_s - self.config.recency_slip_bonus * recency_factor)
        else:
            # Old item - reduce guess probability
            staleness = min(
                (elapsed_days - self.config.recency_threshold_days) /
                self.config.max_recency_days,
                1.0
            )
            p_g = max(0.01, p_g - self.config.recency_guess_penalty * staleness)

        return p_s, p_g

    def prediction_step(
        self,
        concept_state: ConceptState,
        current_time: Optional[datetime] = None
    ) -> Tuple[float, Dict]:
        """
        Prediction step: project mastery forward accounting for decay.

        This is the "time update" in the Kalman filter sense.
        Models forgetting that occurs between observations.

        Args:
            concept_state: Current concept state
            current_time: Current timestamp

        Returns:
            (predicted_mastery, prediction_details)
        """
        current_time = current_time or datetime.now()
        elapsed_days = concept_state.get_elapsed_days(current_time)

        # Compute decay
        decay_factor = self._compute_decay_factor(
            elapsed_days,
            concept_state.half_life_days
        )

        # Apply decay to mastery
        prior_mastery = concept_state.mastery_probability
        predicted_mastery = prior_mastery * decay_factor

        details = {
            "prior_mastery": prior_mastery,
            "predicted_mastery": predicted_mastery,
            "elapsed_days": elapsed_days,
            "decay_factor": decay_factor,
            "half_life_days": concept_state.half_life_days,
        }

        return predicted_mastery, details

    def correction_step(
        self,
        predicted_mastery: float,
        correct: bool,
        elapsed_days: float = 0.0
    ) -> Tuple[float, Dict]:
        """
        Correction step: update mastery based on observation.

        This is the "measurement update" in the Kalman filter sense.
        Uses Bayes' rule with the decayed prior.

        Args:
            predicted_mastery: Mastery after prediction step (decayed prior)
            correct: Whether the response was correct
            elapsed_days: Time since last interaction (for param adjustment)

        Returns:
            (posterior_mastery, correction_details)
        """
        # Get time-adjusted slip/guess probabilities
        p_s, p_g = self._get_time_adjusted_params(elapsed_days)

        p_l = predicted_mastery

        if correct:
            # P(Correct) = P(L) * (1 - P(S)) + (1 - P(L)) * P(G)
            p_correct = p_l * (1 - p_s) + (1 - p_l) * p_g

            # Bayes' rule: P(L | Correct) = P(Correct | L) * P(L) / P(Correct)
            numerator = p_l * (1 - p_s)
            p_l_given_obs = numerator / p_correct if p_correct > 0 else p_l
        else:
            # P(Incorrect) = P(L) * P(S) + (1 - P(L)) * (1 - P(G))
            p_incorrect = p_l * p_s + (1 - p_l) * (1 - p_g)

            # P(L | Incorrect) = P(Incorrect | L) * P(L) / P(Incorrect)
            numerator = p_l * p_s
            p_l_given_obs = numerator / p_incorrect if p_incorrect > 0 else p_l

        # Apply learning transition
        # P(Lt) = P(Lt-1 | observation) + (1 - P(Lt-1 | observation)) * P(T)
        posterior_mastery = p_l_given_obs + (1 - p_l_given_obs) * self.config.p_t

        # Ensure bounds
        posterior_mastery = max(0.0, min(1.0, posterior_mastery))

        details = {
            "predicted_mastery": predicted_mastery,
            "posterior_given_obs": p_l_given_obs,
            "posterior_after_learning": posterior_mastery,
            "correct": correct,
            "p_s_adjusted": p_s,
            "p_g_adjusted": p_g,
            "learning_gain": posterior_mastery - predicted_mastery,
        }

        return posterior_mastery, details

    def _update_half_life(
        self,
        current_half_life: float,
        correct: bool,
        elapsed_days: float
    ) -> float:
        """
        Update memory half-life based on review outcome.

        Successful recall increases half-life (spacing effect).
        Failed recall decreases half-life (weakened memory).

        Args:
            current_half_life: Current half-life in days
            correct: Whether recall was successful
            elapsed_days: Time since last review

        Returns:
            New half-life in days
        """
        if correct:
            # Success: increase half-life
            # Bonus for successfully recalling after longer delay
            spacing_bonus = 1.0 + min(elapsed_days / 30.0, 1.0) * 0.5
            new_half_life = current_half_life * self.config.half_life_growth_factor * spacing_bonus
        else:
            # Failure: decrease half-life
            new_half_life = current_half_life * self.config.half_life_decay_factor

        # Clamp to valid range
        new_half_life = max(0.1, min(self.config.max_half_life_days, new_half_life))

        return new_half_life

    def _detect_inflection_point(
        self,
        old_mastery: float,
        new_mastery: float,
        concept_state: ConceptState
    ) -> Tuple[bool, float]:
        """
        Detect cognitive inflection points (moments of rapid learning/forgetting).

        Args:
            old_mastery: Previous mastery
            new_mastery: Updated mastery
            concept_state: Current concept state

        Returns:
            (is_inflection, velocity)
        """
        velocity = new_mastery - old_mastery

        # Track velocity for trend analysis
        alpha = 0.3  # Exponential smoothing factor
        smoothed_velocity = alpha * velocity + (1 - alpha) * concept_state.mastery_velocity

        # Detect inflection if change exceeds threshold
        is_inflection = abs(velocity) >= self.config.inflection_threshold

        return is_inflection, smoothed_velocity

    def update(
        self,
        belief_state: BeliefState,
        concept_id: str,
        correct: bool,
        timestamp: Optional[datetime] = None,
        response_time_ms: Optional[int] = None
    ) -> Tuple[BeliefState, Dict]:
        """
        Update belief state after an observation.

        Implements the full TD-BKT update:
        1. Prediction step (time decay)
        2. Correction step (Bayesian update)
        3. Half-life update (memory strength)
        4. Inflection point detection

        Args:
            belief_state: Current belief state
            concept_id: Concept that was practiced
            correct: Whether response was correct
            timestamp: Time of observation
            response_time_ms: Optional response time for richer inference

        Returns:
            (updated_belief_state, update_details)
        """
        timestamp = timestamp or datetime.now()

        # Get or create concept state
        if concept_id not in belief_state.concept_states:
            belief_state.concept_states[concept_id] = ConceptState(
                concept_id=concept_id,
                mastery_probability=self.config.p_l0,
                half_life_days=self.config.initial_half_life_days,
            )
            if concept_id not in belief_state.concept_order:
                belief_state.concept_order.append(concept_id)

        concept_state = belief_state.concept_states[concept_id]
        old_mastery = concept_state.mastery_probability

        # Step 1: Prediction (time decay)
        predicted_mastery, prediction_details = self.prediction_step(
            concept_state, timestamp
        )

        # Step 2: Correction (Bayesian update)
        elapsed_days = concept_state.get_elapsed_days(timestamp)
        new_mastery, correction_details = self.correction_step(
            predicted_mastery, correct, elapsed_days
        )

        # Step 3: Update half-life
        old_half_life = concept_state.half_life_days
        new_half_life = self._update_half_life(
            old_half_life, correct, elapsed_days
        )

        # Step 4: Detect inflection points
        is_inflection = False
        velocity = 0.0
        if self.config.detect_inflection_points:
            is_inflection, velocity = self._detect_inflection_point(
                old_mastery, new_mastery, concept_state
            )

        # Update concept state
        concept_state.mastery_probability = new_mastery
        concept_state.half_life_days = new_half_life
        concept_state.last_interaction_at = timestamp
        concept_state.total_attempts += 1
        if correct:
            concept_state.correct_attempts += 1
            concept_state.consecutive_correct += 1
            concept_state.consecutive_incorrect = 0
        else:
            concept_state.consecutive_incorrect += 1
            concept_state.consecutive_correct = 0
        concept_state.is_at_inflection = is_inflection
        concept_state.mastery_velocity = velocity

        # Update belief state timestamp
        belief_state.last_updated = timestamp

        # Compile update details
        update_details = {
            "concept_id": concept_id,
            "correct": correct,
            "timestamp": timestamp.isoformat(),
            "old_mastery": old_mastery,
            "new_mastery": new_mastery,
            "mastery_change": new_mastery - old_mastery,
            "old_half_life": old_half_life,
            "new_half_life": new_half_life,
            "is_inflection": is_inflection,
            "mastery_velocity": velocity,
            "prediction": prediction_details,
            "correction": correction_details,
        }

        return belief_state, update_details

    def batch_update(
        self,
        belief_state: BeliefState,
        interactions: List[Tuple[str, bool, datetime]]
    ) -> Tuple[BeliefState, List[Dict]]:
        """
        Process multiple interactions in sequence.

        Args:
            belief_state: Initial belief state
            interactions: List of (concept_id, correct, timestamp) tuples

        Returns:
            (final_belief_state, list_of_update_details)
        """
        all_details = []

        # Sort by timestamp to ensure correct ordering
        sorted_interactions = sorted(interactions, key=lambda x: x[2])

        for concept_id, correct, timestamp in sorted_interactions:
            belief_state, details = self.update(
                belief_state, concept_id, correct, timestamp
            )
            all_details.append(details)

        return belief_state, all_details

    def predict_performance(
        self,
        belief_state: BeliefState,
        concept_id: str,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Predict performance on a concept.

        Args:
            belief_state: Current belief state
            concept_id: Concept to predict for
            timestamp: Prediction time

        Returns:
            Dictionary with performance predictions
        """
        timestamp = timestamp or datetime.now()

        if concept_id in belief_state.concept_states:
            concept_state = belief_state.concept_states[concept_id]
            elapsed_days = concept_state.get_elapsed_days(timestamp)

            # Get decayed mastery
            mastery = concept_state.get_retention_probability(timestamp)

            # Get time-adjusted params
            p_s, p_g = self._get_time_adjusted_params(elapsed_days)

            # Calculate P(Correct)
            p_correct = mastery * (1 - p_s) + (1 - mastery) * p_g
        else:
            mastery = self.config.p_l0
            p_correct = mastery * (1 - self.config.p_s) + (1 - mastery) * self.config.p_g
            elapsed_days = float('inf')

        return {
            "concept_id": concept_id,
            "mastery": mastery,
            "p_correct": p_correct,
            "p_incorrect": 1 - p_correct,
            "confidence": abs(p_correct - 0.5) * 2,
            "elapsed_days": elapsed_days if elapsed_days != float('inf') else None,
        }

    def get_review_urgency(
        self,
        belief_state: BeliefState,
        concept_id: str,
        timestamp: Optional[datetime] = None
    ) -> float:
        """
        Calculate review urgency for a concept.

        Urgency is high when:
        - Concept was well-learned but has decayed significantly
        - Approaching the point where retention drops below threshold

        Args:
            belief_state: Current belief state
            concept_id: Concept to check
            timestamp: Current time

        Returns:
            Urgency score in [0, 1]
        """
        timestamp = timestamp or datetime.now()

        if concept_id not in belief_state.concept_states:
            return 0.0  # Never practiced, not urgent to review

        concept_state = belief_state.concept_states[concept_id]

        # Raw mastery (what they learned)
        raw_mastery = concept_state.mastery_probability

        # Current retention (accounting for decay)
        current_retention = concept_state.get_retention_probability(timestamp)

        # Urgency based on decay from peak
        if raw_mastery > 0.2:  # Only urgent if they learned something
            decay_amount = raw_mastery - current_retention
            urgency = decay_amount / raw_mastery
        else:
            urgency = 0.0

        # Boost urgency if approaching mastery threshold
        if raw_mastery >= self.config.mastery_threshold:
            if current_retention < self.config.mastery_threshold:
                # Was mastered, now below threshold - very urgent!
                urgency = max(urgency, 0.8)
            elif current_retention < self.config.mastery_threshold * 1.1:
                # Close to dropping below threshold
                urgency = max(urgency, 0.5)

        return min(1.0, max(0.0, urgency))

    def get_all_urgencies(
        self,
        belief_state: BeliefState,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Get review urgencies for all concepts.

        Args:
            belief_state: Current belief state
            timestamp: Current time

        Returns:
            Dictionary mapping concept_id to urgency score
        """
        timestamp = timestamp or datetime.now()

        return {
            concept_id: self.get_review_urgency(belief_state, concept_id, timestamp)
            for concept_id in belief_state.concept_states.keys()
        }

    def serialize_config(self) -> Dict:
        """Serialize configuration for persistence"""
        return self.config.to_dict()

    @classmethod
    def from_config_dict(cls, config_dict: Dict) -> "TemporalDifferenceBKT":
        """Create instance from configuration dictionary"""
        config = TDBKTConfig.from_dict(config_dict)
        return cls(config)
