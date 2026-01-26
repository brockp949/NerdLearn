"""
Hybrid LinUCB with Interaction Terms for Educational Modality Selection

Implements the contextual bandit algorithm from ALGA-Next that models both
user-specific preferences and shared content features with explicit interaction terms.

Mathematical Formulation:
    E[r_t,a | x_t,a] = x_t,a^T * β* + z_t,a^T * θ*_a

    Where:
    - x_t,a: shared features (fatigue, device_type, time_of_day)
    - β*: unknown coefficients for shared features
    - z_t,a: arm-specific features (video_duration, text_reading_level)
    - θ*_a: unknown coefficients specific to arm a

Key Interaction Terms (polynomial expansion):
    - Fatigue × Difficulty: User tolerance threshold
    - Device × Modality: Mobile penalty for long-form text
    - Time × Complexity: Evening preference for simpler content

Research basis:
- Li et al., "A Contextual-Bandit Approach to Personalized News Article Recommendation"
- "Scalable LinUCB: Low-Rank Design Matrix Updates"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
import numpy as np
from scipy import linalg
import logging

logger = logging.getLogger(__name__)


class Modality(str, Enum):
    """Supported content modalities"""
    TEXT = "text"
    VIDEO = "video"
    AUDIO = "audio"
    INTERACTIVE = "interactive"
    DIAGRAM = "diagram"
    PODCAST = "podcast"


@dataclass
class ModalityArm:
    """
    Represents a modality option (arm) in the bandit

    Attributes:
        modality: The content modality type
        content_id: Specific content instance ID
        features: Arm-specific features (z_t,a)
        metadata: Additional content metadata
    """
    modality: Modality
    content_id: str
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def arm_id(self) -> str:
        return f"{self.modality.value}_{self.content_id}"

    def get_feature_vector(self, feature_dim: int) -> np.ndarray:
        """Convert features dict to vector with consistent ordering"""
        # Standard arm features
        feature_keys = [
            "duration_minutes",
            "reading_level",
            "complexity_score",
            "interaction_level",
            "cognitive_load_estimate",
        ]

        vector = np.zeros(feature_dim)
        for i, key in enumerate(feature_keys[:feature_dim]):
            vector[i] = self.features.get(key, 0.5)  # Default to 0.5 (neutral)

        return vector


@dataclass
class ContextVector:
    """
    Context features for the current decision point

    Shared features (x_t,a) that apply across all arms:
    - User state: fatigue, attention, cognitive_capacity
    - Environmental: device_type, time_of_day, bandwidth
    - Content: current_difficulty, topic_familiarity
    """
    # User state features
    fatigue_level: float = 0.5  # 0 = fresh, 1 = exhausted
    attention_level: float = 0.5  # 0 = distracted, 1 = focused
    cognitive_capacity: float = 0.5  # Current available mental resources

    # Environmental features
    device_type: str = "desktop"  # "desktop", "mobile", "tablet"
    time_of_day: float = 0.5  # 0 = morning, 0.5 = afternoon, 1 = evening
    bandwidth_quality: float = 1.0  # Network quality for video/audio

    # Content context
    current_difficulty: float = 0.5  # Difficulty of current concept
    topic_familiarity: float = 0.5  # User's familiarity with topic
    session_duration_minutes: float = 0.0  # How long user has been studying

    # Historical performance
    recent_success_rate: float = 0.5  # Success rate on recent cards
    modality_preferences: Dict[str, float] = field(default_factory=dict)

    def to_shared_vector(self, feature_dim: int) -> np.ndarray:
        """Convert to shared feature vector x_t,a"""
        # Encode device type as one-hot
        device_encoding = {
            "desktop": [1, 0, 0],
            "mobile": [0, 1, 0],
            "tablet": [0, 0, 1],
        }.get(self.device_type, [0.33, 0.33, 0.33])

        features = [
            self.fatigue_level,
            self.attention_level,
            self.cognitive_capacity,
            *device_encoding,
            self.time_of_day,
            self.bandwidth_quality,
            self.current_difficulty,
            self.topic_familiarity,
            min(1.0, self.session_duration_minutes / 60),  # Normalize to 1 hour
            self.recent_success_rate,
        ]

        # Pad or truncate to feature_dim
        if len(features) < feature_dim:
            features.extend([0.0] * (feature_dim - len(features)))

        return np.array(features[:feature_dim])


class InteractionFeatureBuilder:
    """
    Builds polynomial interaction terms for feature engineering

    Key interactions from ALGA-Next:
    - Fatigue × Difficulty: Captures tolerance threshold when tired
    - Device × TextHeavy: Mobile penalty for long-form text
    - Time × Complexity: Evening preference for lighter content
    """

    def __init__(self, include_quadratic: bool = True, include_cross_terms: bool = True):
        self.include_quadratic = include_quadratic
        self.include_cross_terms = include_cross_terms

    def build_interaction_features(
        self,
        context: ContextVector,
        arm: ModalityArm
    ) -> np.ndarray:
        """
        Build interaction features between context and arm

        Returns combined feature vector with interaction terms
        """
        interactions = []

        # Primary interaction: Fatigue × Difficulty
        fatigue_difficulty = context.fatigue_level * context.current_difficulty
        interactions.append(fatigue_difficulty)

        # Device × Modality interactions
        is_mobile = 1.0 if context.device_type == "mobile" else 0.0
        is_text_heavy = 1.0 if arm.modality in [Modality.TEXT, Modality.DIAGRAM] else 0.0
        mobile_text_penalty = is_mobile * is_text_heavy
        interactions.append(mobile_text_penalty)

        # Video suitability on mobile
        is_video = 1.0 if arm.modality in [Modality.VIDEO, Modality.PODCAST] else 0.0
        mobile_video_boost = is_mobile * is_video * context.bandwidth_quality
        interactions.append(mobile_video_boost)

        # Time × Complexity: Evening penalty for complex content
        evening_factor = max(0, context.time_of_day - 0.6)  # After 6pm
        complexity = arm.features.get("complexity_score", 0.5)
        evening_complexity_penalty = evening_factor * complexity
        interactions.append(evening_complexity_penalty)

        # Session fatigue × Interactive content
        # Reduce interactivity preference as session progresses
        session_fatigue = min(1.0, context.session_duration_minutes / 45)  # 45 min threshold
        is_interactive = 1.0 if arm.modality == Modality.INTERACTIVE else 0.0
        interaction_fatigue = session_fatigue * is_interactive
        interactions.append(interaction_fatigue)

        # Cognitive capacity × Content cognitive load
        cognitive_load = arm.features.get("cognitive_load_estimate", 0.5)
        capacity_load_mismatch = abs(context.cognitive_capacity - (1 - cognitive_load))
        interactions.append(capacity_load_mismatch)

        # Success rate × Difficulty adjustment
        # If struggling, prefer lower difficulty content
        struggling = max(0, 0.5 - context.recent_success_rate)
        difficulty = context.current_difficulty
        struggle_difficulty = struggling * difficulty
        interactions.append(struggle_difficulty)

        if self.include_quadratic:
            # Quadratic terms for non-linear relationships
            interactions.append(context.fatigue_level ** 2)
            interactions.append(context.attention_level ** 2)

        return np.array(interactions)


@dataclass
class ModalityPolicy:
    """Result of modality selection"""
    selected_arm: ModalityArm
    confidence: float  # UCB confidence in selection
    exploration_bonus: float  # How much this was exploration
    expected_reward: float  # Predicted reward
    alternatives: List[Tuple[str, float]]  # Other arms and their scores
    explanation: str  # Human-readable explanation


class HybridLinUCB:
    """
    Hybrid LinUCB algorithm for adaptive modality selection

    Maintains both:
    - Shared parameters β (learned across all arms)
    - Arm-specific parameters θ_a (learned per modality)

    Uses ridge regression with online updates for both parameter sets.

    Args:
        shared_dim: Dimension of shared context features
        arm_dim: Dimension of arm-specific features
        interaction_dim: Dimension of interaction features
        alpha: Exploration parameter (controls UCB width)
        lambda_reg: Ridge regularization parameter
    """

    def __init__(
        self,
        shared_dim: int = 12,
        arm_dim: int = 5,
        interaction_dim: int = 9,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
    ):
        self.shared_dim = shared_dim
        self.arm_dim = arm_dim
        self.interaction_dim = interaction_dim
        self.alpha = alpha
        self.lambda_reg = lambda_reg

        # Combined dimension for shared + interaction features
        self.combined_shared_dim = shared_dim + interaction_dim

        # Shared parameters (β)
        # A_0: d×d matrix for shared features
        self.A_0 = lambda_reg * np.eye(self.combined_shared_dim)
        # b_0: d-dimensional vector
        self.b_0 = np.zeros(self.combined_shared_dim)

        # Per-arm parameters (θ_a)
        # A_a: k×k matrices for arm-specific features
        self.A_arms: Dict[str, np.ndarray] = {}
        # B_a: k×d matrices for cross terms
        self.B_arms: Dict[str, np.ndarray] = {}
        # b_a: k-dimensional vectors
        self.b_arms: Dict[str, np.ndarray] = {}

        # Interaction feature builder
        self.interaction_builder = InteractionFeatureBuilder()

        # Statistics
        self.total_selections = 0
        self.arm_selections: Dict[str, int] = {}

        logger.info(f"HybridLinUCB initialized: shared_dim={shared_dim}, arm_dim={arm_dim}")

    def _init_arm(self, arm_id: str):
        """Initialize parameters for a new arm"""
        if arm_id not in self.A_arms:
            self.A_arms[arm_id] = self.lambda_reg * np.eye(self.arm_dim)
            self.B_arms[arm_id] = np.zeros((self.arm_dim, self.combined_shared_dim))
            self.b_arms[arm_id] = np.zeros(self.arm_dim)
            self.arm_selections[arm_id] = 0

    def select_arm(
        self,
        context: ContextVector,
        available_arms: List[ModalityArm],
    ) -> ModalityPolicy:
        """
        Select the best modality arm using Hybrid LinUCB

        For each arm a, computes:
            UCB_a = z_a^T * θ_a + x^T * β + α * sqrt(uncertainty_a)

        Returns the arm with highest UCB value.
        """
        if not available_arms:
            raise ValueError("No arms available for selection")

        # Get shared context features
        x = context.to_shared_vector(self.shared_dim)

        # Compute β estimate (shared parameters)
        try:
            A_0_inv = linalg.inv(self.A_0)
        except linalg.LinAlgError:
            A_0_inv = np.eye(self.combined_shared_dim)

        beta_hat = A_0_inv @ self.b_0

        ucb_scores = {}
        expected_rewards = {}

        for arm in available_arms:
            self._init_arm(arm.arm_id)

            # Get arm-specific features
            z = arm.get_feature_vector(self.arm_dim)

            # Get interaction features
            interactions = self.interaction_builder.build_interaction_features(context, arm)

            # Combined shared features
            x_combined = np.concatenate([x, interactions])

            # Get arm parameters
            A_a = self.A_arms[arm.arm_id]
            B_a = self.B_arms[arm.arm_id]
            b_a = self.b_arms[arm.arm_id]

            # Compute A_a inverse
            try:
                A_a_inv = linalg.inv(A_a)
            except linalg.LinAlgError:
                A_a_inv = np.eye(self.arm_dim)

            # Compute θ_a estimate
            theta_hat = A_a_inv @ (b_a - B_a @ beta_hat)

            # Expected reward: z^T * θ + x^T * β
            expected_reward = z.T @ theta_hat + x_combined.T @ beta_hat
            expected_rewards[arm.arm_id] = expected_reward

            # Compute variance term (uncertainty)
            # s_a = z^T * A_a^{-1} * z + x^T * A_0^{-1} * x
            #       - 2 * x^T * A_0^{-1} * B_a^T * A_a^{-1} * z
            #       + x^T * A_0^{-1} * B_a^T * A_a^{-1} * B_a * A_0^{-1} * x

            # Simplified variance computation
            term1 = z.T @ A_a_inv @ z
            term2 = x_combined.T @ A_0_inv @ x_combined
            term3 = 2 * x_combined.T @ A_0_inv @ B_a.T @ A_a_inv @ z
            term4 = x_combined.T @ A_0_inv @ B_a.T @ A_a_inv @ B_a @ A_0_inv @ x_combined

            variance = term1 + term2 - term3 + term4
            variance = max(0.001, variance)  # Ensure non-negative

            # UCB score
            exploration_bonus = self.alpha * np.sqrt(variance)
            ucb_score = expected_reward + exploration_bonus
            ucb_scores[arm.arm_id] = (ucb_score, exploration_bonus)

        # Select arm with highest UCB
        best_arm_id = max(ucb_scores, key=lambda k: ucb_scores[k][0])
        best_arm = next(a for a in available_arms if a.arm_id == best_arm_id)

        # Get alternatives
        sorted_arms = sorted(
            ucb_scores.items(),
            key=lambda x: x[1][0],
            reverse=True
        )
        alternatives = [(arm_id, score) for arm_id, (score, _) in sorted_arms[1:5]]

        # Calculate confidence (based on gap between best and second-best)
        if len(sorted_arms) > 1:
            gap = sorted_arms[0][1][0] - sorted_arms[1][1][0]
            confidence = min(1.0, gap / (abs(sorted_arms[0][1][0]) + 0.001))
        else:
            confidence = 1.0

        # Generate explanation
        explanation = self._generate_explanation(context, best_arm, ucb_scores)

        self.total_selections += 1
        self.arm_selections[best_arm_id] = self.arm_selections.get(best_arm_id, 0) + 1

        return ModalityPolicy(
            selected_arm=best_arm,
            confidence=float(confidence),
            exploration_bonus=float(ucb_scores[best_arm_id][1]),
            expected_reward=float(expected_rewards[best_arm_id]),
            alternatives=alternatives,
            explanation=explanation,
        )

    def update(
        self,
        context: ContextVector,
        arm: ModalityArm,
        reward: float,
    ):
        """
        Update model parameters after observing reward

        Uses online ridge regression update for both shared and arm-specific params.
        """
        arm_id = arm.arm_id
        self._init_arm(arm_id)

        # Get features
        x = context.to_shared_vector(self.shared_dim)
        z = arm.get_feature_vector(self.arm_dim)
        interactions = self.interaction_builder.build_interaction_features(context, arm)
        x_combined = np.concatenate([x, interactions])

        # Update shared parameters (A_0, b_0)
        self.A_0 += self.B_arms[arm_id].T @ linalg.inv(self.A_arms[arm_id]) @ self.B_arms[arm_id]
        self.b_0 += self.B_arms[arm_id].T @ linalg.inv(self.A_arms[arm_id]) @ self.b_arms[arm_id]

        # Update arm-specific parameters
        self.A_arms[arm_id] += np.outer(z, z)
        self.B_arms[arm_id] += np.outer(z, x_combined)
        self.b_arms[arm_id] += reward * z

        # Update shared parameters after arm update
        self.A_0 += np.outer(x_combined, x_combined) - self.B_arms[arm_id].T @ linalg.inv(self.A_arms[arm_id]) @ self.B_arms[arm_id]
        self.b_0 += reward * x_combined - self.B_arms[arm_id].T @ linalg.inv(self.A_arms[arm_id]) @ self.b_arms[arm_id]

        logger.debug(f"Updated HybridLinUCB for arm {arm_id} with reward {reward:.3f}")

    def _generate_explanation(
        self,
        context: ContextVector,
        arm: ModalityArm,
        ucb_scores: Dict[str, Tuple[float, float]],
    ) -> str:
        """Generate human-readable explanation for selection"""
        explanations = []

        # Explain context factors
        if context.fatigue_level > 0.7:
            explanations.append("User appears fatigued; prefer passive content")

        if context.device_type == "mobile":
            if arm.modality in [Modality.VIDEO, Modality.PODCAST]:
                explanations.append("Mobile device favors video/audio content")
            else:
                explanations.append("Text content less optimal for mobile")

        if context.time_of_day > 0.75:
            explanations.append("Late evening; lighter content preferred")

        if context.recent_success_rate < 0.4:
            explanations.append("Recent struggles; may need scaffolding")

        # Explain selection
        score, exploration = ucb_scores[arm.arm_id]
        if exploration > score * 0.3:
            explanations.append("Exploring less-tried modality")
        else:
            explanations.append("Exploiting known effective modality")

        if not explanations:
            explanations.append("Default selection based on learned preferences")

        return "; ".join(explanations)

    def get_statistics(self) -> Dict[str, Any]:
        """Get bandit statistics for monitoring"""
        return {
            "total_selections": self.total_selections,
            "arm_selections": self.arm_selections,
            "alpha": self.alpha,
            "arms_initialized": list(self.A_arms.keys()),
        }

    def set_exploration_rate(self, alpha: float):
        """Adjust exploration parameter"""
        self.alpha = max(0.01, alpha)
        logger.info(f"Exploration rate set to {self.alpha}")


# Convenience function for creating modality arms from content
def create_modality_arms(
    content_options: List[Dict[str, Any]],
) -> List[ModalityArm]:
    """
    Create ModalityArm objects from content options

    Expected content format:
    {
        "id": "content_123",
        "modality": "video",
        "duration_minutes": 5,
        "reading_level": 0.6,
        "complexity_score": 0.5,
        ...
    }
    """
    arms = []
    for content in content_options:
        modality = Modality(content.get("modality", "text"))
        arm = ModalityArm(
            modality=modality,
            content_id=content.get("id", "unknown"),
            features={
                "duration_minutes": content.get("duration_minutes", 5) / 30,  # Normalize
                "reading_level": content.get("reading_level", 0.5),
                "complexity_score": content.get("complexity_score", 0.5),
                "interaction_level": content.get("interaction_level", 0.5),
                "cognitive_load_estimate": content.get("cognitive_load_estimate", 0.5),
            },
            metadata=content,
        )
        arms.append(arm)

    return arms
