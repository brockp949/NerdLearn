"""
Composite Reward Function for ALGA-Next

Implements a multi-objective reward function that aligns the contextual bandit's
optimization with pedagogical goals rather than naive "clickbait" behavior.

Reward Formula:
    R_t = w1 * E_norm + w2 * M_norm + w3 * 1/(1 + exp(k * (F_t - τ)))

Where:
- E_norm: Normalized engagement (dwell time relative to content length)
- M_norm: Mastery (performance on subsequent micro-assessment)
- Fatigue Penalty: Sigmoid decay when fatigue exceeds threshold τ

This prevents the bandit from selecting high-intensity content that burns
the user out, even if it generates short-term engagement.

References:
- "Understanding contextual bandits: a guide to dynamic decision-making" (Kameleoon)
- "Contextual Bandits with Stage-wise Constraints" (JMLR)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class RewardComponents:
    """
    Components of the composite reward function

    All values normalized to [0, 1] range
    """
    # Engagement component (E_norm)
    engagement_score: float = 0.0
    dwell_time_ms: float = 0.0
    expected_dwell_ms: float = 0.0
    dwell_ratio: float = 0.0  # actual/expected
    scroll_completion: float = 0.0  # How much content was viewed
    interaction_rate: float = 0.0  # Interactions per minute

    # Mastery component (M_norm)
    assessment_score: Optional[float] = None  # Quiz/micro-assessment result
    concept_recall: float = 0.0  # Did user remember from previous session
    transfer_success: float = 0.0  # Success on related concepts

    # Fatigue indicators
    fatigue_level: float = 0.0  # Current fatigue estimate
    session_duration_minutes: float = 0.0
    consecutive_sessions: int = 0
    time_since_break: float = 0.0  # Minutes since last break

    # Meta
    modality_type: str = "text"
    content_complexity: float = 0.5
    content_duration_minutes: float = 5.0


@dataclass
class FatiguePenalty:
    """
    Configurable fatigue penalty using sigmoid decay

    When F_t > τ, reward is heavily penalized to prevent burnout.
    """
    # Threshold τ: Fatigue level above which penalty kicks in
    threshold: float = 0.6

    # Steepness k: How sharp the penalty curve is
    steepness: float = 10.0

    # Weight w3: Importance of fatigue penalty in final reward
    weight: float = 0.3

    def calculate(self, fatigue_level: float) -> float:
        """
        Calculate fatigue penalty

        Returns value in [0, 1]:
        - 1.0 when fatigue is low (no penalty)
        - ~0.5 when fatigue equals threshold
        - ~0 when fatigue is high (severe penalty)
        """
        return 1.0 / (1.0 + np.exp(self.steepness * (fatigue_level - self.threshold)))

    def get_penalty_curve(self, points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Get the penalty curve for visualization"""
        x = np.linspace(0, 1, points)
        y = np.array([self.calculate(f) for f in x])
        return x, y


class RewardObjective(str, Enum):
    """Different optimization objectives"""
    ENGAGEMENT = "engagement"  # Maximize time on task
    MASTERY = "mastery"  # Maximize learning outcomes
    BALANCED = "balanced"  # Balance engagement and mastery
    RETENTION = "retention"  # Maximize long-term retention
    WELLBEING = "wellbeing"  # Include fatigue/burnout prevention


@dataclass
class RewardConfig:
    """Configuration for the composite reward function"""
    # Component weights (should sum to 1.0)
    engagement_weight: float = 0.4
    mastery_weight: float = 0.4
    fatigue_weight: float = 0.2

    # Fatigue penalty configuration
    fatigue_penalty: FatiguePenalty = field(default_factory=FatiguePenalty)

    # Engagement scoring parameters
    min_engagement_threshold: float = 0.3  # Below this = failure
    optimal_dwell_ratio: float = 1.0  # Ideal actual/expected ratio
    dwell_ratio_tolerance: float = 0.5  # Acceptable deviation from optimal

    # Mastery scoring parameters
    mastery_floor: float = 0.0  # Minimum mastery when no assessment
    assessment_boost: float = 0.2  # Extra weight when assessment available

    # Time decay (rewards older observations less)
    time_decay_hours: float = 24.0
    decay_rate: float = 0.1

    # Objective preset
    objective: RewardObjective = RewardObjective.BALANCED

    @classmethod
    def for_objective(cls, objective: RewardObjective) -> "RewardConfig":
        """Create configuration for a specific objective"""
        if objective == RewardObjective.ENGAGEMENT:
            return cls(
                engagement_weight=0.7,
                mastery_weight=0.2,
                fatigue_weight=0.1,
                objective=objective,
            )
        elif objective == RewardObjective.MASTERY:
            return cls(
                engagement_weight=0.2,
                mastery_weight=0.7,
                fatigue_weight=0.1,
                objective=objective,
            )
        elif objective == RewardObjective.WELLBEING:
            return cls(
                engagement_weight=0.3,
                mastery_weight=0.3,
                fatigue_weight=0.4,
                fatigue_penalty=FatiguePenalty(threshold=0.5, steepness=15, weight=0.4),
                objective=objective,
            )
        elif objective == RewardObjective.RETENTION:
            return cls(
                engagement_weight=0.3,
                mastery_weight=0.5,
                fatigue_weight=0.2,
                objective=objective,
            )
        else:  # BALANCED
            return cls(objective=objective)


class CompositeRewardFunction:
    """
    Composite Reward Function aligned with pedagogical goals

    Calculates reward as weighted sum of:
    1. Engagement (E_norm): Did user engage with content?
    2. Mastery (M_norm): Did user learn from content?
    3. Fatigue Penalty: Prevents burnout-inducing content selection
    """

    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()

        # History for baseline computation
        self.reward_history: List[float] = []
        self.component_history: List[Dict[str, float]] = []

        logger.info(f"CompositeRewardFunction initialized with {self.config.objective.value} objective")

    def calculate(self, components: RewardComponents) -> Tuple[float, Dict[str, float]]:
        """
        Calculate composite reward from components

        Returns:
            reward: Final reward value in [0, 1]
            breakdown: Dictionary with component contributions
        """
        # Calculate engagement score
        engagement = self._calculate_engagement(components)

        # Calculate mastery score
        mastery = self._calculate_mastery(components)

        # Calculate fatigue penalty
        fatigue_penalty = self.config.fatigue_penalty.calculate(components.fatigue_level)

        # Composite reward
        w1 = self.config.engagement_weight
        w2 = self.config.mastery_weight
        w3 = self.config.fatigue_weight

        # Weighted sum with fatigue penalty
        raw_reward = w1 * engagement + w2 * mastery

        # Apply fatigue penalty
        reward = raw_reward * fatigue_penalty

        # Additional penalty for very high fatigue (safety valve)
        if components.fatigue_level > 0.9:
            reward *= 0.5

        # Ensure bounds
        reward = max(0.0, min(1.0, reward))

        # Create breakdown
        breakdown = {
            "engagement": engagement,
            "engagement_weighted": w1 * engagement,
            "mastery": mastery,
            "mastery_weighted": w2 * mastery,
            "fatigue_level": components.fatigue_level,
            "fatigue_penalty": fatigue_penalty,
            "fatigue_weighted": w3 * fatigue_penalty,
            "raw_reward": raw_reward,
            "final_reward": reward,
        }

        # Track history
        self.reward_history.append(reward)
        self.component_history.append(breakdown)

        return reward, breakdown

    def _calculate_engagement(self, components: RewardComponents) -> float:
        """
        Calculate normalized engagement score

        Considers:
        - Dwell time relative to expected
        - Scroll completion
        - Interaction rate
        """
        # Dwell time ratio scoring
        if components.expected_dwell_ms > 0:
            dwell_ratio = components.dwell_time_ms / components.expected_dwell_ms
        else:
            dwell_ratio = 1.0

        # Score dwell ratio (optimal = 1.0, too fast or too slow is penalized)
        optimal = self.config.optimal_dwell_ratio
        tolerance = self.config.dwell_ratio_tolerance

        if dwell_ratio < optimal - tolerance:
            # Too fast - likely skipped
            dwell_score = max(0, dwell_ratio / (optimal - tolerance))
        elif dwell_ratio > optimal + tolerance:
            # Too slow - might be distracted
            excess = dwell_ratio - (optimal + tolerance)
            dwell_score = max(0.3, 1.0 - excess * 0.3)
        else:
            # In optimal range
            dwell_score = 1.0

        # Scroll completion
        scroll_score = components.scroll_completion

        # Interaction rate (normalized, assuming 5/min is optimal)
        interaction_score = min(1.0, components.interaction_rate / 5.0)

        # Weighted combination
        engagement = (
            0.5 * dwell_score +
            0.3 * scroll_score +
            0.2 * interaction_score
        )

        # Apply minimum threshold
        if engagement < self.config.min_engagement_threshold:
            engagement = engagement * 0.5  # Heavily penalize below threshold

        return engagement

    def _calculate_mastery(self, components: RewardComponents) -> float:
        """
        Calculate normalized mastery score

        Considers:
        - Assessment score (if available)
        - Concept recall from previous sessions
        - Transfer to related concepts
        """
        if components.assessment_score is not None:
            # Assessment available - weight it heavily
            assessment = components.assessment_score
            recall = components.concept_recall
            transfer = components.transfer_success

            mastery = (
                0.6 * assessment +
                0.2 * recall +
                0.2 * transfer
            )
        else:
            # No assessment - use proxy signals
            recall = components.concept_recall
            transfer = components.transfer_success

            # Use engagement as weak proxy for learning
            dwell_ratio = components.dwell_ratio if components.dwell_ratio > 0 else 0.5
            engagement_proxy = min(1.0, dwell_ratio)

            mastery = (
                0.3 * engagement_proxy +  # Weak proxy
                0.4 * recall +
                0.3 * transfer
            )

            # Apply floor since we don't have direct measurement
            mastery = max(self.config.mastery_floor, mastery)

        return mastery

    def calculate_batch_reward(
        self,
        observations: List[RewardComponents],
        time_decay: bool = True,
    ) -> Tuple[float, List[Dict[str, float]]]:
        """
        Calculate reward for a batch of observations

        Optionally applies time decay to older observations.
        """
        rewards = []
        breakdowns = []

        for i, obs in enumerate(observations):
            reward, breakdown = self.calculate(obs)

            if time_decay and len(observations) > 1:
                # Apply time decay (newer observations weighted more)
                age_factor = (len(observations) - i) / len(observations)
                decay = np.exp(-self.config.decay_rate * (1 - age_factor))
                reward *= decay
                breakdown["time_decay"] = decay

            rewards.append(reward)
            breakdowns.append(breakdown)

        avg_reward = np.mean(rewards) if rewards else 0.0
        return avg_reward, breakdowns

    def get_baseline(self, n_recent: int = 20) -> float:
        """Get baseline reward from recent history"""
        if not self.reward_history:
            return 0.5

        recent = self.reward_history[-n_recent:]
        return np.mean(recent)

    def get_relative_reward(self, reward: float) -> float:
        """Get reward relative to baseline"""
        baseline = self.get_baseline()
        if baseline == 0:
            return reward
        return (reward - baseline) / baseline

    def suggest_intervention(
        self,
        components: RewardComponents,
    ) -> Optional[Dict[str, Any]]:
        """
        Suggest intervention based on reward components

        Returns intervention suggestion if reward is likely to be low.
        """
        interventions = []

        # Check fatigue
        if components.fatigue_level > 0.7:
            interventions.append({
                "type": "break_prompt",
                "reason": "high_fatigue",
                "message": "You've been studying for a while. Consider a short break!",
                "priority": "high",
            })

        # Check engagement
        if components.dwell_ratio < 0.5:
            interventions.append({
                "type": "modality_switch",
                "reason": "low_dwell_time",
                "message": "Content might be too complex. Try a different format?",
                "priority": "medium",
            })

        # Check scroll completion
        if components.scroll_completion < 0.3 and components.dwell_time_ms > 30000:
            interventions.append({
                "type": "scaffold",
                "reason": "stuck_at_top",
                "message": "Need help understanding this content?",
                "priority": "medium",
            })

        # Check session duration
        if components.session_duration_minutes > 45:
            interventions.append({
                "type": "break_prompt",
                "reason": "long_session",
                "message": "You've been at it for 45+ minutes. Time for a stretch!",
                "priority": "medium",
            })

        if interventions:
            # Return highest priority
            priority_order = {"high": 0, "medium": 1, "low": 2}
            interventions.sort(key=lambda x: priority_order.get(x["priority"], 2))
            return interventions[0]

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get reward statistics for monitoring"""
        if not self.reward_history:
            return {"total_rewards": 0}

        recent = self.reward_history[-100:]
        return {
            "total_rewards": len(self.reward_history),
            "recent_mean": float(np.mean(recent)),
            "recent_std": float(np.std(recent)),
            "recent_min": float(np.min(recent)),
            "recent_max": float(np.max(recent)),
            "engagement_weight": self.config.engagement_weight,
            "mastery_weight": self.config.mastery_weight,
            "fatigue_weight": self.config.fatigue_weight,
            "objective": self.config.objective.value,
        }


# Convenience function for quick reward calculation
def calculate_reward(
    dwell_time_ms: float,
    expected_dwell_ms: float,
    engagement_score: float = 0.5,
    assessment_score: Optional[float] = None,
    fatigue_level: float = 0.0,
    objective: RewardObjective = RewardObjective.BALANCED,
) -> float:
    """
    Quick utility to calculate reward

    Returns reward value in [0, 1]
    """
    config = RewardConfig.for_objective(objective)
    reward_fn = CompositeRewardFunction(config)

    components = RewardComponents(
        engagement_score=engagement_score,
        dwell_time_ms=dwell_time_ms,
        expected_dwell_ms=expected_dwell_ms,
        dwell_ratio=dwell_time_ms / expected_dwell_ms if expected_dwell_ms > 0 else 1.0,
        assessment_score=assessment_score,
        fatigue_level=fatigue_level,
    )

    reward, _ = reward_fn.calculate(components)
    return reward
