"""
Multi-Armed Bandit for Adaptive Content Selection
Implements exploration-exploitation strategies for optimal learning path selection

Research basis:
- Multi-Armed Bandits for Adaptive Learning: A Technical Implementation Guide
- Thompson Sampling for contextual bandits
- UCB1 for deterministic exploration
- Epsilon-greedy for simple exploration

Use cases:
- Content selection (which module to show next)
- Difficulty calibration (which difficulty level)
- Strategy selection (which teaching approach)
- A/B testing learning interventions
"""
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import random
import math
import numpy as np
from abc import ABC, abstractmethod


class BanditStrategy(str, Enum):
    """Available bandit strategies"""
    THOMPSON_SAMPLING = "thompson_sampling"
    UCB1 = "ucb1"
    EPSILON_GREEDY = "epsilon_greedy"
    CONTEXTUAL = "contextual"


@dataclass
class ContentArm:
    """Represents a content option (arm) in the bandit"""
    arm_id: str
    content_type: str  # "module", "difficulty", "strategy"
    content_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Statistics
    pulls: int = 0  # Number of times selected
    rewards: float = 0.0  # Cumulative reward
    successes: int = 0  # For beta distribution (Thompson)
    failures: int = 0

    # For UCB
    last_pulled: Optional[datetime] = None

    @property
    def mean_reward(self) -> float:
        """Average reward per pull"""
        return self.rewards / self.pulls if self.pulls > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Success rate for binary outcomes"""
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.5


@dataclass
class SelectionResult:
    """Result of arm selection"""
    arm: ContentArm
    strategy: BanditStrategy
    confidence: float  # Confidence in selection
    exploration_score: float  # How much this was exploration vs exploitation
    alternatives: List[Tuple[str, float]]  # Other arms and their scores


class MultiArmedBandit(ABC):
    """Abstract base class for bandit algorithms"""

    def __init__(self, arms: List[ContentArm]):
        self.arms = {arm.arm_id: arm for arm in arms}
        self.total_pulls = 0

    @abstractmethod
    def select_arm(self) -> SelectionResult:
        """Select an arm to pull"""
        pass

    def update(self, arm_id: str, reward: float, success: bool = True):
        """Update arm statistics after observing reward"""
        if arm_id not in self.arms:
            return

        arm = self.arms[arm_id]
        arm.pulls += 1
        arm.rewards += reward
        arm.last_pulled = datetime.now()

        if success:
            arm.successes += 1
        else:
            arm.failures += 1

        self.total_pulls += 1

    def add_arm(self, arm: ContentArm):
        """Add a new arm"""
        self.arms[arm.arm_id] = arm

    def remove_arm(self, arm_id: str):
        """Remove an arm"""
        if arm_id in self.arms:
            del self.arms[arm_id]

    def get_statistics(self) -> Dict[str, Any]:
        """Get bandit statistics"""
        return {
            "total_arms": len(self.arms),
            "total_pulls": self.total_pulls,
            "arms": [
                {
                    "arm_id": arm.arm_id,
                    "pulls": arm.pulls,
                    "mean_reward": round(arm.mean_reward, 4),
                    "success_rate": round(arm.success_rate, 4),
                }
                for arm in self.arms.values()
            ]
        }


class ThompsonSampling(MultiArmedBandit):
    """
    Thompson Sampling - Bayesian approach to exploration-exploitation

    Uses Beta distribution for binary rewards:
    - Sample from Beta(successes + 1, failures + 1) for each arm
    - Select arm with highest sampled value

    Advantages:
    - Naturally balances exploration/exploitation
    - Handles uncertainty well
    - Good theoretical guarantees
    """

    def __init__(self, arms: List[ContentArm], prior_alpha: float = 1.0, prior_beta: float = 1.0):
        super().__init__(arms)
        self.prior_alpha = prior_alpha  # Prior successes
        self.prior_beta = prior_beta    # Prior failures

    def select_arm(self) -> SelectionResult:
        """Select arm using Thompson Sampling"""
        samples = {}

        for arm_id, arm in self.arms.items():
            # Sample from Beta distribution
            alpha = arm.successes + self.prior_alpha
            beta = arm.failures + self.prior_beta
            samples[arm_id] = np.random.beta(alpha, beta)

        # Select arm with highest sample
        best_arm_id = max(samples, key=samples.get)
        best_arm = self.arms[best_arm_id]

        # Calculate exploration score (higher variance = more exploration)
        alpha = best_arm.successes + self.prior_alpha
        beta = best_arm.failures + self.prior_beta
        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        exploration_score = min(1.0, variance * 10)  # Normalize

        # Confidence based on number of pulls
        confidence = 1 - 1 / (1 + best_arm.pulls)

        # Get alternatives
        sorted_arms = sorted(samples.items(), key=lambda x: x[1], reverse=True)
        alternatives = [(arm_id, round(score, 4)) for arm_id, score in sorted_arms[1:5]]

        return SelectionResult(
            arm=best_arm,
            strategy=BanditStrategy.THOMPSON_SAMPLING,
            confidence=round(confidence, 4),
            exploration_score=round(exploration_score, 4),
            alternatives=alternatives,
        )


class UCB1(MultiArmedBandit):
    """
    Upper Confidence Bound (UCB1) Algorithm

    Formula: UCB = mean_reward + c * sqrt(ln(total_pulls) / arm_pulls)

    Advantages:
    - Deterministic (no randomness in selection)
    - Strong theoretical bounds
    - Good for stationary environments
    """

    def __init__(self, arms: List[ContentArm], exploration_constant: float = 2.0):
        super().__init__(arms)
        self.c = exploration_constant

    def select_arm(self) -> SelectionResult:
        """Select arm using UCB1"""
        # First, pull each arm at least once
        for arm_id, arm in self.arms.items():
            if arm.pulls == 0:
                return SelectionResult(
                    arm=arm,
                    strategy=BanditStrategy.UCB1,
                    confidence=0.0,
                    exploration_score=1.0,  # Pure exploration
                    alternatives=[],
                )

        ucb_scores = {}

        for arm_id, arm in self.arms.items():
            # UCB formula
            exploitation = arm.mean_reward
            exploration = self.c * math.sqrt(math.log(self.total_pulls) / arm.pulls)
            ucb_scores[arm_id] = exploitation + exploration

        # Select arm with highest UCB
        best_arm_id = max(ucb_scores, key=ucb_scores.get)
        best_arm = self.arms[best_arm_id]

        # Calculate exploration score
        exploitation = best_arm.mean_reward
        exploration = self.c * math.sqrt(math.log(self.total_pulls) / best_arm.pulls)
        total = exploitation + exploration
        exploration_score = exploration / total if total > 0 else 0.5

        # Confidence based on UCB gap
        sorted_scores = sorted(ucb_scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            gap = sorted_scores[0] - sorted_scores[1]
            confidence = min(1.0, gap * 2)
        else:
            confidence = 1.0

        # Alternatives
        sorted_arms = sorted(ucb_scores.items(), key=lambda x: x[1], reverse=True)
        alternatives = [(arm_id, round(score, 4)) for arm_id, score in sorted_arms[1:5]]

        return SelectionResult(
            arm=best_arm,
            strategy=BanditStrategy.UCB1,
            confidence=round(confidence, 4),
            exploration_score=round(exploration_score, 4),
            alternatives=alternatives,
        )


class EpsilonGreedy(MultiArmedBandit):
    """
    Epsilon-Greedy Algorithm

    With probability epsilon: explore (random arm)
    With probability 1-epsilon: exploit (best known arm)

    Advantages:
    - Simple to implement and understand
    - Easy to tune
    - Good baseline
    """

    def __init__(self, arms: List[ContentArm], epsilon: float = 0.1, decay: float = 0.999):
        super().__init__(arms)
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.decay = decay

    def select_arm(self) -> SelectionResult:
        """Select arm using epsilon-greedy"""
        explore = random.random() < self.epsilon

        if explore or self.total_pulls == 0:
            # Random exploration
            arm = random.choice(list(self.arms.values()))
            exploration_score = 1.0
        else:
            # Exploit best arm
            arm = max(self.arms.values(), key=lambda a: a.mean_reward)
            exploration_score = 0.0

        # Decay epsilon
        self.epsilon *= self.decay

        # Calculate confidence
        if arm.pulls > 0:
            confidence = 1 - 1 / (1 + arm.pulls)
        else:
            confidence = 0.0

        # Alternatives (sorted by mean reward)
        sorted_arms = sorted(
            self.arms.items(),
            key=lambda x: x[1].mean_reward,
            reverse=True
        )
        alternatives = [
            (arm_id, round(a.mean_reward, 4))
            for arm_id, a in sorted_arms[1:5]
        ]

        return SelectionResult(
            arm=arm,
            strategy=BanditStrategy.EPSILON_GREEDY,
            confidence=round(confidence, 4),
            exploration_score=exploration_score,
            alternatives=alternatives,
        )

    def reset_epsilon(self):
        """Reset epsilon to initial value"""
        self.epsilon = self.initial_epsilon


class ContextualBandit(MultiArmedBandit):
    """
    Contextual Bandit - Uses learner context for arm selection

    Context features:
    - Mastery level
    - Expertise level
    - Cognitive load
    - Time of day
    - Recent performance

    Uses linear regression to predict reward for each arm given context
    """

    def __init__(
        self,
        arms: List[ContentArm],
        context_dim: int = 5,
        learning_rate: float = 0.1,
        exploration_bonus: float = 0.5,
    ):
        super().__init__(arms)
        self.context_dim = context_dim
        self.learning_rate = learning_rate
        self.exploration_bonus = exploration_bonus

        # Initialize weights for each arm (linear model)
        self.weights: Dict[str, np.ndarray] = {
            arm_id: np.zeros(context_dim)
            for arm_id in self.arms
        }

        # Store context history for each arm
        self.context_history: Dict[str, List[Tuple[np.ndarray, float]]] = {
            arm_id: [] for arm_id in self.arms
        }

    def select_arm(self, context: Optional[List[float]] = None) -> SelectionResult:
        """
        Select arm based on context

        Args:
            context: Feature vector [mastery, expertise, cog_load, time_factor, recent_perf]
        """
        if context is None:
            context = [0.5] * self.context_dim

        context = np.array(context)

        # Predict reward for each arm
        predictions = {}
        uncertainties = {}

        for arm_id, arm in self.arms.items():
            # Linear prediction
            pred = np.dot(self.weights[arm_id], context)

            # Add exploration bonus based on uncertainty
            n_samples = len(self.context_history[arm_id])
            uncertainty = self.exploration_bonus / math.sqrt(1 + n_samples)

            predictions[arm_id] = pred + uncertainty
            uncertainties[arm_id] = uncertainty

        # Select best arm
        best_arm_id = max(predictions, key=predictions.get)
        best_arm = self.arms[best_arm_id]

        # Calculate exploration score
        exploration_score = uncertainties[best_arm_id] / (
            predictions[best_arm_id] + 0.001
        ) if predictions[best_arm_id] > 0 else 1.0
        exploration_score = min(1.0, exploration_score)

        # Confidence
        n_samples = len(self.context_history[best_arm_id])
        confidence = 1 - 1 / (1 + n_samples)

        # Alternatives
        sorted_arms = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        alternatives = [(arm_id, round(score, 4)) for arm_id, score in sorted_arms[1:5]]

        return SelectionResult(
            arm=best_arm,
            strategy=BanditStrategy.CONTEXTUAL,
            confidence=round(confidence, 4),
            exploration_score=round(exploration_score, 4),
            alternatives=alternatives,
        )

    def update_with_context(
        self,
        arm_id: str,
        context: List[float],
        reward: float,
        success: bool = True
    ):
        """Update arm with contextual feedback"""
        # Standard update
        self.update(arm_id, reward, success)

        if arm_id not in self.weights:
            return

        context = np.array(context)

        # Store in history
        self.context_history[arm_id].append((context, reward))

        # Update weights using gradient descent
        prediction = np.dot(self.weights[arm_id], context)
        error = reward - prediction
        self.weights[arm_id] += self.learning_rate * error * context

    def add_arm(self, arm: ContentArm):
        """Add new arm with initialized weights"""
        super().add_arm(arm)
        self.weights[arm.arm_id] = np.zeros(self.context_dim)
        self.context_history[arm.arm_id] = []


class ContentSelector:
    """
    High-level content selector using MAB strategies

    Provides easy-to-use interface for content selection in learning systems
    """

    def __init__(
        self,
        strategy: BanditStrategy = BanditStrategy.THOMPSON_SAMPLING,
        **kwargs
    ):
        self.strategy = strategy
        self.kwargs = kwargs
        self.bandits: Dict[str, MultiArmedBandit] = {}

    def get_or_create_bandit(
        self,
        bandit_id: str,
        arms: List[ContentArm]
    ) -> MultiArmedBandit:
        """Get existing bandit or create new one"""
        if bandit_id not in self.bandits:
            self.bandits[bandit_id] = self._create_bandit(arms)
        return self.bandits[bandit_id]

    def _create_bandit(self, arms: List[ContentArm]) -> MultiArmedBandit:
        """Create bandit based on strategy"""
        if self.strategy == BanditStrategy.THOMPSON_SAMPLING:
            return ThompsonSampling(arms, **self.kwargs)
        elif self.strategy == BanditStrategy.UCB1:
            return UCB1(arms, **self.kwargs)
        elif self.strategy == BanditStrategy.EPSILON_GREEDY:
            return EpsilonGreedy(arms, **self.kwargs)
        elif self.strategy == BanditStrategy.CONTEXTUAL:
            return ContextualBandit(arms, **self.kwargs)
        else:
            return ThompsonSampling(arms)

    def select_content(
        self,
        user_id: int,
        content_type: str,
        available_content: List[Dict],
        context: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Select optimal content for user

        Args:
            user_id: User ID
            content_type: Type of content (module, difficulty, etc.)
            available_content: List of available content options
            context: Optional context features for contextual bandit

        Returns:
            Selected content with metadata
        """
        bandit_id = f"{user_id}_{content_type}"

        # Create arms from content
        arms = [
            ContentArm(
                arm_id=f"{content_type}_{c['id']}",
                content_type=content_type,
                content_id=c['id'],
                metadata=c,
            )
            for c in available_content
        ]

        bandit = self.get_or_create_bandit(bandit_id, arms)

        # Select arm
        if isinstance(bandit, ContextualBandit) and context:
            result = bandit.select_arm(context)
        else:
            result = bandit.select_arm()

        return {
            "selected_content_id": result.arm.content_id,
            "content_metadata": result.arm.metadata,
            "strategy": result.strategy.value,
            "confidence": result.confidence,
            "exploration_score": result.exploration_score,
            "alternatives": [
                {"arm_id": arm_id, "score": score}
                for arm_id, score in result.alternatives
            ],
        }

    def record_outcome(
        self,
        user_id: int,
        content_type: str,
        content_id: int,
        reward: float,
        success: bool = True,
        context: Optional[List[float]] = None,
    ):
        """Record outcome of content selection"""
        bandit_id = f"{user_id}_{content_type}"
        arm_id = f"{content_type}_{content_id}"

        if bandit_id in self.bandits:
            bandit = self.bandits[bandit_id]
            if isinstance(bandit, ContextualBandit) and context:
                bandit.update_with_context(arm_id, context, reward, success)
            else:
                bandit.update(arm_id, reward, success)
