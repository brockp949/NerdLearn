"""
Differential Privacy for Educational AI
Privacy-preserving machine learning and analytics

Research basis:
- Privacy-Preserving Machine Learning for Educational AI
- FERPA and GDPR compliance requirements
- Differential Privacy: ε-δ privacy guarantees
- Local vs Global differential privacy

Key concepts:
- Epsilon (ε): Privacy budget - lower = more private
- Delta (δ): Probability of privacy breach
- Sensitivity: Maximum change from one individual
- Noise mechanisms: Laplace, Gaussian, Exponential

Use cases:
- Private learning analytics aggregation
- Privacy-preserving model updates
- Secure data sharing for research
"""
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import math
import random
import numpy as np


class NoiseMechanism(str, Enum):
    """Noise addition mechanisms"""
    LAPLACE = "laplace"        # For ε-DP
    GAUSSIAN = "gaussian"      # For (ε,δ)-DP
    EXPONENTIAL = "exponential"  # For discrete outputs


@dataclass
class PrivacyBudget:
    """Privacy budget tracking"""
    epsilon: float = 1.0       # Privacy parameter (lower = more private)
    delta: float = 1e-5        # Probability of privacy failure
    epsilon_spent: float = 0.0
    delta_spent: float = 0.0

    @property
    def epsilon_remaining(self) -> float:
        return max(0, self.epsilon - self.epsilon_spent)

    @property
    def delta_remaining(self) -> float:
        return max(0, self.delta - self.delta_spent)

    @property
    def is_exhausted(self) -> bool:
        return self.epsilon_remaining <= 0 or self.delta_remaining <= 0

    def can_spend(self, epsilon: float, delta: float = 0) -> bool:
        """Check if we can spend this much budget"""
        return (self.epsilon_remaining >= epsilon and
                self.delta_remaining >= delta)

    def spend(self, epsilon: float, delta: float = 0):
        """Spend privacy budget"""
        self.epsilon_spent += epsilon
        self.delta_spent += delta


@dataclass
class PrivateStatistics:
    """Container for differentially private statistics"""
    value: float
    true_value: Optional[float] = None  # For debugging only, never expose
    epsilon_used: float = 0.0
    mechanism: NoiseMechanism = NoiseMechanism.LAPLACE
    noise_scale: float = 0.0
    confidence_interval: Tuple[float, float] = (0, 0)


class DifferentialPrivacy:
    """
    Core differential privacy implementation

    Provides mechanisms for adding calibrated noise to achieve
    ε-differential privacy guarantees.
    """

    @staticmethod
    def laplace_mechanism(
        value: float,
        sensitivity: float,
        epsilon: float,
    ) -> PrivateStatistics:
        """
        Laplace mechanism for ε-differential privacy

        Args:
            value: True value
            sensitivity: L1 sensitivity (max change from one record)
            epsilon: Privacy parameter

        Returns:
            Noisy value with privacy guarantees
        """
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)
        noisy_value = value + noise

        # 95% confidence interval
        ci_width = scale * math.log(20)  # log(1/0.05)
        ci = (noisy_value - ci_width, noisy_value + ci_width)

        return PrivateStatistics(
            value=noisy_value,
            true_value=value,  # For internal use only
            epsilon_used=epsilon,
            mechanism=NoiseMechanism.LAPLACE,
            noise_scale=scale,
            confidence_interval=ci,
        )

    @staticmethod
    def gaussian_mechanism(
        value: float,
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> PrivateStatistics:
        """
        Gaussian mechanism for (ε,δ)-differential privacy

        Args:
            value: True value
            sensitivity: L2 sensitivity
            epsilon: Privacy parameter
            delta: Privacy failure probability

        Returns:
            Noisy value with privacy guarantees
        """
        # Calculate sigma for Gaussian noise
        sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        noise = np.random.normal(0, sigma)
        noisy_value = value + noise

        # 95% confidence interval
        ci_width = 1.96 * sigma
        ci = (noisy_value - ci_width, noisy_value + ci_width)

        return PrivateStatistics(
            value=noisy_value,
            true_value=value,
            epsilon_used=epsilon,
            mechanism=NoiseMechanism.GAUSSIAN,
            noise_scale=sigma,
            confidence_interval=ci,
        )

    @staticmethod
    def exponential_mechanism(
        scores: Dict[str, float],
        sensitivity: float,
        epsilon: float,
    ) -> Tuple[str, PrivateStatistics]:
        """
        Exponential mechanism for selecting from discrete options

        Args:
            scores: Dictionary of option -> utility score
            sensitivity: Sensitivity of utility function
            epsilon: Privacy parameter

        Returns:
            (Selected option, Privacy statistics)
        """
        # Calculate selection probabilities
        options = list(scores.keys())
        utilities = [scores[opt] for opt in options]

        # Compute exponential weights
        weights = [
            math.exp(epsilon * u / (2 * sensitivity))
            for u in utilities
        ]
        total = sum(weights)
        probabilities = [w / total for w in weights]

        # Sample according to probabilities
        selected_idx = np.random.choice(len(options), p=probabilities)
        selected = options[selected_idx]

        return selected, PrivateStatistics(
            value=scores[selected],
            epsilon_used=epsilon,
            mechanism=NoiseMechanism.EXPONENTIAL,
            noise_scale=sensitivity,
        )

    @staticmethod
    def clip_value(value: float, lower: float, upper: float) -> float:
        """Clip value to bounds (for sensitivity control)"""
        return max(lower, min(upper, value))

    @staticmethod
    def compute_sensitivity(
        func: Callable,
        sample_data: List[float],
        clip_bounds: Tuple[float, float] = None,
    ) -> float:
        """
        Estimate sensitivity of a function

        Args:
            func: Function to compute sensitivity for
            sample_data: Sample data points
            clip_bounds: Optional bounds for clipping

        Returns:
            Estimated L1 sensitivity
        """
        if not sample_data:
            return 1.0

        # Clip if bounds provided
        if clip_bounds:
            sample_data = [
                DifferentialPrivacy.clip_value(x, clip_bounds[0], clip_bounds[1])
                for x in sample_data
            ]

        # Estimate by leaving one out
        base_result = func(sample_data)
        max_change = 0

        for i in range(len(sample_data)):
            # Leave one out
            subset = sample_data[:i] + sample_data[i+1:]
            if subset:
                subset_result = func(subset)
                change = abs(base_result - subset_result)
                max_change = max(max_change, change)

        return max_change if max_change > 0 else 1.0


class PrivateAggregator:
    """
    Aggregates data with differential privacy

    Useful for computing statistics over user data while
    preserving individual privacy.
    """

    def __init__(
        self,
        budget: PrivacyBudget,
        default_clip_bounds: Tuple[float, float] = (0, 100),
    ):
        self.budget = budget
        self.default_clip_bounds = default_clip_bounds

    def private_count(
        self,
        data: List[Any],
        epsilon: Optional[float] = None,
    ) -> PrivateStatistics:
        """
        Private count of records

        Sensitivity = 1 (adding/removing one person changes count by 1)
        """
        epsilon = epsilon or 0.1
        if not self.budget.can_spend(epsilon):
            raise ValueError("Insufficient privacy budget")

        true_count = len(data)
        result = DifferentialPrivacy.laplace_mechanism(
            value=float(true_count),
            sensitivity=1.0,
            epsilon=epsilon,
        )

        self.budget.spend(epsilon)
        result.value = max(0, round(result.value))  # Count should be non-negative integer

        return result

    def private_sum(
        self,
        values: List[float],
        epsilon: Optional[float] = None,
        clip_bounds: Optional[Tuple[float, float]] = None,
    ) -> PrivateStatistics:
        """
        Private sum with clipping

        Sensitivity = upper_bound - lower_bound
        """
        epsilon = epsilon or 0.1
        bounds = clip_bounds or self.default_clip_bounds

        if not self.budget.can_spend(epsilon):
            raise ValueError("Insufficient privacy budget")

        # Clip values
        clipped = [
            DifferentialPrivacy.clip_value(v, bounds[0], bounds[1])
            for v in values
        ]

        sensitivity = bounds[1] - bounds[0]
        true_sum = sum(clipped)

        result = DifferentialPrivacy.laplace_mechanism(
            value=true_sum,
            sensitivity=sensitivity,
            epsilon=epsilon,
        )

        self.budget.spend(epsilon)
        return result

    def private_mean(
        self,
        values: List[float],
        epsilon: Optional[float] = None,
        clip_bounds: Optional[Tuple[float, float]] = None,
    ) -> PrivateStatistics:
        """
        Private mean with clipping

        Uses composition: sum + count
        """
        epsilon = epsilon or 0.2
        bounds = clip_bounds or self.default_clip_bounds

        if not self.budget.can_spend(epsilon):
            raise ValueError("Insufficient privacy budget")

        # Split budget between sum and count
        epsilon_sum = epsilon * 0.7
        epsilon_count = epsilon * 0.3

        private_sum = self.private_sum(values, epsilon_sum, bounds)
        private_count = self.private_count(values, epsilon_count)

        # Compute mean
        if private_count.value > 0:
            mean_value = private_sum.value / private_count.value
        else:
            mean_value = 0

        # Clip to bounds
        mean_value = DifferentialPrivacy.clip_value(mean_value, bounds[0], bounds[1])

        return PrivateStatistics(
            value=mean_value,
            epsilon_used=epsilon,
            mechanism=NoiseMechanism.LAPLACE,
            noise_scale=private_sum.noise_scale,
        )

    def private_histogram(
        self,
        values: List[float],
        bins: List[float],
        epsilon: Optional[float] = None,
    ) -> Dict[str, PrivateStatistics]:
        """
        Private histogram

        Each bin count is perturbed independently
        """
        epsilon = epsilon or 0.5
        epsilon_per_bin = epsilon / len(bins)

        if not self.budget.can_spend(epsilon):
            raise ValueError("Insufficient privacy budget")

        # Compute true histogram
        histogram = {}
        for i in range(len(bins) - 1):
            bin_name = f"{bins[i]}-{bins[i+1]}"
            count = sum(1 for v in values if bins[i] <= v < bins[i+1])
            histogram[bin_name] = count

        # Add noise to each bin
        private_histogram = {}
        for bin_name, count in histogram.items():
            result = DifferentialPrivacy.laplace_mechanism(
                value=float(count),
                sensitivity=1.0,
                epsilon=epsilon_per_bin,
            )
            result.value = max(0, round(result.value))
            private_histogram[bin_name] = result

        self.budget.spend(epsilon)
        return private_histogram

    def private_percentile(
        self,
        values: List[float],
        percentile: float,
        epsilon: Optional[float] = None,
    ) -> PrivateStatistics:
        """
        Private percentile using exponential mechanism

        Args:
            values: Data values
            percentile: Desired percentile (0-100)
            epsilon: Privacy parameter
        """
        epsilon = epsilon or 0.2

        if not self.budget.can_spend(epsilon):
            raise ValueError("Insufficient privacy budget")

        if not values:
            return PrivateStatistics(value=0, epsilon_used=epsilon)

        sorted_values = sorted(values)

        # Create candidate set (unique values)
        candidates = list(set(sorted_values))

        # Score each candidate by how close it is to desired percentile
        target_rank = int(len(sorted_values) * percentile / 100)
        scores = {}

        for candidate in candidates:
            rank = sum(1 for v in sorted_values if v <= candidate)
            # Score = negative distance from target rank
            scores[str(candidate)] = -abs(rank - target_rank)

        selected, stats = DifferentialPrivacy.exponential_mechanism(
            scores=scores,
            sensitivity=1.0,  # Rank sensitivity
            epsilon=epsilon,
        )

        self.budget.spend(epsilon)

        return PrivateStatistics(
            value=float(selected),
            epsilon_used=epsilon,
            mechanism=NoiseMechanism.EXPONENTIAL,
        )


class PrivateLearningAnalytics:
    """
    Privacy-preserving learning analytics

    Provides common educational analytics with DP guarantees
    """

    def __init__(self, privacy_budget: PrivacyBudget = None):
        self.budget = privacy_budget or PrivacyBudget(epsilon=5.0, delta=1e-5)
        self.aggregator = PrivateAggregator(
            self.budget,
            default_clip_bounds=(0, 100),
        )

    def private_course_completion_rate(
        self,
        completion_flags: List[bool],
        epsilon: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Private course completion rate

        Args:
            completion_flags: List of completion status per user
            epsilon: Privacy budget for this query
        """
        # Convert to floats
        values = [1.0 if c else 0.0 for c in completion_flags]

        result = self.aggregator.private_mean(
            values,
            epsilon=epsilon,
            clip_bounds=(0, 1),
        )

        return {
            "completion_rate": round(result.value * 100, 1),
            "confidence_interval": (
                round(max(0, result.confidence_interval[0]) * 100, 1),
                round(min(1, result.confidence_interval[1]) * 100, 1),
            ),
            "privacy_cost": epsilon,
            "privacy_remaining": self.budget.epsilon_remaining,
        }

    def private_average_score(
        self,
        scores: List[float],
        epsilon: float = 0.2,
        max_score: float = 100,
    ) -> Dict[str, Any]:
        """
        Private average assessment score
        """
        result = self.aggregator.private_mean(
            scores,
            epsilon=epsilon,
            clip_bounds=(0, max_score),
        )

        return {
            "average_score": round(result.value, 1),
            "confidence_interval": (
                round(result.confidence_interval[0], 1),
                round(result.confidence_interval[1], 1),
            ),
            "privacy_cost": epsilon,
            "privacy_remaining": self.budget.epsilon_remaining,
        }

    def private_time_spent_distribution(
        self,
        minutes_list: List[float],
        epsilon: float = 0.3,
        max_minutes: float = 480,  # 8 hours max
    ) -> Dict[str, Any]:
        """
        Private distribution of time spent

        Returns histogram of time spent
        """
        bins = [0, 15, 30, 60, 120, 240, max_minutes]
        histogram = self.aggregator.private_histogram(
            [min(m, max_minutes) for m in minutes_list],
            bins=bins,
            epsilon=epsilon,
        )

        return {
            "distribution": {
                k: int(v.value) for k, v in histogram.items()
            },
            "privacy_cost": epsilon,
            "privacy_remaining": self.budget.epsilon_remaining,
        }

    def private_mastery_percentiles(
        self,
        mastery_levels: List[float],
        epsilon: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Private mastery level percentiles
        """
        percentiles = {}

        for p in [25, 50, 75]:
            result = self.aggregator.private_percentile(
                mastery_levels,
                percentile=p,
                epsilon=epsilon / 3,
            )
            percentiles[f"p{p}"] = round(result.value, 3)

        return {
            "percentiles": percentiles,
            "privacy_cost": epsilon,
            "privacy_remaining": self.budget.epsilon_remaining,
        }

    def get_privacy_report(self) -> Dict[str, Any]:
        """Get privacy budget usage report"""
        return {
            "total_epsilon": self.budget.epsilon,
            "epsilon_spent": round(self.budget.epsilon_spent, 4),
            "epsilon_remaining": round(self.budget.epsilon_remaining, 4),
            "total_delta": self.budget.delta,
            "delta_spent": self.budget.delta_spent,
            "budget_exhausted": self.budget.is_exhausted,
            "percentage_used": round(
                self.budget.epsilon_spent / self.budget.epsilon * 100, 1
            ),
        }
