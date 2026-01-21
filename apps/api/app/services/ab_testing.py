"""
A/B Testing Framework

Provides feature flags and experiment management for:
- Feature rollouts
- UI/UX experiments
- Algorithm comparisons
- Content effectiveness testing

Features:
- Feature flags with targeting rules
- Experiment assignment and tracking
- Statistical significance calculation
- Gradual rollouts
- Segment targeting
"""

import hashlib
import logging
import math
from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import random
import json

logger = logging.getLogger(__name__)


# ==================== Enums and Data Classes ====================

class ExperimentStatus(str, Enum):
    """Experiment lifecycle status"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class VariantType(str, Enum):
    """Types of experiment variants"""
    CONTROL = "control"
    TREATMENT = "treatment"


class TargetingOperator(str, Enum):
    """Operators for targeting rules"""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    IN_LIST = "in_list"
    NOT_IN_LIST = "not_in_list"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    REGEX = "regex"
    PERCENTAGE = "percentage"


@dataclass
class TargetingRule:
    """Rule for targeting users to experiments/features"""
    attribute: str
    operator: TargetingOperator
    value: Any
    negate: bool = False


@dataclass
class Variant:
    """Experiment variant configuration"""
    id: str
    name: str
    variant_type: VariantType
    weight: float = 50.0  # Percentage of traffic
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Experiment:
    """A/B experiment configuration"""
    id: str
    name: str
    description: str
    status: ExperimentStatus
    variants: List[Variant]
    targeting_rules: List[TargetingRule] = field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    metrics: List[str] = field(default_factory=list)  # Metrics to track
    min_sample_size: int = 100
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FeatureFlag:
    """Feature flag configuration"""
    id: str
    name: str
    description: str
    enabled: bool
    targeting_rules: List[TargetingRule] = field(default_factory=list)
    default_value: Any = False
    variants: Dict[str, Any] = field(default_factory=dict)  # For multivariate flags
    rollout_percentage: float = 100.0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExperimentAssignment:
    """User's assignment to an experiment variant"""
    user_id: str
    experiment_id: str
    variant_id: str
    assigned_at: datetime
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Statistical results for an experiment"""
    experiment_id: str
    variant_id: str
    metric: str
    sample_size: int
    mean: float
    std_dev: float
    confidence_interval: tuple
    p_value: Optional[float] = None
    is_significant: bool = False
    lift: Optional[float] = None  # Percentage improvement over control


# ==================== Feature Flag Manager ====================

class FeatureFlagManager:
    """
    Manages feature flags with targeting rules.
    """

    def __init__(self):
        self._flags: Dict[str, FeatureFlag] = {}
        self._overrides: Dict[str, Dict[str, Any]] = {}  # user_id -> flag_id -> value

    def register_flag(self, flag: FeatureFlag):
        """Register a feature flag"""
        self._flags[flag.id] = flag
        logger.info(f"Registered feature flag: {flag.id}")

    def set_override(self, user_id: str, flag_id: str, value: Any):
        """Set user-specific override for a flag"""
        if user_id not in self._overrides:
            self._overrides[user_id] = {}
        self._overrides[user_id][flag_id] = value

    def clear_override(self, user_id: str, flag_id: str):
        """Clear user-specific override"""
        if user_id in self._overrides:
            self._overrides[user_id].pop(flag_id, None)

    def is_enabled(
        self,
        flag_id: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if feature flag is enabled for user.

        Args:
            flag_id: Feature flag ID
            user_id: Optional user ID for targeting
            context: Optional context for targeting rules

        Returns:
            Whether the flag is enabled
        """
        flag = self._flags.get(flag_id)
        if not flag:
            return False

        # Check override first
        if user_id and user_id in self._overrides:
            if flag_id in self._overrides[user_id]:
                return bool(self._overrides[user_id][flag_id])

        # Check if globally disabled
        if not flag.enabled:
            return flag.default_value

        # Check targeting rules
        context = context or {}
        if user_id:
            context["user_id"] = user_id

        if flag.targeting_rules:
            if not self._evaluate_rules(flag.targeting_rules, context):
                return flag.default_value

        # Check rollout percentage
        if flag.rollout_percentage < 100:
            if not self._in_rollout(user_id or "", flag_id, flag.rollout_percentage):
                return flag.default_value

        return True

    def get_value(
        self,
        flag_id: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        default: Any = None
    ) -> Any:
        """
        Get feature flag value (for multivariate flags).

        Args:
            flag_id: Feature flag ID
            user_id: Optional user ID
            context: Optional context
            default: Default value if flag not found

        Returns:
            Flag value
        """
        flag = self._flags.get(flag_id)
        if not flag:
            return default

        if not self.is_enabled(flag_id, user_id, context):
            return default

        # Return variant value if multivariate
        if flag.variants and user_id:
            variant_key = self._get_consistent_variant(user_id, flag_id, list(flag.variants.keys()))
            return flag.variants.get(variant_key, default)

        return True

    def _evaluate_rules(
        self,
        rules: List[TargetingRule],
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate targeting rules against context"""
        for rule in rules:
            attr_value = context.get(rule.attribute)
            result = self._evaluate_rule(rule, attr_value)

            if rule.negate:
                result = not result

            if not result:
                return False

        return True

    def _evaluate_rule(self, rule: TargetingRule, attr_value: Any) -> bool:
        """Evaluate a single targeting rule"""
        if attr_value is None:
            return False

        if rule.operator == TargetingOperator.EQUALS:
            return attr_value == rule.value
        elif rule.operator == TargetingOperator.NOT_EQUALS:
            return attr_value != rule.value
        elif rule.operator == TargetingOperator.CONTAINS:
            return str(rule.value) in str(attr_value)
        elif rule.operator == TargetingOperator.IN_LIST:
            return attr_value in rule.value
        elif rule.operator == TargetingOperator.NOT_IN_LIST:
            return attr_value not in rule.value
        elif rule.operator == TargetingOperator.GREATER_THAN:
            return float(attr_value) > float(rule.value)
        elif rule.operator == TargetingOperator.LESS_THAN:
            return float(attr_value) < float(rule.value)
        elif rule.operator == TargetingOperator.PERCENTAGE:
            # Hash-based percentage bucketing
            return self._in_rollout(str(attr_value), str(rule.attribute), float(rule.value))

        return False

    def _in_rollout(self, user_id: str, flag_id: str, percentage: float) -> bool:
        """Determine if user is in rollout percentage"""
        hash_input = f"{user_id}:{flag_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
        bucket = hash_value % 100
        return bucket < percentage

    def _get_consistent_variant(
        self,
        user_id: str,
        flag_id: str,
        variants: List[str]
    ) -> str:
        """Get consistent variant for user (deterministic assignment)"""
        hash_input = f"{user_id}:{flag_id}:variant"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
        index = hash_value % len(variants)
        return variants[index]

    def get_all_flags(self) -> List[Dict[str, Any]]:
        """Get all registered flags"""
        return [
            {
                "id": f.id,
                "name": f.name,
                "description": f.description,
                "enabled": f.enabled,
                "rollout_percentage": f.rollout_percentage
            }
            for f in self._flags.values()
        ]


# ==================== Experiment Manager ====================

class ExperimentManager:
    """
    Manages A/B experiments with statistical analysis.
    """

    def __init__(self):
        self._experiments: Dict[str, Experiment] = {}
        self._assignments: Dict[str, Dict[str, ExperimentAssignment]] = {}  # user -> exp -> assignment
        self._metrics: Dict[str, Dict[str, List[float]]] = {}  # exp -> variant -> metric values

    def create_experiment(self, experiment: Experiment):
        """Create a new experiment"""
        # Validate variant weights sum to 100
        total_weight = sum(v.weight for v in experiment.variants)
        if abs(total_weight - 100) > 0.01:
            raise ValueError(f"Variant weights must sum to 100, got {total_weight}")

        self._experiments[experiment.id] = experiment
        self._metrics[experiment.id] = {v.id: [] for v in experiment.variants}
        logger.info(f"Created experiment: {experiment.id}")

    def start_experiment(self, experiment_id: str):
        """Start an experiment"""
        exp = self._experiments.get(experiment_id)
        if exp:
            exp.status = ExperimentStatus.RUNNING
            exp.start_date = datetime.utcnow()
            exp.updated_at = datetime.utcnow()
            logger.info(f"Started experiment: {experiment_id}")

    def pause_experiment(self, experiment_id: str):
        """Pause an experiment"""
        exp = self._experiments.get(experiment_id)
        if exp:
            exp.status = ExperimentStatus.PAUSED
            exp.updated_at = datetime.utcnow()

    def complete_experiment(self, experiment_id: str):
        """Complete an experiment"""
        exp = self._experiments.get(experiment_id)
        if exp:
            exp.status = ExperimentStatus.COMPLETED
            exp.end_date = datetime.utcnow()
            exp.updated_at = datetime.utcnow()

    def get_variant(
        self,
        experiment_id: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Variant]:
        """
        Get experiment variant for user.

        Args:
            experiment_id: Experiment ID
            user_id: User ID
            context: Optional context for targeting

        Returns:
            Assigned variant or None if not eligible
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return None

        # Check if experiment is running
        if experiment.status != ExperimentStatus.RUNNING:
            return None

        # Check existing assignment
        if user_id in self._assignments:
            if experiment_id in self._assignments[user_id]:
                assignment = self._assignments[user_id][experiment_id]
                variant = next(
                    (v for v in experiment.variants if v.id == assignment.variant_id),
                    None
                )
                return variant

        # Check targeting rules
        context = context or {}
        context["user_id"] = user_id

        if experiment.targeting_rules:
            flag_manager = FeatureFlagManager()
            if not flag_manager._evaluate_rules(experiment.targeting_rules, context):
                return None

        # Assign variant
        variant = self._assign_variant(experiment, user_id)

        # Record assignment
        if user_id not in self._assignments:
            self._assignments[user_id] = {}

        self._assignments[user_id][experiment_id] = ExperimentAssignment(
            user_id=user_id,
            experiment_id=experiment_id,
            variant_id=variant.id,
            assigned_at=datetime.utcnow(),
            context=context
        )

        return variant

    def _assign_variant(self, experiment: Experiment, user_id: str) -> Variant:
        """Assign user to variant based on weights (deterministic)"""
        # Use hash for deterministic assignment
        hash_input = f"{user_id}:{experiment.id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
        bucket = hash_value % 100

        cumulative = 0
        for variant in experiment.variants:
            cumulative += variant.weight
            if bucket < cumulative:
                return variant

        return experiment.variants[-1]  # Fallback

    def record_metric(
        self,
        experiment_id: str,
        user_id: str,
        metric_name: str,
        value: float
    ):
        """
        Record a metric value for experiment analysis.

        Args:
            experiment_id: Experiment ID
            user_id: User ID
            metric_name: Name of the metric
            value: Metric value
        """
        if user_id not in self._assignments:
            return
        if experiment_id not in self._assignments[user_id]:
            return

        assignment = self._assignments[user_id][experiment_id]
        variant_id = assignment.variant_id

        if experiment_id not in self._metrics:
            self._metrics[experiment_id] = {}
        if variant_id not in self._metrics[experiment_id]:
            self._metrics[experiment_id][variant_id] = []

        self._metrics[experiment_id][variant_id].append(value)

    def get_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get experiment results with statistical analysis.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment results with statistical significance
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return {"error": "Experiment not found"}

        results = {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "status": experiment.status.value,
            "variants": []
        }

        # Get control variant
        control_variant = next(
            (v for v in experiment.variants if v.variant_type == VariantType.CONTROL),
            experiment.variants[0]
        )
        control_data = self._metrics.get(experiment_id, {}).get(control_variant.id, [])

        for variant in experiment.variants:
            variant_data = self._metrics.get(experiment_id, {}).get(variant.id, [])

            if not variant_data:
                results["variants"].append({
                    "variant_id": variant.id,
                    "name": variant.name,
                    "sample_size": 0,
                    "mean": None,
                    "std_dev": None
                })
                continue

            # Calculate statistics
            mean = sum(variant_data) / len(variant_data)
            variance = sum((x - mean) ** 2 for x in variant_data) / len(variant_data)
            std_dev = math.sqrt(variance)

            # Calculate confidence interval (95%)
            z = 1.96
            margin = z * std_dev / math.sqrt(len(variant_data)) if len(variant_data) > 0 else 0
            ci = (mean - margin, mean + margin)

            # Calculate p-value and lift compared to control
            p_value = None
            lift = None
            is_significant = False

            if variant.id != control_variant.id and control_data:
                control_mean = sum(control_data) / len(control_data)

                if control_mean > 0:
                    lift = ((mean - control_mean) / control_mean) * 100

                # Two-sample t-test (simplified)
                p_value = self._calculate_p_value(control_data, variant_data)
                is_significant = p_value < 0.05 if p_value else False

            results["variants"].append({
                "variant_id": variant.id,
                "name": variant.name,
                "type": variant.variant_type.value,
                "sample_size": len(variant_data),
                "mean": round(mean, 4),
                "std_dev": round(std_dev, 4),
                "confidence_interval": (round(ci[0], 4), round(ci[1], 4)),
                "p_value": round(p_value, 4) if p_value else None,
                "is_significant": is_significant,
                "lift": round(lift, 2) if lift else None
            })

        # Check if experiment has reached significance
        significant_variants = [
            v for v in results["variants"]
            if v.get("is_significant") and v.get("lift", 0) > 0
        ]
        results["has_winner"] = len(significant_variants) > 0
        results["winner"] = significant_variants[0] if significant_variants else None

        return results

    def _calculate_p_value(
        self,
        control_data: List[float],
        treatment_data: List[float]
    ) -> Optional[float]:
        """Calculate p-value using two-sample t-test"""
        if len(control_data) < 2 or len(treatment_data) < 2:
            return None

        # Calculate means and variances
        n1, n2 = len(control_data), len(treatment_data)
        mean1 = sum(control_data) / n1
        mean2 = sum(treatment_data) / n2
        var1 = sum((x - mean1) ** 2 for x in control_data) / (n1 - 1)
        var2 = sum((x - mean2) ** 2 for x in treatment_data) / (n2 - 1)

        # Pooled standard error
        se = math.sqrt(var1/n1 + var2/n2)

        if se == 0:
            return 1.0

        # T-statistic
        t = abs(mean2 - mean1) / se

        # Approximate p-value using normal distribution (for large samples)
        # For small samples, should use t-distribution
        # Using approximation: p ≈ 2 * (1 - Φ(|t|))
        p_value = 2 * (1 - self._normal_cdf(t))

        return max(0, min(1, p_value))

    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """Get all experiments"""
        return [
            {
                "id": e.id,
                "name": e.name,
                "status": e.status.value,
                "variants": len(e.variants),
                "start_date": e.start_date.isoformat() if e.start_date else None
            }
            for e in self._experiments.values()
        ]


# ==================== Singleton Instances ====================

feature_flag_manager = FeatureFlagManager()
experiment_manager = ExperimentManager()


# ==================== Default Feature Flags ====================

def register_default_flags():
    """Register default feature flags"""
    flags = [
        FeatureFlag(
            id="new_dashboard",
            name="New Dashboard UI",
            description="Enable new dashboard design",
            enabled=True,
            rollout_percentage=50
        ),
        FeatureFlag(
            id="ml_recommendations",
            name="ML-Based Recommendations",
            description="Use ML model for content recommendations",
            enabled=True,
            rollout_percentage=100
        ),
        FeatureFlag(
            id="audio_overviews",
            name="Audio Course Overviews",
            description="Enable audio generation for courses",
            enabled=True,
            rollout_percentage=25
        ),
        FeatureFlag(
            id="social_features",
            name="Social Gamification",
            description="Enable friends, challenges, and leaderboards",
            enabled=True,
            rollout_percentage=75
        ),
        FeatureFlag(
            id="advanced_analytics",
            name="Advanced Analytics Dashboard",
            description="Show advanced analytics to users",
            enabled=False,
            rollout_percentage=0
        )
    ]

    for flag in flags:
        feature_flag_manager.register_flag(flag)


# Initialize default flags
register_default_flags()
