"""
Algorithmic Recourse for Adaptive Learning

Explanation explains the past; recourse changes the future.

This module finds minimum-cost actions a student can take to improve their
predicted learning outcomes. It implements:
- Effort-aware distance metrics (some changes are harder than others)
- Immutable vs actionable feature constraints
- Time-constrained optimization
- Retrospective ("what went wrong") analysis
- Prospective ("what to do next") planning

Key Concepts:
- Validity: Counterfactual must achieve target outcome
- Proximity: Changes should be minimal (effort-aware)
- Feasibility: Changes must be possible (can't change past grades)

Optimization Problem:
    minimize effort_aware_distance(x, x')
    subject to:
        f(x') >= target_success_probability
        x' in feasible_set
        sum(action_time) <= time_budget

Research Basis:
- Counterfactual Explanations for Learning Paths (NerdLearn Research)
- Algorithmic Recourse: From Definitions to Practice (Karimi et al., 2021)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import math
import numpy as np
from scipy import optimize
import logging

from app.adaptive.counterfactual.counterfactual_engine import (
    CounterfactualEngine,
    CounterfactualQuery,
    CounterfactualResult,
    CounterfactualType,
)
from app.adaptive.counterfactual.scm import (
    StructuralCausalModel,
    ExogenousVariables,
    EndogenousVariables,
    ActionType,
)
from app.adaptive.td_bkt.temporal_difference_bkt import (
    BeliefState,
    ConceptState,
)

logger = logging.getLogger(__name__)


class FeatureType(str, Enum):
    """Types of features for recourse"""
    IMMUTABLE = "immutable"        # Cannot be changed (past grades, demographics)
    HIGH_EFFORT = "high_effort"    # Difficult to change (deep mastery)
    LOW_EFFORT = "low_effort"      # Easy to change (study time, hints)
    ACTIONABLE = "actionable"      # General actionable feature


@dataclass
class FeatureConstraint:
    """
    Constraint on a feature for recourse optimization.

    Defines what can and cannot be changed, and the effort cost
    of making changes.

    Attributes:
        feature_name: Name of the feature
        is_immutable: Cannot be changed at all
        is_actionable: Student can actively change this
        effort_weight: Cost multiplier for changing (1.0 = baseline)
        min_value: Minimum allowed value (if applicable)
        max_value: Maximum allowed value (if applicable)
        time_cost_per_unit: Minutes required per unit change
        description: Human-readable description
    """
    feature_name: str
    is_immutable: bool = False
    is_actionable: bool = True
    effort_weight: float = 1.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    time_cost_per_unit: float = 0.0  # Minutes per unit change
    description: str = ""

    def __post_init__(self):
        if self.is_immutable:
            self.is_actionable = False
            self.effort_weight = float('inf')


@dataclass
class RecourseAction:
    """
    A single actionable change in the recourse plan.

    Attributes:
        feature: Feature to change
        original_value: Current value
        target_value: Recommended new value
        effort_cost: Estimated effort (weighted)
        time_cost_minutes: Estimated time in minutes
        expected_impact: Expected change in success probability
        description: Human-readable action description
        priority: Action priority (1 = highest)
    """
    feature: str
    original_value: float
    target_value: float
    effort_cost: float
    time_cost_minutes: float
    expected_impact: float
    description: str
    priority: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "original_value": self.original_value,
            "target_value": self.target_value,
            "effort_cost": self.effort_cost,
            "time_cost_minutes": self.time_cost_minutes,
            "expected_impact": self.expected_impact,
            "description": self.description,
            "priority": self.priority,
        }


@dataclass
class RecoursePlan:
    """
    Complete recourse plan with multiple actions.

    Represents a set of changes the student can make to improve
    their predicted outcome.

    Attributes:
        actions: List of recommended actions
        total_effort: Total weighted effort cost
        estimated_time_minutes: Total estimated time
        success_probability_before: Current predicted success
        success_probability_after: Predicted success after actions
        confidence: Confidence in the estimate
        explanation: Human-readable explanation
        achievable: Whether target is achievable within constraints
        adjusted_target: Adjusted target if original not achievable
    """
    actions: List[RecourseAction]
    total_effort: float
    estimated_time_minutes: float
    success_probability_before: float
    success_probability_after: float
    confidence: float
    explanation: str
    achievable: bool = True
    adjusted_target: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actions": [a.to_dict() for a in self.actions],
            "total_effort": self.total_effort,
            "estimated_time_minutes": self.estimated_time_minutes,
            "success_probability_before": self.success_probability_before,
            "success_probability_after": self.success_probability_after,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "achievable": self.achievable,
            "adjusted_target": self.adjusted_target,
        }


@dataclass
class CriticalDecisionPoint:
    """A critical decision point in retrospective analysis"""
    timestamp: datetime
    concept_id: str
    action_taken: str
    alternative_action: str
    impact_on_outcome: float
    explanation: str


class AlgorithmicRecourse:
    """
    Finds minimum-cost actions to change predicted outcomes.

    Supports:
    - Effort-aware distance metrics
    - Immutable vs actionable feature constraints
    - Time-constrained optimization
    - Retrospective ("what went wrong") analysis
    - Prospective ("what to do next") planning

    Usage:
        recourse = AlgorithmicRecourse(counterfactual_engine, constraints)

        # Find minimum cost plan
        plan = recourse.find_minimum_cost_recourse(
            current_state=belief_state,
            target_success_probability=0.85,
            time_budget_minutes=60
        )

        # Retrospective analysis
        analysis = recourse.retrospective_analysis(user_id, assessment_id, history)

        # Prospective planning
        plan = recourse.prospective_planning(belief_state, concepts, deadline, time)
    """

    def __init__(
        self,
        counterfactual_engine: CounterfactualEngine,
        feature_constraints: Optional[Dict[str, FeatureConstraint]] = None,
    ):
        """
        Initialize algorithmic recourse.

        Args:
            counterfactual_engine: Engine for counterfactual computation
            feature_constraints: Constraints on features (uses defaults if None)
        """
        self.cf_engine = counterfactual_engine
        self.constraints = feature_constraints or self.default_learning_constraints()

        logger.info("AlgorithmicRecourse initialized")

    @staticmethod
    def default_learning_constraints() -> Dict[str, FeatureConstraint]:
        """
        Default constraints for educational recourse.

        Categories:
        - Immutable: past grades, demographics, prior history
        - High effort: deep conceptual mastery
        - Low effort: study time, hint usage, review sessions
        """
        return {
            # Immutable features (cannot change the past)
            "past_quiz_score": FeatureConstraint(
                feature_name="past_quiz_score",
                is_immutable=True,
                description="Previous quiz score (cannot be changed)",
            ),
            "prior_attempts": FeatureConstraint(
                feature_name="prior_attempts",
                is_immutable=True,
                description="Number of previous attempts (historical)",
            ),

            # Low effort features (easy to change)
            "study_time_hours": FeatureConstraint(
                feature_name="study_time_hours",
                is_actionable=True,
                effort_weight=0.2,
                min_value=0,
                max_value=8,  # Max 8 hours in a session
                time_cost_per_unit=60,  # 1 hour = 60 minutes
                description="Additional study time in hours",
            ),
            "review_prerequisites": FeatureConstraint(
                feature_name="review_prerequisites",
                is_actionable=True,
                effort_weight=0.3,
                min_value=0,
                max_value=1,  # Binary: did review or not
                time_cost_per_unit=20,  # 20 min to review
                description="Review prerequisite concepts",
            ),
            "use_hints": FeatureConstraint(
                feature_name="use_hints",
                is_actionable=True,
                effort_weight=0.1,
                min_value=0,
                max_value=1,
                time_cost_per_unit=2,  # 2 min per hint usage
                description="Use available hints and scaffolding",
            ),
            "watch_video": FeatureConstraint(
                feature_name="watch_video",
                is_actionable=True,
                effort_weight=0.25,
                min_value=0,
                max_value=1,
                time_cost_per_unit=15,  # 15 min video
                description="Watch instructional video",
            ),

            # Medium effort features
            "practice_problems": FeatureConstraint(
                feature_name="practice_problems",
                is_actionable=True,
                effort_weight=0.5,
                min_value=0,
                max_value=20,
                time_cost_per_unit=5,  # 5 min per problem
                description="Complete additional practice problems",
            ),
            "spacing_days": FeatureConstraint(
                feature_name="spacing_days",
                is_actionable=True,
                effort_weight=0.3,
                min_value=0,
                max_value=7,
                time_cost_per_unit=0,  # No direct time cost, but requires waiting
                description="Space practice over multiple days",
            ),

            # High effort features (harder to change quickly)
            "concept_mastery": FeatureConstraint(
                feature_name="concept_mastery",
                is_actionable=True,
                effort_weight=1.0,
                min_value=0,
                max_value=1,
                time_cost_per_unit=120,  # 2 hours to improve mastery by 1.0
                description="Deepen conceptual understanding",
            ),
            "prerequisite_mastery": FeatureConstraint(
                feature_name="prerequisite_mastery",
                is_actionable=True,
                effort_weight=1.5,
                min_value=0,
                max_value=1,
                time_cost_per_unit=180,  # 3 hours for prerequisite mastery
                description="Master prerequisite concepts first",
            ),
        }

    def effort_aware_distance(
        self,
        original: Dict[str, float],
        counterfactual: Dict[str, float],
    ) -> float:
        """
        Weighted distance metric accounting for effort.

        d(x, x') = sum(w_i * |x_i - x'_i|) for actionable features
        Returns infinity if immutable features differ.

        Args:
            original: Original feature values
            counterfactual: Counterfactual feature values

        Returns:
            Weighted distance (infinity if infeasible)
        """
        total_distance = 0.0

        for feature, orig_value in original.items():
            if feature not in counterfactual:
                continue

            cf_value = counterfactual[feature]
            diff = abs(cf_value - orig_value)

            if diff < 1e-6:
                continue  # No change

            if feature in self.constraints:
                constraint = self.constraints[feature]

                # Check if immutable
                if constraint.is_immutable:
                    return float('inf')

                # Check bounds
                if constraint.min_value is not None and cf_value < constraint.min_value:
                    return float('inf')
                if constraint.max_value is not None and cf_value > constraint.max_value:
                    return float('inf')

                # Weighted distance
                total_distance += constraint.effort_weight * diff
            else:
                # Unknown feature - use default weight
                total_distance += diff

        return total_distance

    def compute_time_cost(
        self,
        original: Dict[str, float],
        counterfactual: Dict[str, float],
    ) -> float:
        """
        Compute total time cost for changes.

        Args:
            original: Original feature values
            counterfactual: Counterfactual feature values

        Returns:
            Total time cost in minutes
        """
        total_time = 0.0

        for feature, orig_value in original.items():
            if feature not in counterfactual:
                continue

            cf_value = counterfactual[feature]
            diff = abs(cf_value - orig_value)

            if feature in self.constraints:
                constraint = self.constraints[feature]
                total_time += constraint.time_cost_per_unit * diff

        return total_time

    def find_minimum_cost_recourse(
        self,
        current_state: BeliefState,
        concept_id: str,
        target_success_probability: float = 0.85,
        time_budget_minutes: Optional[float] = None,
        max_iterations: int = 100,
    ) -> RecoursePlan:
        """
        Find minimum-cost recourse plan.

        Optimization problem:
            minimize effort_aware_distance(x, x')
            subject to:
                P(success | x') >= target
                x' in feasible_set
                time_cost(x') <= time_budget

        Args:
            current_state: Current belief state
            concept_id: Concept to improve
            target_success_probability: Target success probability
            time_budget_minutes: Maximum time available (None = unlimited)
            max_iterations: Maximum optimization iterations

        Returns:
            RecoursePlan with recommended actions
        """
        logger.info(f"Finding recourse for concept {concept_id}, target={target_success_probability:.0%}")

        # Get current success probability
        current_mastery = current_state.get_concept_mastery(concept_id)
        current_success = self._estimate_success_probability(current_mastery)

        if current_success >= target_success_probability:
            # Already at target
            return RecoursePlan(
                actions=[],
                total_effort=0.0,
                estimated_time_minutes=0.0,
                success_probability_before=current_success,
                success_probability_after=current_success,
                confidence=1.0,
                explanation="You're already on track to meet your goal!",
                achievable=True,
            )

        # Define actionable features to optimize
        actionable_features = {
            name: constraint
            for name, constraint in self.constraints.items()
            if constraint.is_actionable and not constraint.is_immutable
        }

        # Current feature values
        current_features = self._extract_features(current_state, concept_id)

        # Optimization: find minimum effort change to achieve target
        best_plan = None
        best_effort = float('inf')

        # Try different combinations of actions
        for feature_name, constraint in actionable_features.items():
            # Test incremental changes
            for delta in [0.25, 0.5, 1.0, 2.0]:
                if constraint.max_value is not None:
                    target_value = min(
                        current_features.get(feature_name, 0) + delta,
                        constraint.max_value
                    )
                else:
                    target_value = current_features.get(feature_name, 0) + delta

                # Compute counterfactual
                cf_features = current_features.copy()
                cf_features[feature_name] = target_value

                # Estimate new success probability
                new_success = self._estimate_success_from_features(cf_features, current_mastery)

                # Check constraints
                effort = self.effort_aware_distance(current_features, cf_features)
                time_cost = self.compute_time_cost(current_features, cf_features)

                if time_budget_minutes is not None and time_cost > time_budget_minutes:
                    continue

                if new_success >= target_success_probability and effort < best_effort:
                    best_effort = effort
                    best_plan = {
                        "features": cf_features,
                        "success": new_success,
                        "effort": effort,
                        "time": time_cost,
                        "primary_action": feature_name,
                        "primary_delta": target_value - current_features.get(feature_name, 0),
                    }

        # Build recourse plan
        if best_plan is None:
            # Target not achievable within constraints
            # Find best achievable outcome
            best_plan = self._find_best_achievable(
                current_features,
                current_mastery,
                time_budget_minutes,
                actionable_features,
            )

            return RecoursePlan(
                actions=self._build_actions(current_features, best_plan["features"]),
                total_effort=best_plan["effort"],
                estimated_time_minutes=best_plan["time"],
                success_probability_before=current_success,
                success_probability_after=best_plan["success"],
                confidence=0.7,  # Lower confidence when target not met
                explanation=self._generate_adjusted_explanation(
                    target_success_probability,
                    best_plan["success"],
                    time_budget_minutes,
                ),
                achievable=False,
                adjusted_target=best_plan["success"],
            )

        return RecoursePlan(
            actions=self._build_actions(current_features, best_plan["features"]),
            total_effort=best_plan["effort"],
            estimated_time_minutes=best_plan["time"],
            success_probability_before=current_success,
            success_probability_after=best_plan["success"],
            confidence=0.85,
            explanation=self._generate_explanation(best_plan),
            achievable=True,
        )

    def _estimate_success_probability(self, mastery: float) -> float:
        """Estimate success probability from mastery"""
        # BKT-style: P(success) = mastery * (1 - slip) + (1 - mastery) * guess
        p_slip = 0.1
        p_guess = 0.2
        return mastery * (1 - p_slip) + (1 - mastery) * p_guess

    def _estimate_success_from_features(
        self,
        features: Dict[str, float],
        base_mastery: float,
    ) -> float:
        """Estimate success probability from feature changes"""
        # Model how features improve mastery
        mastery_boost = 0.0

        # Study time improves mastery
        study_hours = features.get("study_time_hours", 0)
        mastery_boost += 0.05 * study_hours

        # Practice problems improve mastery
        problems = features.get("practice_problems", 0)
        mastery_boost += 0.02 * problems

        # Prerequisites help
        prereq_review = features.get("review_prerequisites", 0)
        mastery_boost += 0.1 * prereq_review

        # Video watching helps
        video = features.get("watch_video", 0)
        mastery_boost += 0.08 * video

        # Hints help somewhat
        hints = features.get("use_hints", 0)
        mastery_boost += 0.03 * hints

        # Spacing helps retention
        spacing = features.get("spacing_days", 0)
        spacing_bonus = 0.02 * min(spacing, 3)  # Up to 3 days helps
        mastery_boost += spacing_bonus

        # New mastery (capped at 1.0)
        new_mastery = min(1.0, base_mastery + mastery_boost)

        return self._estimate_success_probability(new_mastery)

    def _extract_features(
        self,
        belief_state: BeliefState,
        concept_id: str,
    ) -> Dict[str, float]:
        """Extract current feature values from belief state"""
        # Default current features (all at minimum)
        return {
            "study_time_hours": 0.0,
            "review_prerequisites": 0.0,
            "use_hints": 0.0,
            "watch_video": 0.0,
            "practice_problems": 0.0,
            "spacing_days": 0.0,
        }

    def _find_best_achievable(
        self,
        current_features: Dict[str, float],
        current_mastery: float,
        time_budget: Optional[float],
        actionable_features: Dict[str, FeatureConstraint],
    ) -> Dict[str, Any]:
        """Find best achievable outcome within constraints"""
        best_features = current_features.copy()
        total_time = 0.0
        total_effort = 0.0

        # Greedily add features up to budget
        for feature_name, constraint in sorted(
            actionable_features.items(),
            key=lambda x: x[1].effort_weight
        ):
            if constraint.max_value is None:
                continue

            current_val = current_features.get(feature_name, 0)
            max_val = constraint.max_value

            # How much can we add within time budget?
            if time_budget is not None:
                max_delta = (time_budget - total_time) / (constraint.time_cost_per_unit + 0.001)
                max_delta = max(0, min(max_delta, max_val - current_val))
            else:
                max_delta = max_val - current_val

            if max_delta > 0:
                best_features[feature_name] = current_val + max_delta
                total_time += constraint.time_cost_per_unit * max_delta
                total_effort += constraint.effort_weight * max_delta

        best_success = self._estimate_success_from_features(best_features, current_mastery)

        return {
            "features": best_features,
            "success": best_success,
            "effort": total_effort,
            "time": total_time,
        }

    def _build_actions(
        self,
        original: Dict[str, float],
        counterfactual: Dict[str, float],
    ) -> List[RecourseAction]:
        """Build action list from feature changes"""
        actions = []
        priority = 1

        for feature, orig_val in original.items():
            if feature not in counterfactual:
                continue

            cf_val = counterfactual[feature]
            diff = cf_val - orig_val

            if abs(diff) < 0.01:
                continue

            if feature in self.constraints:
                constraint = self.constraints[feature]
                effort = constraint.effort_weight * abs(diff)
                time_cost = constraint.time_cost_per_unit * abs(diff)
                description = self._action_description(feature, orig_val, cf_val, constraint)
            else:
                effort = abs(diff)
                time_cost = 0
                description = f"Change {feature} from {orig_val:.2f} to {cf_val:.2f}"

            # Estimate impact (simplified)
            impact = 0.05 * abs(diff)

            actions.append(RecourseAction(
                feature=feature,
                original_value=orig_val,
                target_value=cf_val,
                effort_cost=effort,
                time_cost_minutes=time_cost,
                expected_impact=impact,
                description=description,
                priority=priority,
            ))
            priority += 1

        # Sort by effort (lowest first)
        actions.sort(key=lambda a: a.effort_cost)

        # Reassign priorities
        for i, action in enumerate(actions):
            action.priority = i + 1

        return actions

    def _action_description(
        self,
        feature: str,
        orig_val: float,
        cf_val: float,
        constraint: FeatureConstraint,
    ) -> str:
        """Generate human-readable action description"""
        diff = cf_val - orig_val

        if feature == "study_time_hours":
            if diff >= 1:
                return f"Study for {diff:.0f} more hours"
            else:
                return f"Study for {diff*60:.0f} more minutes"
        elif feature == "practice_problems":
            return f"Complete {diff:.0f} additional practice problems"
        elif feature == "review_prerequisites":
            return "Review the prerequisite concepts before proceeding"
        elif feature == "watch_video":
            return "Watch the instructional video for this topic"
        elif feature == "use_hints":
            return "Use the available hints and scaffolding"
        elif feature == "spacing_days":
            return f"Space your practice over {diff:.0f} days"
        else:
            return constraint.description or f"Adjust {feature}"

    def _generate_explanation(self, plan: Dict[str, Any]) -> str:
        """Generate explanation for recourse plan"""
        primary = plan.get("primary_action", "study")
        delta = plan.get("primary_delta", 0)
        success = plan.get("success", 0)

        explanations = {
            "study_time_hours": f"By studying {delta:.0f} more hours",
            "practice_problems": f"By completing {delta:.0f} more practice problems",
            "review_prerequisites": "By reviewing the prerequisite concepts",
            "watch_video": "By watching the instructional video",
            "use_hints": "By using the available hints",
            "spacing_days": f"By spacing your practice over {delta:.0f} days",
        }

        action_text = explanations.get(primary, f"By making these changes")

        return (
            f"{action_text}, you can improve your predicted success rate to "
            f"{success:.0%}. This is the most efficient path to reach your goal."
        )

    def _generate_adjusted_explanation(
        self,
        original_target: float,
        achievable: float,
        time_budget: Optional[float],
    ) -> str:
        """Generate explanation when target is not fully achievable"""
        if time_budget is not None:
            return (
                f"Within your {time_budget:.0f}-minute time budget, the maximum "
                f"achievable success rate is {achievable:.0%} (target was {original_target:.0%}). "
                f"Consider allocating more time or adjusting your goal."
            )
        else:
            return (
                f"The best achievable success rate is {achievable:.0%} "
                f"(target was {original_target:.0%}). "
                f"This may require more time or addressing foundational gaps."
            )

    def retrospective_analysis(
        self,
        user_id: str,
        assessment_id: str,
        interaction_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Retrospective recourse: "What went wrong?"

        Identifies critical decision points where different choices
        would have significantly altered the outcome.

        Args:
            user_id: User identifier
            assessment_id: Failed assessment identifier
            interaction_history: List of past interactions

        Returns:
            Analysis with critical decision points and recommendations
        """
        logger.info(f"Retrospective analysis for user {user_id}, assessment {assessment_id}")

        critical_points: List[CriticalDecisionPoint] = []
        root_causes: List[Dict[str, Any]] = []

        # Analyze interaction sequence
        for i, interaction in enumerate(interaction_history):
            # Look for missed opportunities
            concept_id = interaction.get("concept_id", "")
            action = interaction.get("action", "practice")
            correct = interaction.get("correct", False)
            timestamp = interaction.get("timestamp", datetime.now())

            if not correct:
                # This was a failure - was it predictable?
                mastery_before = interaction.get("mastery_before", 0.5)

                if mastery_before < 0.5:
                    # Low mastery before attempt - should have reviewed first
                    critical_points.append(CriticalDecisionPoint(
                        timestamp=timestamp,
                        concept_id=concept_id,
                        action_taken=action,
                        alternative_action="review_prerequisites",
                        impact_on_outcome=0.15,  # Estimated improvement
                        explanation=f"Your mastery of '{concept_id}' was low ({mastery_before:.0%}). "
                                   f"Reviewing prerequisites first would have helped.",
                    ))

            # Look for spacing issues
            if i > 0:
                prev_time = interaction_history[i-1].get("timestamp", datetime.now())
                gap_hours = (timestamp - prev_time).total_seconds() / 3600

                if gap_hours < 0.5:  # Less than 30 min between attempts
                    critical_points.append(CriticalDecisionPoint(
                        timestamp=timestamp,
                        concept_id=concept_id,
                        action_taken="cramming",
                        alternative_action="spacing",
                        impact_on_outcome=0.08,
                        explanation="Cramming (multiple attempts in quick succession) is less "
                                   "effective than spaced practice.",
                    ))

        # Identify root causes
        if len([p for p in critical_points if "prerequisite" in p.alternative_action]) > 2:
            root_causes.append({
                "factor": "missing_prerequisites",
                "contribution": 0.4,
                "evidence": "Multiple concepts attempted with insufficient prerequisite mastery",
            })

        if len([p for p in critical_points if "spacing" in p.alternative_action]) > 2:
            root_causes.append({
                "factor": "poor_spacing",
                "contribution": 0.25,
                "evidence": "Practice was crammed rather than spaced over time",
            })

        # Generate recommendations
        recommendations = []
        for cause in root_causes:
            if cause["factor"] == "missing_prerequisites":
                recommendations.append({
                    "action": "Review prerequisites before new concepts",
                    "expected_impact": 0.15,
                    "effort": "medium",
                })
            elif cause["factor"] == "poor_spacing":
                recommendations.append({
                    "action": "Space practice sessions over multiple days",
                    "expected_impact": 0.10,
                    "effort": "low",
                })

        return {
            "critical_decision_points": [
                {
                    "timestamp": p.timestamp.isoformat(),
                    "concept_id": p.concept_id,
                    "action_taken": p.action_taken,
                    "alternative_action": p.alternative_action,
                    "impact_on_outcome": p.impact_on_outcome,
                    "explanation": p.explanation,
                }
                for p in critical_points[:5]  # Top 5
            ],
            "root_causes": root_causes,
            "recommended_remediation": recommendations,
        }

    def prospective_planning(
        self,
        belief_state: BeliefState,
        target_concept_ids: List[str],
        deadline: datetime,
        available_time_minutes: float,
    ) -> RecoursePlan:
        """
        Prospective recourse: "What should I do next?"

        Finds optimal study allocation given time constraints.
        Calculates gradient of success probability with respect to
        time for each topic.

        Args:
            belief_state: Current belief state
            target_concept_ids: Concepts to master
            deadline: When mastery is needed
            available_time_minutes: Total available study time

        Returns:
            RecoursePlan with optimal time allocation
        """
        logger.info(f"Prospective planning for {len(target_concept_ids)} concepts, "
                   f"{available_time_minutes:.0f} min available")

        # Compute current success probability for each concept
        concept_priorities = []
        for concept_id in target_concept_ids:
            mastery = belief_state.get_concept_mastery(concept_id)
            success = self._estimate_success_probability(mastery)

            # Gradient: how much does success improve per hour of study?
            # Lower mastery = higher gradient (more room to improve)
            gradient = 0.1 * (1 - mastery)  # Simplified model

            concept_priorities.append({
                "concept_id": concept_id,
                "current_mastery": mastery,
                "current_success": success,
                "gradient": gradient,
            })

        # Sort by gradient (highest first)
        concept_priorities.sort(key=lambda x: x["gradient"], reverse=True)

        # Allocate time proportionally to gradient
        total_gradient = sum(c["gradient"] for c in concept_priorities)
        if total_gradient == 0:
            total_gradient = 1  # Avoid division by zero

        actions = []
        total_time_used = 0
        new_successes = {}

        for concept in concept_priorities:
            # Allocate time proportional to gradient
            allocated_time = (concept["gradient"] / total_gradient) * available_time_minutes
            allocated_time = min(allocated_time, available_time_minutes - total_time_used)

            if allocated_time < 5:  # Skip if less than 5 minutes
                continue

            total_time_used += allocated_time

            # Estimate improvement
            hours = allocated_time / 60
            mastery_gain = concept["gradient"] * hours
            new_mastery = min(1.0, concept["current_mastery"] + mastery_gain)
            new_success = self._estimate_success_probability(new_mastery)
            new_successes[concept["concept_id"]] = new_success

            actions.append(RecourseAction(
                feature=f"study_{concept['concept_id']}",
                original_value=0,
                target_value=hours,
                effort_cost=hours * 0.5,  # Medium effort
                time_cost_minutes=allocated_time,
                expected_impact=new_success - concept["current_success"],
                description=f"Study '{concept['concept_id']}' for {allocated_time:.0f} minutes",
                priority=len(actions) + 1,
            ))

        # Compute overall success
        if new_successes:
            overall_before = np.mean([c["current_success"] for c in concept_priorities])
            overall_after = np.mean(list(new_successes.values()))
        else:
            overall_before = 0.5
            overall_after = 0.5

        return RecoursePlan(
            actions=actions,
            total_effort=sum(a.effort_cost for a in actions),
            estimated_time_minutes=total_time_used,
            success_probability_before=overall_before,
            success_probability_after=overall_after,
            confidence=0.8,
            explanation=self._generate_prospective_explanation(actions, deadline),
            achievable=True,
        )

    def _generate_prospective_explanation(
        self,
        actions: List[RecourseAction],
        deadline: datetime,
    ) -> str:
        """Generate explanation for prospective plan"""
        if not actions:
            return "You're well prepared! Keep up the good work."

        total_time = sum(a.time_cost_minutes for a in actions)
        days_until = (deadline - datetime.now()).days

        return (
            f"Focus on {len(actions)} key areas over the next {days_until} days. "
            f"Total study time: {total_time:.0f} minutes. "
            f"Priority: {actions[0].description.split('Study ')[-1] if actions else 'review materials'}."
        )
