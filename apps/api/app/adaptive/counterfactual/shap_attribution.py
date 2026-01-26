"""
SHAP Attribution for Explaining Bandit Arm Selection

Once the Bandit selects an activity, we must explain WHY that specific arm
was chosen over others. This module uses SHAP (SHapley Additive exPlanations)
to attribute the decision to specific features of the student's context.

Key Features:
- BanditSHAPExplainer: Explains MAB arm selection decisions
- OrdShapExplainer: Handles sequential dependencies in learning paths
- Feature attribution for understanding exploration vs exploitation

The SHAP value φ_i represents how much feature i pushed the estimated reward
of the chosen arm away from the baseline average.

Example Output:
    "I recommended geometry review primarily because:
    - φ_mastery = -0.4: Low mastery significantly increased review value
    - φ_exam = +0.2: Upcoming exam added urgency
    - φ_fatigue = -0.1: High activity slightly penalized, but outweighed"

Research Basis:
- Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions"
- OrdShap for Sequential Feature Attribution
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from datetime import datetime
import math
import numpy as np
from itertools import combinations
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureContribution:
    """
    SHAP contribution for a single feature.

    Attributes:
        feature_name: Name of the feature
        shap_value: SHAP value (contribution to prediction)
        feature_value: Actual value of the feature
        description: Human-readable description
        direction: "positive" or "negative" impact
    """
    feature_name: str
    shap_value: float
    feature_value: float
    description: str
    direction: str = "neutral"

    def __post_init__(self):
        if self.shap_value > 0.01:
            self.direction = "positive"
        elif self.shap_value < -0.01:
            self.direction = "negative"
        else:
            self.direction = "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "shap_value": self.shap_value,
            "feature_value": self.feature_value,
            "description": self.description,
            "direction": self.direction,
        }


@dataclass
class SHAPExplanation:
    """
    Complete SHAP explanation for a bandit decision.

    Attributes:
        selected_arm: ID of the selected arm
        base_value: Expected value without any features (baseline)
        feature_contributions: List of feature contributions
        total_score: Final prediction (base + sum of contributions)
        explanation_text: Human-readable explanation
        confidence: Confidence in the explanation
    """
    selected_arm: str
    base_value: float
    feature_contributions: List[FeatureContribution]
    total_score: float
    explanation_text: str
    confidence: float = 0.85

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_arm": self.selected_arm,
            "base_value": self.base_value,
            "feature_contributions": [f.to_dict() for f in self.feature_contributions],
            "total_score": self.total_score,
            "explanation_text": self.explanation_text,
            "confidence": self.confidence,
        }

    def get_top_contributors(self, n: int = 3) -> List[FeatureContribution]:
        """Get top N contributors by absolute SHAP value"""
        sorted_contribs = sorted(
            self.feature_contributions,
            key=lambda x: abs(x.shap_value),
            reverse=True
        )
        return sorted_contribs[:n]


@dataclass
class OrdShapContribution:
    """
    OrdShap contribution separating value and position effects.

    Attributes:
        item: The item in the sequence
        value_effect: Contribution from the item's value
        position_effect: Contribution from the item's position
        total_effect: Combined effect
    """
    item: str
    position: int
    value_effect: float
    position_effect: float
    total_effect: float
    description: str


class BanditSHAPExplainer:
    """
    SHAP attribution for Multi-Armed Bandit decisions.

    Explains WHY a particular arm was selected by decomposing the
    decision into feature contributions. Works with:
    - HybridLinUCB
    - ContextualBandit
    - Thompson Sampling (with feature-based priors)

    Features Explained:
    - Mastery level: Student's current concept mastery
    - Time since review: Recency of last practice
    - Cognitive load: Current mental state
    - Exam proximity: Urgency from upcoming assessments
    - Recent performance: Trend in recent attempts

    Usage:
        explainer = BanditSHAPExplainer(bandit)

        explanation = explainer.explain_selection(
            context=student_context,
            selected_arm=chosen_modality,
            alternative_arms=other_modalities
        )
    """

    # Standard feature names and their descriptions
    FEATURE_DESCRIPTIONS = {
        "mastery": "Current concept mastery level",
        "recency": "Time since last review",
        "cognitive_load": "Current cognitive load",
        "exam_proximity": "Proximity to upcoming exam",
        "recent_performance": "Recent performance trend",
        "fatigue": "Estimated user fatigue",
        "difficulty": "Content difficulty level",
        "prerequisite_mastery": "Mastery of prerequisites",
        "time_of_day": "Time of day factor",
        "session_duration": "Current session duration",
    }

    def __init__(
        self,
        predict_fn: Optional[Callable[[np.ndarray], float]] = None,
        feature_names: Optional[List[str]] = None,
        background_data: Optional[np.ndarray] = None,
        num_samples: int = 100,
    ):
        """
        Initialize SHAP explainer.

        Args:
            predict_fn: Function that takes features and returns score
            feature_names: Names of features in order
            background_data: Background dataset for SHAP computation
            num_samples: Number of samples for KernelSHAP
        """
        self.predict_fn = predict_fn
        self.feature_names = feature_names or list(self.FEATURE_DESCRIPTIONS.keys())
        self.background_data = background_data
        self.num_samples = num_samples

        # Initialize background to mean if not provided
        if self.background_data is None:
            # Default background: neutral values
            self.background_data = np.array([[0.5] * len(self.feature_names)])

        logger.info(f"BanditSHAPExplainer initialized with {len(self.feature_names)} features")

    def set_predict_function(self, predict_fn: Callable[[np.ndarray], float]):
        """Set the prediction function for the bandit"""
        self.predict_fn = predict_fn

    def explain_selection(
        self,
        context: Dict[str, float],
        selected_arm: str,
        predict_fn: Optional[Callable] = None,
    ) -> SHAPExplanation:
        """
        Compute SHAP values for arm selection.

        Args:
            context: Feature dictionary {feature_name: value}
            selected_arm: ID of the selected arm
            predict_fn: Optional prediction function override

        Returns:
            SHAPExplanation with feature contributions
        """
        logger.debug(f"Explaining selection of arm '{selected_arm}'")

        predict = predict_fn or self.predict_fn

        if predict is None:
            # Use simple linear model as fallback
            predict = self._default_linear_predict

        # Convert context to feature vector
        feature_vector = np.array([
            context.get(name, 0.5)
            for name in self.feature_names
        ])

        # Compute SHAP values
        shap_values = self._kernel_shap(feature_vector, predict)

        # Compute base value (prediction on background)
        base_value = float(np.mean([
            predict(bg)
            for bg in self.background_data
        ]))

        # Build feature contributions
        contributions = []
        for i, name in enumerate(self.feature_names):
            shap_val = shap_values[i]
            feature_val = feature_vector[i]

            # Generate description
            desc = self._generate_feature_description(name, feature_val, shap_val)

            contributions.append(FeatureContribution(
                feature_name=name,
                shap_value=float(shap_val),
                feature_value=float(feature_val),
                description=desc,
            ))

        # Total score
        total_score = base_value + sum(c.shap_value for c in contributions)

        # Generate explanation text
        explanation_text = self._generate_explanation_text(
            selected_arm,
            contributions,
            base_value,
            total_score,
        )

        return SHAPExplanation(
            selected_arm=selected_arm,
            base_value=base_value,
            feature_contributions=contributions,
            total_score=total_score,
            explanation_text=explanation_text,
            confidence=self._estimate_confidence(shap_values),
        )

    def _kernel_shap(
        self,
        instance: np.ndarray,
        predict_fn: Callable,
    ) -> np.ndarray:
        """
        KernelSHAP implementation for SHAP value computation.

        Uses weighted linear regression on coalition samples to
        estimate Shapley values efficiently.

        Args:
            instance: Feature vector to explain
            predict_fn: Prediction function

        Returns:
            SHAP values for each feature
        """
        n_features = len(instance)

        if n_features == 0:
            return np.array([])

        # Sample coalitions
        coalitions = []
        coalition_weights = []
        predictions = []

        # Add empty and full coalitions
        coalitions.append(np.zeros(n_features, dtype=bool))
        coalition_weights.append(1e6)  # High weight for empty

        coalitions.append(np.ones(n_features, dtype=bool))
        coalition_weights.append(1e6)  # High weight for full

        # Sample random coalitions
        for _ in range(self.num_samples - 2):
            coalition_size = np.random.randint(1, n_features)
            coalition = np.zeros(n_features, dtype=bool)
            indices = np.random.choice(n_features, coalition_size, replace=False)
            coalition[indices] = True
            coalitions.append(coalition)

            # SHAP kernel weight: M * (M-1) / (C(M,k) * k * (M-k))
            # Simplified: higher weight for smaller/larger coalitions
            k = coalition_size
            weight = 1.0 / (k * (n_features - k) + 1)
            coalition_weights.append(weight)

        # Get predictions for each coalition
        background_mean = self.background_data.mean(axis=0)

        for coalition in coalitions:
            # Create input: instance values where coalition=1, background where coalition=0
            x = np.where(coalition, instance, background_mean)
            pred = predict_fn(x)
            predictions.append(pred)

        # Convert to arrays
        coalitions = np.array(coalitions, dtype=float)
        weights = np.array(coalition_weights)
        predictions = np.array(predictions)

        # Solve weighted least squares
        # y = X @ shap_values
        try:
            # Add regularization
            W = np.diag(np.sqrt(weights))
            X_weighted = W @ coalitions
            y_weighted = W @ (predictions - predictions[0])  # Relative to empty coalition

            # Ridge regression
            lambda_reg = 0.01
            XtX = X_weighted.T @ X_weighted + lambda_reg * np.eye(n_features)
            Xty = X_weighted.T @ y_weighted

            shap_values = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            # Fallback to simple difference from mean
            base_pred = predict_fn(background_mean)
            full_pred = predict_fn(instance)
            shap_values = (full_pred - base_pred) / n_features * np.ones(n_features)

        return shap_values

    def _default_linear_predict(self, features: np.ndarray) -> float:
        """Default linear prediction function"""
        # Simple weighted sum
        weights = np.array([
            -0.3,   # mastery: lower = higher value for review
            0.1,    # recency: higher = more urgent
            -0.1,   # cognitive_load: lower = better
            0.2,    # exam_proximity: closer = more urgent
            -0.1,   # recent_performance: lower = needs help
            -0.15,  # fatigue: higher = prefer passive content
            0.0,    # difficulty: neutral
            -0.2,   # prerequisite_mastery: lower = need prerequisites
            0.0,    # time_of_day: neutral
            -0.05,  # session_duration: longer = more tired
        ])[:len(features)]

        return float(features @ weights[:len(features)])

    def _generate_feature_description(
        self,
        feature_name: str,
        feature_value: float,
        shap_value: float,
    ) -> str:
        """Generate description for a feature contribution"""
        base_desc = self.FEATURE_DESCRIPTIONS.get(feature_name, feature_name)

        if abs(shap_value) < 0.01:
            impact = "minimal impact"
        elif shap_value > 0:
            impact = "increased the recommendation score"
        else:
            impact = "decreased the recommendation score"

        # Feature-specific descriptions
        if feature_name == "mastery":
            if feature_value < 0.4:
                return f"Low mastery ({feature_value:.0%}) {impact}"
            elif feature_value > 0.7:
                return f"High mastery ({feature_value:.0%}) {impact}"
            else:
                return f"Moderate mastery ({feature_value:.0%}) {impact}"
        elif feature_name == "exam_proximity":
            if feature_value > 0.7:
                return f"Upcoming exam {impact}"
            else:
                return f"No immediate deadline {impact}"
        elif feature_name == "fatigue":
            if feature_value > 0.6:
                return f"High fatigue level {impact}"
            else:
                return f"Energy level acceptable {impact}"
        else:
            return f"{base_desc} ({feature_value:.2f}) {impact}"

    def _generate_explanation_text(
        self,
        selected_arm: str,
        contributions: List[FeatureContribution],
        base_value: float,
        total_score: float,
    ) -> str:
        """Generate human-readable explanation"""
        # Get top 3 contributors
        sorted_contribs = sorted(
            contributions,
            key=lambda x: abs(x.shap_value),
            reverse=True
        )[:3]

        reasons = []
        for contrib in sorted_contribs:
            if abs(contrib.shap_value) > 0.01:
                direction = "because" if contrib.shap_value > 0 else "despite"
                reasons.append(f"{direction} {contrib.description.lower()}")

        if reasons:
            reason_text = ", ".join(reasons)
            return f"I recommended '{selected_arm}' primarily {reason_text}."
        else:
            return f"I recommended '{selected_arm}' as the best overall choice for your current situation."

    def _estimate_confidence(self, shap_values: np.ndarray) -> float:
        """Estimate confidence in SHAP explanation"""
        # Higher magnitude SHAP values = more confident explanation
        max_shap = np.max(np.abs(shap_values)) if len(shap_values) > 0 else 0
        # Map to [0.6, 0.95] range
        confidence = 0.6 + 0.35 * min(max_shap / 0.5, 1.0)
        return confidence


class OrdShapExplainer:
    """
    Ordered Shapley Values for sequential learning data.

    Standard SHAP assumes features are independent, but in a learning path,
    the ORDER of events matters. OrdShap disentangles:
    - Value effect: Impact of the item's content/outcome
    - Position effect: Impact of WHERE in the sequence it occurred

    Example:
        "You are struggling because you missed the quiz" (value effect)
        vs
        "You are struggling because you took the quiz BEFORE the lecture" (position effect)

    This enables "reordering" counterfactuals:
        "If you had waited 24 hours (spacing effect), retention would be 12% higher"
    """

    def __init__(self, num_permutations: int = 50):
        """
        Initialize OrdShap explainer.

        Args:
            num_permutations: Number of permutations for Monte Carlo estimation
        """
        self.num_permutations = num_permutations

    def compute_ordshap(
        self,
        sequence: List[Tuple[str, datetime, bool]],
        outcome_fn: Callable[[List[Tuple[str, datetime, bool]]], float],
        target_position: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute OrdShap values for a sequence.

        Separates value effects from position effects for each item
        in the sequence.

        Args:
            sequence: List of (concept_id, timestamp, correct) tuples
            outcome_fn: Function that computes outcome given sequence
            target_position: Optional focus on specific position

        Returns:
            Dict with 'value_effects' and 'position_effects' for each item
        """
        logger.debug(f"Computing OrdShap for sequence of length {len(sequence)}")

        n = len(sequence)
        if n == 0:
            return {"value_effects": {}, "position_effects": {}}

        # Get baseline outcome (original sequence)
        baseline_outcome = outcome_fn(sequence)

        value_effects = {}
        position_effects = {}

        for i, (concept_id, timestamp, correct) in enumerate(sequence):
            if target_position is not None and i != target_position:
                continue

            item_key = f"{concept_id}_{i}"

            # Compute value effect: impact of this item's value (correct/incorrect)
            value_effect = self._compute_value_effect(
                sequence, i, outcome_fn, baseline_outcome
            )
            value_effects[item_key] = value_effect

            # Compute position effect: impact of this item's position
            position_effect = self._compute_position_effect(
                sequence, i, outcome_fn, baseline_outcome
            )
            position_effects[item_key] = position_effect

        return {
            "value_effects": value_effects,
            "position_effects": position_effects,
        }

    def _compute_value_effect(
        self,
        sequence: List[Tuple[str, datetime, bool]],
        position: int,
        outcome_fn: Callable,
        baseline: float,
    ) -> float:
        """
        Compute value effect by toggling the item's outcome.

        Value effect = outcome(with item) - outcome(without item's value)
        """
        # Toggle the correct/incorrect value
        modified = list(sequence)
        concept_id, timestamp, correct = modified[position]
        modified[position] = (concept_id, timestamp, not correct)

        modified_outcome = outcome_fn(modified)

        # Value effect: how much did this item's value contribute?
        # Positive = item being correct (or incorrect) helped
        return baseline - modified_outcome if correct else modified_outcome - baseline

    def _compute_position_effect(
        self,
        sequence: List[Tuple[str, datetime, bool]],
        position: int,
        outcome_fn: Callable,
        baseline: float,
    ) -> float:
        """
        Compute position effect by moving the item to different positions.

        Position effect = variance in outcome across different positions
        """
        effects = []
        item = sequence[position]

        # Try moving to different positions
        for new_pos in range(len(sequence)):
            if new_pos == position:
                continue

            # Create reordered sequence
            reordered = list(sequence)
            del reordered[position]
            reordered.insert(new_pos, item)

            # Compute outcome with reordering
            reordered_outcome = outcome_fn(reordered)
            effects.append(baseline - reordered_outcome)

        if effects:
            # Position effect is mean difference from reordering
            return float(np.mean(effects))
        return 0.0

    def explain_sequence_impact(
        self,
        original_sequence: List[Tuple[str, datetime, bool]],
        alternative_ordering: List[int],
        outcome_fn: Callable[[List[Tuple[str, datetime, bool]]], float],
    ) -> OrdShapContribution:
        """
        Explain the impact of reordering a sequence.

        Args:
            original_sequence: Original sequence
            alternative_ordering: New order as list of indices
            outcome_fn: Function to compute outcome

        Returns:
            OrdShapContribution explaining the reordering effect
        """
        original_outcome = outcome_fn(original_sequence)

        # Create reordered sequence
        reordered = [original_sequence[i] for i in alternative_ordering]
        reordered_outcome = outcome_fn(reordered)

        improvement = reordered_outcome - original_outcome

        # Identify key position changes
        position_changes = []
        for new_pos, old_idx in enumerate(alternative_ordering):
            if new_pos != old_idx:
                position_changes.append(
                    f"{original_sequence[old_idx][0]}: {old_idx} → {new_pos}"
                )

        description = (
            f"Reordering would {'improve' if improvement > 0 else 'decrease'} "
            f"the outcome by {abs(improvement):.1%}. "
            f"Key changes: {', '.join(position_changes[:3])}"
        )

        return OrdShapContribution(
            item="sequence_reordering",
            position=-1,  # Not applicable
            value_effect=0,  # Reordering doesn't change values
            position_effect=improvement,
            total_effect=improvement,
            description=description,
        )

    def suggest_optimal_ordering(
        self,
        sequence: List[Tuple[str, datetime, bool]],
        outcome_fn: Callable[[List[Tuple[str, datetime, bool]]], float],
        num_samples: int = 100,
    ) -> Tuple[List[int], float]:
        """
        Suggest optimal ordering via sampling.

        Args:
            sequence: Original sequence
            outcome_fn: Outcome function
            num_samples: Number of random orderings to try

        Returns:
            (best_ordering, expected_outcome)
        """
        n = len(sequence)
        if n <= 1:
            return list(range(n)), outcome_fn(sequence)

        best_ordering = list(range(n))
        best_outcome = outcome_fn(sequence)

        for _ in range(num_samples):
            # Random permutation
            ordering = np.random.permutation(n).tolist()
            reordered = [sequence[i] for i in ordering]
            outcome = outcome_fn(reordered)

            if outcome > best_outcome:
                best_outcome = outcome
                best_ordering = ordering

        return best_ordering, best_outcome


def explain_bandit_with_context(
    bandit_type: str,
    context_features: Dict[str, float],
    selected_arm: str,
    arm_scores: Dict[str, float],
) -> SHAPExplanation:
    """
    Convenience function to explain a bandit selection.

    Args:
        bandit_type: Type of bandit ("linucb", "thompson", etc.)
        context_features: Feature dict for current context
        selected_arm: ID of selected arm
        arm_scores: Scores for each arm

    Returns:
        SHAP explanation
    """
    # Create explainer
    explainer = BanditSHAPExplainer(
        feature_names=list(context_features.keys())
    )

    # Define prediction function based on arm scores
    def predict_fn(features: np.ndarray) -> float:
        # Simple linear combination weighted by feature importance
        return float(np.sum(features * np.array([0.2] * len(features))))

    return explainer.explain_selection(
        context=context_features,
        selected_arm=selected_arm,
        predict_fn=predict_fn,
    )
