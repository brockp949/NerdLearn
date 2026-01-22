"""
Zone of Proximal Development (ZPD) Regulator
The "Cognitive Thermostat" that maintains optimal challenge level

Based on Vygotsky's ZPD theory and Flow State research (Csikszentmihalyi)

Target: Maintain success rate between 35-70% (the "Goldilocks Zone")
- <35%: Frustration zone → provide scaffolding
- 35-70%: ZPD (optimal learning) → maintain
- >70%: Comfort zone → increase challenge
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum
import numpy as np
from collections import deque


class ZPDZone(Enum):
    """Current zone relative to ZPD"""
    FRUSTRATION = "frustration"  # Too hard, need scaffolding
    OPTIMAL = "optimal"  # Just right, maintain
    COMFORT = "comfort"  # Too easy, increase challenge


class ScaffoldingType(Enum):
    """Types of instructional scaffolding"""
    WORKED_EXAMPLE = "worked_example"  # Show complete solution
    PARTIAL_SOLUTION = "partial_solution"  # Provide hints/structure
    PREREQUISITE_REVIEW = "prerequisite_review"  # Review foundational concepts
    DIFFICULTY_REDUCTION = "difficulty_reduction"  # Simpler version of task
    COLLABORATIVE = "collaborative"  # Peer learning opportunity


class AdaptiveAction(Enum):
    """Actions the system can take to regulate difficulty"""
    PROVIDE_SCAFFOLDING = "provide_scaffolding"
    REMOVE_SCAFFOLDING = "remove_scaffolding"  # Fading
    INCREASE_DIFFICULTY = "increase_difficulty"
    DECREASE_DIFFICULTY = "decrease_difficulty"
    INTRODUCE_INTERLEAVING = "introduce_interleaving"
    ENABLE_BLOCKED_PRACTICE = "enable_blocked_practice"
    NO_ACTION = "no_action"


@dataclass
class PerformanceWindow:
    """Rolling window of recent performance"""
    attempts: deque  # Recent attempt results (bool)
    window_size: int = 10

    def __init__(self, window_size: int = 10):
        self.attempts = deque(maxlen=window_size)
        self.window_size = window_size

    def add_attempt(self, success: bool):
        """Add a new attempt result"""
        self.attempts.append(success)

    def get_success_rate(self) -> float:
        """Calculate current success rate"""
        if not self.attempts:
            return 0.5  # Neutral default

        return sum(1 for a in self.attempts if a) / len(self.attempts)

    def get_trend(self) -> str:
        """Detect trend: improving, declining, or stable"""
        if len(self.attempts) < 4:
            return "stable"

        # Split into first half and second half
        mid = len(self.attempts) // 2
        first_half = list(self.attempts)[:mid]
        second_half = list(self.attempts)[mid:]

        first_rate = sum(1 for a in first_half if a) / len(first_half)
        second_rate = sum(1 for a in second_half if a) / len(second_half)

        if second_rate - first_rate > 0.15:
            return "improving"
        elif first_rate - second_rate > 0.15:
            return "declining"
        else:
            return "stable"


@dataclass
class ZPDState:
    """Current ZPD state for a learner on a concept"""
    concept_id: str
    current_zone: ZPDZone
    success_rate: float
    trend: str
    recommended_actions: List[AdaptiveAction]
    confidence: float  # Confidence in assessment (0-1)


class ZPDRegulator:
    """
    Maintains learners in their Zone of Proximal Development

    This is the "thermostat" that continuously adjusts difficulty to
    maintain optimal challenge level, preventing both frustration and boredom.
    """

    def __init__(
        self,
        zpd_lower: float = 0.35,
        zpd_upper: float = 0.70,
        window_size: int = 10,
        min_attempts_for_confidence: int = 5
    ):
        """
        Args:
            zpd_lower: Lower bound of ZPD (below = frustration)
            zpd_upper: Upper bound of ZPD (above = comfort)
            window_size: Number of recent attempts to consider
            min_attempts_for_confidence: Minimum attempts before high confidence
        """
        self.zpd_lower = zpd_lower
        self.zpd_upper = zpd_upper
        self.window_size = window_size
        self.min_attempts = min_attempts_for_confidence

        # Track performance windows per (learner, concept)
        self.performance_windows: Dict[str, PerformanceWindow] = {}

        # Scaffolding state tracking
        self.active_scaffolds: Dict[str, List[ScaffoldingType]] = {}

    def record_attempt(
        self,
        learner_id: str,
        concept_id: str,
        success: bool
    ):
        """Record a learning attempt"""
        key = f"{learner_id}:{concept_id}"

        if key not in self.performance_windows:
            self.performance_windows[key] = PerformanceWindow(self.window_size)

        self.performance_windows[key].add_attempt(success)

    def assess_zpd_state(
        self,
        learner_id: str,
        concept_id: str
    ) -> ZPDState:
        """
        Assess current ZPD state and recommend actions

        This is the core regulation logic:
        1. Calculate success rate from recent attempts
        2. Determine which zone learner is in
        3. Recommend appropriate adaptive actions
        """
        key = f"{learner_id}:{concept_id}"

        # Get performance data
        if key not in self.performance_windows:
            # No data yet, assume optimal zone
            return ZPDState(
                concept_id=concept_id,
                current_zone=ZPDZone.OPTIMAL,
                success_rate=0.5,
                trend="stable",
                recommended_actions=[AdaptiveAction.NO_ACTION],
                confidence=0.0
            )

        window = self.performance_windows[key]
        success_rate = window.get_success_rate()
        trend = window.get_trend()

        # Calculate confidence (increases with more data)
        num_attempts = len(window.attempts)
        confidence = min(1.0, num_attempts / self.min_attempts)

        # Determine zone
        if success_rate < self.zpd_lower:
            zone = ZPDZone.FRUSTRATION
        elif success_rate > self.zpd_upper:
            zone = ZPDZone.COMFORT
        else:
            zone = ZPDZone.OPTIMAL

        # Recommend actions
        actions = self._determine_actions(
            zone,
            success_rate,
            trend,
            key
        )

        return ZPDState(
            concept_id=concept_id,
            current_zone=zone,
            success_rate=success_rate,
            trend=trend,
            recommended_actions=actions,
            confidence=confidence
        )

    def _determine_actions(
        self,
        zone: ZPDZone,
        success_rate: float,
        trend: str,
        key: str
    ) -> List[AdaptiveAction]:
        """
        Determine appropriate adaptive actions based on zone and trend

        Decision Logic:
        - Frustration Zone:
          - Very low (<25%): Strong scaffolding (worked examples)
          - Low (25-35%): Moderate scaffolding (hints)
          - Declining: Review prerequisites
        - Optimal Zone:
          - Stable: No action (maintain)
          - Improving: Prepare to fade scaffolding
        - Comfort Zone:
          - High (70-85%): Remove scaffolding, increase difficulty
          - Very high (>85%): Introduce interleaving, advance
        """
        actions = []

        if zone == ZPDZone.FRUSTRATION:
            if success_rate < 0.25:
                # Severe difficulty - strong intervention
                actions.append(AdaptiveAction.PROVIDE_SCAFFOLDING)
                actions.append(AdaptiveAction.DECREASE_DIFFICULTY)

                # Store scaffolding preference
                if key not in self.active_scaffolds:
                    self.active_scaffolds[key] = []
                if ScaffoldingType.WORKED_EXAMPLE not in self.active_scaffolds[key]:
                    self.active_scaffolds[key].append(ScaffoldingType.WORKED_EXAMPLE)

            elif trend == "declining":
                # Getting worse - review prerequisites
                actions.append(AdaptiveAction.PROVIDE_SCAFFOLDING)
                if key not in self.active_scaffolds:
                    self.active_scaffolds[key] = []
                if ScaffoldingType.PREREQUISITE_REVIEW not in self.active_scaffolds[key]:
                    self.active_scaffolds[key].append(ScaffoldingType.PREREQUISITE_REVIEW)

            else:
                # Struggling but not critically - moderate support
                actions.append(AdaptiveAction.PROVIDE_SCAFFOLDING)
                if key not in self.active_scaffolds:
                    self.active_scaffolds[key] = []
                if ScaffoldingType.PARTIAL_SOLUTION not in self.active_scaffolds[key]:
                    self.active_scaffolds[key].append(ScaffoldingType.PARTIAL_SOLUTION)

            # Use blocked practice in frustration zone
            actions.append(AdaptiveAction.ENABLE_BLOCKED_PRACTICE)

        elif zone == ZPDZone.OPTIMAL:
            # In the sweet spot - maintain current level
            if trend == "improving":
                # Prepare to fade scaffolding
                if key in self.active_scaffolds and self.active_scaffolds[key]:
                    actions.append(AdaptiveAction.REMOVE_SCAFFOLDING)
            else:
                actions.append(AdaptiveAction.NO_ACTION)

        else:  # COMFORT zone
            # Too easy - increase challenge
            if success_rate > 0.85:
                # Very high success - time for major challenge increase
                actions.append(AdaptiveAction.INCREASE_DIFFICULTY)
                actions.append(AdaptiveAction.INTRODUCE_INTERLEAVING)

                # Remove all scaffolding
                if key in self.active_scaffolds:
                    actions.append(AdaptiveAction.REMOVE_SCAFFOLDING)
                    self.active_scaffolds[key] = []

            else:
                # Moderately high - gradual increase
                actions.append(AdaptiveAction.REMOVE_SCAFFOLDING)
                if key in self.active_scaffolds:
                    # Fade gradually (remove one scaffold at a time)
                    if self.active_scaffolds[key]:
                        self.active_scaffolds[key].pop()

        return actions if actions else [AdaptiveAction.NO_ACTION]

    def get_active_scaffolding(
        self,
        learner_id: str,
        concept_id: str
    ) -> List[ScaffoldingType]:
        """Get currently active scaffolding for learner/concept"""
        key = f"{learner_id}:{concept_id}"
        return self.active_scaffolds.get(key, [])

    def apply_scaffolding(
        self,
        learner_id: str,
        concept_id: str,
        scaffold_type: ScaffoldingType
    ):
        """Manually apply scaffolding"""
        key = f"{learner_id}:{concept_id}"
        if key not in self.active_scaffolds:
            self.active_scaffolds[key] = []
        if scaffold_type not in self.active_scaffolds[key]:
            self.active_scaffolds[key].append(scaffold_type)

    def remove_scaffolding(
        self,
        learner_id: str,
        concept_id: str,
        scaffold_type: Optional[ScaffoldingType] = None
    ):
        """Remove scaffolding (all or specific type)"""
        key = f"{learner_id}:{concept_id}"
        if key not in self.active_scaffolds:
            return

        if scaffold_type is None:
            # Remove all
            self.active_scaffolds[key] = []
        else:
            # Remove specific
            if scaffold_type in self.active_scaffolds[key]:
                self.active_scaffolds[key].remove(scaffold_type)

    def get_optimal_difficulty(
        self,
        learner_id: str,
        concept_id: str,
        base_difficulty: float
    ) -> float:
        """
        Calculate optimal difficulty adjustment

        Args:
            learner_id: Learner ID
            concept_id: Concept ID
            base_difficulty: Base difficulty of content (1-10)

        Returns:
            Adjusted difficulty (1-10)
        """
        key = f"{learner_id}:{concept_id}"

        if key not in self.performance_windows:
            return base_difficulty

        success_rate = self.performance_windows[key].get_success_rate()

        # Calculate difficulty adjustment
        # Target: adjust so success rate moves toward ZPD midpoint (52.5%)
        target_rate = (self.zpd_lower + self.zpd_upper) / 2
        rate_diff = success_rate - target_rate

        # Difficulty adjustment (inverse relationship with success rate)
        # If doing too well (high success), increase difficulty
        # If struggling (low success), decrease difficulty
        adjustment = rate_diff * 2.0  # Scale factor

        new_difficulty = base_difficulty + adjustment
        return np.clip(new_difficulty, 1.0, 10.0)

    def reset_learner_state(self, learner_id: str, concept_id: str):
        """Reset state for learner/concept (e.g., when starting new unit)"""
        key = f"{learner_id}:{concept_id}"
        if key in self.performance_windows:
            del self.performance_windows[key]
        if key in self.active_scaffolds:
            del self.active_scaffolds[key]


# ============================================================================
# INTEGRATION WITH KNOWLEDGE TRACING
# ============================================================================

class AdaptiveEngine:
    """
    Combines ZPD Regulation with Knowledge Tracing for full adaptation

    This is the "brain" that integrates:
    1. DKT predictions (what learner knows)
    2. ZPD regulation (optimal difficulty)
    3. FSRS scheduling (when to review)
    """

    def __init__(self, zpd_regulator: ZPDRegulator):
        self.zpd_regulator = zpd_regulator

    def recommend_next_activity(
        self,
        learner_id: str,
        knowledge_state: np.ndarray,  # From DKT
        available_concepts: List[str],
        concept_difficulties: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Recommend next learning activity based on:
        - Current knowledge state (DKT)
        - ZPD assessment
        - Available content

        Returns optimal concept and difficulty level
        """
        recommendations = []

        for concept_id in available_concepts:
            # Get ZPD state
            zpd_state = self.zpd_regulator.assess_zpd_state(learner_id, concept_id)

            # Skip if in frustration zone (need prerequisite work first)
            if zpd_state.current_zone == ZPDZone.FRUSTRATION and zpd_state.success_rate < 0.2:
                continue

            # Calculate priority score
            # Factors:
            # - ZPD optimality (prefer concepts in ZPD)
            # - Knowledge gap (prefer concepts not yet mastered)
            # - Confidence (prefer high-confidence assessments)

            concept_idx = int(concept_id) if concept_id.isdigit() else 0
            if concept_idx < len(knowledge_state):
                current_mastery = knowledge_state[concept_idx]
            else:
                current_mastery = 0.5

            # ZPD score: highest when in optimal zone
            if zpd_state.current_zone == ZPDZone.OPTIMAL:
                zpd_score = 1.0
            elif zpd_state.current_zone == ZPDZone.COMFORT:
                zpd_score = 0.7
            else:
                zpd_score = 0.5

            # Knowledge gap score: highest when partially known
            gap_score = 1.0 - abs(current_mastery - 0.5) * 2

            # Combined priority
            priority = (
                zpd_score * 0.4 +
                gap_score * 0.4 +
                zpd_state.confidence * 0.2
            )

            recommendations.append({
                'concept_id': concept_id,
                'priority': priority,
                'zpd_zone': zpd_state.current_zone.value,
                'success_rate': zpd_state.success_rate,
                'mastery_estimate': float(current_mastery),
                'recommended_difficulty': concept_difficulties.get(concept_id, 5.0),
                'scaffolding_needed': zpd_state.current_zone == ZPDZone.FRUSTRATION,
                'actions': [a.value for a in zpd_state.recommended_actions]
            })

        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'], reverse=True)

        return {
            'recommendations': recommendations[:5],  # Top 5
            'top_choice': recommendations[0] if recommendations else None
        }


# Example usage
if __name__ == "__main__":
    # Initialize regulator
    regulator = ZPDRegulator()

    # Simulate learning sequence
    learner_id = "student_123"
    concept_id = "python_loops"

    print("=== ZPD Regulation Simulation ===\n")

    # Simulate attempts with varying success
    attempts = [
        (False, "Initial attempt - failed"),
        (False, "Second attempt - failed"),
        (False, "Third attempt - failed (frustration zone)"),
        (True, "Success after scaffolding"),
        (True, "Another success"),
        (True, "Improving..."),
        (True, "Consistently succeeding (comfort zone)"),
        (True, "Too easy now"),
    ]

    for success, description in attempts:
        regulator.record_attempt(learner_id, concept_id, success)
        state = regulator.assess_zpd_state(learner_id, concept_id)

        print(f"{description}")
        print(f"  Zone: {state.current_zone.value}")
        print(f"  Success Rate: {state.success_rate:.2%}")
        print(f"  Trend: {state.trend}")
        print(f"  Actions: {[a.value for a in state.recommended_actions]}")
        print(f"  Active Scaffolds: {regulator.get_active_scaffolding(learner_id, concept_id)}")
        print()
