"""
Bayesian Knowledge Tracing (BKT) with Temporal Difference Extension (TD-BKT)
Probabilistic model for tracking knowledge mastery over time

Standard BKT Parameters:
- P(L0): Prior probability of knowing the skill
- P(T): Probability of learning (transition)
- P(G): Probability of guessing correctly when not knowing
- P(S): Probability of slip (error when knowing)

TD-BKT Extension (from research):
- Uses behavioral telemetry signals for continuous knowledge updates
- Integrates frustration, engagement, and time signals
- Enables stealth assessment without explicit quizzes

Formula:
P(Lt) = P(Lt-1 | evidence) + (1 - P(Lt-1 | evidence)) * P(T)

TD Update:
δ = r + γ * V(s') - V(s)
V(s) ← V(s) + α * δ
"""
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import math


class BayesianKnowledgeTracer:
    """
    Bayesian Knowledge Tracing for mastery estimation
    """

    # Default BKT parameters (can be learned from data)
    DEFAULT_PARAMS = {
        "p_l0": 0.1,  # Prior probability of knowing (10%)
        "p_t": 0.15,  # Probability of learning per opportunity (15%)
        "p_g": 0.2,   # Probability of guessing correctly (20%)
        "p_s": 0.1,   # Probability of slip/error (10%)
    }

    def __init__(self, params: Optional[Dict[str, float]] = None):
        """
        Initialize BKT with parameters

        Args:
            params: Optional custom parameters (p_l0, p_t, p_g, p_s)
        """
        self.params = params or self.DEFAULT_PARAMS
        self.p_l0 = self.params["p_l0"]
        self.p_t = self.params["p_t"]
        self.p_g = self.params["p_g"]
        self.p_s = self.params["p_s"]

    def update_from_observation(
        self, current_mastery: float, correct: bool
    ) -> Tuple[float, Dict]:
        """
        Update mastery probability given an observation (Bayesian update)

        Args:
            current_mastery: Current P(L) - probability of knowing
            correct: Whether the observation was correct

        Returns:
            (New mastery probability, update details)
        """
        p_l = current_mastery

        # Calculate P(Correct | L) - probability of correct answer given knowledge state
        if correct:
            # P(Correct) = P(L) * (1 - P(S)) + (1 - P(L)) * P(G)
            p_correct = p_l * (1 - self.p_s) + (1 - p_l) * self.p_g

            # Bayes' rule: P(L | Correct) = P(Correct | L) * P(L) / P(Correct)
            numerator = p_l * (1 - self.p_s)
            p_l_given_obs = numerator / p_correct if p_correct > 0 else p_l
        else:
            # P(Incorrect) = P(L) * P(S) + (1 - P(L)) * (1 - P(G))
            p_incorrect = p_l * self.p_s + (1 - p_l) * (1 - self.p_g)

            # P(L | Incorrect) = P(Incorrect | L) * P(L) / P(Incorrect)
            numerator = p_l * self.p_s
            p_l_given_obs = numerator / p_incorrect if p_incorrect > 0 else p_l

        # Apply learning transition
        # P(Lt) = P(Lt-1 | observation) + (1 - P(Lt-1 | observation)) * P(T)
        new_mastery = p_l_given_obs + (1 - p_l_given_obs) * self.p_t

        # Ensure bounds
        new_mastery = max(0.0, min(1.0, new_mastery))

        update_details = {
            "prior_mastery": current_mastery,
            "posterior_mastery": p_l_given_obs,
            "new_mastery_after_learning": new_mastery,
            "correct": correct,
            "learning_gain": new_mastery - current_mastery,
        }

        return new_mastery, update_details

    def update_from_evidence(
        self, current_mastery: float, evidence_score: float
    ) -> Tuple[float, Dict]:
        """
        Update mastery from stealth assessment evidence (continuous score)

        Args:
            current_mastery: Current P(L)
            evidence_score: Evidence score from 0 to 1

        Returns:
            (New mastery probability, update details)
        """
        # Interpret evidence score as "correctness probability"
        # High evidence score = more likely to know the concept

        # Weighted update based on evidence confidence
        # Strong evidence (close to 0 or 1) updates more
        # Weak evidence (around 0.5) updates less
        evidence_strength = abs(evidence_score - 0.5) * 2  # 0 to 1

        # Treat as partial observation
        # evidence_score > 0.7 = likely knows
        # evidence_score < 0.3 = likely doesn't know
        # evidence_score ~ 0.5 = uncertain

        if evidence_score > 0.7:
            # Positive evidence - treat as correct answer with confidence
            correct_prob = evidence_score
            new_mastery, _ = self.update_from_observation(
                current_mastery, correct=True
            )
            # Scale update by evidence strength
            new_mastery = current_mastery + (
                new_mastery - current_mastery
            ) * evidence_strength * correct_prob
        elif evidence_score < 0.3:
            # Negative evidence - treat as incorrect answer
            incorrect_prob = 1 - evidence_score
            new_mastery, _ = self.update_from_observation(
                current_mastery, correct=False
            )
            # Scale update by evidence strength
            new_mastery = current_mastery + (
                new_mastery - current_mastery
            ) * evidence_strength * incorrect_prob
        else:
            # Weak evidence - minimal update
            # Slight nudge toward evidence score
            new_mastery = current_mastery * 0.9 + evidence_score * 0.1

        # Ensure bounds
        new_mastery = max(0.0, min(1.0, new_mastery))

        update_details = {
            "prior_mastery": current_mastery,
            "evidence_score": evidence_score,
            "evidence_strength": evidence_strength,
            "new_mastery": new_mastery,
            "mastery_change": new_mastery - current_mastery,
        }

        return new_mastery, update_details

    def predict_performance(self, mastery: float) -> Dict[str, float]:
        """
        Predict performance given mastery level

        Args:
            mastery: Current mastery probability P(L)

        Returns:
            Dictionary with performance predictions
        """
        # P(Correct) = P(L) * (1 - P(S)) + (1 - P(L)) * P(G)
        p_correct = mastery * (1 - self.p_s) + (1 - mastery) * self.p_g

        return {
            "mastery": mastery,
            "p_correct": p_correct,
            "p_incorrect": 1 - p_correct,
            "confidence": abs(p_correct - 0.5) * 2,  # How certain we are
        }

    def is_mastered(self, mastery: float, threshold: float = 0.95) -> bool:
        """
        Determine if concept is mastered

        Args:
            mastery: Current mastery probability
            threshold: Mastery threshold (default 95%)

        Returns:
            True if mastered
        """
        return mastery >= threshold

    def sessions_to_mastery(
        self, current_mastery: float, success_rate: float = 0.8, threshold: float = 0.95
    ) -> int:
        """
        Estimate number of practice sessions needed to reach mastery

        Args:
            current_mastery: Current mastery level
            success_rate: Expected success rate in practice
            threshold: Mastery threshold

        Returns:
            Estimated number of sessions
        """
        if current_mastery >= threshold:
            return 0

        # Simulate learning trajectory
        mastery = current_mastery
        sessions = 0
        max_sessions = 100  # Safety limit

        while mastery < threshold and sessions < max_sessions:
            # Assume average performance based on success rate
            correct = success_rate > 0.5  # Simplified
            mastery, _ = self.update_from_observation(mastery, correct)
            sessions += 1

        return sessions if sessions < max_sessions else -1  # -1 means unlikely to master

    def calibrate_parameters(
        self, history: List[Tuple[float, bool]], learning_rate: float = 0.05
    ) -> Dict[str, float]:
        """
        Simple parameter learning loop to adjust P(G) and P(S) based on residuals
        
        Args:
            history: List of (mastery_before, observation_correct)
            learning_rate: How quickly to adjust parameters
            
        Returns:
            Updated parameters
        """
        g_deltas = []
        s_deltas = []

        for mastery, correct in history:
            prediction = self.predict_performance(mastery)
            p_correct = prediction["p_correct"]
            
            # If we predict wrong but user is correct (and mastery is low) => P(G) might be higher
            if correct and mastery < 0.3:
                # Residue positive: user outperformed expectation
                g_deltas.append(1.0 - p_correct)
            
            # If we predict correct but user is wrong (and mastery is high) => P(S) might be higher
            if not correct and mastery > 0.7:
                # Residue negative: user underperformed expectation
                s_deltas.append(p_correct)

        if g_deltas:
            avg_g_delta = sum(g_deltas) / len(g_deltas)
            self.p_g = max(0.01, min(0.5, self.p_g + avg_g_delta * learning_rate))
        
        if s_deltas:
            avg_s_delta = sum(s_deltas) / len(s_deltas)
            self.p_s = max(0.01, min(0.5, self.p_s + avg_s_delta * learning_rate))

        # Update params dict
        self.params["p_g"] = self.p_g
        self.params["p_s"] = self.p_s

        return self.params


class AffectState(str, Enum):
    """Affect states from telemetry (matches telemetry-tracker.ts)"""
    FLOW = "flow"
    FRUSTRATED = "frustrated"
    BORED = "bored"
    CONFUSED = "confused"
    NEUTRAL = "neutral"


@dataclass
class TelemetrySignal:
    """Telemetry signal for TD-BKT updates"""
    affect_state: AffectState
    frustration_score: float  # 0-1
    engagement_score: float   # 0-1 (derived from activity)
    time_on_task: float       # seconds
    scroll_depth: float       # 0-100 percentage
    idle_time_ratio: float    # 0-1 ratio of idle to active time


class TDBKT(BayesianKnowledgeTracer):
    """
    Temporal Difference Bayesian Knowledge Tracing (TD-BKT)

    Research alignment: Combines BKT with temporal difference learning
    to update knowledge estimates using continuous behavioral signals
    from telemetry (frustration, engagement, time patterns) without
    requiring explicit assessment.

    Key innovations:
    1. Real-time knowledge estimation from behavioral signals
    2. Affect-aware learning rate adjustment
    3. Signal-to-evidence mapping for stealth assessment
    """

    # Telemetry signal weights for evidence computation
    SIGNAL_WEIGHTS = {
        "affect": 0.30,        # Affect state contribution
        "engagement": 0.25,    # Engagement level
        "frustration": 0.25,   # Inverse frustration (confusion indicator)
        "time_quality": 0.20,  # Time spent quality
    }

    # Affect state to learning signal mapping
    AFFECT_LEARNING_SIGNALS = {
        AffectState.FLOW: 0.85,       # High learning signal
        AffectState.NEUTRAL: 0.50,    # Baseline
        AffectState.CONFUSED: 0.30,   # Struggling but engaged
        AffectState.BORED: 0.20,      # Disengaged - low learning
        AffectState.FRUSTRATED: 0.10, # Blocked - minimal learning
    }

    def __init__(
        self,
        params: Optional[Dict[str, float]] = None,
        td_learning_rate: float = 0.1,
        discount_factor: float = 0.9
    ):
        """
        Initialize TD-BKT

        Args:
            params: BKT parameters
            td_learning_rate: TD learning rate (alpha)
            discount_factor: TD discount factor (gamma)
        """
        super().__init__(params)
        self.alpha = td_learning_rate
        self.gamma = discount_factor
        self._value_estimates: Dict[str, float] = {}  # concept_id -> value

    def telemetry_to_evidence(self, signal: TelemetrySignal) -> float:
        """
        Convert telemetry signal to evidence score for BKT update.

        Research alignment: Maps behavioral signals to knowledge evidence
        using a weighted combination of affect, engagement, and time metrics.

        Args:
            signal: Telemetry signal from client

        Returns:
            Evidence score (0-1) for knowledge estimation
        """
        # Affect component
        affect_signal = self.AFFECT_LEARNING_SIGNALS.get(signal.affect_state, 0.5)

        # Engagement component (inverse of idle time ratio)
        engagement_signal = signal.engagement_score

        # Frustration component (inverse - low frustration = good)
        frustration_signal = 1.0 - signal.frustration_score

        # Time quality component
        # Optimal time on task varies, but we assume 30-300 seconds is good
        if signal.time_on_task < 10:
            time_signal = 0.2  # Too fast - likely skipped
        elif signal.time_on_task < 30:
            time_signal = 0.5  # Quick review
        elif signal.time_on_task < 300:
            time_signal = 0.8  # Good engagement
        else:
            time_signal = 0.4  # Too long - might be stuck

        # Scroll depth bonus (reading through content)
        scroll_bonus = min(signal.scroll_depth / 100, 1.0) * 0.2

        # Weighted combination
        evidence = (
            self.SIGNAL_WEIGHTS["affect"] * affect_signal +
            self.SIGNAL_WEIGHTS["engagement"] * engagement_signal +
            self.SIGNAL_WEIGHTS["frustration"] * frustration_signal +
            self.SIGNAL_WEIGHTS["time_quality"] * (time_signal + scroll_bonus)
        )

        return max(0.0, min(1.0, evidence))

    def td_update(
        self,
        concept_id: str,
        current_mastery: float,
        signal: TelemetrySignal,
        next_mastery_estimate: Optional[float] = None
    ) -> Tuple[float, Dict]:
        """
        Perform temporal difference update on knowledge estimate.

        Research alignment: TD(0) learning applied to knowledge tracing:
        - State: Current mastery estimate
        - Reward: Evidence from telemetry signals
        - Transition: BKT learning dynamics

        Args:
            concept_id: Concept being learned
            current_mastery: Current P(L)
            signal: Telemetry signal
            next_mastery_estimate: Optional next state estimate

        Returns:
            (Updated mastery, TD update details)
        """
        # Convert telemetry to evidence
        evidence = self.telemetry_to_evidence(signal)

        # Get or initialize value estimate
        if concept_id not in self._value_estimates:
            self._value_estimates[concept_id] = current_mastery

        current_value = self._value_estimates[concept_id]

        # Compute TD target (reward + discounted next value)
        # Reward is the evidence signal (interpreted as learning success)
        reward = evidence

        # Next value estimate
        if next_mastery_estimate is not None:
            next_value = next_mastery_estimate
        else:
            # Estimate next value using BKT transition
            next_value = current_value + (1 - current_value) * self.p_t

        # TD error: δ = r + γ * V(s') - V(s)
        td_error = reward + self.gamma * next_value - current_value

        # Update value estimate: V(s) ← V(s) + α * δ
        new_value = current_value + self.alpha * td_error
        new_value = max(0.0, min(1.0, new_value))

        # Store updated value
        self._value_estimates[concept_id] = new_value

        # Also perform BKT update with evidence
        bkt_mastery, bkt_details = self.update_from_evidence(current_mastery, evidence)

        # Combine TD and BKT estimates (weighted average)
        combined_mastery = 0.6 * new_value + 0.4 * bkt_mastery

        update_details = {
            "concept_id": concept_id,
            "prior_mastery": current_mastery,
            "evidence_from_telemetry": evidence,
            "td_error": td_error,
            "td_value_estimate": new_value,
            "bkt_mastery": bkt_mastery,
            "combined_mastery": combined_mastery,
            "affect_state": signal.affect_state.value,
            "frustration_score": signal.frustration_score,
            "engagement_score": signal.engagement_score,
        }

        return combined_mastery, update_details

    def update_from_telemetry(
        self,
        concept_id: str,
        current_mastery: float,
        affect_state: str,
        frustration_score: float,
        engagement_score: float = 0.5,
        time_on_task: float = 60.0,
        scroll_depth: float = 50.0,
        idle_time_ratio: float = 0.2
    ) -> Tuple[float, Dict]:
        """
        Convenience method to update mastery from raw telemetry values.

        Args:
            concept_id: Concept being learned
            current_mastery: Current mastery estimate
            affect_state: String affect state from telemetry
            frustration_score: Frustration index (0-1)
            engagement_score: Derived engagement (0-1)
            time_on_task: Seconds on current content
            scroll_depth: Percentage scrolled (0-100)
            idle_time_ratio: Ratio of idle time (0-1)

        Returns:
            (Updated mastery, update details)
        """
        # Parse affect state
        try:
            affect = AffectState(affect_state)
        except ValueError:
            affect = AffectState.NEUTRAL

        # Build signal
        signal = TelemetrySignal(
            affect_state=affect,
            frustration_score=frustration_score,
            engagement_score=engagement_score,
            time_on_task=time_on_task,
            scroll_depth=scroll_depth,
            idle_time_ratio=idle_time_ratio
        )

        return self.td_update(concept_id, current_mastery, signal)

    def get_intervention_recommendation(
        self,
        mastery: float,
        signal: TelemetrySignal
    ) -> Dict[str, any]:
        """
        Recommend intervention based on mastery and telemetry.

        Research alignment: Real-time micro-interventions triggered
        by behavioral signals, not just mastery thresholds.

        Args:
            mastery: Current mastery estimate
            signal: Current telemetry signal

        Returns:
            Intervention recommendation with type and urgency
        """
        recommendations = {
            "should_intervene": False,
            "intervention_type": None,
            "urgency": "low",
            "message": None
        }

        # High frustration intervention
        if signal.frustration_score > 0.7:
            recommendations["should_intervene"] = True
            recommendations["intervention_type"] = "simplify"
            recommendations["urgency"] = "high"
            recommendations["message"] = "Consider providing a simpler explanation or hint"

        # Confusion intervention
        elif signal.affect_state == AffectState.CONFUSED and mastery < 0.5:
            recommendations["should_intervene"] = True
            recommendations["intervention_type"] = "scaffold"
            recommendations["urgency"] = "medium"
            recommendations["message"] = "Provide step-by-step guidance"

        # Boredom intervention (might be too easy)
        elif signal.affect_state == AffectState.BORED:
            if mastery > 0.7:
                recommendations["should_intervene"] = True
                recommendations["intervention_type"] = "challenge"
                recommendations["urgency"] = "low"
                recommendations["message"] = "Increase difficulty or move to next concept"
            else:
                recommendations["should_intervene"] = True
                recommendations["intervention_type"] = "engage"
                recommendations["urgency"] = "medium"
                recommendations["message"] = "Try interactive exercise or different modality"

        # Flow state - no intervention needed, but track for positive reinforcement
        elif signal.affect_state == AffectState.FLOW:
            recommendations["intervention_type"] = "none"
            recommendations["message"] = "Learner in flow state - maintain current approach"

        # Stuck too long intervention
        elif signal.time_on_task > 300 and signal.idle_time_ratio > 0.5:
            recommendations["should_intervene"] = True
            recommendations["intervention_type"] = "prompt"
            recommendations["urgency"] = "medium"
            recommendations["message"] = "Check if learner needs help"

        return recommendations

    def reset_value_estimates(self, concept_id: Optional[str] = None):
        """Reset TD value estimates"""
        if concept_id:
            self._value_estimates.pop(concept_id, None)
        else:
            self._value_estimates.clear()
