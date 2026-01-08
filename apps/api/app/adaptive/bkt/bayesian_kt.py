"""
Bayesian Knowledge Tracing (BKT)
Probabilistic model for tracking knowledge mastery over time

Parameters:
- P(L0): Prior probability of knowing the skill
- P(T): Probability of learning (transition)
- P(G): Probability of guessing correctly when not knowing
- P(S): Probability of slip (error when knowing)

Formula:
P(Lt) = P(Lt-1 | evidence) + (1 - P(Lt-1 | evidence)) * P(T)
"""
from typing import Dict, Tuple, Optional
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
