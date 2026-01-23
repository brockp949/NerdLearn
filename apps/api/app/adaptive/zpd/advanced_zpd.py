"""
Advanced ZPD (Zone of Proximal Development) Module

Extends basic ZPD with:
- Multi-dimensional difficulty assessment
- Real-time frustration detection
- Affective state modeling
- Adaptive scaffolding recommendations
- Learning momentum tracking

Research basis:
- Vygotsky's ZPD theory
- Csikszentmihalyi's Flow theory
- Affective computing in education
- Self-Determination Theory (SDT)
"""

import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import statistics
from app.adaptive.cognitive import FrustrationDetector

logger = logging.getLogger(__name__)


# ==================== Enums and Data Classes ====================

class AffectiveState(str, Enum):
    """Learner affective/emotional states"""
    FLOW = "flow"              # Optimal engagement
    BOREDOM = "boredom"        # Under-challenged
    FRUSTRATION = "frustration" # Over-challenged
    ANXIETY = "anxiety"        # High stakes, uncertain
    CONFUSION = "confusion"    # Cognitive conflict
    CURIOSITY = "curiosity"    # Engaged, exploring
    ENGAGED = "engaged"        # Active learning
    DISENGAGED = "disengaged"  # Passive, distracted


class DifficultyDimension(str, Enum):
    """Dimensions of content difficulty"""
    COGNITIVE = "cognitive"           # Mental processing required
    PRIOR_KNOWLEDGE = "prior_knowledge"  # Required background
    COMPLEXITY = "complexity"         # Number of interacting elements
    ABSTRACTNESS = "abstractness"     # Concrete vs abstract
    NOVELTY = "novelty"              # Familiarity of content
    PRECISION = "precision"          # Required accuracy
    TIME_PRESSURE = "time_pressure"  # Urgency/deadline stress


@dataclass
class DifficultyProfile:
    """Multi-dimensional difficulty profile for content"""
    cognitive: float = 0.5      # 0-1 scale
    prior_knowledge: float = 0.5
    complexity: float = 0.5
    abstractness: float = 0.5
    novelty: float = 0.5
    precision: float = 0.5
    time_pressure: float = 0.3

    def overall_difficulty(self, weights: Dict[str, float] = None) -> float:
        """Calculate weighted overall difficulty"""
        default_weights = {
            "cognitive": 0.25,
            "prior_knowledge": 0.20,
            "complexity": 0.20,
            "abstractness": 0.10,
            "novelty": 0.10,
            "precision": 0.10,
            "time_pressure": 0.05,
        }
        weights = weights or default_weights

        total = 0.0
        for dim, weight in weights.items():
            total += getattr(self, dim, 0.5) * weight
        return total

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            "cognitive": self.cognitive,
            "prior_knowledge": self.prior_knowledge,
            "complexity": self.complexity,
            "abstractness": self.abstractness,
            "novelty": self.novelty,
            "precision": self.precision,
            "time_pressure": self.time_pressure,
            "overall": self.overall_difficulty()
        }


@dataclass
class LearnerProfile:
    """Multi-dimensional learner capability profile"""
    cognitive_capacity: float = 0.5    # Working memory, processing speed
    prior_knowledge: float = 0.5       # Domain knowledge
    complexity_tolerance: float = 0.5  # Ability to handle complex info
    abstract_reasoning: float = 0.5    # Abstract vs concrete preference
    novelty_preference: float = 0.5    # Comfort with new material
    precision_capability: float = 0.5  # Attention to detail
    stress_tolerance: float = 0.5      # Performance under pressure

    def matches_difficulty(self, difficulty: DifficultyProfile) -> float:
        """Calculate match score between learner and content difficulty"""
        # Each dimension: closer to 0 = better match (learner capability >= difficulty)
        gaps = [
            max(0, difficulty.cognitive - self.cognitive_capacity),
            max(0, difficulty.prior_knowledge - self.prior_knowledge),
            max(0, difficulty.complexity - self.complexity_tolerance),
            max(0, difficulty.abstractness - self.abstract_reasoning),
            max(0, difficulty.novelty - self.novelty_preference),
            max(0, difficulty.precision - self.precision_capability),
            max(0, difficulty.time_pressure - self.stress_tolerance),
        ]

        # Average gap (0 = perfect match, 1 = completely mismatched)
        avg_gap = sum(gaps) / len(gaps)

        # Convert to match score (1 = perfect, 0 = poor)
        return 1 - avg_gap


@dataclass
class FrustrationIndicator:
    """Indicators of learner frustration"""
    timestamp: datetime
    indicator_type: str  # "error_rate", "time_pattern", "behavior", etc.
    severity: float      # 0-1
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningMomentum:
    """Tracks learning velocity and acceleration"""
    velocity: float = 0.0      # Rate of mastery gain
    acceleration: float = 0.0  # Change in velocity
    trend: str = "stable"      # "accelerating", "decelerating", "stable"
    confidence: float = 0.5    # Confidence in momentum estimate


# ==================== Frustration Detection ====================

# FrustrationDetector removed - imported from app.adaptive.cognitive


# ==================== Advanced ZPD Regulator ====================

class AdvancedZPDRegulator:
    """
    Advanced Zone of Proximal Development regulator with:
    - Multi-dimensional difficulty matching
    - Real-time affective state tracking
    - Frustration detection and intervention
    - Learning momentum optimization
    """

    def __init__(
        self,
        zpd_width: float = 0.3,
        optimal_challenge: float = 0.15,  # Optimal difficulty above mastery
        frustration_detector: Optional[FrustrationDetector] = None
    ):
        """
        Initialize advanced ZPD regulator.

        Args:
            zpd_width: Width of the ZPD zone
            optimal_challenge: Optimal difficulty increment above mastery
            frustration_detector: Optional frustration detector instance
        """
        self.zpd_width = zpd_width
        self.optimal_challenge = optimal_challenge
        self.frustration_detector = frustration_detector or FrustrationDetector()

        # Learning momentum tracking
        self._momentum_history: Dict[str, deque] = {}

    def calculate_multidimensional_zpd(
        self,
        learner: LearnerProfile,
        content: DifficultyProfile,
        concept_mastery: float
    ) -> Dict[str, Any]:
        """
        Calculate ZPD fit using multi-dimensional analysis.

        Args:
            learner: Learner capability profile
            content: Content difficulty profile
            concept_mastery: Current mastery of target concept

        Returns:
            Detailed ZPD analysis with per-dimension breakdown
        """
        # Per-dimension ZPD analysis
        dimension_analysis = {}

        dimensions = [
            ("cognitive", learner.cognitive_capacity, content.cognitive),
            ("prior_knowledge", learner.prior_knowledge, content.prior_knowledge),
            ("complexity", learner.complexity_tolerance, content.complexity),
            ("abstractness", learner.abstract_reasoning, content.abstractness),
            ("novelty", learner.novelty_preference, content.novelty),
            ("precision", learner.precision_capability, content.precision),
            ("time_pressure", learner.stress_tolerance, content.time_pressure),
        ]

        in_zpd_count = 0
        challenge_levels = []

        for dim_name, capability, difficulty in dimensions:
            gap = difficulty - capability

            # Determine zone for this dimension
            if gap < -self.zpd_width:
                zone = "too_easy"
            elif gap > self.zpd_width:
                zone = "too_hard"
            elif 0 <= gap <= self.optimal_challenge:
                zone = "optimal"
                in_zpd_count += 1
            elif gap < 0:
                zone = "easy_side"
                in_zpd_count += 0.5
            else:
                zone = "hard_side"
                in_zpd_count += 0.5

            dimension_analysis[dim_name] = {
                "capability": capability,
                "difficulty": difficulty,
                "gap": gap,
                "zone": zone
            }
            challenge_levels.append(gap)

        # Overall ZPD score
        zpd_score = in_zpd_count / len(dimensions)

        # Overall challenge level
        avg_challenge = sum(challenge_levels) / len(challenge_levels)

        # Determine overall state
        if zpd_score >= 0.7:
            state = AffectiveState.FLOW
        elif avg_challenge < -0.2:
            state = AffectiveState.BOREDOM
        elif avg_challenge > 0.3:
            state = AffectiveState.FRUSTRATION
        elif zpd_score >= 0.4:
            state = AffectiveState.ENGAGED
        else:
            state = AffectiveState.CONFUSION

        return {
            "zpd_score": round(zpd_score, 3),
            "predicted_state": state.value,
            "overall_challenge": round(avg_challenge, 3),
            "dimension_analysis": dimension_analysis,
            "match_score": round(learner.matches_difficulty(content), 3),
            "recommendations": self._get_dimension_recommendations(dimension_analysis)
        }

    def _get_dimension_recommendations(
        self,
        analysis: Dict[str, Dict]
    ) -> List[str]:
        """Generate recommendations based on dimension analysis"""
        recommendations = []

        too_hard = [dim for dim, data in analysis.items() if data["zone"] == "too_hard"]
        too_easy = [dim for dim, data in analysis.items() if data["zone"] == "too_easy"]

        if too_hard:
            if "prior_knowledge" in too_hard:
                recommendations.append("Review prerequisite material before this content")
            if "cognitive" in too_hard:
                recommendations.append("Break content into smaller, simpler chunks")
            if "complexity" in too_hard:
                recommendations.append("Reduce number of interacting elements")
            if "abstractness" in too_hard:
                recommendations.append("Add concrete examples and visualizations")
            if "novelty" in too_hard:
                recommendations.append("Connect to familiar concepts first")
            if "time_pressure" in too_hard:
                recommendations.append("Remove time constraints for this learner")

        if too_easy:
            if len(too_easy) >= 3:
                recommendations.append("Consider advancing to more challenging content")
            else:
                recommendations.append(f"Can handle more challenge in: {', '.join(too_easy)}")

        return recommendations

    def track_learning_momentum(
        self,
        user_id: str,
        mastery: float,
        timestamp: Optional[datetime] = None
    ) -> LearningMomentum:
        """
        Track and calculate learning momentum.

        Args:
            user_id: User identifier
            mastery: Current mastery level
            timestamp: Observation timestamp

        Returns:
            Learning momentum analysis
        """
        timestamp = timestamp or datetime.utcnow()

        if user_id not in self._momentum_history:
            self._momentum_history[user_id] = deque(maxlen=20)

        self._momentum_history[user_id].append((timestamp, mastery))

        history = list(self._momentum_history[user_id])

        if len(history) < 3:
            return LearningMomentum(
                velocity=0.0,
                acceleration=0.0,
                trend="stable",
                confidence=0.2
            )

        # Calculate velocity (mastery change per hour)
        velocities = []
        for i in range(1, len(history)):
            time_diff = (history[i][0] - history[i-1][0]).total_seconds() / 3600
            if time_diff > 0:
                mastery_diff = history[i][1] - history[i-1][1]
                velocities.append(mastery_diff / time_diff)

        if not velocities:
            return LearningMomentum()

        current_velocity = velocities[-1]
        avg_velocity = statistics.mean(velocities)

        # Calculate acceleration
        if len(velocities) >= 2:
            recent_velocity = statistics.mean(velocities[-3:])
            older_velocity = statistics.mean(velocities[:-3]) if len(velocities) > 3 else velocities[0]
            acceleration = recent_velocity - older_velocity
        else:
            acceleration = 0.0

        # Determine trend
        if acceleration > 0.01:
            trend = "accelerating"
        elif acceleration < -0.01:
            trend = "decelerating"
        else:
            trend = "stable"

        # Confidence based on data points
        confidence = min(1.0, len(history) / 10)

        return LearningMomentum(
            velocity=round(avg_velocity, 4),
            acceleration=round(acceleration, 4),
            trend=trend,
            confidence=round(confidence, 2)
        )

    def get_adaptive_recommendation(
        self,
        user_id: str,
        learner: LearnerProfile,
        available_content: List[Dict[str, Any]],
        current_mastery: float,
        recent_performance: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Get adaptive content recommendation considering all factors.

        Args:
            user_id: User identifier
            learner: Learner capability profile
            available_content: List of content with difficulty profiles
            current_mastery: Current concept mastery
            recent_performance: Recent performance data for frustration detection

        Returns:
            Comprehensive recommendation with reasoning
        """
        # Check frustration state
        if recent_performance:
            for event in recent_performance:
                self.frustration_detector.record_event(
                    user_id, event.get("type", "unknown"), event
                )

        frustration = self.frustration_detector.detect_frustration(user_id)

        # Get learning momentum
        momentum = self.track_learning_momentum(user_id, current_mastery)

        # Score all available content
        scored_content = []

        for content in available_content:
            difficulty = DifficultyProfile(**content.get("difficulty", {}))

            # Calculate ZPD fit
            zpd_analysis = self.calculate_multidimensional_zpd(
                learner, difficulty, current_mastery
            )

            # Adjust score based on frustration
            base_score = zpd_analysis["zpd_score"]

            if frustration["frustrated"]:
                # Prefer easier content when frustrated
                if zpd_analysis["overall_challenge"] > 0:
                    base_score *= 0.7
                else:
                    base_score *= 1.2

            # Adjust based on momentum
            if momentum.trend == "accelerating":
                # Can handle more challenge
                if zpd_analysis["overall_challenge"] > 0:
                    base_score *= 1.1
            elif momentum.trend == "decelerating":
                # Reduce challenge
                if zpd_analysis["overall_challenge"] > 0.1:
                    base_score *= 0.8

            scored_content.append({
                "content_id": content.get("id"),
                "title": content.get("title", ""),
                "score": round(base_score, 3),
                "zpd_analysis": zpd_analysis,
                "difficulty": difficulty.to_dict()
            })

        # Sort by score
        scored_content.sort(key=lambda x: x["score"], reverse=True)

        # Get top recommendation
        top_rec = scored_content[0] if scored_content else None

        return {
            "recommendation": top_rec,
            "alternatives": scored_content[1:4],
            "frustration_state": frustration,
            "learning_momentum": {
                "velocity": momentum.velocity,
                "acceleration": momentum.acceleration,
                "trend": momentum.trend,
                "confidence": momentum.confidence
            },
            "learner_state": {
                "current_mastery": current_mastery,
                "predicted_state": top_rec["zpd_analysis"]["predicted_state"] if top_rec else "unknown"
            },
            "scaffolding_recommendation": self._get_scaffolding_level(frustration, momentum)
        }

    def _get_scaffolding_level(
        self,
        frustration: Dict,
        momentum: LearningMomentum
    ) -> Dict[str, Any]:
        """Determine appropriate scaffolding level"""
        base_level = 0.5  # Default moderate scaffolding

        # Increase scaffolding if frustrated
        if frustration["frustrated"]:
            if frustration["level"] == "high":
                base_level = 0.9
            elif frustration["level"] == "moderate":
                base_level = 0.7
            else:
                base_level = 0.6

        # Decrease scaffolding if momentum is good
        if momentum.trend == "accelerating" and momentum.velocity > 0:
            base_level *= 0.8
        elif momentum.trend == "decelerating":
            base_level = min(1.0, base_level * 1.2)

        # Scaffolding recommendations
        if base_level >= 0.8:
            level_name = "high"
            strategies = [
                "Provide complete worked examples",
                "Break into small steps with feedback",
                "Offer multiple hints available"
            ]
        elif base_level >= 0.5:
            level_name = "moderate"
            strategies = [
                "Provide partial worked examples",
                "Offer hints after first attempt",
                "Give immediate feedback on errors"
            ]
        else:
            level_name = "low"
            strategies = [
                "Let learner attempt independently",
                "Provide hints only on request",
                "Delayed feedback to encourage reflection"
            ]

        return {
            "level": level_name,
            "level_value": round(base_level, 2),
            "strategies": strategies
        }


# Singleton instances
frustration_detector = FrustrationDetector()
advanced_zpd_regulator = AdvancedZPDRegulator(frustration_detector=frustration_detector)
