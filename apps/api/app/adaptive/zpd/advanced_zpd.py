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

class FrustrationDetector:
    """
    Real-time frustration detection from behavioral signals.

    Monitors:
    - Error patterns (consecutive errors, error types)
    - Time patterns (long pauses, rushed responses)
    - Behavioral signals (hint overuse, giving up, backtracking)
    - Implicit feedback (abandonment, disengagement)
    """

    # Thresholds for frustration detection
    CONSECUTIVE_ERROR_THRESHOLD = 3
    ERROR_RATE_THRESHOLD = 0.6
    LONG_PAUSE_MULTIPLIER = 3.0
    RUSH_MULTIPLIER = 0.3
    HINT_OVERUSE_THRESHOLD = 3

    def __init__(self, window_size: int = 10):
        """
        Initialize frustration detector.

        Args:
            window_size: Number of recent events to consider
        """
        self.window_size = window_size
        self._event_history: Dict[str, deque] = {}  # user_id -> events
        self._indicators: Dict[str, List[FrustrationIndicator]] = {}

    def record_event(
        self,
        user_id: str,
        event_type: str,
        data: Dict[str, Any]
    ):
        """Record a learning event for frustration analysis"""
        if user_id not in self._event_history:
            self._event_history[user_id] = deque(maxlen=self.window_size * 3)

        event = {
            "timestamp": datetime.utcnow(),
            "type": event_type,
            "data": data
        }
        self._event_history[user_id].append(event)

    def detect_frustration(
        self,
        user_id: str,
        expected_time: float = 60.0
    ) -> Dict[str, Any]:
        """
        Detect frustration from recent events.

        Args:
            user_id: User identifier
            expected_time: Expected time for typical response (seconds)

        Returns:
            Frustration analysis with indicators and recommendations
        """
        if user_id not in self._event_history:
            return {
                "frustrated": False,
                "confidence": 0.0,
                "indicators": [],
                "level": "none",
                "recommendations": []
            }

        events = list(self._event_history[user_id])
        indicators = []

        # 1. Check error patterns
        error_indicators = self._check_error_patterns(events)
        indicators.extend(error_indicators)

        # 2. Check time patterns
        time_indicators = self._check_time_patterns(events, expected_time)
        indicators.extend(time_indicators)

        # 3. Check behavioral signals
        behavior_indicators = self._check_behavioral_signals(events)
        indicators.extend(behavior_indicators)

        # 4. Check engagement patterns
        engagement_indicators = self._check_engagement_patterns(events)
        indicators.extend(engagement_indicators)

        # Calculate overall frustration level
        if not indicators:
            frustration_level = 0.0
        else:
            # Weighted average of indicator severities
            recent_weight = 2.0
            weights = [
                recent_weight if i.timestamp > datetime.utcnow() - timedelta(minutes=5) else 1.0
                for i in indicators
            ]
            frustration_level = sum(
                i.severity * w for i, w in zip(indicators, weights)
            ) / sum(weights)

        # Determine frustration state
        if frustration_level >= 0.7:
            level = "high"
            frustrated = True
        elif frustration_level >= 0.4:
            level = "moderate"
            frustrated = True
        elif frustration_level >= 0.2:
            level = "low"
            frustrated = False
        else:
            level = "none"
            frustrated = False

        # Generate recommendations
        recommendations = self._generate_recommendations(indicators, level)

        return {
            "frustrated": frustrated,
            "frustration_level": round(frustration_level, 3),
            "confidence": self._calculate_confidence(indicators, events),
            "level": level,
            "indicators": [
                {
                    "type": i.indicator_type,
                    "severity": i.severity,
                    "details": i.details
                }
                for i in indicators
            ],
            "recommendations": recommendations
        }

    def _check_error_patterns(
        self,
        events: List[Dict]
    ) -> List[FrustrationIndicator]:
        """Check for frustrating error patterns"""
        indicators = []

        # Get recent attempts
        attempts = [
            e for e in events
            if e["type"] in ["quiz_attempt", "practice_attempt", "answer_submit"]
        ][-self.window_size:]

        if not attempts:
            return indicators

        # Consecutive errors
        consecutive_errors = 0
        for attempt in reversed(attempts):
            if not attempt["data"].get("correct", True):
                consecutive_errors += 1
            else:
                break

        if consecutive_errors >= self.CONSECUTIVE_ERROR_THRESHOLD:
            indicators.append(FrustrationIndicator(
                timestamp=datetime.utcnow(),
                indicator_type="consecutive_errors",
                severity=min(1.0, consecutive_errors / 5),
                details={"count": consecutive_errors}
            ))

        # Error rate
        if len(attempts) >= 5:
            error_count = sum(
                1 for a in attempts if not a["data"].get("correct", True)
            )
            error_rate = error_count / len(attempts)

            if error_rate >= self.ERROR_RATE_THRESHOLD:
                indicators.append(FrustrationIndicator(
                    timestamp=datetime.utcnow(),
                    indicator_type="high_error_rate",
                    severity=error_rate,
                    details={"rate": error_rate, "window": len(attempts)}
                ))

        # Same error repeated
        error_types = [
            a["data"].get("error_type", "unknown")
            for a in attempts
            if not a["data"].get("correct", True)
        ]
        if error_types:
            from collections import Counter
            most_common = Counter(error_types).most_common(1)[0]
            if most_common[1] >= 3:
                indicators.append(FrustrationIndicator(
                    timestamp=datetime.utcnow(),
                    indicator_type="repeated_error_type",
                    severity=min(1.0, most_common[1] / 4),
                    details={"error_type": most_common[0], "count": most_common[1]}
                ))

        return indicators

    def _check_time_patterns(
        self,
        events: List[Dict],
        expected_time: float
    ) -> List[FrustrationIndicator]:
        """Check for frustrating time patterns"""
        indicators = []

        time_events = [
            e for e in events
            if "response_time" in e["data"] or "duration" in e["data"]
        ][-self.window_size:]

        if not time_events:
            return indicators

        times = [
            e["data"].get("response_time") or e["data"].get("duration", expected_time)
            for e in time_events
        ]

        # Long pauses (struggle indicator)
        long_pauses = sum(
            1 for t in times if t > expected_time * self.LONG_PAUSE_MULTIPLIER
        )
        if long_pauses >= 2:
            indicators.append(FrustrationIndicator(
                timestamp=datetime.utcnow(),
                indicator_type="long_pauses",
                severity=min(1.0, long_pauses / 4),
                details={"count": long_pauses, "threshold": expected_time * self.LONG_PAUSE_MULTIPLIER}
            ))

        # Rushed responses (giving up indicator)
        rushed = sum(
            1 for t in times if t < expected_time * self.RUSH_MULTIPLIER
        )
        if rushed >= 3:
            indicators.append(FrustrationIndicator(
                timestamp=datetime.utcnow(),
                indicator_type="rushed_responses",
                severity=min(1.0, rushed / 5),
                details={"count": rushed}
            ))

        # Increasing response times (fatigue/struggle)
        if len(times) >= 5:
            first_half = statistics.mean(times[:len(times)//2])
            second_half = statistics.mean(times[len(times)//2:])
            if second_half > first_half * 1.5:
                indicators.append(FrustrationIndicator(
                    timestamp=datetime.utcnow(),
                    indicator_type="increasing_response_time",
                    severity=min(1.0, (second_half / first_half - 1) / 2),
                    details={"increase_ratio": second_half / first_half}
                ))

        return indicators

    def _check_behavioral_signals(
        self,
        events: List[Dict]
    ) -> List[FrustrationIndicator]:
        """Check behavioral frustration signals"""
        indicators = []

        # Hint overuse
        hint_events = [e for e in events if e["type"] == "hint_request"]
        recent_hints = [
            h for h in hint_events
            if h["timestamp"] > datetime.utcnow() - timedelta(minutes=10)
        ]
        if len(recent_hints) >= self.HINT_OVERUSE_THRESHOLD:
            indicators.append(FrustrationIndicator(
                timestamp=datetime.utcnow(),
                indicator_type="hint_overuse",
                severity=min(1.0, len(recent_hints) / 5),
                details={"count": len(recent_hints)}
            ))

        # Skip/abandon patterns
        skip_events = [
            e for e in events
            if e["type"] in ["skip", "abandon", "give_up"]
        ]
        if len(skip_events) >= 2:
            indicators.append(FrustrationIndicator(
                timestamp=datetime.utcnow(),
                indicator_type="skip_abandon",
                severity=min(1.0, len(skip_events) / 3),
                details={"count": len(skip_events)}
            ))

        # Rapid backtracking
        nav_events = [e for e in events if e["type"] == "navigation"]
        back_nav = sum(
            1 for e in nav_events
            if e["data"].get("direction") == "back"
        )
        if back_nav >= 4:
            indicators.append(FrustrationIndicator(
                timestamp=datetime.utcnow(),
                indicator_type="rapid_backtracking",
                severity=min(1.0, back_nav / 6),
                details={"count": back_nav}
            ))

        return indicators

    def _check_engagement_patterns(
        self,
        events: List[Dict]
    ) -> List[FrustrationIndicator]:
        """Check engagement-related frustration patterns"""
        indicators = []

        if len(events) < 5:
            return indicators

        # Check for disengagement (long gaps between events)
        timestamps = [e["timestamp"] for e in events]
        gaps = []
        for i in range(1, len(timestamps)):
            gap = (timestamps[i] - timestamps[i-1]).total_seconds()
            gaps.append(gap)

        if gaps:
            avg_gap = statistics.mean(gaps)
            # Long gap followed by quick abandon
            if avg_gap > 300:  # 5 minutes average gap
                indicators.append(FrustrationIndicator(
                    timestamp=datetime.utcnow(),
                    indicator_type="disengagement",
                    severity=min(1.0, avg_gap / 600),
                    details={"avg_gap_seconds": avg_gap}
                ))

        return indicators

    def _calculate_confidence(
        self,
        indicators: List[FrustrationIndicator],
        events: List[Dict]
    ) -> float:
        """Calculate confidence in frustration detection"""
        # More events = higher confidence
        event_factor = min(1.0, len(events) / self.window_size)

        # More indicators = higher confidence
        indicator_factor = min(1.0, len(indicators) / 3) if indicators else 0.3

        # Recency of indicators
        if indicators:
            recent = sum(
                1 for i in indicators
                if i.timestamp > datetime.utcnow() - timedelta(minutes=5)
            )
            recency_factor = recent / len(indicators)
        else:
            recency_factor = 0.5

        return 0.4 * event_factor + 0.3 * indicator_factor + 0.3 * recency_factor

    def _generate_recommendations(
        self,
        indicators: List[FrustrationIndicator],
        level: str
    ) -> List[str]:
        """Generate intervention recommendations"""
        recommendations = []

        indicator_types = {i.indicator_type for i in indicators}

        if level == "high":
            recommendations.append("Consider immediate intervention - offer break or simpler content")

        if "consecutive_errors" in indicator_types:
            recommendations.append("Provide step-by-step guidance or worked example")

        if "high_error_rate" in indicator_types:
            recommendations.append("Review prerequisite concepts before continuing")

        if "repeated_error_type" in indicator_types:
            recommendations.append("Address specific misconception with targeted feedback")

        if "long_pauses" in indicator_types:
            recommendations.append("Offer hints or break content into smaller steps")

        if "rushed_responses" in indicator_types:
            recommendations.append("Encourage deliberate practice - quality over speed")

        if "hint_overuse" in indicator_types:
            recommendations.append("Provide scaffolded practice with fading hints")

        if "skip_abandon" in indicator_types:
            recommendations.append("Reduce difficulty or provide more support")

        if "disengagement" in indicator_types:
            recommendations.append("Re-engage with interesting example or gamification")

        if not recommendations:
            recommendations.append("Continue current approach")

        return recommendations


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
