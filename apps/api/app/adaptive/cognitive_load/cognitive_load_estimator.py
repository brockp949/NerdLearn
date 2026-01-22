"""
Cognitive Load Estimation System
Real-time cognitive load detection and adaptive scaffolding

Research basis:
- Cognitive Load Theory in Adaptive Learning: Expertise Detection, Scaffolding Fading,
  and Real-Time Estimation Systems
- Expertise reversal effect (d=0.45-2.99): scaffolding helps novices but hinders experts
- Backward fading strategy: remove scaffolds from end of problem first
- Response time analysis: key indicator of cognitive load

Three types of cognitive load:
1. Intrinsic: Inherent complexity of the material
2. Extraneous: Load from poor instructional design
3. Germane: Load from schema construction (desirable)

Goal: Minimize extraneous, optimize intrinsic + germane
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import statistics
import math


class CognitiveLoadLevel(str, Enum):
    """Cognitive load levels"""
    LOW = "low"           # Under-challenged, potential boredom
    OPTIMAL = "optimal"   # In flow state, effective learning
    HIGH = "high"         # Challenged but manageable
    OVERLOAD = "overload" # Exceeds working memory capacity


class ExpertiseLevel(str, Enum):
    """Learner expertise levels for expertise reversal effect"""
    NOVICE = "novice"           # Needs full scaffolding
    BEGINNER = "beginner"       # Needs partial scaffolding
    INTERMEDIATE = "intermediate"  # Minimal scaffolding
    ADVANCED = "advanced"       # No scaffolding needed
    EXPERT = "expert"           # Scaffolding may hinder


@dataclass
class ResponseMetrics:
    """Metrics from a single response/interaction"""
    response_time_ms: int      # Time to respond in milliseconds
    correct: bool              # Whether response was correct
    confidence: Optional[float] = None  # Self-reported confidence (0-1)
    hint_used: bool = False    # Whether hints were used
    attempts: int = 1          # Number of attempts before correct
    content_difficulty: float = 5.0  # Content difficulty (1-10)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CognitiveLoadEstimate:
    """Estimated cognitive load state"""
    level: CognitiveLoadLevel
    score: float  # 0-1 scale (0=low, 1=overload)
    intrinsic_load: float  # From content complexity
    extraneous_load: float  # From presentation/interface
    germane_load: float  # From active learning
    confidence: float  # Confidence in estimate (0-1)
    indicators: Dict[str, float]  # Individual indicator scores


@dataclass
class ScaffoldingRecommendation:
    """Recommendation for scaffolding level"""
    expertise_level: ExpertiseLevel
    scaffolding_level: float  # 0-1 (0=none, 1=full)
    fade_strategy: str  # "none", "backward", "forward", "middle-out"
    specific_recommendations: List[str]
    rationale: str


class CognitiveLoadEstimator:
    """
    Real-time cognitive load estimation based on behavioral indicators

    Research-backed indicators:
    1. Response time patterns (primary indicator)
    2. Error rates and patterns
    3. Hint usage frequency
    4. Re-reading/backtracking behavior
    5. Answer changes
    6. Pause patterns

    Expertise reversal effect:
    - Novices: Full scaffolding beneficial (d=0.45-2.99)
    - Experts: Scaffolding can be detrimental
    - Implement adaptive fading based on expertise detection
    """

    # Response time thresholds (in ms) - calibrated per content type
    DEFAULT_RT_THRESHOLDS = {
        "question": {
            "fast": 3000,      # < 3s = possibly guessing
            "normal_min": 5000,  # 5-15s = normal processing
            "normal_max": 15000,
            "slow": 30000,     # > 30s = struggling
        },
        "video": {
            "fast": 0.5,       # ratio of video length watched
            "normal_min": 0.8,
            "normal_max": 1.2,
            "slow": 2.0,
        },
        "reading": {
            "wpm_fast": 400,   # words per minute
            "wpm_normal_min": 200,
            "wpm_normal_max": 300,
            "wpm_slow": 100,
        }
    }

    # Expertise detection thresholds
    EXPERTISE_THRESHOLDS = {
        "novice": {"accuracy": 0.5, "rt_consistency": 0.3},
        "beginner": {"accuracy": 0.65, "rt_consistency": 0.4},
        "intermediate": {"accuracy": 0.8, "rt_consistency": 0.6},
        "advanced": {"accuracy": 0.9, "rt_consistency": 0.75},
        "expert": {"accuracy": 0.95, "rt_consistency": 0.85},
    }

    def __init__(
        self,
        rt_thresholds: Optional[Dict] = None,
        window_size: int = 10,  # Number of recent responses to consider
        expertise_window: int = 20,  # Responses for expertise detection
    ):
        """
        Initialize cognitive load estimator

        Args:
            rt_thresholds: Custom response time thresholds
            window_size: Number of recent responses for load estimation
            expertise_window: Number of responses for expertise detection
        """
        self.rt_thresholds = rt_thresholds or self.DEFAULT_RT_THRESHOLDS
        self.window_size = window_size
        self.expertise_window = expertise_window

    def estimate_cognitive_load(
        self,
        recent_metrics: List[ResponseMetrics],
        content_difficulty: float = 5.0,
        user_expertise: Optional[ExpertiseLevel] = None,
    ) -> CognitiveLoadEstimate:
        """
        Estimate current cognitive load from recent behavior

        Args:
            recent_metrics: List of recent response metrics
            content_difficulty: Current content difficulty (1-10)
            user_expertise: Known expertise level (if available)

        Returns:
            Cognitive load estimate with breakdown
        """
        if not recent_metrics:
            return CognitiveLoadEstimate(
                level=CognitiveLoadLevel.OPTIMAL,
                score=0.5,
                intrinsic_load=content_difficulty / 10,
                extraneous_load=0.0,
                germane_load=0.3,
                confidence=0.0,
                indicators={}
            )

        # Use most recent responses up to window size
        metrics = recent_metrics[-self.window_size:]

        # Calculate individual indicators
        indicators = {}

        # 1. Response time indicator
        rt_indicator = self._calculate_rt_indicator(metrics)
        indicators["response_time"] = rt_indicator

        # 2. Accuracy indicator (inverse - low accuracy = high load)
        accuracy = sum(1 for m in metrics if m.correct) / len(metrics)
        indicators["accuracy"] = 1 - accuracy

        # 3. Hint usage indicator
        hint_rate = sum(1 for m in metrics if m.hint_used) / len(metrics)
        indicators["hint_usage"] = hint_rate

        # 4. Multiple attempts indicator
        avg_attempts = statistics.mean(m.attempts for m in metrics)
        indicators["retry_rate"] = min(1.0, (avg_attempts - 1) / 3)

        # 5. Response time variability (high variability = struggle)
        if len(metrics) > 2:
            rt_values = [m.response_time_ms for m in metrics]
            rt_cv = statistics.stdev(rt_values) / statistics.mean(rt_values) if statistics.mean(rt_values) > 0 else 0
            indicators["rt_variability"] = min(1.0, rt_cv)
        else:
            indicators["rt_variability"] = 0.5

        # 6. Self-reported confidence (if available)
        confidences = [m.confidence for m in metrics if m.confidence is not None]
        if confidences:
            indicators["self_confidence"] = 1 - statistics.mean(confidences)
        else:
            indicators["self_confidence"] = 0.5

        # Calculate weighted cognitive load score
        weights = {
            "response_time": 0.25,
            "accuracy": 0.25,
            "hint_usage": 0.15,
            "retry_rate": 0.15,
            "rt_variability": 0.1,
            "self_confidence": 0.1,
        }

        load_score = sum(
            indicators[k] * weights[k]
            for k in weights.keys()
        )

        # Adjust for content difficulty
        intrinsic_load = content_difficulty / 10
        load_score = load_score * 0.7 + intrinsic_load * 0.3

        # Bound the score
        load_score = max(0.0, min(1.0, load_score))

        # Determine load level
        if load_score < 0.25:
            level = CognitiveLoadLevel.LOW
        elif load_score < 0.5:
            level = CognitiveLoadLevel.OPTIMAL
        elif load_score < 0.75:
            level = CognitiveLoadLevel.HIGH
        else:
            level = CognitiveLoadLevel.OVERLOAD

        # Calculate load components
        # Intrinsic: from content
        intrinsic = intrinsic_load

        # Extraneous: from hint usage, retries (poor interface/instruction)
        extraneous = (indicators["hint_usage"] + indicators["retry_rate"]) / 2

        # Germane: from active processing (moderate RT, some struggle but success)
        if accuracy > 0.6 and 0.3 < rt_indicator < 0.7:
            germane = 0.5 + (accuracy - 0.6) * 0.5
        else:
            germane = 0.2

        # Confidence based on sample size and consistency
        confidence = min(1.0, len(metrics) / self.window_size)
        if indicators["rt_variability"] > 0.5:
            confidence *= 0.8  # High variability reduces confidence

        return CognitiveLoadEstimate(
            level=level,
            score=load_score,
            intrinsic_load=intrinsic,
            extraneous_load=extraneous,
            germane_load=germane,
            confidence=confidence,
            indicators=indicators
        )

    def _calculate_rt_indicator(self, metrics: List[ResponseMetrics]) -> float:
        """
        Calculate response time indicator

        Returns:
            0 = very fast (possibly guessing)
            0.5 = normal processing
            1 = very slow (overloaded)
        """
        if not metrics:
            return 0.5

        thresholds = self.rt_thresholds["question"]
        rt_values = [m.response_time_ms for m in metrics]
        avg_rt = statistics.mean(rt_values)

        if avg_rt < thresholds["fast"]:
            return 0.1  # Too fast - possibly guessing
        elif avg_rt < thresholds["normal_min"]:
            # Linear interpolation from fast to normal
            return 0.1 + 0.3 * (avg_rt - thresholds["fast"]) / (thresholds["normal_min"] - thresholds["fast"])
        elif avg_rt <= thresholds["normal_max"]:
            return 0.4 + 0.2 * (avg_rt - thresholds["normal_min"]) / (thresholds["normal_max"] - thresholds["normal_min"])
        elif avg_rt < thresholds["slow"]:
            return 0.6 + 0.3 * (avg_rt - thresholds["normal_max"]) / (thresholds["slow"] - thresholds["normal_max"])
        else:
            return min(1.0, 0.9 + 0.1 * (avg_rt - thresholds["slow"]) / thresholds["slow"])

    def detect_expertise(
        self,
        all_metrics: List[ResponseMetrics],
        concept_id: Optional[int] = None,
    ) -> Tuple[ExpertiseLevel, float]:
        """
        Detect user's expertise level from performance history

        Uses:
        - Accuracy patterns
        - Response time consistency
        - Learning trajectory

        Args:
            all_metrics: Complete performance history
            concept_id: Optional concept to filter by

        Returns:
            (Expertise level, confidence score)
        """
        if not all_metrics:
            return ExpertiseLevel.NOVICE, 0.0

        metrics = all_metrics[-self.expertise_window:]

        # Calculate accuracy
        accuracy = sum(1 for m in metrics if m.correct) / len(metrics)

        # Calculate response time consistency (lower CV = more consistent)
        rt_values = [m.response_time_ms for m in metrics if m.response_time_ms > 0]
        if len(rt_values) > 1:
            rt_mean = statistics.mean(rt_values)
            rt_std = statistics.stdev(rt_values)
            rt_consistency = 1 - min(1.0, rt_std / rt_mean) if rt_mean > 0 else 0
        else:
            rt_consistency = 0.5

        # Calculate learning trajectory (improvement over time)
        if len(metrics) >= 4:
            first_half = metrics[:len(metrics)//2]
            second_half = metrics[len(metrics)//2:]
            first_accuracy = sum(1 for m in first_half if m.correct) / len(first_half)
            second_accuracy = sum(1 for m in second_half if m.correct) / len(second_half)
            improvement = second_accuracy - first_accuracy
        else:
            improvement = 0

        # Determine expertise level
        expertise_level = ExpertiseLevel.NOVICE

        for level_name, thresholds in self.EXPERTISE_THRESHOLDS.items():
            if accuracy >= thresholds["accuracy"] and rt_consistency >= thresholds["rt_consistency"]:
                expertise_level = ExpertiseLevel(level_name)

        # Calculate confidence based on sample size
        confidence = min(1.0, len(metrics) / self.expertise_window)

        # Boost confidence if metrics are consistent
        if rt_consistency > 0.7:
            confidence = min(1.0, confidence * 1.1)

        return expertise_level, confidence

    def recommend_scaffolding(
        self,
        expertise_level: ExpertiseLevel,
        cognitive_load: CognitiveLoadEstimate,
        content_type: str = "problem",
    ) -> ScaffoldingRecommendation:
        """
        Recommend scaffolding level based on expertise and cognitive load

        Implements expertise reversal effect:
        - Novices benefit from scaffolding (d=0.45-2.99)
        - Experts may be hindered by scaffolding

        Implements backward fading strategy:
        - Remove scaffolds from end of problem first
        - Maintains initial guidance while building independence

        Args:
            expertise_level: Detected expertise level
            cognitive_load: Current cognitive load estimate
            content_type: Type of content ("problem", "reading", "video")

        Returns:
            Scaffolding recommendation
        """
        recommendations = []
        rationale_parts = []

        # Base scaffolding level from expertise
        expertise_scaffolding = {
            ExpertiseLevel.NOVICE: 1.0,
            ExpertiseLevel.BEGINNER: 0.8,
            ExpertiseLevel.INTERMEDIATE: 0.5,
            ExpertiseLevel.ADVANCED: 0.2,
            ExpertiseLevel.EXPERT: 0.0,
        }

        base_level = expertise_scaffolding[expertise_level]

        # Adjust for cognitive load
        if cognitive_load.level == CognitiveLoadLevel.OVERLOAD:
            # Increase scaffolding if overloaded (even for advanced users)
            adjusted_level = min(1.0, base_level + 0.3)
            rationale_parts.append("Increased scaffolding due to cognitive overload")
        elif cognitive_load.level == CognitiveLoadLevel.LOW:
            # Decrease scaffolding if under-challenged
            adjusted_level = max(0.0, base_level - 0.2)
            rationale_parts.append("Reduced scaffolding to increase challenge")
        else:
            adjusted_level = base_level

        # Determine fade strategy
        if expertise_level in [ExpertiseLevel.NOVICE, ExpertiseLevel.BEGINNER]:
            fade_strategy = "none"  # No fading yet
            rationale_parts.append("Full scaffolding maintained for novice learner")
        elif expertise_level == ExpertiseLevel.INTERMEDIATE:
            fade_strategy = "backward"  # Remove end scaffolds first
            rationale_parts.append("Backward fading: removing final step hints")
        elif expertise_level == ExpertiseLevel.ADVANCED:
            fade_strategy = "middle-out"  # Remove middle scaffolds
            rationale_parts.append("Middle-out fading: maintaining only start/end guidance")
        else:
            fade_strategy = "forward"  # Remove initial scaffolds
            rationale_parts.append("Forward fading: expert needs minimal guidance")

        # Generate specific recommendations based on level
        if adjusted_level > 0.8:
            recommendations = [
                "Provide worked examples before practice",
                "Show step-by-step solution process",
                "Offer hints at each step",
                "Use completion problems (partial solutions)",
            ]
        elif adjusted_level > 0.5:
            recommendations = [
                "Provide initial problem setup only",
                "Offer hints on request",
                "Show similar solved examples",
                "Use faded worked examples",
            ]
        elif adjusted_level > 0.2:
            recommendations = [
                "Minimal guidance - goal only",
                "Hints available but not prominent",
                "Encourage self-explanation",
                "Use interleaved practice",
            ]
        else:
            recommendations = [
                "No scaffolding - full problem",
                "Hide hints unless requested",
                "Focus on transfer problems",
                "Introduce variations and challenges",
            ]

        # Add expertise reversal warning if applicable
        if expertise_level in [ExpertiseLevel.ADVANCED, ExpertiseLevel.EXPERT] and adjusted_level > 0.3:
            rationale_parts.append(
                "WARNING: Excessive scaffolding may hinder expert learner (expertise reversal effect)"
            )

        return ScaffoldingRecommendation(
            expertise_level=expertise_level,
            scaffolding_level=adjusted_level,
            fade_strategy=fade_strategy,
            specific_recommendations=recommendations,
            rationale=" | ".join(rationale_parts)
        )

    def calculate_optimal_difficulty(
        self,
        expertise_level: ExpertiseLevel,
        current_mastery: float,
        cognitive_load: CognitiveLoadEstimate,
    ) -> Tuple[float, float]:
        """
        Calculate optimal content difficulty range

        Based on:
        - Expertise level
        - Current mastery
        - Cognitive load state
        - ZPD principles (slightly above current ability)

        Args:
            expertise_level: User's expertise level
            current_mastery: Mastery level (0-1)
            cognitive_load: Current cognitive load

        Returns:
            (min_difficulty, max_difficulty) on 1-10 scale
        """
        # Base difficulty from mastery (1-10 scale)
        base_min = current_mastery * 10
        base_max = min(10, base_min + 2)  # ZPD range

        # Adjust for cognitive load
        if cognitive_load.level == CognitiveLoadLevel.OVERLOAD:
            # Reduce difficulty if overloaded
            adjustment = -1.5
        elif cognitive_load.level == CognitiveLoadLevel.HIGH:
            # Slight reduction
            adjustment = -0.5
        elif cognitive_load.level == CognitiveLoadLevel.LOW:
            # Increase difficulty if bored
            adjustment = 1.0
        else:
            adjustment = 0

        # Adjust for expertise
        expertise_bonus = {
            ExpertiseLevel.NOVICE: -1.0,
            ExpertiseLevel.BEGINNER: -0.5,
            ExpertiseLevel.INTERMEDIATE: 0,
            ExpertiseLevel.ADVANCED: 0.5,
            ExpertiseLevel.EXPERT: 1.0,
        }

        total_adjustment = adjustment + expertise_bonus[expertise_level]

        min_difficulty = max(1, min(9, base_min + total_adjustment))
        max_difficulty = max(2, min(10, base_max + total_adjustment))

        # Ensure min < max
        if min_difficulty >= max_difficulty:
            max_difficulty = min(10, min_difficulty + 1)

        return min_difficulty, max_difficulty

    def should_intervene(
        self,
        cognitive_load: CognitiveLoadEstimate,
        consecutive_errors: int = 0,
        time_on_task_minutes: float = 0,
    ) -> Tuple[bool, str]:
        """
        Determine if intervention is needed

        Triggers:
        - Cognitive overload detected
        - Multiple consecutive errors
        - Extended time without progress
        - Declining performance trend

        Args:
            cognitive_load: Current cognitive load estimate
            consecutive_errors: Number of consecutive wrong answers
            time_on_task_minutes: Time spent on current task

        Returns:
            (Should intervene?, Intervention type)
        """
        # Immediate intervention for overload
        if cognitive_load.level == CognitiveLoadLevel.OVERLOAD:
            return True, "cognitive_overload"

        # Intervention after multiple consecutive errors
        if consecutive_errors >= 3:
            return True, "repeated_failure"

        # Intervention for extended struggle (> 10 min on single item)
        if time_on_task_minutes > 10:
            return True, "time_limit"

        # Check for high extraneous load (interface/design issues)
        if cognitive_load.extraneous_load > 0.6:
            return True, "extraneous_load"

        # No intervention needed
        return False, "none"

    def get_intervention_recommendation(
        self,
        intervention_type: str,
        expertise_level: ExpertiseLevel,
    ) -> Dict[str, any]:
        """
        Get specific intervention recommendation

        Args:
            intervention_type: Type of intervention needed
            expertise_level: User's expertise level

        Returns:
            Intervention details
        """
        interventions = {
            "cognitive_overload": {
                "action": "reduce_complexity",
                "message": "Let's break this down into smaller steps.",
                "recommendations": [
                    "Simplify the current problem",
                    "Provide worked example",
                    "Take a short break",
                    "Review prerequisite concepts",
                ],
            },
            "repeated_failure": {
                "action": "provide_hint",
                "message": "Here's a hint to help you along.",
                "recommendations": [
                    "Show progressive hints",
                    "Demonstrate first step",
                    "Link to related concept review",
                ],
            },
            "time_limit": {
                "action": "offer_help",
                "message": "Would you like some help with this?",
                "recommendations": [
                    "Offer to skip or simplify",
                    "Suggest taking a break",
                    "Provide scaffolded version",
                ],
            },
            "extraneous_load": {
                "action": "simplify_interface",
                "message": "Let me simplify this for you.",
                "recommendations": [
                    "Remove distracting elements",
                    "Focus on essential information",
                    "Provide clearer instructions",
                ],
            },
            "none": {
                "action": "none",
                "message": "",
                "recommendations": [],
            },
        }

        intervention = interventions.get(intervention_type, interventions["none"])

        # Adjust for expertise
        if expertise_level in [ExpertiseLevel.ADVANCED, ExpertiseLevel.EXPERT]:
            intervention["message"] = intervention["message"].replace(
                "Let's break this down",
                "This is a challenging problem"
            )

        return intervention
