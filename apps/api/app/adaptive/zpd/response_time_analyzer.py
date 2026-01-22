"""
Response Time Analysis for ZPD Enhancement
Integrates response time patterns into difficulty calibration

Research basis:
- Operationalizing ZPD in Adaptive Learning Systems
- Response time as indicator of cognitive load
- Dual-Process Theory (fast System 1 vs slow System 2)

Key patterns:
- Fast + Correct → Content may be too easy (automaticity achieved)
- Slow + Correct → Optimal challenge (effortful processing)
- Fast + Wrong → Possible misconception or guessing
- Slow + Wrong → Content too hard or confusion

Uses:
- Refine ZPD calculations with time data
- Detect cognitive load levels
- Identify when mastery is superficial vs deep
- Improve difficulty adaptation
"""
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
import math


class ResponsePattern(str, Enum):
    """Response pattern based on time and accuracy"""
    FAST_CORRECT = "fast_correct"      # Mastered, possibly too easy
    SLOW_CORRECT = "slow_correct"      # Optimal challenge
    FAST_WRONG = "fast_wrong"          # Possible misconception
    SLOW_WRONG = "slow_wrong"          # Too hard or confused
    NORMAL_CORRECT = "normal_correct"  # Good difficulty
    NORMAL_WRONG = "normal_wrong"      # Learning opportunity


class CognitiveState(str, Enum):
    """Inferred cognitive state from response patterns"""
    AUTOMATICITY = "automaticity"      # Effortless, automated responses
    FLOW = "flow"                      # Optimal engagement
    STRUGGLING = "struggling"          # Having difficulty
    GUESSING = "guessing"              # Not engaging properly
    CONFUSED = "confused"              # Needs scaffolding


@dataclass
class ResponseTimeData:
    """Single response with timing data"""
    question_id: int
    concept_id: int
    is_correct: bool
    response_time_ms: int
    timestamp: datetime = field(default_factory=datetime.now)
    difficulty: float = 0.5  # Question difficulty 0-1
    confidence: Optional[float] = None


@dataclass
class ConceptTimeProfile:
    """Response time profile for a concept"""
    concept_id: int
    mean_time_correct: float = 0.0
    mean_time_wrong: float = 0.0
    std_time_correct: float = 0.0
    std_time_wrong: float = 0.0
    sample_count: int = 0
    baseline_time: float = 5000.0  # Default baseline in ms


@dataclass
class TimeAnalysisResult:
    """Result of response time analysis"""
    pattern: ResponsePattern
    cognitive_state: CognitiveState
    zpd_adjustment: float  # How much to adjust ZPD score (-1 to 1)
    time_percentile: float  # Where this time falls in distribution
    signals: List[str]
    recommendations: List[str]


@dataclass
class UserTimeProfile:
    """User's overall response time profile"""
    user_id: int
    base_speed: float = 1.0  # Relative to average (1.0 = average)
    consistency: float = 0.5  # How consistent their times are (0-1)
    concept_profiles: Dict[int, ConceptTimeProfile] = field(default_factory=dict)
    response_history: List[ResponseTimeData] = field(default_factory=list)


class ResponseTimeAnalyzer:
    """
    Analyzes response times to enhance ZPD calculations

    Integrates with ZPDRegulator to provide more nuanced
    difficulty recommendations based on timing patterns.
    """

    # Time thresholds (in standard deviations from mean)
    FAST_THRESHOLD = -1.0   # Below 1 SD = fast
    SLOW_THRESHOLD = 1.5    # Above 1.5 SD = slow

    # Baseline times by question type (ms)
    DEFAULT_BASELINES = {
        "multiple_choice": 8000,
        "true_false": 4000,
        "short_answer": 15000,
        "calculation": 20000,
        "default": 10000,
    }

    # Minimum responses for reliable analysis
    MIN_SAMPLES = 5

    def __init__(
        self,
        fast_threshold: float = -1.0,
        slow_threshold: float = 1.5,
    ):
        self.fast_threshold = fast_threshold
        self.slow_threshold = slow_threshold
        self.user_profiles: Dict[int, UserTimeProfile] = {}
        self.concept_baselines: Dict[int, float] = {}

    def record_response(
        self,
        user_id: int,
        question_id: int,
        concept_id: int,
        is_correct: bool,
        response_time_ms: int,
        difficulty: float = 0.5,
        confidence: float = None,
    ):
        """Record a response for time analysis"""
        profile = self._get_or_create_profile(user_id)

        data = ResponseTimeData(
            question_id=question_id,
            concept_id=concept_id,
            is_correct=is_correct,
            response_time_ms=response_time_ms,
            difficulty=difficulty,
            confidence=confidence,
        )

        profile.response_history.append(data)

        # Update concept profile
        self._update_concept_profile(profile, concept_id, is_correct, response_time_ms)

        # Update user base speed
        self._update_user_speed_profile(profile)

        # Prune old history (keep last 500 responses)
        if len(profile.response_history) > 500:
            profile.response_history = profile.response_history[-500:]

    def analyze_response(
        self,
        user_id: int,
        response: ResponseTimeData,
    ) -> TimeAnalysisResult:
        """
        Analyze a single response for patterns and cognitive state

        Returns analysis with ZPD adjustment recommendation
        """
        profile = self._get_or_create_profile(user_id)
        concept_profile = profile.concept_profiles.get(response.concept_id)

        # Determine baseline for comparison
        if concept_profile and concept_profile.sample_count >= self.MIN_SAMPLES:
            if response.is_correct:
                baseline = concept_profile.mean_time_correct
                std = concept_profile.std_time_correct or baseline * 0.3
            else:
                baseline = concept_profile.mean_time_wrong
                std = concept_profile.std_time_wrong or baseline * 0.3
        else:
            baseline = self.concept_baselines.get(
                response.concept_id,
                self.DEFAULT_BASELINES["default"]
            )
            std = baseline * 0.3  # Assume 30% standard deviation

        # Calculate z-score
        if std > 0:
            z_score = (response.response_time_ms - baseline) / std
        else:
            z_score = 0

        # Determine pattern
        pattern = self._classify_pattern(z_score, response.is_correct)

        # Infer cognitive state
        cognitive_state = self._infer_cognitive_state(pattern, response, profile)

        # Calculate ZPD adjustment
        zpd_adjustment = self._calculate_zpd_adjustment(pattern, cognitive_state)

        # Calculate time percentile
        time_percentile = self._calculate_percentile(z_score)

        # Generate signals and recommendations
        signals = self._generate_signals(pattern, z_score, response)
        recommendations = self._generate_recommendations(pattern, cognitive_state)

        return TimeAnalysisResult(
            pattern=pattern,
            cognitive_state=cognitive_state,
            zpd_adjustment=zpd_adjustment,
            time_percentile=time_percentile,
            signals=signals,
            recommendations=recommendations,
        )

    def get_zpd_enhancement(
        self,
        user_id: int,
        concept_id: int,
        base_zpd_score: float,
    ) -> Dict[str, Any]:
        """
        Enhance ZPD score with response time analysis

        Args:
            user_id: User ID
            concept_id: Concept being assessed
            base_zpd_score: Original ZPD score from ZPDRegulator

        Returns:
            Enhanced ZPD analysis with time-based adjustments
        """
        profile = self._get_or_create_profile(user_id)

        # Get recent responses for this concept
        recent = [
            r for r in profile.response_history[-50:]
            if r.concept_id == concept_id
        ]

        if len(recent) < 3:
            return {
                "enhanced_zpd_score": base_zpd_score,
                "adjustment": 0.0,
                "confidence": "low",
                "time_patterns": {},
                "recommendation": "Insufficient data for time-based enhancement",
            }

        # Analyze patterns
        patterns = {}
        for r in recent:
            analysis = self.analyze_response(user_id, r)
            patterns[analysis.pattern.value] = patterns.get(analysis.pattern.value, 0) + 1

        # Calculate overall adjustment
        total_adjustment = 0.0
        for r in recent:
            analysis = self.analyze_response(user_id, r)
            total_adjustment += analysis.zpd_adjustment

        avg_adjustment = total_adjustment / len(recent)

        # Apply adjustment to ZPD score
        enhanced_score = max(0.0, min(1.0, base_zpd_score + avg_adjustment))

        # Determine primary pattern
        primary_pattern = max(patterns, key=patterns.get) if patterns else "unknown"

        # Generate recommendation based on patterns
        if patterns.get("fast_correct", 0) > len(recent) * 0.5:
            recommendation = "Content may be too easy - consider increasing difficulty"
        elif patterns.get("slow_wrong", 0) > len(recent) * 0.3:
            recommendation = "Content may be too hard - consider scaffolding"
        elif patterns.get("fast_wrong", 0) > len(recent) * 0.3:
            recommendation = "Possible misconception - review fundamentals"
        else:
            recommendation = "Difficulty level appears appropriate"

        return {
            "enhanced_zpd_score": round(enhanced_score, 4),
            "base_zpd_score": base_zpd_score,
            "adjustment": round(avg_adjustment, 4),
            "confidence": "high" if len(recent) >= 10 else "medium",
            "time_patterns": patterns,
            "primary_pattern": primary_pattern,
            "recommendation": recommendation,
        }

    def estimate_cognitive_load(
        self,
        user_id: int,
        concept_id: int,
    ) -> Dict[str, Any]:
        """
        Estimate cognitive load based on response times

        Uses response time patterns to infer cognitive load level
        """
        profile = self._get_or_create_profile(user_id)
        concept_profile = profile.concept_profiles.get(concept_id)

        if not concept_profile or concept_profile.sample_count < self.MIN_SAMPLES:
            return {
                "cognitive_load": "unknown",
                "load_score": 0.5,
                "confidence": "low",
                "indicators": [],
            }

        recent = [
            r for r in profile.response_history[-20:]
            if r.concept_id == concept_id
        ]

        if not recent:
            return {
                "cognitive_load": "unknown",
                "load_score": 0.5,
                "confidence": "low",
                "indicators": [],
            }

        # Calculate indicators
        avg_time = statistics.mean(r.response_time_ms for r in recent)
        time_variance = statistics.variance(
            [r.response_time_ms for r in recent]
        ) if len(recent) > 1 else 0
        accuracy = sum(1 for r in recent if r.is_correct) / len(recent)

        # Compare to baseline
        baseline = concept_profile.mean_time_correct or self.DEFAULT_BASELINES["default"]

        # Cognitive load indicators
        indicators = []

        # Time relative to baseline
        time_ratio = avg_time / baseline if baseline > 0 else 1.0

        if time_ratio > 1.5:
            indicators.append("Response times elevated")
        elif time_ratio < 0.7:
            indicators.append("Response times very fast")

        # Accuracy pattern
        if accuracy < 0.5:
            indicators.append("Low accuracy suggesting overload")
        elif accuracy > 0.9:
            indicators.append("High accuracy suggesting low load")

        # Variance in times (high variance = fluctuating load)
        if concept_profile.std_time_correct > 0:
            cv = math.sqrt(time_variance) / avg_time if avg_time > 0 else 0
            if cv > 0.5:
                indicators.append("High time variability")

        # Calculate load score (0 = low load, 1 = high load)
        load_score = 0.5

        if time_ratio > 1.0:
            load_score += min(0.3, (time_ratio - 1.0) * 0.3)
        else:
            load_score -= min(0.3, (1.0 - time_ratio) * 0.3)

        if accuracy < 0.6:
            load_score += 0.2
        elif accuracy > 0.8:
            load_score -= 0.1

        load_score = max(0.0, min(1.0, load_score))

        # Determine load level
        if load_score < 0.3:
            load_level = "low"
        elif load_score < 0.6:
            load_level = "optimal"
        elif load_score < 0.8:
            load_level = "high"
        else:
            load_level = "overload"

        return {
            "cognitive_load": load_level,
            "load_score": round(load_score, 3),
            "confidence": "high" if len(recent) >= 10 else "medium",
            "indicators": indicators,
            "time_ratio": round(time_ratio, 2),
            "accuracy": round(accuracy, 3),
        }

    def get_optimal_pacing(
        self,
        user_id: int,
        concept_id: int,
    ) -> Dict[str, Any]:
        """
        Recommend optimal pacing based on response patterns

        Returns timing recommendations for content delivery
        """
        profile = self._get_or_create_profile(user_id)
        concept_profile = profile.concept_profiles.get(concept_id)

        if not concept_profile or concept_profile.sample_count < self.MIN_SAMPLES:
            return {
                "recommended_time_per_question_ms": self.DEFAULT_BASELINES["default"],
                "recommended_questions_per_session": 10,
                "break_frequency": "every 15 minutes",
                "confidence": "low",
            }

        # Calculate user's typical response time
        typical_time = concept_profile.mean_time_correct

        # Estimate optimal session length
        # Research: cognitive fatigue increases after ~20 minutes of focused work
        optimal_questions = int(min(20, 1200000 / typical_time))  # Max 20 min worth

        # Break frequency based on load
        load_estimate = self.estimate_cognitive_load(user_id, concept_id)
        if load_estimate["cognitive_load"] == "high":
            break_frequency = "every 10 minutes"
        elif load_estimate["cognitive_load"] == "overload":
            break_frequency = "every 5 minutes"
        else:
            break_frequency = "every 15-20 minutes"

        return {
            "recommended_time_per_question_ms": int(typical_time * 1.2),  # Allow 20% buffer
            "recommended_questions_per_session": optimal_questions,
            "break_frequency": break_frequency,
            "user_speed_factor": round(profile.base_speed, 2),
            "confidence": "high" if concept_profile.sample_count >= 20 else "medium",
        }

    def _get_or_create_profile(self, user_id: int) -> UserTimeProfile:
        """Get or create user time profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserTimeProfile(user_id=user_id)
        return self.user_profiles[user_id]

    def _update_concept_profile(
        self,
        profile: UserTimeProfile,
        concept_id: int,
        is_correct: bool,
        response_time_ms: int,
    ):
        """Update concept time profile with new response"""
        if concept_id not in profile.concept_profiles:
            profile.concept_profiles[concept_id] = ConceptTimeProfile(
                concept_id=concept_id
            )

        cp = profile.concept_profiles[concept_id]
        cp.sample_count += 1

        # Get all times for this concept
        concept_responses = [
            r for r in profile.response_history
            if r.concept_id == concept_id
        ]

        correct_times = [r.response_time_ms for r in concept_responses if r.is_correct]
        wrong_times = [r.response_time_ms for r in concept_responses if not r.is_correct]

        if correct_times:
            cp.mean_time_correct = statistics.mean(correct_times)
            if len(correct_times) > 1:
                cp.std_time_correct = statistics.stdev(correct_times)

        if wrong_times:
            cp.mean_time_wrong = statistics.mean(wrong_times)
            if len(wrong_times) > 1:
                cp.std_time_wrong = statistics.stdev(wrong_times)

    def _update_user_speed_profile(self, profile: UserTimeProfile):
        """Update user's overall speed profile"""
        if len(profile.response_history) < self.MIN_SAMPLES:
            return

        # Calculate user's average speed relative to baseline
        times = [r.response_time_ms for r in profile.response_history[-50:]]
        avg_time = statistics.mean(times)

        # Compare to global baseline
        global_baseline = self.DEFAULT_BASELINES["default"]
        profile.base_speed = global_baseline / avg_time if avg_time > 0 else 1.0

        # Calculate consistency
        if len(times) > 1:
            cv = statistics.stdev(times) / avg_time if avg_time > 0 else 0
            profile.consistency = max(0, 1 - cv)  # Lower CV = higher consistency

    def _classify_pattern(self, z_score: float, is_correct: bool) -> ResponsePattern:
        """Classify response pattern based on z-score and correctness"""
        if z_score < self.fast_threshold:
            # Fast response
            return ResponsePattern.FAST_CORRECT if is_correct else ResponsePattern.FAST_WRONG
        elif z_score > self.slow_threshold:
            # Slow response
            return ResponsePattern.SLOW_CORRECT if is_correct else ResponsePattern.SLOW_WRONG
        else:
            # Normal response
            return ResponsePattern.NORMAL_CORRECT if is_correct else ResponsePattern.NORMAL_WRONG

    def _infer_cognitive_state(
        self,
        pattern: ResponsePattern,
        response: ResponseTimeData,
        profile: UserTimeProfile,
    ) -> CognitiveState:
        """Infer cognitive state from response pattern"""
        # Check recent history for context
        recent = profile.response_history[-10:]
        recent_patterns = [
            self.analyze_response(profile.user_id, r).pattern
            if hasattr(self, '_analyzing') else pattern  # Avoid recursion
            for r in recent
            if r.concept_id == response.concept_id
        ]

        if pattern == ResponsePattern.FAST_CORRECT:
            # Fast correct - likely automated/mastered
            return CognitiveState.AUTOMATICITY
        elif pattern == ResponsePattern.SLOW_CORRECT:
            # Slow correct - effortful but successful
            return CognitiveState.FLOW
        elif pattern == ResponsePattern.FAST_WRONG:
            # Fast wrong - guessing or misconception
            return CognitiveState.GUESSING
        elif pattern == ResponsePattern.SLOW_WRONG:
            # Slow wrong - struggling
            return CognitiveState.STRUGGLING
        elif pattern == ResponsePattern.NORMAL_CORRECT:
            return CognitiveState.FLOW
        else:
            return CognitiveState.STRUGGLING

    def _calculate_zpd_adjustment(
        self,
        pattern: ResponsePattern,
        cognitive_state: CognitiveState,
    ) -> float:
        """Calculate ZPD score adjustment based on pattern"""
        # Adjustments based on pattern
        adjustments = {
            ResponsePattern.FAST_CORRECT: -0.15,    # Too easy, decrease ZPD match
            ResponsePattern.SLOW_CORRECT: 0.1,      # Good challenge, boost ZPD match
            ResponsePattern.FAST_WRONG: -0.1,       # Problematic, needs intervention
            ResponsePattern.SLOW_WRONG: -0.2,       # Too hard, decrease ZPD match
            ResponsePattern.NORMAL_CORRECT: 0.05,   # Good fit
            ResponsePattern.NORMAL_WRONG: -0.05,    # Slight adjustment
        }

        return adjustments.get(pattern, 0.0)

    def _calculate_percentile(self, z_score: float) -> float:
        """Calculate percentile from z-score (simplified normal CDF)"""
        # Approximate standard normal CDF
        # Using logistic approximation: 1 / (1 + exp(-1.7 * z))
        return 1 / (1 + math.exp(-1.7 * z_score))

    def _generate_signals(
        self,
        pattern: ResponsePattern,
        z_score: float,
        response: ResponseTimeData,
    ) -> List[str]:
        """Generate signals based on pattern analysis"""
        signals = []

        if abs(z_score) > 2:
            signals.append(f"Response time significantly {'fast' if z_score < 0 else 'slow'}")

        if pattern == ResponsePattern.FAST_WRONG:
            signals.append("Fast incorrect answer - possible misconception or guess")
        elif pattern == ResponsePattern.SLOW_WRONG:
            signals.append("Slow incorrect answer - content may be too challenging")
        elif pattern == ResponsePattern.FAST_CORRECT:
            signals.append("Fast correct answer - content well mastered")

        if response.confidence and response.confidence > 0.8 and not response.is_correct:
            signals.append("High confidence but incorrect - likely misconception")

        return signals

    def _generate_recommendations(
        self,
        pattern: ResponsePattern,
        cognitive_state: CognitiveState,
    ) -> List[str]:
        """Generate recommendations based on cognitive state"""
        recommendations = []

        if cognitive_state == CognitiveState.AUTOMATICITY:
            recommendations.append("Consider increasing difficulty")
            recommendations.append("Move to more advanced content")
        elif cognitive_state == CognitiveState.STRUGGLING:
            recommendations.append("Provide scaffolding or hints")
            recommendations.append("Review prerequisite concepts")
        elif cognitive_state == CognitiveState.GUESSING:
            recommendations.append("Check for misconceptions")
            recommendations.append("Encourage more careful consideration")
        elif cognitive_state == CognitiveState.CONFUSED:
            recommendations.append("Simplify content presentation")
            recommendations.append("Break into smaller steps")

        return recommendations
