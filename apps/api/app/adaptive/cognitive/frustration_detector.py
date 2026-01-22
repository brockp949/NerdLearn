"""
Frustration Detection System - Real-Time Affective State Monitoring

Research alignment:
- Affective Computing in Education: Detecting and responding to learner emotions
- Frustration Detection: Response time patterns, error sequences, behavioral signals
- Productive Struggle vs Unproductive Frustration: Optimal challenge zone

Key Indicators:
1. Response Time Patterns: Rapid guessing or extended pauses
2. Error Sequences: Consecutive incorrect answers
3. Behavioral Signals: Rapid clicking, erratic navigation
4. Engagement Patterns: Abandonment, skipping content
5. Help-Seeking: Excessive hint usage, repeated help requests

Goal: Distinguish productive struggle (leads to learning) from unproductive
frustration (leads to disengagement) and intervene appropriately.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
import math
import logging

logger = logging.getLogger(__name__)


class FrustrationLevel(str, Enum):
    """Frustration levels for intervention decisions"""
    NONE = "none"              # No frustration detected
    MILD = "mild"              # Slight struggle, may be productive
    MODERATE = "moderate"      # Clear signs, monitor closely
    HIGH = "high"              # Significant frustration, intervene
    SEVERE = "severe"          # Risk of abandonment, immediate intervention


class StruggleType(str, Enum):
    """Type of struggle detected"""
    NONE = "none"
    PRODUCTIVE = "productive"    # Desirable difficulty, leads to learning
    UNPRODUCTIVE = "unproductive"  # Frustration without learning benefit


class BehavioralSignal(str, Enum):
    """Types of behavioral signals indicating frustration"""
    RAPID_GUESSING = "rapid_guessing"
    EXTENDED_PAUSE = "extended_pause"
    CONSECUTIVE_ERRORS = "consecutive_errors"
    ERRATIC_NAVIGATION = "erratic_navigation"
    HELP_SEEKING_SPIKE = "help_seeking_spike"
    CONTENT_ABANDONMENT = "content_abandonment"
    REPEATED_ATTEMPTS = "repeated_attempts"
    REGRESSION = "regression"  # Worse performance on previously mastered content


@dataclass
class InteractionEvent:
    """Single interaction event for frustration analysis"""
    timestamp: datetime
    event_type: str  # "answer", "click", "navigation", "hint", "pause"
    correct: Optional[bool] = None
    response_time_ms: Optional[int] = None
    content_id: Optional[str] = None
    hint_used: bool = False
    attempts: int = 1
    metadata: Dict = field(default_factory=dict)


@dataclass
class FrustrationIndicators:
    """Collection of frustration indicators"""
    response_time_variance: float = 0.0
    rapid_response_ratio: float = 0.0
    extended_pause_ratio: float = 0.0
    consecutive_error_count: int = 0
    max_consecutive_errors: int = 0
    hint_usage_rate: float = 0.0
    navigation_entropy: float = 0.0
    abandonment_signals: int = 0
    regression_count: int = 0


@dataclass
class FrustrationEstimate:
    """Estimated frustration state"""
    level: FrustrationLevel
    score: float  # 0-1 scale
    struggle_type: StruggleType
    indicators: FrustrationIndicators
    active_signals: List[BehavioralSignal]
    confidence: float
    recommended_action: str
    details: Dict = field(default_factory=dict)


class FrustrationDetector:
    """
    Real-time frustration detection from behavioral signals

    Research basis:
    - Response time analysis (Baker 2007)
    - Gaming the system detection (Baker 2004)
    - Affect detection in educational games (Conati 2002)
    - Productive failure theory (Kapur 2008)

    The detector distinguishes between:
    1. Productive struggle: Challenging but learner is progressing
    2. Unproductive frustration: Spinning wheels, risk of disengagement
    """

    # Thresholds for frustration detection
    THRESHOLDS = {
        # Response time (ms)
        "rapid_response": 2000,      # < 2s = possibly guessing
        "normal_min": 3000,
        "normal_max": 30000,
        "extended_pause": 60000,     # > 60s = extended struggle

        # Error patterns
        "consecutive_errors_mild": 2,
        "consecutive_errors_moderate": 4,
        "consecutive_errors_severe": 6,

        # Hint seeking
        "hint_rate_normal": 0.2,
        "hint_rate_high": 0.5,

        # Navigation
        "navigation_entropy_high": 0.8,  # Erratic = high entropy

        # Frustration score thresholds
        "mild": 0.3,
        "moderate": 0.5,
        "high": 0.7,
        "severe": 0.85,
    }

    # Time windows for analysis
    WINDOWS = {
        "short": timedelta(minutes=5),   # Recent behavior
        "medium": timedelta(minutes=15), # Session trend
        "long": timedelta(minutes=30),   # Overall session
    }

    def __init__(
        self,
        custom_thresholds: Optional[Dict] = None,
        baseline_response_time: int = 10000,  # Baseline RT in ms
    ):
        self.thresholds = {**self.THRESHOLDS, **(custom_thresholds or {})}
        self.baseline_rt = baseline_response_time
        self.user_baselines: Dict[str, Dict] = {}  # Per-user baseline storage

    def update_user_baseline(
        self,
        user_id: str,
        events: List[InteractionEvent]
    ):
        """Update user's baseline metrics from historical events"""
        if not events:
            return

        response_times = [
            e.response_time_ms for e in events
            if e.response_time_ms and e.correct is not None
        ]

        if len(response_times) >= 10:
            self.user_baselines[user_id] = {
                "avg_rt": statistics.mean(response_times),
                "std_rt": statistics.stdev(response_times) if len(response_times) > 1 else 0,
                "accuracy": sum(1 for e in events if e.correct) / len([e for e in events if e.correct is not None]),
                "updated_at": datetime.utcnow(),
            }

    def _get_user_baseline(self, user_id: str) -> Dict:
        """Get user's baseline or default"""
        return self.user_baselines.get(user_id, {
            "avg_rt": self.baseline_rt,
            "std_rt": self.baseline_rt * 0.3,
            "accuracy": 0.7,
        })

    def _calculate_response_time_signals(
        self,
        events: List[InteractionEvent],
        user_baseline: Dict
    ) -> Tuple[float, float, float]:
        """
        Analyze response time patterns for frustration signals

        Returns:
            (variance_signal, rapid_ratio, pause_ratio)
        """
        response_times = [
            e.response_time_ms for e in events
            if e.response_time_ms is not None
        ]

        if len(response_times) < 2:
            return 0.0, 0.0, 0.0

        avg_rt = user_baseline["avg_rt"]
        std_rt = user_baseline["std_rt"]

        # Calculate variance signal (high variance = erratic behavior)
        current_std = statistics.stdev(response_times) if len(response_times) > 1 else 0
        variance_signal = min(1.0, current_std / (std_rt + 1) - 1) if std_rt > 0 else 0

        # Calculate rapid response ratio (guessing)
        rapid_count = sum(1 for rt in response_times if rt < self.thresholds["rapid_response"])
        rapid_ratio = rapid_count / len(response_times)

        # Calculate extended pause ratio (struggling)
        pause_count = sum(1 for rt in response_times if rt > self.thresholds["extended_pause"])
        pause_ratio = pause_count / len(response_times)

        return max(0, variance_signal), rapid_ratio, pause_ratio

    def _calculate_error_signals(
        self,
        events: List[InteractionEvent]
    ) -> Tuple[int, int]:
        """
        Analyze error patterns

        Returns:
            (current_consecutive_errors, max_consecutive_errors)
        """
        answer_events = [e for e in events if e.correct is not None]

        if not answer_events:
            return 0, 0

        # Count consecutive errors from most recent
        current_streak = 0
        for event in reversed(answer_events):
            if not event.correct:
                current_streak += 1
            else:
                break

        # Find max consecutive errors in window
        max_streak = 0
        current = 0
        for event in answer_events:
            if not event.correct:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0

        return current_streak, max_streak

    def _calculate_help_seeking_signal(
        self,
        events: List[InteractionEvent]
    ) -> float:
        """Calculate hint/help usage rate"""
        total_interactions = len([e for e in events if e.event_type == "answer"])
        if total_interactions == 0:
            return 0.0

        hint_count = sum(1 for e in events if e.hint_used or e.event_type == "hint")
        return hint_count / total_interactions

    def _calculate_navigation_entropy(
        self,
        events: List[InteractionEvent]
    ) -> float:
        """
        Calculate navigation entropy (erratic = high entropy)

        High entropy indicates jumping between content without completion
        """
        nav_events = [e for e in events if e.event_type == "navigation"]

        if len(nav_events) < 3:
            return 0.0

        # Count transitions between different content
        content_ids = [e.content_id for e in nav_events if e.content_id]
        if len(content_ids) < 2:
            return 0.0

        transitions = sum(
            1 for i in range(1, len(content_ids))
            if content_ids[i] != content_ids[i-1]
        )

        # Normalize by max possible transitions
        max_transitions = len(content_ids) - 1
        entropy = transitions / max_transitions if max_transitions > 0 else 0

        return entropy

    def _detect_productive_vs_unproductive(
        self,
        events: List[InteractionEvent],
        indicators: FrustrationIndicators,
        frustration_score: float
    ) -> StruggleType:
        """
        Distinguish productive struggle from unproductive frustration

        Productive struggle characteristics:
        - Making progress (some correct answers after errors)
        - Reasonable response times (thinking, not guessing)
        - Limited hint dependence
        - Persisting through difficulty

        Unproductive frustration:
        - No progress (all errors)
        - Rapid guessing or giving up
        - Excessive help-seeking
        - Signs of disengagement
        """
        if frustration_score < self.thresholds["mild"]:
            return StruggleType.NONE

        answer_events = [e for e in events if e.correct is not None]
        if not answer_events:
            return StruggleType.NONE

        # Check for progress (any correct after errors)
        recent_events = answer_events[-10:]
        has_correct = any(e.correct for e in recent_events)
        all_wrong = all(not e.correct for e in recent_events)

        # Check response time patterns
        is_guessing = indicators.rapid_response_ratio > 0.5
        is_thoughtful = indicators.rapid_response_ratio < 0.2 and indicators.extended_pause_ratio < 0.3

        # Determine struggle type
        if all_wrong and (is_guessing or indicators.consecutive_error_count >= 5):
            return StruggleType.UNPRODUCTIVE
        elif has_correct and is_thoughtful:
            return StruggleType.PRODUCTIVE
        elif indicators.hint_usage_rate > 0.5:
            return StruggleType.UNPRODUCTIVE
        elif frustration_score > self.thresholds["high"]:
            return StruggleType.UNPRODUCTIVE
        else:
            return StruggleType.PRODUCTIVE

    def _identify_active_signals(
        self,
        indicators: FrustrationIndicators
    ) -> List[BehavioralSignal]:
        """Identify which behavioral signals are currently active"""
        signals = []

        if indicators.rapid_response_ratio > 0.3:
            signals.append(BehavioralSignal.RAPID_GUESSING)

        if indicators.extended_pause_ratio > 0.2:
            signals.append(BehavioralSignal.EXTENDED_PAUSE)

        if indicators.consecutive_error_count >= self.thresholds["consecutive_errors_mild"]:
            signals.append(BehavioralSignal.CONSECUTIVE_ERRORS)

        if indicators.navigation_entropy > self.thresholds["navigation_entropy_high"]:
            signals.append(BehavioralSignal.ERRATIC_NAVIGATION)

        if indicators.hint_usage_rate > self.thresholds["hint_rate_high"]:
            signals.append(BehavioralSignal.HELP_SEEKING_SPIKE)

        if indicators.abandonment_signals > 0:
            signals.append(BehavioralSignal.CONTENT_ABANDONMENT)

        if indicators.regression_count > 0:
            signals.append(BehavioralSignal.REGRESSION)

        return signals

    def _get_recommended_action(
        self,
        level: FrustrationLevel,
        struggle_type: StruggleType,
        signals: List[BehavioralSignal]
    ) -> str:
        """Get recommended intervention action"""
        if level == FrustrationLevel.NONE:
            return "continue"

        if level == FrustrationLevel.MILD:
            if struggle_type == StruggleType.PRODUCTIVE:
                return "encourage"  # Let them work through it
            return "monitor"

        if level == FrustrationLevel.MODERATE:
            if BehavioralSignal.RAPID_GUESSING in signals:
                return "slow_down_prompt"
            if BehavioralSignal.CONSECUTIVE_ERRORS in signals:
                return "offer_hint"
            return "check_in"

        if level == FrustrationLevel.HIGH:
            if BehavioralSignal.HELP_SEEKING_SPIKE in signals:
                return "provide_scaffold"
            return "reduce_difficulty"

        # SEVERE
        return "break_suggestion"

    def detect_frustration(
        self,
        user_id: str,
        events: List[InteractionEvent],
        context: Optional[Dict] = None
    ) -> FrustrationEstimate:
        """
        Detect frustration level from recent interaction events

        Args:
            user_id: User identifier
            events: Recent interaction events
            context: Optional context (content difficulty, etc.)

        Returns:
            FrustrationEstimate with level, signals, and recommendations
        """
        context = context or {}

        if len(events) < 3:
            return FrustrationEstimate(
                level=FrustrationLevel.NONE,
                score=0.0,
                struggle_type=StruggleType.NONE,
                indicators=FrustrationIndicators(),
                active_signals=[],
                confidence=0.3,
                recommended_action="continue",
                details={"reason": "insufficient_data"}
            )

        # Get user baseline
        baseline = self._get_user_baseline(user_id)

        # Calculate indicators
        rt_variance, rapid_ratio, pause_ratio = self._calculate_response_time_signals(
            events, baseline
        )
        consecutive_errors, max_consecutive = self._calculate_error_signals(events)
        hint_rate = self._calculate_help_seeking_signal(events)
        nav_entropy = self._calculate_navigation_entropy(events)

        indicators = FrustrationIndicators(
            response_time_variance=rt_variance,
            rapid_response_ratio=rapid_ratio,
            extended_pause_ratio=pause_ratio,
            consecutive_error_count=consecutive_errors,
            max_consecutive_errors=max_consecutive,
            hint_usage_rate=hint_rate,
            navigation_entropy=nav_entropy,
        )

        # Calculate overall frustration score
        score_components = [
            rapid_ratio * 0.15,                    # Guessing weight
            pause_ratio * 0.1,                     # Extended pause weight
            min(1.0, consecutive_errors / 6) * 0.3,  # Error streak weight
            hint_rate * 0.15,                      # Help seeking weight
            nav_entropy * 0.1,                     # Navigation chaos weight
            rt_variance * 0.2,                     # RT variability weight
        ]

        frustration_score = min(1.0, sum(score_components))

        # Determine level
        if frustration_score < self.thresholds["mild"]:
            level = FrustrationLevel.NONE
        elif frustration_score < self.thresholds["moderate"]:
            level = FrustrationLevel.MILD
        elif frustration_score < self.thresholds["high"]:
            level = FrustrationLevel.MODERATE
        elif frustration_score < self.thresholds["severe"]:
            level = FrustrationLevel.HIGH
        else:
            level = FrustrationLevel.SEVERE

        # Determine struggle type
        struggle_type = self._detect_productive_vs_unproductive(
            events, indicators, frustration_score
        )

        # Identify active signals
        active_signals = self._identify_active_signals(indicators)

        # Get recommendation
        recommended_action = self._get_recommended_action(
            level, struggle_type, active_signals
        )

        # Calculate confidence based on data quality
        confidence = min(0.95, 0.5 + len(events) * 0.03)

        return FrustrationEstimate(
            level=level,
            score=round(frustration_score, 3),
            struggle_type=struggle_type,
            indicators=indicators,
            active_signals=active_signals,
            confidence=round(confidence, 2),
            recommended_action=recommended_action,
            details={
                "event_count": len(events),
                "content_difficulty": context.get("difficulty", "unknown"),
            }
        )


# Global instance
frustration_detector = FrustrationDetector()


def get_frustration_detector() -> FrustrationDetector:
    """Dependency injection"""
    return frustration_detector
