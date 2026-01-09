"""
Gamification Novelty Tracking System
Monitors engagement decay and refreshes gamification elements

Research basis:
- Gamification novelty effects decay after ~4 weeks (Hamari et al.)
- Variable reward schedules maintain engagement longer
- Periodic refresh of game elements prevents habituation
- Personalized novelty detection based on individual patterns

Key metrics:
- Engagement rate over time
- Feature interaction frequency
- Achievement pursuit intensity
- Login/session patterns
"""
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math
import statistics


class NoveltyState(str, Enum):
    """State of novelty for a gamification element"""
    FRESH = "fresh"           # New, high engagement
    ENGAGING = "engaging"     # Still effective
    DECLINING = "declining"   # Starting to lose effectiveness
    STALE = "stale"          # Needs refresh
    EXHAUSTED = "exhausted"  # Consider replacing


class GamificationElement(str, Enum):
    """Types of gamification elements to track"""
    XP_SYSTEM = "xp_system"
    ACHIEVEMENTS = "achievements"
    STREAKS = "streaks"
    LEADERBOARDS = "leaderboards"
    SKILL_TREE = "skill_tree"
    CHALLENGES = "challenges"
    REWARDS = "rewards"
    BADGES = "badges"


@dataclass
class EngagementDataPoint:
    """Single engagement measurement"""
    timestamp: datetime
    element: GamificationElement
    interaction_count: int
    session_duration_minutes: float
    actions_per_minute: float = 0.0
    voluntary: bool = True  # vs prompted/forced interaction


@dataclass
class NoveltyAssessment:
    """Assessment of novelty state for an element"""
    element: GamificationElement
    state: NoveltyState
    engagement_trend: float  # -1 to 1 (declining to growing)
    days_since_introduction: int
    predicted_days_until_stale: int
    refresh_recommendations: List[str]
    urgency_score: float  # 0-1


@dataclass
class UserEngagementProfile:
    """User's engagement profile across gamification elements"""
    user_id: int
    element_states: Dict[GamificationElement, NoveltyState]
    overall_engagement_score: float
    engagement_history: List[EngagementDataPoint] = field(default_factory=list)
    element_introduction_dates: Dict[GamificationElement, datetime] = field(default_factory=dict)
    last_refresh_dates: Dict[GamificationElement, datetime] = field(default_factory=dict)


class NoveltyTracker:
    """
    Tracks and manages gamification novelty

    Research-backed thresholds:
    - Fresh: 0-7 days, engagement typically high
    - Engaging: 7-21 days, sustained engagement
    - Declining: 21-35 days, noticeable drop
    - Stale: 35-56 days, significant decline
    - Exhausted: 56+ days, minimal effect
    """

    # Novelty decay thresholds (days)
    NOVELTY_THRESHOLDS = {
        NoveltyState.FRESH: 7,
        NoveltyState.ENGAGING: 21,
        NoveltyState.DECLINING: 35,
        NoveltyState.STALE: 56,
        NoveltyState.EXHAUSTED: float('inf'),
    }

    # Expected engagement decay rate per week (research: ~15-25% per week after novelty wears off)
    DECAY_RATE_PER_WEEK = 0.20

    # Minimum data points for reliable assessment
    MIN_DATA_POINTS = 5

    def __init__(
        self,
        decay_rate: float = 0.20,
        refresh_threshold: float = 0.3,  # Engagement drop threshold to trigger refresh
    ):
        self.decay_rate = decay_rate
        self.refresh_threshold = refresh_threshold
        self.user_profiles: Dict[int, UserEngagementProfile] = {}

    def record_engagement(
        self,
        user_id: int,
        element: GamificationElement,
        interaction_count: int,
        session_duration_minutes: float,
        voluntary: bool = True,
    ):
        """Record an engagement data point"""
        profile = self._get_or_create_profile(user_id)

        data_point = EngagementDataPoint(
            timestamp=datetime.now(),
            element=element,
            interaction_count=interaction_count,
            session_duration_minutes=session_duration_minutes,
            actions_per_minute=interaction_count / max(0.1, session_duration_minutes),
            voluntary=voluntary,
        )

        profile.engagement_history.append(data_point)

        # Prune old history (keep last 90 days)
        cutoff = datetime.now() - timedelta(days=90)
        profile.engagement_history = [
            dp for dp in profile.engagement_history
            if dp.timestamp > cutoff
        ]

    def introduce_element(
        self,
        user_id: int,
        element: GamificationElement,
    ):
        """Record introduction of a new gamification element"""
        profile = self._get_or_create_profile(user_id)
        profile.element_introduction_dates[element] = datetime.now()
        profile.element_states[element] = NoveltyState.FRESH

    def refresh_element(
        self,
        user_id: int,
        element: GamificationElement,
    ):
        """Record refresh of a gamification element"""
        profile = self._get_or_create_profile(user_id)
        profile.last_refresh_dates[element] = datetime.now()
        profile.element_states[element] = NoveltyState.FRESH

    def assess_novelty(
        self,
        user_id: int,
        element: GamificationElement,
    ) -> NoveltyAssessment:
        """Assess current novelty state of an element for a user"""
        profile = self._get_or_create_profile(user_id)

        # Get introduction date
        intro_date = profile.element_introduction_dates.get(element)
        last_refresh = profile.last_refresh_dates.get(element)

        # Use most recent of introduction or refresh
        effective_start = max(
            intro_date or datetime.min,
            last_refresh or datetime.min
        )

        if effective_start == datetime.min:
            effective_start = datetime.now() - timedelta(days=30)  # Default assumption

        days_since = (datetime.now() - effective_start).days

        # Get engagement data for this element
        element_data = [
            dp for dp in profile.engagement_history
            if dp.element == element
        ]

        # Calculate engagement trend
        engagement_trend = self._calculate_engagement_trend(element_data)

        # Determine novelty state
        state = self._determine_novelty_state(days_since, engagement_trend)

        # Predict days until stale
        predicted_stale = self._predict_days_until_stale(
            days_since, engagement_trend, state
        )

        # Generate refresh recommendations
        recommendations = self._generate_refresh_recommendations(
            element, state, engagement_trend
        )

        # Calculate urgency
        urgency = self._calculate_urgency(state, engagement_trend)

        # Update profile
        profile.element_states[element] = state

        return NoveltyAssessment(
            element=element,
            state=state,
            engagement_trend=round(engagement_trend, 3),
            days_since_introduction=days_since,
            predicted_days_until_stale=predicted_stale,
            refresh_recommendations=recommendations,
            urgency_score=round(urgency, 3),
        )

    def get_user_engagement_summary(
        self,
        user_id: int,
    ) -> Dict[str, Any]:
        """Get comprehensive engagement summary for user"""
        profile = self._get_or_create_profile(user_id)

        assessments = {}
        for element in GamificationElement:
            assessments[element.value] = self.assess_novelty(user_id, element)

        # Overall engagement score
        recent_data = [
            dp for dp in profile.engagement_history
            if dp.timestamp > datetime.now() - timedelta(days=7)
        ]

        if recent_data:
            avg_actions = statistics.mean(dp.actions_per_minute for dp in recent_data)
            overall_score = min(1.0, avg_actions / 5)  # Normalize to 5 actions/min as max
        else:
            overall_score = 0.0

        profile.overall_engagement_score = overall_score

        # Elements needing attention
        urgent_elements = [
            element.value for element, assessment in assessments.items()
            if assessment.urgency_score > 0.7
        ]

        return {
            "user_id": user_id,
            "overall_engagement_score": round(overall_score, 3),
            "element_assessments": {
                element: {
                    "state": assessment.state.value,
                    "trend": assessment.engagement_trend,
                    "days_active": assessment.days_since_introduction,
                    "urgency": assessment.urgency_score,
                }
                for element, assessment in assessments.items()
            },
            "elements_needing_refresh": urgent_elements,
            "recommendations": self._get_priority_recommendations(assessments),
        }

    def get_refresh_schedule(
        self,
        user_id: int,
    ) -> List[Dict[str, Any]]:
        """Get recommended refresh schedule for gamification elements"""
        profile = self._get_or_create_profile(user_id)

        schedule = []

        for element in GamificationElement:
            assessment = self.assess_novelty(user_id, element)

            if assessment.state in [NoveltyState.DECLINING, NoveltyState.STALE, NoveltyState.EXHAUSTED]:
                priority = {
                    NoveltyState.EXHAUSTED: 1,
                    NoveltyState.STALE: 2,
                    NoveltyState.DECLINING: 3,
                }[assessment.state]

                schedule.append({
                    "element": element.value,
                    "priority": priority,
                    "recommended_refresh_within_days": max(1, assessment.predicted_days_until_stale),
                    "refresh_type": self._recommend_refresh_type(element, assessment),
                    "suggestions": assessment.refresh_recommendations,
                })

        # Sort by priority
        schedule.sort(key=lambda x: x["priority"])

        return schedule

    def _get_or_create_profile(self, user_id: int) -> UserEngagementProfile:
        """Get or create user engagement profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserEngagementProfile(
                user_id=user_id,
                element_states={},
                overall_engagement_score=0.5,
            )
        return self.user_profiles[user_id]

    def _calculate_engagement_trend(
        self,
        data: List[EngagementDataPoint],
    ) -> float:
        """
        Calculate engagement trend from data points

        Returns:
            -1 to 1 (strongly declining to strongly growing)
        """
        if len(data) < self.MIN_DATA_POINTS:
            return 0.0  # Insufficient data

        # Sort by timestamp
        sorted_data = sorted(data, key=lambda d: d.timestamp)

        # Split into first and second half
        mid = len(sorted_data) // 2
        first_half = sorted_data[:mid]
        second_half = sorted_data[mid:]

        # Compare average actions per minute
        first_avg = statistics.mean(dp.actions_per_minute for dp in first_half)
        second_avg = statistics.mean(dp.actions_per_minute for dp in second_half)

        if first_avg == 0:
            return 0.0

        # Calculate relative change
        change = (second_avg - first_avg) / first_avg

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, change))

    def _determine_novelty_state(
        self,
        days_since: int,
        engagement_trend: float,
    ) -> NoveltyState:
        """Determine novelty state based on time and engagement"""

        # Time-based initial state
        if days_since <= self.NOVELTY_THRESHOLDS[NoveltyState.FRESH]:
            time_state = NoveltyState.FRESH
        elif days_since <= self.NOVELTY_THRESHOLDS[NoveltyState.ENGAGING]:
            time_state = NoveltyState.ENGAGING
        elif days_since <= self.NOVELTY_THRESHOLDS[NoveltyState.DECLINING]:
            time_state = NoveltyState.DECLINING
        elif days_since <= self.NOVELTY_THRESHOLDS[NoveltyState.STALE]:
            time_state = NoveltyState.STALE
        else:
            time_state = NoveltyState.EXHAUSTED

        # Adjust based on engagement trend
        if engagement_trend > 0.2:
            # Positive trend - upgrade state
            state_order = list(NoveltyState)
            idx = state_order.index(time_state)
            return state_order[max(0, idx - 1)]
        elif engagement_trend < -0.3:
            # Strong negative trend - downgrade state
            state_order = list(NoveltyState)
            idx = state_order.index(time_state)
            return state_order[min(len(state_order) - 1, idx + 1)]

        return time_state

    def _predict_days_until_stale(
        self,
        days_since: int,
        engagement_trend: float,
        current_state: NoveltyState,
    ) -> int:
        """Predict days until element becomes stale"""

        if current_state in [NoveltyState.STALE, NoveltyState.EXHAUSTED]:
            return 0

        # Base prediction from thresholds
        stale_threshold = self.NOVELTY_THRESHOLDS[NoveltyState.STALE]
        base_days = max(0, stale_threshold - days_since)

        # Adjust for engagement trend
        if engagement_trend < 0:
            # Declining faster
            multiplier = 1 + abs(engagement_trend)
            base_days = int(base_days / multiplier)
        elif engagement_trend > 0:
            # Declining slower
            multiplier = 1 + engagement_trend
            base_days = int(base_days * multiplier)

        return max(0, base_days)

    def _generate_refresh_recommendations(
        self,
        element: GamificationElement,
        state: NoveltyState,
        trend: float,
    ) -> List[str]:
        """Generate refresh recommendations for an element"""

        recommendations = []

        element_specific = {
            GamificationElement.XP_SYSTEM: [
                "Introduce XP multiplier events",
                "Add new XP earning activities",
                "Create limited-time XP challenges",
            ],
            GamificationElement.ACHIEVEMENTS: [
                "Add new achievement categories",
                "Introduce seasonal achievements",
                "Create progressive achievement chains",
            ],
            GamificationElement.STREAKS: [
                "Add streak freeze tokens",
                "Introduce streak milestones with rewards",
                "Create streak recovery challenges",
            ],
            GamificationElement.LEADERBOARDS: [
                "Add weekly/monthly reset leaderboards",
                "Create skill-specific leaderboards",
                "Introduce friend leaderboards",
            ],
            GamificationElement.SKILL_TREE: [
                "Unlock new skill branches",
                "Add prestige/mastery tiers",
                "Introduce skill specializations",
            ],
            GamificationElement.CHALLENGES: [
                "Create new challenge types",
                "Add collaborative challenges",
                "Introduce difficulty tiers",
            ],
            GamificationElement.REWARDS: [
                "Introduce new reward types",
                "Add mystery/random rewards",
                "Create reward customization options",
            ],
            GamificationElement.BADGES: [
                "Design new badge series",
                "Add badge levels (bronze/silver/gold)",
                "Create limited-edition badges",
            ],
        }

        if element in element_specific:
            recommendations.extend(element_specific[element][:2])

        # State-specific recommendations
        if state == NoveltyState.DECLINING:
            recommendations.append("Consider introducing a limited-time event")
        elif state == NoveltyState.STALE:
            recommendations.append("Major refresh recommended - add new mechanics")
        elif state == NoveltyState.EXHAUSTED:
            recommendations.append("Consider replacing or significantly redesigning this element")

        return recommendations

    def _calculate_urgency(
        self,
        state: NoveltyState,
        trend: float,
    ) -> float:
        """Calculate urgency score for refresh"""

        base_urgency = {
            NoveltyState.FRESH: 0.0,
            NoveltyState.ENGAGING: 0.1,
            NoveltyState.DECLINING: 0.5,
            NoveltyState.STALE: 0.8,
            NoveltyState.EXHAUSTED: 1.0,
        }

        urgency = base_urgency[state]

        # Adjust for trend
        if trend < -0.3:
            urgency = min(1.0, urgency + 0.2)
        elif trend > 0.2:
            urgency = max(0.0, urgency - 0.1)

        return urgency

    def _recommend_refresh_type(
        self,
        element: GamificationElement,
        assessment: NoveltyAssessment,
    ) -> str:
        """Recommend type of refresh based on assessment"""

        if assessment.state == NoveltyState.EXHAUSTED:
            return "major_overhaul"
        elif assessment.state == NoveltyState.STALE:
            return "significant_update"
        elif assessment.state == NoveltyState.DECLINING:
            return "minor_refresh"
        else:
            return "monitoring"

    def _get_priority_recommendations(
        self,
        assessments: Dict[GamificationElement, NoveltyAssessment],
    ) -> List[str]:
        """Get priority recommendations across all elements"""

        recommendations = []

        # Find most urgent
        urgent = [
            (elem, assess) for elem, assess in assessments.items()
            if assess.urgency_score > 0.5
        ]

        if not urgent:
            recommendations.append("All gamification elements are performing well")
            return recommendations

        # Sort by urgency
        urgent.sort(key=lambda x: x[1].urgency_score, reverse=True)

        for elem, assess in urgent[:3]:
            recommendations.append(
                f"{elem.value}: {assess.refresh_recommendations[0] if assess.refresh_recommendations else 'Consider refresh'}"
            )

        return recommendations
