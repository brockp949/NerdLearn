"""
Stealth Assessment Telemetry Collector
Tracks user behavior without explicit testing to infer mastery

Evidence types:
- Dwell time on concepts
- Click patterns and navigation
- Content engagement
- Video seeking behavior
- Search queries
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel
from enum import Enum


class TelemetryEventType(str, Enum):
    """Types of stealth assessment events"""
    PAGE_VIEW = "page_view"
    CONTENT_DWELL = "content_dwell"
    VIDEO_PLAY = "video_play"
    VIDEO_PAUSE = "video_pause"
    VIDEO_SEEK = "video_seek"
    CHAT_QUERY = "chat_query"
    CONCEPT_CLICK = "concept_click"
    MODULE_COMPLETE = "module_complete"
    QUIZ_ATTEMPT = "quiz_attempt"
    SEARCH = "search"


class TelemetryEvent(BaseModel):
    """Individual telemetry event"""
    event_type: TelemetryEventType
    timestamp: datetime
    user_id: int
    course_id: int
    module_id: Optional[int] = None
    concept_id: Optional[int] = None

    # Event-specific data
    data: Dict[str, Any] = {}

    # Metadata
    session_id: str
    device_type: Optional[str] = None


class EvidenceRule:
    """
    Base class for evidence interpretation rules
    Converts raw telemetry into mastery evidence
    """

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    def evaluate(self, events: List[TelemetryEvent]) -> Optional[float]:
        """
        Evaluate telemetry events and return evidence score

        Args:
            events: List of related telemetry events

        Returns:
            Evidence score (0-1) or None if not applicable
        """
        raise NotImplementedError


class DwellTimeRule(EvidenceRule):
    """
    Evidence from time spent on content

    Assumes:
    - Longer dwell time = better engagement (up to a point)
    - Normalized by content difficulty and reading speed
    """

    def __init__(self):
        super().__init__("dwell_time", weight=0.8)
        self.optimal_wpm = 250  # Average reading speed

    def evaluate(self, events: List[TelemetryEvent]) -> Optional[float]:
        """
        Calculate dwell time evidence

        Expected events: PAGE_VIEW or CONTENT_DWELL
        """
        if not events:
            return None

        dwell_events = [
            e for e in events
            if e.event_type in [TelemetryEventType.PAGE_VIEW, TelemetryEventType.CONTENT_DWELL]
        ]

        if not dwell_events:
            return None

        # Calculate total dwell time
        total_dwell_seconds = sum(
            e.data.get("duration_seconds", 0) for e in dwell_events
        )

        # Get expected reading time based on content length
        word_count = dwell_events[0].data.get("word_count", 0)
        if word_count == 0:
            return None

        expected_time = (word_count / self.optimal_wpm) * 60  # seconds

        # Calculate ratio
        time_ratio = total_dwell_seconds / expected_time if expected_time > 0 else 0

        # Score based on ratio
        # Too fast = skimming (low score)
        # Optimal = 0.8-1.5x expected time (high score)
        # Too slow = struggling (medium score)
        if time_ratio < 0.5:
            score = 0.3  # Skimmed
        elif 0.5 <= time_ratio < 0.8:
            score = 0.6  # Fast but engaged
        elif 0.8 <= time_ratio <= 1.5:
            score = 0.9  # Optimal engagement
        elif 1.5 < time_ratio <= 3.0:
            score = 0.7  # Took time to understand
        else:
            score = 0.5  # Struggling or distracted

        return score


class VideoEngagementRule(EvidenceRule):
    """
    Evidence from video watching behavior

    Indicators:
    - Completion rate
    - Replay frequency (seeking back)
    - Pause frequency (thinking)
    """

    def __init__(self):
        super().__init__("video_engagement", weight=0.9)

    def evaluate(self, events: List[TelemetryEvent]) -> Optional[float]:
        """Calculate video engagement score"""
        video_events = [
            e for e in events
            if e.event_type in [
                TelemetryEventType.VIDEO_PLAY,
                TelemetryEventType.VIDEO_PAUSE,
                TelemetryEventType.VIDEO_SEEK,
            ]
        ]

        if not video_events:
            return None

        # Get video duration
        video_duration = video_events[0].data.get("video_duration", 0)
        if video_duration == 0:
            return None

        # Calculate watch time
        play_events = [e for e in video_events if e.event_type == TelemetryEventType.VIDEO_PLAY]
        pause_events = [e for e in video_events if e.event_type == TelemetryEventType.VIDEO_PAUSE]

        total_watch_time = 0
        for play_event in play_events:
            # Find corresponding pause
            play_time = play_event.timestamp
            matching_pauses = [
                p for p in pause_events
                if p.timestamp > play_time
            ]
            if matching_pauses:
                pause_time = matching_pauses[0].timestamp
                total_watch_time += (pause_time - play_time).total_seconds()

        # Calculate completion rate
        completion_rate = min(1.0, total_watch_time / video_duration)

        # Count replays (backwards seeks)
        seek_events = [e for e in video_events if e.event_type == TelemetryEventType.VIDEO_SEEK]
        backward_seeks = sum(
            1 for e in seek_events
            if e.data.get("direction") == "backward"
        )

        # Calculate score
        # High completion + few replays = good understanding
        # High completion + many replays = engaged but needs review
        # Low completion = low engagement or mastery
        base_score = completion_rate * 0.7

        if backward_seeks > 0:
            # Some replays indicate careful learning
            replay_bonus = min(0.2, backward_seeks * 0.05)
            score = base_score + replay_bonus
        else:
            # No replays might mean easy content or skimming
            if completion_rate > 0.9:
                score = base_score + 0.3  # Mastered
            else:
                score = base_score  # Skipped

        return min(1.0, score)


class ChatQueryRule(EvidenceRule):
    """
    Evidence from chat interactions

    Indicators:
    - Query complexity
    - Follow-up questions
    - Concept references
    """

    def __init__(self):
        super().__init__("chat_query", weight=0.7)

    def evaluate(self, events: List[TelemetryEvent]) -> Optional[float]:
        """Evaluate chat query evidence"""
        chat_events = [
            e for e in events
            if e.event_type == TelemetryEventType.CHAT_QUERY
        ]

        if not chat_events:
            return None

        # Analyze queries
        total_queries = len(chat_events)

        # Check for concept references
        concept_queries = sum(
            1 for e in chat_events
            if e.concept_id is not None
        )

        # Check query quality (length as proxy)
        avg_query_length = sum(
            len(e.data.get("query", ""))
            for e in chat_events
        ) / total_queries

        # Score based on engagement
        # More questions = deeper engagement
        # Concept-specific questions = targeted learning
        engagement_score = min(1.0, total_queries / 5) * 0.4
        concept_score = (concept_queries / total_queries) * 0.3 if total_queries > 0 else 0
        quality_score = min(0.3, avg_query_length / 100)  # Longer = more thoughtful

        score = engagement_score + concept_score + quality_score

        return min(1.0, score)


class TelemetryCollector:
    """
    Collects and processes telemetry events for stealth assessment
    """

    def __init__(self):
        self.rules: List[EvidenceRule] = [
            DwellTimeRule(),
            VideoEngagementRule(),
            ChatQueryRule(),
        ]
        self.event_buffer: Dict[str, List[TelemetryEvent]] = {}

    def add_event(self, event: TelemetryEvent):
        """Add an event to the buffer"""
        session_key = f"{event.user_id}_{event.session_id}"

        if session_key not in self.event_buffer:
            self.event_buffer[session_key] = []

        self.event_buffer[session_key].append(event)

    def get_session_events(
        self, user_id: int, session_id: str
    ) -> List[TelemetryEvent]:
        """Get events for a session"""
        session_key = f"{user_id}_{session_id}"
        return self.event_buffer.get(session_key, [])

    def collect_evidence(
        self, user_id: int, concept_id: int, session_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Collect all evidence for a user-concept pair

        Args:
            user_id: User ID
            concept_id: Concept ID
            session_id: Optional session filter

        Returns:
            Dictionary of evidence scores by rule name
        """
        # Get relevant events
        if session_id:
            events = self.get_session_events(user_id, session_id)
        else:
            # Get all events for user
            events = [
                e for events in self.event_buffer.values()
                for e in events
                if e.user_id == user_id
            ]

        # Filter by concept if specified
        if concept_id:
            events = [e for e in events if e.concept_id == concept_id]

        # Apply evidence rules
        evidence = {}
        for rule in self.rules:
            score = rule.evaluate(events)
            if score is not None:
                evidence[rule.name] = {
                    "score": score,
                    "weight": rule.weight,
                    "weighted_score": score * rule.weight,
                }

        return evidence

    def aggregate_evidence(self, evidence: Dict[str, Dict]) -> float:
        """
        Aggregate weighted evidence into overall mastery probability

        Args:
            evidence: Evidence dictionary from collect_evidence()

        Returns:
            Overall mastery probability (0-1)
        """
        if not evidence:
            return 0.0

        total_weight = sum(e["weight"] for e in evidence.values())
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(e["weighted_score"] for e in evidence.values())

        return weighted_sum / total_weight
