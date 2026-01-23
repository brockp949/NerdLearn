
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
from pydantic import BaseModel

from .frustration_detector import InteractionEvent
from app.schemas.cognitive import BehaviorPattern, BehaviorType

class GamingType(Enum):
    RAPID_GUESSING = "rapid_guessing"
    HINT_ABUSE = "hint_abuse"
    SYSTEM_GAMING = "system_gaming" # Generic
    NONE = "none"

class WheelSpinningType(Enum):
    REPETITIVE_ERROR = "repetitive_error"
    NO_PROGRESS = "no_progress"
    NONE = "none"

def detect_gaming(events: List[InteractionEvent], time_window_seconds: int = 60) -> Optional[BehaviorPattern]:
    """
    Detects 'Gaming the System' behaviors:
    1. Rapid Guessing: Answers submitted too quickly (< 3-5s) relative to problem complexity.
    2. Hint Abuse: Rapidly clicking through hints without reading/attempting.
    """
    if not events:
        return None
        
    # filter recent events
    recent_events = [e for e in events if (datetime.utcnow() - e.timestamp).total_seconds() < time_window_seconds]
    if not recent_events:
        return None

    # Check for Rapid Guessing
    short_attempts = [e for e in recent_events if e.event_type == 'answer' and (e.response_time_ms or 0) < 4000]
    if len(short_attempts) >= 3:
        return BehaviorPattern(
            pattern_type=BehaviorType.GAMING,
            subtype=GamingType.RAPID_GUESSING.value,
            intensity=min(1.0, len(short_attempts) * 0.2), 
            confidence=0.8,
            evidence=[f"{len(short_attempts)} attempts < 4s"],
            detected_at=datetime.utcnow()
        )

    # Check for Hint Abuse
    hint_events = [e for e in recent_events if e.event_type == 'hint']
    if len(hint_events) >= 3:
        # Check time between hints
        short_intervals = 0
        sorted_hints = sorted(hint_events, key=lambda x: x.timestamp)
        for i in range(len(sorted_hints) - 1):
            diff = (sorted_hints[i+1].timestamp - sorted_hints[i].timestamp).total_seconds()
            if diff < 2:
                short_intervals += 1
                
        if short_intervals >= 2:
             return BehaviorPattern(
                pattern_type=BehaviorType.GAMING,
                subtype=GamingType.HINT_ABUSE.value,
                intensity=0.9,
                confidence=0.9,
                evidence=[f"Rapid hint requests ({short_intervals} < 2s)"],
                detected_at=datetime.utcnow()
            )
            
    return None

def detect_wheel_spinning(events: List[InteractionEvent], max_attempts: int = 3) -> Optional[BehaviorPattern]:
    """
    Detects 'Wheel Spinning':
    - Student makes many attempts on the *same* skill/problem without success and without asking for effective help.
    """
    if not events:
        return None
        
    # Group by content_id
    # Assuming events are sorted desc or we sort them
    sorted_events = sorted(events, key=lambda x: x.timestamp)
    
    # Check last N attempts
    # We look for consecutive incorrect answers on same content or same skill (if metadata has skill_id)
    
    consecutive_failures = 0
    last_content_id = None
    
    for e in reversed(sorted_events):
        if e.event_type != 'answer':
            continue
            
        if e.correct is False:
             if last_content_id is None or e.content_id == last_content_id:
                 consecutive_failures += 1
                 last_content_id = e.content_id
             else:
                 break # Different content, reset logic for simple wheel spinning
        elif e.correct is True:
            consecutive_failures = 0
            break
            
    if consecutive_failures >= max_attempts:
         return BehaviorPattern(
            pattern_type=BehaviorType.WHEEL_SPINNING,
            subtype=WheelSpinningType.REPETITIVE_ERROR.value,
            intensity=min(1.0, consecutive_failures * 0.2),
            confidence=0.7,
            evidence=[f"{consecutive_failures} consecutive failures on content {last_content_id}"],
            detected_at=datetime.utcnow()
        )
        
    return None
