
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

class BehaviorType(str, Enum):
    GAMING = "gaming"
    WHEEL_SPINNING = "wheel_spinning"
    PRODUCTIVE = "productive"
    ACTIVE_LEARNING = "active_learning"
    AVOIDANCE = "avoidance"

class BehaviorPattern(BaseModel):
    pattern_type: BehaviorType
    subtype: str
    intensity: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    evidence: List[str]
    detected_at: datetime = Field(default_factory=datetime.utcnow)

class ObservationRequest(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    recent_events: List[Dict[str, Any]] # Raw events or InteractionEvent dicts
    context: Optional[Dict[str, Any]] = None

class ObservationResponse(BaseModel):
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    patterns_detected: List[BehaviorPattern]
    recommended_intervention: Optional[Dict[str, Any]] = None # Intervention object
    should_intervene: bool = False
