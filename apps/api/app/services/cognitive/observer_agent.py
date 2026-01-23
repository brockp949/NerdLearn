
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.adaptive.cognitive.behavior_detectors import detect_gaming, detect_wheel_spinning, InteractionEvent
from app.schemas.cognitive import BehaviorPattern, ObservationRequest, ObservationResponse, BehaviorType
from app.adaptive.cognitive import get_intervention_engine, InterventionEngine
from app.adaptive.cognitive import InterventionPriority

logger = logging.getLogger(__name__)

class MetacognitiveObserver:
    """
    Background agent that monitors user behavior for counter-productive learning strategies.
    Uses heuristic detectors and can trigger LLM-based analysis or interventions.
    """
    
    def __init__(self, intervention_engine: Optional[InterventionEngine] = None):
        self.intervention_engine = intervention_engine or get_intervention_engine()

    async def observe(self, request: ObservationRequest) -> ObservationResponse:
        """
        Analyze recent User interactions and detect patterns.
        """
        user_id = request.user_id
        
        # Convert dicts to InteractionEvents
        events = []
        for e_dict in request.recent_events:
            try:
                # Handle timestamp string parsing if needed
                ts = e_dict.get("timestamp")
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    
                events.append(InteractionEvent(
                    timestamp=ts or datetime.utcnow(),
                    event_type=e_dict.get("event_type", "unknown"),
                    correct=e_dict.get("correct"),
                    response_time_ms=e_dict.get("response_time_ms"),
                    content_id=e_dict.get("content_id"),
                    hint_used=e_dict.get("hint_used", False),
                    attempts=e_dict.get("attempts", 1),
                    metadata=e_dict.get("metadata", {})
                ))
            except Exception as e:
                logger.warning(f"Failed to parse event for observation: {e}")
                
        patterns: List[BehaviorPattern] = []
        
        # Run Detectors
        # 1. Gaming
        gaming_pattern = detect_gaming(events)
        if gaming_pattern:
            patterns.append(gaming_pattern)
            
        # 2. Wheel Spinning
        wheel_pattern = detect_wheel_spinning(events)
        if wheel_pattern:
            patterns.append(wheel_pattern)
            
        should_intervene = False
        intervention = None
        
        # Decide on Intervention
        if patterns:
            # Simple logic for now: intervene if high confidence
            # In future, this could be more complex or delegate to InterventionEngine completely
            
            top_pattern = max(patterns, key=lambda p: p.intensity * p.confidence)
            
            if top_pattern.confidence > 0.7:
                should_intervene = True
                intervention = self._generate_intervention(top_pattern, user_id)
                
        return ObservationResponse(
            user_id=user_id,
            patterns_detected=patterns,
            should_intervene=should_intervene,
            recommended_intervention=intervention
        )

    def _generate_intervention(self, pattern: BehaviorPattern, user_id: str) -> Dict[str, Any]:
        """
        Generate a specific intervention based on the detected pattern.
        """
        import uuid
        
        if pattern.pattern_type == BehaviorType.GAMING:
             return {
                "intervention_id": str(uuid.uuid4())[:8],
                "type": "strategy_suggestion",
                "priority": "high",
                "title": "Slow Down",
                "message": "You seem to be rushing. Try reading the full question before answering.",
                "action": "I'll slow down",
                "dismissible": True
            }
        elif pattern.pattern_type == BehaviorType.WHEEL_SPINNING:
            return {
                "intervention_id": str(uuid.uuid4())[:8],
                "type": "scaffold",
                "priority": "medium",
                "title": "Let's Review",
                "message": "This concept seems tricky. Want to see a similar example?",
                "action": "Show Example",
                "dismissible": True
            }
            
        return {}

# Global instance
observer_agent = MetacognitiveObserver()

def get_observer_agent() -> MetacognitiveObserver:
    return observer_agent
