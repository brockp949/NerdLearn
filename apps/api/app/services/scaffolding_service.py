
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel

from app.adaptive.cognitive import get_intervention_engine, InterventionEngine
from app.adaptive.zpd.advanced_zpd import AdvancedZPDRegulator, LearnerProfile, DifficultyProfile
from app.adaptive.cognitive import FrustrationDetector, get_frustration_detector

logger = logging.getLogger(__name__)

class HintRequest(BaseModel):
    user_id: str
    content_id: str
    step_id: Optional[str] = None
    context: Dict[str, Any] = {}

class HintResponse(BaseModel):
    hint_text: str
    hint_level: str # "low", "medium", "high"
    remaining_hints: int

class ScaffoldingService:
    """
    Service for providing Just-in-Time scaffolding and adaptive support.
    """
    
    def __init__(
        self, 
        intervention_engine: Optional[InterventionEngine] = None,
        zpd_regulator: Optional[AdvancedZPDRegulator] = None
    ):
        self.intervention_engine = intervention_engine or get_intervention_engine()
        self.zpd_regulator = zpd_regulator or AdvancedZPDRegulator(frustration_detector=get_frustration_detector())
        self._hint_history: Dict[str, List[datetime]] = {} # user_id:content_id -> timestamps

    async def get_adaptive_hint(self, request: HintRequest) -> HintResponse:
        """
        Get an adaptive hint based on user's current context and history.
        """
        key = f"{request.user_id}:{request.content_id}"
        if key not in self._hint_history:
            self._hint_history[key] = []
            
        history = self._hint_history[key]
        history.append(datetime.utcnow())
        
        # Determine hint level based on count
        count = len(history)
        
        if count == 1:
            level = "low"
            text = "Try identifying the key variables first." # Placeholder logic
        elif count == 2:
            level = "medium"
            text = "Consider the relationship between X and Y."
        else:
            level = "high"
            text = "Here is a similar example: ..."
            
        return HintResponse(
            hint_level=level,
            hint_text=text,
            remaining_hints=max(0, 3 - count)
        )

    async def analyze_zpd_fit(self, user_id: str, content_difficulty: float, user_mastery: float) -> Dict[str, Any]:
        """
        Analyze whether content fits the user's ZPD.
        """
        # Create dummy profiles for now - in real implementation would load from DB
        learner = LearnerProfile(
            cognitive_capacity=0.7, 
            prior_knowledge=user_mastery
        )
        content = DifficultyProfile(
            cognitive=content_difficulty,
            prior_knowledge=content_difficulty # Improving assumption
        )
        
        analysis = self.zpd_regulator.calculate_multidimensional_zpd(
            learner, content, user_mastery
        )
        
        return analysis

# Global instance
scaffolding_service = ScaffoldingService()

def get_scaffolding_service() -> ScaffoldingService:
    return scaffolding_service
