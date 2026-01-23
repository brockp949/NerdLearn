from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from app.models.assessment import UserConceptMastery

class RetentionService:
    """
    Analyzes learning retention following teaching sessions.
    """

    async def calculate_retention_gain(
        self, 
        user_id: int, 
        concept_id: str, 
        db: AsyncSession
    ) -> Dict[str, float]:
        """
        Calculate the 'teaching effect': improvement in mastery after teaching.
        """
        # Get mastery record
        result = await db.execute(
            select(UserConceptMastery).where(
                and_(
                    UserConceptMastery.user_id == user_id,
                    UserConceptMastery.concept_id == concept_id
                )
            )
        )
        mastery = result.scalar_one_or_none()
        
        if not mastery:
            return {"gain": 0.0, "current_mastery": 0.0}
            
        # We assume baseline was 0.5 or tracked separately. 
        # For MVP, we treat current probability > 0.6 as positive retention.
        
        baseline = 0.5 # Default prior
        gain = mastery.mastery_probability - baseline
        
        return {
            "gain": max(0.0, gain),
            "current_mastery": mastery.mastery_probability,
            "retention_score": mastery.mastery_probability * 100
        }

    async def schedule_followup_quiz(self, user_id: int, concept_id: str) -> datetime:
        """
        Schedule a spaced repetition check-in.
        Returns the scheduled time (mocked for now).
        """
        # In a real system, this would insert into a scheduler queue
        scheduled_time = datetime.utcnow() + timedelta(days=1)
        return scheduled_time

    async def analyze_decay(
        self,
        user_id: int,
        concept_id: str,
        db: AsyncSession
    ) -> float:
        """
        Estimate memory decay since last interaction.
        """
        result = await db.execute(
            select(UserConceptMastery).where(
                and_(
                    UserConceptMastery.user_id == user_id,
                    UserConceptMastery.concept_id == concept_id
                )
            )
        )
        mastery = result.scalar_one_or_none()
        
        if not mastery or not mastery.last_updated_at:
            return 0.0
            
        # Simple Ebbinghaus decay model approximation
        days_elapsed = (datetime.utcnow() - mastery.last_updated_at).total_seconds() / 86400
        retention = mastery.mastery_probability * (0.9 ** days_elapsed)
        
        return retention
