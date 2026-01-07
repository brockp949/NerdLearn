"""
Inference Engine Microservice
Provides Knowledge Tracing (DKT/SAINT) and ZPD Regulation

This service acts as the "cognitive core" - it:
1. Tracks learner knowledge state over time
2. Predicts performance on new concepts
3. Regulates difficulty to maintain optimal challenge (ZPD)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np
import asyncio

from dkt_model import KnowledgeStateTracker
from zpd_regulator import ZPDRegulator, ZPDState, ScaffoldingType, AdaptiveAction, AdaptiveEngine


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class InteractionRecord(BaseModel):
    concept_id: int
    is_correct: bool
    timestamp: Optional[datetime] = None


class PredictionRequest(BaseModel):
    learner_id: str
    interaction_history: List[InteractionRecord]
    target_concept_id: int


class PredictionResponse(BaseModel):
    learner_id: str
    target_concept_id: int
    predicted_probability: float
    confidence: float
    timestamp: datetime


class KnowledgeStateRequest(BaseModel):
    learner_id: str
    interaction_history: List[InteractionRecord]


class KnowledgeStateResponse(BaseModel):
    learner_id: str
    knowledge_vector: List[float]
    mastered_concepts: List[int]
    struggling_concepts: List[int]
    timestamp: datetime


class ZPDAssessmentRequest(BaseModel):
    learner_id: str
    concept_id: str
    success: bool


class ZPDAssessmentResponse(BaseModel):
    learner_id: str
    concept_id: str
    zone: str
    success_rate: float
    trend: str
    recommended_actions: List[str]
    active_scaffolding: List[str]
    confidence: float


class AdaptiveRecommendationRequest(BaseModel):
    learner_id: str
    interaction_history: List[InteractionRecord]
    available_concepts: List[str]
    concept_difficulties: Dict[str, float]


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="NerdLearn Inference Engine",
    description="Knowledge Tracing and ZPD Regulation Service",
    version="0.1.0"
)

# Global instances
NUM_CONCEPTS = 1000  # Adjust based on curriculum
knowledge_tracker = KnowledgeStateTracker(
    num_concepts=NUM_CONCEPTS,
    model_type="saint",  # Use SAINT+ for better performance
    device="cpu"
)
zpd_regulator = ZPDRegulator()
adaptive_engine = AdaptiveEngine(zpd_regulator)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "service": "NerdLearn Inference Engine",
        "status": "operational",
        "version": "0.1.0",
        "model": "SAINT+ (Transformer-based Knowledge Tracing)",
        "num_concepts": NUM_CONCEPTS
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_performance(request: PredictionRequest):
    """
    Predict probability of correct response on target concept

    Uses Deep Knowledge Tracing (DKT/SAINT) to model knowledge evolution
    """
    try:
        # Extract exercise and response sequences
        exercises = [int(record.concept_id) for record in request.interaction_history]
        responses = [record.is_correct for record in request.interaction_history]

        # Validate concept IDs
        if not all(0 <= e < NUM_CONCEPTS for e in exercises + [request.target_concept_id]):
            raise HTTPException(status_code=400, detail="Invalid concept ID")

        # Predict
        probability = knowledge_tracker.predict_performance(
            exercises,
            responses,
            request.target_concept_id
        )

        # Confidence based on amount of data
        confidence = min(1.0, len(exercises) / 20)

        return PredictionResponse(
            learner_id=request.learner_id,
            target_concept_id=request.target_concept_id,
            predicted_probability=probability,
            confidence=confidence,
            timestamp=datetime.now()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/knowledge-state", response_model=KnowledgeStateResponse)
async def get_knowledge_state(request: KnowledgeStateRequest):
    """
    Get complete knowledge state vector across all concepts

    Returns predicted mastery level for every concept in curriculum
    """
    try:
        # Extract sequences
        exercises = [int(record.concept_id) for record in request.interaction_history]
        responses = [record.is_correct for record in request.interaction_history]

        # Get knowledge state
        state_vector = knowledge_tracker.get_knowledge_state(exercises, responses)

        # Identify mastered and struggling concepts
        mastered = [i for i, p in enumerate(state_vector) if p > 0.8]
        struggling = [i for i, p in enumerate(state_vector) if p < 0.3]

        return KnowledgeStateResponse(
            learner_id=request.learner_id,
            knowledge_vector=state_vector.tolist(),
            mastered_concepts=mastered,
            struggling_concepts=struggling,
            timestamp=datetime.now()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State error: {str(e)}")


@app.post("/zpd/assess", response_model=ZPDAssessmentResponse)
async def assess_zpd(request: ZPDAssessmentRequest):
    """
    Assess ZPD state and get adaptive recommendations

    Records attempt and returns current zone + recommended actions
    """
    try:
        # Record attempt
        zpd_regulator.record_attempt(
            request.learner_id,
            request.concept_id,
            request.success
        )

        # Assess state
        state = zpd_regulator.assess_zpd_state(request.learner_id, request.concept_id)

        # Get active scaffolding
        scaffolds = zpd_regulator.get_active_scaffolding(
            request.learner_id,
            request.concept_id
        )

        return ZPDAssessmentResponse(
            learner_id=request.learner_id,
            concept_id=request.concept_id,
            zone=state.current_zone.value,
            success_rate=state.success_rate,
            trend=state.trend,
            recommended_actions=[a.value for a in state.recommended_actions],
            active_scaffolding=[s.value for s in scaffolds],
            confidence=state.confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ZPD assessment error: {str(e)}")


@app.post("/zpd/scaffold")
async def apply_scaffolding(
    learner_id: str,
    concept_id: str,
    scaffold_type: str
):
    """Manually apply scaffolding"""
    try:
        scaffold = ScaffoldingType(scaffold_type)
        zpd_regulator.apply_scaffolding(learner_id, concept_id, scaffold)

        return {
            "status": "applied",
            "learner_id": learner_id,
            "concept_id": concept_id,
            "scaffold_type": scaffold_type
        }

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid scaffold type")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scaffold error: {str(e)}")


@app.delete("/zpd/scaffold")
async def remove_scaffold(
    learner_id: str,
    concept_id: str,
    scaffold_type: Optional[str] = None
):
    """Remove scaffolding (all or specific type)"""
    try:
        if scaffold_type:
            scaffold = ScaffoldingType(scaffold_type)
            zpd_regulator.remove_scaffolding(learner_id, concept_id, scaffold)
        else:
            zpd_regulator.remove_scaffolding(learner_id, concept_id)

        return {
            "status": "removed",
            "learner_id": learner_id,
            "concept_id": concept_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Remove error: {str(e)}")


@app.post("/recommend")
async def get_adaptive_recommendations(request: AdaptiveRecommendationRequest):
    """
    Get adaptive recommendations combining DKT + ZPD

    This is the main "brain" endpoint that integrates:
    - Knowledge state (what learner knows)
    - ZPD assessment (optimal difficulty)
    - Available content

    Returns prioritized learning recommendations
    """
    try:
        # Get knowledge state
        exercises = [int(record.concept_id) for record in request.interaction_history]
        responses = [record.is_correct for record in request.interaction_history]
        knowledge_state = knowledge_tracker.get_knowledge_state(exercises, responses)

        # Get recommendations
        recommendations = adaptive_engine.recommend_next_activity(
            request.learner_id,
            knowledge_state,
            request.available_concepts,
            request.concept_difficulties
        )

        return recommendations

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "num_concepts": NUM_CONCEPTS
    }


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    print("✅ Inference Engine started")
    print(f"📚 Knowledge Tracing: SAINT+ with {NUM_CONCEPTS} concepts")
    print("🎯 ZPD Regulation: Active")


@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 Inference Engine stopped")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
