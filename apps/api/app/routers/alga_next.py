"""
ALGA-Next API Router

REST and WebSocket endpoints for the Adaptive Learning via Generative Allocation system.

Endpoints:
- POST /select-modality: Get optimal modality for a concept
- POST /record-outcome: Record learning outcome
- POST /telemetry: Process telemetry batch
- GET /user-state/{user_id}: Get current user state
- GET /transfer-matrix: Get cross-modality transfer matrix
- WS /ws/{user_id}: Real-time telemetry and adaptation
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging

from app.adaptive.alga_next import (
    HybridLinUCB,
    ContextVector,
    MMSAFNet,
    UserStateVector,
    BehavioralFeatures,
    ContextualFeatures,
    ContentFeatures,
    AttentionTransferNetwork,
    CompositeRewardFunction,
    RewardComponents,
    RewardObjective,
    GenerativeUIRegistry,
    MouStressAnalyzer,
    LearnerState,
)
from app.adaptive.alga_next.orchestrator import (
    ALGANextOrchestrator,
    TelemetrySnapshot,
    MouseEvent,
    get_orchestrator,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alga-next", tags=["ALGA-Next Adaptive Learning"])


# ============================================================================
# Pydantic Models
# ============================================================================

class MouseEventRequest(BaseModel):
    x: int
    y: int
    timestamp_ms: float
    event_type: str = "move"


class TelemetryRequest(BaseModel):
    session_id: str
    user_id: str
    mouse_events: List[MouseEventRequest] = Field(default_factory=list)
    dwell_time_ms: float = 0.0
    scroll_depth: float = 0.0
    click_count: int = 0
    interaction_count: int = 0
    session_duration_minutes: float = 0.0
    cards_completed: int = 0
    recent_success_rate: float = 0.5
    current_concept_id: str = ""
    current_content_id: str = ""
    current_modality: str = "text"


class ContentOption(BaseModel):
    id: str
    modality: str
    duration_minutes: float = 5.0
    reading_level: float = 0.5
    complexity_score: float = 0.5
    interaction_level: float = 0.0
    cognitive_load_estimate: float = 0.5


class ModalitySelectionRequest(BaseModel):
    user_id: str
    concept_id: str
    available_content: List[ContentOption]
    device: str = "desktop"
    telemetry: Optional[TelemetryRequest] = None


class OutcomeRequest(BaseModel):
    user_id: str
    session_id: str
    content_id: str
    modality: str
    engagement_score: float = 0.5
    dwell_time_ms: float = 0.0
    expected_dwell_ms: float = 60000.0
    scroll_completion: float = 0.5
    interaction_rate: float = 0.0
    assessment_score: Optional[float] = None
    completion_rate: float = 0.5
    success: float = 0.5


class UserStateResponse(BaseModel):
    user_id: str
    cognitive_capacity: float
    fatigue_level: float
    focus_level: float
    engagement: float
    frustration: float
    confidence: float
    flow_state: float
    confusion_indicator: float
    dominant_state: str
    summary: str


class ModalitySelectionResponse(BaseModel):
    selected_modality: str
    selected_content_id: str
    confidence: float
    user_state: UserStateResponse
    learner_state: str
    scaffolding_level: str
    explanation: str
    alternatives: List[Dict[str, Any]]
    exploration_bonus: float
    intervention: Optional[Dict[str, Any]]
    ui_schema: Dict[str, Any]


class OutcomeResponse(BaseModel):
    reward: float
    breakdown: Dict[str, float]


# ============================================================================
# REST Endpoints
# ============================================================================

@router.post("/select-modality", response_model=ModalitySelectionResponse)
async def select_modality(
    request: ModalitySelectionRequest,
    orchestrator: ALGANextOrchestrator = Depends(get_orchestrator),
):
    """
    Select optimal content modality for a user and concept

    Uses:
    - Hybrid LinUCB contextual bandit
    - MMSAF-Net for state inference
    - Attention Transfer for cold-start handling
    """
    try:
        # Convert telemetry if provided
        telemetry = None
        if request.telemetry:
            telemetry = TelemetrySnapshot(
                session_id=request.telemetry.session_id,
                user_id=request.telemetry.user_id,
                timestamp=datetime.now(),
                mouse_events=[
                    MouseEvent(
                        x=e.x,
                        y=e.y,
                        timestamp_ms=e.timestamp_ms,
                        event_type=e.event_type,
                    )
                    for e in request.telemetry.mouse_events
                ],
                dwell_time_ms=request.telemetry.dwell_time_ms,
                scroll_depth=request.telemetry.scroll_depth,
                click_count=request.telemetry.click_count,
                interaction_count=request.telemetry.interaction_count,
                session_duration_minutes=request.telemetry.session_duration_minutes,
                cards_completed=request.telemetry.cards_completed,
                recent_success_rate=request.telemetry.recent_success_rate,
                current_concept_id=request.telemetry.current_concept_id,
                current_content_id=request.telemetry.current_content_id,
                current_modality=request.telemetry.current_modality,
            )

        # Convert content options
        available_content = [c.dict() for c in request.available_content]

        # Run selection
        result = await orchestrator.select_modality(
            user_id=request.user_id,
            concept_id=request.concept_id,
            available_content=available_content,
            telemetry=telemetry,
            device=request.device,
        )

        return ModalitySelectionResponse(
            selected_modality=result.selected_modality.value,
            selected_content_id=result.selected_content_id,
            confidence=result.confidence,
            user_state=UserStateResponse(
                user_id=request.user_id,
                cognitive_capacity=result.user_state.cognitive_capacity,
                fatigue_level=result.user_state.fatigue_level,
                focus_level=result.user_state.focus_level,
                engagement=result.user_state.engagement,
                frustration=result.user_state.frustration,
                confidence=result.user_state.confidence,
                flow_state=result.user_state.flow_state,
                confusion_indicator=result.user_state.confusion_indicator,
                dominant_state=result.user_state.dominant_state,
                summary=result.user_state.get_summary(),
            ),
            learner_state=result.learner_state.value,
            scaffolding_level=result.ui_schema.scaffolding_level.value,
            explanation=result.explanation,
            alternatives=result.alternatives,
            exploration_bonus=result.exploration_bonus,
            intervention=result.intervention,
            ui_schema=result.ui_schema.to_dict(),
        )

    except Exception as e:
        logger.error(f"Error selecting modality: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/record-outcome", response_model=OutcomeResponse)
async def record_outcome(
    request: OutcomeRequest,
    orchestrator: ALGANextOrchestrator = Depends(get_orchestrator),
):
    """
    Record learning outcome and update models

    Updates:
    - Hybrid LinUCB with reward
    - Attention Transfer with observation
    """
    try:
        result = await orchestrator.record_outcome(
            user_id=request.user_id,
            session_id=request.session_id,
            content_id=request.content_id,
            modality=request.modality,
            outcome={
                "engagement_score": request.engagement_score,
                "dwell_time_ms": request.dwell_time_ms,
                "expected_dwell_ms": request.expected_dwell_ms,
                "dwell_ratio": request.dwell_time_ms / max(1, request.expected_dwell_ms),
                "scroll_completion": request.scroll_completion,
                "interaction_rate": request.interaction_rate,
                "assessment_score": request.assessment_score,
                "completion_rate": request.completion_rate,
                "success": request.success,
            },
        )

        return OutcomeResponse(
            reward=result["reward"],
            breakdown=result["breakdown"],
        )

    except Exception as e:
        logger.error(f"Error recording outcome: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-telemetry")
async def process_telemetry(
    request: TelemetryRequest,
    orchestrator: ALGANextOrchestrator = Depends(get_orchestrator),
):
    """
    Process telemetry batch and return user state

    Useful for periodic state updates without modality selection.
    """
    try:
        telemetry = TelemetrySnapshot(
            session_id=request.session_id,
            user_id=request.user_id,
            timestamp=datetime.now(),
            mouse_events=[
                MouseEvent(
                    x=e.x,
                    y=e.y,
                    timestamp_ms=e.timestamp_ms,
                    event_type=e.event_type,
                )
                for e in request.mouse_events
            ],
            dwell_time_ms=request.dwell_time_ms,
            scroll_depth=request.scroll_depth,
            click_count=request.click_count,
            interaction_count=request.interaction_count,
            session_duration_minutes=request.session_duration_minutes,
            cards_completed=request.cards_completed,
            recent_success_rate=request.recent_success_rate,
            current_concept_id=request.current_concept_id,
            current_content_id=request.current_content_id,
            current_modality=request.current_modality,
        )

        user_state = await orchestrator.process_telemetry(telemetry)

        # Get mouse analyzer result
        analyzer = orchestrator.get_mouse_analyzer(request.user_id)
        mouse_result = analyzer.analyze()

        return {
            "user_id": request.user_id,
            "user_state": {
                "cognitive_capacity": user_state.cognitive_capacity,
                "fatigue_level": user_state.fatigue_level,
                "focus_level": user_state.focus_level,
                "engagement": user_state.engagement,
                "dominant_state": user_state.dominant_state,
                "summary": user_state.get_summary(),
            },
            "mouse_analysis": {
                "learner_state": mouse_result.learner_state.value,
                "cognitive_load": mouse_result.cognitive_load,
                "attention_level": mouse_result.attention_level,
                "confidence": mouse_result.confidence,
                "recommendations": mouse_result.recommendations,
            },
        }

    except Exception as e:
        logger.error(f"Error processing telemetry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user-state/{user_id}", response_model=UserStateResponse)
async def get_user_state(
    user_id: str,
    orchestrator: ALGANextOrchestrator = Depends(get_orchestrator),
):
    """Get current user state"""
    user_state = orchestrator.user_states.get(user_id)

    if not user_state:
        # Return default state
        user_state = UserStateVector()

    return UserStateResponse(
        user_id=user_id,
        cognitive_capacity=user_state.cognitive_capacity,
        fatigue_level=user_state.fatigue_level,
        focus_level=user_state.focus_level,
        engagement=user_state.engagement,
        frustration=user_state.frustration,
        confidence=user_state.confidence,
        flow_state=user_state.flow_state,
        confusion_indicator=user_state.confusion_indicator,
        dominant_state=user_state.dominant_state,
        summary=user_state.get_summary(),
    )


@router.get("/transfer-matrix")
async def get_transfer_matrix(
    orchestrator: ALGANextOrchestrator = Depends(get_orchestrator),
):
    """Get the cross-modality transfer matrix"""
    return orchestrator.attention_transfer.get_transfer_matrix()


@router.get("/statistics")
async def get_statistics(
    orchestrator: ALGANextOrchestrator = Depends(get_orchestrator),
):
    """Get orchestrator statistics"""
    return orchestrator.get_statistics()


@router.get("/modality-predictions/{user_id}")
async def get_modality_predictions(
    user_id: str,
    orchestrator: ALGANextOrchestrator = Depends(get_orchestrator),
):
    """
    Get predicted performance across all modalities for a user

    Uses Attention Transfer to predict even for untested modalities.
    """
    predictions = orchestrator.attention_transfer.predict(user_id)

    return {
        "user_id": user_id,
        "predictions": {
            modality.value: {
                "predicted_engagement": pred.predicted_engagement,
                "predicted_completion": pred.predicted_completion,
                "predicted_mastery": pred.predicted_mastery,
                "confidence": pred.confidence,
                "uncertainty": pred.uncertainty,
            }
            for modality, pred in predictions.predictions.items()
        },
    }


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@router.websocket("/ws/{user_id}/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str,
    session_id: str,
):
    """
    Real-time WebSocket for telemetry streaming and adaptive feedback

    Receives:
    - mouse_events: Batches of mouse events
    - interaction: User interactions
    - heartbeat: Keep-alive

    Sends:
    - state_update: User state changes
    - intervention: Recommended interventions
    - modality_suggestion: When modality switch recommended
    """
    await websocket.accept()

    orchestrator = get_orchestrator()
    analyzer = orchestrator.get_mouse_analyzer(user_id)

    logger.info(f"WebSocket connected: user={user_id}, session={session_id}")

    try:
        # Initialize session
        current_telemetry = TelemetrySnapshot(
            session_id=session_id,
            user_id=user_id,
            timestamp=datetime.now(),
        )

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            msg_type = message.get("type")

            if msg_type == "mouse_events":
                # Process mouse events
                events = message.get("events", [])
                for event in events:
                    analyzer.add_event(
                        x=event.get("x", 0),
                        y=event.get("y", 0),
                        timestamp_ms=event.get("timestamp", 0),
                        event_type=event.get("type", "move"),
                    )

                # Analyze
                result = analyzer.analyze()

                # Send state update if significant change
                if result.confidence > 0.5:
                    await websocket.send_json({
                        "type": "state_update",
                        "learner_state": result.learner_state.value,
                        "cognitive_load": result.cognitive_load,
                        "attention_level": result.attention_level,
                        "confidence": result.confidence,
                    })

                    # Check for interventions
                    if result.learner_state in [LearnerState.FRUSTRATION, LearnerState.FATIGUE]:
                        for rec in result.recommendations[:1]:
                            await websocket.send_json({
                                "type": "intervention",
                                "message": rec,
                                "priority": "high" if result.learner_state == LearnerState.FRUSTRATION else "medium",
                            })

            elif msg_type == "interaction":
                # Track interaction
                current_telemetry.interaction_count += 1

            elif msg_type == "dwell_update":
                # Update dwell time
                current_telemetry.dwell_time_ms = message.get("dwell_ms", 0)

            elif msg_type == "scroll":
                # Update scroll depth
                current_telemetry.scroll_depth = message.get("depth", 0)

            elif msg_type == "heartbeat":
                # Keep-alive
                current_telemetry.session_duration_minutes += 0.5  # 30 sec heartbeat

                # Process full telemetry periodically
                user_state = await orchestrator.process_telemetry(current_telemetry)

                await websocket.send_json({
                    "type": "heartbeat_ack",
                    "engagement": user_state.engagement,
                    "fatigue": user_state.fatigue_level,
                })

            elif msg_type == "outcome":
                # Record outcome
                await orchestrator.record_outcome(
                    user_id=user_id,
                    session_id=session_id,
                    content_id=message.get("content_id", ""),
                    modality=message.get("modality", "text"),
                    outcome=message,
                )

                await websocket.send_json({
                    "type": "outcome_ack",
                    "status": "recorded",
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: user={user_id}, session={session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))
