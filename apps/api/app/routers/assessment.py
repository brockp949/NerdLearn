from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
from typing import Dict
from app.core.database import get_db
from app.models.assessment import UserConceptMastery
from app.models.user import User
from app.adaptive.stealth import TelemetryCollector, TelemetryEvent, TelemetryEventType
from app.adaptive.bkt import BayesianKnowledgeTracer
import json

router = APIRouter()

# Global telemetry collector (in production, use Redis for multi-worker support)
telemetry_collector = TelemetryCollector()
bkt = BayesianKnowledgeTracer()


@router.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """
    WebSocket endpoint for stealth assessment telemetry

    Expected message format:
    {
        "event_type": "page_view" | "video_play" | "chat_query" | ...,
        "user_id": 123,
        "course_id": 456,
        "module_id": 789,
        "concept_id": 101,
        "session_id": "abc-123",
        "data": {
            "duration_seconds": 45,
            "word_count": 500,
            ...
        }
    }
    """
    await websocket.accept()

    try:
        while True:
            # Receive telemetry event
            message = await websocket.receive_json()

            # Parse event
            try:
                event = TelemetryEvent(
                    event_type=TelemetryEventType(message["event_type"]),
                    timestamp=datetime.fromisoformat(message.get("timestamp", datetime.now().isoformat())),
                    user_id=message["user_id"],
                    course_id=message["course_id"],
                    module_id=message.get("module_id"),
                    concept_id=message.get("concept_id"),
                    data=message.get("data", {}),
                    session_id=message["session_id"],
                    device_type=message.get("device_type"),
                )

                # Add to collector
                telemetry_collector.add_event(event)

                # Check if we should update mastery
                if event.concept_id:
                    # Collect evidence for this concept
                    evidence = telemetry_collector.collect_evidence(
                        event.user_id,
                        event.concept_id,
                        event.session_id
                    )

                    if evidence:
                        # Aggregate evidence into mastery score
                        evidence_score = telemetry_collector.aggregate_evidence(evidence)

                        # Send real-time feedback
                        await websocket.send_json({
                            "status": "processed",
                            "event_id": f"{event.user_id}_{event.timestamp.isoformat()}",
                            "concept_id": event.concept_id,
                            "evidence_score": evidence_score,
                            "evidence_details": evidence,
                            "message": "Stealth assessment updated"
                        })
                    else:
                        await websocket.send_json({
                            "status": "received",
                            "message": "Event logged"
                        })
                else:
                    await websocket.send_json({
                        "status": "received",
                        "message": "Event logged (no concept)"
                    })

            except Exception as e:
                await websocket.send_json({
                    "status": "error",
                    "message": str(e)
                })

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass
