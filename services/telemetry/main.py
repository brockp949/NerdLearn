"""
Telemetry Microservice - Stealth Assessment Engine
Captures and analyzes implicit behavioral signals for Evidence-Centered Design

Research alignment:
- TD-BKT (Temporal Difference Bayesian Knowledge Tracing) for real-time mastery updates
- WebSocket micro-interventions triggered by behavioral signals
- 30-second idle threshold for engagement monitoring
- Affect-aware intervention routing

Key Signals:
- Mouse dynamics (velocity, entropy, trajectory)
- Dwell time and reading patterns
- Click sequences and navigation
- Keyboard patterns
- Pause/hesitation detection
"""

from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import asyncio
import json
import numpy as np
from collections import deque
import os
import redis.asyncio as redis


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class EventType(str, Enum):
    MOUSE_MOVE = "mouse_move"
    MOUSE_CLICK = "mouse_click"
    KEY_PRESS = "key_press"
    SCROLL = "scroll"
    FOCUS = "focus"
    BLUR = "blur"
    PAGE_VIEW = "page_view"
    CONTENT_INTERACTION = "content_interaction"
    IDLE_START = "idle_start"
    IDLE_END = "idle_end"
    AFFECT_UPDATE = "affect_update"  # Client sends affect state


class AffectState(str, Enum):
    """Affect states from telemetry (matches telemetry-tracker.ts)"""
    FLOW = "flow"
    FRUSTRATED = "frustrated"
    BORED = "bored"
    CONFUSED = "confused"
    NEUTRAL = "neutral"


class InterventionType(str, Enum):
    """Types of micro-interventions"""
    PROMPT = "prompt"          # Generic check-in
    SIMPLIFY = "simplify"      # Suggest simpler content
    SCAFFOLD = "scaffold"      # Provide step-by-step help
    CHALLENGE = "challenge"    # Increase difficulty
    ENGAGE = "engage"          # Try different modality
    ENCOURAGE = "encourage"    # Positive reinforcement



class MouseEvent(BaseModel):
    x: int
    y: int
    timestamp: float
    event_type: EventType = EventType.MOUSE_MOVE


class TelemetryEvent(BaseModel):
    user_id: str
    session_id: str
    event_type: EventType
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    data: Dict[str, Any] = Field(default_factory=dict)
    resource_id: Optional[str] = None
    concept_id: Optional[str] = None


class DwellTimeAnalysis(BaseModel):
    resource_id: str
    total_dwell_ms: int
    valid_engagement: bool
    reading_rate: float  # words per minute
    expected_min_time: float
    actual_time: float


class MouseDynamicsAnalysis(BaseModel):
    avg_velocity: float
    peak_velocity: float
    trajectory_entropy: float
    saccade_count: int
    cognitive_load_indicator: str  # "low", "medium", "high"


class EngagementScore(BaseModel):
    user_id: str
    session_id: str
    overall_score: float  # 0-1
    attention_score: float
    struggle_indicator: float
    confidence_score: float
    timestamp: datetime


class MicroIntervention(BaseModel):
    """Real-time intervention sent via WebSocket"""
    type: str = "intervention"
    intervention_type: InterventionType
    urgency: str  # low, medium, high
    message: str
    action: Optional[str] = None
    concept_id: Optional[str] = None
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())


@dataclass
class SessionState:
    """Track state for TD-BKT micro-interventions"""
    user_id: str
    session_id: str
    current_concept_id: Optional[str] = None
    mastery_estimate: float = 0.5
    affect_state: AffectState = AffectState.NEUTRAL
    frustration_score: float = 0.0
    time_on_concept: float = 0.0
    scroll_depth: float = 0.0
    idle_time: float = 0.0
    last_update: float = 0.0
    interventions_sent: int = 0


# ============================================================================
# TD-BKT MICRO-INTERVENTION ENGINE
# ============================================================================

class TDBKTInterventionEngine:
    """
    Temporal Difference BKT-based intervention engine.

    Research alignment:
    - Combines behavioral signals with knowledge estimates
    - Real-time intervention triggers via WebSocket
    - Affect-aware intervention selection
    """

    # Intervention cooldown (seconds) to avoid spam
    INTERVENTION_COOLDOWN = 60

    # Idle threshold from research (30 seconds)
    IDLE_THRESHOLD = 30.0

    # Frustration threshold for immediate intervention
    FRUSTRATION_THRESHOLD = 0.7

    # TD-BKT parameters
    TD_LEARNING_RATE = 0.1
    TD_DISCOUNT = 0.9

    # Signal weights for mastery update
    SIGNAL_WEIGHTS = {
        "affect": 0.30,
        "engagement": 0.25,
        "frustration": 0.25,
        "time_quality": 0.20,
    }

    # Affect to learning signal mapping
    AFFECT_SIGNALS = {
        AffectState.FLOW: 0.85,
        AffectState.NEUTRAL: 0.50,
        AffectState.CONFUSED: 0.30,
        AffectState.BORED: 0.20,
        AffectState.FRUSTRATED: 0.10,
    }

    def __init__(self):
        self.session_states: Dict[str, SessionState] = {}
        self.last_intervention_time: Dict[str, float] = {}

    def get_or_create_session(self, user_id: str, session_id: str) -> SessionState:
        """Get or create session state"""
        key = f"{user_id}:{session_id}"
        if key not in self.session_states:
            self.session_states[key] = SessionState(
                user_id=user_id,
                session_id=session_id,
                last_update=datetime.now().timestamp()
            )
        return self.session_states[key]

    def update_from_telemetry(
        self,
        state: SessionState,
        affect_state: Optional[str] = None,
        frustration_score: Optional[float] = None,
        engagement_score: Optional[float] = None,
        time_on_task: Optional[float] = None,
        scroll_depth: Optional[float] = None,
        concept_id: Optional[str] = None
    ) -> SessionState:
        """Update session state from telemetry signals"""
        now = datetime.now().timestamp()

        if concept_id:
            if state.current_concept_id != concept_id:
                # Switched concept - reset time tracking
                state.current_concept_id = concept_id
                state.time_on_concept = 0.0
            state.current_concept_id = concept_id

        if affect_state:
            try:
                state.affect_state = AffectState(affect_state)
            except ValueError:
                state.affect_state = AffectState.NEUTRAL

        if frustration_score is not None:
            state.frustration_score = frustration_score

        if time_on_task is not None:
            state.time_on_concept = time_on_task

        if scroll_depth is not None:
            state.scroll_depth = scroll_depth

        # Update mastery using TD-BKT approach
        if affect_state or frustration_score is not None:
            state.mastery_estimate = self._td_update_mastery(
                state.mastery_estimate,
                state.affect_state,
                state.frustration_score,
                engagement_score or 0.5,
                state.time_on_concept
            )

        state.last_update = now
        return state

    def _td_update_mastery(
        self,
        current_mastery: float,
        affect: AffectState,
        frustration: float,
        engagement: float,
        time_on_task: float
    ) -> float:
        """
        TD-BKT mastery update using behavioral signals.

        Research alignment: Uses temporal difference learning to update
        knowledge estimates from behavioral telemetry.
        """
        # Compute evidence from signals
        affect_signal = self.AFFECT_SIGNALS.get(affect, 0.5)
        frustration_signal = 1.0 - frustration  # Inverse
        engagement_signal = engagement

        # Time quality signal
        if time_on_task < 10:
            time_signal = 0.2  # Too fast
        elif time_on_task < 30:
            time_signal = 0.5  # Quick
        elif time_on_task < 300:
            time_signal = 0.8  # Good
        else:
            time_signal = 0.4  # Too long

        # Weighted evidence
        evidence = (
            self.SIGNAL_WEIGHTS["affect"] * affect_signal +
            self.SIGNAL_WEIGHTS["engagement"] * engagement_signal +
            self.SIGNAL_WEIGHTS["frustration"] * frustration_signal +
            self.SIGNAL_WEIGHTS["time_quality"] * time_signal
        )

        # TD update
        td_error = evidence - current_mastery
        new_mastery = current_mastery + self.TD_LEARNING_RATE * td_error

        return max(0.0, min(1.0, new_mastery))

    def check_intervention_needed(
        self,
        state: SessionState,
        mouse_analysis: Optional[MouseDynamicsAnalysis] = None
    ) -> Optional[MicroIntervention]:
        """
        Check if intervention is needed based on state.

        Returns intervention if needed, None otherwise.
        """
        key = f"{state.user_id}:{state.session_id}"
        now = datetime.now().timestamp()

        # Check cooldown
        last_intervention = self.last_intervention_time.get(key, 0)
        if now - last_intervention < self.INTERVENTION_COOLDOWN:
            return None

        intervention = None

        # High frustration - immediate intervention
        if state.frustration_score > self.FRUSTRATION_THRESHOLD:
            intervention = MicroIntervention(
                intervention_type=InterventionType.SIMPLIFY,
                urgency="high",
                message="It seems like this might be challenging. Would you like a simpler explanation or a hint?",
                action="Get Help",
                concept_id=state.current_concept_id
            )

        # Confused state with low mastery
        elif state.affect_state == AffectState.CONFUSED and state.mastery_estimate < 0.5:
            intervention = MicroIntervention(
                intervention_type=InterventionType.SCAFFOLD,
                urgency="medium",
                message="Let me break this down step by step. Would you like guided practice?",
                action="Start Guided Practice",
                concept_id=state.current_concept_id
            )

        # Bored state - check if mastered or disengaged
        elif state.affect_state == AffectState.BORED:
            if state.mastery_estimate > 0.7:
                intervention = MicroIntervention(
                    intervention_type=InterventionType.CHALLENGE,
                    urgency="low",
                    message="You seem to have this down! Ready for a challenge?",
                    action="Take Challenge",
                    concept_id=state.current_concept_id
                )
            else:
                intervention = MicroIntervention(
                    intervention_type=InterventionType.ENGAGE,
                    urgency="medium",
                    message="Would you prefer a different way to learn this? Try an interactive exercise!",
                    action="Try Interactive",
                    concept_id=state.current_concept_id
                )

        # Stuck too long with high idle time
        elif state.time_on_concept > 300 and state.idle_time > 60:
            intervention = MicroIntervention(
                intervention_type=InterventionType.PROMPT,
                urgency="medium",
                message="Taking a moment to reflect? Let me know if you'd like to discuss this concept.",
                action="Ask Question",
                concept_id=state.current_concept_id
            )

        # Flow state - optional encouragement
        elif state.affect_state == AffectState.FLOW and state.mastery_estimate > 0.8:
            # Don't interrupt flow, but track for positive reinforcement later
            pass

        # Record intervention time
        if intervention:
            self.last_intervention_time[key] = now
            state.interventions_sent += 1

        return intervention

    def clear_session(self, user_id: str, session_id: str):
        """Clear session state"""
        key = f"{user_id}:{session_id}"
        self.session_states.pop(key, None)
        self.last_intervention_time.pop(key, None)


# ============================================================================
# EVIDENCE RULES ENGINE
# ============================================================================

class EvidenceRulesEngine:
    """
    Translates raw telemetry into Evidence for the Competency Model
    Implements Evidence-Centered Design (ECD) principles
    """

    def __init__(self):
        self.avg_reading_rate = 250  # words per minute (average adult)
        self.min_engagement_ratio = 0.6  # 60% of expected time minimum

    def analyze_dwell_time(
        self,
        dwell_ms: int,
        word_count: int,
        content_type: str = "text"
    ) -> DwellTimeAnalysis:
        """
        Analyze if dwell time indicates valid engagement

        Research shows:
        - Valid reading: 200-300 WPM for comprehension
        - Scanning: 400-700 WPM
        - Skimming: 700+ WPM
        - Too slow: <150 WPM (possible distraction)
        """
        dwell_minutes = dwell_ms / 60000
        expected_min_minutes = word_count / self.avg_reading_rate

        if dwell_minutes == 0:
            reading_rate = 0
            valid_engagement = False
        else:
            reading_rate = word_count / dwell_minutes

            # Valid engagement criteria
            valid_engagement = (
                dwell_minutes >= (expected_min_minutes * self.min_engagement_ratio) and
                150 <= reading_rate <= 700
            )

        return DwellTimeAnalysis(
            resource_id="",
            total_dwell_ms=dwell_ms,
            valid_engagement=valid_engagement,
            reading_rate=reading_rate,
            expected_min_time=expected_min_minutes * 60000,
            actual_time=dwell_ms
        )

    def analyze_mouse_dynamics(self, mouse_events: List[MouseEvent]) -> MouseDynamicsAnalysis:
        """
        Analyze mouse movement patterns to infer cognitive load

        High cognitive load signatures:
        - Slower peak velocity
        - Lower trajectory deviation
        - More saccades (rapid movements)

        Confusion signatures:
        - High entropy (chaotic movement)
        - High saccade rate
        """
        if len(mouse_events) < 2:
            return MouseDynamicsAnalysis(
                avg_velocity=0,
                peak_velocity=0,
                trajectory_entropy=0,
                saccade_count=0,
                cognitive_load_indicator="unknown"
            )

        # Calculate velocities
        velocities = []
        for i in range(1, len(mouse_events)):
            prev = mouse_events[i - 1]
            curr = mouse_events[i]

            dx = curr.x - prev.x
            dy = curr.y - prev.y
            dt = curr.timestamp - prev.timestamp

            if dt > 0:
                distance = np.sqrt(dx**2 + dy**2)
                velocity = distance / dt
                velocities.append(velocity)

        avg_velocity = np.mean(velocities) if velocities else 0
        peak_velocity = np.max(velocities) if velocities else 0

        # Calculate trajectory entropy (measure of randomness)
        # Higher entropy = more chaotic movement = possible confusion
        angles = []
        for i in range(2, len(mouse_events)):
            p1 = mouse_events[i - 2]
            p2 = mouse_events[i - 1]
            p3 = mouse_events[i]

            v1 = np.array([p2.x - p1.x, p2.y - p1.y])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y])

            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                angles.append(angle)

        trajectory_entropy = np.std(angles) if angles else 0

        # Detect saccades (rapid movements > threshold velocity)
        saccade_threshold = avg_velocity * 2 if avg_velocity > 0 else 100
        saccade_count = sum(1 for v in velocities if v > saccade_threshold)

        # Determine cognitive load
        if avg_velocity < 50 and trajectory_entropy < 0.5:
            load_indicator = "high"  # Slow, deliberate = struggling
        elif trajectory_entropy > 1.0 and saccade_count > len(velocities) * 0.3:
            load_indicator = "high"  # Chaotic = confused
        elif avg_velocity > 200:
            load_indicator = "low"  # Fast, confident
        else:
            load_indicator = "medium"

        return MouseDynamicsAnalysis(
            avg_velocity=float(avg_velocity),
            peak_velocity=float(peak_velocity),
            trajectory_entropy=float(trajectory_entropy),
            saccade_count=saccade_count,
            cognitive_load_indicator=load_indicator
        )

    def calculate_engagement_score(
        self,
        dwell_analysis: DwellTimeAnalysis,
        mouse_analysis: MouseDynamicsAnalysis,
        interaction_count: int,
        expected_interactions: int = 5
    ) -> float:
        """
        Calculate overall engagement score (0-1)

        Combines:
        - Dwell time validity (40%)
        - Mouse dynamics (30%)
        - Interaction frequency (30%)
        """
        # Dwell score
        dwell_score = 1.0 if dwell_analysis.valid_engagement else 0.3

        # Mouse dynamics score (inverse of cognitive load)
        mouse_score_map = {"low": 1.0, "medium": 0.7, "high": 0.4, "unknown": 0.5}
        mouse_score = mouse_score_map.get(mouse_analysis.cognitive_load_indicator, 0.5)

        # Interaction score
        interaction_score = min(1.0, interaction_count / expected_interactions)

        # Weighted average
        engagement = (
            dwell_score * 0.4 +
            mouse_score * 0.3 +
            interaction_score * 0.3
        )

        return engagement


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="NerdLearn Telemetry Service",
    description="Real-time behavioral signal capture and stealth assessment",
    version="0.1.0"
)

# Global state
evidence_engine = EvidenceRulesEngine()
intervention_engine = TDBKTInterventionEngine()
redis_client: Optional[redis.Redis] = None

# In-memory buffers for real-time analysis
session_mouse_events: Dict[str, deque] = {}
session_interactions: Dict[str, List[TelemetryEvent]] = {}


async def get_redis_client() -> redis.Redis:
    """Get or create Redis client"""
    global redis_client
    if redis_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/1")
        redis_client = redis.from_url(redis_url, decode_responses=True)
    return redis_client


async def publish_event(topic: str, event: dict):
    """Publish event to Redis Stream for persistence"""
    try:
        client = await get_redis_client()
        # Redis Streams XADD
        # We start the stream with 'telemetry:' prefix
        stream_name = f"telemetry:{topic}"
        await client.xadd(stream_name, {"json": json.dumps(event)})
    except Exception as e:
        print(f"Redis publish error: {e}")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "service": "NerdLearn Telemetry Service",
        "status": "operational",
        "version": "0.1.0"
    }


@app.post("/event")
async def ingest_event(event: TelemetryEvent, background_tasks: BackgroundTasks):
    """
    Ingest a single telemetry event

    This is the main ingestion endpoint for all behavioral signals
    """
    # Store in session buffer
    session_key = f"{event.user_id}:{event.session_id}"

    if event.event_type == EventType.MOUSE_MOVE:
        if session_key not in session_mouse_events:
            session_mouse_events[session_key] = deque(maxlen=1000)

        mouse_event = MouseEvent(
            x=event.data.get('x', 0),
            y=event.data.get('y', 0),
            timestamp=event.timestamp,
            event_type=EventType.MOUSE_MOVE
        )
        session_mouse_events[session_key].append(mouse_event)

    # Track all interactions
    if session_key not in session_interactions:
        session_interactions[session_key] = []
    session_interactions[session_key].append(event)

    # Publish to Kafka for persistence
    background_tasks.add_task(
        publish_event,
        "telemetry.raw",
        event.dict()
    )

    return {"status": "ingested", "event_id": f"{event.user_id}_{event.timestamp}"}


@app.post("/batch")
async def ingest_batch(events: List[TelemetryEvent], background_tasks: BackgroundTasks):
    """
    Batch ingest multiple events
    More efficient for high-frequency events like mouse movements
    """
    for event in events:
        session_key = f"{event.user_id}:{event.session_id}"

        if event.event_type == EventType.MOUSE_MOVE:
            if session_key not in session_mouse_events:
                session_mouse_events[session_key] = deque(maxlen=1000)

            mouse_event = MouseEvent(
                x=event.data.get('x', 0),
                y=event.data.get('y', 0),
                timestamp=event.timestamp
            )
            session_mouse_events[session_key].append(mouse_event)

        if session_key not in session_interactions:
            session_interactions[session_key] = []
        session_interactions[session_key].append(event)

    # Publish batch to Kafka
    background_tasks.add_task(
        publish_event,
        "telemetry.raw",
        {"events": [e.dict() for e in events], "batch_size": len(events)}
    )

    return {"status": "ingested", "count": len(events)}


@app.get("/analysis/mouse/{user_id}/{session_id}")
async def analyze_mouse_dynamics(user_id: str, session_id: str):
    """
    Analyze mouse dynamics for cognitive load inference
    """
    session_key = f"{user_id}:{session_id}"

    if session_key not in session_mouse_events:
        raise HTTPException(status_code=404, detail="No mouse data for session")

    mouse_events = list(session_mouse_events[session_key])
    analysis = evidence_engine.analyze_mouse_dynamics(mouse_events)

    return analysis


@app.get("/analysis/engagement/{user_id}/{session_id}")
async def calculate_engagement(user_id: str, session_id: str):
    """
    Calculate comprehensive engagement score for session
    """
    session_key = f"{user_id}:{session_id}"

    # Get mouse analysis
    mouse_events = list(session_mouse_events.get(session_key, []))
    mouse_analysis = evidence_engine.analyze_mouse_dynamics(mouse_events)

    # Get interaction count
    interactions = session_interactions.get(session_key, [])
    interaction_count = len([e for e in interactions if e.event_type == EventType.CONTENT_INTERACTION])

    # Calculate dwell time
    if interactions:
        timestamps = [e.timestamp for e in interactions]
        total_dwell_ms = (max(timestamps) - min(timestamps)) * 1000
        dwell_analysis = evidence_engine.analyze_dwell_time(
            int(total_dwell_ms),
            word_count=500  # Default, should be passed from content
        )
    else:
        dwell_analysis = DwellTimeAnalysis(
            resource_id="",
            total_dwell_ms=0,
            valid_engagement=False,
            reading_rate=0,
            expected_min_time=0,
            actual_time=0
        )

    # Calculate overall engagement
    engagement_score = evidence_engine.calculate_engagement_score(
        dwell_analysis,
        mouse_analysis,
        interaction_count
    )

    return EngagementScore(
        user_id=user_id,
        session_id=session_id,
        overall_score=engagement_score,
        attention_score=1.0 if dwell_analysis.valid_engagement else 0.3,
        struggle_indicator=1.0 if mouse_analysis.cognitive_load_indicator == "high" else 0.0,
        confidence_score=1.0 if mouse_analysis.cognitive_load_indicator == "low" else 0.5,
        timestamp=datetime.now()
    )


@app.websocket("/ws/{user_id}/{session_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, session_id: str):
    """
    WebSocket endpoint for real-time telemetry streaming with TD-BKT micro-interventions.

    Research alignment:
    - Sub-100ms latency for live adaptation
    - TD-BKT mastery updates from behavioral signals
    - Affect-aware micro-interventions
    - 30-second idle threshold monitoring
    """
    await websocket.accept()

    # Get or create session state for TD-BKT
    session_state = intervention_engine.get_or_create_session(user_id, session_id)
    session_key = f"{user_id}:{session_id}"

    try:
        while True:
            # Receive event
            data = await websocket.receive_text()
            event_data = json.loads(data)

            # Create event
            event_type_str = event_data.get('event_type', 'content_interaction')
            try:
                event_type = EventType(event_type_str)
            except ValueError:
                event_type = EventType.CONTENT_INTERACTION

            event = TelemetryEvent(
                user_id=user_id,
                session_id=session_id,
                event_type=event_type,
                timestamp=event_data.get('timestamp', datetime.now().timestamp()),
                data=event_data.get('data', {}),
                resource_id=event_data.get('resource_id'),
                concept_id=event_data.get('concept_id')
            )

            # Process mouse events
            if event.event_type == EventType.MOUSE_MOVE:
                if session_key not in session_mouse_events:
                    session_mouse_events[session_key] = deque(maxlen=1000)

                mouse_event = MouseEvent(
                    x=event.data.get('x', 0),
                    y=event.data.get('y', 0),
                    timestamp=event.timestamp
                )
                session_mouse_events[session_key].append(mouse_event)

            # Handle affect update from client (TD-BKT integration)
            elif event.event_type == EventType.AFFECT_UPDATE:
                # Update session state with telemetry
                session_state = intervention_engine.update_from_telemetry(
                    session_state,
                    affect_state=event.data.get('affect_state'),
                    frustration_score=event.data.get('frustration_score'),
                    engagement_score=event.data.get('engagement_score'),
                    time_on_task=event.data.get('time_on_task'),
                    scroll_depth=event.data.get('scroll_depth'),
                    concept_id=event.concept_id
                )

                # Check if intervention is needed
                mouse_events = list(session_mouse_events.get(session_key, []))
                mouse_analysis = evidence_engine.analyze_mouse_dynamics(mouse_events) if mouse_events else None

                intervention = intervention_engine.check_intervention_needed(
                    session_state,
                    mouse_analysis
                )

                if intervention:
                    # Send micro-intervention via WebSocket
                    await websocket.send_json(intervention.dict())

                # Send mastery update back to client
                await websocket.send_json({
                    "type": "mastery_update",
                    "concept_id": session_state.current_concept_id,
                    "mastery_estimate": round(session_state.mastery_estimate, 3),
                    "affect_state": session_state.affect_state.value
                })

            # Handle idle start - check for intervention
            elif event.event_type == EventType.IDLE_START:
                session_state.idle_time += 30  # Increment idle time

                # Update and check for intervention
                session_state = intervention_engine.update_from_telemetry(
                    session_state,
                    affect_state=event.data.get('affect_state', 'neutral'),
                    concept_id=event.concept_id
                )

                intervention = intervention_engine.check_intervention_needed(session_state)

                if intervention:
                    await websocket.send_json(intervention.dict())
                else:
                    # Default idle intervention if no TD-BKT intervention triggered
                    default_intervention = MicroIntervention(
                        intervention_type=InterventionType.PROMPT,
                        urgency="low",
                        message="It looks like you've been inactive. Need any help with this concept?",
                        action="Ask AI",
                        concept_id=session_state.current_concept_id
                    )
                    await websocket.send_json(default_intervention.dict())

            # Handle idle end - reset idle time
            elif event.event_type == EventType.IDLE_END:
                session_state.idle_time = 0

            # Store interaction
            if session_key not in session_interactions:
                session_interactions[session_key] = []
            session_interactions[session_key].append(event)

            # Send acknowledgment with current state
            await websocket.send_json({
                "status": "received",
                "mastery_estimate": round(session_state.mastery_estimate, 3),
                "interventions_sent": session_state.interventions_sent
            })

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


@app.delete("/session/{user_id}/{session_id}")
async def clear_session(user_id: str, session_id: str):
    """Clear session data (called when session ends)"""
    session_key = f"{user_id}:{session_id}"

    removed_mouse = session_mouse_events.pop(session_key, None)
    removed_interactions = session_interactions.pop(session_key, None)

    # Clear TD-BKT intervention engine state
    intervention_engine.clear_session(user_id, session_id)

    return {
        "status": "cleared",
        "mouse_events_removed": len(removed_mouse) if removed_mouse else 0,
        "interactions_removed": len(removed_interactions) if removed_interactions else 0
    }


@app.get("/mastery/{user_id}/{session_id}")
async def get_mastery_estimate(user_id: str, session_id: str):
    """
    Get current TD-BKT mastery estimate for session.

    Research alignment: Provides real-time knowledge estimate
    updated from behavioral telemetry signals.
    """
    session_state = intervention_engine.get_or_create_session(user_id, session_id)

    return {
        "user_id": user_id,
        "session_id": session_id,
        "concept_id": session_state.current_concept_id,
        "mastery_estimate": round(session_state.mastery_estimate, 3),
        "affect_state": session_state.affect_state.value,
        "frustration_score": round(session_state.frustration_score, 3),
        "time_on_concept": session_state.time_on_concept,
        "interventions_sent": session_state.interventions_sent
    }


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    print("✅ Telemetry service started")
    print("📊 Real-time stealth assessment active")


@app.on_event("shutdown")
async def shutdown_event():
    if redis_client:
        await redis_client.close()
    print("🛑 Telemetry service stopped")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, ws_ping_interval=20)
