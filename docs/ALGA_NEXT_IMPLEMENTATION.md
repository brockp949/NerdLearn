# ALGA-Next: Adaptive Learning via Generative Allocation

## Implementation Status: COMPLETE (January 2025)

Successfully implemented the ALGA-Next framework from the research paper "Adaptive Multimodal Learning with Attention Transfer" as part of NerdLearn's adaptive learning system.

---

## Overview

ALGA-Next is a research-aligned implementation of an Adaptive Content Modality Selection System (ACMSS) based on the **Engagement-Mediated Learning Hypothesis**. The system selects optimal content modalities (text, video, interactive, audio, diagram) based on real-time learner state inference.

### Key Design Principles
1. **Engagement-Mediated Learning**: Optimizes for context-dependent engagement rather than static "learning styles"
2. **Closed Feedback Loop**: Sensors -> Feature Engineering -> Bandit Core -> Generative UI -> Feedback
3. **Cold-Start Handling**: Attention Transfer enables prediction even without modality-specific data
4. **Pedagogical Alignment**: Reward function prevents "clickbait" optimization

---

## Components Implemented

### 1. Hybrid LinUCB Contextual Bandit
**File:** `apps/api/app/adaptive/alga_next/hybrid_linucb.py`

Implements the Hybrid LinUCB algorithm with interaction terms for modality selection:

```
E[r_t,a | x_t,a] = x_t,a^T * beta* + z_t,a^T * theta*_a
```

**Features:**
- Shared parameters (beta*) across all modalities
- Per-arm parameters (theta*_a) for modality-specific effects
- Interaction feature builder: Fatigue x Difficulty, Device x Modality, Time x Complexity
- Ridge regression with UCB exploration bonus
- Thompson Sampling option for exploration

**Key Classes:**
- `HybridLinUCB`: Main bandit implementation
- `ModalityArm`: Represents a content modality option
- `ContextVector`: User/session context features
- `InteractionFeatureBuilder`: Creates interaction terms

---

### 2. MouStress Mouse Dynamics Analysis
**File:** `apps/api/app/adaptive/alga_next/mouse_stress.py`

Based on the MouStress framework (Sun et al.) for cognitive state inference from mouse dynamics.

**Temporal Thresholds (from research):**
- Idle threshold: 310ms (pause detection)
- Perception threshold: 100ms
- Micro-hesitation: 100-200ms

**Kinematic Features:**
- Damping ratio (zeta)
- Natural frequency (omega_n)
- Stiffness (K)
- Tremor amplitude/frequency

**Learner States Detected:**
- FLOW: Optimal learning state
- CONFUSION: Needs scaffolding
- FRUSTRATION: Consider modality switch
- FATIGUE: Suggest break
- DISTRACTION: Re-engagement needed
- READING: Normal content consumption

**Key Classes:**
- `MouStressAnalyzer`: Main analysis class
- `TrajectoryAnalysis`: Path-based features
- `KinematicStiffness`: Mass-spring-damper model
- `LearnerState`: Enumeration of states

---

### 3. MMSAF-Net (Multi-Modal Self-Attention Fusion)
**File:** `apps/api/app/adaptive/alga_next/mmsaf_net.py`

Neural network for fusing multi-modal telemetry into a User State Vector.

**Architecture:**
1. Feature encoders for behavioral, contextual, and content features
2. Multi-head self-attention (4 heads, 64 dimensions)
3. Cross-modal attention for dynamic feature weighting
4. Fusion layer producing 8-dimensional state vector

**User State Vector Components:**
- cognitive_capacity (0-1)
- fatigue_level (0-1)
- focus_level (0-1)
- engagement (0-1)
- frustration (0-1)
- confidence (0-1)
- flow_state (0-1)
- confusion_indicator (0-1)

**Key Classes:**
- `MMSAFNet`: Main fusion network (PyTorch + NumPy fallback)
- `UserStateVector`: Output state representation
- `BehavioralFeatures`: Mouse/interaction features
- `ContextualFeatures`: Session/device context
- `ContentFeatures`: Current content metadata

---

### 4. Attention Transfer / Multi-Task Learning
**File:** `apps/api/app/adaptive/alga_next/attention_transfer.py`

Solves the cold-start problem using cross-modality transfer.

**Architecture:**
- Shared encoder (user behavior patterns)
- Modality-specific prediction heads
- Transfer matrix with learned correlations

**Transfer Matrix Priors (from research):**
```
Interactive -> Video: 0.75
Video -> Text: 0.65
Text -> Audio: 0.60
Audio -> Interactive: 0.50
```

**Key Classes:**
- `AttentionTransferNetwork`: Main network
- `CrossModalityTransferMatrix`: Transfer relationships
- `UserObservation`: Recorded learning observation
- `TransferPredictions`: Predictions for all modalities

---

### 5. Composite Reward Function
**File:** `apps/api/app/adaptive/alga_next/reward_function.py`

Multi-objective reward aligned with pedagogical goals.

**Formula:**
```
R_t = w1 * E_norm + w2 * M_norm * (1 / (1 + exp(k * (F_t - tau))))
```

Where:
- E_norm: Normalized engagement (dwell time, scroll, interaction)
- M_norm: Mastery (assessment score, concept recall)
- Fatigue Penalty: Sigmoid decay when fatigue > threshold tau

**Reward Objectives:**
- ENGAGEMENT: Maximize time on task
- MASTERY: Maximize learning outcomes
- BALANCED: Default balance (0.4/0.4/0.2)
- RETENTION: Long-term memory focus
- WELLBEING: Include burnout prevention

**Key Classes:**
- `CompositeRewardFunction`: Main reward calculator
- `RewardComponents`: Input metrics
- `FatiguePenalty`: Configurable sigmoid penalty
- `RewardConfig`: Objective presets

---

### 6. Generative UI / SDUI Registry
**File:** `apps/api/app/adaptive/alga_next/generative_ui.py`

Server-Driven UI based on Microsoft Adaptive Cards specification with IEEE LOM extensions.

**Atomic Content Units:**
- Based on IEEE Learning Object Metadata standard
- Types: Text, Image, Video, Audio, Interactive, Assessment, CodeBlock, etc.
- Adaptivity metadata: difficulty modifiers, prerequisite visibility, time constraints

**Scaffolding Levels:**
- NONE: Full content
- MINIMAL: Hints available
- MODERATE: Guided questions
- INTENSIVE: Step-by-step breakdown

**Layout Types:**
- SINGLE_COLUMN: Default vertical layout
- TWO_COLUMN: Side-by-side content
- HERO_MEDIA: Prominent media + text
- QUIZ_FOCUS: Question-centric
- INTERACTIVE: Simulation-focused

**Key Classes:**
- `GenerativeUIRegistry`: Content registration and schema generation
- `AtomicContentUnit`: IEEE LOM-based content unit
- `SDUISchema`: Complete UI schema with layout instructions
- `AdaptiveCard`: Individual card specification

---

### 7. ALGA-Next Orchestrator
**File:** `apps/api/app/adaptive/alga_next/orchestrator.py`

Main orchestration service integrating all components.

**Feedback Loop:**
```
1. Telemetry -> MouStress (cognitive state)
2. Features -> MMSAF-Net (user state vector)
3. Context -> Hybrid LinUCB (modality selection)
4. Cold-start -> Attention Transfer (predictions)
5. Selection -> Generative UI (SDUI schema)
6. Outcome -> Reward Function (learning signal)
7. Update -> LinUCB + Transfer (model update)
```

**Key Methods:**
- `process_telemetry()`: Convert raw telemetry to user state
- `select_modality()`: Run full selection pipeline
- `record_outcome()`: Update models with learning result
- `get_statistics()`: Monitor system health

---

### 8. FastAPI Router
**File:** `apps/api/app/routers/alga_next.py`

REST and WebSocket endpoints for the ALGA-Next system.

**REST Endpoints:**
- `POST /alga-next/select-modality`: Get optimal modality for a concept
- `POST /alga-next/record-outcome`: Record learning outcome
- `POST /alga-next/process-telemetry`: Process telemetry batch
- `GET /alga-next/user-state/{user_id}`: Get current user state
- `GET /alga-next/transfer-matrix`: Get cross-modality transfer matrix
- `GET /alga-next/statistics`: Get orchestrator statistics
- `GET /alga-next/modality-predictions/{user_id}`: Predict all modalities

**WebSocket Endpoint:**
- `WS /alga-next/ws/{user_id}/{session_id}`: Real-time telemetry and adaptation

**Message Types (WebSocket):**
- `mouse_events`: Batch of mouse events
- `interaction`: User interactions
- `dwell_update`: Dwell time updates
- `scroll`: Scroll depth updates
- `heartbeat`: Keep-alive + state sync
- `outcome`: Learning outcome recording

---

## Integration

The ALGA-Next router is registered in `apps/api/app/main.py`:

```python
from app.routers import alga_next

app.include_router(alga_next.router, prefix="/api", tags=["ALGA-Next Adaptive Learning"])
```

---

## Usage Example

### REST API

```python
import httpx

# Select optimal modality
response = await client.post("/api/alga-next/select-modality", json={
    "user_id": "user-123",
    "concept_id": "python-functions",
    "available_content": [
        {"id": "text-1", "modality": "text", "duration_minutes": 5.0},
        {"id": "video-1", "modality": "video", "duration_minutes": 8.0},
        {"id": "interactive-1", "modality": "interactive", "duration_minutes": 10.0},
    ],
    "device": "desktop",
    "telemetry": {
        "session_id": "session-456",
        "user_id": "user-123",
        "dwell_time_ms": 45000,
        "scroll_depth": 0.7,
        "mouse_events": [...]
    }
})

result = response.json()
# {
#     "selected_modality": "video",
#     "selected_content_id": "video-1",
#     "confidence": 0.85,
#     "user_state": {...},
#     "learner_state": "flow",
#     "scaffolding_level": "none",
#     "ui_schema": {...}
# }
```

### WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/api/alga-next/ws/user-123/session-456');

ws.onopen = () => {
    // Send mouse events
    ws.send(JSON.stringify({
        type: 'mouse_events',
        events: [
            { x: 100, y: 200, timestamp: 1706000000000, type: 'move' },
            // ...
        ]
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'state_update') {
        console.log('Learner state:', data.learner_state);
        console.log('Cognitive load:', data.cognitive_load);
    } else if (data.type === 'intervention') {
        showIntervention(data.message, data.priority);
    }
};
```

---

## Research References

1. **Hybrid LinUCB**: Li et al., "A Contextual-Bandit Approach to Personalized News Article Recommendation"
2. **MouStress**: Sun et al., "MouStress: Detecting Stress from Mouse Motion Dynamics"
3. **Attention Transfer**: "Adaptive Multimodal Orchestration: A Context-Aware Framework"
4. **IEEE LOM**: IEEE 1484.12.1-2002 Learning Object Metadata Standard
5. **Adaptive Cards**: Microsoft Adaptive Cards Specification

---

## Files Created

| File | Description | Lines |
|------|-------------|-------|
| `alga_next/__init__.py` | Package exports | 111 |
| `alga_next/hybrid_linucb.py` | Contextual bandit | ~580 |
| `alga_next/mouse_stress.py` | Mouse dynamics | ~650 |
| `alga_next/mmsaf_net.py` | Neural fusion | ~570 |
| `alga_next/attention_transfer.py` | Cold-start MTL | ~510 |
| `alga_next/reward_function.py` | Composite reward | ~480 |
| `alga_next/generative_ui.py` | SDUI registry | ~640 |
| `alga_next/orchestrator.py` | Main orchestrator | ~520 |
| `routers/alga_next.py` | API endpoints | ~530 |

**Total:** ~4,600 lines of research-aligned adaptive learning code

---

## Next Steps

1. **Frontend Integration**: Connect telemetry client to ALGA-Next WebSocket
2. **Model Training**: Train MMSAF-Net on real telemetry data
3. **A/B Testing**: Compare ALGA-Next with existing content selection
4. **Evaluation**: Measure engagement and learning outcomes
