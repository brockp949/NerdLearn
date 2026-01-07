# Phase 3: Full Integration - Implementation Plan

## Overview

Phase 3 transforms NerdLearn from independent services into a **unified, intelligent learning system**. This phase delivers the MVP (Minimum Viable Product) with full end-to-end learning flows.

## Goals

1. ✅ **Working End-to-End Flow**: Upload content → Learn → Track progress
2. ✅ **Authentication & Security**: JWT-based auth, protected routes
3. ✅ **Real-Time Adaptation**: ZPD regulator actively adjusting difficulty
4. ✅ **Gamification**: XP, levels, achievements, streaks
5. ✅ **Analytics Dashboard**: Progress visualization, Knowledge Graph
6. ✅ **Service Integration**: All 4 microservices working together
7. ✅ **Basic Testing**: 50%+ coverage on critical paths

## Architecture Changes

### New Components

```
Frontend (Next.js 14)
├── /app
│   ├── (auth)
│   │   ├── login/
│   │   └── register/
│   ├── (protected)
│   │   ├── dashboard/          ← Main hub
│   │   ├── learn/              ← Learning interface
│   │   ├── progress/           ← Analytics
│   │   └── graph/              ← Knowledge Graph viz
│   └── api/
│       └── auth/               ← Auth endpoints
│
├── /components
│   ├── auth/
│   ├── learning/
│   │   ├── ContentViewer
│   │   ├── QuestionCard
│   │   └── ScaffoldingPanel
│   ├── dashboard/
│   │   ├── ProgressChart
│   │   ├── XPCounter
│   │   └── StreakTracker
│   └── graph/
│       └── KnowledgeGraphViz
│
└── /lib
    ├── auth.ts                 ← Auth utilities
    ├── websocket.ts            ← Real-time connection
    └── state/                  ← Zustand stores
```

### Backend Additions

```
services/
├── api-gateway/                ← NEW: Central API gateway
│   ├── main.py
│   ├── auth.py                ← JWT middleware
│   └── routes.py              ← Unified routes
│
└── orchestrator/               ← NEW: Learning flow coordinator
    ├── main.py
    └── learning_session.py    ← Session management
```

## Implementation Phases

### Week 1: Foundation (Days 1-3)

#### Day 1: Authentication System
- [ ] JWT token generation/validation
- [ ] User registration/login API
- [ ] Protected route middleware
- [ ] Frontend auth context
- [ ] Login/register pages

#### Day 2: API Gateway
- [ ] FastAPI gateway service
- [ ] Route all service calls through gateway
- [ ] CORS configuration
- [ ] Error handling middleware

#### Day 3: Database Setup & Seeding
- [ ] Run Prisma migrations
- [ ] Create demo user accounts
- [ ] Seed sample concepts
- [ ] Initialize Knowledge Graph

### Week 2: Core Features (Days 4-7)

#### Day 4: Learning Interface
- [ ] Content viewer component
- [ ] Question/answer cards
- [ ] Progress indicator
- [ ] Review interface

#### Day 5: Service Integration
- [ ] Orchestrator service (coordinates learning flow)
- [ ] Content → Scheduler integration
- [ ] Telemetry → Inference integration
- [ ] Real-time WebSocket setup

#### Day 6: ZPD Adaptation
- [ ] Live difficulty adjustment
- [ ] Scaffolding delivery system
- [ ] Success rate tracking
- [ ] Zone indicator UI

#### Day 7: Gamification
- [ ] XP calculation system
- [ ] Level progression
- [ ] Achievement tracking
- [ ] Streak system with freezes

### Week 3: Polish (Days 8-10)

#### Day 8: Dashboard & Analytics
- [ ] Progress charts (Recharts)
- [ ] Knowledge state visualization
- [ ] Learning statistics
- [ ] Recent activity feed

#### Day 9: Knowledge Graph Visualization
- [ ] Interactive graph (react-force-graph)
- [ ] Prerequisite highlighting
- [ ] Mastery coloring
- [ ] Learning path display

#### Day 10: Testing & Demo Content
- [ ] Unit tests (critical paths)
- [ ] Integration tests
- [ ] Create "Intro to Python" course
- [ ] End-to-end demo flow

## Detailed Component Specifications

### 1. Authentication System

**Tech Stack**: JWT + bcrypt + HTTP-only cookies

**Flow**:
```
1. User submits credentials
2. Backend validates → bcrypt.compare()
3. Generate JWT (15min access + 7day refresh)
4. Return HTTP-only cookie
5. Frontend stores user context
6. Middleware validates on protected routes
```

**Endpoints**:
- `POST /api/auth/register`
- `POST /api/auth/login`
- `POST /api/auth/logout`
- `POST /api/auth/refresh`
- `GET /api/auth/me`

### 2. Learning Interface

**Components**:

**ContentViewer**: Displays learning material
- Markdown rendering
- Code syntax highlighting
- Math equation support (KaTeX)
- Dwell time tracking (telemetry)

**QuestionCard**: Interactive questions
- Multiple choice
- Code input
- Free response
- Rating buttons (Again/Hard/Good/Easy)

**ScaffoldingPanel**: Adaptive help
- Worked examples (frustration zone)
- Hints (approaching frustration)
- Prerequisites links (knowledge gaps)
- Fades automatically (comfort zone)

### 3. Orchestrator Service

**Purpose**: Coordinates learning sessions

**Responsibilities**:
1. Load next due card (from Scheduler)
2. Get current ZPD state (from Inference)
3. Apply scaffolding if needed
4. Track session metrics
5. Update competency state
6. Emit events (Redpanda)

**API**:
```python
POST /session/start
  → Returns session_id, first card

POST /session/answer
  Body: {card_id, rating, response_data}
  → Updates all services
  → Returns next card + ZPD state

GET /session/state/{session_id}
  → Current session metrics
```

### 4. Gamification System

**XP Formula**:
```python
base_xp = 10
difficulty_multiplier = card.difficulty / 5  # 0.2 - 2.0
performance_bonus = {
    'again': 0.5,
    'hard': 0.8,
    'good': 1.0,
    'easy': 1.2
}

xp = base_xp * difficulty_multiplier * performance_bonus[rating]

# Streak bonus
if streak_days > 0:
    xp *= (1 + (streak_days * 0.05))  # 5% per day, max 50%
```

**Level Formula**:
```python
xp_for_level(n) = 100 * (n ** 1.5)

# Level 1: 100 XP
# Level 2: 283 XP
# Level 3: 520 XP
# Level 5: 1,118 XP
# Level 10: 3,162 XP
```

**Achievement Types**:
1. **Streak Milestones**: 3, 7, 14, 30, 100 days
2. **XP Milestones**: 1k, 5k, 10k, 50k, 100k
3. **Concept Mastery**: Master 10, 50, 100 concepts
4. **Speed Demon**: 10 cards in <5min
5. **Perfectionist**: 10 "Easy" ratings in a row
6. **Explorer**: Visit 5 different domains
7. **Helper**: (Future: peer learning)

### 5. Dashboard

**Widgets**:

**Progress Chart**: Line graph showing:
- Daily XP earned
- Success rate over time
- Cards reviewed per day
- Time spent learning

**Knowledge State Radar**: Radar chart
- 6 axes (one per Bloom's level)
- Shows cognitive distribution

**Recent Activity**: Timeline
- Last 10 learning events
- Achievements unlocked
- Level ups

**Streak Tracker**: Visual calendar
- Green = studied that day
- Gray = skipped
- Flame icon = current streak

**ZPD Zone Indicator**: Thermometer
- Red (frustration)
- Green (optimal)
- Yellow (comfort)

### 6. Knowledge Graph Visualization

**Tech**: react-force-graph-2d

**Features**:
- **Nodes**: Concepts (sized by frequency)
- **Colors**:
  - Green: Mastered (>80%)
  - Yellow: In progress (30-80%)
  - Red: Not started (<30%)
  - Gray: Locked (prerequisites not met)
- **Edges**: Prerequisites (arrows)
- **Interactions**:
  - Click node → view details
  - Hover → show tooltip
  - Drag to rearrange
  - Zoom/pan

**Layout**: Force-directed graph
- Prerequisites push nodes apart
- Related concepts pull together
- Creates natural learning paths

## Service Integration Flow

### End-to-End Learning Session

```
1. USER: Click "Start Learning"
   ↓
2. FRONTEND: POST /session/start
   ↓
3. ORCHESTRATOR:
   a. Create LearningSession (PostgreSQL)
   b. GET /due/{learner_id} (Scheduler)
   c. Load content (PostgreSQL)
   d. GET /zpd/assess (Inference)
   e. Return card + state
   ↓
4. FRONTEND: Display card, track telemetry
   ↓
5. USER: Submit answer (rating)
   ↓
6. FRONTEND: POST /session/answer
   Parallel:
   - POST /event (Telemetry) [mouse, dwell time]
   - WebSocket: Live engagement score
   ↓
7. ORCHESTRATOR:
   a. POST /review (Scheduler)
   b. POST /zpd/assess (Inference)
   c. Update CompetencyState (PostgreSQL)
   d. Calculate XP, check achievements
   e. Publish event (Redpanda)
   f. Return next card + updates
   ↓
8. FRONTEND: Update UI
   - New card
   - XP animation
   - Achievement popup (if earned)
   - ZPD zone indicator
   ↓
9. REPEAT steps 4-8 until session ends
```

## Data Synchronization

### Event-Driven Architecture

**Events Published to Redpanda**:

1. `concept.mastered` - Learner masters a concept
2. `achievement.unlocked` - New achievement earned
3. `level.up` - Level increased
4. `card.reviewed` - Card reviewed
5. `session.completed` - Learning session ended
6. `zpd.zone_change` - Moved between zones

**Consumers**:
- Analytics service (future)
- Notification service (future)
- Dashboard real-time updates

## Testing Strategy

### Unit Tests (pytest + jest)

**Backend**:
- `test_auth.py`: JWT generation, validation
- `test_fsrs.py`: Scheduling algorithm
- `test_zpd.py`: Zone detection, scaffolding
- `test_gamification.py`: XP, levels, achievements

**Frontend**:
- `auth.test.ts`: Auth context, hooks
- `QuestionCard.test.tsx`: Component rendering
- `gamification.test.ts`: XP calculations

### Integration Tests

- `test_learning_flow.py`: Full session E2E
- `test_service_integration.py`: Service-to-service calls
- `test_database_consistency.py`: PostgreSQL ↔ Neo4j sync

**Target Coverage**: 50%+ on critical paths

## Demo Content: "Intro to Python"

### Concepts (20):
1. Variables
2. Data Types (int, str, bool)
3. Operators
4. Control Flow (if/else)
5. Loops (for, while)
6. Functions
7. Lists
8. Dictionaries
9. Tuples
10. Sets
11. String Methods
12. List Comprehensions
13. Error Handling (try/except)
14. File I/O
15. Modules
16. Classes
17. Inheritance
18. Recursion
19. Decorators
20. Generators

### Prerequisites (examples):
- Variables → Functions
- Data Types → Operators
- Control Flow → Loops
- Functions → Recursion
- Lists → List Comprehensions

### Cards per Concept: 5-10
- Total: ~100-150 cards

## Performance Targets

| Metric | Target |
|--------|--------|
| Page load (dashboard) | <2s |
| Card transition | <300ms |
| ZPD calculation | <100ms |
| XP update | <50ms |
| WebSocket latency | <100ms |
| Graph rendering (100 nodes) | <1s |

## Security Checklist

- [x] JWT tokens (short-lived)
- [x] HTTP-only cookies
- [x] CORS whitelist
- [x] Input validation (Pydantic)
- [x] SQL injection prevention (Prisma)
- [x] XSS prevention (React escaping)
- [x] Rate limiting (future)
- [x] HTTPS (production only)

## Deployment Considerations

### Development:
- Run services locally (Python scripts)
- Docker for databases only
- Hot reload enabled

### Production (future):
- Containerize all services
- Kubernetes orchestration
- Load balancer (Nginx)
- CDN for static assets
- Database replicas

## Success Criteria

Phase 3 is complete when:

1. ✅ User can register → login → start learning
2. ✅ Cards are scheduled optimally (FSRS)
3. ✅ Difficulty adapts in real-time (ZPD)
4. ✅ XP/levels update automatically
5. ✅ Dashboard shows accurate progress
6. ✅ Knowledge Graph visualizes relationships
7. ✅ 50%+ test coverage
8. ✅ Demo video recordable (3-5 minutes)

## Timeline

**Week 1**: Foundation (auth, gateway, DB)
**Week 2**: Core features (learning, adaptation, gamification)
**Week 3**: Polish (dashboard, graph, tests)

**Total**: 15-20 working days

## Next Steps

1. Create API Gateway service
2. Implement authentication
3. Build Orchestrator service
4. Create learning interface
5. Integrate services
6. Add gamification
7. Build dashboard
8. Test end-to-end

Let's begin! 🚀
