# Phase 3 Remaining Tasks

## Current Status

### ✅ Completed (What We Just Built)
1. **API Gateway Service** (services/api-gateway/)
   - JWT authentication & refresh tokens
   - User registration/login endpoints
   - Protected route middleware
   - CORS configuration

2. **Orchestrator Service** (services/orchestrator/)
   - Session management
   - Gamification engine (XP, levels, achievements)
   - ZPD zone detection
   - Scaffolding delivery
   - Demo content (5 Python Function cards)

3. **Frontend Learning Interface**
   - QuestionCard component
   - ContentViewer component
   - ScaffoldingPanel component
   - LearningStats component
   - Learn page (/learn)
   - Auth integration
   - Dashboard links

---

## 🔴 Critical Missing Items (Blockers for Full MVP)

### 1. **Database Initialization** ⚠️ HIGH PRIORITY
**Status**: Schemas exist, but databases not initialized

**What's Missing:**
```bash
# Prisma not migrated
cd packages/db
npx prisma generate
npx prisma db push

# Neo4j not seeded
# No initial data in Knowledge Graph
```

**Impact**:
- Frontend works but uses **in-memory demo data**
- No persistence between sessions
- Can't test real database integration

**Effort**: 1-2 hours

**Tasks:**
- [ ] Run Prisma migrations to PostgreSQL
- [ ] Verify PostgreSQL connection
- [ ] Create initial Neo4j schema
- [ ] Seed demo user accounts (3 test users)
- [ ] Create initial concepts in Knowledge Graph

---

### 2. **Real Service Integration** ⚠️ HIGH PRIORITY
**Status**: Orchestrator uses in-memory data, doesn't call other services

**What's Missing:**
```python
# Orchestrator currently has:
DEMO_CARDS = [...]  # In-memory cards
LEARNER_PROFILES = {}  # In-memory profiles

# Should call:
- Scheduler service (port 8001) for FSRS scheduling
- Inference service (port 8003) for ZPD assessment
- Content service (port 8004) for loading concepts
- Telemetry service (port 8002) for tracking
```

**Impact**:
- Not using FSRS algorithm (using fake due dates)
- Not using DKT/ZPD models (using simple success rate)
- No real-time adaptation
- No telemetry tracking

**Effort**: 4-6 hours

**Tasks:**
- [ ] Connect Orchestrator → Scheduler (GET /due, POST /review)
- [ ] Connect Orchestrator → Inference (POST /zpd/assess)
- [ ] Connect Orchestrator → Content (GET /concepts)
- [ ] Replace in-memory data with database queries
- [ ] Add error handling for service failures

---

### 3. **WebSocket Integration for Telemetry** ⚠️ MEDIUM PRIORITY
**Status**: Telemetry service has WebSocket, but frontend doesn't connect

**What's Missing:**
```typescript
// Frontend needs:
- WebSocket connection to telemetry service
- Mouse tracking events
- Dwell time tracking (currently basic)
- Hesitation detection
- Real-time engagement scoring
```

**Impact**:
- No stealth assessment
- No real-time cognitive load detection
- Missing Evidence-Centered Design data

**Effort**: 2-3 hours

**Tasks:**
- [ ] Create WebSocket client in frontend
- [ ] Track mouse events on learning page
- [ ] Send events to telemetry service
- [ ] Display engagement score in UI
- [ ] Connect telemetry → inference for adaptive updates

---

## 🟡 Important Missing Items (MVP Nice-to-Have)

### 4. **Progress Analytics Page** 📊
**Status**: Dashboard has static stats, no analytics page

**What's Missing:**
- `/progress` page with charts
- Daily XP line chart (Recharts)
- Success rate over time
- Knowledge state radar chart
- Recent activity timeline
- Streak calendar visualization

**Effort**: 4-6 hours

**Tasks:**
- [ ] Create /app/(protected)/progress/page.tsx
- [ ] Install Recharts (may already be in package.json)
- [ ] Build ProgressChart component (line chart)
- [ ] Build RadarChart component (Bloom's levels)
- [ ] Build ActivityFeed component
- [ ] Build StreakCalendar component
- [ ] Fetch real data from API

---

### 5. **Knowledge Graph Visualization** 🌳
**Status**: Neo4j has structure, no visualization exists

**What's Missing:**
- `/graph` page with interactive graph
- react-force-graph-2d integration (already in package.json!)
- Prerequisite highlighting
- Mastery coloring (green/yellow/red)
- Learning path display
- Click to view concept details

**Effort**: 4-6 hours

**Tasks:**
- [ ] Create /app/(protected)/graph/page.tsx
- [ ] Create KnowledgeGraphViz component
- [ ] Query Neo4j for graph data (nodes + edges)
- [ ] Implement force-directed layout
- [ ] Color nodes by mastery level
- [ ] Add click handlers for concept details
- [ ] Add zoom/pan controls

---

### 6. **Demo Content Creation** 📚
**Status**: 5 demo cards for Python Functions only

**What's Missing:**
- Full "Intro to Python" course (20 concepts)
- 100-150 total learning cards
- Prerequisites properly set
- Content with:
  - Markdown explanations
  - Code examples
  - Questions with answers
  - Difficulty ratings

**Effort**: 8-12 hours (content creation)

**Tasks:**
- [ ] Write content for 20 Python concepts (see Phase 3 plan)
- [ ] Create 5-10 cards per concept
- [ ] Set up prerequisite relationships
- [ ] Load into PostgreSQL + Neo4j
- [ ] Test complete learning paths

---

### 7. **Testing Infrastructure** 🧪
**Status**: 0% test coverage

**What's Missing:**
- Unit tests (pytest for Python, jest for TypeScript)
- Integration tests
- E2E tests
- CI/CD pipeline

**Effort**: 1-2 days

**Tasks:**
- [ ] Set up pytest in services/
- [ ] Write tests for:
  - FSRS algorithm (scheduler)
  - XP calculation (orchestrator)
  - JWT auth (api-gateway)
  - ZPD zone detection
- [ ] Set up jest in apps/web
- [ ] Write component tests:
  - QuestionCard
  - Auth context
- [ ] Add integration test for learning flow
- [ ] Target: 50% coverage minimum

---

## 🟢 Minor Enhancements (Future)

### 8. **Error Handling & Polish**
**Tasks:**
- [ ] Better error messages in UI
- [ ] Loading states for all API calls
- [ ] Retry logic for failed requests
- [ ] Toast notifications for achievements
- [ ] Offline mode detection

**Effort**: 2-3 hours

---

### 9. **Service Startup Scripts**
**Tasks:**
- [ ] Add `if __name__ == "__main__"` to services without it
- [ ] Create start-all.sh script
- [ ] Create stop-all.sh script
- [ ] Add health check endpoints
- [ ] Document startup order

**Effort**: 1 hour

---

### 10. **Documentation Updates**
**Tasks:**
- [ ] API reference documentation
- [ ] Deployment guide
- [ ] Troubleshooting guide
- [ ] Contributing guide
- [ ] Demo video recording

**Effort**: 2-4 hours

---

## Priority Roadmap

### Phase 3A: Core MVP (Highest Priority) - 1-2 days
1. ✅ ~~Learning Interface~~ (DONE!)
2. **Database Initialization** (1-2 hours)
3. **Real Service Integration** (4-6 hours)
4. **Basic Testing** (4 hours minimum)

**Goal**: Fully functional end-to-end flow with real algorithms

---

### Phase 3B: Analytics & Visualization - 1 day
5. **Progress Analytics Page** (4-6 hours)
6. **Knowledge Graph Visualization** (4-6 hours)

**Goal**: Complete MVP with full feature set

---

### Phase 3C: Content & Polish - 2-3 days
7. **Demo Content Creation** (8-12 hours)
8. **Error Handling & Polish** (2-3 hours)
9. **Service Startup Scripts** (1 hour)
10. **Documentation Updates** (2-4 hours)

**Goal**: Production-ready, demo-able system

---

## What Can Be Tested NOW

Even without the remaining work, you can test:

✅ **Working Features:**
- User registration & login
- Protected routes
- Learning session flow (with demo data)
- Question answering with ratings
- XP earning & level progression
- Achievement unlocking
- ZPD zone detection (basic)
- Scaffolding display
- Session progress tracking
- Beautiful, responsive UI

**Services Running:**
- 🟢 Orchestrator (port 8005)
- 🟢 Next.js Frontend (port 3000)

**Services NOT Running:**
- ⚪ API Gateway (port 8000) - exists but not started
- ⚪ Scheduler (port 8001) - exists but not integrated
- ⚪ Telemetry (port 8002) - exists but not integrated
- ⚪ Inference (port 8003) - exists but not integrated
- ⚪ Content Ingestion (port 8004) - exists but not integrated
- ⚪ Databases (Docker) - exist but not initialized with data

---

## Recommended Next Steps

### Option A: Complete Core MVP (Recommended)
**Focus on making everything work end-to-end**

1. Initialize databases (1-2 hours)
2. Integrate real services (4-6 hours)
3. Add basic tests (4 hours)
4. Record demo video

**Timeline**: 2 days
**Output**: Fully functional system with real algorithms

---

### Option B: Add Visualization First
**Make it look amazing before integrating**

1. Build Progress Analytics page (4-6 hours)
2. Build Knowledge Graph visualization (4-6 hours)
3. Polish UI/UX
4. Then integrate services

**Timeline**: 2 days
**Output**: Beautiful UI with demo data

---

### Option C: Content Creation First
**Make the learning experience comprehensive**

1. Write full Python course content (8-12 hours)
2. Create 100+ learning cards
3. Set up Knowledge Graph properly
4. Then integrate everything

**Timeline**: 2-3 days
**Output**: Rich content with basic integration

---

## Summary

### Completed Today:
- ✅ Orchestrator service (557 lines)
- ✅ Complete learning interface (4 components)
- ✅ Gamification system
- ✅ ZPD adaptation UI
- ✅ Session management
- ✅ Auth integration

### Total Phase 3 Progress:
**~40% Complete**

### Remaining for Full MVP:
- Database initialization & seeding
- Real service integration (not using in-memory data)
- WebSocket telemetry connection
- Progress analytics page
- Knowledge Graph visualization
- Demo content (full course)
- Testing (50% coverage)

### Time to Full MVP:
**5-7 working days** (conservative estimate)

---

**What would you like to focus on next?**
