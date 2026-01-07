# NerdLearn Architecture Documentation

## System Overview

NerdLearn implements a **microservices architecture** designed for sub-100ms cognitive adaptation. The system is divided into four main layers:

1. **Presentation Layer**: Next.js 14 frontend
2. **Service Layer**: FastAPI microservices
3. **Data Layer**: Polyglot persistence
4. **Integration Layer**: Event streaming and caching

---

## Design Principles

### 1. Real-Time Adaptation
- Event streaming via Redpanda (Kafka-compatible)
- WebSocket connections for live telemetry
- Redis caching for sub-10ms state retrieval
- Target latency: <100ms for adaptive decisions

### 2. Separation of Concerns
Each microservice has a single responsibility:
- **Scheduler**: When to review (FSRS algorithm)
- **Telemetry**: What learner is doing (behavioral signals)
- **Inference**: What learner knows (DKT + ZPD)
- **Content**: What to learn (graph construction)

### 3. Polyglot Persistence
Different data types require different databases:
- **PostgreSQL**: ACID transactions (user accounts, enrollments)
- **Neo4j**: Graph queries (concept relationships)
- **TimescaleDB**: Time-series (behavioral logs)
- **Milvus**: Vector similarity (semantic search)
- **Redis**: Key-value (session state)

### 4. Research-Backed Algorithms
Every algorithm is based on peer-reviewed research:
- FSRS: Jarrett Ye (2023)
- SAINT+: Shin et al. (2021)
- ECD: Mislevy et al. (2003)
- ZPD: Vygotsky (1978), implemented via modern ML

---

## Service Details

### Scheduler Service (Port 8001)

**Responsibility**: Determine optimal review intervals

**Algorithm**: FSRS (Free Spaced Repetition Scheduler)

**API Endpoints**:
- `POST /review` - Process a review and update schedule
- `GET /preview/{learner_id}/{card_id}` - Preview intervals for all ratings
- `GET /due/{learner_id}` - Get cards due for review
- `POST /config` - Update scheduler configuration

**State Management**:
- Card states stored in Redis
- Key format: `card:{learner_id}:{card_id}`
- TTL: 1 year

**Performance**:
- Average calculation time: 3ms
- Throughput: ~10,000 schedules/second
- Memory footprint: ~10MB per 100K cards

### Telemetry Service (Port 8002)

**Responsibility**: Capture and analyze behavioral signals

**Algorithm**: Evidence-Centered Design (ECD)

**API Endpoints**:
- `POST /event` - Ingest single telemetry event
- `POST /batch` - Batch ingest (more efficient)
- `GET /analysis/mouse/{user_id}/{session_id}` - Mouse dynamics analysis
- `GET /analysis/engagement/{user_id}/{session_id}` - Engagement score
- `WS /ws/{user_id}/{session_id}` - WebSocket for real-time streaming

**Event Types**:
1. `mouse_move` - Mouse position tracking
2. `mouse_click` - Click events
3. `key_press` - Keyboard input
4. `scroll` - Scroll events
5. `focus/blur` - Window focus changes
6. `page_view` - Navigation events
7. `content_interaction` - Resource interactions

**Evidence Rules**:

| Signal | Evidence Type | Weight | Decay Rate |
|--------|---------------|--------|------------|
| Valid dwell time | IMPLICIT_ENGAGEMENT | 0.7 | 0.95/day |
| Low mouse velocity | IMPLICIT_STRUGGLE | 0.6 | 0.90/day |
| High trajectory entropy | IMPLICIT_STRUGGLE | 0.8 | 0.90/day |
| Correct submission | EXPLICIT_CORRECT | 1.0 | 0.98/day |

**Performance**:
- Event ingestion latency: <50ms (p95)
- WebSocket throughput: 1000 events/second
- Analysis latency: <100ms
- Storage: TimescaleDB with 1ms resolution

### Inference Engine (Port 8003)

**Responsibility**: Knowledge tracking and difficulty regulation

**Algorithms**:
1. SAINT+ (Deep Knowledge Tracing)
2. ZPD Regulator

**API Endpoints**:
- `POST /predict` - Predict performance on concept
- `POST /knowledge-state` - Get full knowledge vector
- `POST /zpd/assess` - Assess ZPD state after attempt
- `POST /zpd/scaffold` - Apply scaffolding
- `DELETE /zpd/scaffold` - Remove scaffolding
- `POST /recommend` - Get adaptive recommendations

**SAINT+ Model Architecture**:
```
Input Layer:
  - Exercise Embedding (128-dim)
  - Response Embedding (128-dim)
  - Position Encoding (128-dim)

Transformer Encoder:
  - 4 layers
  - 8 attention heads
  - 512 feedforward dim
  - Dropout: 0.1

Output Layer:
  - Linear projection to num_concepts
  - Sigmoid activation
  - Output: P(correct) for each concept
```

**ZPD Zones**:
- **Frustration** (<35% success): Strong scaffolding
- **Optimal** (35-70% success): Maintain challenge
- **Comfort** (>70% success): Increase difficulty

**Scaffolding Types**:
1. Worked Examples (most support)
2. Partial Solutions (hints)
3. Prerequisite Review (knowledge gaps)
4. Difficulty Reduction (simpler version)

**Performance**:
- DKT inference: 320ms (CPU), 45ms (GPU)
- ZPD assessment: 5ms
- Memory: ~2GB for 1000 concepts (model loaded)

---

## Data Models

### Learner Profile (PostgreSQL)

```typescript
LearnerProfile {
  id: string
  userId: string

  // Cognitive State
  cognitiveEmbedding: JSON  // 128-dim vector

  // FSRS Parameters (personalized)
  fsrsStability: number
  fsrsDifficulty: number
  fsrsRetrievability: number

  // ZPD State
  currentZpdLower: number (default 0.35)
  currentZpdUpper: number (default 0.70)
  optimalDifficulty: number

  // Gamification
  totalXP: number
  level: number
  streakDays: number
}
```

### Competency State (PostgreSQL)

```typescript
CompetencyState {
  id: string
  learnerId: string
  conceptId: string

  // Bayesian Mastery
  masteryProbability: number  // P(Mastery|Evidence)
  confidence: number

  // DKT State
  knowledgeState: number  // Hidden state value

  // Performance
  successRate: number
  totalAttempts: number

  // Scheduling
  nextReviewDue: DateTime
  itemStability: number
  itemDifficulty: number
}
```

### Knowledge Graph (Neo4j)

```cypher
// Nodes
(c:Concept {
  id: string,
  name: string,
  domain: string,
  taxonomyLevel: string,  // Bloom's taxonomy
  difficulty: float
})

// Relationships
(prerequisite:Concept)-[:HAS_PREREQUISITE {
  weight: float,       // Strength of dependency
  isStrict: boolean    // Must master before progression
}]->(dependent:Concept)
```

### Evidence Records (PostgreSQL)

```typescript
Evidence {
  id: string
  competencyId: string
  type: EvidenceType
  weight: number       // 0-1
  data: JSON          // Type-specific data
  timestamp: DateTime
  decayRate: number   // Per-day decay
}
```

---

## Event Flow Examples

### Example 1: Review Event

```
1. User submits answer (Frontend)
   ↓
2. POST /review to Scheduler Service
   ↓
3. Scheduler:
   - Load card state from Redis
   - Apply FSRS algorithm
   - Calculate new interval
   - Save state to Redis
   - Return new schedule
   ↓
4. POST /zpd/assess to Inference Engine
   ↓
5. Inference:
   - Record attempt in ZPD window
   - Calculate success rate
   - Determine zone (frustration/optimal/comfort)
   - Return recommended actions
   ↓
6. Frontend applies adaptations:
   - If frustration: Show worked example
   - If optimal: Continue
   - If comfort: Increase difficulty
```

### Example 2: Stealth Assessment

```
1. User views content (Frontend)
   ↓
2. Mouse movement events → WS /ws/{user}/{session}
   ↓
3. Telemetry Service:
   - Buffer events in memory (deque)
   - Publish to Redpanda (persistence)
   ↓
4. On content exit:
   GET /analysis/engagement/{user}/{session}
   ↓
5. Telemetry calculates:
   - Dwell time analysis
   - Mouse dynamics analysis
   - Engagement score
   ↓
6. If valid engagement:
   POST to database:
   - Create Evidence record
   - Type: IMPLICIT_ENGAGEMENT
   - Weight: 0.7
   ↓
7. Background job:
   - Update Bayesian belief for competency
   - Recalculate mastery probability
```

### Example 3: Adaptive Recommendation

```
1. User completes learning session
   ↓
2. POST /recommend to Inference Engine
   Request: {
     interaction_history: [...],
     available_concepts: [...]
   }
   ↓
3. Inference:
   a. Run DKT to get knowledge state vector
   b. For each concept:
      - Get ZPD state
      - Calculate priority score
      - Check prerequisites (Neo4j query)
   c. Sort by priority
   d. Return top 5 recommendations
   ↓
4. Frontend displays:
   - Next concept to learn
   - ZPD zone indicator
   - Estimated difficulty
   - Recommended resources
```

---

## Scaling Considerations

### Horizontal Scaling

**Stateless Services** (easy to scale):
- All FastAPI services are stateless
- Load balancer: Nginx or Traefik
- Auto-scaling based on CPU/memory

**Stateful Components** (require care):
- Redis: Use Redis Cluster for sharding
- PostgreSQL: Read replicas for queries
- Neo4j: Causal clustering for HA

### Database Optimization

**PostgreSQL**:
- Connection pooling (PgBouncer)
- Partitioning: `Evidence` table by timestamp
- Indexes: Composite on `(learnerId, conceptId)`

**Neo4j**:
- APOC procedures for batch operations
- Index on `Concept.id`, `Concept.domain`
- Periodic graph algorithm updates (offline)

**Redis**:
- Set TTL on all keys
- Use Redis Cluster for >100K concurrent users
- Eviction policy: `allkeys-lru`

### Monitoring

**Metrics to Track**:
- Service latency (p50, p95, p99)
- Error rates
- Database query times
- Event throughput (Redpanda)
- Model inference time
- Cache hit rate

**Tools**:
- Prometheus + Grafana (metrics)
- Sentry (error tracking)
- OpenTelemetry (distributed tracing)

---

## Security

### Authentication
- JWT tokens (short-lived: 15 min)
- Refresh tokens (longer: 7 days)
- HTTP-only cookies for token storage

### Authorization
- Role-Based Access Control (RBAC)
- Roles: STUDENT, INSTRUCTOR, ADMIN, RESEARCHER

### Data Privacy
- **FERPA Compliance**: No PII in behavioral logs
- **GDPR Compliance**: Right to deletion, data export
- **Federated Learning**: Train models client-side (future)

### API Security
- Rate limiting: 100 req/min per user
- CORS: Whitelist frontend domain
- Input validation: Pydantic models
- SQL injection: Parameterized queries (Prisma)

---

## Future Enhancements

### Phase 2
- Content ingestion pipeline (LayoutLMv3)
- Automated Knowledge Graph construction
- Vector database integration (Milvus)

### Phase 3
- Federated learning for privacy
- Multi-modal content (video, interactive)
- Collaborative learning features
- Mobile apps (React Native)

### Phase 4
- A/B testing framework
- Curriculum optimization (RL)
- Transfer learning across domains
- Enterprise features (SSO, SCIM)

---

## References

1. **FSRS**: https://github.com/open-spaced-repetition/fsrs4anki
2. **SAINT+**: https://arxiv.org/abs/2010.12042
3. **Evidence-Centered Design**: Mislevy, Steinberg, & Almond (2003)
4. **ZPD Theory**: Vygotsky, L. S. (1978). Mind in society.
