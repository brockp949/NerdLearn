# Phase 3B: Analytics & Visualization

## 🎯 Goal
Transform NerdLearn's dashboard from basic stats into a **comprehensive analytics and visualization platform** that shows learners their progress, knowledge mastery, and learning patterns.

---

## 📋 Overview

### Current State
- ✅ Basic dashboard with session stats
- ✅ XP and level display
- ✅ Streak counter
- ⚠️ **No progress visualization** (no charts)
- ⚠️ **No Knowledge Graph visualization**
- ⚠️ **No learning analytics** (no insights)
- ⚠️ **No activity timeline** (no history view)

### Target State
- ✅ Interactive progress charts (XP over time, success rate, etc.)
- ✅ Knowledge Graph visualization (concepts and prerequisites)
- ✅ Learning analytics dashboard (insights and recommendations)
- ✅ Activity timeline (recent sessions, achievements)
- ✅ Concept mastery view (strengths and weaknesses)
- ✅ Performance metrics (response time, accuracy, engagement)

---

## 🗓️ Timeline: 2-3 Days

### Day 1: Progress Analytics (Chunk 1)
**Morning:** Data fetching and aggregation
**Afternoon:** Chart components and visualizations

### Day 2: Knowledge Graph Visualization (Chunk 2)
**Morning:** Neo4j data fetching
**Afternoon:** Interactive graph component

### Day 3: Dashboard Polish & Integration (Chunk 3)
**Morning:** Activity timeline and insights
**Afternoon:** Responsive design and final touches

---

## 📝 Detailed Task Breakdown

## CHUNK 1: Progress Analytics & Charts

### Goal
Create interactive charts showing learning progress over time

### Tasks

#### 1.1: Analytics API Endpoints
**File:** `services/api-gateway/main.py` (or new analytics service)

**Endpoints to Create:**
```python
# GET /api/analytics/progress/{learner_id}
# Returns: XP over time, success rate, sessions completed

# GET /api/analytics/concepts/{learner_id}
# Returns: Mastery level per concept, strengths/weaknesses

# GET /api/analytics/performance/{learner_id}
# Returns: Average dwell time, accuracy, engagement scores

# GET /api/analytics/activity/{learner_id}
# Returns: Recent sessions, achievements, streak history
```

**Sample Response:**
```json
{
  "progress": {
    "xp_history": [
      {"date": "2026-01-01", "xp": 50},
      {"date": "2026-01-02", "xp": 120},
      {"date": "2026-01-07", "xp": 245}
    ],
    "success_rate_history": [
      {"date": "2026-01-01", "rate": 0.65},
      {"date": "2026-01-07", "rate": 0.72}
    ],
    "sessions_per_day": [
      {"date": "2026-01-01", "count": 2},
      {"date": "2026-01-07", "count": 3}
    ]
  }
}
```

---

#### 1.2: Chart Components
**Files:**
- `apps/web/src/components/analytics/ProgressChart.tsx`
- `apps/web/src/components/analytics/ConceptMasteryChart.tsx`
- `apps/web/src/components/analytics/PerformanceMetrics.tsx`

**Libraries to Install:**
```bash
cd apps/web
pnpm add recharts
pnpm add @tremor/react  # Alternative: lightweight chart library
pnpm add date-fns       # For date formatting
```

**ProgressChart Component:**
```typescript
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts'

export function ProgressChart({ data }: ProgressChartProps) {
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold mb-4">📈 Learning Progress</h3>

      <LineChart width={600} height={300} data={data.xp_history}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="xp" stroke="#8884d8" name="Total XP" />
      </LineChart>
    </div>
  )
}
```

**ConceptMasteryChart (Radar Chart):**
```typescript
import { RadarChart, Radar, PolarGrid, PolarAngleAxis } from 'recharts'

export function ConceptMasteryChart({ concepts }: ConceptMasteryChartProps) {
  return (
    <RadarChart width={400} height={400} data={concepts}>
      <PolarGrid />
      <PolarAngleAxis dataKey="concept" />
      <Radar name="Mastery" dataKey="mastery" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
    </RadarChart>
  )
}
```

**PerformanceMetrics (Stat Cards):**
```typescript
export function PerformanceMetrics({ stats }: PerformanceMetricsProps) {
  return (
    <div className="grid grid-cols-3 gap-4">
      <StatCard
        title="Average Accuracy"
        value={`${(stats.avg_accuracy * 100).toFixed(0)}%`}
        trend={stats.accuracy_trend}
        icon="🎯"
      />
      <StatCard
        title="Study Time"
        value={formatDuration(stats.total_study_time_ms)}
        trend={stats.time_trend}
        icon="⏱️"
      />
      <StatCard
        title="Engagement"
        value={`${(stats.avg_engagement * 100).toFixed(0)}%`}
        trend={stats.engagement_trend}
        icon="🧠"
      />
    </div>
  )
}
```

---

#### 1.3: Progress Page
**File:** `apps/web/src/app/(protected)/progress/page.tsx`

**Layout:**
```
┌─────────────────────────────────────────────────┐
│ 📊 Your Learning Analytics                      │
├─────────────────────────────────────────────────┤
│                                                 │
│ ┌─────────────────┐  ┌─────────────────┐      │
│ │  🎯 Accuracy    │  │  ⏱️ Study Time   │      │
│ │     72%         │  │    12.5 hours    │      │
│ └─────────────────┘  └─────────────────┘      │
│                                                 │
│ ┌──────────────────────────────────────┐       │
│ │ 📈 XP Progress Over Time             │       │
│ │   [Line Chart: XP by date]           │       │
│ └──────────────────────────────────────┘       │
│                                                 │
│ ┌──────────────────────────────────────┐       │
│ │ 🎯 Success Rate Trend                │       │
│ │   [Line Chart: Success rate by date] │       │
│ └──────────────────────────────────────┘       │
│                                                 │
│ ┌──────────────────────────────────────┐       │
│ │ 📚 Concept Mastery                   │       │
│ │   [Radar Chart: Mastery per concept] │       │
│ └──────────────────────────────────────┘       │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## CHUNK 2: Knowledge Graph Visualization

### Goal
Interactive visualization of concept relationships and prerequisites

### Tasks

#### 2.1: Knowledge Graph API Endpoint
**File:** `services/api-gateway/main.py` or `services/content-ingestion/main.py`

**Endpoint:**
```python
# GET /api/knowledge-graph/{learner_id}
# Returns: Concept nodes, prerequisite edges, mastery status

@app.get("/api/knowledge-graph/{learner_id}")
async def get_knowledge_graph(learner_id: str):
    # Query Neo4j for concepts and prerequisites
    query = """
    MATCH (c:Concept)
    OPTIONAL MATCH (c)-[r:PREREQUISITE]->(p:Concept)
    RETURN c, r, p
    """

    # Get mastery status from PostgreSQL
    mastery = await get_concept_mastery(learner_id)

    return {
        "nodes": [
            {
                "id": concept.id,
                "name": concept.name,
                "domain": concept.domain,
                "mastery": mastery.get(concept.id, 0),
                "color": get_color_by_mastery(mastery.get(concept.id, 0))
            }
            for concept in concepts
        ],
        "edges": [
            {
                "source": prereq.source_id,
                "target": prereq.target_id,
                "weight": prereq.weight,
                "type": "prerequisite"
            }
            for prereq in prerequisites
        ]
    }
```

---

#### 2.2: Graph Visualization Component
**File:** `apps/web/src/components/analytics/KnowledgeGraphView.tsx`

**Install Dependencies:**
```bash
cd apps/web
pnpm add react-force-graph-2d
pnpm add d3-force
```

**Component:**
```typescript
import ForceGraph2D from 'react-force-graph-2d'

export function KnowledgeGraphView({ learnerId }: Props) {
  const [graphData, setGraphData] = useState<GraphData | null>(null)

  useEffect(() => {
    fetch(`/api/knowledge-graph/${learnerId}`)
      .then(r => r.json())
      .then(data => setGraphData(data))
  }, [learnerId])

  if (!graphData) return <div>Loading graph...</div>

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold mb-4">🕸️ Knowledge Graph</h3>

      <ForceGraph2D
        graphData={graphData}
        nodeLabel="name"
        nodeColor={(node) => node.color}
        nodeCanvasObject={(node, ctx, globalScale) => {
          // Custom node rendering
          const label = node.name
          const fontSize = 12 / globalScale
          ctx.font = `${fontSize}px Sans-Serif`
          ctx.textAlign = 'center'
          ctx.textBaseline = 'middle'
          ctx.fillStyle = node.color
          ctx.fillText(label, node.x, node.y)
        }}
        linkDirectionalArrowLength={6}
        linkDirectionalArrowRelPos={1}
        linkColor={() => '#cccccc'}
      />

      {/* Legend */}
      <div className="mt-4 flex gap-4">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-green-500" />
          <span className="text-sm">Mastered (≥80%)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-yellow-500" />
          <span className="text-sm">Learning (40-79%)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-red-500" />
          <span className="text-sm">Not Started (<40%)</span>
        </div>
      </div>
    </div>
  )
}
```

---

#### 2.3: Concept Detail View
**File:** `apps/web/src/components/analytics/ConceptDetail.tsx`

**Features:**
- Show concept details when clicked
- Display prerequisites
- Show related cards
- Display mastery progress bar
- Show next review date

```typescript
export function ConceptDetail({ concept, mastery }: Props) {
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-xl font-bold mb-2">{concept.name}</h3>
      <p className="text-gray-600 mb-4">{concept.description}</p>

      {/* Mastery Progress */}
      <div className="mb-4">
        <div className="flex justify-between mb-1">
          <span className="text-sm font-medium">Mastery</span>
          <span className="text-sm font-medium">{(mastery * 100).toFixed(0)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-green-500 h-2 rounded-full"
            style={{ width: `${mastery * 100}%` }}
          />
        </div>
      </div>

      {/* Prerequisites */}
      <div className="mb-4">
        <h4 className="font-semibold mb-2">Prerequisites</h4>
        <ul className="list-disc list-inside">
          {concept.prerequisites.map(prereq => (
            <li key={prereq.id} className="text-sm text-gray-700">
              {prereq.name}
            </li>
          ))}
        </ul>
      </div>

      {/* Cards */}
      <div>
        <h4 className="font-semibold mb-2">Cards ({concept.card_count})</h4>
        <div className="text-sm text-gray-600">
          Reviewed: {concept.cards_reviewed} / {concept.card_count}
        </div>
      </div>
    </div>
  )
}
```

---

## CHUNK 3: Dashboard Polish & Integration

### Goal
Enhance dashboard with activity timeline, insights, and responsive design

### Tasks

#### 3.1: Activity Timeline Component
**File:** `apps/web/src/components/dashboard/ActivityTimeline.tsx`

**Features:**
- Recent learning sessions
- Achievements unlocked
- Level ups
- Streak milestones
- Concept mastery

```typescript
export function ActivityTimeline({ activities }: Props) {
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold mb-4">📅 Recent Activity</h3>

      <div className="space-y-4">
        {activities.map(activity => (
          <ActivityItem key={activity.id} activity={activity} />
        ))}
      </div>
    </div>
  )
}

function ActivityItem({ activity }: { activity: Activity }) {
  const icon = getActivityIcon(activity.type)
  const color = getActivityColor(activity.type)

  return (
    <div className="flex items-start gap-3">
      <div className={`w-8 h-8 rounded-full ${color} flex items-center justify-center`}>
        {icon}
      </div>
      <div className="flex-1">
        <p className="font-medium">{activity.title}</p>
        <p className="text-sm text-gray-600">{activity.description}</p>
        <p className="text-xs text-gray-400 mt-1">
          {formatDistanceToNow(activity.timestamp)} ago
        </p>
      </div>
    </div>
  )
}
```

**Activity Types:**
- `session_completed` - Completed learning session
- `achievement_unlocked` - Unlocked achievement
- `level_up` - Leveled up
- `concept_mastered` - Mastered a concept
- `streak_milestone` - Reached streak milestone

---

#### 3.2: Insights & Recommendations
**File:** `apps/web/src/components/dashboard/InsightsPanel.tsx`

**Insights to Show:**
- Learning patterns (best time of day, most productive days)
- Strengths (concepts with high mastery)
- Weaknesses (concepts needing review)
- Recommendations (what to study next)
- Motivation (encouraging messages based on progress)

```typescript
export function InsightsPanel({ insights }: Props) {
  return (
    <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold mb-4">💡 Insights</h3>

      <div className="space-y-4">
        {/* Best Time */}
        <Insight
          icon="🌟"
          title="Peak Performance"
          description={`You learn best in the ${insights.best_time_of_day}`}
        />

        {/* Strengths */}
        <Insight
          icon="💪"
          title="Strengths"
          description={`You're excelling at ${insights.top_concepts.join(', ')}`}
        />

        {/* Areas for Improvement */}
        <Insight
          icon="📚"
          title="Focus Areas"
          description={`Consider reviewing ${insights.weak_concepts.join(', ')}`}
        />

        {/* Recommendation */}
        <Insight
          icon="🎯"
          title="Next Steps"
          description={insights.recommendation}
          actionButton={
            <button className="text-sm text-blue-600 hover:underline">
              Start Learning →
            </button>
          }
        />
      </div>
    </div>
  )
}
```

---

#### 3.3: Enhanced Dashboard Layout
**File:** `apps/web/src/app/(protected)/dashboard/page.tsx`

**Updated Layout:**
```
┌─────────────────────────────────────────────────┐
│ Welcome back, [Name]! 👋                        │
├─────────────────────────────────────────────────┤
│                                                 │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│ │ Level 3 │ │ 245 XP  │ │ 7 Days  │           │
│ │    ⬆️    │ │   📊    │ │   🔥    │           │
│ └─────────┘ └─────────┘ └─────────┘           │
│                                                 │
│ ┌─────────────────────────────────────┐        │
│ │ 💡 Insights                         │        │
│ │ ✨ You learn best in the morning    │        │
│ │ 💪 Strengths: Variables, Functions  │        │
│ │ 📚 Review: Recursion, Classes       │        │
│ └─────────────────────────────────────┘        │
│                                                 │
│ ┌─────────────────┐  ┌─────────────────┐      │
│ │ 📅 Activity     │  │ 📈 Progress     │      │
│ │                 │  │                 │      │
│ │ • Session       │  │ [Mini chart]    │      │
│ │ • Achievement   │  │                 │      │
│ │ • Level up      │  │                 │      │
│ └─────────────────┘  └─────────────────┘      │
│                                                 │
│ [Start Learning] [View Progress] [Knowledge Graph] │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

#### 3.4: Navigation Updates
**File:** `apps/web/src/components/layout/Sidebar.tsx`

**Add New Nav Items:**
- Dashboard (existing)
- Learn (existing)
- **Progress** ← NEW
- **Knowledge Graph** ← NEW
- Profile (existing)

---

#### 3.5: Responsive Design
**Updates to all components:**
- Mobile-friendly layouts (stack on small screens)
- Touch-friendly interactions
- Optimized chart sizes for mobile
- Hamburger menu for navigation

---

## 📊 Success Metrics

### Must Have (Minimum Viable)
- [ ] Progress page shows XP chart
- [ ] Success rate trend visible
- [ ] Concept mastery visualization
- [ ] Knowledge Graph renders correctly
- [ ] Activity timeline shows recent sessions
- [ ] Dashboard shows basic insights
- [ ] Responsive on mobile

### Nice to Have (Stretch Goals)
- [ ] Interactive graph (zoom, pan, click)
- [ ] Multiple chart types (bar, pie, radar)
- [ ] Export data (CSV, PDF)
- [ ] Customizable dashboards
- [ ] Advanced insights (ML-based)

---

## 🚧 Implementation Order

### Chunk 1: Progress Analytics (8-10 hours)
1. Create analytics API endpoints (2 hours)
2. Install chart libraries (30 min)
3. Create chart components (3 hours)
4. Build progress page (2 hours)
5. Test and polish (2 hours)

### Chunk 2: Knowledge Graph (6-8 hours)
1. Create knowledge graph API endpoint (2 hours)
2. Install graph visualization library (30 min)
3. Build graph component (3 hours)
4. Add concept detail view (2 hours)
5. Test and polish (2 hours)

### Chunk 3: Dashboard Polish (4-6 hours)
1. Activity timeline component (2 hours)
2. Insights panel (2 hours)
3. Dashboard layout updates (1 hour)
4. Navigation updates (30 min)
5. Responsive design (2 hours)
6. Final testing and polish (2 hours)

**Total: 18-24 hours (2-3 days)**

---

## 📦 Deliverables

By end of Phase 3B, we should have:

1. **Progress Analytics Page**
   - XP over time chart
   - Success rate trend chart
   - Concept mastery radar chart
   - Performance metrics (accuracy, study time, engagement)

2. **Knowledge Graph Visualization**
   - Interactive 2D force-directed graph
   - Node coloring by mastery level
   - Prerequisite relationships shown
   - Click to view concept details

3. **Enhanced Dashboard**
   - Activity timeline (recent sessions, achievements)
   - Insights panel (patterns, recommendations)
   - Quick stats (level, XP, streak)
   - Action buttons (start learning, view progress)

4. **Responsive Design**
   - Mobile-friendly layouts
   - Touch-optimized interactions
   - Adaptive chart sizes

5. **Documentation**
   - API documentation for analytics endpoints
   - Component usage guide
   - Testing documentation

---

## 🎯 Definition of Done

Phase 3B is **COMPLETE** when:

- [ ] Progress page accessible via navigation
- [ ] XP chart shows historical data
- [ ] Concept mastery chart displays all concepts
- [ ] Knowledge Graph renders all concepts and prerequisites
- [ ] Activity timeline shows last 10 activities
- [ ] Dashboard shows personalized insights
- [ ] All charts responsive on mobile
- [ ] No console errors
- [ ] All components documented

---

## 📁 Files to Create/Modify

### New Files (18+)
```
services/api-gateway/
  routes/analytics.py                    ← Analytics endpoints

apps/web/src/components/analytics/
  ProgressChart.tsx                      ← XP over time chart
  SuccessRateChart.tsx                   ← Success rate trend
  ConceptMasteryChart.tsx                ← Radar chart for concepts
  PerformanceMetrics.tsx                 ← Stat cards
  KnowledgeGraphView.tsx                 ← Interactive graph
  ConceptDetail.tsx                      ← Concept detail panel

apps/web/src/components/dashboard/
  ActivityTimeline.tsx                   ← Recent activity
  InsightsPanel.tsx                      ← Insights and recommendations
  QuickStats.tsx                         ← Level, XP, streak cards

apps/web/src/app/(protected)/
  progress/page.tsx                      ← Progress analytics page
  knowledge-graph/page.tsx               ← Knowledge Graph page

docs/
  CHUNK5_PROGRESS_ANALYTICS.md           ← Chunk 5 documentation
  CHUNK6_KNOWLEDGE_GRAPH.md              ← Chunk 6 documentation
  CHUNK7_DASHBOARD_POLISH.md             ← Chunk 7 documentation
  PHASE_3B_COMPLETE.md                   ← Phase summary
```

### Modified Files
```
apps/web/src/app/(protected)/dashboard/page.tsx  ← Enhanced layout
apps/web/src/components/layout/Sidebar.tsx       ← New nav items
apps/web/package.json                             ← New dependencies
```

---

**Ready to build analytics that make learning visible! 📊**
