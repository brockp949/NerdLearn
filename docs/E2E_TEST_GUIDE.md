# End-to-End Testing Guide

## Overview

This guide provides comprehensive instructions for testing the complete NerdLearn adaptive learning system across all services and features.

---

## Prerequisites

### Required Services Running

Ensure all services are running before starting tests:

```bash
# Start all services
./scripts/start-all-services.sh

# Verify all services are healthy
curl http://localhost:8000/health  # API Gateway
curl http://localhost:8001/health  # Scheduler
curl http://localhost:8002/health  # Telemetry
curl http://localhost:8003/health  # Inference
curl http://localhost:8004/health  # Content
curl http://localhost:8005/health  # Orchestrator
curl http://localhost:3000          # Frontend (should return HTML)
```

### Expected Response Times

| Service | Expected Health Check Response Time |
|---------|-------------------------------------|
| API Gateway | < 50ms |
| Scheduler | < 100ms |
| Telemetry | < 50ms |
| Inference | < 200ms (model loading) |
| Content | < 50ms |
| Orchestrator | < 100ms |
| Frontend | < 500ms (SSR) |

---

## Test Suite 1: User Registration & Authentication

### Test 1.1: New User Registration

**Objective:** Verify user can create account and profile is initialized

**Steps:**
1. Navigate to `http://localhost:3000`
2. Click "Register" or "Sign Up"
3. Fill in registration form:
   - Email: `testuser@example.com`
   - Username: `testuser`
   - Password: `Test123!`
4. Submit form
5. Verify redirect to dashboard

**Expected Results:**
- ✅ User created in database
- ✅ LearnerProfile created with default values:
  - `fsrsStability`: 2.5
  - `fsrsDifficulty`: 5.0
  - `currentZpdLower`: 0.35
  - `currentZpdUpper`: 0.70
  - `totalXP`: 0
  - `level`: 1
  - `streakDays`: 0
- ✅ Session token issued
- ✅ Redirect to dashboard page

**Verification:**
```sql
-- Check user exists
SELECT * FROM "User" WHERE email = 'testuser@example.com';

-- Check profile created
SELECT * FROM "LearnerProfile" WHERE "userId" = (
  SELECT id FROM "User" WHERE email = 'testuser@example.com'
);
```

---

### Test 1.2: User Login

**Objective:** Verify existing user can login

**Steps:**
1. Navigate to `http://localhost:3000/login`
2. Enter credentials:
   - Email: `demo@nerdlearn.com`
   - Password: `demo123`
3. Submit form

**Expected Results:**
- ✅ JWT token issued
- ✅ Session created
- ✅ Redirect to dashboard
- ✅ User profile data loaded

**Verification:**
```bash
# Check session token is valid
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/user/me
```

---

### Test 1.3: Authentication Persistence

**Objective:** Verify session persists across page reloads

**Steps:**
1. Login as demo user
2. Refresh page (F5)
3. Navigate to different pages

**Expected Results:**
- ✅ User remains logged in
- ✅ No redirect to login page
- ✅ Session token valid in localStorage/cookies

---

## Test Suite 2: Learning Session Flow

### Test 2.1: Start Learning Session

**Objective:** Verify learning session initializes with FSRS-scheduled cards

**Steps:**
1. Login as demo user
2. Navigate to Dashboard
3. Click "Start Learning" button
4. Observe loading state
5. Verify first card displays

**Expected Results:**
- ✅ Session created in Orchestrator
- ✅ Scheduler service called to get due cards
- ✅ Cards loaded from database (not demo data)
- ✅ First card displays with:
  - Content section
  - "Continue" button
  - Progress indicator (1/10)
  - XP display
  - Engagement meter (if telemetry connected)

**Verification:**
```bash
# Check session created
curl http://localhost:8005/session/<session_id>

# Check scheduler was called (check logs)
tail -f logs/scheduler.log
# Should see: "GET /due/<learner_profile_id>"
```

**Expected Card Structure:**
```json
{
  "card_id": "clx...",
  "concept_name": "Python Variables",
  "content": "**Variables** are containers for storing data...",
  "question": "What keyword is used to create a variable?",
  "correct_answer": "=",
  "difficulty": 3.5,
  "card_type": "BASIC"
}
```

---

### Test 2.2: Card Content Display

**Objective:** Verify card content renders correctly with markdown

**Steps:**
1. Start learning session
2. Read first card's content
3. Verify markdown formatting

**Expected Results:**
- ✅ Markdown rendered correctly:
  - **Bold** text
  - *Italic* text
  - `Code snippets`
  - ```python code blocks```
  - Links
  - Lists (bullet and numbered)
- ✅ Syntax highlighting for code blocks
- ✅ Responsive layout
- ✅ Readable typography

---

### Test 2.3: Answer Question - "Again" Rating

**Objective:** Verify FSRS scheduling for "again" rating (failure)

**Steps:**
1. View card content
2. Click "Continue" to reveal question
3. Think about answer
4. Click "Show Answer"
5. Click "Again" (didn't remember)

**Expected Results:**
- ✅ FSRS review endpoint called
- ✅ New scheduling calculated:
  - `interval_minutes`: ~10 minutes (very short)
  - `new_stability`: Lower than before
  - `new_difficulty`: Higher than before
- ✅ XP earned: ~5 XP (minimal for failure)
- ✅ ZPD zone: "frustration" (if multiple failures)
- ✅ Scaffolding offered (hint or worked example)
- ✅ Database updated:
  - `ScheduledItem.nextReview` set to ~10 minutes from now
  - `LearnerProfile.totalXP` increased
  - `CompetencyState` updated for concept

**Verification:**
```bash
# Check scheduler review call
tail -f logs/scheduler.log
# Should see: POST /review with rating=1

# Check database
SELECT * FROM "ScheduledItem"
WHERE "cardId" = '<card_id>' AND "learnerProfileId" = '<profile_id>'
ORDER BY "lastReview" DESC LIMIT 1;
```

**Expected Scheduling (FSRS for "Again"):**
```json
{
  "rating": 1,
  "interval_minutes": 10,
  "new_stability": 1.5,
  "new_difficulty": 6.2,
  "next_review": "2026-01-07T12:40:00Z"
}
```

---

### Test 2.4: Answer Question - "Good" Rating

**Objective:** Verify FSRS scheduling for "good" rating (success)

**Steps:**
1. View card content
2. Click "Continue"
3. Think about answer
4. Click "Show Answer"
5. Click "Good" (remembered with effort)

**Expected Results:**
- ✅ FSRS review endpoint called
- ✅ New scheduling calculated:
  - `interval_days`: ~2-4 days (moderate)
  - `new_stability`: Higher than before
  - `new_difficulty`: Slightly adjusted
- ✅ XP earned: ~15 XP (moderate reward)
- ✅ ZPD zone: "optimal" (within 35-70% success rate)
- ✅ No scaffolding needed
- ✅ Database updated

**Expected Scheduling (FSRS for "Good"):**
```json
{
  "rating": 3,
  "interval_days": 3.2,
  "new_stability": 4.5,
  "new_difficulty": 5.1,
  "next_review": "2026-01-10T12:30:00Z"
}
```

---

### Test 2.5: Answer Question - "Easy" Rating

**Objective:** Verify FSRS scheduling for "easy" rating (mastery)

**Steps:**
1. View card content
2. Click "Continue"
3. Click "Show Answer" immediately
4. Click "Easy" (instant recall)

**Expected Results:**
- ✅ FSRS review endpoint called
- ✅ New scheduling calculated:
  - `interval_days`: ~7-14 days (long)
  - `new_stability`: Significantly higher
  - `new_difficulty`: Lower (easier)
- ✅ XP earned: ~25 XP (high reward)
- ✅ ZPD zone: "comfort" (if consistently easy)
- ✅ Difficulty increase suggested
- ✅ Database updated

**Expected Scheduling (FSRS for "Easy"):**
```json
{
  "rating": 4,
  "interval_days": 10.5,
  "new_stability": 8.2,
  "new_difficulty": 4.3,
  "next_review": "2026-01-17T12:30:00Z"
}
```

---

### Test 2.6: Complete Full Session (10 Cards)

**Objective:** Verify complete learning session from start to finish

**Steps:**
1. Start session with 10 cards
2. Answer all 10 cards with mixed ratings:
   - Cards 1-2: "Good"
   - Cards 3-4: "Again"
   - Cards 5-7: "Good"
   - Card 8: "Easy"
   - Cards 9-10: "Hard"
3. Complete session

**Expected Results:**
- ✅ All 10 cards answered
- ✅ Total XP earned: ~120-150 XP
- ✅ Session completion screen shows:
  - Cards reviewed: 10
  - Success rate: 60-70%
  - Total XP: 120-150
  - Streak maintained/broken
  - Next session suggestion
- ✅ All 10 ScheduledItems updated in database
- ✅ LearnerProfile updated:
  - `totalXP` increased
  - `fsrsStability` adjusted
  - `fsrsDifficulty` adjusted
  - `currentZpdLower`/`Upper` potentially adjusted
  - `lastSessionDate` updated
- ✅ CompetencyState records created for each concept
- ✅ Evidence records created for behavioral data

**Verification:**
```sql
-- Check all scheduled items updated
SELECT COUNT(*) FROM "ScheduledItem"
WHERE "learnerProfileId" = '<profile_id>'
  AND "lastReview" > NOW() - INTERVAL '1 hour';
-- Expected: 10

-- Check XP updated
SELECT "totalXP", level FROM "LearnerProfile"
WHERE "userId" = '<user_id>';
-- Expected: totalXP ≈ 120-150, level = 1 or 2

-- Check evidence records
SELECT COUNT(*) FROM "Evidence"
WHERE "learnerId" = '<profile_id>'
  AND "createdAt" > NOW() - INTERVAL '1 hour';
-- Expected: 20+ (multiple events per card)
```

---

## Test Suite 3: Zone of Proximal Development (ZPD)

### Test 3.1: Frustration Zone Detection

**Objective:** Verify system detects frustration zone and provides scaffolding

**Steps:**
1. Start learning session
2. Intentionally answer "Again" for 3-4 consecutive cards
3. Observe ZPD indicator and scaffolding

**Expected Results:**
- ✅ ZPD zone changes to "frustration"
- ✅ Inference service called with recent performance data
- ✅ Success rate < 35%
- ✅ Scaffolding provided:
  - Hint or worked example
  - Difficulty decreased for next card
  - Encouragement message
- ✅ UI shows frustration indicator (red/orange)

**Verification:**
```bash
# Check inference service called
tail -f logs/inference.log
# Should see: POST /zpd/assess with recent_performance=[1,1,1,1]
```

**Expected Inference Response:**
```json
{
  "zone": "frustration",
  "success_rate": 0.25,
  "scaffolding": {
    "type": "worked_example",
    "content": "Let's break this down step by step...",
    "difficulty_adjustment": -1.0
  },
  "message": "You're in the frustration zone. Let's make this easier.",
  "recommendation": "decrease_difficulty"
}
```

---

### Test 3.2: Optimal Zone Maintenance

**Objective:** Verify system maintains optimal challenge (35-70% success)

**Steps:**
1. Start learning session
2. Answer with mixed "Again", "Hard", "Good" ratings
3. Maintain success rate around 50-60%

**Expected Results:**
- ✅ ZPD zone remains "optimal"
- ✅ Success rate: 35-70%
- ✅ No scaffolding needed
- ✅ Difficulty maintained
- ✅ UI shows optimal indicator (green/blue)
- ✅ Positive feedback messages

**Expected Inference Response:**
```json
{
  "zone": "optimal",
  "success_rate": 0.58,
  "scaffolding": null,
  "message": "Great! You're being challenged at the right level.",
  "recommendation": "maintain"
}
```

---

### Test 3.3: Comfort Zone Detection

**Objective:** Verify system detects comfort zone and increases difficulty

**Steps:**
1. Start learning session
2. Answer "Easy" for 4-5 consecutive cards
3. Observe ZPD indicator and difficulty adjustment

**Expected Results:**
- ✅ ZPD zone changes to "comfort"
- ✅ Success rate > 70%
- ✅ Difficulty increased for next cards
- ✅ UI shows comfort indicator (blue)
- ✅ Message: "You're mastering this! Let's increase the challenge."

**Expected Inference Response:**
```json
{
  "zone": "comfort",
  "success_rate": 0.85,
  "scaffolding": null,
  "message": "You're mastering this! Let's increase the challenge.",
  "recommendation": "increase_difficulty"
}
```

---

## Test Suite 4: Telemetry & Engagement Tracking

### Test 4.1: WebSocket Connection

**Objective:** Verify telemetry WebSocket connects successfully

**Steps:**
1. Start learning session
2. Open browser DevTools → Network → WS tab
3. Verify WebSocket connection

**Expected Results:**
- ✅ WebSocket connection to `ws://localhost:8002/ws`
- ✅ Connection status: "Connected" (green indicator in UI)
- ✅ Handshake message sent:
  ```json
  {
    "type": "init",
    "session_id": "session_...",
    "learner_id": "user_...",
    "timestamp": 1704636000000
  }
  ```
- ✅ Server acknowledgment received

---

### Test 4.2: Mouse Tracking

**Objective:** Verify mouse movements tracked and sent to telemetry service

**Steps:**
1. Start learning session
2. Move mouse around card content
3. Open DevTools → Network → WS → Messages
4. Observe mouse event batches

**Expected Results:**
- ✅ Mouse events sent every ~50ms
- ✅ Batched events (10-20 per message):
  ```json
  {
    "type": "mouse_events",
    "session_id": "session_...",
    "events": [
      {"x": 450, "y": 320, "timestamp": 1704636001000},
      {"x": 451, "y": 321, "timestamp": 1704636001050},
      ...
    ],
    "count": 15
  }
  ```
- ✅ Throttling working (not overwhelming network)
- ✅ No performance degradation

**Verification:**
```bash
# Check telemetry service processing
tail -f logs/telemetry.log
# Should see: "Received mouse_events batch: 15 events"
```

---

### Test 4.3: Dwell Time Tracking

**Objective:** Verify dwell time measured accurately

**Steps:**
1. Start learning session
2. Read content for exactly 30 seconds (use timer)
3. Click "Continue"
4. Check dwell time recorded

**Expected Results:**
- ✅ Dwell time measured: ~30,000ms (±500ms)
- ✅ Dwell time sent to telemetry:
  ```json
  {
    "type": "dwell_time",
    "session_id": "session_...",
    "card_id": "card_...",
    "dwell_time_ms": 30250,
    "timestamp": 1704636030250
  }
  ```
- ✅ Dwell time stored in Evidence table

**Verification:**
```sql
SELECT "observableData"->>'dwell_time_ms' as dwell_time
FROM "Evidence"
WHERE "cardId" = '<card_id>'
  AND "evidenceType" = 'ENGAGEMENT'
ORDER BY "createdAt" DESC LIMIT 1;
-- Expected: ~30000
```

---

### Test 4.4: Engagement Score Display

**Objective:** Verify engagement score updates in real-time

**Steps:**
1. Start learning session
2. Observe engagement meter in sidebar
3. Move mouse actively (high engagement)
4. Stop moving mouse (low engagement)
5. Watch engagement score change

**Expected Results:**
- ✅ Engagement meter visible in sidebar
- ✅ Initial engagement: ~50% (neutral)
- ✅ Active mouse movement:
  - Engagement increases to 70-85%
  - Meter turns green
  - Cognitive load: "low" or "medium"
  - Attention level: ●●● (high)
- ✅ No mouse movement for 10s:
  - Engagement decreases to 20-35%
  - Meter turns red/yellow
  - Cognitive load: "low"
  - Attention level: ●○○ (low)
- ✅ Real-time updates (no lag)

**Expected WebSocket Message:**
```json
{
  "type": "engagement_score",
  "score": 0.72,
  "cognitive_load": "medium",
  "attention_level": "high",
  "timestamp": "2026-01-07T12:30:05Z"
}
```

---

### Test 4.5: Hesitation Detection

**Objective:** Verify hesitation tracking when user pauses

**Steps:**
1. View question
2. Click "Show Answer"
3. Pause for 5 seconds
4. Hover over rating buttons without clicking
5. Pause again
6. Click "Good"

**Expected Results:**
- ✅ Hesitation count tracked: 2
- ✅ Hesitation data sent with answer:
  ```json
  {
    "type": "interaction",
    "interaction_type": "answer_submitted",
    "data": {
      "rating": "good",
      "hesitation_count": 2,
      "dwell_time_ms": 15000
    }
  }
  ```
- ✅ Higher hesitation → Lower confidence inference

---

## Test Suite 5: Gamification & Progression

### Test 5.1: XP Calculation

**Objective:** Verify XP earned based on rating and difficulty

**Steps:**
1. Answer card (difficulty: 5.0) with different ratings
2. Observe XP earned

**Expected XP Formula:**
```
base_xp = difficulty * 2
rating_multiplier = {again: 0.5, hard: 0.75, good: 1.0, easy: 1.25}
xp_earned = base_xp * rating_multiplier
```

**Expected Results:**
| Rating | Difficulty | Base XP | Multiplier | XP Earned |
|--------|-----------|---------|-----------|-----------|
| Again  | 5.0       | 10      | 0.5       | 5         |
| Hard   | 5.0       | 10      | 0.75      | 8         |
| Good   | 5.0       | 10      | 1.0       | 10        |
| Easy   | 5.0       | 10      | 1.25      | 13        |

---

### Test 5.2: Level Up

**Objective:** Verify level progression when XP threshold reached

**Steps:**
1. Complete enough cards to reach 100 XP (Level 2 threshold)
2. Observe level up notification

**Expected Results:**
- ✅ Level up notification appears
- ✅ Level displayed: 2
- ✅ XP formula for next level: `level * 100`
- ✅ Progress bar updates
- ✅ Confetti or celebration animation
- ✅ Database updated: `LearnerProfile.level = 2`

**XP Thresholds:**
- Level 1: 0 XP
- Level 2: 100 XP
- Level 3: 200 XP
- Level 4: 300 XP
- Level N: (N-1) * 100 XP

---

### Test 5.3: Streak Tracking

**Objective:** Verify daily streak maintained

**Steps:**
1. Complete session today
2. Check streak: 1 day
3. Wait until next day (or simulate)
4. Complete another session
5. Check streak: 2 days

**Expected Results:**
- ✅ Streak increments on consecutive days
- ✅ Streak resets if day skipped
- ✅ Streak displayed in dashboard
- ✅ Database updated: `LearnerProfile.streakDays`

---

### Test 5.4: Achievement Unlock

**Objective:** Verify achievements unlock at milestones

**Steps:**
1. Complete specific achievement conditions
2. Observe achievement notification

**Expected Achievements:**
- "First Steps" - Complete first card (1 XP)
- "Getting Started" - Complete 10 cards
- "Dedicated Learner" - 7-day streak
- "Quick Learner" - 50 cards rated "Easy"
- "Persistent" - 100 total cards reviewed
- "Python Novice" - Master 5 Python concepts

**Expected Results:**
- ✅ Achievement notification appears
- ✅ Achievement badge displayed
- ✅ Achievement stored in database
- ✅ Achievement list accessible in profile

---

## Test Suite 6: Database Persistence

### Test 6.1: Session Data Persistence

**Objective:** Verify session data persists correctly

**Steps:**
1. Start session
2. Answer 3 cards
3. Close browser (simulate crash)
4. Reopen and login
5. Check if progress saved

**Expected Results:**
- ✅ All 3 reviews saved
- ✅ XP persisted
- ✅ Scheduling updated
- ✅ Can start new session (doesn't resume old one)

---

### Test 6.2: Scheduled Items Due Query

**Objective:** Verify correct cards scheduled for review

**Steps:**
1. Review card and rate "Good" (due in 3 days)
2. Wait 3 days (or simulate with database update)
3. Start new session
4. Verify card appears in due list

**Expected SQL Query:**
```sql
SELECT * FROM "ScheduledItem"
WHERE "learnerProfileId" = '<profile_id>'
  AND "nextReview" <= NOW()
ORDER BY "nextReview" ASC
LIMIT 10;
```

**Expected Results:**
- ✅ Card appears when due
- ✅ Card not appear before due date
- ✅ Earliest due cards prioritized

---

### Test 6.3: Evidence Storage

**Objective:** Verify behavioral evidence stored for analysis

**Steps:**
1. Complete learning session
2. Query Evidence table

**Expected Evidence Types:**
```sql
SELECT "evidenceType", COUNT(*)
FROM "Evidence"
WHERE "learnerId" = '<profile_id>'
GROUP BY "evidenceType";
```

**Expected Results:**
| Evidence Type | Count (per card) |
|--------------|------------------|
| ENGAGEMENT   | 1                |
| PERFORMANCE  | 1                |
| BEHAVIORAL   | 5-10 (mouse)     |

---

## Test Suite 7: Error Handling & Edge Cases

### Test 7.1: Service Unavailable - Scheduler Down

**Objective:** Verify graceful degradation when Scheduler unavailable

**Steps:**
1. Stop Scheduler service: `pkill -f scheduler`
2. Start learning session

**Expected Results:**
- ✅ Session still starts (fallback to database query)
- ✅ Cards loaded from database (all cards or random sample)
- ✅ Warning message: "Scheduling service unavailable, using fallback"
- ✅ Reviews still processed (stored for later sync)
- ✅ No crash or error page

---

### Test 7.2: Service Unavailable - Telemetry Down

**Objective:** Verify learning works without telemetry

**Steps:**
1. Stop Telemetry service: `pkill -f telemetry`
2. Start learning session

**Expected Results:**
- ✅ Learning session works normally
- ✅ Engagement meter shows "Offline" status
- ✅ No telemetry data sent
- ✅ No browser console errors
- ✅ No performance degradation

---

### Test 7.3: Empty Due List

**Objective:** Verify behavior when no cards due

**Steps:**
1. Complete all cards
2. Start new session immediately (no cards due)

**Expected Results:**
- ✅ Message: "No cards due for review. Check back later!"
- ✅ Next review time displayed
- ✅ Option to browse content library
- ✅ Option to practice random cards

---

### Test 7.4: Database Connection Lost

**Objective:** Verify error handling for database failures

**Steps:**
1. Stop PostgreSQL: `docker-compose stop postgres`
2. Attempt to start session

**Expected Results:**
- ✅ Error page: "Database connection lost"
- ✅ Retry button
- ✅ No crash
- ✅ Graceful error message

---

## Test Suite 8: Performance & Load

### Test 8.1: Session Load Time

**Objective:** Verify session starts quickly

**Expected Results:**
- ✅ Session start: < 500ms
- ✅ Card load: < 200ms
- ✅ Answer processing: < 300ms
- ✅ No blocking UI operations

---

### Test 8.2: WebSocket Message Throughput

**Objective:** Verify telemetry handles high message volume

**Steps:**
1. Generate rapid mouse movements
2. Observe network tab

**Expected Results:**
- ✅ Messages throttled to 50ms
- ✅ No message queue buildup
- ✅ No lag in UI
- ✅ Bandwidth: < 10 KB/s

---

### Test 8.3: Concurrent Users

**Objective:** Verify system handles multiple users

**Steps:**
1. Open 5 browser windows
2. Login as different users
3. All start sessions simultaneously

**Expected Results:**
- ✅ All sessions work correctly
- ✅ No data mixing between users
- ✅ Database handles concurrent writes
- ✅ Services responsive

---

## Test Suite 9: Knowledge Graph

### Test 9.1: Prerequisite Enforcement

**Objective:** Verify prerequisites respected in card selection

**Steps:**
1. Start as new user
2. Verify only beginner cards (no prerequisites) appear
3. Master beginner concepts
4. Verify advanced cards now available

**Expected Results:**
- ✅ Beginner cards first
- ✅ Prerequisites checked via Neo4j query
- ✅ Advanced cards locked until prerequisites met

**Expected Neo4j Query:**
```cypher
MATCH (c:Concept {id: $conceptId})
MATCH (c)<-[:PREREQUISITE]-(prereq:Concept)
RETURN prereq.id, prereq.name
```

---

### Test 9.2: Concept Mastery

**Objective:** Verify concept marked as mastered

**Steps:**
1. Complete all cards for "Python Variables" with "Easy"
2. Check CompetencyState

**Expected Results:**
- ✅ Competency level: "MASTERED"
- ✅ Mastery threshold: 80%+ success rate
- ✅ Concept badge unlocked

**Expected SQL:**
```sql
SELECT "competencyLevel", "currentScore"
FROM "CompetencyState"
WHERE "learnerProfileId" = '<profile_id>'
  AND "conceptId" = '<concept_id>'
ORDER BY "lastUpdated" DESC LIMIT 1;

-- Expected: competencyLevel = 'MASTERED', currentScore > 0.8
```

---

## Success Criteria Summary

### Critical (Must Pass)

- [ ] User registration and login works
- [ ] Learning session starts with database cards
- [ ] FSRS scheduling calculates intervals correctly
- [ ] XP updates in database
- [ ] Cards can be answered with all 4 ratings
- [ ] Session completes successfully
- [ ] Data persists across sessions

### Important (Should Pass)

- [ ] ZPD zones detected correctly (frustration/optimal/comfort)
- [ ] Scaffolding provided in frustration zone
- [ ] Telemetry WebSocket connects
- [ ] Mouse tracking works
- [ ] Engagement score updates
- [ ] Level up triggers at thresholds
- [ ] Streak tracking works

### Nice to Have (May Pass)

- [ ] All services start without errors
- [ ] Performance targets met
- [ ] Graceful degradation for service failures
- [ ] Achievements unlock
- [ ] Knowledge Graph enforces prerequisites

---

## Test Results Template

```markdown
## E2E Test Results - [Date]

**Tester:** [Name]
**Environment:** [Local/Staging/Production]
**Services Running:** [List]

### Test Suite 1: Authentication
- [ ] Test 1.1: Registration - PASS/FAIL - [Notes]
- [ ] Test 1.2: Login - PASS/FAIL - [Notes]
- [ ] Test 1.3: Persistence - PASS/FAIL - [Notes]

### Test Suite 2: Learning Session
- [ ] Test 2.1: Start Session - PASS/FAIL - [Notes]
- [ ] Test 2.2: Card Display - PASS/FAIL - [Notes]
- [ ] Test 2.3: Answer "Again" - PASS/FAIL - [Notes]
- [ ] Test 2.4: Answer "Good" - PASS/FAIL - [Notes]
- [ ] Test 2.5: Answer "Easy" - PASS/FAIL - [Notes]
- [ ] Test 2.6: Complete Session - PASS/FAIL - [Notes]

[Continue for all test suites...]

### Issues Found
1. [Issue description] - Priority: High/Medium/Low
2. [Issue description] - Priority: High/Medium/Low

### Performance Metrics
- Session start time: [X]ms
- Average card load: [X]ms
- Answer processing: [X]ms

### Conclusion
[Overall assessment of system readiness]
```

---

## Debugging Tips

### Check Service Logs

```bash
# View all logs in real-time
tail -f logs/*.log

# Check specific service
tail -f logs/orchestrator.log
tail -f logs/scheduler.log
tail -f logs/telemetry.log
```

### Check Database State

```bash
# Open Prisma Studio
cd packages/db
npx prisma studio

# Or use psql
psql -U nerdlearn -d nerdlearn -h localhost
```

### Check Neo4j Graph

```
# Open Neo4j Browser
http://localhost:7474

# Query all concepts
MATCH (c:Concept) RETURN c

# Query prerequisites
MATCH (c:Concept)-[r:PREREQUISITE]->(p:Concept)
RETURN c.name, p.name, r.weight
```

### Monitor WebSocket

```javascript
// In browser console
const ws = new WebSocket('ws://localhost:8002/ws')
ws.onmessage = (event) => console.log('WS:', event.data)
ws.send(JSON.stringify({type: 'init', session_id: 'test'}))
```

---

## Next Steps After Testing

1. **Document all failures** in GitHub Issues
2. **Fix critical bugs** before demo
3. **Optimize slow operations**
4. **Add error handling** for edge cases
5. **Write automated tests** for failures found
6. **Update documentation** with findings
7. **Plan next iteration** improvements

---

**Testing is not just finding bugs - it's validating that NerdLearn delivers on its promise of adaptive, cognitive-focused learning.**
