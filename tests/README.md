# NerdLearn Testing Guide

## Overview

This directory contains all tests for the NerdLearn adaptive learning platform, organized by test type and scope. The structure supports end-to-end user journeys, cross-service integration tests, isolated unit tests, and performance benchmarks.

---

## Test Structure

```
tests/
├── e2e/                          # End-to-end user journey tests
│   ├── conftest.py               # E2E fixtures (auth, http clients)
│   ├── learning/                 # Learning flow tests
│   │   └── test_learning_flow.py # Complete learning session tests
│   ├── auth/                     # Authentication flow tests
│   └── gamification/             # Gamification flow tests
│
├── integration/                  # Cross-service integration tests
│   ├── conftest.py               # Integration fixtures
│   ├── services/                 # Service-to-service tests
│   │   ├── test_api_scheduler.py # API <-> Scheduler tests
│   │   └── test_orchestrator.py  # Orchestrator coordination tests
│   └── data/                     # Data persistence tests
│       └── test_data_persistence.py
│
├── unit/                         # Isolated unit tests
│   ├── conftest.py               # Unit test fixtures (mocks)
│   ├── swarm/                    # Swarm package tests
│   │   └── test_fuel_meter.py    # Fuel metering tests
│   └── packages/                 # Shared package tests
│
├── performance/                  # Performance & load tests
│   ├── conftest.py               # Performance fixtures (timers, thresholds)
│   ├── locustfiles/              # Locust load test scenarios
│   │   └── learning_flow.py      # Learning flow load test
│   └── benchmarks/               # Response time benchmarks
│       └── test_response_times.py
│
├── fixtures/                     # Shared test data factories
│   ├── __init__.py               # Factory exports
│   ├── users.py                  # User data factories
│   ├── cards.py                  # Card data factories
│   └── sessions.py               # Session data factories
│
├── docs/                         # Project documentation & research
│   ├── phases/                   # Phase implementation summaries
│   ├── planning/                 # Project planning documents
│   ├── operations/               # Deployment & operations guides
│   ├── tracks/                   # Conductor tracking system
│   └── research/                 # 40+ academic research PDFs
│       ├── adaptive-learning/
│       ├── algorithms/
│       ├── architecture/
│       ├── assessment/
│       ├── cognitive-science/
│       ├── gamification/
│       ├── knowledge-graphs/
│       └── ...
│
├── conftest.py                   # Root-level shared fixtures
├── pytest.ini                    # Pytest configuration
├── requirements.txt              # Test dependencies
└── README.md                     # This file
```

### Test Tiers

| Tier | Purpose | Speed | Services Required |
|------|---------|-------|-------------------|
| **e2e** | Full user journeys | Slow | All services |
| **integration** | Cross-service communication | Medium | Specific services |
| **unit** | Isolated components | Fast | None |
| **performance** | Load & response times | Variable | All services |

### Documentation & Research

All project documentation and research papers are organized in `docs/`:

| Folder | Contents |
|--------|----------|
| `docs/phases/` | Phase implementation summaries (Phase 2, 3, 4) |
| `docs/planning/` | Bonus features plan, verification report |
| `docs/operations/` | Deployment and production ops guides |
| `docs/tracks/` | Conductor tracking system (current work) |
| `docs/research/` | 40+ academic PDFs organized by topic |

See [docs/README.md](docs/README.md) for full documentation index.

---

## Prerequisites

### 1. Install Test Dependencies

```bash
# From project root
pip install -r tests/requirements.txt

# Or using specific Python version
python3.10 -m pip install -r tests/requirements.txt
```

### 2. Start All Services

```bash
# Ensure all services are running
./scripts/start-all-services.sh

# Verify services are healthy
curl http://localhost:8000/health  # API Gateway
curl http://localhost:8001/health  # Scheduler
curl http://localhost:8002/health  # Telemetry
curl http://localhost:8003/health  # Inference
curl http://localhost:8005/health  # Orchestrator
```

### 3. Seed Database

```bash
# Ensure database has demo data
cd packages/db
npx tsx prisma/seed.ts
```

---

## Running Tests

### Run All Tests

```bash
# From tests/ directory (uses pytest.ini)
cd tests
pytest

# Or from project root
pytest tests/
```

### Run Tests by Tier

```bash
# End-to-end tests (requires all services)
pytest tests/e2e/ -v

# Integration tests (requires specific services)
pytest tests/integration/ -v

# Unit tests (no services required)
pytest tests/unit/ -v

# Performance tests
pytest tests/performance/ -v
```

### Run Specific Test File

```bash
# E2E learning flow tests
pytest tests/e2e/learning/test_learning_flow.py

# Unit tests for fuel meter
pytest tests/unit/swarm/test_fuel_meter.py
```

### Run Specific Test

```bash
# Run single test by name
pytest tests/e2e/learning/test_learning_flow.py::test_start_learning_session

# Run tests matching pattern
pytest tests/ -k "session"

# Run tests matching multiple patterns
pytest tests/ -k "session or zpd"
```

### Run Tests by Marker

```bash
# Run only E2E tests
pytest tests/ -m e2e

# Run only integration tests
pytest tests/ -m integration

# Run only slow tests
pytest tests/ -m slow

# Skip slow tests
pytest tests/ -m "not slow"

# Run only database tests
pytest tests/ -m requires_db

# Run performance benchmarks
pytest tests/ -m benchmark
```

### Run with Coverage

```bash
# Generate coverage report
pytest tests/ --cov=../apps/api/app --cov-report=html

# Open coverage report
open htmlcov/index.html
```

---

## Test Suites

### 1. Service Health Checks

**File:** `e2e/learning/test_learning_flow.py::test_all_services_healthy`

Verifies all microservices are running and responding to health checks.

**Expected Result:** ✅ All services return 200 status

**Troubleshooting:**
- If fails: Check `./scripts/start-all-services.sh` ran successfully
- Check logs: `tail -f logs/*.log`

---

### 2. Authentication Tests

**Tests:**
- `test_user_registration` - Register new user
- `test_user_login` - Login with credentials
- `test_user_login` (from fixture) - Token-based auth

**Expected Results:**
- ✅ User created in database
- ✅ LearnerProfile initialized with defaults
- ✅ JWT token issued
- ✅ Token valid for API requests

**Common Issues:**
- Email already exists → Use unique timestamp in email
- Password too weak → Minimum 8 chars, includes number/symbol
- Database connection lost → Restart PostgreSQL

---

### 3. Learning Session Tests

**Tests:**
- `test_start_learning_session` - Initialize session
- `test_answer_card_good_rating` - Answer with "good"
- `test_complete_full_session` - Complete 10 cards

**Expected Results:**
- ✅ Session ID generated
- ✅ Cards loaded from Scheduler (FSRS)
- ✅ First card displayed
- ✅ XP earned on each answer
- ✅ Progress tracked
- ✅ All data persists in database

**Performance Targets:**
- Session start: < 500ms
- Card load: < 200ms
- Answer processing: < 300ms

**Common Issues:**
- No cards available → Run seed script
- Cards not from Scheduler → Check Scheduler service running
- Timeout → Increase `TIMEOUT` in test config

---

### 4. FSRS Scheduling Tests

**Test:** `test_fsrs_scheduling_intervals`

Verifies FSRS algorithm calculates correct review intervals.

**Expected Intervals:**
| Rating | Expected Interval |
|--------|------------------|
| Again (1) | Minutes to hours |
| Hard (2) | Hours to 1 day |
| Good (3) | 1-5 days |
| Easy (4) | 3-15 days |

**Note:** Exact intervals depend on current stability/difficulty parameters.

**Common Issues:**
- Intervals too short → Check FSRS parameters initialized
- Intervals too long → Check card difficulty setting
- Endpoint not found → Scheduler service not implementing `/review`

---

### 5. ZPD Adaptation Tests

**Tests:**
- `test_zpd_frustration_zone` - Detect low performance
- `test_zpd_comfort_zone` - Detect high performance

**Expected ZPD Zones:**
| Success Rate | ZPD Zone | Action |
|-------------|----------|--------|
| < 35% | Frustration | Provide scaffolding |
| 35-70% | Optimal | Maintain difficulty |
| > 70% | Comfort | Increase difficulty |

**Expected Results:**
- ✅ Frustration detected after 3-4 consecutive "again"
- ✅ Scaffolding provided (hint/worked example)
- ✅ Comfort detected after 4-5 consecutive "easy"
- ✅ Difficulty adjustment recommended

**Common Issues:**
- Zone not detected → Check Inference service running
- Wrong thresholds → Adjust in Orchestrator `assess_zpd_state()`

---

### 6. Gamification Tests

**Test:** `test_xp_calculation`

Verifies XP formula: `base_xp * rating_multiplier`

**Expected XP (difficulty = 5.0):**
| Rating | Base XP | Multiplier | XP Earned |
|--------|---------|------------|-----------|
| Again | 10 | 0.5 | 5 |
| Hard | 10 | 0.75 | 8 |
| Good | 10 | 1.0 | 10 |
| Easy | 10 | 1.25 | 13 |

**Common Issues:**
- XP not ordered correctly → Check `calculate_xp()` function
- XP too high/low → Verify difficulty values in database

---

### 7. Database Persistence Tests

**Test:** `test_session_data_persists`

Verifies reviews stored in database.

**Expected Database Records:**
- `ScheduledItem` - Updated with `nextReview`, `lastReview`
- `LearnerProfile` - `totalXP`, `fsrsStability`, `fsrsDifficulty` updated
- `CompetencyState` - Concept mastery tracking
- `Evidence` - Behavioral data (mouse, dwell time)

**Verification Queries:**
```sql
-- Check scheduled items
SELECT * FROM "ScheduledItem"
WHERE "learnerProfileId" = '<profile_id>'
ORDER BY "lastReview" DESC;

-- Check XP updated
SELECT "totalXP", level FROM "LearnerProfile"
WHERE "userId" = '<user_id>';

-- Check evidence records
SELECT COUNT(*) FROM "Evidence"
WHERE "learnerId" = '<profile_id>'
  AND "createdAt" > NOW() - INTERVAL '1 hour';
```

---

### 8. Error Handling Tests

**Tests:**
- `test_invalid_session_id` - Invalid session
- `test_invalid_rating` - Invalid rating value

**Expected Results:**
- ✅ Return 400/404/422 (not 500)
- ✅ Error message in response
- ✅ No server crash

---

### 9. Performance Tests

**Tests:**
- `test_session_start_performance` - < 1000ms
- `test_answer_processing_performance` - < 500ms

**Performance Targets:**
| Operation | Target | Critical Threshold |
|-----------|--------|-------------------|
| Session start | < 500ms | < 1000ms |
| Card load | < 200ms | < 500ms |
| Answer processing | < 300ms | < 500ms |
| WebSocket latency | < 50ms | < 100ms |

**Optimization Tips:**
- Use database connection pooling
- Cache concept/card metadata
- Optimize FSRS calculations
- Use async/await for service calls

---

## Test Results Interpretation

### All Tests Pass ✅

```
======================== 20 passed in 15.23s ========================
```

**Meaning:**
- All services working correctly
- Database operations successful
- FSRS scheduling functional
- ZPD adaptation working
- Ready for demo/production

---

### Some Tests Skipped ⏭️

```
=================== 15 passed, 5 skipped in 12.45s ===================
```

**Meaning:**
- Some services unavailable (e.g., Scheduler, Inference)
- Missing demo data
- Tests work with graceful degradation

**Action:** Review skip reasons:
```bash
pytest tests/integration/ -v -rs
```

---

### Tests Failed ❌

```
=================== 5 passed, 3 failed in 10.12s ===================
```

**Meaning:**
- Critical bug or service failure
- Database connection issue
- Service integration problem

**Action:**
1. Review error messages
2. Check service logs: `tail -f logs/*.log`
3. Verify database state: `npx prisma studio`
4. Run single failing test: `pytest tests/integration/ -k test_name -v -s`

---

## Common Test Failures

### 1. Connection Refused

```
httpx.ConnectError: [Errno 111] Connection refused
```

**Cause:** Service not running

**Fix:**
```bash
# Check which services are down
./scripts/start-all-services.sh

# Or start specific service
cd services/orchestrator
python main.py
```

---

### 2. Database Connection Error

```
psycopg2.OperationalError: could not connect to server
```

**Cause:** PostgreSQL not running

**Fix:**
```bash
# Start database
docker-compose up -d postgres

# Verify connection
psql -U nerdlearn -h localhost -d nerdlearn
```

---

### 3. No Cards Available

```
pytest.skip: No cards available for testing
```

**Cause:** Database not seeded

**Fix:**
```bash
cd packages/db
npx tsx prisma/seed.ts
```

---

### 4. Timeout Error

```
httpx.TimeoutException: Request timed out
```

**Cause:** Service responding slowly or hung

**Fix:**
- Check service logs for errors
- Increase timeout in test config
- Restart service

---

### 5. Authentication Failed

```
AssertionError: Login failed: Unauthorized
```

**Cause:** User not in database or wrong credentials

**Fix:**
- Verify user seeded: Check Prisma Studio
- Check password hash algorithm matches
- Re-run seed script

---

## Writing New Tests

### Test Template

```python
@pytest.mark.asyncio
async def test_my_feature(
    http_client: httpx.AsyncClient,
    test_user: Dict,
    auth_headers: Dict
):
    """Test description"""

    # Arrange
    # ... setup test data ...

    # Act
    response = await http_client.post(
        f"{ORCHESTRATOR_URL}/my/endpoint",
        json={"data": "value"},
        headers=auth_headers
    )

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "expected_key" in data

    print(f"✅ Test passed: {data}")
```

### Best Practices

1. **Use fixtures** for common setup (test_user, auth_headers)
2. **Clean up** test data after tests
3. **Use descriptive names** (test_what_when_then)
4. **Add markers** for organization (@pytest.mark.slow)
5. **Print results** for debugging
6. **Handle errors gracefully** (check status codes)
7. **Test edge cases** (empty data, invalid inputs)

---

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/integration-tests.yml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: nerdlearn_dev_password
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r tests/requirements.txt

      - name: Start services
        run: |
          ./scripts/start-all-services.sh

      - name: Run integration tests
        run: |
          pytest tests/integration/ -v

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## Debugging Tests

### Enable Verbose Logging

```bash
# Show all print statements
pytest tests/integration/ -v -s

# Show log messages
pytest tests/integration/ -v --log-cli-level=DEBUG
```

### Run Single Test with Debugger

```bash
# Drop into pdb on failure
pytest tests/integration/ --pdb

# Drop into pdb immediately
pytest tests/integration/ --trace
```

### Check Service Logs

```bash
# Watch all logs
tail -f logs/*.log

# Filter specific service
tail -f logs/orchestrator.log | grep ERROR
```

### Use pytest-watch for TDD

```bash
# Install pytest-watch
pip install pytest-watch

# Auto-run tests on file change
ptw tests/integration/
```

---

## Performance Benchmarking

### Load Testing with Locust

```bash
# Install locust
pip install locust

# Create locustfile.py (see examples/)

# Run load test
locust -f tests/load/locustfile.py --host http://localhost:8005
```

### Database Query Performance

```sql
-- Enable query timing
\timing

-- Check slow queries
SELECT * FROM "ScheduledItem"
WHERE "nextReview" <= NOW()
ORDER BY "nextReview" ASC
LIMIT 10;

-- Time: 45.123 ms
```

---

## Test Coverage Goals

| Component | Current Coverage | Target |
|-----------|-----------------|--------|
| Orchestrator | TBD | 60% |
| Scheduler | TBD | 70% |
| Inference | TBD | 50% |
| Telemetry | TBD | 60% |
| API Gateway | TBD | 60% |
| Overall | TBD | 60% |

---

## Next Steps

### Phase 1: Basic Integration (Current)
- [x] Service health checks
- [x] Authentication flow
- [x] Learning session flow
- [x] FSRS scheduling
- [x] ZPD adaptation
- [x] Gamification (XP, levels)

### Phase 2: Advanced Features
- [ ] Knowledge Graph traversal
- [ ] Scaffolding content loading
- [ ] Achievement unlocking
- [ ] Telemetry WebSocket tests
- [ ] Real-time engagement scoring

### Phase 3: Performance & Scale
- [ ] Load testing (100+ concurrent users)
- [ ] Database query optimization
- [ ] Service response time benchmarks
- [ ] WebSocket message throughput
- [ ] Memory leak detection

### Phase 4: Edge Cases
- [ ] Service failure scenarios
- [ ] Network latency simulation
- [ ] Database transaction rollbacks
- [ ] Concurrent write conflicts
- [ ] Data consistency checks

---

## Resources

- **Pytest Documentation:** https://docs.pytest.org/
- **httpx Documentation:** https://www.python-httpx.org/
- **Prisma Studio:** `npx prisma studio` (view database)
- **Neo4j Browser:** http://localhost:7474 (view Knowledge Graph)
- **Service Logs:** `./logs/` directory

---

## Support

If tests are failing:

1. **Check service logs**: `tail -f logs/*.log`
2. **Verify database**: `npx prisma studio`
3. **Review test output**: `pytest tests/integration/ -v -s`
4. **Check GitHub Issues**: Known test failures
5. **Ask for help**: Include error message and logs

---

**Happy Testing! 🧪**

*Remember: Tests are not just finding bugs - they're validating that NerdLearn delivers on its promise of adaptive, cognitive-focused learning.*
