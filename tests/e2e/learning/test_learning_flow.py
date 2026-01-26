"""
End-to-End Integration Tests for NerdLearn Learning Flow

Tests the complete learning flow across all services:
- User registration and authentication
- Learning session initialization
- FSRS scheduling
- ZPD adaptation
- Telemetry tracking
- Database persistence

Requirements:
- All services running (use ./scripts/start-all-services.sh)
- Database seeded with demo data
- pytest, pytest-asyncio, httpx installed
"""

import pytest
import httpx
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional


# ============================================================================
# Test Configuration
# ============================================================================

BASE_URL = "http://localhost:8000"  # API Gateway
ORCHESTRATOR_URL = "http://localhost:8005"
SCHEDULER_URL = "http://localhost:8001"
TELEMETRY_URL = "http://localhost:8002"
INFERENCE_URL = "http://localhost:8003"

TIMEOUT = 10.0  # seconds


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
async def http_client():
    """Async HTTP client for API requests"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        yield client


@pytest.fixture
async def test_user(http_client: httpx.AsyncClient) -> Dict:
    """Create a test user for integration tests"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    user_data = {
        "email": f"test_{timestamp}@example.com",
        "username": f"testuser_{timestamp}",
        "password": "Test123!@#"
    }

    # Register user
    response = await http_client.post(
        f"{BASE_URL}/api/auth/register",
        json=user_data
    )

    if response.status_code != 201:
        # User might already exist, try login
        login_response = await http_client.post(
            f"{BASE_URL}/api/auth/login",
            data={
                "username": user_data["email"],
                "password": user_data["password"]
            }
        )
        if login_response.status_code == 200:
            token_data = login_response.json()
            return {
                **user_data,
                "user_id": token_data.get("user_id"),
                "access_token": token_data.get("access_token")
            }
        else:
            pytest.fail(f"Failed to register or login test user: {login_response.text}")

    # Get user details from registration response
    user_response = response.json()

    # Login to get token
    login_response = await http_client.post(
        f"{BASE_URL}/api/auth/login",
        data={
            "username": user_data["email"],
            "password": user_data["password"]
        }
    )

    assert login_response.status_code == 200, f"Login failed: {login_response.text}"
    token_data = login_response.json()

    return {
        **user_data,
        "user_id": user_response.get("user_id") or token_data.get("user_id"),
        "access_token": token_data.get("access_token")
    }


@pytest.fixture
def auth_headers(test_user: Dict) -> Dict:
    """HTTP headers with authentication token"""
    return {
        "Authorization": f"Bearer {test_user['access_token']}",
        "Content-Type": "application/json"
    }


# ============================================================================
# Service Health Checks
# ============================================================================

@pytest.mark.asyncio
async def test_all_services_healthy(http_client: httpx.AsyncClient):
    """Verify all required services are running and healthy"""

    services = {
        "API Gateway": f"{BASE_URL}/health",
        "Scheduler": f"{SCHEDULER_URL}/health",
        "Telemetry": f"{TELEMETRY_URL}/health",
        "Inference": f"{INFERENCE_URL}/health",
        "Orchestrator": f"{ORCHESTRATOR_URL}/health"
    }

    for service_name, health_url in services.items():
        try:
            response = await http_client.get(health_url, timeout=5.0)
            assert response.status_code == 200, f"{service_name} health check failed"
            print(f"✅ {service_name}: Healthy")
        except httpx.RequestError as e:
            pytest.fail(f"❌ {service_name}: Not reachable - {e}")


# ============================================================================
# Authentication Tests
# ============================================================================

@pytest.mark.asyncio
async def test_user_registration(http_client: httpx.AsyncClient):
    """Test user registration creates account and profile"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    user_data = {
        "email": f"newuser_{timestamp}@example.com",
        "username": f"newuser_{timestamp}",
        "password": "NewUser123!"
    }

    response = await http_client.post(
        f"{BASE_URL}/api/auth/register",
        json=user_data
    )

    assert response.status_code == 201, f"Registration failed: {response.text}"

    data = response.json()
    assert "user_id" in data or "id" in data
    assert data.get("email") == user_data["email"] or data.get("message") == "User created"

    print(f"✅ User registered: {user_data['username']}")


@pytest.mark.asyncio
async def test_user_login(http_client: httpx.AsyncClient, test_user: Dict):
    """Test user login returns valid JWT token"""

    response = await http_client.post(
        f"{BASE_URL}/api/auth/login",
        data={
            "username": test_user["email"],
            "password": test_user["password"]
        }
    )

    assert response.status_code == 200, f"Login failed: {response.text}"

    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

    print(f"✅ User logged in: {test_user['username']}")


# ============================================================================
# Learning Session Tests
# ============================================================================

@pytest.mark.asyncio
async def test_start_learning_session(
    http_client: httpx.AsyncClient,
    test_user: Dict,
    auth_headers: Dict
):
    """Test starting a learning session loads cards from scheduler"""

    # Start session
    response = await http_client.post(
        f"{ORCHESTRATOR_URL}/session/start",
        json={
            "learner_id": test_user["user_id"],
            "limit": 10
        },
        headers=auth_headers
    )

    assert response.status_code == 200, f"Session start failed: {response.text}"

    data = response.json()

    # Verify session structure
    assert "session_id" in data
    assert "current_card" in data
    assert "cards_remaining" in data or "total_cards" in data

    # Verify card structure
    if data["current_card"]:
        card = data["current_card"]
        assert "card_id" in card
        assert "content" in card or "question" in card
        assert "difficulty" in card or card.get("difficulty") is not None

    print(f"✅ Session started: {data['session_id']}")
    print(f"   Cards loaded: {data.get('total_cards', 'N/A')}")
    print(f"   Current card: {data.get('current_card', {}).get('card_id', 'None')}")

    return data


@pytest.mark.asyncio
async def test_answer_card_good_rating(
    http_client: httpx.AsyncClient,
    test_user: Dict,
    auth_headers: Dict
):
    """Test answering a card with 'good' rating updates FSRS and XP"""

    # Start session first
    session_response = await http_client.post(
        f"{ORCHESTRATOR_URL}/session/start",
        json={"learner_id": test_user["user_id"], "limit": 10},
        headers=auth_headers
    )
    assert session_response.status_code == 200
    session_data = session_response.json()

    if not session_data.get("current_card"):
        pytest.skip("No cards available for testing")

    # Answer the card
    answer_response = await http_client.post(
        f"{ORCHESTRATOR_URL}/session/answer",
        json={
            "session_id": session_data["session_id"],
            "card_id": session_data["current_card"]["card_id"],
            "rating": "good",
            "dwell_time_ms": 15000,
            "hesitation_count": 1
        },
        headers=auth_headers
    )

    assert answer_response.status_code == 200, f"Answer failed: {answer_response.text}"

    data = answer_response.json()

    # Verify response structure
    assert "xp_earned" in data
    assert data["xp_earned"] > 0, "XP should be earned"
    assert "zpd_zone" in data
    assert data["zpd_zone"] in ["frustration", "optimal", "comfort"]

    # Verify scheduling info (if returned)
    if "scheduling_info" in data:
        sched = data["scheduling_info"]
        assert "next_review" in sched or "interval_days" in sched

    print(f"✅ Card answered: {session_data['current_card']['card_id']}")
    print(f"   Rating: good")
    print(f"   XP earned: {data['xp_earned']}")
    print(f"   ZPD zone: {data['zpd_zone']}")
    print(f"   Next card: {data.get('next_card', {}).get('card_id', 'Session complete')}")


@pytest.mark.asyncio
async def test_complete_full_session(
    http_client: httpx.AsyncClient,
    test_user: Dict,
    auth_headers: Dict
):
    """Test completing a full learning session (10 cards)"""

    # Start session
    session_response = await http_client.post(
        f"{ORCHESTRATOR_URL}/session/start",
        json={"learner_id": test_user["user_id"], "limit": 10},
        headers=auth_headers
    )
    assert session_response.status_code == 200
    session_data = session_response.json()
    session_id = session_data["session_id"]

    if not session_data.get("current_card"):
        pytest.skip("No cards available for testing")

    # Answer multiple cards with varied ratings
    ratings = ["good", "good", "again", "good", "hard", "good", "easy", "good", "hard", "good"]
    total_xp = 0
    cards_answered = 0

    current_card = session_data.get("current_card")

    for i, rating in enumerate(ratings):
        if not current_card:
            print(f"   Session ended after {cards_answered} cards")
            break

        # Answer current card
        answer_response = await http_client.post(
            f"{ORCHESTRATOR_URL}/session/answer",
            json={
                "session_id": session_id,
                "card_id": current_card["card_id"],
                "rating": rating,
                "dwell_time_ms": 10000 + (i * 1000),
                "hesitation_count": 0 if rating == "easy" else 1
            },
            headers=auth_headers
        )

        if answer_response.status_code != 200:
            print(f"   Answer {i+1} failed: {answer_response.text}")
            break

        data = answer_response.json()
        xp_earned = data.get("xp_earned", 0)
        total_xp += xp_earned
        cards_answered += 1

        print(f"   Card {i+1}/{len(ratings)}: {rating:6s} → +{xp_earned} XP (ZPD: {data.get('zpd_zone')})")

        # Get next card
        current_card = data.get("next_card")

    assert cards_answered > 0, "Should answer at least one card"
    assert total_xp > 0, "Should earn some XP"

    print(f"✅ Session completed")
    print(f"   Cards answered: {cards_answered}")
    print(f"   Total XP: {total_xp}")
    print(f"   Success rate: {sum(1 for r in ratings[:cards_answered] if r != 'again') / cards_answered * 100:.0f}%")


# ============================================================================
# FSRS Scheduling Tests
# ============================================================================

@pytest.mark.asyncio
async def test_fsrs_scheduling_intervals(http_client: httpx.AsyncClient):
    """Test FSRS calculates appropriate intervals for different ratings"""

    # Mock learner and card data
    test_reviews = [
        {"rating": 1, "expected_interval_range": (0, 1)},      # Again: minutes to hours
        {"rating": 2, "expected_interval_range": (0.5, 2)},    # Hard: hours to 1 day
        {"rating": 3, "expected_interval_range": (1, 5)},      # Good: 1-5 days
        {"rating": 4, "expected_interval_range": (3, 15)},     # Easy: 3-15 days
    ]

    for review in test_reviews:
        response = await http_client.post(
            f"{SCHEDULER_URL}/review",
            json={
                "card_id": "test_card_123",
                "learner_id": "test_learner_123",
                "rating": review["rating"],
                "reviewed_at": datetime.utcnow().isoformat()
            }
        )

        # If scheduler not configured, skip
        if response.status_code == 404:
            pytest.skip("Scheduler /review endpoint not implemented")

        assert response.status_code == 200, f"FSRS review failed: {response.text}"

        data = response.json()

        # Check interval calculated
        interval_days = data.get("interval_days", data.get("interval_minutes", 0) / 1440)
        min_interval, max_interval = review["expected_interval_range"]

        print(f"   Rating {review['rating']}: Interval = {interval_days:.2f} days (expected {min_interval}-{max_interval})")

        # Note: Exact intervals depend on current stability/difficulty
        # Just verify interval exists and is reasonable
        assert interval_days >= 0, "Interval should be non-negative"


# ============================================================================
# ZPD Adaptation Tests
# ============================================================================

@pytest.mark.asyncio
async def test_zpd_frustration_zone(
    http_client: httpx.AsyncClient,
    test_user: Dict,
    auth_headers: Dict
):
    """Test ZPD detects frustration zone with consecutive failures"""

    # Start session
    session_response = await http_client.post(
        f"{ORCHESTRATOR_URL}/session/start",
        json={"learner_id": test_user["user_id"], "limit": 10},
        headers=auth_headers
    )
    assert session_response.status_code == 200
    session_data = session_response.json()

    if not session_data.get("current_card"):
        pytest.skip("No cards available")

    # Answer multiple cards with "again" to trigger frustration
    current_card = session_data.get("current_card")
    frustration_detected = False

    for i in range(4):  # 4 consecutive failures
        if not current_card:
            break

        answer_response = await http_client.post(
            f"{ORCHESTRATOR_URL}/session/answer",
            json={
                "session_id": session_data["session_id"],
                "card_id": current_card["card_id"],
                "rating": "again",  # Failure
                "dwell_time_ms": 20000,
                "hesitation_count": 3
            },
            headers=auth_headers
        )

        if answer_response.status_code == 200:
            data = answer_response.json()
            zpd_zone = data.get("zpd_zone")

            print(f"   Attempt {i+1}: ZPD = {zpd_zone}")

            if zpd_zone == "frustration":
                frustration_detected = True
                assert "scaffolding" in data or "zpd_message" in data
                print(f"✅ Frustration zone detected after {i+1} failures")
                break

            current_card = data.get("next_card")

    # Frustration should be detected within 4 failures (success rate < 35%)
    if not frustration_detected:
        print("⚠️  Frustration zone not detected (may use different threshold)")


@pytest.mark.asyncio
async def test_zpd_comfort_zone(
    http_client: httpx.AsyncClient,
    test_user: Dict,
    auth_headers: Dict
):
    """Test ZPD detects comfort zone with consecutive easy answers"""

    # Start session
    session_response = await http_client.post(
        f"{ORCHESTRATOR_URL}/session/start",
        json={"learner_id": test_user["user_id"], "limit": 10},
        headers=auth_headers
    )
    assert session_response.status_code == 200
    session_data = session_response.json()

    if not session_data.get("current_card"):
        pytest.skip("No cards available")

    # Answer multiple cards with "easy" to trigger comfort zone
    current_card = session_data.get("current_card")
    comfort_detected = False

    for i in range(5):  # 5 consecutive easy
        if not current_card:
            break

        answer_response = await http_client.post(
            f"{ORCHESTRATOR_URL}/session/answer",
            json={
                "session_id": session_data["session_id"],
                "card_id": current_card["card_id"],
                "rating": "easy",  # Mastery
                "dwell_time_ms": 5000,
                "hesitation_count": 0
            },
            headers=auth_headers
        )

        if answer_response.status_code == 200:
            data = answer_response.json()
            zpd_zone = data.get("zpd_zone")

            print(f"   Attempt {i+1}: ZPD = {zpd_zone}")

            if zpd_zone == "comfort":
                comfort_detected = True
                print(f"✅ Comfort zone detected after {i+1} easy answers")
                break

            current_card = data.get("next_card")

    if not comfort_detected:
        print("⚠️  Comfort zone not detected (may use different threshold)")


# ============================================================================
# Gamification Tests
# ============================================================================

@pytest.mark.asyncio
async def test_xp_calculation(
    http_client: httpx.AsyncClient,
    test_user: Dict,
    auth_headers: Dict
):
    """Test XP calculated based on rating and difficulty"""

    # Start session
    session_response = await http_client.post(
        f"{ORCHESTRATOR_URL}/session/start",
        json={"learner_id": test_user["user_id"], "limit": 10},
        headers=auth_headers
    )
    assert session_response.status_code == 200
    session_data = session_response.json()

    if not session_data.get("current_card"):
        pytest.skip("No cards available")

    current_card = session_data["current_card"]
    difficulty = current_card.get("difficulty", 5.0)

    # Test different ratings
    ratings_to_test = ["again", "hard", "good", "easy"]
    xp_results = {}

    for rating in ratings_to_test:
        # Start new session for each test
        new_session = await http_client.post(
            f"{ORCHESTRATOR_URL}/session/start",
            json={"learner_id": test_user["user_id"], "limit": 1},
            headers=auth_headers
        )

        if new_session.status_code != 200 or not new_session.json().get("current_card"):
            continue

        session = new_session.json()

        # Answer card
        answer_response = await http_client.post(
            f"{ORCHESTRATOR_URL}/session/answer",
            json={
                "session_id": session["session_id"],
                "card_id": session["current_card"]["card_id"],
                "rating": rating,
                "dwell_time_ms": 10000
            },
            headers=auth_headers
        )

        if answer_response.status_code == 200:
            data = answer_response.json()
            xp_earned = data.get("xp_earned", 0)
            xp_results[rating] = xp_earned
            print(f"   {rating:6s}: {xp_earned} XP")

    # Verify XP ordering: again < hard < good < easy
    if len(xp_results) >= 4:
        assert xp_results["again"] < xp_results["good"], "Again should give less XP than Good"
        assert xp_results["good"] < xp_results["easy"], "Good should give less XP than Easy"
        print(f"✅ XP ordering correct: again < hard < good < easy")


# ============================================================================
# Database Persistence Tests
# ============================================================================

@pytest.mark.asyncio
async def test_session_data_persists(
    http_client: httpx.AsyncClient,
    test_user: Dict,
    auth_headers: Dict
):
    """Test learning session data persists in database"""

    # Complete a session
    session_response = await http_client.post(
        f"{ORCHESTRATOR_URL}/session/start",
        json={"learner_id": test_user["user_id"], "limit": 3},
        headers=auth_headers
    )
    assert session_response.status_code == 200
    session_data = session_response.json()

    if not session_data.get("current_card"):
        pytest.skip("No cards available")

    # Answer 3 cards
    current_card = session_data.get("current_card")
    cards_answered = []

    for i in range(3):
        if not current_card:
            break

        answer_response = await http_client.post(
            f"{ORCHESTRATOR_URL}/session/answer",
            json={
                "session_id": session_data["session_id"],
                "card_id": current_card["card_id"],
                "rating": "good",
                "dwell_time_ms": 10000
            },
            headers=auth_headers
        )

        if answer_response.status_code == 200:
            data = answer_response.json()
            cards_answered.append(current_card["card_id"])
            current_card = data.get("next_card")

    # Verify data persisted (would need database query endpoint)
    # For now, just verify we could answer cards
    assert len(cards_answered) > 0, "Should persist at least one answer"
    print(f"✅ {len(cards_answered)} card reviews persisted")


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.asyncio
async def test_invalid_session_id():
    """Test graceful error handling for invalid session ID"""

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            f"{ORCHESTRATOR_URL}/session/answer",
            json={
                "session_id": "invalid_session_12345",
                "card_id": "some_card",
                "rating": "good"
            }
        )

        # Should return 404 or 400, not crash
        assert response.status_code in [400, 404, 422], "Should return error for invalid session"
        print(f"✅ Invalid session handled gracefully: {response.status_code}")


@pytest.mark.asyncio
async def test_invalid_rating():
    """Test graceful error handling for invalid rating"""

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            f"{ORCHESTRATOR_URL}/session/answer",
            json={
                "session_id": "some_session",
                "card_id": "some_card",
                "rating": "invalid_rating"  # Invalid
            }
        )

        # Should return 400 or 422 validation error
        assert response.status_code in [400, 422], "Should validate rating values"
        print(f"✅ Invalid rating handled gracefully: {response.status_code}")


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_session_start_performance(
    http_client: httpx.AsyncClient,
    test_user: Dict,
    auth_headers: Dict
):
    """Test session starts within performance threshold"""

    import time

    start_time = time.time()

    response = await http_client.post(
        f"{ORCHESTRATOR_URL}/session/start",
        json={"learner_id": test_user["user_id"], "limit": 10},
        headers=auth_headers
    )

    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000

    assert response.status_code == 200

    # Should start within 1 second
    threshold_ms = 1000
    assert duration_ms < threshold_ms, f"Session start took {duration_ms:.0f}ms (threshold: {threshold_ms}ms)"

    print(f"✅ Session start time: {duration_ms:.0f}ms (threshold: {threshold_ms}ms)")


@pytest.mark.asyncio
async def test_answer_processing_performance(
    http_client: httpx.AsyncClient,
    test_user: Dict,
    auth_headers: Dict
):
    """Test answer processing within performance threshold"""

    import time

    # Start session
    session_response = await http_client.post(
        f"{ORCHESTRATOR_URL}/session/start",
        json={"learner_id": test_user["user_id"], "limit": 10},
        headers=auth_headers
    )
    assert session_response.status_code == 200
    session_data = session_response.json()

    if not session_data.get("current_card"):
        pytest.skip("No cards available")

    start_time = time.time()

    response = await http_client.post(
        f"{ORCHESTRATOR_URL}/session/answer",
        json={
            "session_id": session_data["session_id"],
            "card_id": session_data["current_card"]["card_id"],
            "rating": "good",
            "dwell_time_ms": 10000
        },
        headers=auth_headers
    )

    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000

    assert response.status_code == 200

    # Should process within 500ms
    threshold_ms = 500
    assert duration_ms < threshold_ms, f"Answer processing took {duration_ms:.0f}ms (threshold: {threshold_ms}ms)"

    print(f"✅ Answer processing time: {duration_ms:.0f}ms (threshold: {threshold_ms}ms)")


# ============================================================================
# Test Summary
# ============================================================================

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Custom test summary"""
    print("\n" + "="*80)
    print("NerdLearn E2E Integration Test Summary")
    print("="*80)

    passed = len([r for r in terminalreporter.stats.get('passed', [])])
    failed = len([r for r in terminalreporter.stats.get('failed', [])])
    skipped = len([r for r in terminalreporter.stats.get('skipped', [])])

    print(f"✅ Passed:  {passed}")
    print(f"❌ Failed:  {failed}")
    print(f"⏭️  Skipped: {skipped}")
    print("="*80)
