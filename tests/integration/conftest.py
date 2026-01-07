"""
Pytest configuration and shared fixtures for integration tests
"""

import pytest
import asyncio
import httpx
from typing import Generator


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires services running)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_db: mark test as requiring database"
    )


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Service Availability Checks
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
async def check_services_available():
    """Check if required services are running before starting tests"""

    services = {
        "Orchestrator": "http://localhost:8005/health",
        "Scheduler": "http://localhost:8001/health",
        "Telemetry": "http://localhost:8002/health",
        "Inference": "http://localhost:8003/health"
    }

    print("\n" + "="*80)
    print("Checking service availability...")
    print("="*80)

    unavailable = []

    async with httpx.AsyncClient(timeout=5.0) as client:
        for service_name, health_url in services.items():
            try:
                response = await client.get(health_url)
                if response.status_code == 200:
                    print(f"✅ {service_name:15s} - Available")
                else:
                    print(f"⚠️  {service_name:15s} - Unhealthy (status {response.status_code})")
                    unavailable.append(service_name)
            except httpx.RequestError:
                print(f"❌ {service_name:15s} - Not reachable")
                unavailable.append(service_name)

    print("="*80)

    if unavailable:
        print(f"\n⚠️  Warning: {len(unavailable)} service(s) unavailable: {', '.join(unavailable)}")
        print("   Some tests may be skipped or fail.")
        print("   Run './scripts/start-all-services.sh' to start services.\n")
    else:
        print("\n✅ All services available!\n")


# ============================================================================
# Database Utilities
# ============================================================================

@pytest.fixture
def db_cleanup():
    """Cleanup test data after tests (if needed)"""
    yield
    # Add cleanup logic here if needed
    # For now, we create unique test users so no cleanup needed


# ============================================================================
# Mock Data Generators
# ============================================================================

@pytest.fixture
def mock_learner_profile():
    """Generate mock learner profile data"""
    return {
        "fsrs_stability": 2.5,
        "fsrs_difficulty": 5.0,
        "current_zpd_lower": 0.35,
        "current_zpd_upper": 0.70,
        "total_xp": 0,
        "level": 1,
        "streak_days": 0
    }


@pytest.fixture
def mock_card_data():
    """Generate mock card data"""
    return {
        "card_id": "test_card_123",
        "concept_id": "test_concept_python_vars",
        "concept_name": "Python Variables",
        "content": "**Variables** are containers for storing data values.",
        "question": "What keyword is used to assign a value to a variable?",
        "correct_answer": "=",
        "difficulty": 3.5,
        "card_type": "BASIC"
    }


# ============================================================================
# Test Helpers
# ============================================================================

@pytest.fixture
def assert_response_time():
    """Helper to assert response time is within threshold"""
    def _assert(duration_ms: float, threshold_ms: float, operation: str):
        assert duration_ms < threshold_ms, (
            f"{operation} took {duration_ms:.0f}ms "
            f"(threshold: {threshold_ms}ms)"
        )
        print(f"   ⏱️  {operation}: {duration_ms:.0f}ms")

    return _assert


@pytest.fixture
def assert_xp_order():
    """Helper to assert XP follows expected ordering"""
    def _assert(xp_results: dict):
        """Verify again < hard < good < easy"""
        if "again" in xp_results and "good" in xp_results:
            assert xp_results["again"] < xp_results["good"]
        if "hard" in xp_results and "good" in xp_results:
            assert xp_results["hard"] < xp_results["good"]
        if "good" in xp_results and "easy" in xp_results:
            assert xp_results["good"] < xp_results["easy"]
        print(f"   ✅ XP ordering correct")

    return _assert
