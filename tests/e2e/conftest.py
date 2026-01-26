"""
E2E test configuration and fixtures.

E2E tests validate complete user journeys across all services.
These tests require all services to be running.
"""

import pytest
import httpx
from typing import Dict

# Import shared fixtures
from tests.fixtures import create_test_user_data, create_mock_learner_profile


# ============================================================================
# Service Configuration
# ============================================================================

BASE_URL = "http://localhost:8000"  # API Gateway
ORCHESTRATOR_URL = "http://localhost:8005"
SCHEDULER_URL = "http://localhost:8001"
TELEMETRY_URL = "http://localhost:8002"
INFERENCE_URL = "http://localhost:8003"

TIMEOUT = 10.0  # seconds


# ============================================================================
# HTTP Client Fixtures
# ============================================================================

@pytest.fixture
async def http_client():
    """Async HTTP client for API requests."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        yield client


# ============================================================================
# User Fixtures
# ============================================================================

@pytest.fixture
async def test_user(http_client: httpx.AsyncClient) -> Dict:
    """
    Create and authenticate a test user.

    Registers a new user, logs in, and returns user data with access token.
    """
    user_data = create_test_user_data()

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
    """HTTP headers with authentication token."""
    return {
        "Authorization": f"Bearer {test_user['access_token']}",
        "Content-Type": "application/json"
    }


# ============================================================================
# Mock Data Fixtures
# ============================================================================

@pytest.fixture
def mock_learner_profile():
    """Generate mock learner profile data."""
    return create_mock_learner_profile()


# ============================================================================
# Service Health Checks
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
async def check_services_available():
    """Check if required services are running before starting E2E tests."""
    services = {
        "API Gateway": f"{BASE_URL}/health",
        "Orchestrator": f"{ORCHESTRATOR_URL}/health",
        "Scheduler": f"{SCHEDULER_URL}/health",
        "Telemetry": f"{TELEMETRY_URL}/health",
        "Inference": f"{INFERENCE_URL}/health"
    }

    print("\n" + "=" * 80)
    print("E2E Tests - Checking service availability...")
    print("=" * 80)

    unavailable = []

    async with httpx.AsyncClient(timeout=5.0) as client:
        for service_name, health_url in services.items():
            try:
                response = await client.get(health_url)
                if response.status_code == 200:
                    print(f"  {service_name:15s} - Available")
                else:
                    print(f"  {service_name:15s} - Unhealthy (status {response.status_code})")
                    unavailable.append(service_name)
            except httpx.RequestError:
                print(f"  {service_name:15s} - Not reachable")
                unavailable.append(service_name)

    print("=" * 80)

    if unavailable:
        print(f"\n  Warning: {len(unavailable)} service(s) unavailable: {', '.join(unavailable)}")
        print("  Some tests may be skipped or fail.")
        print("  Run './scripts/start-all-services.sh' to start services.\n")


# ============================================================================
# Test Helpers
# ============================================================================

@pytest.fixture
def assert_response_time():
    """Helper to assert response time is within threshold."""
    def _assert(duration_ms: float, threshold_ms: float, operation: str):
        assert duration_ms < threshold_ms, (
            f"{operation} took {duration_ms:.0f}ms "
            f"(threshold: {threshold_ms}ms)"
        )
        print(f"  {operation}: {duration_ms:.0f}ms")

    return _assert


@pytest.fixture
def assert_xp_order():
    """Helper to assert XP follows expected ordering."""
    def _assert(xp_results: dict):
        """Verify again < hard < good < easy"""
        if "again" in xp_results and "good" in xp_results:
            assert xp_results["again"] < xp_results["good"]
        if "hard" in xp_results and "good" in xp_results:
            assert xp_results["hard"] < xp_results["good"]
        if "good" in xp_results and "easy" in xp_results:
            assert xp_results["good"] < xp_results["easy"]
        print("  XP ordering correct")

    return _assert
