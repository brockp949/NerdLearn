"""
Integration test configuration and fixtures.

Integration tests validate cross-service communication and data flow.
These tests verify that services work correctly together.
"""

import pytest
import httpx
from typing import Dict

# Import shared fixtures
from tests.fixtures import (
    create_test_user_data,
    create_mock_learner_profile,
    create_mock_card
)


# ============================================================================
# Service Configuration
# ============================================================================

API_GATEWAY_URL = "http://localhost:8000"
ORCHESTRATOR_URL = "http://localhost:8005"
SCHEDULER_URL = "http://localhost:8001"
TELEMETRY_URL = "http://localhost:8002"
INFERENCE_URL = "http://localhost:8003"

TIMEOUT = 10.0


# ============================================================================
# HTTP Client Fixtures
# ============================================================================

@pytest.fixture
async def http_client():
    """Async HTTP client for service requests."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        yield client


@pytest.fixture
async def api_client():
    """HTTP client configured for API Gateway."""
    async with httpx.AsyncClient(
        base_url=API_GATEWAY_URL,
        timeout=TIMEOUT
    ) as client:
        yield client


@pytest.fixture
async def orchestrator_client():
    """HTTP client configured for Orchestrator service."""
    async with httpx.AsyncClient(
        base_url=ORCHESTRATOR_URL,
        timeout=TIMEOUT
    ) as client:
        yield client


@pytest.fixture
async def scheduler_client():
    """HTTP client configured for Scheduler service."""
    async with httpx.AsyncClient(
        base_url=SCHEDULER_URL,
        timeout=TIMEOUT
    ) as client:
        yield client


# ============================================================================
# Service Health Fixtures
# ============================================================================

@pytest.fixture
async def ensure_api_healthy(api_client):
    """Ensure API Gateway is healthy before test."""
    try:
        response = await api_client.get("/health")
        if response.status_code != 200:
            pytest.skip("API Gateway not healthy")
    except httpx.RequestError:
        pytest.skip("API Gateway not reachable")


@pytest.fixture
async def ensure_orchestrator_healthy(orchestrator_client):
    """Ensure Orchestrator is healthy before test."""
    try:
        response = await orchestrator_client.get("/health")
        if response.status_code != 200:
            pytest.skip("Orchestrator not healthy")
    except httpx.RequestError:
        pytest.skip("Orchestrator not reachable")


@pytest.fixture
async def ensure_scheduler_healthy(scheduler_client):
    """Ensure Scheduler is healthy before test."""
    try:
        response = await scheduler_client.get("/health")
        if response.status_code != 200:
            pytest.skip("Scheduler not healthy")
    except httpx.RequestError:
        pytest.skip("Scheduler not reachable")


# ============================================================================
# Mock Data Fixtures
# ============================================================================

@pytest.fixture
def mock_learner_profile():
    """Generate mock learner profile data."""
    return create_mock_learner_profile()


@pytest.fixture
def mock_card_data():
    """Generate mock card data."""
    return create_mock_card()


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
