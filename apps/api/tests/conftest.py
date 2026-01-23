"""
Pytest configuration and fixtures for API tests
"""
import pytest
import asyncio
from typing import AsyncGenerator
from httpx import AsyncClient, ASGITransport

# Import app lazily to avoid import errors
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client for testing"""
    from app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client

@pytest.fixture
def telemetry_service():
    """Fixture for TelemetryService that resets metrics after each test"""
    from app.core.telemetry import TelemetryService
    service = TelemetryService()
    yield service
    service.reset_metrics()

@pytest.fixture
def mock_executor():
    """Fixture for SafePythonExecutor"""
    from unittest.mock import MagicMock
    executor = MagicMock()
    # mock execute return
    executor.execute.return_value = {
        "success": True,
        "output": "Simulated Output",
        "error": None
    }
    return executor
