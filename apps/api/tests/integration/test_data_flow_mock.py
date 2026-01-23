import pytest
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.core.database import get_db

@pytest.mark.asyncio
async def test_health_check_with_mock_db():
    """
    Integration test demonstrating data flow with a mocked database session.
    Verifies that the API correctly handles database interactions.
    """
    
    # 1. Create a Mock Database Session
    mock_session = AsyncMock()
    # Mock the execute result for "SELECT 1"
    mock_result = MagicMock()
    mock_result.scalar.return_value = 1
    mock_session.execute.return_value = mock_result

    # 2. Override the get_db dependency to return our mock session
    async def override_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db

    try:
        # 3. Simulate Client Request
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.get("/health")

        # 4. Verify Data Flow
        assert response.status_code == 200
        data = response.json()
        
        # Check that the service "read" from our mock DB
        assert data["services"]["database"] == "healthy"
        
        # Verify the DB was actually called (Interaction Verification)
        mock_session.execute.assert_called_once()
        
    finally:
        # Clean up overrides
        app.dependency_overrides = {}

@pytest.mark.asyncio
async def test_health_check_db_failure():
    """
    Test scenario where the database fails (mock raising exception).
    Verifies system resilience and error reporting.
    """
    mock_session = AsyncMock()
    mock_session.execute.side_effect = Exception("Connection Timeout")

    async def override_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
        
        # Expect degraded status code (503 as per code)
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "degraded"
        assert data["services"]["database"] == "unhealthy"

    finally:
        app.dependency_overrides = {}
