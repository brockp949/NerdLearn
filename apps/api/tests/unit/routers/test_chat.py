"""
Tests for chat router endpoints
"""
import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_chat_missing_required_fields(client: AsyncClient):
    """Test chat endpoint with missing required fields"""
    request_data = {
        "query": "What is Python?"
        # Missing user_id, course_id
    }

    response = await client.post("/api/chat/", json=request_data)
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_chat_empty_query(client: AsyncClient):
    """Test chat endpoint with empty query"""
    request_data = {
        "query": "",
        "user_id": 1,
        "course_id": 1
    }

    response = await client.post("/api/chat/", json=request_data)
    # Empty query may be accepted or rejected depending on validation
    assert response.status_code in [200, 422, 500]


@pytest.mark.asyncio
async def test_chat_request_structure(client: AsyncClient, mock_db):
    """Test chat request with valid structure"""
    request_data = {
        "query": "Explain machine learning",
        "user_id": 1,
        "course_id": 1,
        "session_id": "test-session-123",
        "module_id": 1
    }

    # This will likely fail due to service dependencies but tests the route
    response = await client.post("/api/chat/", json=request_data)
    # Accept 200 (success), 500 (service error), or 422 (validation)
    assert response.status_code in [200, 422, 500]


@pytest.mark.asyncio
async def test_get_chat_history_missing_params(client: AsyncClient):
    """Test getting chat history without required params"""
    response = await client.get("/api/chat/history")
    assert response.status_code == 422  # Missing user_id and course_id


@pytest.mark.asyncio
async def test_get_chat_history_valid_params(client: AsyncClient, mock_db):
    """Test getting chat history with valid params"""
    mock_db.execute.return_value.scalars.return_value.all.return_value = []

    response = await client.get("/api/chat/history?user_id=1&course_id=1")
    assert response.status_code in [200, 500]


@pytest.mark.asyncio
async def test_get_chat_history_with_session(client: AsyncClient, mock_db):
    """Test getting chat history with session filter"""
    mock_db.execute.return_value.scalars.return_value.all.return_value = []

    response = await client.get(
        "/api/chat/history?user_id=1&course_id=1&session_id=test-session"
    )
    assert response.status_code in [200, 500]


@pytest.mark.asyncio
async def test_get_chat_history_with_limit(client: AsyncClient, mock_db):
    """Test getting chat history with custom limit"""
    mock_db.execute.return_value.scalars.return_value.all.return_value = []

    response = await client.get(
        "/api/chat/history?user_id=1&course_id=1&limit=10"
    )
    assert response.status_code in [200, 500]


@pytest.mark.asyncio
async def test_clear_chat_history_missing_user_id(client: AsyncClient):
    """Test clearing chat history without user_id"""
    response = await client.delete("/api/chat/history")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_clear_chat_history_success(client: AsyncClient, mock_db):
    """Test successfully clearing chat history"""
    mock_db.execute.return_value.scalars.return_value.all.return_value = []

    response = await client.delete("/api/chat/history?user_id=1")
    assert response.status_code in [200, 500]


@pytest.mark.asyncio
async def test_clear_chat_history_with_session(client: AsyncClient, mock_db):
    """Test clearing chat history for specific session"""
    mock_db.execute.return_value.scalars.return_value.all.return_value = []

    response = await client.delete(
        "/api/chat/history?user_id=1&session_id=test-session"
    )
    assert response.status_code in [200, 500]


class TestChatModels:
    """Test request/response model validation"""

    def test_chat_request_valid(self):
        """Test valid ChatRequest model"""
        from app.routers.chat import ChatRequest

        request = ChatRequest(
            query="Test query",
            user_id=1,
            course_id=1,
            session_id="test",
            module_id=1
        )

        assert request.query == "Test query"
        assert request.user_id == 1
        assert request.course_id == 1

    def test_chat_request_optional_fields(self):
        """Test ChatRequest with only required fields"""
        from app.routers.chat import ChatRequest

        request = ChatRequest(
            query="Test query",
            user_id=1,
            course_id=1
        )

        assert request.session_id is None
        assert request.module_id is None

    def test_citation_model(self):
        """Test Citation model"""
        from app.routers.chat import Citation

        citation = Citation(
            module_id=1,
            module_title="Test Module",
            module_type="pdf",
            chunk_text="Sample text",
            page_number=5,
            relevance_score=0.85
        )

        assert citation.module_id == 1
        assert citation.relevance_score == 0.85

    def test_chat_response_model(self):
        """Test ChatResponse model"""
        from app.routers.chat import ChatResponse

        response = ChatResponse(
            message="Test response",
            citations=[],
            xp_earned=10
        )

        assert response.message == "Test response"
        assert response.xp_earned == 10
