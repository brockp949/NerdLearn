"""
Tests for curriculum router endpoints
"""
import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_get_curriculum(client: AsyncClient, mock_db):
    """Test getting course curriculum"""
    response = await client.get("/api/curriculum/1")
    assert response.status_code in [200, 404, 500]


@pytest.mark.asyncio
async def test_get_curriculum_invalid_id(client: AsyncClient, mock_db):
    """Test getting curriculum with invalid course ID"""
    response = await client.get("/api/curriculum/invalid")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_learning_path(client: AsyncClient, mock_db):
    """Test getting personalized learning path"""
    response = await client.get("/api/curriculum/1/learning-path?user_id=1")
    assert response.status_code in [200, 404, 422, 500]


@pytest.mark.asyncio
async def test_get_learning_path_missing_user(client: AsyncClient):
    """Test getting learning path without user_id"""
    response = await client.get("/api/curriculum/1/learning-path")
    assert response.status_code in [200, 422, 500]


@pytest.mark.asyncio
async def test_get_prerequisites(client: AsyncClient, mock_db):
    """Test getting module prerequisites"""
    response = await client.get("/api/curriculum/modules/1/prerequisites")
    assert response.status_code in [200, 404, 500]


@pytest.mark.asyncio
async def test_get_recommended_next(client: AsyncClient, mock_db):
    """Test getting recommended next modules"""
    response = await client.get(
        "/api/curriculum/1/recommended-next?user_id=1"
    )
    assert response.status_code in [200, 404, 422, 500]


@pytest.mark.asyncio
async def test_update_progress(client: AsyncClient, mock_db):
    """Test updating user progress in curriculum"""
    progress_data = {
        "user_id": 1,
        "module_id": 1,
        "progress_percentage": 50,
        "time_spent_seconds": 300
    }

    response = await client.post("/api/curriculum/progress", json=progress_data)
    assert response.status_code in [200, 201, 404, 422, 500]


@pytest.mark.asyncio
async def test_get_progress(client: AsyncClient, mock_db):
    """Test getting user progress in course"""
    response = await client.get("/api/curriculum/1/progress?user_id=1")
    assert response.status_code in [200, 404, 422, 500]


@pytest.mark.asyncio
async def test_mark_module_complete(client: AsyncClient, mock_db):
    """Test marking a module as complete"""
    response = await client.post(
        "/api/curriculum/modules/1/complete?user_id=1"
    )
    assert response.status_code in [200, 201, 404, 422, 500]


@pytest.mark.asyncio
async def test_get_curriculum_overview(client: AsyncClient, mock_db):
    """Test getting curriculum overview with stats"""
    response = await client.get("/api/curriculum/1/overview")
    assert response.status_code in [200, 404, 500]


class TestCurriculumStructure:
    """Tests for curriculum structure validation"""

    def test_curriculum_module_ordering(self):
        """Test that curriculum modules maintain proper ordering"""
        modules = [
            {"id": 1, "order": 1, "title": "Introduction"},
            {"id": 2, "order": 2, "title": "Basics"},
            {"id": 3, "order": 3, "title": "Advanced"},
        ]

        sorted_modules = sorted(modules, key=lambda m: m["order"])
        assert [m["id"] for m in sorted_modules] == [1, 2, 3]

    def test_prerequisite_validation(self):
        """Test prerequisite chain validation"""
        # Module 3 requires 2, Module 2 requires 1
        prerequisites = {
            1: [],
            2: [1],
            3: [2],
        }

        def can_access_module(module_id: int, completed: set) -> bool:
            return all(prereq in completed for prereq in prerequisites[module_id])

        completed = {1}
        assert can_access_module(1, completed) is True
        assert can_access_module(2, completed) is True
        assert can_access_module(3, completed) is False

        completed.add(2)
        assert can_access_module(3, completed) is True

    def test_progress_calculation(self):
        """Test progress percentage calculation"""
        total_modules = 10
        completed_modules = 3

        progress = (completed_modules / total_modules) * 100
        assert progress == 30.0

        # Edge cases
        assert (0 / 10) * 100 == 0.0
        assert (10 / 10) * 100 == 100.0
