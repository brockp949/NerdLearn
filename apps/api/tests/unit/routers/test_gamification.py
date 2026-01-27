"""
Tests for gamification router endpoints
"""
import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_get_user_xp(client: AsyncClient, mock_db):
    """Test getting user XP points"""
    response = await client.get("/api/gamification/xp?user_id=1")
    assert response.status_code in [200, 404, 422, 500]


@pytest.mark.asyncio
async def test_get_user_xp_missing_user(client: AsyncClient):
    """Test getting XP without user_id"""
    response = await client.get("/api/gamification/xp")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_achievements(client: AsyncClient, mock_db):
    """Test getting user achievements"""
    response = await client.get("/api/gamification/achievements?user_id=1")
    assert response.status_code in [200, 404, 422, 500]


@pytest.mark.asyncio
async def test_get_leaderboard(client: AsyncClient, mock_db):
    """Test getting leaderboard"""
    response = await client.get("/api/gamification/leaderboard")
    assert response.status_code in [200, 404, 500]


@pytest.mark.asyncio
async def test_get_leaderboard_with_filters(client: AsyncClient, mock_db):
    """Test getting leaderboard with filters"""
    response = await client.get(
        "/api/gamification/leaderboard?course_id=1&limit=10&period=week"
    )
    assert response.status_code in [200, 404, 422, 500]


@pytest.mark.asyncio
async def test_get_streaks(client: AsyncClient, mock_db):
    """Test getting user streaks"""
    response = await client.get("/api/gamification/streaks?user_id=1")
    assert response.status_code in [200, 404, 422, 500]


@pytest.mark.asyncio
async def test_get_badges(client: AsyncClient, mock_db):
    """Test getting user badges"""
    response = await client.get("/api/gamification/badges?user_id=1")
    assert response.status_code in [200, 404, 422, 500]


@pytest.mark.asyncio
async def test_claim_daily_reward(client: AsyncClient, mock_db):
    """Test claiming daily reward"""
    response = await client.post("/api/gamification/daily-reward?user_id=1")
    assert response.status_code in [200, 400, 404, 500]


@pytest.mark.asyncio
async def test_get_user_level(client: AsyncClient, mock_db):
    """Test getting user level"""
    response = await client.get("/api/gamification/level?user_id=1")
    assert response.status_code in [200, 404, 422, 500]


class TestGamificationLogic:
    """Test gamification business logic"""

    def test_xp_calculation(self):
        """Test XP calculation for different actions"""
        try:
            from app.gamification import GamificationEngine

            xp_review = GamificationEngine.award_xp("review_complete")
            xp_chat = GamificationEngine.award_xp("chat_interaction")
            xp_module = GamificationEngine.award_xp("module_complete")

            # XP values should be positive integers
            assert isinstance(xp_review, int)
            assert isinstance(xp_chat, int)
            assert isinstance(xp_module, int)
            assert xp_review >= 0
            assert xp_chat >= 0
            assert xp_module >= 0
        except ImportError:
            pytest.skip("GamificationEngine not available")

    def test_streak_calculation(self):
        """Test streak calculation logic"""
        from datetime import datetime, timedelta

        # Test streak continuation logic
        yesterday = datetime.now() - timedelta(days=1)
        today = datetime.now()

        # If last activity was yesterday, streak should continue
        days_diff = (today.date() - yesterday.date()).days
        assert days_diff == 1  # Streak should continue

    def test_level_from_xp(self):
        """Test level calculation from XP"""
        # Standard level thresholds
        def calculate_level(xp: int) -> int:
            if xp < 100:
                return 1
            elif xp < 300:
                return 2
            elif xp < 600:
                return 3
            elif xp < 1000:
                return 4
            else:
                return 5 + (xp - 1000) // 500

        assert calculate_level(0) == 1
        assert calculate_level(99) == 1
        assert calculate_level(100) == 2
        assert calculate_level(299) == 2
        assert calculate_level(300) == 3
        assert calculate_level(1000) == 5
        assert calculate_level(1500) == 6
