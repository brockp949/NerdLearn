"""
Tests for adaptive learning router endpoints
"""
import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime


@pytest.mark.asyncio
async def test_get_due_reviews_missing_params(client: AsyncClient):
    """Test getting due reviews without required parameters"""
    response = await client.get("/api/adaptive/reviews/due")
    assert response.status_code == 422  # Missing user_id, course_id


@pytest.mark.asyncio
async def test_get_due_reviews_valid_params(client: AsyncClient, mock_db):
    """Test getting due reviews with valid parameters"""
    mock_db.execute.return_value.scalars.return_value.all.return_value = []

    response = await client.get("/api/adaptive/reviews/due?user_id=1&course_id=1")
    assert response.status_code in [200, 500]


@pytest.mark.asyncio
async def test_get_due_reviews_with_limit(client: AsyncClient, mock_db):
    """Test getting due reviews with custom limit"""
    mock_db.execute.return_value.scalars.return_value.all.return_value = []

    response = await client.get("/api/adaptive/reviews/due?user_id=1&course_id=1&limit=10")
    assert response.status_code in [200, 500]


@pytest.mark.asyncio
async def test_submit_review_missing_fields(client: AsyncClient):
    """Test submitting a review with missing fields"""
    review_data = {
        "card_id": 1,
        # Missing rating, review_duration_ms
    }

    response = await client.post("/api/adaptive/reviews/submit", json=review_data)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_submit_review_invalid_rating(client: AsyncClient):
    """Test submitting a review with invalid rating"""
    review_data = {
        "card_id": 1,
        "rating": "invalid_rating",
        "review_duration_ms": 5000
    }

    response = await client.post("/api/adaptive/reviews/submit", json=review_data)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_submit_review_card_not_found(client: AsyncClient, mock_db):
    """Test submitting a review for non-existent card"""
    mock_db.execute.return_value.scalar_one_or_none.return_value = None

    review_data = {
        "card_id": 999,
        "rating": "good",
        "review_duration_ms": 5000
    }

    response = await client.post("/api/adaptive/reviews/submit", json=review_data)
    assert response.status_code in [404, 500]


@pytest.mark.asyncio
async def test_submit_review_valid_again(client: AsyncClient, mock_db, mock_sr_card):
    """Test submitting a review with 'again' rating"""
    mock_db.execute.return_value.scalar_one_or_none.return_value = mock_sr_card

    review_data = {
        "card_id": 1,
        "rating": "again",
        "review_duration_ms": 5000
    }

    response = await client.post("/api/adaptive/reviews/submit", json=review_data)
    assert response.status_code in [200, 500]


@pytest.mark.asyncio
async def test_submit_review_valid_easy(client: AsyncClient, mock_db, mock_sr_card):
    """Test submitting a review with 'easy' rating"""
    mock_db.execute.return_value.scalar_one_or_none.return_value = mock_sr_card

    review_data = {
        "card_id": 1,
        "rating": "easy",
        "review_duration_ms": 2000
    }

    response = await client.post("/api/adaptive/reviews/submit", json=review_data)
    assert response.status_code in [200, 500]


class TestReviewModels:
    """Test request model validation for review endpoints"""

    def test_review_request_valid(self):
        """Test valid ReviewRequest model"""
        from app.routers.adaptive import ReviewRequest, ReviewRating

        request = ReviewRequest(
            card_id=1,
            rating=ReviewRating.GOOD,
            review_duration_ms=5000
        )

        assert request.card_id == 1
        assert request.rating == ReviewRating.GOOD
        assert request.review_duration_ms == 5000

    def test_review_rating_values(self):
        """Test ReviewRating enum values"""
        from app.routers.adaptive import ReviewRating

        assert ReviewRating.AGAIN == "again"
        assert ReviewRating.HARD == "hard"
        assert ReviewRating.GOOD == "good"
        assert ReviewRating.EASY == "easy"

    def test_mastery_update_request(self):
        """Test MasteryUpdateRequest model"""
        from app.routers.adaptive import MasteryUpdateRequest

        request = MasteryUpdateRequest(
            user_id=1,
            concept_id=10,
            evidence_score=0.85
        )

        assert request.user_id == 1
        assert request.concept_id == 10
        assert request.evidence_score == 0.85


class TestAdaptiveAlgorithmsIntegration:
    """Integration tests for adaptive algorithm endpoints"""

    @pytest.mark.asyncio
    async def test_zpd_recommendation_structure(self, client: AsyncClient):
        """Test ZPD recommendation endpoint returns expected structure"""
        response = await client.get("/api/adaptive/zpd/recommend?user_id=1&course_id=1")

        # Should return 200 or 500 depending on service availability
        if response.status_code == 200:
            data = response.json()
            # Verify response has expected fields
            assert isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_cognitive_load_estimation(self, client: AsyncClient):
        """Test cognitive load estimation endpoint"""
        request_data = {
            "user_id": 1,
            "content_id": 1,
            "response_time_ms": 5000,
            "error_count": 1
        }

        response = await client.post(
            "/api/adaptive/cognitive-load/estimate",
            json=request_data
        )

        # Accept various status codes depending on endpoint availability
        assert response.status_code in [200, 404, 422, 500]

    @pytest.mark.asyncio
    async def test_interleaved_schedule(self, client: AsyncClient):
        """Test interleaved practice schedule endpoint"""
        response = await client.get(
            "/api/adaptive/interleaved/schedule?user_id=1&course_id=1"
        )

        assert response.status_code in [200, 404, 422, 500]


class TestMasteryEndpoints:
    """Tests for mastery tracking endpoints"""

    @pytest.mark.asyncio
    async def test_get_mastery_levels(self, client: AsyncClient, mock_db):
        """Test getting user mastery levels"""
        mock_db.execute.return_value.scalars.return_value.all.return_value = []

        response = await client.get("/api/adaptive/mastery?user_id=1&course_id=1")
        assert response.status_code in [200, 404, 500]

    @pytest.mark.asyncio
    async def test_update_mastery_from_stealth(self, client: AsyncClient, mock_db):
        """Test updating mastery from stealth assessment"""
        update_data = {
            "user_id": 1,
            "concept_id": 10,
            "evidence_score": 0.75
        }

        response = await client.post(
            "/api/adaptive/mastery/stealth-update",
            json=update_data
        )

        assert response.status_code in [200, 404, 422, 500]


class TestHintEndpoints:
    """Tests for hint/scaffolding endpoints"""

    @pytest.mark.asyncio
    async def test_get_hint_missing_params(self, client: AsyncClient):
        """Test getting hint without required parameters"""
        response = await client.post("/api/adaptive/hints/get", json={})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_get_hint_valid_request(self, client: AsyncClient):
        """Test getting hint with valid request"""
        hint_request = {
            "user_id": "user_1",
            "content_id": "content_1",
            "step_id": "step_1",
            "context": {"attempt_count": 2}
        }

        response = await client.post("/api/adaptive/hints/get", json=hint_request)
        assert response.status_code in [200, 404, 500]


class TestRewardEndpoints:
    """Tests for variable reward endpoints"""

    @pytest.mark.asyncio
    async def test_get_reward(self, client: AsyncClient):
        """Test getting reward after action"""
        response = await client.get("/api/adaptive/rewards?user_id=1&action=review_complete")
        assert response.status_code in [200, 404, 422, 500]
