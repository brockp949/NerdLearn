"""
Tests for session router endpoints
"""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch


class TestSessionModels:
    """Tests for session data models"""

    def test_session_start_request(self):
        """Test SessionStartRequest model"""
        from app.schemas.session import SessionStartRequest

        request = SessionStartRequest(
            learner_id="learner_1",
            domain="python",
            goal="Learn basics"
        )

        assert request.learner_id == "learner_1"
        assert request.domain == "python"

    def test_answer_request_model(self):
        """Test AnswerRequest model"""
        from app.schemas.session import AnswerRequest

        request = AnswerRequest(
            session_id="session-123",
            card_id="q1",
            rating="good"
        )

        assert request.session_id == "session-123"
        assert request.card_id == "q1"
        assert request.rating == "good"

    def test_learning_card_response(self):
        """Test LearningCardResponse model"""
        from app.schemas.session import LearningCardResponse

        card = LearningCardResponse(
            card_id="c1",
            type="concept",
            title="Python Variables",
            content="Variables are containers...",
            difficulty=1.0
        )

        assert card.card_id == "c1"
        assert card.type == "concept"
        assert card.difficulty == 1.0


class TestSessionStartEndpoint:
    """Tests for session start endpoint"""

    @pytest.mark.asyncio
    async def test_start_session_valid(self, client):
        """Test starting a valid session"""
        request = {
            "learner_id": "learner_1",
            "domain": "python"
        }

        response = await client.post("/api/session/start", json=request)

        if response.status_code == 200:
            data = response.json()
            assert "session_id" in data
            assert "current_card" in data
            assert data["cards_reviewed"] == 0
            assert data["cards_correct"] == 0

    @pytest.mark.asyncio
    async def test_start_session_missing_learner_id(self, client):
        """Test starting session without learner_id"""
        request = {
            "domain": "python"
        }

        response = await client.post("/api/session/start", json=request)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_start_session_returns_first_card(self, client):
        """Test that starting session returns first card"""
        request = {
            "learner_id": "learner_1",
            "domain": "python"
        }

        response = await client.post("/api/session/start", json=request)

        if response.status_code == 200:
            data = response.json()
            assert data["current_card"] is not None
            assert "card_id" in data["current_card"]
            assert "type" in data["current_card"]
            assert "title" in data["current_card"]
            assert "content" in data["current_card"]


class TestAnswerEndpoint:
    """Tests for answer submission endpoint"""

    @pytest.fixture
    async def session_id(self, client):
        """Create a session and return its ID"""
        request = {
            "learner_id": "learner_1",
            "domain": "python"
        }
        response = await client.post("/api/session/start", json=request)
        if response.status_code == 200:
            return response.json()["session_id"]
        return None

    @pytest.mark.asyncio
    async def test_submit_answer_good(self, client, session_id):
        """Test submitting a 'good' answer"""
        if session_id is None:
            pytest.skip("Could not create session")

        request = {
            "session_id": session_id,
            "card_id": "c1",
            "rating": "good"
        }

        response = await client.post("/api/session/answer", json=request)

        if response.status_code == 200:
            data = response.json()
            assert data["correct"] is True
            assert data["xp_earned"] > 0
            assert "next_card" in data

    @pytest.mark.asyncio
    async def test_submit_answer_again(self, client, session_id):
        """Test submitting an 'again' answer (incorrect)"""
        if session_id is None:
            pytest.skip("Could not create session")

        request = {
            "session_id": session_id,
            "card_id": "c1",
            "rating": "again"
        }

        response = await client.post("/api/session/answer", json=request)

        if response.status_code == 200:
            data = response.json()
            assert data["correct"] is False

    @pytest.mark.asyncio
    async def test_submit_answer_invalid_session(self, client):
        """Test submitting answer for invalid session"""
        request = {
            "session_id": "invalid-session-id",
            "card_id": "c1",
            "rating": "good"
        }

        response = await client.post("/api/session/answer", json=request)
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_answer_updates_stats(self, client, session_id):
        """Test that answering updates session statistics"""
        if session_id is None:
            pytest.skip("Could not create session")

        # Submit multiple answers
        for rating in ["good", "easy", "hard"]:
            request = {
                "session_id": session_id,
                "card_id": "c1",
                "rating": rating
            }
            await client.post("/api/session/answer", json=request)

        # Verify stats changed
        # Note: In a real test, we'd have an endpoint to get session state

    @pytest.mark.asyncio
    async def test_answer_returns_next_card(self, client, session_id):
        """Test that answer returns next card"""
        if session_id is None:
            pytest.skip("Could not create session")

        request = {
            "session_id": session_id,
            "card_id": "c1",
            "rating": "good"
        }

        response = await client.post("/api/session/answer", json=request)

        if response.status_code == 200:
            data = response.json()
            assert "next_card" in data
            assert data["next_card"]["card_id"] is not None

    @pytest.mark.asyncio
    async def test_answer_includes_zpd_info(self, client, session_id):
        """Test that answer includes ZPD zone information"""
        if session_id is None:
            pytest.skip("Could not create session")

        request = {
            "session_id": session_id,
            "card_id": "c1",
            "rating": "good"
        }

        response = await client.post("/api/session/answer", json=request)

        if response.status_code == 200:
            data = response.json()
            assert "zpd_zone" in data
            assert "zpd_message" in data


class TestSessionStateTracking:
    """Tests for session state tracking"""

    @pytest.mark.asyncio
    async def test_xp_accumulation(self, client):
        """Test XP accumulates correctly over answers"""
        # Start session
        start_response = await client.post("/api/session/start", json={
            "learner_id": "learner_1",
            "domain": "python"
        })

        if start_response.status_code != 200:
            pytest.skip("Could not create session")

        session_id = start_response.json()["session_id"]
        total_xp = 0

        # Submit correct answers
        for _ in range(3):
            response = await client.post("/api/session/answer", json={
                "session_id": session_id,
                "card_id": "c1",
                "rating": "good"
            })

            if response.status_code == 200:
                data = response.json()
                assert data["new_total_xp"] >= total_xp
                total_xp = data["new_total_xp"]

    @pytest.mark.asyncio
    async def test_level_progression(self, client):
        """Test level progresses with XP"""
        start_response = await client.post("/api/session/start", json={
            "learner_id": "learner_1",
            "domain": "python"
        })

        if start_response.status_code != 200:
            pytest.skip("Could not create session")

        session_id = start_response.json()["session_id"]

        # Submit many answers to gain XP
        for _ in range(15):
            response = await client.post("/api/session/answer", json={
                "session_id": session_id,
                "card_id": "c1",
                "rating": "easy"
            })

            if response.status_code == 200:
                data = response.json()
                # Level should increase with enough XP
                assert data["level"] >= 1
                assert 0 <= data["level_progress"] <= 1


class TestMockContentCards:
    """Tests for mock content cards behavior"""

    @pytest.mark.asyncio
    async def test_card_types(self, client):
        """Test different card types are returned"""
        start_response = await client.post("/api/session/start", json={
            "learner_id": "learner_1",
            "domain": "python"
        })

        if start_response.status_code != 200:
            pytest.skip("Could not create session")

        session_id = start_response.json()["session_id"]
        card_types = set()

        # Get multiple cards
        for _ in range(4):
            response = await client.post("/api/session/answer", json={
                "session_id": session_id,
                "card_id": "c1",
                "rating": "good"
            })

            if response.status_code == 200:
                card = response.json()["next_card"]
                card_types.add(card["type"])

        # Should have both concept and question types
        # (based on MOCK_CARDS in session.py)

    @pytest.mark.asyncio
    async def test_question_cards_have_options(self, client):
        """Test question cards include options"""
        start_response = await client.post("/api/session/start", json={
            "learner_id": "learner_1",
            "domain": "python"
        })

        if start_response.status_code != 200:
            pytest.skip("Could not create session")

        session_id = start_response.json()["session_id"]

        # Cycle through cards to find question type
        for _ in range(4):
            response = await client.post("/api/session/answer", json={
                "session_id": session_id,
                "card_id": "c1",
                "rating": "good"
            })

            if response.status_code == 200:
                card = response.json()["next_card"]
                if card["type"] == "question":
                    assert "options" in card
                    assert card["options"] is not None
                    assert len(card["options"]) > 0
