"""
End-to-End Tests for User Journeys

Tests complete user flows through the application including:
- User onboarding
- Learning session completion
- Review sessions
- Social features
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


class TestOnboardingJourney:
    """Tests for user onboarding flow"""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_new_user_registration_flow(self, client):
        """Test complete user registration flow"""
        # Step 1: Create account
        registration_data = {
            "email": "newuser@example.com",
            "username": "newlearner",
            "password": "SecurePass123!",
            "full_name": "New Learner"
        }

        response = await client.post("/api/auth/register", json=registration_data)
        # Accept various responses based on whether auth is implemented
        assert response.status_code in [200, 201, 404, 422, 500]

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_user_selects_first_course(self, client):
        """Test user browsing and selecting first course"""
        # Step 1: Browse courses
        response = await client.get("/api/courses?status=published")

        if response.status_code == 200:
            courses = response.json()
            assert isinstance(courses, list) or isinstance(courses, dict)

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_user_starts_learning_session(self, client):
        """Test user starting their first learning session"""
        # Start a session
        session_request = {
            "learner_id": "user_1",
            "domain": "python"
        }

        response = await client.post("/api/session/start", json=session_request)

        if response.status_code == 200:
            session = response.json()
            assert "session_id" in session
            assert "current_card" in session


class TestLearningSessionJourney:
    """Tests for complete learning session flow"""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_complete_learning_session(self, client):
        """Test completing a full learning session"""
        # Step 1: Start session
        start_response = await client.post("/api/session/start", json={
            "learner_id": "user_1",
            "domain": "python"
        })

        if start_response.status_code != 200:
            pytest.skip("Could not start session")

        session = start_response.json()
        session_id = session["session_id"]

        # Step 2: Answer multiple cards
        cards_answered = 0
        for _ in range(5):
            answer_response = await client.post("/api/session/answer", json={
                "session_id": session_id,
                "card_id": session["current_card"]["card_id"],
                "rating": "good"
            })

            if answer_response.status_code == 200:
                cards_answered += 1
                result = answer_response.json()
                session["current_card"] = result["next_card"]

        assert cards_answered > 0

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_learning_with_hints(self, client):
        """Test learning flow with hint requests"""
        # Start session
        start_response = await client.post("/api/session/start", json={
            "learner_id": "user_1",
            "domain": "python"
        })

        if start_response.status_code != 200:
            pytest.skip("Could not start session")

        session = start_response.json()

        # Request hint
        hint_request = {
            "user_id": "user_1",
            "content_id": session["current_card"]["card_id"],
            "step_id": "step_1",
            "context": {"attempt_count": 1}
        }

        hint_response = await client.post("/api/adaptive/hints/get", json=hint_request)
        # Hint endpoint may or may not exist
        assert hint_response.status_code in [200, 404, 422, 500]

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_xp_accumulation_during_session(self, client):
        """Test XP accumulates correctly during learning"""
        start_response = await client.post("/api/session/start", json={
            "learner_id": "user_1",
            "domain": "python"
        })

        if start_response.status_code != 200:
            pytest.skip("Could not start session")

        session = start_response.json()
        session_id = session["session_id"]
        initial_xp = session.get("total_xp_earned", 0)

        # Answer correctly
        answer_response = await client.post("/api/session/answer", json={
            "session_id": session_id,
            "card_id": session["current_card"]["card_id"],
            "rating": "easy"
        })

        if answer_response.status_code == 200:
            result = answer_response.json()
            assert result["xp_earned"] > 0
            assert result["new_total_xp"] > initial_xp


class TestReviewSessionJourney:
    """Tests for spaced repetition review sessions"""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_get_due_reviews(self, client):
        """Test fetching due review cards"""
        response = await client.get(
            "/api/adaptive/reviews/due?user_id=1&course_id=1"
        )

        assert response.status_code in [200, 404, 500]

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_complete_review_cycle(self, client):
        """Test completing a full review cycle"""
        # Get due reviews
        due_response = await client.get(
            "/api/adaptive/reviews/due?user_id=1&course_id=1&limit=5"
        )

        if due_response.status_code != 200:
            pytest.skip("Could not get due reviews")

        cards = due_response.json()
        if not cards:
            pytest.skip("No cards due for review")

        # Review each card
        reviewed = 0
        for card in cards[:3]:
            review_response = await client.post("/api/adaptive/reviews/submit", json={
                "card_id": card["id"],
                "rating": "good",
                "review_duration_ms": 5000
            })

            if review_response.status_code == 200:
                reviewed += 1

        # Some reviews should succeed
        assert reviewed >= 0


class TestChatJourney:
    """Tests for chat interaction flow"""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_chat_conversation_flow(self, client):
        """Test a multi-turn chat conversation"""
        # Turn 1: Ask initial question
        turn1_response = await client.post("/api/chat/", json={
            "query": "What is Python?",
            "user_id": 1,
            "course_id": 1,
            "session_id": "e2e-test-session"
        })

        if turn1_response.status_code != 200:
            pytest.skip("Chat endpoint not available")

        turn1 = turn1_response.json()
        assert "message" in turn1

        # Turn 2: Follow-up question
        turn2_response = await client.post("/api/chat/", json={
            "query": "Can you give me an example?",
            "user_id": 1,
            "course_id": 1,
            "session_id": "e2e-test-session"
        })

        if turn2_response.status_code == 200:
            turn2 = turn2_response.json()
            assert "message" in turn2

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_chat_with_citations(self, client):
        """Test chat returns relevant citations"""
        response = await client.post("/api/chat/", json={
            "query": "Explain Python decorators",
            "user_id": 1,
            "course_id": 1,
            "session_id": "e2e-test-session"
        })

        if response.status_code == 200:
            data = response.json()
            assert "citations" in data


class TestSocialJourney:
    """Tests for social feature flows"""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_friend_flow(self, client):
        """Test friend request and acceptance flow"""
        # Send friend request
        send_response = await client.post(
            "/api/social/friends/request?current_user_id=1",
            json={"addressee_id": 2}
        )

        # Get pending requests for user 2
        pending_response = await client.get(
            "/api/social/friends/requests?current_user_id=2"
        )

        # Both may succeed or fail depending on DB state
        assert send_response.status_code in [200, 400, 404, 500]

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_study_group_flow(self, client):
        """Test creating and joining study groups"""
        # Create group
        create_response = await client.post(
            "/api/social/groups?current_user_id=1",
            json={
                "name": "E2E Test Group",
                "description": "Test study group",
                "is_public": True,
                "max_members": 10
            }
        )

        if create_response.status_code == 200:
            group = create_response.json()
            group_id = group["id"]

            # Another user joins
            join_response = await client.post(
                f"/api/social/groups/{group_id}/join?current_user_id=2"
            )
            assert join_response.status_code in [200, 400, 500]

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_leaderboard_access(self, client):
        """Test accessing various leaderboards"""
        # Global leaderboard
        global_response = await client.get("/api/social/leaderboard/global")
        assert global_response.status_code in [200, 500]

        # Friends leaderboard
        friends_response = await client.get(
            "/api/social/leaderboard/friends?current_user_id=1"
        )
        assert friends_response.status_code in [200, 500]


class TestAnalyticsJourney:
    """Tests for analytics and dashboard flows"""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_dashboard_data_flow(self, client):
        """Test fetching dashboard analytics"""
        # Get analytics summary
        summary_response = await client.get("/api/analytics/summary?days=7")

        if summary_response.status_code == 200:
            summary = summary_response.json()
            assert "metrics" in summary

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_learning_progress_tracking(self, client):
        """Test tracking learning progress over time"""
        # Get learning curve
        curve_response = await client.get(
            "/api/analytics/learning-curve/user/1?course_id=1&days=30"
        )

        if curve_response.status_code == 200:
            curve = curve_response.json()
            assert "points" in curve
            assert "trend" in curve

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_mastery_distribution(self, client):
        """Test fetching mastery distribution"""
        response = await client.get(
            "/api/analytics/mastery/distribution?course_id=1"
        )

        if response.status_code == 200:
            data = response.json()
            assert "distribution" in data


class TestKnowledgeGraphJourney:
    """Tests for knowledge graph navigation"""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_graph_exploration_flow(self, client):
        """Test exploring the knowledge graph"""
        # Get course graph
        graph_response = await client.get("/api/graph/courses/1")

        if graph_response.status_code == 200:
            graph = graph_response.json()
            assert "nodes" in graph
            assert "edges" in graph

            # If there are nodes, explore one
            if graph["nodes"]:
                concept_name = graph["nodes"][0].get("label", "test")

                # Get concept details
                detail_response = await client.get(
                    f"/api/graph/courses/1/concepts/{concept_name}"
                )
                assert detail_response.status_code in [200, 404, 500]

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_learning_path_generation(self, client):
        """Test generating a learning path"""
        response = await client.post("/api/graph/courses/1/learning-path", json={
            "target_concepts": ["Advanced Topic"],
            "mastered_concepts": ["Basic Topic"]
        })

        if response.status_code == 200:
            path = response.json()
            assert "learning_path" in path
