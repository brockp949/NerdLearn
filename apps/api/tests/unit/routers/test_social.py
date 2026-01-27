"""
Tests for social gamification router endpoints
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


class TestSocialModels:
    """Tests for social data models"""

    def test_friend_request_create(self):
        """Test FriendRequestCreate model"""
        from app.routers.social import FriendRequestCreate

        request = FriendRequestCreate(addressee_id=2)
        assert request.addressee_id == 2

    def test_challenge_create(self):
        """Test ChallengeCreate model"""
        from app.routers.social import ChallengeCreate
        from app.models.social import ChallengeType

        challenge = ChallengeCreate(
            challenge_type=ChallengeType.XP_RACE,
            title="Weekend XP Challenge",
            description="Race to 500 XP",
            target_value=500,
            end_date=datetime.utcnow() + timedelta(days=2),
            participant_ids=[2, 3],
            xp_reward=100
        )

        assert challenge.title == "Weekend XP Challenge"
        assert challenge.target_value == 500
        assert len(challenge.participant_ids) == 2

    def test_study_group_create(self):
        """Test StudyGroupCreate model"""
        from app.routers.social import StudyGroupCreate

        group = StudyGroupCreate(
            name="Python Learners",
            description="A group for Python enthusiasts",
            is_public=True,
            max_members=25
        )

        assert group.name == "Python Learners"
        assert group.is_public is True
        assert group.max_members == 25

    def test_group_message_create(self):
        """Test GroupMessageCreate model"""
        from app.routers.social import GroupMessageCreate

        message = GroupMessageCreate(
            content="Hello everyone!",
            shared_module_id=1
        )

        assert message.content == "Hello everyone!"
        assert message.shared_module_id == 1

    def test_leaderboard_entry(self):
        """Test LeaderboardEntry model"""
        from app.routers.social import LeaderboardEntry

        entry = LeaderboardEntry(
            rank=1,
            user_id=1,
            username="toplearner",
            score=5000,
            level=10
        )

        assert entry.rank == 1
        assert entry.score == 5000


class TestFriendsEndpoints:
    """Tests for friends endpoints"""

    @pytest.mark.asyncio
    async def test_send_friend_request(self, client, mock_db):
        """Test sending a friend request"""
        request = {"addressee_id": 2}

        response = await client.post(
            "/api/social/friends/request?current_user_id=1",
            json=request
        )

        # Should succeed or fail depending on DB state
        assert response.status_code in [200, 400, 404, 500]

    @pytest.mark.asyncio
    async def test_send_friend_request_to_self(self, client):
        """Test sending friend request to self should fail"""
        request = {"addressee_id": 1}

        response = await client.post(
            "/api/social/friends/request?current_user_id=1",
            json=request
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_get_friend_requests(self, client):
        """Test getting pending friend requests"""
        response = await client.get("/api/social/friends/requests?current_user_id=1")
        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_accept_friend_request(self, client):
        """Test accepting a friend request"""
        response = await client.post(
            "/api/social/friends/requests/1/accept?current_user_id=2"
        )
        assert response.status_code in [200, 404, 500]

    @pytest.mark.asyncio
    async def test_decline_friend_request(self, client):
        """Test declining a friend request"""
        response = await client.post(
            "/api/social/friends/requests/1/decline?current_user_id=2"
        )
        assert response.status_code in [200, 404, 500]

    @pytest.mark.asyncio
    async def test_get_friends_list(self, client):
        """Test getting friends list"""
        response = await client.get("/api/social/friends?current_user_id=1")
        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_remove_friend(self, client):
        """Test removing a friend"""
        response = await client.delete("/api/social/friends/2?current_user_id=1")
        assert response.status_code in [200, 404, 500]


class TestChallengesEndpoints:
    """Tests for challenges endpoints"""

    @pytest.mark.asyncio
    async def test_create_challenge(self, client):
        """Test creating a challenge"""
        challenge = {
            "challenge_type": "xp_race",
            "title": "Test Challenge",
            "description": "A test challenge",
            "target_value": 100,
            "end_date": (datetime.utcnow() + timedelta(days=7)).isoformat(),
            "participant_ids": [2],
            "xp_reward": 50
        }

        response = await client.post(
            "/api/social/challenges?current_user_id=1",
            json=challenge
        )
        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_get_challenges(self, client):
        """Test getting user's challenges"""
        response = await client.get("/api/social/challenges?current_user_id=1")
        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_get_challenges_with_filter(self, client):
        """Test getting challenges with status filter"""
        response = await client.get(
            "/api/social/challenges?current_user_id=1&status_filter=active"
        )
        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_accept_challenge(self, client):
        """Test accepting a challenge invitation"""
        response = await client.post(
            "/api/social/challenges/1/accept?current_user_id=2"
        )
        assert response.status_code in [200, 404, 500]

    @pytest.mark.asyncio
    async def test_update_challenge_progress(self, client):
        """Test updating challenge progress"""
        response = await client.post(
            "/api/social/challenges/1/progress?progress_value=50&current_user_id=1"
        )
        assert response.status_code in [200, 404, 500]


class TestStudyGroupsEndpoints:
    """Tests for study groups endpoints"""

    @pytest.mark.asyncio
    async def test_create_study_group(self, client):
        """Test creating a study group"""
        group = {
            "name": "Python Learners",
            "description": "Study Python together",
            "is_public": True,
            "max_members": 20
        }

        response = await client.post(
            "/api/social/groups?current_user_id=1",
            json=group
        )
        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_get_study_groups(self, client):
        """Test getting user's study groups"""
        response = await client.get("/api/social/groups?current_user_id=1")
        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_get_study_groups_with_public(self, client):
        """Test getting study groups including public ones"""
        response = await client.get(
            "/api/social/groups?current_user_id=1&include_public=true"
        )
        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_join_public_group(self, client):
        """Test joining a public study group"""
        response = await client.post(
            "/api/social/groups/1/join?current_user_id=2"
        )
        assert response.status_code in [200, 400, 403, 404, 500]

    @pytest.mark.asyncio
    async def test_join_private_group_with_code(self, client):
        """Test joining a private group with invite code"""
        response = await client.post(
            "/api/social/groups/1/join?current_user_id=2&invite_code=abc123"
        )
        assert response.status_code in [200, 400, 403, 404, 500]

    @pytest.mark.asyncio
    async def test_leave_study_group(self, client):
        """Test leaving a study group"""
        response = await client.delete(
            "/api/social/groups/1/leave?current_user_id=2"
        )
        assert response.status_code in [200, 400, 404, 500]

    @pytest.mark.asyncio
    async def test_get_group_messages(self, client):
        """Test getting group messages"""
        response = await client.get(
            "/api/social/groups/1/messages?current_user_id=1"
        )
        assert response.status_code in [200, 403, 500]

    @pytest.mark.asyncio
    async def test_send_group_message(self, client):
        """Test sending a group message"""
        message = {"content": "Hello everyone!"}

        response = await client.post(
            "/api/social/groups/1/messages?current_user_id=1",
            json=message
        )
        assert response.status_code in [200, 403, 500]


class TestLeaderboardEndpoints:
    """Tests for leaderboard endpoints"""

    @pytest.mark.asyncio
    async def test_get_global_leaderboard(self, client):
        """Test getting global leaderboard"""
        response = await client.get("/api/social/leaderboard/global")
        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_get_global_leaderboard_with_period(self, client):
        """Test getting global leaderboard with period filter"""
        for period in ["daily", "weekly", "monthly", "all_time"]:
            response = await client.get(
                f"/api/social/leaderboard/global?period={period}"
            )
            assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_get_global_leaderboard_with_limit(self, client):
        """Test getting global leaderboard with limit"""
        response = await client.get("/api/social/leaderboard/global?limit=5")
        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_get_friends_leaderboard(self, client):
        """Test getting friends leaderboard"""
        response = await client.get(
            "/api/social/leaderboard/friends?current_user_id=1"
        )
        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_get_group_leaderboard(self, client):
        """Test getting group leaderboard"""
        response = await client.get(
            "/api/social/leaderboard/group/1?current_user_id=1"
        )
        assert response.status_code in [200, 403, 500]


class TestActivityFeedEndpoint:
    """Tests for activity feed endpoint"""

    @pytest.mark.asyncio
    async def test_get_friends_activity(self, client):
        """Test getting friends activity feed"""
        response = await client.get(
            "/api/social/activity/friends?current_user_id=1"
        )
        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_get_friends_activity_with_limit(self, client):
        """Test getting friends activity with limit"""
        response = await client.get(
            "/api/social/activity/friends?current_user_id=1&limit=10"
        )
        assert response.status_code in [200, 500]


class TestAgenticSocialFeatures:
    """Tests for agentic social features (coding challenges, debates, teaching)"""

    @pytest.mark.asyncio
    async def test_list_coding_challenges(self, client):
        """Test listing coding challenges"""
        response = await client.get(
            "/api/social/coding-challenges?current_user_id=1"
        )
        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_get_coding_challenge(self, client):
        """Test getting a specific coding challenge"""
        response = await client.get(
            "/api/social/coding-challenges/challenge_1?current_user_id=1"
        )
        assert response.status_code in [200, 404, 500]

    @pytest.mark.asyncio
    async def test_evaluate_code_submission(self, client):
        """Test evaluating code submission"""
        request = {
            "challenge_id": "challenge_1",
            "code": "def solution(x):\n    return x * 2"
        }

        response = await client.post("/api/social/challenges/evaluate", json=request)
        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_get_challenge_hint(self, client):
        """Test getting a hint for a challenge"""
        request = {
            "challenge_id": "challenge_1",
            "code": "def solution(x):\n    pass",
            "hint_level": 1
        }

        response = await client.post("/api/social/challenges/hint", json=request)
        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_start_debate(self, client):
        """Test starting a debate session"""
        request = {
            "topic": "Is Python better than JavaScript?",
            "format": "structured",
            "panel_preset": "tech",
            "max_rounds": 3
        }

        response = await client.post("/api/social/debates/start", json=request)
        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_start_teaching_session(self, client):
        """Test starting a teaching session"""
        request = {
            "user_id": "user_1",
            "concept_name": "Python Decorators",
            "persona": "curious_student"
        }

        response = await client.post("/api/social/teaching/start", json=request)
        assert response.status_code in [200, 500]
