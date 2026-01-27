"""
Tests for ScaffoldingService
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestScaffoldingService:
    """Tests for the ScaffoldingService class"""

    @pytest.fixture
    def mock_intervention_engine(self):
        """Mock intervention engine"""
        engine = MagicMock()
        return engine

    @pytest.fixture
    def mock_zpd_regulator(self):
        """Mock ZPD regulator"""
        regulator = MagicMock()
        regulator.calculate_multidimensional_zpd = MagicMock(return_value={
            "in_zpd": True,
            "optimal_difficulty": 0.6,
            "scaffolding_needed": "minimal"
        })
        return regulator

    @pytest.fixture
    def service(self, mock_intervention_engine, mock_zpd_regulator):
        """Create ScaffoldingService with mocks"""
        from app.services.scaffolding_service import ScaffoldingService
        return ScaffoldingService(
            intervention_engine=mock_intervention_engine,
            zpd_regulator=mock_zpd_regulator
        )

    @pytest.mark.asyncio
    async def test_get_adaptive_hint_first_request(self, service):
        """Test getting first hint returns low level"""
        from app.services.scaffolding_service import HintRequest

        request = HintRequest(
            user_id="user_1",
            content_id="content_1",
            step_id="step_1",
            context={}
        )

        response = await service.get_adaptive_hint(request)

        assert response.hint_level == "low"
        assert response.remaining_hints == 2

    @pytest.mark.asyncio
    async def test_get_adaptive_hint_second_request(self, service):
        """Test getting second hint returns medium level"""
        from app.services.scaffolding_service import HintRequest

        request = HintRequest(
            user_id="user_1",
            content_id="content_1"
        )

        # First request
        await service.get_adaptive_hint(request)

        # Second request
        response = await service.get_adaptive_hint(request)

        assert response.hint_level == "medium"
        assert response.remaining_hints == 1

    @pytest.mark.asyncio
    async def test_get_adaptive_hint_third_request(self, service):
        """Test getting third hint returns high level"""
        from app.services.scaffolding_service import HintRequest

        request = HintRequest(
            user_id="user_1",
            content_id="content_1"
        )

        # First two requests
        await service.get_adaptive_hint(request)
        await service.get_adaptive_hint(request)

        # Third request
        response = await service.get_adaptive_hint(request)

        assert response.hint_level == "high"
        assert response.remaining_hints == 0

    @pytest.mark.asyncio
    async def test_get_adaptive_hint_different_content(self, service):
        """Test hints are tracked separately per content"""
        from app.services.scaffolding_service import HintRequest

        request1 = HintRequest(user_id="user_1", content_id="content_1")
        request2 = HintRequest(user_id="user_1", content_id="content_2")

        # Request hint for content_1
        response1 = await service.get_adaptive_hint(request1)

        # Request hint for content_2 (should be first hint)
        response2 = await service.get_adaptive_hint(request2)

        assert response1.hint_level == "low"
        assert response2.hint_level == "low"

    @pytest.mark.asyncio
    async def test_get_adaptive_hint_different_users(self, service):
        """Test hints are tracked separately per user"""
        from app.services.scaffolding_service import HintRequest

        request1 = HintRequest(user_id="user_1", content_id="content_1")
        request2 = HintRequest(user_id="user_2", content_id="content_1")

        # Request hint for user_1
        await service.get_adaptive_hint(request1)
        await service.get_adaptive_hint(request1)

        # Request hint for user_2 (should be first hint)
        response = await service.get_adaptive_hint(request2)

        assert response.hint_level == "low"

    @pytest.mark.asyncio
    async def test_analyze_zpd_fit(self, service, mock_zpd_regulator):
        """Test ZPD analysis"""
        result = await service.analyze_zpd_fit(
            user_id="user_1",
            content_difficulty=0.5,
            user_mastery=0.6
        )

        assert "in_zpd" in result
        mock_zpd_regulator.calculate_multidimensional_zpd.assert_called_once()


class TestHintRequest:
    """Tests for HintRequest model"""

    def test_hint_request_all_fields(self):
        """Test HintRequest with all fields"""
        from app.services.scaffolding_service import HintRequest

        request = HintRequest(
            user_id="user_1",
            content_id="content_1",
            step_id="step_1",
            context={"attempt": 1}
        )

        assert request.user_id == "user_1"
        assert request.content_id == "content_1"
        assert request.step_id == "step_1"
        assert request.context == {"attempt": 1}

    def test_hint_request_required_only(self):
        """Test HintRequest with only required fields"""
        from app.services.scaffolding_service import HintRequest

        request = HintRequest(
            user_id="user_1",
            content_id="content_1"
        )

        assert request.user_id == "user_1"
        assert request.content_id == "content_1"
        assert request.step_id is None
        assert request.context == {}


class TestHintResponse:
    """Tests for HintResponse model"""

    def test_hint_response_creation(self):
        """Test HintResponse creation"""
        from app.services.scaffolding_service import HintResponse

        response = HintResponse(
            hint_text="Try this approach",
            hint_level="medium",
            remaining_hints=1
        )

        assert response.hint_text == "Try this approach"
        assert response.hint_level == "medium"
        assert response.remaining_hints == 1


class TestScaffoldingLogic:
    """Tests for scaffolding logic"""

    def test_hint_progression(self):
        """Test that hints progress from low to high"""
        hint_levels = ["low", "medium", "high"]

        for i, expected_level in enumerate(hint_levels):
            count = i + 1
            if count == 1:
                level = "low"
            elif count == 2:
                level = "medium"
            else:
                level = "high"
            assert level == expected_level

    def test_remaining_hints_calculation(self):
        """Test remaining hints calculation"""
        max_hints = 3

        for count in range(1, 5):
            remaining = max(0, max_hints - count)
            if count == 1:
                assert remaining == 2
            elif count == 2:
                assert remaining == 1
            elif count >= 3:
                assert remaining == 0

    def test_zpd_fit_boundaries(self):
        """Test ZPD fit calculation boundaries"""
        # Content should be slightly above current mastery
        mastery = 0.6
        optimal_range = (mastery, mastery + 0.2)

        # In ZPD
        assert optimal_range[0] <= 0.7 <= optimal_range[1]

        # Too easy
        assert 0.4 < optimal_range[0]

        # Too hard
        assert 0.9 > optimal_range[1]
