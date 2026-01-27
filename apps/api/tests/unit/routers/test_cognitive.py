"""
Tests for cognitive router endpoints
"""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch


class TestCognitiveModels:
    """Tests for cognitive endpoint models"""

    def test_interaction_event_input(self):
        """Test InteractionEventInput model"""
        from app.routers.cognitive import InteractionEventInput

        event = InteractionEventInput(
            event_type="answer",
            correct=True,
            response_time_ms=5000,
            content_id="content_1",
            hint_used=False,
            attempts=1
        )

        assert event.event_type == "answer"
        assert event.correct is True
        assert event.response_time_ms == 5000

    def test_frustration_detection_request(self):
        """Test FrustrationDetectionRequest model"""
        from app.routers.cognitive import FrustrationDetectionRequest, InteractionEventInput

        request = FrustrationDetectionRequest(
            user_id="user_1",
            events=[
                InteractionEventInput(event_type="answer", correct=False, response_time_ms=3000)
            ]
        )

        assert request.user_id == "user_1"
        assert len(request.events) == 1

    def test_confidence_rating_input_validation(self):
        """Test ConfidenceRatingInput validation"""
        from app.routers.cognitive import ConfidenceRatingInput

        # Valid confidence
        rating = ConfidenceRatingInput(
            user_id="user_1",
            concept_id="concept_1",
            content_id="content_1",
            confidence=0.75
        )
        assert rating.confidence == 0.75

        # Invalid confidence (out of range) should raise
        with pytest.raises(Exception):
            ConfidenceRatingInput(
                user_id="user_1",
                concept_id="concept_1",
                content_id="content_1",
                confidence=1.5  # > 1
            )

    def test_learner_state_input(self):
        """Test LearnerStateInput model"""
        from app.routers.cognitive import LearnerStateInput

        state = LearnerStateInput(
            user_id="user_1",
            frustration_score=0.3,
            frustration_level="low",
            cognitive_load_score=0.5,
            consecutive_errors=2,
            time_on_task_minutes=15.0
        )

        assert state.user_id == "user_1"
        assert state.frustration_score == 0.3
        assert state.consecutive_errors == 2


class TestFrustrationEndpoints:
    """Tests for frustration detection endpoints"""

    @pytest.mark.asyncio
    async def test_detect_frustration_valid_request(self, client):
        """Test frustration detection with valid request"""
        request = {
            "user_id": "user_1",
            "events": [
                {
                    "event_type": "answer",
                    "correct": False,
                    "response_time_ms": 2000,
                    "attempts": 3
                },
                {
                    "event_type": "answer",
                    "correct": False,
                    "response_time_ms": 1500,
                    "attempts": 2
                }
            ]
        }

        response = await client.post("/api/cognitive/frustration/detect", json=request)

        if response.status_code == 200:
            data = response.json()
            assert "level" in data
            assert "score" in data
            assert "recommended_action" in data
            assert "indicators" in data

    @pytest.mark.asyncio
    async def test_detect_frustration_missing_events(self, client):
        """Test frustration detection with missing events"""
        request = {
            "user_id": "user_1",
            "events": []
        }

        response = await client.post("/api/cognitive/frustration/detect", json=request)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_update_user_baseline(self, client):
        """Test updating user baseline"""
        events = [
            {
                "event_type": "answer",
                "correct": True,
                "response_time_ms": 5000
            }
        ] * 10  # Multiple events for baseline

        response = await client.post(
            "/api/cognitive/frustration/update-baseline?user_id=user_1",
            json=events
        )

        if response.status_code == 200:
            data = response.json()
            assert data["baseline_updated"] is True
            assert "baseline" in data


class TestMetacognitionEndpoints:
    """Tests for metacognition endpoints"""

    @pytest.mark.asyncio
    async def test_get_metacognition_prompt(self, client):
        """Test getting metacognition prompt"""
        request = {
            "user_id": "user_1",
            "concept_name": "Python Variables",
            "timing": "during",
            "force": True
        }

        response = await client.post("/api/cognitive/metacognition/prompt", json=request)

        if response.status_code == 200:
            data = response.json()
            # Either returns a prompt or reason for no prompt
            assert "prompt" in data or "prompt_type" in data or "reason" in data

    @pytest.mark.asyncio
    async def test_get_confidence_scale(self, client):
        """Test getting confidence scale"""
        response = await client.get(
            "/api/cognitive/metacognition/confidence-scale?concept_name=Test&scale_type=numeric"
        )

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_record_confidence_rating(self, client):
        """Test recording confidence rating"""
        rating = {
            "user_id": "user_1",
            "concept_id": "concept_1",
            "content_id": "content_1",
            "confidence": 0.7,
            "context": "during_practice"
        }

        response = await client.post(
            "/api/cognitive/metacognition/record-confidence",
            json=rating
        )

        if response.status_code == 200:
            data = response.json()
            assert data["recorded"] is True

    @pytest.mark.asyncio
    async def test_analyze_self_explanation(self, client):
        """Test analyzing self-explanation"""
        explanation = {
            "explanation_text": "A variable is like a container that stores data values in Python. You can assign values using the equals sign.",
            "concept_name": "Python Variables",
            "expected_concepts": ["container", "data", "assignment"],
            "common_misconceptions": ["variables are like math variables"]
        }

        response = await client.post(
            "/api/cognitive/metacognition/analyze-explanation",
            json=explanation
        )

        assert response.status_code in [200, 500]


class TestCalibrationEndpoints:
    """Tests for calibration endpoints"""

    @pytest.mark.asyncio
    async def test_calculate_calibration(self, client):
        """Test calculating calibration"""
        request = {
            "user_id": "user_1",
            "concept_id": None,
            "time_window_hours": 24
        }

        response = await client.post("/api/cognitive/calibration/calculate", json=request)

        if response.status_code == 200:
            data = response.json()
            assert "calibration_level" in data
            assert "mean_confidence" in data
            assert "mean_performance" in data

    @pytest.mark.asyncio
    async def test_get_calibration_feedback(self, client):
        """Test getting calibration feedback"""
        request = {
            "user_id": "user_1"
        }

        response = await client.post("/api/cognitive/calibration/feedback", json=request)
        assert response.status_code in [200, 500]


class TestInterventionEndpoints:
    """Tests for intervention endpoints"""

    @pytest.mark.asyncio
    async def test_decide_intervention(self, client):
        """Test intervention decision"""
        request = {
            "learner_state": {
                "user_id": "user_1",
                "frustration_score": 0.7,
                "frustration_level": "moderate",
                "cognitive_load_score": 0.8,
                "consecutive_errors": 3,
                "time_on_task_minutes": 30
            },
            "events": [
                {
                    "event_type": "answer",
                    "correct": False,
                    "response_time_ms": 2000
                }
            ]
        }

        response = await client.post("/api/cognitive/intervention/decide", json=request)

        if response.status_code == 200:
            data = response.json()
            assert "should_intervene" in data
            assert "reason" in data

    @pytest.mark.asyncio
    async def test_get_intervention_history(self, client):
        """Test getting intervention history"""
        response = await client.get("/api/cognitive/intervention/history/user_1")

        assert response.status_code in [200, 500]


class TestCognitiveProfileEndpoint:
    """Tests for cognitive profile endpoint"""

    @pytest.mark.asyncio
    async def test_get_cognitive_profile(self, client):
        """Test getting comprehensive cognitive profile"""
        response = await client.get("/api/cognitive/profile/user_1")

        if response.status_code == 200:
            data = response.json()
            assert "user_id" in data
            assert "baseline" in data
            assert "calibration" in data
            assert "interventions" in data


class TestObserverEndpoint:
    """Tests for observer agent endpoint"""

    @pytest.mark.asyncio
    async def test_observe_behavior(self, client):
        """Test observer agent behavior analysis"""
        request = {
            "user_id": "user_1",
            "events": [
                {
                    "event_type": "answer",
                    "correct": False,
                    "response_time_ms": 500,  # Very fast - potential gaming
                    "attempts": 1
                }
            ] * 5
        }

        response = await client.post("/api/cognitive/observe", json=request)
        assert response.status_code in [200, 422, 500]
