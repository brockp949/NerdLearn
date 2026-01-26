"""
Unit tests for Hybrid FSRS/Neural ODE Scheduler.

Tests control mode transitions, interval blending, and fallback behavior.
"""

import pytest
import torch
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from app.adaptive.neural_ode.hybrid_scheduler import (
    HybridFSRSScheduler,
    HybridConfig,
    ODEState,
    ControlMode,
    SchedulingResult,
)


@pytest.fixture
def mock_ode_model():
    """Create a mock ODE model for testing."""
    model = MagicMock()
    model.to = MagicMock(return_value=model)
    model.eval = MagicMock()
    model.forward = MagicMock(return_value=torch.randn(1, 10, 32))
    model.readout = MagicMock(return_value=torch.tensor([[0.85]]))  # Mock retention prediction
    model.jump = MagicMock(return_value=torch.randn(1, 32))
    return model


@pytest.fixture
def scheduler(mock_ode_model):
    """Create a scheduler with mock ODE model."""
    config = HybridConfig()
    return HybridFSRSScheduler(ode_model=mock_ode_model, config=config)


def create_ode_state(
    review_count: int = 10,
    ode_confidence: float = 0.5,
    user_id: int = 1,
    concept_id: int = 1,
):
    """Helper to create ODEState instances."""
    return ODEState(
        user_id=user_id,
        concept_id=concept_id,
        latent_state=torch.randn(32),
        last_state_time=datetime.utcnow(),
        control_mode=ControlMode.SHADOW,
        ode_confidence=ode_confidence,
        review_count=review_count,
    )


class TestControlModeTransitions:
    """Tests for control mode determination logic."""

    def test_shadow_mode_low_reviews(self, scheduler):
        """Shadow mode when review count < 20."""
        ode_state = create_ode_state(
            review_count=10,  # < 20
            ode_confidence=0.9,  # High confidence shouldn't matter
        )

        mode = scheduler.determine_control_mode(ode_state)
        assert mode == ControlMode.SHADOW

    def test_shadow_mode_low_confidence(self, scheduler):
        """Shadow mode when confidence < 0.5."""
        ode_state = create_ode_state(
            review_count=100,  # High review count
            ode_confidence=0.3,  # < 0.5
        )

        mode = scheduler.determine_control_mode(ode_state)
        assert mode == ControlMode.SHADOW

    def test_hybrid_mode_moderate_reviews_and_confidence(self, scheduler):
        """Hybrid mode when 20 <= reviews < 50 and confidence >= 0.5."""
        ode_state = create_ode_state(
            review_count=30,  # 20 <= x < 50
            ode_confidence=0.6,  # >= 0.5
        )

        mode = scheduler.determine_control_mode(ode_state)
        assert mode == ControlMode.HYBRID

    def test_active_mode_high_reviews_and_confidence(self, scheduler):
        """Active mode when reviews >= 50 and confidence >= 0.8."""
        ode_state = create_ode_state(
            review_count=60,  # >= 50
            ode_confidence=0.85,  # >= 0.8
        )

        mode = scheduler.determine_control_mode(ode_state)
        assert mode == ControlMode.ACTIVE

    def test_hybrid_mode_high_reviews_moderate_confidence(self, scheduler):
        """Hybrid mode when reviews >= 50 but confidence < 0.8."""
        ode_state = create_ode_state(
            review_count=60,  # >= 50
            ode_confidence=0.7,  # 0.5 <= x < 0.8
        )

        mode = scheduler.determine_control_mode(ode_state)
        assert mode == ControlMode.HYBRID


class TestSchedulingDecisions:
    """Tests for scheduling decisions."""

    def test_schedule_returns_result(self, scheduler, mock_ode_model):
        """Scheduler should return a SchedulingResult."""
        ode_state = create_ode_state(review_count=5, ode_confidence=0.2)
        card_features = torch.randn(64)

        # Mock predict_next_review to return an interval
        mock_ode_model.predict_next_review = MagicMock(return_value=torch.tensor([24.0]))
        mock_ode_model.predict_retention = MagicMock(return_value=torch.tensor([[[0.85]]]))

        result = scheduler.schedule_review(
            ode_state=ode_state,
            fsrs_interval_days=1.0,  # 24 hours
            card_features=card_features,
        )

        assert isinstance(result, SchedulingResult)
        assert result.interval_hours > 0
        assert result.control_mode in ControlMode

    def test_shadow_mode_follows_fsrs(self, scheduler, mock_ode_model):
        """In shadow mode, interval should follow FSRS."""
        ode_state = create_ode_state(review_count=5, ode_confidence=0.2)
        card_features = torch.randn(64)

        # Mock ODE predictions
        mock_ode_model.predict_next_review = MagicMock(return_value=torch.tensor([48.0]))
        mock_ode_model.predict_retention = MagicMock(return_value=torch.tensor([[[0.85]]]))

        result = scheduler.schedule_review(
            ode_state=ode_state,
            fsrs_interval_days=3.0,  # 72 hours
            card_features=card_features,
        )

        assert result.control_mode == ControlMode.SHADOW
        # In shadow mode, should use FSRS interval
        assert result.interval_hours == pytest.approx(72.0, rel=0.1)
        assert result.blend_weight == pytest.approx(0.0)

    def test_result_has_next_review_time(self, scheduler, mock_ode_model):
        """Result should include next review datetime."""
        ode_state = create_ode_state(review_count=5, ode_confidence=0.2)
        card_features = torch.randn(64)

        mock_ode_model.predict_next_review = MagicMock(return_value=torch.tensor([24.0]))
        mock_ode_model.predict_retention = MagicMock(return_value=torch.tensor([[[0.85]]]))

        result = scheduler.schedule_review(
            ode_state=ode_state,
            fsrs_interval_days=1.0,
            card_features=card_features,
        )

        assert result.next_review > datetime.utcnow()
        expected_next = datetime.utcnow() + timedelta(hours=result.interval_hours)
        # Allow 1 second tolerance
        assert abs((result.next_review - expected_next).total_seconds()) < 1


class TestSafetyBounds:
    """Tests for safety bounds on intervals."""

    def test_interval_minimum_enforced(self, scheduler, mock_ode_model):
        """Interval should not go below minimum."""
        ode_state = create_ode_state(review_count=5, ode_confidence=0.2)
        card_features = torch.randn(64)

        mock_ode_model.predict_next_review = MagicMock(return_value=torch.tensor([0.1]))
        mock_ode_model.predict_retention = MagicMock(return_value=torch.tensor([[[0.85]]]))

        result = scheduler.schedule_review(
            ode_state=ode_state,
            fsrs_interval_days=0.004,  # ~0.1 hours, very short
            card_features=card_features,
        )

        assert result.interval_hours >= scheduler.config.min_interval_hours

    def test_interval_maximum_enforced(self, scheduler, mock_ode_model):
        """Interval should not exceed maximum."""
        ode_state = create_ode_state(review_count=5, ode_confidence=0.2)
        card_features = torch.randn(64)

        mock_ode_model.predict_next_review = MagicMock(return_value=torch.tensor([10000.0]))
        mock_ode_model.predict_retention = MagicMock(return_value=torch.tensor([[[0.85]]]))

        result = scheduler.schedule_review(
            ode_state=ode_state,
            fsrs_interval_days=500.0,  # ~12000 hours, very long
            card_features=card_features,
        )

        assert result.interval_hours <= scheduler.config.max_interval_hours


class TestConfidenceUpdates:
    """Tests for ODE confidence updates."""

    def test_update_on_correct_prediction(self, scheduler):
        """Confidence should update based on prediction accuracy."""
        ode_state = create_ode_state(review_count=30, ode_confidence=0.6)

        # Record a prediction
        predicted_retention = 0.85
        actual_recalled = True

        new_confidence = scheduler.update_confidence(
            ode_state=ode_state,
            predicted_retention=predicted_retention,
            actual_recalled=actual_recalled,
        )

        # Good prediction should maintain or increase confidence
        assert new_confidence >= 0.0
        assert new_confidence <= 1.0

    def test_update_on_wrong_prediction(self, scheduler):
        """Confidence should decrease when ODE prediction is wrong."""
        ode_state = create_ode_state(review_count=30, ode_confidence=0.7)

        # ODE predicted high retention, but user forgot
        new_confidence = scheduler.update_confidence(
            ode_state=ode_state,
            predicted_retention=0.9,
            actual_recalled=False,
        )

        # Bad prediction should decrease confidence
        assert new_confidence < 0.7

    def test_confidence_stays_bounded(self, scheduler):
        """Confidence should always be in [0, 1]."""
        ode_state = create_ode_state(review_count=30, ode_confidence=0.99)

        new_confidence = scheduler.update_confidence(
            ode_state=ode_state,
            predicted_retention=0.99,
            actual_recalled=True,
        )

        assert 0.0 <= new_confidence <= 1.0


class TestODEStateManagement:
    """Tests for ODEState dataclass."""

    def test_ode_state_creation(self):
        """ODEState should be created with required fields."""
        state = ODEState(
            user_id=1,
            concept_id=2,
            latent_state=torch.randn(32),
            last_state_time=datetime.utcnow(),
        )

        assert state.user_id == 1
        assert state.concept_id == 2
        assert state.latent_state.shape == (32,)
        assert state.control_mode == ControlMode.SHADOW
        assert state.ode_confidence == 0.0
        assert state.review_count == 0

    def test_ode_state_to_dict(self):
        """ODEState should serialize to dict."""
        state = create_ode_state()
        state_dict = state.to_dict()

        assert 'user_id' in state_dict
        assert 'concept_id' in state_dict
        assert 'current_latent_state' in state_dict
        assert 'control_mode' in state_dict
        assert 'ode_confidence' in state_dict


class TestHybridConfig:
    """Tests for HybridConfig."""

    def test_default_config(self):
        """Default config should have reasonable values."""
        config = HybridConfig()

        assert config.confidence_threshold_shadow_to_hybrid == 0.5
        assert config.confidence_threshold_hybrid_to_active == 0.8
        assert config.min_reviews_for_hybrid == 20
        assert config.min_reviews_for_active == 50
        assert config.min_interval_hours == 1.0
        assert config.target_retention == 0.9

    def test_custom_config(self):
        """Custom config values should be respected."""
        config = HybridConfig(
            confidence_threshold_shadow_to_hybrid=0.6,
            min_reviews_for_hybrid=30,
            target_retention=0.85,
        )

        assert config.confidence_threshold_shadow_to_hybrid == 0.6
        assert config.min_reviews_for_hybrid == 30
        assert config.target_retention == 0.85


class TestSchedulingResult:
    """Tests for SchedulingResult structure."""

    def test_scheduling_result_creation(self):
        """SchedulingResult should have all fields."""
        result = SchedulingResult(
            interval_hours=24.0,
            interval_days=1.0,
            next_review=datetime.utcnow() + timedelta(days=1),
            control_mode=ControlMode.SHADOW,
            ode_prediction=20.0,
            fsrs_prediction=24.0,
            ode_confidence=0.5,
            blend_weight=0.0,
            predicted_retention=0.9,
        )

        assert result.interval_hours == 24.0
        assert result.interval_days == 1.0
        assert result.control_mode == ControlMode.SHADOW
        assert result.blend_weight == 0.0


class TestEdgeCases:
    """Edge case tests for the hybrid scheduler."""

    def test_zero_review_count(self, scheduler, mock_ode_model):
        """Should handle zero review count gracefully."""
        ode_state = create_ode_state(review_count=0, ode_confidence=0.0)
        card_features = torch.randn(64)

        mock_ode_model.predict_next_review = MagicMock(return_value=torch.tensor([24.0]))
        mock_ode_model.predict_retention = MagicMock(return_value=torch.tensor([[[0.85]]]))

        result = scheduler.schedule_review(
            ode_state=ode_state,
            fsrs_interval_days=1.0,
            card_features=card_features,
        )

        assert result.control_mode == ControlMode.SHADOW
        assert result.interval_hours > 0

    def test_very_old_last_update(self, scheduler, mock_ode_model):
        """Should handle very old last update time."""
        ode_state = ODEState(
            user_id=1,
            concept_id=1,
            latent_state=torch.randn(32),
            last_state_time=datetime.utcnow() - timedelta(days=365),  # 1 year ago
            ode_confidence=0.8,
            review_count=50,
        )
        card_features = torch.randn(64)

        mock_ode_model.predict_next_review = MagicMock(return_value=torch.tensor([72.0]))
        mock_ode_model.predict_retention = MagicMock(return_value=torch.tensor([[[0.85]]]))

        result = scheduler.schedule_review(
            ode_state=ode_state,
            fsrs_interval_days=3.0,
            card_features=card_features,
        )

        # Should still produce valid result
        assert result.interval_hours > 0
        assert result.interval_hours <= scheduler.config.max_interval_hours

    def test_negative_fsrs_interval_clamped(self, scheduler, mock_ode_model):
        """Negative FSRS interval should be clamped to minimum."""
        ode_state = create_ode_state(review_count=5, ode_confidence=0.2)
        card_features = torch.randn(64)

        mock_ode_model.predict_next_review = MagicMock(return_value=torch.tensor([24.0]))
        mock_ode_model.predict_retention = MagicMock(return_value=torch.tensor([[[0.85]]]))

        result = scheduler.schedule_review(
            ode_state=ode_state,
            fsrs_interval_days=-0.5,  # Invalid negative
            card_features=card_features,
        )

        assert result.interval_hours >= scheduler.config.min_interval_hours
