"""
Hybrid FSRS + Neural ODE Scheduler.

Combines the reliability of FSRS with the personalization of Neural ODE.
Uses confidence-based control modes to safely transition between algorithms.
"""

import torch
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
import logging
import math

from .model import NeuralMemoryODE

logger = logging.getLogger(__name__)


class ControlMode(str, Enum):
    """
    Control mode determines which algorithm makes scheduling decisions.

    SHADOW: Neural ODE runs in shadow mode - FSRS decides, ODE predictions logged
    HYBRID: Confidence-weighted blend of FSRS and ODE predictions
    ACTIVE: Neural ODE fully controls scheduling (FSRS as safety backup)
    """
    SHADOW = "shadow"
    HYBRID = "hybrid"
    ACTIVE = "active"


@dataclass
class HybridConfig:
    """Configuration for hybrid scheduler."""
    # Thresholds for mode transitions
    confidence_threshold_shadow_to_hybrid: float = 0.5
    confidence_threshold_hybrid_to_active: float = 0.8

    # Minimum reviews required for each mode
    min_reviews_for_hybrid: int = 20
    min_reviews_for_active: int = 50

    # Blending weights for HYBRID mode
    # Final interval = fsrs_weight * FSRS + (1 - fsrs_weight) * ODE
    fsrs_weight_in_hybrid: float = 0.3

    # Safety bounds
    min_interval_hours: float = 1.0  # 1 hour minimum
    max_interval_hours: float = 365 * 24  # 1 year maximum

    # Confidence update parameters
    confidence_learning_rate: float = 0.1  # How fast confidence updates
    confidence_decay_on_error: float = 0.15  # Confidence drop on prediction error

    # Target retention for scheduling
    target_retention: float = 0.9


@dataclass
class ODEState:
    """
    State tracking for Neural ODE on a specific user-concept pair.

    Maps to ODECardState database model.
    """
    user_id: int
    concept_id: int
    latent_state: torch.Tensor  # [32] dimensional
    last_state_time: datetime
    control_mode: ControlMode = ControlMode.SHADOW
    ode_confidence: float = 0.0
    review_count: int = 0
    prediction_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'user_id': self.user_id,
            'concept_id': self.concept_id,
            'current_latent_state': self.latent_state.tolist(),
            'last_state_time': self.last_state_time.isoformat(),
            'control_mode': self.control_mode.value,
            'ode_confidence': self.ode_confidence,
            'review_count': self.review_count,
        }


@dataclass
class SchedulingResult:
    """Result of scheduling decision."""
    interval_hours: float
    interval_days: float
    next_review: datetime
    control_mode: ControlMode
    ode_prediction: Optional[float]  # ODE interval in hours
    fsrs_prediction: Optional[float]  # FSRS interval in hours
    ode_confidence: float
    blend_weight: float  # Weight given to ODE (0 = pure FSRS, 1 = pure ODE)
    predicted_retention: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class HybridFSRSScheduler:
    """
    Hybrid scheduler combining FSRS reliability with Neural ODE personalization.

    Control Modes:
    - SHADOW: FSRS decides, ODE runs in background for comparison
    - HYBRID: Weighted blend of FSRS and ODE based on confidence
    - ACTIVE: ODE controls scheduling with FSRS safety bounds

    Transition Logic:
    - SHADOW → HYBRID: When review_count >= 20 AND ode_confidence >= 0.5
    - HYBRID → ACTIVE: When review_count >= 50 AND ode_confidence >= 0.8
    - Any → SHADOW: When ode_confidence drops below 0.3 (safety fallback)
    """

    def __init__(
        self,
        ode_model: NeuralMemoryODE,
        config: Optional[HybridConfig] = None,
        device: str = "cpu",
    ):
        """
        Initialize hybrid scheduler.

        Args:
            ode_model: Trained NeuralMemoryODE model
            config: HybridConfig with thresholds and parameters
            device: Device for model inference
        """
        self.ode_model = ode_model.to(device)
        self.ode_model.eval()
        self.config = config or HybridConfig()
        self.device = device

    def determine_control_mode(self, ode_state: ODEState) -> ControlMode:
        """
        Determine appropriate control mode based on state.

        Args:
            ode_state: Current ODE state for user-concept pair

        Returns:
            ControlMode to use for scheduling
        """
        review_count = ode_state.review_count
        confidence = ode_state.ode_confidence

        # Safety fallback: Low confidence → SHADOW
        if confidence < 0.3:
            return ControlMode.SHADOW

        # Check for ACTIVE mode
        if (review_count >= self.config.min_reviews_for_active and
                confidence >= self.config.confidence_threshold_hybrid_to_active):
            return ControlMode.ACTIVE

        # Check for HYBRID mode
        if (review_count >= self.config.min_reviews_for_hybrid and
                confidence >= self.config.confidence_threshold_shadow_to_hybrid):
            return ControlMode.HYBRID

        # Default to SHADOW
        return ControlMode.SHADOW

    def schedule_review(
        self,
        ode_state: ODEState,
        fsrs_interval_days: float,
        card_features: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
        current_time: Optional[datetime] = None,
    ) -> SchedulingResult:
        """
        Schedule next review using hybrid approach.

        Args:
            ode_state: Current ODE state for this card
            fsrs_interval_days: Interval recommended by FSRS (days)
            card_features: Card feature vector [64]
            user_features: User feature vector [16] or None
            current_time: Current time (defaults to now)

        Returns:
            SchedulingResult with interval and metadata
        """
        if current_time is None:
            current_time = datetime.utcnow()

        fsrs_interval_hours = fsrs_interval_days * 24

        # Get ODE prediction
        with torch.no_grad():
            ode_interval_hours, ode_retention = self._get_ode_prediction(
                ode_state, card_features, user_features
            )

        # Determine control mode
        control_mode = self.determine_control_mode(ode_state)

        # Calculate final interval based on mode
        if control_mode == ControlMode.SHADOW:
            final_interval_hours, blend_weight = self._shadow_mode(
                ode_interval_hours, fsrs_interval_hours
            )
        elif control_mode == ControlMode.HYBRID:
            final_interval_hours, blend_weight = self._hybrid_mode(
                ode_interval_hours, fsrs_interval_hours, ode_state.ode_confidence
            )
        else:  # ACTIVE
            final_interval_hours, blend_weight = self._active_mode(
                ode_interval_hours, fsrs_interval_hours
            )

        # Apply safety bounds
        final_interval_hours = self._apply_bounds(final_interval_hours)
        final_interval_days = final_interval_hours / 24.0

        # Calculate next review time
        next_review = current_time + timedelta(hours=final_interval_hours)

        return SchedulingResult(
            interval_hours=final_interval_hours,
            interval_days=final_interval_days,
            next_review=next_review,
            control_mode=control_mode,
            ode_prediction=ode_interval_hours,
            fsrs_prediction=fsrs_interval_hours,
            ode_confidence=ode_state.ode_confidence,
            blend_weight=blend_weight,
            predicted_retention=ode_retention,
            metadata={
                'review_count': ode_state.review_count,
                'mode_thresholds': {
                    'hybrid_reviews': self.config.min_reviews_for_hybrid,
                    'active_reviews': self.config.min_reviews_for_active,
                    'hybrid_confidence': self.config.confidence_threshold_shadow_to_hybrid,
                    'active_confidence': self.config.confidence_threshold_hybrid_to_active,
                },
            }
        )

    def _get_ode_prediction(
        self,
        ode_state: ODEState,
        card_features: torch.Tensor,
        user_features: Optional[torch.Tensor],
    ) -> Tuple[float, float]:
        """
        Get interval prediction from Neural ODE.

        Returns:
            (interval_hours, predicted_retention_at_interval)
        """
        # Ensure tensors are on correct device
        h = ode_state.latent_state.to(self.device).unsqueeze(0)  # [1, 32]

        # Predict optimal interval using binary search
        interval_tensor = self.ode_model.predict_next_review(
            h,
            target_retention=self.config.target_retention,
            max_interval=self.config.max_interval_hours,
        )

        interval_hours = interval_tensor.squeeze().item()

        # Get predicted retention at this interval
        t_eval = torch.tensor([0.0, interval_hours], device=self.device)
        retention = self.ode_model.predict_retention(h, t_eval)
        predicted_retention = retention[-1, 0, 0].item()

        return interval_hours, predicted_retention

    def _shadow_mode(
        self,
        ode_interval: float,
        fsrs_interval: float,
    ) -> Tuple[float, float]:
        """
        Shadow mode: FSRS decides, ODE logged for comparison.

        Returns:
            (final_interval, blend_weight=0.0)
        """
        # Log comparison for analysis
        logger.debug(
            f"SHADOW mode - FSRS: {fsrs_interval:.1f}h, ODE: {ode_interval:.1f}h, "
            f"diff: {abs(ode_interval - fsrs_interval):.1f}h"
        )
        return fsrs_interval, 0.0

    def _hybrid_mode(
        self,
        ode_interval: float,
        fsrs_interval: float,
        confidence: float,
    ) -> Tuple[float, float]:
        """
        Hybrid mode: Confidence-weighted blend.

        Higher confidence → more weight to ODE.

        Returns:
            (final_interval, blend_weight)
        """
        # Scale confidence to blend weight
        # At confidence 0.5: fsrs_weight = 0.3 (70% FSRS, 30% ODE)
        # At confidence 0.8: fsrs_weight approaches 0.1 (more ODE)
        base_fsrs_weight = self.config.fsrs_weight_in_hybrid
        confidence_factor = (confidence - 0.5) / 0.3  # 0 at 0.5, 1 at 0.8
        confidence_factor = max(0, min(1, confidence_factor))

        fsrs_weight = base_fsrs_weight * (1 - 0.5 * confidence_factor)
        ode_weight = 1 - fsrs_weight

        # Blend intervals
        final_interval = fsrs_weight * fsrs_interval + ode_weight * ode_interval

        logger.debug(
            f"HYBRID mode - FSRS: {fsrs_interval:.1f}h ({fsrs_weight:.2f}), "
            f"ODE: {ode_interval:.1f}h ({ode_weight:.2f}), "
            f"Final: {final_interval:.1f}h"
        )

        return final_interval, ode_weight

    def _active_mode(
        self,
        ode_interval: float,
        fsrs_interval: float,
    ) -> Tuple[float, float]:
        """
        Active mode: ODE controls with FSRS safety bounds.

        FSRS provides reasonable bounds, ODE decides within bounds.

        Returns:
            (final_interval, blend_weight=1.0)
        """
        # Use FSRS as soft bounds (allow 50% deviation)
        min_bound = fsrs_interval * 0.5
        max_bound = fsrs_interval * 2.0

        # Clamp ODE to reasonable range relative to FSRS
        final_interval = max(min_bound, min(max_bound, ode_interval))

        if final_interval != ode_interval:
            logger.debug(
                f"ACTIVE mode - ODE clamped from {ode_interval:.1f}h to {final_interval:.1f}h "
                f"(FSRS bounds: {min_bound:.1f}-{max_bound:.1f}h)"
            )
        else:
            logger.debug(f"ACTIVE mode - Using ODE interval: {ode_interval:.1f}h")

        return final_interval, 1.0

    def _apply_bounds(self, interval_hours: float) -> float:
        """Apply safety bounds to interval."""
        return max(
            self.config.min_interval_hours,
            min(self.config.max_interval_hours, interval_hours)
        )

    def update_confidence(
        self,
        ode_state: ODEState,
        actual_recalled: bool,
        predicted_retention: float,
    ) -> float:
        """
        Update ODE confidence based on prediction accuracy.

        Args:
            ode_state: Current ODE state
            actual_recalled: Whether user actually recalled (grade >= 2)
            predicted_retention: Model's predicted retention probability

        Returns:
            Updated confidence value
        """
        # Calculate prediction error
        actual = 1.0 if actual_recalled else 0.0
        error = abs(actual - predicted_retention)

        # Update confidence using exponential moving average
        if actual_recalled and predicted_retention >= 0.5:
            # Correct prediction of recall
            confidence_delta = self.config.confidence_learning_rate * (1 - error)
        elif not actual_recalled and predicted_retention < 0.5:
            # Correct prediction of forgetting
            confidence_delta = self.config.confidence_learning_rate * (1 - error)
        else:
            # Incorrect prediction
            confidence_delta = -self.config.confidence_decay_on_error * error

        new_confidence = ode_state.ode_confidence + confidence_delta
        new_confidence = max(0.0, min(1.0, new_confidence))

        logger.debug(
            f"Confidence update: {ode_state.ode_confidence:.3f} → {new_confidence:.3f} "
            f"(predicted={predicted_retention:.3f}, actual={'recall' if actual_recalled else 'forgot'})"
        )

        return new_confidence

    def process_review_outcome(
        self,
        ode_state: ODEState,
        grade: int,
        telemetry: Optional[torch.Tensor],
        review_time: Optional[datetime] = None,
    ) -> ODEState:
        """
        Process a review outcome and update ODE state.

        Args:
            ode_state: Current ODE state
            grade: Review grade (1-4)
            telemetry: Telemetry tensor [4] or None
            review_time: Time of review (defaults to now)

        Returns:
            Updated ODEState
        """
        if review_time is None:
            review_time = datetime.utcnow()

        # Get current state
        h = ode_state.latent_state.to(self.device).unsqueeze(0)

        # Time since last state
        time_delta = review_time - ode_state.last_state_time
        hours_elapsed = time_delta.total_seconds() / 3600.0

        # Evolve state to review time
        if hours_elapsed > 0:
            h = self.ode_model._integrate(h, 0.0, hours_elapsed)

        # Get predicted retention before jump
        retention_before = torch.sigmoid(self.ode_model.readout(h)).squeeze().item()

        # Apply jump for review
        grade_tensor = torch.tensor([grade], device=self.device)
        h = self.ode_model.apply_jump(h, grade_tensor, telemetry)

        # Update confidence based on outcome
        actual_recalled = grade >= 2
        new_confidence = self.update_confidence(
            ode_state, actual_recalled, retention_before
        )

        # Determine new control mode
        new_review_count = ode_state.review_count + 1

        # Create updated state
        updated_state = ODEState(
            user_id=ode_state.user_id,
            concept_id=ode_state.concept_id,
            latent_state=h.squeeze(0).cpu(),
            last_state_time=review_time,
            control_mode=ode_state.control_mode,  # Will be recalculated on next schedule
            ode_confidence=new_confidence,
            review_count=new_review_count,
            prediction_history=ode_state.prediction_history + [{
                'time': review_time.isoformat(),
                'predicted_retention': retention_before,
                'actual_recalled': actual_recalled,
                'grade': grade,
                'confidence_after': new_confidence,
            }],
        )

        # Update control mode
        updated_state.control_mode = self.determine_control_mode(updated_state)

        return updated_state

    def initialize_state(
        self,
        user_id: int,
        concept_id: int,
        card_features: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
        phenotype_id: Optional[int] = None,
    ) -> ODEState:
        """
        Initialize ODE state for a new user-concept pair.

        Args:
            user_id: User ID
            concept_id: Concept ID
            card_features: Card feature vector [64]
            user_features: User feature vector [16] or None
            phenotype_id: Learner phenotype for cold-start

        Returns:
            Initial ODEState
        """
        # Ensure tensors on device
        card_features = card_features.to(self.device).unsqueeze(0)
        if user_features is not None:
            user_features = user_features.to(self.device).unsqueeze(0)

        phenotype_tensor = None
        if phenotype_id is not None:
            phenotype_tensor = torch.tensor([phenotype_id], device=self.device)

        # Get initial state from encoder
        with torch.no_grad():
            h0 = self.ode_model.encode_initial_state(
                card_features, user_features, phenotype_tensor
            )

        return ODEState(
            user_id=user_id,
            concept_id=concept_id,
            latent_state=h0.squeeze(0).cpu(),
            last_state_time=datetime.utcnow(),
            control_mode=ControlMode.SHADOW,
            ode_confidence=0.0,
            review_count=0,
        )
