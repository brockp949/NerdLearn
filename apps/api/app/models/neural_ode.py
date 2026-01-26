"""
Neural ODE Models
SQLAlchemy models for CT-MCN (Continuous-Time Memory Calibration Network)
"""
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, ForeignKey,
    UniqueConstraint, Index
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class MemoryStateTrajectory(Base):
    """
    Stores h(t) latent state time series for Neural ODE memory model.
    Each record represents a snapshot of the memory state at a specific time.
    """
    __tablename__ = "memory_state_trajectory"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    concept_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    # 32-dimensional latent state vector
    latent_state = Column(JSONB, nullable=False)  # List[float] of length 32

    # Predictions from latent state
    predicted_retrievability = Column(Float, nullable=False)  # P(recall) 0-1

    # Uncertainty estimates (Evidential Deep Learning)
    uncertainty_epistemic = Column(Float)  # Model uncertainty
    uncertainty_aleatoric = Column(Float)  # Data uncertainty

    # Contextual factors at this time
    circadian_factor = Column(Float)  # λ_circ(t) modulation factor
    sleep_consolidated = Column(Boolean, default=False)  # Whether sleep consolidation occurred
    stress_coefficient = Column(Float)  # s(t) stress/load factor

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User")

    __table_args__ = (
        Index('ix_memory_state_trajectory_user_concept', 'user_id', 'concept_id'),
    )


class UserCircadianPattern(Base):
    """
    Per-user circadian rhythm parameters for memory modeling.
    Inferred from activity patterns and performance data.
    """
    __tablename__ = "user_circadian_pattern"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)

    # Circadian parameters
    peak_hours = Column(JSONB, nullable=False, default=[9, 10, 14, 15, 16])  # Hours of peak performance
    amplitude = Column(Float, default=0.2)  # Oscillation amplitude (0-0.5 typical)
    phase_offset = Column(Float, default=0.0)  # Phase shift from standard cycle

    # Sleep patterns
    typical_sleep_start = Column(Integer, default=23)  # Hour 0-23
    typical_sleep_end = Column(Integer, default=7)  # Hour 0-23
    chronotype = Column(String(20), default="neutral")  # morning_lark, night_owl, neutral

    # Last detected sleep (inferred from inactivity)
    last_detected_sleep_start = Column(DateTime(timezone=True))
    last_detected_sleep_end = Column(DateTime(timezone=True))

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User")


class LearnerPhenotype(Base):
    """
    Learner type cluster assignment for cold-start initialization.
    Maps users to one of several "learner phenotypes" based on learning patterns.
    """
    __tablename__ = "learner_phenotype"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)

    # Phenotype assignment
    phenotype_id = Column(Integer, nullable=False, index=True)  # Cluster index 0-6
    phenotype_name = Column(String(50), nullable=False)  # Human-readable name

    # Centroid parameters for MAML initialization
    centroid_params = Column(JSONB, nullable=False)  # Dict of model parameters

    # Assignment metadata
    assignment_confidence = Column(Float, default=0.5)  # Confidence in cluster assignment
    reviews_at_assignment = Column(Integer, default=0)  # Reviews when assigned

    # Derived characteristics
    decay_rate_factor = Column(Float, default=1.0)  # Relative to population mean
    learning_rate_factor = Column(Float, default=1.0)  # Relative to population mean

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User")


class ODECardState(Base):
    """
    Per user-concept ODE state tracking.
    Stores current latent state and control mode for hybrid FSRS/ODE system.
    """
    __tablename__ = "ode_card_state"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    concept_id = Column(Integer, nullable=False)

    # Current ODE state
    current_latent_state = Column(JSONB, nullable=False)  # 32-dim vector
    last_state_time = Column(DateTime(timezone=True), nullable=False)

    # Control mode
    control_mode = Column(String(20), default="shadow", index=True)  # shadow, hybrid, active
    ode_confidence = Column(Float, default=0.0)  # Model confidence 0-1

    # Training statistics
    review_count = Column(Integer, default=0)  # Total reviews for this card
    total_training_samples = Column(Integer, default=0)
    last_loss = Column(Float)  # Last training loss
    model_version = Column(Integer, default=0)  # Increments on retrain

    # Safety bounds
    min_interval_days = Column(Integer, default=1)
    max_interval_days = Column(Integer, default=365)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User")

    __table_args__ = (
        UniqueConstraint('user_id', 'concept_id', name='uq_ode_card_state_user_concept'),
    )


class ResponseTimeObservation(Base):
    """
    Response time and hesitation metrics for TD-BKT integration.
    Provides implicit observations for memory state calibration.
    """
    __tablename__ = "response_time_observation"

    id = Column(Integer, primary_key=True, index=True)
    review_log_id = Column(
        Integer,
        ForeignKey("review_logs.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Raw response time
    response_time_ms = Column(Integer, nullable=False)
    normalized_rt = Column(Float)  # Log-normalized: ln(min(RT, 60000) + 1)

    # Hesitation metrics
    cursor_tortuosity = Column(Float)  # path_length / euclidean_distance (1.0 = direct)
    hesitation_count = Column(Integer)  # Pauses > 500ms
    backspace_count = Column(Integer)  # Corrections
    click_count = Column(Integer)  # Total clicks before answer

    # Derived metrics
    retrieval_fluency = Column(Float)  # Computed metric 0-1
    time_to_first_action_ms = Column(Integer)  # Time before any input

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    review_log = relationship("ReviewLog")


# Phenotype definitions for reference
PHENOTYPE_DEFINITIONS = {
    0: {
        "name": "Fast Forgetter",
        "description": "High initial retention but steep decay curve",
        "decay_rate_factor": 1.4,
        "learning_rate_factor": 1.2,
    },
    1: {
        "name": "Steady Learner",
        "description": "Consistent moderate retention and decay",
        "decay_rate_factor": 1.0,
        "learning_rate_factor": 1.0,
    },
    2: {
        "name": "Cramper",
        "description": "Very high short-term retention, rapid long-term decay",
        "decay_rate_factor": 1.6,
        "learning_rate_factor": 1.5,
    },
    3: {
        "name": "Deep Processor",
        "description": "Slow initial learning but very stable long-term retention",
        "decay_rate_factor": 0.7,
        "learning_rate_factor": 0.8,
    },
    4: {
        "name": "Night Owl",
        "description": "Better performance in evening sessions",
        "decay_rate_factor": 1.0,
        "learning_rate_factor": 1.0,
        "phase_offset": 4.0,  # Peak shifted 4 hours later
    },
    5: {
        "name": "Morning Lark",
        "description": "Better performance in morning sessions",
        "decay_rate_factor": 1.0,
        "learning_rate_factor": 1.0,
        "phase_offset": -2.0,  # Peak shifted 2 hours earlier
    },
    6: {
        "name": "Variable Learner",
        "description": "High variance in retention depending on context",
        "decay_rate_factor": 1.1,
        "learning_rate_factor": 1.1,
    },
}
