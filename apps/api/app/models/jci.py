"""
Joint Causal Inference (JCI) Models.

SQLAlchemy models for linking A/B experiments to causal graph edges,
enabling Bayesian updates of edge confidence from experimental data.

The JCI framework combines:
- Observational data (correlations from user behavior)
- Experimental data (A/B tests with randomized assignments)

This allows us to strengthen causal claims about learning relationships
by validating observational findings with interventional evidence.
"""
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey,
    Index, Enum as SQLEnum, Text
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from enum import Enum
from app.core.database import Base


class ExperimentEdgeStatus(str, Enum):
    """Status of an experiment-edge link."""
    PENDING = "pending"          # Experiment created, not yet complete
    RUNNING = "running"          # Experiment actively collecting data
    COMPLETED = "completed"      # Experiment finished, results available
    VALIDATED = "validated"      # Results validated, confidence updated
    INVALIDATED = "invalidated"  # Results inconclusive or invalidated


class CausalDirection(str, Enum):
    """Direction of causal relationship."""
    FORWARD = "forward"   # source -> target
    REVERSE = "reverse"   # target -> source
    BIDIRECTIONAL = "bidirectional"  # Both directions
    NONE = "none"         # No causal relationship


class ExperimentEdge(Base):
    """
    Links A/B testing experiments to causal graph edges.

    When we observe a potential causal relationship A → B in the knowledge
    graph (from observational data), we can design an A/B test to validate
    this relationship through intervention.

    The JCI framework uses Bayesian updating to combine:
    - Prior confidence: P(A→B) from observational discovery
    - Experimental evidence: Effect size and significance from A/B test
    - Posterior confidence: P(A→B | experiment) after updating
    """
    __tablename__ = "experiment_edges"

    id = Column(Integer, primary_key=True, index=True)

    # Link to A/B testing framework
    experiment_id = Column(String(100), nullable=False, index=True)

    # Source and target concepts (from knowledge graph)
    source_concept = Column(String(255), nullable=False)
    target_concept = Column(String(255), nullable=False)

    # Course context (experiments may be course-specific)
    course_id = Column(Integer, index=True)

    # Edge confidence before and after experiment
    prior_confidence = Column(Float, nullable=False, default=0.5)
    posterior_confidence = Column(Float)  # Updated after experiment

    # Observational evidence (before experiment)
    observational_correlation = Column(Float)  # Correlation coefficient
    observational_sample_size = Column(Integer)  # N for observational data
    observational_p_value = Column(Float)  # From observational analysis

    # Experimental results
    experiment_effect_size = Column(Float)  # Cohen's d or similar
    experiment_p_value = Column(Float)  # From A/B test
    experiment_sample_size = Column(Integer)  # N per variant
    experiment_power = Column(Float)  # Statistical power achieved

    # Confidence intervals
    effect_ci_lower = Column(Float)  # 95% CI lower bound
    effect_ci_upper = Column(Float)  # 95% CI upper bound

    # Treatment details
    treatment_description = Column(Text)  # What intervention was applied
    control_description = Column(Text)  # What was the control condition

    # Causal direction inference
    inferred_direction = Column(
        SQLEnum(CausalDirection),
        default=CausalDirection.FORWARD
    )
    direction_confidence = Column(Float)  # Confidence in direction

    # Status tracking
    status = Column(
        SQLEnum(ExperimentEdgeStatus),
        default=ExperimentEdgeStatus.PENDING,
        index=True
    )

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    experiment_started_at = Column(DateTime(timezone=True))
    experiment_completed_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Additional data
    extra_metadata = Column(JSONB, default={})  # Additional experiment metadata

    __table_args__ = (
        Index('ix_experiment_edges_source_target', 'source_concept', 'target_concept'),
        Index('ix_experiment_edges_course_status', 'course_id', 'status'),
    )


class CausalEdgeHistory(Base):
    """
    Tracks the history of confidence updates for causal edges.

    Each time an experiment completes or new evidence arrives,
    we record the confidence change for auditability and analysis.
    """
    __tablename__ = "causal_edge_history"

    id = Column(Integer, primary_key=True, index=True)

    # Reference to experiment edge
    experiment_edge_id = Column(
        Integer,
        ForeignKey("experiment_edges.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Confidence change
    confidence_before = Column(Float, nullable=False)
    confidence_after = Column(Float, nullable=False)
    confidence_delta = Column(Float, nullable=False)  # after - before

    # What triggered this update
    update_reason = Column(String(100), nullable=False)  # experiment_complete, new_data, manual
    update_source = Column(String(255))  # experiment_id or data source

    # Evidence that drove the update
    evidence_type = Column(String(50))  # experimental, observational, meta
    evidence_strength = Column(Float)  # How strongly evidence supports update

    # Bayesian update details
    likelihood_ratio = Column(Float)  # P(data|H1) / P(data|H0)
    prior = Column(Float)
    posterior = Column(Float)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    experiment_edge = relationship("ExperimentEdge")


class EdgeValidationQueue(Base):
    """
    Queue of causal edges that need experimental validation.

    Populated by active learning module which identifies high-uncertainty
    edges that would benefit most from experimental validation.
    """
    __tablename__ = "edge_validation_queue"

    id = Column(Integer, primary_key=True, index=True)

    # Edge to validate
    source_concept = Column(String(255), nullable=False)
    target_concept = Column(String(255), nullable=False)
    course_id = Column(Integer, nullable=False, index=True)

    # Priority and scoring
    priority_score = Column(Float, nullable=False, default=0.0)  # Higher = more urgent
    information_gain = Column(Float)  # Expected info gain from experiment
    uncertainty = Column(Float)  # Current edge uncertainty

    # Feasibility
    estimated_sample_size = Column(Integer)  # Required N for significance
    estimated_duration_days = Column(Integer)  # How long experiment would take
    feasibility_score = Column(Float)  # 0-1, how feasible is this experiment

    # Current evidence summary
    current_confidence = Column(Float)
    evidence_sources = Column(JSONB, default=[])  # List of evidence types available

    # Queue management
    queued_at = Column(DateTime(timezone=True), server_default=func.now())
    last_reviewed_at = Column(DateTime(timezone=True))
    experiment_id = Column(String(100))  # Set when experiment is created

    # Status
    status = Column(String(50), default="pending", index=True)  # pending, approved, rejected, assigned
    rejection_reason = Column(Text)

    __table_args__ = (
        Index('ix_edge_validation_queue_course_priority', 'course_id', 'priority_score'),
    )


class MetaAnalysisResult(Base):
    """
    Stores meta-analysis results combining multiple experiments.

    When multiple experiments test similar causal relationships,
    we can combine their results through meta-analysis for stronger
    evidence.
    """
    __tablename__ = "meta_analysis_results"

    id = Column(Integer, primary_key=True, index=True)

    # What relationship was analyzed
    source_concept = Column(String(255), nullable=False)
    target_concept = Column(String(255), nullable=False)

    # Experiments included
    experiment_ids = Column(JSONB, nullable=False)  # List of experiment_edge IDs
    num_experiments = Column(Integer, nullable=False)
    total_sample_size = Column(Integer, nullable=False)

    # Combined effect estimate
    pooled_effect_size = Column(Float, nullable=False)
    pooled_effect_ci_lower = Column(Float)
    pooled_effect_ci_upper = Column(Float)
    pooled_p_value = Column(Float)

    # Heterogeneity measures
    heterogeneity_i2 = Column(Float)  # I² statistic (0-100%)
    heterogeneity_q = Column(Float)  # Cochran's Q
    heterogeneity_p_value = Column(Float)  # Significance of heterogeneity

    # Final conclusion
    combined_confidence = Column(Float, nullable=False)  # Final P(A→B)
    conclusion = Column(String(50))  # supported, refuted, inconclusive

    # Analysis details
    analysis_method = Column(String(50))  # fixed_effects, random_effects
    weights = Column(JSONB)  # Weights for each study

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('ix_meta_analysis_source_target', 'source_concept', 'target_concept'),
    )
