"""
PACER Learning Protocol Models
Information taxonomy for adaptive learning content classification

P.A.C.E.R. Framework:
- P: Procedural (Practice) - How-to steps and instructions
- A: Analogous (Critique) - Metaphors and comparisons to prior knowledge
- C: Conceptual (Mapping) - Theories, principles, causal relationships
- E: Evidence (Store & Rehearse) - Facts/statistics that validate concepts
- R: Reference (Store & Rehearse) - Arbitrary details for rote memorization
"""
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    ForeignKey,
    JSON,
    Enum,
    Boolean,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
import enum


class PACERType(str, enum.Enum):
    """Five information types in the PACER taxonomy"""

    PROCEDURAL = "procedural"  # How-to instructions -> Practice
    ANALOGOUS = "analogous"  # Metaphors/comparisons -> Critique
    CONCEPTUAL = "conceptual"  # Theories/principles -> Map
    EVIDENCE = "evidence"  # Supporting facts -> Link to concepts
    REFERENCE = "reference"  # Arbitrary details -> SRS flashcards


class EvidenceRelationshipType(str, enum.Enum):
    """Relationship between evidence and concepts"""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    QUALIFIES = "qualifies"


class BreakdownSeverity(str, enum.Enum):
    """Severity of analogy breakdown points"""

    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"


class PACERContentItem(Base):
    """
    Content item classified by PACER type.
    Links to source content (modules, chunks) with PACER classification.
    """

    __tablename__ = "pacer_content_items"

    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False, index=True)
    module_id = Column(Integer, ForeignKey("modules.id"), nullable=True, index=True)
    chunk_id = Column(String, nullable=True)  # Vector store chunk reference

    # PACER Classification
    pacer_type = Column(Enum(PACERType), nullable=False, index=True)
    classification_confidence = Column(Float, default=0.0)
    classification_method = Column(
        String, default="ai_triage"
    )  # ai_triage, manual, hybrid

    # Content
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String, index=True)  # For deduplication

    # Type-specific metadata (JSON for flexibility)
    # For Procedural: {"steps": [...], "prerequisites": [...], "estimated_time_mins": int}
    # For Analogous: {"source_domain": "", "target_domain": ""}
    # For Conceptual: {"related_concepts": [...], "causal_relationships": [...]}
    # For Evidence: {"statistical_data": {...}, "source_citation": ""}
    # For Reference: {"category": "", "mnemonics": [...]}
    item_metadata = Column(JSON, default={})

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    course = relationship("Course")
    module = relationship("Module")
    analogy = relationship(
        "AnalogyRecord", back_populates="pacer_item", uselist=False, cascade="all, delete-orphan"
    )
    evidence_links = relationship(
        "EvidenceConceptLink", back_populates="evidence_item", cascade="all, delete-orphan"
    )
    procedural_progress = relationship(
        "ProceduralProgress", back_populates="pacer_item", cascade="all, delete-orphan"
    )


class AnalogyRecord(Base):
    """
    Detailed analogy storage for critique learning.
    Stores the structural mapping between source (familiar) and target (new) domains.
    """

    __tablename__ = "analogy_records"

    id = Column(Integer, primary_key=True, index=True)
    pacer_item_id = Column(
        Integer, ForeignKey("pacer_content_items.id"), nullable=False, unique=True
    )

    # Domains
    source_domain = Column(String, nullable=False)  # Familiar concept (e.g., "Water Flow")
    target_domain = Column(String, nullable=False)  # New concept (e.g., "Electricity")

    # Structural mapping: how elements of source map to target
    # Format: [{"source_element": "pressure", "target_element": "voltage", "relationship": "corresponds_to"}, ...]
    structural_mapping = Column(JSON, default=[])

    # Valid aspects where analogy holds
    valid_aspects = Column(JSON, default=[])

    # Breakdown points where analogy fails
    # Format: [{"aspect": "visibility", "reason": "Water is visible, electricity is not", "severity": "minor", "educational_note": "..."}]
    breakdown_points = Column(JSON, default=[])

    # Critique prompt to show learners
    critique_prompt = Column(Text, nullable=True)

    # Whether breakdown points have been validated by instructor/expert
    is_validated = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    pacer_item = relationship("PACERContentItem", back_populates="analogy")
    critiques = relationship(
        "AnalogyCritique", back_populates="analogy", cascade="all, delete-orphan"
    )


class EvidenceConceptLink(Base):
    """
    Links Evidence items to Concepts they support/contradict/qualify.
    Central to PACER's Evidence-Concept relationship model.
    """

    __tablename__ = "evidence_concept_links"

    id = Column(Integer, primary_key=True, index=True)
    evidence_item_id = Column(
        Integer, ForeignKey("pacer_content_items.id"), nullable=False, index=True
    )
    concept_id = Column(Integer, ForeignKey("concepts.id"), nullable=False, index=True)

    # Link metadata
    relationship_type = Column(
        Enum(EvidenceRelationshipType), default=EvidenceRelationshipType.SUPPORTS
    )
    strength = Column(Float, default=0.5)  # 0-1 how strongly evidence relates to concept

    # Citation/source information
    citation = Column(String, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Ensure unique evidence-concept pairs
    __table_args__ = (
        UniqueConstraint("evidence_item_id", "concept_id", name="uq_evidence_concept"),
    )

    # Relationships
    evidence_item = relationship("PACERContentItem", back_populates="evidence_links")
    concept = relationship("Concept")


class ProceduralProgress(Base):
    """
    Tracks user progress through procedural (practice) content.
    Implements step-by-step practice tracking per PACER protocol.
    """

    __tablename__ = "procedural_progress"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    pacer_item_id = Column(
        Integer, ForeignKey("pacer_content_items.id"), nullable=False, index=True
    )

    # Progress tracking
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer, nullable=False)
    attempts = Column(Integer, default=0)
    completed = Column(Boolean, default=False)

    # Performance metrics
    step_times_ms = Column(JSON, default=[])  # Time spent on each step
    errors_per_step = Column(JSON, default=[])  # Errors at each step
    step_completions = Column(JSON, default=[])  # Boolean for each step

    # Timestamps
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    last_activity_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Ensure unique user-item pairs
    __table_args__ = (
        UniqueConstraint("user_id", "pacer_item_id", name="uq_user_procedural"),
    )

    # Relationships
    user = relationship("User")
    pacer_item = relationship("PACERContentItem", back_populates="procedural_progress")


class AnalogyCritique(Base):
    """
    User's critique of analogies - identifies breakdown points.
    Core to PACER's Analogous learning action (Critique).
    """

    __tablename__ = "analogy_critiques"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    analogy_id = Column(
        Integer, ForeignKey("analogy_records.id"), nullable=False, index=True
    )

    # User's identified breakdown points
    # Format: [{"aspect": "...", "explanation": "..."}]
    identified_breakdowns = Column(JSON, default=[])

    # Evaluation metrics
    critique_score = Column(Float, nullable=True)  # F1 score 0-1
    precision = Column(Float, nullable=True)  # Correct / Total identified
    recall = Column(Float, nullable=True)  # Correct / Total actual

    # Correctly identified breakdown points
    correct_breakdowns = Column(JSON, default=[])
    # Breakdown points user missed
    missed_breakdowns = Column(JSON, default=[])
    # Breakdown points user identified incorrectly (false positives)
    false_positives = Column(JSON, default=[])

    # Feedback provided to user
    feedback = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User")
    analogy = relationship("AnalogyRecord", back_populates="critiques")


class UserPACERProfile(Base):
    """
    User's proficiency and preferences across PACER types.
    Tracks learning effectiveness for each information type.
    """

    __tablename__ = "user_pacer_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)

    # Type-specific proficiencies (0-1 scale)
    procedural_proficiency = Column(Float, default=0.5)
    analogous_proficiency = Column(Float, default=0.5)
    conceptual_proficiency = Column(Float, default=0.5)
    evidence_proficiency = Column(Float, default=0.5)
    reference_proficiency = Column(Float, default=0.5)

    # Learning preferences (which types work best for this user)
    # Ordered list of PACERType values by effectiveness
    preferred_types = Column(JSON, default=[])

    # Statistics
    total_items_processed = Column(Integer, default=0)
    items_by_type = Column(
        JSON, default={"procedural": 0, "analogous": 0, "conceptual": 0, "evidence": 0, "reference": 0}
    )

    # User's own classification attempts (for meta-learning)
    classification_attempts = Column(Integer, default=0)
    classification_accuracy = Column(Float, default=0.0)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User")


class PACERClassificationLog(Base):
    """
    Log of content classifications for analytics and model improvement.
    """

    __tablename__ = "pacer_classification_logs"

    id = Column(Integer, primary_key=True, index=True)
    content_hash = Column(String, index=True)
    content_preview = Column(String(500))  # First 500 chars

    # Classification result
    predicted_type = Column(Enum(PACERType), nullable=False)
    confidence = Column(Float, nullable=False)
    alternative_types = Column(JSON, default=[])  # [(type, confidence), ...]

    # Classification method
    method = Column(String, default="ai_triage")  # ai_triage, rule_based, llm, hybrid

    # Triage decision path
    triage_decisions = Column(JSON, default=[])

    # Optional user correction
    corrected_type = Column(Enum(PACERType), nullable=True)
    corrected_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
