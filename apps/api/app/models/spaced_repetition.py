"""
Spaced Repetition Models
Database models for FSRS cards and review history
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class SpacedRepetitionCard(Base):
    """
    FSRS flashcard for concept mastery tracking
    """

    __tablename__ = "spaced_repetition_cards"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    concept_id = Column(Integer, nullable=False, index=True)  # Concept from knowledge graph
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False, index=True)

    # FSRS Parameters
    stability = Column(Float, default=0.0)  # Memory stability
    difficulty = Column(Float, default=5.0)  # Difficulty (1-10)
    elapsed_days = Column(Integer, default=0)  # Days since last review
    scheduled_days = Column(Integer, default=0)  # Days until next review
    reps = Column(Integer, default=0)  # Total reviews
    lapses = Column(Integer, default=0)  # Times forgotten

    # Card State
    state = Column(String, default="new")  # new, learning, review, relearning
    last_review = Column(DateTime(timezone=True))  # Last review time
    due = Column(DateTime(timezone=True), index=True)  # Next review due date

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="spaced_repetition_cards")
    course = relationship("Course")
    reviews = relationship(
        "ReviewLog", back_populates="card", cascade="all, delete-orphan"
    )


class ReviewLog(Base):
    """
    Log of individual reviews for analytics and algorithm optimization
    """

    __tablename__ = "review_logs"

    id = Column(Integer, primary_key=True, index=True)
    card_id = Column(
        Integer, ForeignKey("spaced_repetition_cards.id"), nullable=False, index=True
    )
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Review Details
    rating = Column(Integer, nullable=False)  # 1-4 (AGAIN, HARD, GOOD, EASY)
    review_time = Column(DateTime(timezone=True), nullable=False, index=True)
    elapsed_days = Column(Integer)  # Days since last review
    scheduled_days = Column(Integer)  # Days scheduled for this review

    # FSRS State Snapshot
    stability = Column(Float)  # Stability after review
    difficulty = Column(Float)  # Difficulty after review
    state = Column(String)  # Card state after review

    # Context
    review_duration_ms = Column(Integer)  # Time taken to review (milliseconds)
    source = Column(String)  # Source of review (manual, stealth, etc.)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    card = relationship("SpacedRepetitionCard", back_populates="reviews")
    user = relationship("User")


class Concept(Base):
    """
    Concept nodes extracted from course content
    Links to Neo4j knowledge graph
    """

    __tablename__ = "concepts"

    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False, index=True)
    name = Column(String, nullable=False, index=True)
    description = Column(Text)

    # Knowledge Graph Reference
    neo4j_node_id = Column(String)  # ID in Neo4j graph

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    course = relationship("Course")
    user_mastery = relationship("UserConceptMastery", back_populates="concept")
