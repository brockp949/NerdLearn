from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class UserConceptMastery(Base):
    """
    Tracks user mastery of individual concepts (nodes in knowledge graph)
    Used for Stealth Assessment and Bayesian updates
    """
    __tablename__ = "user_concept_mastery"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    concept_id = Column(String, nullable=False)  # Neo4j node ID
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)

    # Bayesian probability of mastery (0.0 to 1.0)
    mastery_probability = Column(Float, default=0.5)

    # Evidence counters
    correct_attempts = Column(Integer, default=0)
    total_attempts = Column(Integer, default=0)

    # Engagement metrics (stealth assessment)
    total_dwell_time = Column(Float, default=0.0)  # seconds
    interaction_count = Column(Integer, default=0)

    # Timestamps
    first_seen_at = Column(DateTime(timezone=True), server_default=func.now())
    last_updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="mastery_records")
