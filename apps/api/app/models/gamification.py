"""
Gamification Database Models
Achievements, user progress, and gamification stats
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class UserAchievement(Base):
    """
    User achievements tracking
    """

    __tablename__ = "user_achievements"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    achievement_id = Column(String, nullable=False, index=True)  # Achievement identifier

    # Achievement details (denormalized for historical record)
    name = Column(String, nullable=False)
    description = Column(Text)
    achievement_type = Column(String, nullable=False)  # milestone, streak, mastery, etc.
    xp_reward = Column(Integer, default=0)
    rarity = Column(String, default="common")

    # Metadata
    unlocked_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="achievements")


class UserStats(Base):
    """
    User gamification statistics
    """

    __tablename__ = "user_stats"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True, index=True)

    # Overall stats
    modules_completed = Column(Integer, default=0)
    concepts_mastered = Column(Integer, default=0)
    reviews_completed = Column(Integer, default=0)
    perfect_streak = Column(Integer, default=0)  # Current perfect review streak
    chat_messages = Column(Integer, default=0)
    videos_completed = Column(Integer, default=0)

    # Course progress
    courses_started = Column(Integer, default=0)
    courses_completed = Column(Integer, default=0)

    # Time tracking
    total_study_time_minutes = Column(Integer, default=0)
    avg_session_minutes = Column(Float, default=0)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="stats")


class DailyActivity(Base):
    """
    Daily activity tracking for streaks
    """

    __tablename__ = "daily_activities"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    activity_date = Column(DateTime(timezone=True), nullable=False, index=True)

    # Activity metrics
    xp_earned = Column(Integer, default=0)
    reviews_completed = Column(Integer, default=0)
    modules_completed = Column(Integer, default=0)
    study_time_minutes = Column(Integer, default=0)

    # Goals
    daily_goal_met = Column(Boolean, default=False)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User")


class ChatHistory(Base):
    """
    Chat conversation history
    """

    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False, index=True)
    session_id = Column(String, nullable=False, index=True)

    # Message
    role = Column(String, nullable=False)  # user, assistant
    content = Column(Text, nullable=False)

    # Citations (JSON array)
    citations = Column(JSON)  # List of citation objects

    # Concepts referenced
    concept_ids = Column(JSON)  # List of concept IDs

    # Metadata
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Relationships
    user = relationship("User")
    course = relationship("Course")
