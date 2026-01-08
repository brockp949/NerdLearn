from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_instructor = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Gamification fields
    total_xp = Column(Integer, default=0)
    level = Column(Integer, default=1)
    streak_days = Column(Integer, default=0)
    last_activity_date = Column(DateTime(timezone=True))

    # Relationships
    enrollments = relationship("Enrollment", back_populates="user")
    mastery_records = relationship("UserConceptMastery", back_populates="user")
    spaced_repetition_cards = relationship("SpacedRepetitionCard", back_populates="user")
    achievements = relationship("UserAchievement", back_populates="user")
    stats = relationship("UserStats", back_populates="user", uselist=False)


class Instructor(Base):
    __tablename__ = "instructors"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, unique=True)
    bio = Column(String)
    expertise_areas = Column(String)  # JSON string
    rating = Column(Float, default=0.0)
    total_students = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    courses = relationship("Course", back_populates="instructor")
