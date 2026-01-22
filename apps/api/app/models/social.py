"""
Social Gamification Models

Models for social features including friends, challenges, and study groups.
Enables collaborative learning and competitive elements.
"""
from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Table,
    Text,
    Enum as SQLEnum,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
from enum import Enum
from datetime import datetime


class FriendshipStatus(str, Enum):
    """Status of a friendship request"""
    PENDING = "pending"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    BLOCKED = "blocked"


class ChallengeType(str, Enum):
    """Types of learning challenges"""
    STREAK = "streak"           # Maintain study streak
    XP_RACE = "xp_race"         # Earn most XP in timeframe
    MASTERY = "mastery"         # Master specific concepts
    QUIZ_SCORE = "quiz_score"   # Best quiz performance
    STUDY_TIME = "study_time"   # Most time spent studying


class ChallengeStatus(str, Enum):
    """Status of a challenge"""
    PENDING = "pending"         # Waiting for acceptance
    ACTIVE = "active"           # Challenge in progress
    COMPLETED = "completed"     # Challenge finished
    CANCELLED = "cancelled"     # Challenge was cancelled


class GroupRole(str, Enum):
    """Roles within a study group"""
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"


# Association table for study group members
study_group_members = Table(
    "study_group_members",
    Base.metadata,
    Column("group_id", Integer, ForeignKey("study_groups.id"), primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("role", SQLEnum(GroupRole), default=GroupRole.MEMBER),
    Column("joined_at", DateTime(timezone=True), server_default=func.now()),
)


class Friendship(Base):
    """
    Friendship relationship between users.
    Supports asymmetric requests (pending until accepted).
    """
    __tablename__ = "friendships"

    id = Column(Integer, primary_key=True, index=True)
    requester_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    addressee_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    status = Column(SQLEnum(FriendshipStatus), default=FriendshipStatus.PENDING)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    requester = relationship("User", foreign_keys=[requester_id], backref="sent_friend_requests")
    addressee = relationship("User", foreign_keys=[addressee_id], backref="received_friend_requests")


class Challenge(Base):
    """
    Learning challenge between users.
    Enables competitive learning through various challenge types.
    """
    __tablename__ = "challenges"

    id = Column(Integer, primary_key=True, index=True)
    challenge_type = Column(SQLEnum(ChallengeType), nullable=False)
    status = Column(SQLEnum(ChallengeStatus), default=ChallengeStatus.PENDING)

    # Challenge configuration
    title = Column(String, nullable=False)
    description = Column(Text)
    target_value = Column(Integer, nullable=False)  # Goal to achieve
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=True)  # Optional course scope

    # Timing
    start_date = Column(DateTime(timezone=True))
    end_date = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Creator
    creator_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Rewards
    xp_reward = Column(Integer, default=100)

    # Winner (set when challenge completes)
    winner_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Relationships
    creator = relationship("User", foreign_keys=[creator_id], backref="created_challenges")
    winner = relationship("User", foreign_keys=[winner_id], backref="won_challenges")
    participants = relationship("ChallengeParticipant", back_populates="challenge")
    course = relationship("Course", backref="challenges")


class ChallengeParticipant(Base):
    """
    Participant in a challenge with their progress.
    """
    __tablename__ = "challenge_participants"

    id = Column(Integer, primary_key=True, index=True)
    challenge_id = Column(Integer, ForeignKey("challenges.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Progress tracking
    current_value = Column(Integer, default=0)
    completed = Column(Boolean, default=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Status
    accepted = Column(Boolean, default=False)  # False until user accepts invite
    joined_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    challenge = relationship("Challenge", back_populates="participants")
    user = relationship("User", backref="challenge_participations")


class StudyGroup(Base):
    """
    Study group for collaborative learning.
    Allows users to learn together, share progress, and compete on leaderboards.
    """
    __tablename__ = "study_groups"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=True)  # Optional course focus

    # Settings
    is_public = Column(Boolean, default=False)
    max_members = Column(Integer, default=50)
    invite_code = Column(String, unique=True, nullable=True)  # For private group invites

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Aggregate stats (updated periodically)
    total_xp = Column(Integer, default=0)
    average_streak = Column(Float, default=0.0)
    member_count = Column(Integer, default=1)

    # Relationships
    owner = relationship("User", foreign_keys=[owner_id], backref="owned_study_groups")
    course = relationship("Course", backref="study_groups")
    members = relationship(
        "User",
        secondary=study_group_members,
        backref="study_groups",
    )
    messages = relationship("GroupMessage", back_populates="group")


class GroupMessage(Base):
    """
    Message in a study group chat.
    """
    __tablename__ = "group_messages"

    id = Column(Integer, primary_key=True, index=True)
    group_id = Column(Integer, ForeignKey("study_groups.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Optional: reference to shared content
    shared_module_id = Column(Integer, ForeignKey("modules.id"), nullable=True)

    # Relationships
    group = relationship("StudyGroup", back_populates="messages")
    user = relationship("User", backref="group_messages")


class Leaderboard(Base):
    """
    Cached leaderboard entries for performance.
    Refreshed periodically by background tasks.
    """
    __tablename__ = "leaderboards"

    id = Column(Integer, primary_key=True, index=True)
    leaderboard_type = Column(String, nullable=False, index=True)  # 'global', 'course_{id}', 'group_{id}'
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Ranking
    rank = Column(Integer, nullable=False)
    score = Column(Integer, nullable=False)

    # Period
    period = Column(String, nullable=False)  # 'daily', 'weekly', 'monthly', 'all_time'

    # Timestamps
    calculated_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", backref="leaderboard_entries")


class UserActivity(Base):
    """
    Activity feed for social features.
    Tracks notable actions for friends/group feeds.
    """
    __tablename__ = "user_activities"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    activity_type = Column(String, nullable=False)  # 'achievement', 'level_up', 'streak', 'challenge_win'

    # Activity details
    title = Column(String, nullable=False)
    description = Column(Text)
    metadata = Column(Text)  # JSON string for flexible data

    # Visibility
    is_public = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", backref="activities")
