"""
Database Test Fixtures and Configuration

Provides shared fixtures for database testing including:
- In-memory SQLite for unit tests
- PostgreSQL test containers for integration tests
- Test data factories
- Database session management

This module is self-contained and defines its own test models
to avoid import path issues across different environments.
"""
import os
import sys
from pathlib import Path

# ============================================================================
# Path Setup - Add app directories to Python path
# ============================================================================
_current_file = Path(__file__).resolve()
_tests_database_dir = _current_file.parent
_tests_dir = _tests_database_dir.parent
_project_root = _tests_dir.parent

# Add paths for imports
_api_path = _project_root / "apps" / "api"
_db_path = _project_root / "packages" / "db"

for path in [str(_api_path), str(_db_path), str(_project_root)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# ============================================================================
# Standard imports
# ============================================================================
import pytest
import asyncio
from datetime import datetime, timedelta
from typing import AsyncGenerator, Generator, Optional
from unittest.mock import MagicMock, AsyncMock

from sqlalchemy import (
    create_engine, Column, Integer, String, Boolean, Float,
    DateTime, ForeignKey, Text, event
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import (
    sessionmaker, Session, relationship, declarative_base
)
from sqlalchemy.pool import StaticPool


# ============================================================================
# Test Database Configuration
# ============================================================================
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "sqlite+aiosqlite:///:memory:"
)
TEST_SYNC_DATABASE_URL = os.getenv(
    "TEST_SYNC_DATABASE_URL",
    "sqlite:///:memory:"
)

# ============================================================================
# Self-Contained Test Models
# ============================================================================
# These models mirror the production models but are self-contained
# to avoid import issues in different test environments

TestBase = declarative_base()


class User(TestBase):
    """Test User model."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    is_instructor = Column(Boolean, default=False)
    total_xp = Column(Integer, default=0)
    level = Column(Integer, default=1)
    streak_days = Column(Integer, default=0)
    last_activity_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    enrollments = relationship("Enrollment", back_populates="user", cascade="all, delete-orphan")
    achievements = relationship("UserAchievement", back_populates="user", cascade="all, delete-orphan")
    spaced_repetition_cards = relationship("SpacedRepetitionCard", back_populates="user", cascade="all, delete-orphan")
    stats = relationship("UserStats", back_populates="user", uselist=False, cascade="all, delete-orphan")


class Instructor(TestBase):
    """Test Instructor model."""
    __tablename__ = "instructors"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    bio = Column(Text, nullable=True)
    expertise_areas = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    courses = relationship("Course", back_populates="instructor")


class Course(TestBase):
    """Test Course model."""
    __tablename__ = "courses"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    domain = Column(String(100), index=True, nullable=True)
    instructor_id = Column(Integer, ForeignKey("instructors.id"), nullable=True)
    is_published = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    instructor = relationship("Instructor", back_populates="courses")
    enrollments = relationship("Enrollment", back_populates="course", cascade="all, delete-orphan")


class Enrollment(TestBase):
    """Test Enrollment model."""
    __tablename__ = "enrollments"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    course_id = Column(Integer, ForeignKey("courses.id", ondelete="CASCADE"), nullable=False)
    progress = Column(Float, default=0.0)
    completed_at = Column(DateTime, nullable=True)
    enrolled_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="enrollments")
    course = relationship("Course", back_populates="enrollments")


class Concept(TestBase):
    """Test Concept model."""
    __tablename__ = "concepts"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    domain = Column(String(100), nullable=True)
    subdomain = Column(String(100), nullable=True)
    avg_difficulty = Column(Float, default=5.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    spaced_repetition_cards = relationship("SpacedRepetitionCard", back_populates="concept")


class SpacedRepetitionCard(TestBase):
    """Test SpacedRepetitionCard model (FSRS-based)."""
    __tablename__ = "spaced_repetition_cards"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    concept_id = Column(Integer, ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False)
    difficulty = Column(Float, default=5.0)
    stability = Column(Float, default=2.5)
    retrievability = Column(Float, default=0.9)
    review_count = Column(Integer, default=0)
    next_review_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="spaced_repetition_cards")
    concept = relationship("Concept", back_populates="spaced_repetition_cards")
    review_logs = relationship("ReviewLog", back_populates="card", cascade="all, delete-orphan")


class ReviewLog(TestBase):
    """Test ReviewLog model."""
    __tablename__ = "review_logs"

    id = Column(Integer, primary_key=True, index=True)
    card_id = Column(Integer, ForeignKey("spaced_repetition_cards.id", ondelete="CASCADE"), nullable=False)
    rating = Column(Integer, nullable=False)  # 1-5 rating
    reviewed_at = Column(DateTime, default=datetime.utcnow)
    response_time_ms = Column(Integer, nullable=True)

    # Relationships
    card = relationship("SpacedRepetitionCard", back_populates="review_logs")


class UserAchievement(TestBase):
    """Test UserAchievement model."""
    __tablename__ = "user_achievements"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    achievement_type = Column(String(50), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    xp_reward = Column(Integer, default=0)
    unlocked_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="achievements")


class UserStats(TestBase):
    """Test UserStats model."""
    __tablename__ = "user_stats"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    total_achievements = Column(Integer, default=0)
    total_reviews = Column(Integer, default=0)
    total_correct = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="stats")


class DailyActivity(TestBase):
    """Test DailyActivity model."""
    __tablename__ = "daily_activities"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    date = Column(DateTime, nullable=False)
    xp_earned = Column(Integer, default=0)
    reviews_completed = Column(Integer, default=0)


class ChatHistory(TestBase):
    """Test ChatHistory model."""
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    message = Column(Text, nullable=False)
    response = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class UserConceptMastery(TestBase):
    """Test UserConceptMastery model."""
    __tablename__ = "user_concept_mastery"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    concept_id = Column(Integer, ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False)
    mastery_level = Column(Float, default=0.0)
    last_reviewed = Column(DateTime, nullable=True)


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def sync_engine():
    """Create synchronous test engine for schema tests."""
    engine = create_engine(
        TEST_SYNC_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )

    # Enable foreign keys for SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    # Create all tables
    TestBase.metadata.create_all(bind=engine)
    yield engine
    # Drop all tables after test
    TestBase.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
async def async_engine():
    """Create async test engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )

    # Enable foreign keys for SQLite async
    def _enable_foreign_keys(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    from sqlalchemy import event as sa_event
    sa_event.listen(engine.sync_engine, "connect", _enable_foreign_keys)

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(TestBase.metadata.create_all)

    yield engine

    # Drop all tables after test
    async with engine.begin() as conn:
        await conn.run_sync(TestBase.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
async def async_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Provide async database session for tests."""
    async_session_factory = sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    async with async_session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
def sync_session(sync_engine) -> Generator[Session, None, None]:
    """Provide sync database session for tests."""
    SessionLocal = sessionmaker(bind=sync_engine, autoflush=False, autocommit=False)
    session = SessionLocal()

    try:
        yield session
        session.rollback()
    finally:
        session.close()


# ============================================================================
# Test Data Factories
# ============================================================================

class UserFactory:
    """Factory for creating test User records."""

    _counter = 0

    @classmethod
    def create(cls, **kwargs) -> dict:
        """Create user data dictionary."""
        cls._counter += 1
        defaults = {
            "email": f"testuser{cls._counter}@example.com",
            "username": f"testuser{cls._counter}",
            "hashed_password": "hashed_password_placeholder",
            "full_name": f"Test User {cls._counter}",
            "is_active": True,
            "is_instructor": False,
            "total_xp": 0,
            "level": 1,
            "streak_days": 0,
        }
        defaults.update(kwargs)
        return defaults

    @classmethod
    def reset(cls):
        """Reset counter for clean test runs."""
        cls._counter = 0


class ConceptFactory:
    """Factory for creating test Concept records."""

    _counter = 0
    DOMAINS = ["Mathematics", "Computer Science", "Physics", "Biology"]

    @classmethod
    def create(cls, **kwargs) -> dict:
        """Create concept data dictionary."""
        cls._counter += 1
        defaults = {
            "name": f"Test Concept {cls._counter}",
            "description": f"Description for concept {cls._counter}",
            "domain": cls.DOMAINS[cls._counter % len(cls.DOMAINS)],
            "subdomain": f"Subdomain {cls._counter}",
            "avg_difficulty": 5.0,
        }
        defaults.update(kwargs)
        return defaults

    @classmethod
    def reset(cls):
        cls._counter = 0


class CourseFactory:
    """Factory for creating test Course records."""

    _counter = 0

    @classmethod
    def create(cls, **kwargs) -> dict:
        """Create course data dictionary."""
        cls._counter += 1
        defaults = {
            "title": f"Test Course {cls._counter}",
            "description": f"Description for course {cls._counter}",
            "domain": "Computer Science",
            "is_published": False,
        }
        defaults.update(kwargs)
        return defaults

    @classmethod
    def reset(cls):
        cls._counter = 0


class SpacedRepetitionCardFactory:
    """Factory for creating test SpacedRepetitionCard records."""

    _counter = 0

    @classmethod
    def create(cls, user_id: int, concept_id: int, **kwargs) -> dict:
        """Create spaced repetition card data dictionary."""
        cls._counter += 1
        defaults = {
            "user_id": user_id,
            "concept_id": concept_id,
            "difficulty": 5.0,
            "stability": 2.5,
            "retrievability": 0.9,
            "review_count": 0,
            "next_review_at": datetime.utcnow() + timedelta(days=1),
        }
        defaults.update(kwargs)
        return defaults

    @classmethod
    def reset(cls):
        cls._counter = 0


@pytest.fixture(autouse=True)
def reset_factories():
    """Reset all factories before each test."""
    UserFactory.reset()
    ConceptFactory.reset()
    CourseFactory.reset()
    SpacedRepetitionCardFactory.reset()
    yield


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock = MagicMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=1)
    mock.expire = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_neo4j():
    """Mock Neo4j driver for testing."""
    mock = MagicMock()
    mock.execute_query = MagicMock(return_value=([], None, None))
    return mock


# ============================================================================
# Performance Test Utilities
# ============================================================================

class QueryTimer:
    """Context manager for timing database queries."""

    def __init__(self):
        self.queries = []
        self.total_time = 0

    def __enter__(self):
        self.start_time = datetime.utcnow()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.total_time = (datetime.utcnow() - self.start_time).total_seconds()
        return False

    def record_query(self, query: str, duration: float):
        """Record a query execution."""
        self.queries.append({
            "query": query,
            "duration": duration
        })


@pytest.fixture
def query_timer():
    """Provide query timer for performance tests."""
    return QueryTimer()


# ============================================================================
# Test Report Data Collection
# ============================================================================

class TestReportCollector:
    """Collects test results for reporting."""

    def __init__(self):
        self.results = []
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = datetime.utcnow()

    def end(self):
        self.end_time = datetime.utcnow()

    def record(self, test_name: str, category: str, passed: bool,
               duration: float, details: dict = None):
        """Record a test result."""
        self.results.append({
            "test_name": test_name,
            "category": category,
            "passed": passed,
            "duration": duration,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        })

    def get_summary(self) -> dict:
        """Get test summary statistics."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed

        by_category = {}
        for result in self.results:
            cat = result["category"]
            if cat not in by_category:
                by_category[cat] = {"passed": 0, "failed": 0, "total": 0}
            by_category[cat]["total"] += 1
            if result["passed"]:
                by_category[cat]["passed"] += 1
            else:
                by_category[cat]["failed"] += 1

        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": (passed / total * 100) if total > 0 else 0,
            "by_category": by_category,
            "duration": (self.end_time - self.start_time).total_seconds()
                       if self.end_time and self.start_time else 0
        }


@pytest.fixture(scope="session")
def report_collector():
    """Provide test report collector."""
    collector = TestReportCollector()
    collector.start()
    yield collector
    collector.end()
