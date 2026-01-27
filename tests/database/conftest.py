"""
Database Test Fixtures and Configuration

Provides shared fixtures for database testing including:
- In-memory SQLite for unit tests
- PostgreSQL test containers for integration tests
- Test data factories
- Database session management
"""
import os
import pytest
import asyncio
from datetime import datetime, timedelta
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock, AsyncMock, patch

from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool


# Test database URLs
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "sqlite+aiosqlite:///:memory:"
)
TEST_SYNC_DATABASE_URL = os.getenv(
    "TEST_SYNC_DATABASE_URL",
    "sqlite:///:memory:"
)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def sync_engine():
    """Create synchronous test engine for schema tests."""
    engine = create_engine(
        TEST_SYNC_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
    return engine


@pytest.fixture(scope="session")
async def async_engine():
    """Create async test engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
    return engine


@pytest.fixture
async def async_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Provide async database session for tests."""
    # Import models to ensure they are registered
    from app.core.database import Base
    import app.models  # noqa: F401

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session_factory = sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    async with async_session_factory() as session:
        yield session
        await session.rollback()

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
def sync_session(sync_engine) -> Generator[Session, None, None]:
    """Provide sync database session for tests."""
    from app.core.database import Base
    import app.models  # noqa: F401

    Base.metadata.create_all(bind=sync_engine)

    SessionLocal = sessionmaker(bind=sync_engine, autoflush=False, autocommit=False)
    session = SessionLocal()

    try:
        yield session
        session.rollback()
    finally:
        session.close()
        Base.metadata.drop_all(bind=sync_engine)


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
