"""
Unit test configuration and fixtures.

Unit tests validate isolated components without external dependencies.
These tests should be fast and not require services to be running.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Any, Dict

# Import shared fixtures
from tests.fixtures import (
    create_test_user_data,
    create_mock_learner_profile,
    create_mock_card,
    create_mock_session
)


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_db_session():
    """Mock database session for testing."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for testing."""
    client = AsyncMock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.put = AsyncMock()
    client.delete = AsyncMock()
    return client


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)
    redis.exists = AsyncMock(return_value=False)
    return redis


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture
def user_data():
    """Generate test user data."""
    return create_test_user_data()


@pytest.fixture
def learner_profile():
    """Generate mock learner profile."""
    return create_mock_learner_profile()


@pytest.fixture
def card_data():
    """Generate mock card data."""
    return create_mock_card()


@pytest.fixture
def session_data():
    """Generate mock session data."""
    return create_mock_session()


# ============================================================================
# FSRS Fixtures
# ============================================================================

@pytest.fixture
def fsrs_defaults():
    """Default FSRS algorithm parameters."""
    return {
        "w": [0.4, 0.6, 2.4, 5.8, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34, 1.26, 0.29, 2.61],
        "request_retention": 0.9,
        "maximum_interval": 36500,
        "easy_bonus": 1.3,
        "hard_interval": 1.2
    }


@pytest.fixture
def fsrs_card_new():
    """A new FSRS card (never reviewed)."""
    return {
        "due": None,
        "stability": 0,
        "difficulty": 0,
        "elapsed_days": 0,
        "scheduled_days": 0,
        "reps": 0,
        "lapses": 0,
        "state": "new",
        "last_review": None
    }


@pytest.fixture
def fsrs_card_learning():
    """An FSRS card in learning state."""
    return {
        "due": "2024-01-15T10:00:00Z",
        "stability": 1.5,
        "difficulty": 5.0,
        "elapsed_days": 1,
        "scheduled_days": 1,
        "reps": 2,
        "lapses": 0,
        "state": "learning",
        "last_review": "2024-01-14T10:00:00Z"
    }


@pytest.fixture
def fsrs_card_review():
    """An FSRS card in review state."""
    return {
        "due": "2024-01-20T10:00:00Z",
        "stability": 10.5,
        "difficulty": 4.5,
        "elapsed_days": 5,
        "scheduled_days": 5,
        "reps": 10,
        "lapses": 1,
        "state": "review",
        "last_review": "2024-01-15T10:00:00Z"
    }


# ============================================================================
# BKT Fixtures
# ============================================================================

@pytest.fixture
def bkt_defaults():
    """Default BKT algorithm parameters."""
    return {
        "p_l0": 0.1,    # Initial mastery probability
        "p_t": 0.15,    # Probability of learning
        "p_g": 0.2,     # Probability of guess
        "p_s": 0.1      # Probability of slip
    }
