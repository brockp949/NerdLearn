"""
Unit test conftest - minimal fixtures for isolated unit tests.

These tests don't require the full application context.
"""

import os
import sys

# Set minimal test environment
os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USER"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "test"
os.environ["REDIS_URL"] = "redis://localhost:6379"
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["SECRET_KEY"] = "test-secret"
os.environ["ALLOWED_ORIGINS"] = '["http://localhost:3000"]'

import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime


@pytest.fixture
def mock_llm():
    """Create a mock LangChain LLM for agent tests."""
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=MagicMock(content="Mock response"))
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="Mock response"))
    return llm


@pytest.fixture
def sample_user_id():
    """Sample user ID for tests."""
    return "test_user_123"


@pytest.fixture
def sample_concept_id():
    """Sample concept ID for tests."""
    return "concept_456"


@pytest.fixture
def sample_course_id():
    """Sample course ID for tests."""
    return 1
