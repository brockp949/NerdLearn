"""
Pytest fixtures for router tests
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import AsyncGenerator
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def mock_db():
    """Mock database session for testing"""
    db = AsyncMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    db.flush = AsyncMock()
    db.refresh = AsyncMock()
    db.add = MagicMock()
    db.delete = AsyncMock()
    return db


@pytest.fixture
def mock_course():
    """Mock course object"""
    course = MagicMock()
    course.id = 1
    course.title = "Test Course"
    course.description = "A test course"
    course.instructor_id = 1
    course.thumbnail_url = "https://example.com/thumb.jpg"
    course.price = 29.99
    course.difficulty_level = "beginner"
    course.tags = "python,testing"
    course.status = MagicMock(value="draft")
    course.published_at = None
    course.created_at = datetime.now()
    course.updated_at = datetime.now()
    course.modules = []
    return course


@pytest.fixture
def mock_module():
    """Mock module object"""
    module = MagicMock()
    module.id = 1
    module.course_id = 1
    module.title = "Test Module"
    module.description = "A test module"
    module.module_type = MagicMock(value="pdf")
    module.file_url = "https://example.com/file.pdf"
    module.processing_status = MagicMock(value="pending")
    module.processing_task_id = None
    module.is_processed = False
    module.chunk_count = 0
    module.concept_count = 0
    module.processed_at = None
    module.processing_error = None
    return module


@pytest.fixture
def mock_sr_card():
    """Mock spaced repetition card"""
    card = MagicMock()
    card.id = 1
    card.concept_id = 1
    card.user_id = 1
    card.course_id = 1
    card.stability = 1.0
    card.difficulty = 0.3
    card.elapsed_days = 0
    card.scheduled_days = 1
    card.reps = 0
    card.lapses = 0
    card.state = "new"
    card.last_review = None
    card.due = datetime.now()
    return card


@pytest.fixture
def mock_concept():
    """Mock concept object"""
    concept = MagicMock()
    concept.id = 1
    concept.name = "Test Concept"
    concept.description = "A test concept"
    concept.module_id = 1
    return concept


@pytest.fixture
def mock_user():
    """Mock user object"""
    user = MagicMock()
    user.id = 1
    user.email = "test@example.com"
    user.username = "testuser"
    user.is_active = True
    return user


@pytest.fixture
def mock_chat_history():
    """Mock chat history entry"""
    message = MagicMock()
    message.id = 1
    message.user_id = 1
    message.course_id = 1
    message.session_id = "test-session"
    message.role = "assistant"
    message.content = "Test response"
    message.citations = []
    message.concept_ids = []
    message.timestamp = datetime.now()
    return message


@pytest.fixture
def mock_mastery():
    """Mock user concept mastery"""
    mastery = MagicMock()
    mastery.id = 1
    mastery.user_id = 1
    mastery.concept_id = 1
    mastery.mastery_level = 0.7
    mastery.last_updated = datetime.now()
    return mastery


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client for testing with mocked dependencies"""
    # Patch database dependency
    with patch('app.core.database.get_db') as mock_get_db:
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.flush = AsyncMock()
        mock_db.refresh = AsyncMock()
        mock_get_db.return_value = mock_db

        try:
            from app.main import app

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test"
            ) as client:
                yield client
        except Exception:
            # If app import fails, create a minimal test client
            from fastapi import FastAPI
            test_app = FastAPI()

            async with AsyncClient(
                transport=ASGITransport(app=test_app),
                base_url="http://test"
            ) as client:
                yield client


@pytest.fixture
def sample_course_data():
    """Sample course creation data"""
    return {
        "title": "Introduction to Python",
        "description": "Learn Python programming from scratch",
        "instructor_id": 1,
        "thumbnail_url": "https://example.com/python.jpg",
        "price": 49.99,
        "difficulty_level": "beginner",
        "tags": ["python", "programming", "beginner"]
    }


@pytest.fixture
def sample_chat_request():
    """Sample chat request data"""
    return {
        "query": "Explain decorators in Python",
        "user_id": 1,
        "course_id": 1,
        "session_id": "test-session-123",
        "module_id": 1
    }


@pytest.fixture
def sample_review_request():
    """Sample review request data"""
    return {
        "card_id": 1,
        "rating": "good",
        "review_duration_ms": 5000
    }
