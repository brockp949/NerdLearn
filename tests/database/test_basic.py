"""
Basic tests to verify pytest is working correctly.
"""
import pytest


class TestBasicSync:
    """Basic sync tests - should always pass."""

    def test_simple_assert(self):
        """Most basic test possible."""
        assert True

    def test_math(self):
        """Basic math test."""
        assert 1 + 1 == 2

    def test_string(self):
        """Basic string test."""
        assert "hello".upper() == "HELLO"


class TestBasicAsync:
    """Basic async tests."""

    @pytest.mark.asyncio
    async def test_async_simple(self):
        """Simple async test."""
        import asyncio
        await asyncio.sleep(0.001)
        assert True

    @pytest.mark.asyncio
    async def test_async_math(self):
        """Async math test."""
        result = await self._async_add(2, 3)
        assert result == 5

    async def _async_add(self, a, b):
        return a + b


class TestSQLAlchemyImports:
    """Test that SQLAlchemy imports work."""

    def test_sqlalchemy_import(self):
        """Verify SQLAlchemy can be imported."""
        from sqlalchemy import create_engine
        assert create_engine is not None

    def test_async_sqlalchemy_import(self):
        """Verify async SQLAlchemy can be imported."""
        from sqlalchemy.ext.asyncio import create_async_engine
        assert create_async_engine is not None


class TestConfTestImports:
    """Test that conftest models can be imported."""

    def test_import_user_model(self):
        """Verify User model can be imported from conftest."""
        from .conftest import User
        assert User is not None
        assert User.__tablename__ == "users"

    def test_import_all_models(self):
        """Verify all models can be imported."""
        from .conftest import (
            User, Course, Enrollment, Concept,
            SpacedRepetitionCard, ReviewLog, UserAchievement
        )
        assert all([User, Course, Enrollment, Concept,
                   SpacedRepetitionCard, ReviewLog, UserAchievement])


class TestSyncEngine:
    """Test sync engine fixture."""

    def test_sync_engine_creates_tables(self, sync_engine):
        """Verify sync_engine fixture creates tables."""
        from sqlalchemy import inspect
        inspector = inspect(sync_engine)
        tables = inspector.get_table_names()
        assert "users" in tables
        assert "courses" in tables

    def test_sync_session_works(self, sync_session):
        """Verify sync_session can execute queries."""
        from sqlalchemy import text
        result = sync_session.execute(text("SELECT 1"))
        assert result.scalar() == 1
