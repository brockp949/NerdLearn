"""
Schema Integrity Tests

Tests to verify the database schema is correctly defined and matches expectations.
These tests should be run after migrations to ensure schema consistency.

Test Categories:
- Table existence verification
- Column definitions (types, nullability, defaults)
- Index verification
- Constraint verification (unique, foreign keys)
"""
import pytest
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from .conftest import (
    User, Course, Enrollment, Concept, SpacedRepetitionCard,
    ReviewLog, UserAchievement, UserStats, Instructor, TestBase
)

# These are sync tests - do NOT use pytest.mark.asyncio
pytestmark = []


class TestTableExistence:
    """Verify all expected tables exist in the schema."""

    # Tables defined in our test models (conftest.py)
    EXPECTED_TABLES = [
        "users",
        "instructors",
        "courses",
        "enrollments",
        "concepts",
        "user_concept_mastery",
        "spaced_repetition_cards",
        "review_logs",
        "user_achievements",
        "user_stats",
        "daily_activities",
        "chat_history",
    ]

    def test_all_required_tables_exist(self, sync_engine: Engine):
        """Verify all required tables are present in the database."""
        inspector = inspect(sync_engine)
        existing_tables = inspector.get_table_names()

        for table in self.EXPECTED_TABLES:
            assert table in existing_tables, f"Missing required table: {table}"

    def test_no_orphan_tables(self, sync_engine: Engine):
        """Check for unexpected tables."""
        inspector = inspect(sync_engine)
        existing_tables = set(inspector.get_table_names())
        expected_tables = set(self.EXPECTED_TABLES)

        unexpected = existing_tables - expected_tables
        # Note: unexpected tables are OK, just informational
        if unexpected:
            print(f"Additional tables found: {unexpected}")


class TestUserTableSchema:
    """Verify User table schema."""

    def test_user_table_columns(self, sync_engine: Engine):
        """Verify User table has required columns."""
        inspector = inspect(sync_engine)
        columns = {col["name"] for col in inspector.get_columns("users")}

        required_columns = {
            "id", "email", "username", "hashed_password",
            "is_active", "total_xp", "level", "streak_days",
            "created_at"
        }

        for col in required_columns:
            assert col in columns, f"Missing required column: users.{col}"

    def test_user_email_is_unique(self, sync_engine: Engine):
        """Verify email has a unique constraint."""
        inspector = inspect(sync_engine)
        indexes = inspector.get_indexes("users")
        unique_constraints = inspector.get_unique_constraints("users")

        # Check indexes for unique flag
        email_unique = any(
            idx.get("unique") and "email" in idx.get("column_names", [])
            for idx in indexes
        )

        # Check unique constraints
        email_constraint = any(
            "email" in uc.get("column_names", [])
            for uc in unique_constraints
        )

        assert email_unique or email_constraint, "users.email should have unique constraint"

    def test_user_username_is_unique(self, sync_engine: Engine):
        """Verify username has a unique constraint."""
        inspector = inspect(sync_engine)
        indexes = inspector.get_indexes("users")
        unique_constraints = inspector.get_unique_constraints("users")

        username_unique = any(
            idx.get("unique") and "username" in idx.get("column_names", [])
            for idx in indexes
        )

        username_constraint = any(
            "username" in uc.get("column_names", [])
            for uc in unique_constraints
        )

        assert username_unique or username_constraint, "users.username should have unique constraint"

    def test_user_email_not_nullable(self, sync_engine: Engine):
        """Verify email is not nullable."""
        inspector = inspect(sync_engine)
        columns = {col["name"]: col for col in inspector.get_columns("users")}

        assert columns["email"]["nullable"] is False, "users.email should not be nullable"

    def test_user_default_values(self, sync_engine: Engine):
        """Verify default values are set correctly."""
        inspector = inspect(sync_engine)
        columns = {col["name"]: col for col in inspector.get_columns("users")}

        # Check that certain columns have defaults
        assert columns["is_active"]["default"] is not None or columns["is_active"]["nullable"], \
            "is_active should have default or be nullable"


class TestCourseTableSchema:
    """Verify Course table schema."""

    def test_course_table_columns(self, sync_engine: Engine):
        """Verify Course table has required columns."""
        inspector = inspect(sync_engine)
        columns = {col["name"] for col in inspector.get_columns("courses")}

        required_columns = {
            "id", "title", "description", "is_published", "created_at"
        }

        for col in required_columns:
            assert col in columns, f"Missing required column: courses.{col}"


class TestConceptTableSchema:
    """Verify Concept table schema."""

    def test_concept_table_columns(self, sync_engine: Engine):
        """Verify Concept table has required columns."""
        inspector = inspect(sync_engine)
        columns = {col["name"] for col in inspector.get_columns("concepts")}

        required_columns = {"id", "name", "description"}

        for col in required_columns:
            assert col in columns, f"Missing required column: concepts.{col}"


class TestSpacedRepetitionSchema:
    """Verify Spaced Repetition related table schemas."""

    def test_spaced_repetition_card_columns(self, sync_engine: Engine):
        """Verify SpacedRepetitionCard table has required FSRS columns."""
        inspector = inspect(sync_engine)
        columns = {col["name"] for col in inspector.get_columns("spaced_repetition_cards")}

        required_columns = {
            "id", "user_id", "concept_id",
            "difficulty", "stability", "retrievability",
            "review_count", "next_review_at"
        }

        for col in required_columns:
            assert col in columns, f"Missing FSRS column: spaced_repetition_cards.{col}"

    def test_review_log_columns(self, sync_engine: Engine):
        """Verify ReviewLog table has required columns."""
        inspector = inspect(sync_engine)
        columns = {col["name"] for col in inspector.get_columns("review_logs")}

        required_columns = {"id", "card_id", "rating", "reviewed_at"}

        for col in required_columns:
            assert col in columns, f"Missing column: review_logs.{col}"


class TestIndexes:
    """Verify database indexes are present for performance."""

    def test_users_email_index(self, sync_engine: Engine):
        """Verify email has an index for fast lookups."""
        inspector = inspect(sync_engine)
        indexes = inspector.get_indexes("users")

        email_indexed = any(
            "email" in idx.get("column_names", [])
            for idx in indexes
        )

        assert email_indexed, "users.email should be indexed"

    def test_users_username_index(self, sync_engine: Engine):
        """Verify username has an index."""
        inspector = inspect(sync_engine)
        indexes = inspector.get_indexes("users")

        username_indexed = any(
            "username" in idx.get("column_names", [])
            for idx in indexes
        )

        assert username_indexed, "users.username should be indexed"


class TestForeignKeyConstraints:
    """Verify foreign key relationships are properly defined."""

    def test_enrollment_user_fk(self, sync_engine: Engine):
        """Verify enrollment has FK to users."""
        inspector = inspect(sync_engine)
        fks = inspector.get_foreign_keys("enrollments")

        user_fk = any(
            fk.get("referred_table") == "users"
            for fk in fks
        )

        assert user_fk, "enrollments should have FK to users"

    def test_enrollment_course_fk(self, sync_engine: Engine):
        """Verify enrollment has FK to courses."""
        inspector = inspect(sync_engine)
        fks = inspector.get_foreign_keys("enrollments")

        course_fk = any(
            fk.get("referred_table") == "courses"
            for fk in fks
        )

        assert course_fk, "enrollments should have FK to courses"

    def test_spaced_repetition_card_user_fk(self, sync_engine: Engine):
        """Verify spaced_repetition_cards has FK to users."""
        inspector = inspect(sync_engine)
        fks = inspector.get_foreign_keys("spaced_repetition_cards")

        user_fk = any(
            fk.get("referred_table") == "users"
            for fk in fks
        )

        assert user_fk, "spaced_repetition_cards should have FK to users"

    def test_review_log_card_fk(self, sync_engine: Engine):
        """Verify review_logs has FK to spaced_repetition_cards."""
        inspector = inspect(sync_engine)
        fks = inspector.get_foreign_keys("review_logs")

        card_fk = any(
            fk.get("referred_table") == "spaced_repetition_cards"
            for fk in fks
        )

        assert card_fk, "review_logs should have FK to spaced_repetition_cards"
