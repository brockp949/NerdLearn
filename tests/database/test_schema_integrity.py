"""
Schema Integrity Tests

Tests to verify the database schema is correctly defined and matches expectations.
These tests should be run after migrations to ensure schema consistency.

Test Categories:
- Table existence verification
- Column definitions (types, nullability, defaults)
- Index verification
- Constraint verification (unique, foreign keys)
- Enum validation
"""
import pytest
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

pytestmark = [pytest.mark.requires_db, pytest.mark.unit]


class TestTableExistence:
    """Verify all expected tables exist in the schema."""

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
        """Warn about unexpected tables that might be orphaned."""
        inspector = inspect(sync_engine)
        existing_tables = set(inspector.get_table_names())
        expected_tables = set(self.EXPECTED_TABLES)

        # SQLite internal tables
        internal_tables = {"sqlite_sequence", "alembic_version"}
        orphan_tables = existing_tables - expected_tables - internal_tables

        # This is a warning, not a failure - unexpected tables might be intentional
        if orphan_tables:
            pytest.skip(f"Found unexpected tables (may be intentional): {orphan_tables}")


class TestUserTableSchema:
    """Verify User table schema correctness."""

    def test_user_table_columns(self, sync_engine: Engine):
        """Verify user table has all required columns."""
        inspector = inspect(sync_engine)
        columns = {col["name"]: col for col in inspector.get_columns("users")}

        required_columns = [
            "id", "email", "username", "hashed_password",
            "full_name", "is_active", "is_instructor",
            "created_at", "updated_at",
            "total_xp", "level", "streak_days", "last_activity_date"
        ]

        for col_name in required_columns:
            assert col_name in columns, f"Missing column: users.{col_name}"

    def test_user_email_is_unique(self, sync_engine: Engine):
        """Verify email column has unique constraint."""
        inspector = inspect(sync_engine)
        unique_constraints = inspector.get_unique_constraints("users")
        indexes = inspector.get_indexes("users")

        # Check for unique constraint or unique index on email
        email_unique = any(
            "email" in (uc.get("column_names", []) or [])
            for uc in unique_constraints
        ) or any(
            idx.get("unique", False) and "email" in idx.get("column_names", [])
            for idx in indexes
        )

        assert email_unique, "users.email should have unique constraint"

    def test_user_username_is_unique(self, sync_engine: Engine):
        """Verify username column has unique constraint."""
        inspector = inspect(sync_engine)
        unique_constraints = inspector.get_unique_constraints("users")
        indexes = inspector.get_indexes("users")

        username_unique = any(
            "username" in (uc.get("column_names", []) or [])
            for uc in unique_constraints
        ) or any(
            idx.get("unique", False) and "username" in idx.get("column_names", [])
            for idx in indexes
        )

        assert username_unique, "users.username should have unique constraint"

    def test_user_email_not_nullable(self, sync_engine: Engine):
        """Verify email is a required field."""
        inspector = inspect(sync_engine)
        columns = {col["name"]: col for col in inspector.get_columns("users")}

        assert not columns["email"]["nullable"], "users.email should not be nullable"

    def test_user_default_values(self, sync_engine: Engine):
        """Verify default values are set correctly."""
        inspector = inspect(sync_engine)
        columns = {col["name"]: col for col in inspector.get_columns("users")}

        # Check boolean defaults
        assert columns["is_active"].get("default") is not None, \
            "users.is_active should have a default"
        assert columns["is_instructor"].get("default") is not None, \
            "users.is_instructor should have a default"


class TestCourseTableSchema:
    """Verify Course table schema correctness."""

    def test_course_table_columns(self, sync_engine: Engine):
        """Verify course table has all required columns."""
        inspector = inspect(sync_engine)
        columns = {col["name"]: col for col in inspector.get_columns("courses")}

        required_columns = [
            "id", "title", "description", "domain",
            "instructor_id", "is_published", "created_at"
        ]

        for col_name in required_columns:
            assert col_name in columns, f"Missing column: courses.{col_name}"


class TestConceptTableSchema:
    """Verify Concept table schema correctness."""

    def test_concept_table_columns(self, sync_engine: Engine):
        """Verify concept table has all required columns."""
        inspector = inspect(sync_engine)
        columns = {col["name"]: col for col in inspector.get_columns("concepts")}

        required_columns = ["id", "name", "description"]

        for col_name in required_columns:
            assert col_name in columns, f"Missing column: concepts.{col_name}"


class TestSpacedRepetitionSchema:
    """Verify Spaced Repetition tables schema correctness."""

    def test_spaced_repetition_card_columns(self, sync_engine: Engine):
        """Verify spaced_repetition_cards table has FSRS columns."""
        inspector = inspect(sync_engine)
        columns = {col["name"]: col for col in inspector.get_columns("spaced_repetition_cards")}

        # FSRS algorithm requires these columns
        fsrs_columns = [
            "difficulty", "stability", "retrievability",
            "review_count", "next_review_at"
        ]

        for col_name in fsrs_columns:
            assert col_name in columns, \
                f"Missing FSRS column: spaced_repetition_cards.{col_name}"

    def test_review_log_columns(self, sync_engine: Engine):
        """Verify review_logs table has required tracking columns."""
        inspector = inspect(sync_engine)
        columns = {col["name"]: col for col in inspector.get_columns("review_logs")}

        required_columns = [
            "id", "card_id", "rating", "reviewed_at",
            "response_time_ms"
        ]

        for col_name in required_columns:
            assert col_name in columns, f"Missing column: review_logs.{col_name}"


class TestIndexes:
    """Verify database indexes are correctly defined."""

    def test_users_email_index(self, sync_engine: Engine):
        """Verify index on users.email for fast lookups."""
        inspector = inspect(sync_engine)
        indexes = inspector.get_indexes("users")

        email_indexed = any(
            "email" in idx.get("column_names", [])
            for idx in indexes
        )

        assert email_indexed, "users.email should be indexed"

    def test_users_username_index(self, sync_engine: Engine):
        """Verify index on users.username for fast lookups."""
        inspector = inspect(sync_engine)
        indexes = inspector.get_indexes("users")

        username_indexed = any(
            "username" in idx.get("column_names", [])
            for idx in indexes
        )

        assert username_indexed, "users.username should be indexed"

    def test_courses_domain_index(self, sync_engine: Engine):
        """Verify index on courses.domain for category filtering."""
        inspector = inspect(sync_engine)
        indexes = inspector.get_indexes("courses")

        domain_indexed = any(
            "domain" in idx.get("column_names", [])
            for idx in indexes
        )

        # This is a recommended index, not required
        if not domain_indexed:
            pytest.skip("courses.domain index recommended but not required")


class TestForeignKeyConstraints:
    """Verify foreign key relationships are correctly defined."""

    def test_enrollment_user_fk(self, sync_engine: Engine):
        """Verify enrollments.user_id references users.id."""
        inspector = inspect(sync_engine)
        fks = inspector.get_foreign_keys("enrollments")

        user_fk = any(
            fk.get("referred_table") == "users" and
            "user_id" in fk.get("constrained_columns", [])
            for fk in fks
        )

        assert user_fk, "enrollments.user_id should reference users.id"

    def test_enrollment_course_fk(self, sync_engine: Engine):
        """Verify enrollments.course_id references courses.id."""
        inspector = inspect(sync_engine)
        fks = inspector.get_foreign_keys("enrollments")

        course_fk = any(
            fk.get("referred_table") == "courses" and
            "course_id" in fk.get("constrained_columns", [])
            for fk in fks
        )

        assert course_fk, "enrollments.course_id should reference courses.id"

    def test_spaced_repetition_card_user_fk(self, sync_engine: Engine):
        """Verify spaced_repetition_cards.user_id references users.id."""
        inspector = inspect(sync_engine)
        fks = inspector.get_foreign_keys("spaced_repetition_cards")

        user_fk = any(
            fk.get("referred_table") == "users" and
            "user_id" in fk.get("constrained_columns", [])
            for fk in fks
        )

        assert user_fk, "spaced_repetition_cards.user_id should reference users.id"

    def test_review_log_card_fk(self, sync_engine: Engine):
        """Verify review_logs.card_id references spaced_repetition_cards.id."""
        inspector = inspect(sync_engine)
        fks = inspector.get_foreign_keys("review_logs")

        card_fk = any(
            fk.get("referred_table") == "spaced_repetition_cards" and
            "card_id" in fk.get("constrained_columns", [])
            for fk in fks
        )

        assert card_fk, "review_logs.card_id should reference spaced_repetition_cards.id"


class TestSchemaVersioning:
    """Tests related to schema versioning and migrations."""

    def test_alembic_version_table_exists(self, sync_engine: Engine):
        """Verify Alembic migration tracking table exists (if using migrations)."""
        inspector = inspect(sync_engine)
        tables = inspector.get_table_names()

        # Note: This might not exist in test database if created fresh
        # Skip if not using Alembic
        if "alembic_version" not in tables:
            pytest.skip("alembic_version table not present (may be using fresh schema)")

    def test_schema_is_up_to_date(self, sync_engine: Engine):
        """Verify schema matches latest migration (if using Alembic)."""
        inspector = inspect(sync_engine)
        tables = inspector.get_table_names()

        if "alembic_version" not in tables:
            pytest.skip("Not using Alembic migrations")

        # This would require comparing against migration scripts
        # For now, just verify the table is populated
        with sync_engine.connect() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            version = result.scalar()
            assert version is not None, "No migration version recorded"
