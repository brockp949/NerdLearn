"""
Example Database Verifier Test

Demonstrates how to use the DatabaseVerifier agent within the
Antigravity testing framework for comprehensive database testing.

Usage:
    python -m pytest apps/testing/agents/tests/example_database_test.py -v

Or run specific categories:
    python -c "
    import asyncio
    from apps.testing.agents import DatabaseVerifier, create_database_verifier, MockLLMClient

    async def main():
        verifier = create_database_verifier(MockLLMClient())
        report = await verifier.audit_database()
        print(report.to_markdown())

    asyncio.run(main())
    "
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from apps.testing.agents.database_verifier import (
    DatabaseVerifier,
    DatabaseTestCategory,
    DatabaseTestResult,
    DatabaseAuditReport,
    create_database_verifier,
    DATABASE_GOAL_VECTORS
)
from apps.testing.agents.base_verifier import GoalVector


class MockLLMClient:
    """Mock LLM client for testing."""

    async def generate(self, prompt: str, **kwargs) -> str:
        """Return mock analysis response."""
        return '''```json
{
    "analysis": "Database tests show good overall health. Schema integrity is maintained and CRUD operations function correctly. Some performance optimizations may be beneficial.",
    "recommendations": [
        "Consider adding indexes for frequently queried columns",
        "Review cascade delete behavior for user entity",
        "Add more edge case tests for concurrent operations"
    ]
}
```'''


class TestDatabaseVerifierInitialization:
    """Tests for DatabaseVerifier initialization."""

    def test_create_with_default_goal_vector(self):
        """Test creation with default goal vector."""
        llm = MockLLMClient()
        verifier = DatabaseVerifier(llm)

        assert verifier.goal.name == "Database Integrity"
        assert verifier.goal.pass_threshold == 0.85
        assert len(verifier.goal.constraints) > 0

    def test_create_with_custom_goal_vector(self):
        """Test creation with custom goal vector."""
        llm = MockLLMClient()
        custom_goal = GoalVector(
            name="Custom Database Goal",
            description="Custom testing objective",
            constraints=["Custom constraint 1"],
            pass_threshold=0.90
        )
        verifier = DatabaseVerifier(llm, goal_vector=custom_goal)

        assert verifier.goal.name == "Custom Database Goal"
        assert verifier.goal.pass_threshold == 0.90

    def test_create_with_project_root(self):
        """Test creation with specified project root."""
        llm = MockLLMClient()
        project_root = Path("/custom/project/root")
        verifier = DatabaseVerifier(llm, project_root=project_root)

        assert verifier.project_root == project_root
        assert verifier.tests_path == project_root / "tests/database"


class TestDatabaseGoalVectors:
    """Tests for predefined database goal vectors."""

    def test_schema_integrity_goal_vector(self):
        """Test schema integrity goal vector."""
        goal = DATABASE_GOAL_VECTORS['schema_integrity']
        assert goal.name == "Schema Integrity"
        assert goal.pass_threshold == 0.95

    def test_data_integrity_goal_vector(self):
        """Test data integrity goal vector."""
        goal = DATABASE_GOAL_VECTORS['data_integrity']
        assert goal.name == "Data Integrity"
        assert goal.pass_threshold == 0.90

    def test_performance_goal_vector(self):
        """Test performance goal vector."""
        goal = DATABASE_GOAL_VECTORS['performance']
        assert goal.name == "Performance Requirements"
        assert goal.pass_threshold == 0.80

    def test_concurrency_goal_vector(self):
        """Test concurrency goal vector."""
        goal = DATABASE_GOAL_VECTORS['concurrency']
        assert goal.name == "Concurrency Safety"
        assert goal.pass_threshold == 0.85


class TestDatabaseTestResult:
    """Tests for DatabaseTestResult dataclass."""

    def test_pass_rate_calculation(self):
        """Test pass rate calculation."""
        result = DatabaseTestResult(
            category=DatabaseTestCategory.SCHEMA,
            total_tests=10,
            passed=8,
            failed=1,
            skipped=1,
            duration_seconds=5.0
        )

        assert result.pass_rate == 0.8

    def test_pass_rate_zero_tests(self):
        """Test pass rate with zero tests."""
        result = DatabaseTestResult(
            category=DatabaseTestCategory.CRUD,
            total_tests=0,
            passed=0,
            failed=0,
            skipped=0,
            duration_seconds=0.0
        )

        assert result.pass_rate == 0.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = DatabaseTestResult(
            category=DatabaseTestCategory.PERFORMANCE,
            total_tests=5,
            passed=5,
            failed=0,
            skipped=0,
            duration_seconds=2.5,
            failures=[],
            warnings=["Consider adding more tests"]
        )

        data = result.to_dict()
        assert data['category'] == 'performance'
        assert data['total_tests'] == 5
        assert data['pass_rate'] == 1.0


class TestDatabaseAuditReport:
    """Tests for DatabaseAuditReport dataclass."""

    def test_overall_metrics(self):
        """Test overall metrics calculation."""
        results = [
            DatabaseTestResult(
                category=DatabaseTestCategory.SCHEMA,
                total_tests=10, passed=10, failed=0, skipped=0,
                duration_seconds=1.0
            ),
            DatabaseTestResult(
                category=DatabaseTestCategory.CRUD,
                total_tests=20, passed=18, failed=2, skipped=0,
                duration_seconds=2.0
            ),
        ]

        report = DatabaseAuditReport(
            results=results,
            overall_passed=False,
            overall_confidence=0.85,
            semantic_analysis="Test analysis",
            recommendations=["Recommendation 1"]
        )

        assert report.total_tests == 30
        assert report.total_passed == 28
        assert report.overall_pass_rate == pytest.approx(0.933, rel=0.01)

    def test_to_markdown(self):
        """Test markdown report generation."""
        results = [
            DatabaseTestResult(
                category=DatabaseTestCategory.SCHEMA,
                total_tests=5, passed=5, failed=0, skipped=0,
                duration_seconds=1.0
            ),
        ]

        report = DatabaseAuditReport(
            results=results,
            overall_passed=True,
            overall_confidence=0.95,
            semantic_analysis="All tests passed.",
            recommendations=["Continue monitoring"]
        )

        markdown = report.to_markdown()
        assert "PASSED" in markdown
        assert "95.0%" in markdown
        assert "schema" in markdown


class TestDatabaseVerifierAudit:
    """Tests for DatabaseVerifier audit functionality."""

    @pytest.mark.asyncio
    async def test_audit_with_mocked_tests(self):
        """Test audit with mocked pytest execution."""
        llm = MockLLMClient()
        verifier = DatabaseVerifier(llm)

        # Mock subprocess.run to simulate test results
        mock_result = MagicMock()
        mock_result.stdout = "5 passed, 1 failed in 2.5s"
        mock_result.stderr = ""
        mock_result.returncode = 1

        with patch('subprocess.run', return_value=mock_result):
            result = await verifier.run_category_tests(DatabaseTestCategory.SCHEMA)

        assert result.category == DatabaseTestCategory.SCHEMA
        assert result.passed == 5
        assert result.failed == 1

    @pytest.mark.asyncio
    async def test_full_audit_flow(self):
        """Test complete audit flow."""
        llm = MockLLMClient()
        verifier = DatabaseVerifier(llm)

        # Mock all subprocess calls
        mock_result = MagicMock()
        mock_result.stdout = "10 passed in 1.0s"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch('subprocess.run', return_value=mock_result):
            report = await verifier.audit_database(
                categories=[DatabaseTestCategory.SCHEMA, DatabaseTestCategory.CRUD]
            )

        assert len(report.results) == 2
        assert report.overall_passed is True
        assert report.overall_confidence > 0

    @pytest.mark.asyncio
    async def test_get_quick_status_no_audits(self):
        """Test quick status with no audits run."""
        llm = MockLLMClient()
        verifier = DatabaseVerifier(llm)

        status = verifier.get_quick_status()
        assert status['status'] == 'no_audits'

    @pytest.mark.asyncio
    async def test_get_quick_status_after_audit(self):
        """Test quick status after running audit."""
        llm = MockLLMClient()
        verifier = DatabaseVerifier(llm)

        mock_result = MagicMock()
        mock_result.stdout = "5 passed in 1.0s"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch('subprocess.run', return_value=mock_result):
            await verifier.audit_database(categories=[DatabaseTestCategory.SCHEMA])

        status = verifier.get_quick_status()
        assert status['status'] in ['passed', 'failed']
        assert 'confidence' in status


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_database_verifier(self):
        """Test the convenience function."""
        llm = MockLLMClient()
        verifier = create_database_verifier(llm)

        assert isinstance(verifier, DatabaseVerifier)
        assert verifier.goal.name == "Database Integrity"


# Run tests directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
