"""
Database Verifier Agent for Antigravity Testing Framework.

Integrates comprehensive database tests with semantic verification:
- Schema integrity tests
- Data integrity tests
- CRUD operation tests
- Relationship tests
- Performance benchmarks
- Concurrency tests

Uses goal vectors to ensure database operations meet semantic correctness
beyond just syntactic validity.
"""

import asyncio
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
import json
import logging

from .base_verifier import VerifierAgent, GoalVector, VerificationResult

logger = logging.getLogger(__name__)


class DatabaseTestCategory(Enum):
    """Categories of database tests aligned with goal vectors."""
    SCHEMA = "schema"
    DATA_INTEGRITY = "data_integrity"
    CRUD = "crud"
    RELATIONSHIPS = "relationships"
    PERFORMANCE = "performance"
    CONCURRENCY = "concurrency"


@dataclass
class DatabaseTestResult:
    """Result from running database test suite."""
    category: DatabaseTestCategory
    total_tests: int
    passed: int
    failed: int
    skipped: int
    duration_seconds: float
    failures: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if self.total_tests == 0:
            return 0.0
        return self.passed / self.total_tests

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'category': self.category.value,
            'total_tests': self.total_tests,
            'passed': self.passed,
            'failed': self.failed,
            'skipped': self.skipped,
            'pass_rate': self.pass_rate,
            'duration_seconds': self.duration_seconds,
            'failures': self.failures,
            'warnings': self.warnings,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class DatabaseAuditReport:
    """Comprehensive database audit report."""
    results: List[DatabaseTestResult]
    overall_passed: bool
    overall_confidence: float
    semantic_analysis: str
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total_tests(self) -> int:
        return sum(r.total_tests for r in self.results)

    @property
    def total_passed(self) -> int:
        return sum(r.passed for r in self.results)

    @property
    def overall_pass_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.total_passed / self.total_tests

    def to_dict(self) -> Dict[str, Any]:
        return {
            'results': [r.to_dict() for r in self.results],
            'overall_passed': self.overall_passed,
            'overall_confidence': self.overall_confidence,
            'overall_pass_rate': self.overall_pass_rate,
            'semantic_analysis': self.semantic_analysis,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat()
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        status = "PASSED" if self.overall_passed else "FAILED"
        emoji = "✅" if self.overall_passed else "❌"

        md = f"""# Database Audit Report {emoji}

**Status:** {status}
**Confidence:** {self.overall_confidence:.1%}
**Overall Pass Rate:** {self.overall_pass_rate:.1%}
**Timestamp:** {self.timestamp.isoformat()}

## Test Results by Category

| Category | Total | Passed | Failed | Skipped | Pass Rate | Duration |
|----------|-------|--------|--------|---------|-----------|----------|
"""
        for result in self.results:
            status_icon = "✅" if result.failed == 0 else "❌"
            md += f"| {status_icon} {result.category.value} | {result.total_tests} | {result.passed} | {result.failed} | {result.skipped} | {result.pass_rate:.1%} | {result.duration_seconds:.2f}s |\n"

        md += f"""
## Semantic Analysis

{self.semantic_analysis}

## Recommendations

"""
        for rec in self.recommendations:
            md += f"- {rec}\n"

        if any(r.failures for r in self.results):
            md += "\n## Failures\n\n"
            for result in self.results:
                if result.failures:
                    md += f"### {result.category.value}\n\n"
                    for failure in result.failures:
                        md += f"- **{failure.get('test', 'Unknown')}**: {failure.get('message', 'No message')}\n"

        return md


class LLMClient(Protocol):
    """Protocol for LLM client interface."""
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        ...


class DatabaseVerifier(VerifierAgent):
    """
    Database verification agent using Antigravity testing principles.

    Combines traditional pytest-based database tests with LLM-powered
    semantic analysis to ensure database operations are not just
    syntactically correct but semantically aligned with system goals.

    Goal Vectors for Database Testing:
    - Schema Integrity: Tables, columns, indexes match design
    - Data Integrity: Constraints enforced, no orphans
    - Performance: Operations meet latency requirements
    - Concurrency: Handles parallel access correctly
    """

    # Path to database tests directory (relative to project root)
    DATABASE_TESTS_PATH = Path("tests/database")

    # Category to pytest marker mapping
    CATEGORY_MARKERS = {
        DatabaseTestCategory.SCHEMA: "schema",
        DatabaseTestCategory.DATA_INTEGRITY: "data_integrity",
        DatabaseTestCategory.CRUD: "crud",
        DatabaseTestCategory.RELATIONSHIPS: "relationships",
        DatabaseTestCategory.PERFORMANCE: "performance",
        DatabaseTestCategory.CONCURRENCY: "concurrency",
    }

    def __init__(
        self,
        llm_client: LLMClient,
        project_root: Optional[Path] = None,
        goal_vector: Optional[GoalVector] = None
    ):
        """
        Initialize database verifier.

        Args:
            llm_client: LLM interface for semantic validation
            project_root: Root directory of the project
            goal_vector: Testing objective (uses default if not provided)
        """
        if goal_vector is None:
            goal_vector = self._create_default_goal_vector()

        super().__init__(goal_vector, llm_client)

        self.project_root = project_root or Path.cwd()
        self.tests_path = self.project_root / self.DATABASE_TESTS_PATH
        self.audit_reports: List[DatabaseAuditReport] = []

    def _create_default_goal_vector(self) -> GoalVector:
        """Create default goal vector for database testing."""
        return GoalVector(
            name="Database Integrity",
            description="Ensure database operations maintain data integrity, "
                       "performance requirements, and correct relationships",
            constraints=[
                "All schema elements match design specifications",
                "Foreign key relationships are properly enforced",
                "Unique constraints prevent duplicate data",
                "Performance meets latency thresholds",
                "Concurrent operations maintain consistency",
                "No orphaned records or dangling references"
            ],
            pass_threshold=0.85
        )

    async def run_category_tests(
        self,
        category: DatabaseTestCategory,
        verbose: bool = False
    ) -> DatabaseTestResult:
        """
        Run tests for a specific category.

        Args:
            category: Test category to run
            verbose: Include verbose output

        Returns:
            DatabaseTestResult with test outcomes
        """
        logger.info(f"Running {category.value} tests...")

        marker = self.CATEGORY_MARKERS[category]

        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_path),
            "-m", marker,
            "--tb=short",
            "-q",
            "--json-report",
            "--json-report-file=-"  # Output to stdout
        ]

        if verbose:
            cmd.append("-v")

        start_time = datetime.now()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=str(self.project_root)
            )

            duration = (datetime.now() - start_time).total_seconds()

            # Try to parse JSON report from output
            test_result = self._parse_pytest_output(
                category, result.stdout, result.stderr, result.returncode, duration
            )

            return test_result

        except subprocess.TimeoutExpired:
            logger.error(f"Tests for {category.value} timed out")
            return DatabaseTestResult(
                category=category,
                total_tests=0,
                passed=0,
                failed=1,
                skipped=0,
                duration_seconds=300,
                failures=[{"test": "timeout", "message": "Test suite timed out"}]
            )
        except Exception as e:
            logger.error(f"Error running {category.value} tests: {e}")
            return DatabaseTestResult(
                category=category,
                total_tests=0,
                passed=0,
                failed=1,
                skipped=0,
                duration_seconds=0,
                failures=[{"test": "error", "message": str(e)}]
            )

    def _parse_pytest_output(
        self,
        category: DatabaseTestCategory,
        stdout: str,
        stderr: str,
        returncode: int,
        duration: float
    ) -> DatabaseTestResult:
        """Parse pytest output to extract test results."""

        # Try to extract counts from standard pytest output
        # Format: "X passed, Y failed, Z skipped in Ns"
        passed = failed = skipped = 0
        total = 0
        failures = []

        lines = stdout.split('\n') + stderr.split('\n')

        for line in lines:
            line = line.strip()

            # Parse summary line
            if 'passed' in line or 'failed' in line or 'skipped' in line:
                import re

                match = re.search(r'(\d+)\s+passed', line)
                if match:
                    passed = int(match.group(1))

                match = re.search(r'(\d+)\s+failed', line)
                if match:
                    failed = int(match.group(1))

                match = re.search(r'(\d+)\s+skipped', line)
                if match:
                    skipped = int(match.group(1))

            # Capture failure info
            if line.startswith('FAILED') or line.startswith('ERROR'):
                failures.append({
                    'test': line.split('::')[-1] if '::' in line else line,
                    'message': line
                })

        total = passed + failed + skipped

        # If we couldn't parse, estimate from return code
        if total == 0:
            if returncode == 0:
                passed = 1
                total = 1
            else:
                failed = 1
                total = 1
                failures.append({
                    'test': 'unknown',
                    'message': f'Exit code {returncode}'
                })

        return DatabaseTestResult(
            category=category,
            total_tests=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            duration_seconds=duration,
            failures=failures
        )

    async def run_all_tests(self, verbose: bool = False) -> List[DatabaseTestResult]:
        """
        Run all database test categories.

        Args:
            verbose: Include verbose output

        Returns:
            List of DatabaseTestResult for each category
        """
        results = []

        for category in DatabaseTestCategory:
            result = await self.run_category_tests(category, verbose)
            results.append(result)
            logger.info(
                f"{category.value}: {result.passed}/{result.total_tests} passed "
                f"({result.pass_rate:.1%})"
            )

        return results

    async def audit_database(
        self,
        categories: Optional[List[DatabaseTestCategory]] = None,
        verbose: bool = False
    ) -> DatabaseAuditReport:
        """
        Perform comprehensive database audit with semantic analysis.

        This is the main entry point for Antigravity database testing.
        It runs tests, collects results, and uses LLM to provide
        semantic analysis and recommendations.

        Args:
            categories: Specific categories to test (all if None)
            verbose: Include verbose output

        Returns:
            DatabaseAuditReport with full analysis
        """
        logger.info("Starting database audit...")

        # Run tests
        if categories is None:
            results = await self.run_all_tests(verbose)
        else:
            results = []
            for cat in categories:
                result = await self.run_category_tests(cat, verbose)
                results.append(result)

        # Calculate overall metrics
        total_tests = sum(r.total_tests for r in results)
        total_passed = sum(r.passed for r in results)
        total_failed = sum(r.failed for r in results)

        # Determine if audit passed
        overall_passed = all(r.failed == 0 for r in results)

        # Calculate confidence based on pass rates and coverage
        if total_tests > 0:
            base_confidence = total_passed / total_tests
            # Adjust for coverage (more tests = higher confidence)
            coverage_factor = min(1.0, total_tests / 100)  # Cap at 100 tests
            overall_confidence = base_confidence * (0.7 + 0.3 * coverage_factor)
        else:
            overall_confidence = 0.0

        # Get semantic analysis from LLM
        semantic_analysis, recommendations = await self._get_semantic_analysis(results)

        # Apply goal alignment check
        goal_alignment = await self._compute_goal_alignment(
            {"results": [r.to_dict() for r in results]},
            None
        )

        # Adjust overall_passed based on goal alignment
        if goal_alignment < self.goal.pass_threshold:
            overall_passed = False
            overall_confidence *= goal_alignment

        report = DatabaseAuditReport(
            results=results,
            overall_passed=overall_passed,
            overall_confidence=overall_confidence,
            semantic_analysis=semantic_analysis,
            recommendations=recommendations
        )

        self.audit_reports.append(report)

        logger.info(
            f"Database audit complete: {'PASSED' if overall_passed else 'FAILED'} "
            f"(confidence: {overall_confidence:.1%})"
        )

        return report

    async def _get_semantic_analysis(
        self,
        results: List[DatabaseTestResult]
    ) -> tuple[str, List[str]]:
        """
        Get LLM-powered semantic analysis of test results.

        Returns:
            Tuple of (analysis_text, recommendations_list)
        """
        # Build analysis prompt
        results_summary = "\n".join([
            f"- {r.category.value}: {r.passed}/{r.total_tests} passed "
            f"({r.pass_rate:.1%}), {len(r.failures)} failures"
            for r in results
        ])

        failures_detail = []
        for r in results:
            for f in r.failures:
                failures_detail.append(f"[{r.category.value}] {f.get('test', 'Unknown')}: {f.get('message', 'No details')}")

        failures_text = "\n".join(failures_detail) if failures_detail else "No failures"

        prompt = f"""You are a Database Quality Auditor Agent. Analyze the following database test results and provide:
1. A brief semantic analysis of the database health
2. Specific recommendations for improvement

GOAL VECTOR: {self.goal.name}
CONSTRAINTS: {', '.join(self.goal.constraints)}

TEST RESULTS SUMMARY:
{results_summary}

FAILURE DETAILS:
{failures_text}

Provide your analysis in this exact JSON format:
{{
    "analysis": "Your semantic analysis of database health (2-3 sentences)",
    "recommendations": ["recommendation 1", "recommendation 2", "recommendation 3"]
}}"""

        try:
            response = await self.llm.generate(prompt)

            # Parse response
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown
                if '```json' in response:
                    json_start = response.find('```json') + 7
                    json_end = response.find('```', json_start)
                    data = json.loads(response[json_start:json_end])
                elif '```' in response:
                    json_start = response.find('```') + 3
                    json_end = response.find('```', json_start)
                    data = json.loads(response[json_start:json_end])
                else:
                    raise

            return data.get('analysis', 'Analysis unavailable'), data.get('recommendations', [])

        except Exception as e:
            logger.warning(f"Failed to get semantic analysis: {e}")

            # Provide basic analysis without LLM
            total_passed = sum(r.passed for r in results)
            total_tests = sum(r.total_tests for r in results)
            total_failed = sum(r.failed for r in results)

            if total_failed == 0:
                analysis = "All database tests passed. Schema, data integrity, and relationships are functioning correctly."
                recommendations = ["Continue monitoring performance metrics", "Consider adding more edge case tests"]
            else:
                failed_categories = [r.category.value for r in results if r.failed > 0]
                analysis = f"Database tests failed in: {', '.join(failed_categories)}. Review failures and fix underlying issues."
                recommendations = [
                    f"Fix failures in {cat}" for cat in failed_categories
                ] + ["Run tests again after fixes to verify resolution"]

            return analysis, recommendations

    async def verify(
        self,
        test_output: Any,
        context: Optional[Dict[str, Any]] = None,
        expected: Optional[Any] = None
    ) -> VerificationResult:
        """
        Verify database test output using parent class method.

        This allows using DatabaseVerifier as a standard VerifierAgent
        for integration with the broader Antigravity framework.
        """
        return await super().verify(test_output, context, expected)

    def get_quick_status(self) -> Dict[str, Any]:
        """Get quick status of most recent audit."""
        if not self.audit_reports:
            return {
                'status': 'no_audits',
                'message': 'No database audits have been run'
            }

        latest = self.audit_reports[-1]
        return {
            'status': 'passed' if latest.overall_passed else 'failed',
            'confidence': latest.overall_confidence,
            'pass_rate': latest.overall_pass_rate,
            'total_tests': latest.total_tests,
            'timestamp': latest.timestamp.isoformat()
        }


# Convenience function to create a database verifier
def create_database_verifier(
    llm_client: LLMClient,
    project_root: Optional[Path] = None
) -> DatabaseVerifier:
    """
    Create a DatabaseVerifier with default configuration.

    Args:
        llm_client: LLM client for semantic analysis
        project_root: Project root directory

    Returns:
        Configured DatabaseVerifier instance
    """
    return DatabaseVerifier(llm_client, project_root)


# Goal vectors for database testing (for use in goal_vectors.yaml)
DATABASE_GOAL_VECTORS = {
    'schema_integrity': GoalVector(
        name="Schema Integrity",
        description="Database schema matches design specifications",
        constraints=[
            "All expected tables exist",
            "Column types match specifications",
            "Indexes are properly configured",
            "Constraints are enforced"
        ],
        pass_threshold=0.95
    ),
    'data_integrity': GoalVector(
        name="Data Integrity",
        description="Data constraints are properly enforced",
        constraints=[
            "Unique constraints prevent duplicates",
            "NOT NULL constraints enforced",
            "Foreign keys maintain referential integrity",
            "Check constraints validate data"
        ],
        pass_threshold=0.90
    ),
    'performance': GoalVector(
        name="Performance Requirements",
        description="Database operations meet performance thresholds",
        constraints=[
            "Query latency under threshold",
            "Bulk operations efficient",
            "Indexes improve query performance",
            "No N+1 query patterns"
        ],
        pass_threshold=0.80
    ),
    'concurrency': GoalVector(
        name="Concurrency Safety",
        description="Database handles concurrent access correctly",
        constraints=[
            "No race conditions",
            "Deadlocks handled gracefully",
            "Isolation levels appropriate",
            "Transactions maintain consistency"
        ],
        pass_threshold=0.85
    )
}
