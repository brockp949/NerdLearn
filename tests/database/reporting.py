"""
Database Test Reporting Infrastructure

Provides utilities for generating comprehensive test reports including:
- Test execution summaries
- Performance metrics
- Trend analysis
- HTML and JSON report generation
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict


@dataclass
class TestResult:
    """Individual test result."""
    name: str
    category: str
    status: str  # passed, failed, skipped, error
    duration: float
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class BenchmarkResult:
    """Performance benchmark result."""
    name: str
    metric: str
    value: float
    unit: str
    threshold: Optional[float] = None
    passed: bool = True
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class TestReport:
    """Complete test report."""
    run_id: str
    timestamp: str
    duration: float
    environment: Dict[str, str]
    summary: Dict[str, int]
    tests: List[TestResult]
    benchmarks: List[BenchmarkResult]
    categories: Dict[str, Dict[str, int]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "environment": self.environment,
            "summary": self.summary,
            "tests": [asdict(t) for t in self.tests],
            "benchmarks": [asdict(b) for b in self.benchmarks],
            "categories": self.categories
        }


class DatabaseTestReporter:
    """Reporter for database test results."""

    def __init__(self, output_dir: str = "reports/database"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tests: List[TestResult] = []
        self.benchmarks: List[BenchmarkResult] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.run_id: str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def start_run(self):
        """Mark the start of a test run."""
        self.start_time = datetime.utcnow()
        self.tests = []
        self.benchmarks = []

    def end_run(self):
        """Mark the end of a test run."""
        self.end_time = datetime.utcnow()

    def record_test(
        self,
        name: str,
        category: str,
        status: str,
        duration: float,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Record a test result."""
        self.tests.append(TestResult(
            name=name,
            category=category,
            status=status,
            duration=duration,
            message=message,
            details=details or {}
        ))

    def record_benchmark(
        self,
        name: str,
        metric: str,
        value: float,
        unit: str,
        threshold: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Record a benchmark result."""
        passed = threshold is None or value <= threshold
        self.benchmarks.append(BenchmarkResult(
            name=name,
            metric=metric,
            value=value,
            unit=unit,
            threshold=threshold,
            passed=passed,
            details=details or {}
        ))

    def get_summary(self) -> Dict[str, int]:
        """Get test summary statistics."""
        return {
            "total": len(self.tests),
            "passed": sum(1 for t in self.tests if t.status == "passed"),
            "failed": sum(1 for t in self.tests if t.status == "failed"),
            "skipped": sum(1 for t in self.tests if t.status == "skipped"),
            "errors": sum(1 for t in self.tests if t.status == "error")
        }

    def get_categories(self) -> Dict[str, Dict[str, int]]:
        """Get results by category."""
        categories: Dict[str, Dict[str, int]] = {}
        for test in self.tests:
            if test.category not in categories:
                categories[test.category] = {
                    "total": 0, "passed": 0, "failed": 0, "skipped": 0
                }
            categories[test.category]["total"] += 1
            if test.status in categories[test.category]:
                categories[test.category][test.status] += 1
        return categories

    def get_environment(self) -> Dict[str, str]:
        """Get environment information."""
        return {
            "python_version": os.popen("python --version").read().strip(),
            "database": os.getenv("TEST_DATABASE_URL", "sqlite:///:memory:"),
            "os": os.name,
            "timestamp": datetime.utcnow().isoformat()
        }

    def generate_report(self) -> TestReport:
        """Generate complete test report."""
        duration = 0.0
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()

        return TestReport(
            run_id=self.run_id,
            timestamp=datetime.utcnow().isoformat(),
            duration=duration,
            environment=self.get_environment(),
            summary=self.get_summary(),
            tests=self.tests,
            benchmarks=self.benchmarks,
            categories=self.get_categories()
        )

    def save_json_report(self, filename: Optional[str] = None) -> str:
        """Save report as JSON."""
        report = self.generate_report()
        filename = filename or f"db_test_report_{self.run_id}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        return str(filepath)

    def save_html_report(self, filename: Optional[str] = None) -> str:
        """Generate and save HTML report."""
        report = self.generate_report()
        filename = filename or f"db_test_report_{self.run_id}.html"
        filepath = self.output_dir / filename

        html = self._generate_html(report)
        with open(filepath, "w") as f:
            f.write(html)

        return str(filepath)

    def _generate_html(self, report: TestReport) -> str:
        """Generate HTML report content."""
        summary = report.summary
        pass_rate = (
            (summary["passed"] / summary["total"] * 100)
            if summary["total"] > 0 else 0
        )

        # Generate test rows
        test_rows = ""
        for test in report.tests:
            status_class = {
                "passed": "success",
                "failed": "danger",
                "skipped": "warning",
                "error": "danger"
            }.get(test.status, "secondary")

            test_rows += f"""
            <tr>
                <td>{test.name}</td>
                <td>{test.category}</td>
                <td><span class="badge bg-{status_class}">{test.status}</span></td>
                <td>{test.duration:.4f}s</td>
                <td>{test.message or '-'}</td>
            </tr>
            """

        # Generate benchmark rows
        benchmark_rows = ""
        for bench in report.benchmarks:
            status_class = "success" if bench.passed else "danger"
            threshold_str = f"{bench.threshold}{bench.unit}" if bench.threshold else "-"
            benchmark_rows += f"""
            <tr>
                <td>{bench.name}</td>
                <td>{bench.metric}</td>
                <td>{bench.value:.4f} {bench.unit}</td>
                <td>{threshold_str}</td>
                <td><span class="badge bg-{status_class}">
                    {"PASS" if bench.passed else "FAIL"}
                </span></td>
            </tr>
            """

        # Generate category summary
        category_rows = ""
        for cat, stats in report.categories.items():
            cat_pass_rate = (
                (stats["passed"] / stats["total"] * 100)
                if stats["total"] > 0 else 0
            )
            category_rows += f"""
            <tr>
                <td>{cat}</td>
                <td>{stats["total"]}</td>
                <td>{stats["passed"]}</td>
                <td>{stats["failed"]}</td>
                <td>{stats["skipped"]}</td>
                <td>{cat_pass_rate:.1f}%</td>
            </tr>
            """

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Database Test Report - {report.run_id}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ padding: 20px; }}
        .summary-card {{ margin-bottom: 20px; }}
        .pass-rate {{ font-size: 2rem; font-weight: bold; }}
        .table {{ margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container-fluid">
        <h1>Database Test Report</h1>
        <p class="text-muted">Run ID: {report.run_id} | Generated: {report.timestamp}</p>

        <!-- Summary Cards -->
        <div class="row summary-card">
            <div class="col-md-3">
                <div class="card text-white bg-primary">
                    <div class="card-body">
                        <h5 class="card-title">Total Tests</h5>
                        <p class="pass-rate">{summary["total"]}</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-white bg-success">
                    <div class="card-body">
                        <h5 class="card-title">Passed</h5>
                        <p class="pass-rate">{summary["passed"]}</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-white bg-danger">
                    <div class="card-body">
                        <h5 class="card-title">Failed</h5>
                        <p class="pass-rate">{summary["failed"]}</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-white {"bg-success" if pass_rate >= 80 else "bg-warning"}">
                    <div class="card-body">
                        <h5 class="card-title">Pass Rate</h5>
                        <p class="pass-rate">{pass_rate:.1f}%</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Category Summary -->
        <h2>Results by Category</h2>
        <table class="table table-striped">
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Total</th>
                    <th>Passed</th>
                    <th>Failed</th>
                    <th>Skipped</th>
                    <th>Pass Rate</th>
                </tr>
            </thead>
            <tbody>
                {category_rows}
            </tbody>
        </table>

        <!-- Test Details -->
        <h2>Test Details</h2>
        <table class="table table-hover">
            <thead>
                <tr>
                    <th>Test Name</th>
                    <th>Category</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Message</th>
                </tr>
            </thead>
            <tbody>
                {test_rows}
            </tbody>
        </table>

        <!-- Benchmarks -->
        <h2>Performance Benchmarks</h2>
        <table class="table table-hover">
            <thead>
                <tr>
                    <th>Benchmark</th>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Threshold</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {benchmark_rows if benchmark_rows else "<tr><td colspan='5'>No benchmarks recorded</td></tr>"}
            </tbody>
        </table>

        <!-- Environment -->
        <h2>Environment</h2>
        <table class="table table-sm">
            <tbody>
                {"".join(f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>" for k, v in report.environment.items())}
            </tbody>
        </table>

        <footer class="mt-4 text-muted">
            <p>Duration: {report.duration:.2f}s</p>
        </footer>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
        """
        return html

    def print_summary(self):
        """Print a summary to console."""
        summary = self.get_summary()
        pass_rate = (
            (summary["passed"] / summary["total"] * 100)
            if summary["total"] > 0 else 0
        )

        print("\n" + "=" * 60)
        print("DATABASE TEST SUMMARY")
        print("=" * 60)
        print(f"Total:   {summary['total']}")
        print(f"Passed:  {summary['passed']}")
        print(f"Failed:  {summary['failed']}")
        print(f"Skipped: {summary['skipped']}")
        print(f"Errors:  {summary['errors']}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        print("=" * 60)

        if self.benchmarks:
            print("\nBENCHMARKS:")
            for b in self.benchmarks:
                status = "PASS" if b.passed else "FAIL"
                print(f"  {b.name}: {b.value:.4f}{b.unit} [{status}]")


# Pytest plugin hooks for automatic reporting
class DatabaseTestPlugin:
    """Pytest plugin for database test reporting."""

    def __init__(self):
        self.reporter = DatabaseTestReporter()

    def pytest_sessionstart(self, session):
        """Called before test session starts."""
        self.reporter.start_run()

    def pytest_runtest_logreport(self, report):
        """Called for each test phase (setup, call, teardown)."""
        if report.when == "call":
            # Determine category from test path
            category = "general"
            if "schema" in report.nodeid:
                category = "schema_integrity"
            elif "data" in report.nodeid:
                category = "data_integrity"
            elif "crud" in report.nodeid:
                category = "crud_operations"
            elif "relationship" in report.nodeid:
                category = "relationships"
            elif "performance" in report.nodeid:
                category = "performance"
            elif "concurren" in report.nodeid:
                category = "concurrency"

            # Determine status
            if report.passed:
                status = "passed"
            elif report.failed:
                status = "failed"
            elif report.skipped:
                status = "skipped"
            else:
                status = "error"

            self.reporter.record_test(
                name=report.nodeid.split("::")[-1],
                category=category,
                status=status,
                duration=report.duration,
                message=str(report.longrepr) if report.longrepr else None
            )

    def pytest_sessionfinish(self, session, exitstatus):
        """Called after test session finishes."""
        self.reporter.end_run()
        self.reporter.print_summary()

        # Generate reports
        json_path = self.reporter.save_json_report()
        html_path = self.reporter.save_html_report()

        print(f"\nReports generated:")
        print(f"  JSON: {json_path}")
        print(f"  HTML: {html_path}")


def pytest_configure(config):
    """Register the plugin."""
    config.pluginmanager.register(DatabaseTestPlugin(), "database_reporter")
