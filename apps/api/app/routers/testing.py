"""
Antigravity Testing API Router

Provides endpoints for the Testing panel in the sidebar,
displaying test results from the antigravity testing framework.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import asyncio
import subprocess
import json
import os
from pathlib import Path

router = APIRouter()


# ==================== Models ====================

class TestStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RUNNING = "running"
    PENDING = "pending"
    ERROR = "error"


class TestResult(BaseModel):
    id: str
    name: str
    status: TestStatus
    duration: int  # milliseconds
    category: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    short_trace: Optional[str] = None  # Condensed traceback for display
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    timestamp: str


class TestSuite(BaseModel):
    id: str
    name: str
    category: str
    tests: List[TestResult]
    total_tests: int = Field(alias="totalTests")
    passed: int
    failed: int
    skipped: int
    errors: int = 0
    duration: int
    last_run: str = Field(alias="lastRun")

    class Config:
        populate_by_name = True


class TestSummary(BaseModel):
    total_suites: int = Field(alias="totalSuites")
    total_tests: int = Field(alias="totalTests")
    passed: int
    failed: int
    skipped: int
    errors: int = 0
    pass_rate: float = Field(alias="passRate")
    duration: int
    last_run: str = Field(alias="lastRun")
    suites: List[TestSuite]
    running: bool = False

    class Config:
        populate_by_name = True


class RunTestsRequest(BaseModel):
    suite_id: Optional[str] = None
    test_path: Optional[str] = None


class AntigravityTestResult(BaseModel):
    id: str
    name: str
    status: TestStatus
    goal_vector: Dict[str, Any]
    gravity_intensity: str
    drift_detected: bool
    duration: int
    timestamp: str


# ==================== Test Data Store ====================

# In-memory store for test results
_test_results_cache: Dict[str, Any] = {
    "suites": [],
    "last_run": None,
    "running": False,
}


def _get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    # Navigate up from apps/api/app/routers to project root
    return current.parent.parent.parent.parent.parent


def _parse_pytest_json(json_path: Path) -> List[TestSuite]:
    """Parse pytest JSON output into TestSuite objects."""
    if not json_path.exists():
        return []

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

    project_root = _get_project_root()
    suites_dict: Dict[str, Dict] = {}

    # Group tests by file/category
    for test in data.get("tests", []):
        nodeid = test.get("nodeid", "")

        # Parse the nodeid: tests/database/test_basic.py::TestClass::test_method
        parts = nodeid.split("::")
        file_path = parts[0] if parts else ""
        test_class = parts[1] if len(parts) > 1 else ""
        test_name = parts[-1] if parts else nodeid

        # Determine category from path
        if "database" in file_path:
            category = "Database"
        elif "unit" in file_path:
            category = "Unit"
        elif "integration" in file_path:
            category = "Integration"
        elif "e2e" in file_path:
            category = "E2E"
        elif "performance" in file_path:
            category = "Performance"
        elif "api" in file_path.lower():
            category = "API"
        else:
            category = "Other"

        suite_key = f"{category}_{file_path}"

        if suite_key not in suites_dict:
            # Extract file name for suite name
            file_name = Path(file_path).stem.replace("test_", "").replace("_", " ").title()
            suites_dict[suite_key] = {
                "id": suite_key.replace("/", "_").replace(".", "_"),
                "name": file_name,
                "category": category,
                "tests": [],
                "file_path": file_path,
            }

        # Map pytest outcome to our status
        outcome = test.get("outcome", "pending")
        status_map = {
            "passed": TestStatus.PASSED,
            "failed": TestStatus.FAILED,
            "skipped": TestStatus.SKIPPED,
            "error": TestStatus.ERROR,
        }
        status = status_map.get(outcome, TestStatus.PENDING)

        # Extract failure information
        error_message = None
        error_type = None
        stack_trace = None
        short_trace = None

        call_info = test.get("call", {})
        if status in (TestStatus.FAILED, TestStatus.ERROR):
            # Get the longrepr (full traceback)
            longrepr = call_info.get("longrepr", "")
            if longrepr:
                stack_trace = longrepr

                # Extract short trace (last few lines)
                lines = longrepr.strip().split("\n")
                if len(lines) > 5:
                    short_trace = "\n".join(lines[-5:])
                else:
                    short_trace = longrepr

                # Try to extract error type and message from last line
                for line in reversed(lines):
                    if ": " in line and not line.startswith(" "):
                        parts = line.split(": ", 1)
                        if len(parts) == 2:
                            error_type = parts[0].strip()
                            error_message = parts[1].strip()
                            break
                    elif line.startswith("E "):
                        error_message = line[2:].strip()

            # Also check crash info
            crash = call_info.get("crash", {})
            if crash:
                error_message = error_message or crash.get("message", "")

        # Get stdout/stderr
        stdout = ""
        stderr = ""
        for section in test.get("setup", {}).get("sections", []):
            if section[0] == "Captured stdout":
                stdout += section[1]
            elif section[0] == "Captured stderr":
                stderr += section[1]
        for section in call_info.get("sections", []):
            if section[0] == "Captured stdout call":
                stdout += section[1]
            elif section[0] == "Captured stderr call":
                stderr += section[1]

        # Parse line number from nodeid or file
        line_number = None
        if "lineno" in test:
            line_number = test["lineno"]

        # Build friendly test name
        display_name = test_name
        if test_class:
            display_name = f"{test_class}::{test_name}"

        # Duration in milliseconds
        duration_ms = int((call_info.get("duration", 0) or 0) * 1000)

        test_result = TestResult(
            id=nodeid.replace("::", "_").replace("/", "_").replace(".", "_"),
            name=display_name,
            status=status,
            duration=duration_ms,
            category=category,
            file_path=file_path,
            line_number=line_number,
            error_message=error_message,
            error_type=error_type,
            stack_trace=stack_trace,
            short_trace=short_trace,
            stdout=stdout if stdout else None,
            stderr=stderr if stderr else None,
            timestamp=datetime.utcnow().isoformat()
        )

        suites_dict[suite_key]["tests"].append(test_result)

    # Convert to TestSuite objects
    suites = []
    for suite_data in suites_dict.values():
        tests = suite_data["tests"]
        passed = sum(1 for t in tests if t.status == TestStatus.PASSED)
        failed = sum(1 for t in tests if t.status == TestStatus.FAILED)
        skipped = sum(1 for t in tests if t.status == TestStatus.SKIPPED)
        errors = sum(1 for t in tests if t.status == TestStatus.ERROR)
        total_duration = sum(t.duration for t in tests)

        suite = TestSuite(
            id=suite_data["id"],
            name=suite_data["name"],
            category=suite_data["category"],
            tests=tests,
            totalTests=len(tests),
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            duration=total_duration,
            lastRun=datetime.utcnow().isoformat()
        )
        suites.append(suite)

    # Sort suites: failed first, then by category
    suites.sort(key=lambda s: (-(s.failed + s.errors), s.category, s.name))

    return suites


def _discover_tests() -> List[TestSuite]:
    """Discover and categorize test files in the project."""
    project_root = _get_project_root()
    suites = []

    # Test directories to scan
    test_dirs = [
        ("tests/database", "Database"),
        ("tests/unit", "Unit"),
        ("tests/integration", "Integration"),
        ("tests/e2e", "E2E"),
        ("tests/performance", "Performance"),
        ("apps/api/tests", "API"),
        ("apps/testing/agents/tests", "Antigravity"),
    ]

    for test_dir, category in test_dirs:
        dir_path = project_root / test_dir
        if not dir_path.exists():
            continue

        test_files = list(dir_path.glob("**/test_*.py"))
        if not test_files:
            continue

        tests = []
        for test_file in test_files[:20]:  # Limit for performance
            test_name = test_file.stem.replace("test_", "").replace("_", " ").title()
            tests.append(TestResult(
                id=f"{category.lower()}_{test_file.stem}",
                name=test_name,
                status=TestStatus.PENDING,
                duration=0,
                category=category,
                file_path=str(test_file.relative_to(project_root)),
                timestamp=datetime.utcnow().isoformat()
            ))

        suite = TestSuite(
            id=f"suite_{category.lower()}",
            name=f"{category} Tests",
            category=category,
            tests=tests,
            totalTests=len(tests),
            passed=0,
            failed=0,
            skipped=0,
            errors=0,
            duration=0,
            lastRun=datetime.utcnow().isoformat()
        )
        suites.append(suite)

    return suites


async def _run_pytest_async(test_path: Optional[str] = None) -> List[TestSuite]:
    """Run pytest asynchronously and return parsed results."""
    global _test_results_cache

    project_root = _get_project_root()
    json_output = project_root / "reports" / "pytest_results.json"

    # Ensure reports directory exists
    json_output.parent.mkdir(parents=True, exist_ok=True)

    # Build pytest command
    cmd = [
        "python", "-m", "pytest",
        "--json-report",
        f"--json-report-file={json_output}",
        "-v",
        "--tb=long",  # Full tracebacks
    ]

    if test_path:
        cmd.append(str(project_root / test_path))
    else:
        cmd.append(str(project_root / "tests" / "database"))

    _test_results_cache["running"] = True

    try:
        # Run pytest
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(project_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        # Parse results
        suites = _parse_pytest_json(json_output)

        if not suites:
            # Fallback: try to parse stdout for basic info
            stdout_str = stdout.decode() if stdout else ""
            stderr_str = stderr.decode() if stderr else ""

            # Create a single suite with error info
            suites = [TestSuite(
                id="pytest_run",
                name="Test Run",
                category="Tests",
                tests=[TestResult(
                    id="pytest_output",
                    name="Pytest Output",
                    status=TestStatus.ERROR if process.returncode != 0 else TestStatus.PASSED,
                    duration=0,
                    category="Tests",
                    error_message=stderr_str[:500] if stderr_str else None,
                    stdout=stdout_str[:2000] if stdout_str else None,
                    timestamp=datetime.utcnow().isoformat()
                )],
                totalTests=1,
                passed=1 if process.returncode == 0 else 0,
                failed=1 if process.returncode != 0 else 0,
                skipped=0,
                errors=0,
                duration=0,
                lastRun=datetime.utcnow().isoformat()
            )]

        _test_results_cache["suites"] = suites
        _test_results_cache["last_run"] = datetime.utcnow()

        return suites

    except Exception as e:
        # Return error suite
        return [TestSuite(
            id="error",
            name="Test Run Error",
            category="Error",
            tests=[TestResult(
                id="error",
                name="Failed to run tests",
                status=TestStatus.ERROR,
                duration=0,
                category="Error",
                error_message=str(e),
                timestamp=datetime.utcnow().isoformat()
            )],
            totalTests=1,
            passed=0,
            failed=0,
            skipped=0,
            errors=1,
            duration=0,
            lastRun=datetime.utcnow().isoformat()
        )]
    finally:
        _test_results_cache["running"] = False


# ==================== Endpoints ====================

@router.get("/summary", response_model=TestSummary)
async def get_test_summary():
    """
    Get a summary of all test suites and their results.
    Returns cached results if available, otherwise discovers tests.
    """
    # Use cached results if available
    if _test_results_cache.get("suites"):
        suites = _test_results_cache["suites"]
    else:
        suites = _discover_tests()

    total_tests = sum(s.total_tests for s in suites)
    passed = sum(s.passed for s in suites)
    failed = sum(s.failed for s in suites)
    skipped = sum(s.skipped for s in suites)
    errors = sum(s.errors for s in suites)
    duration = sum(s.duration for s in suites)

    pass_rate = round((passed / total_tests * 100) if total_tests > 0 else 0, 1)
    last_run = _test_results_cache.get("last_run")

    return TestSummary(
        totalSuites=len(suites),
        totalTests=total_tests,
        passed=passed,
        failed=failed,
        skipped=skipped,
        errors=errors,
        passRate=pass_rate,
        duration=duration,
        lastRun=last_run.isoformat() if last_run else datetime.utcnow().isoformat(),
        suites=suites,
        running=_test_results_cache.get("running", False)
    )


@router.get("/suites/{suite_id}", response_model=TestSuite)
async def get_test_suite(suite_id: str):
    """
    Get details for a specific test suite.
    """
    suites = _test_results_cache.get("suites") or _discover_tests()

    for suite in suites:
        if suite.id == suite_id:
            return suite

    raise HTTPException(status_code=404, detail=f"Test suite '{suite_id}' not found")


@router.get("/tests")
async def get_tests(category: Optional[str] = None, status: Optional[str] = None):
    """
    Get all tests, optionally filtered by category or status.
    """
    suites = _test_results_cache.get("suites") or _discover_tests()
    all_tests = []

    for suite in suites:
        if category and suite.category.lower() != category.lower():
            continue
        for test in suite.tests:
            if status and test.status.value != status.lower():
                continue
            all_tests.append(test)

    # Sort: failed first, then errors, then others
    priority = {TestStatus.FAILED: 0, TestStatus.ERROR: 1, TestStatus.SKIPPED: 2, TestStatus.PASSED: 3, TestStatus.PENDING: 4}
    all_tests.sort(key=lambda t: priority.get(t.status, 5))

    return {"tests": all_tests, "total": len(all_tests)}


@router.get("/failed")
async def get_failed_tests():
    """
    Get only failed and error tests with full details.
    """
    suites = _test_results_cache.get("suites") or []
    failed_tests = []

    for suite in suites:
        for test in suite.tests:
            if test.status in (TestStatus.FAILED, TestStatus.ERROR):
                failed_tests.append({
                    "suite": suite.name,
                    "category": suite.category,
                    **test.dict()
                })

    return {
        "tests": failed_tests,
        "total": len(failed_tests),
        "message": f"{len(failed_tests)} test(s) failed" if failed_tests else "All tests passed!"
    }


@router.post("/run")
async def run_tests(
    request: RunTestsRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger test execution.

    If suite_id or test_path is provided, only run those tests.
    Otherwise, run all database tests.
    """
    global _test_results_cache

    if _test_results_cache.get("running"):
        return {
            "status": "already_running",
            "message": "Test run already in progress"
        }

    _test_results_cache["running"] = True

    # Run tests in background
    background_tasks.add_task(_run_pytest_async, request.test_path)

    return {
        "status": "started",
        "message": f"Test run initiated at {datetime.utcnow().isoformat()}",
        "suite_id": request.suite_id
    }


@router.get("/antigravity")
async def get_antigravity_results():
    """
    Get results from antigravity testing framework.

    Returns tests with Goal Vector and Gravitational Well information.
    """
    project_root = _get_project_root()
    antigravity_dir = project_root / "apps" / "testing" / "agents" / "tests"

    results = []

    if antigravity_dir.exists():
        for test_file in antigravity_dir.glob("example_*.py"):
            test_name = test_file.stem.replace("example_", "").replace("_test", "")

            results.append(AntigravityTestResult(
                id=f"ag_{test_file.stem}",
                name=test_name.replace("_", " ").title(),
                status=TestStatus.PENDING,
                goal_vector={
                    "primaryObjective": f"Verify {test_name.replace('_', ' ')}",
                    "successCriteria": ["Passes validation", "No drift detected"],
                    "failureConditions": ["Validation error", "Goal drift"]
                },
                gravity_intensity="HIGH",
                drift_detected=False,
                duration=0,
                timestamp=datetime.utcnow().isoformat()
            ))

    return {
        "results": results,
        "total": len(results),
        "framework": "Antigravity Testing Framework",
        "description": "Tests with semantic gravity to prevent goal drift"
    }


@router.get("/categories")
async def get_test_categories():
    """
    Get list of available test categories.
    """
    return {
        "categories": [
            {"id": "database", "name": "Database", "description": "Database integrity and performance tests"},
            {"id": "unit", "name": "Unit", "description": "Unit tests for individual components"},
            {"id": "integration", "name": "Integration", "description": "Integration tests for component interactions"},
            {"id": "e2e", "name": "E2E", "description": "End-to-end workflow tests"},
            {"id": "performance", "name": "Performance", "description": "Performance and load tests"},
            {"id": "api", "name": "API", "description": "API endpoint tests"},
            {"id": "antigravity", "name": "Antigravity", "description": "Antigravity testing framework tests"},
        ]
    }


@router.delete("/cache")
async def clear_cache():
    """Clear the test results cache."""
    global _test_results_cache
    _test_results_cache = {
        "suites": [],
        "last_run": None,
        "running": False,
    }
    return {"status": "cleared", "message": "Test results cache cleared"}
