"""
Root pytest configuration and shared fixtures for all test tiers.

This conftest.py is automatically loaded by pytest for all tests in the tests/ directory.
It provides common configuration and fixtures used across e2e, integration, unit, and
performance tests.
"""

import pytest
import asyncio
from typing import Generator


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    # Test tiers
    config.addinivalue_line(
        "markers", "e2e: end-to-end tests (full user journeys)"
    )
    config.addinivalue_line(
        "markers", "integration: cross-service integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: unit tests for isolated components"
    )
    config.addinivalue_line(
        "markers", "benchmark: performance benchmark tests"
    )

    # Test characteristics
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_db: mark test as requiring database"
    )
    config.addinivalue_line(
        "markers", "requires_services: mark test as requiring services running"
    )


# ============================================================================
# Async Support
# ============================================================================

@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests (session-scoped for efficiency)."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Service Configuration
# ============================================================================

@pytest.fixture(scope="session")
def service_urls():
    """Service URLs for integration/e2e tests."""
    return {
        "api_gateway": "http://localhost:8000",
        "orchestrator": "http://localhost:8005",
        "scheduler": "http://localhost:8001",
        "telemetry": "http://localhost:8002",
        "inference": "http://localhost:8003"
    }


@pytest.fixture(scope="session")
def timeout_config():
    """Timeout configuration for HTTP clients."""
    return {
        "default": 10.0,
        "slow": 30.0,
        "health_check": 5.0
    }


# ============================================================================
# Test Data Cleanup
# ============================================================================

@pytest.fixture
def db_cleanup():
    """
    Cleanup test data after tests.

    Currently a no-op since we create unique test users.
    Can be extended to clean up specific test data.
    """
    yield
    # Add cleanup logic here if needed


# ============================================================================
# Test Reporting
# ============================================================================

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Custom test summary at end of test run."""
    print("\n" + "=" * 80)
    print("NerdLearn Test Summary")
    print("=" * 80)

    passed = len(terminalreporter.stats.get('passed', []))
    failed = len(terminalreporter.stats.get('failed', []))
    skipped = len(terminalreporter.stats.get('skipped', []))
    errors = len(terminalreporter.stats.get('error', []))

    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors:  {errors}")
    print("=" * 80)
