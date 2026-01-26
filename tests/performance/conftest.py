"""
Performance test configuration and fixtures.

Performance tests validate response times and throughput.
"""

import pytest
import time
from typing import Callable, Dict
from dataclasses import dataclass


# ============================================================================
# Performance Thresholds
# ============================================================================

@pytest.fixture(scope="session")
def performance_thresholds() -> Dict[str, float]:
    """
    Performance thresholds in milliseconds.

    These are target thresholds for critical operations.
    Tests will fail if operations exceed these times.
    """
    return {
        # API Operations
        "session_start": 1000,      # 1 second
        "card_answer": 500,         # 500ms
        "card_load": 200,           # 200ms
        "health_check": 100,        # 100ms

        # Real-time Operations
        "websocket_latency": 50,    # 50ms
        "event_propagation": 100,   # 100ms

        # Database Operations
        "db_query_simple": 50,      # 50ms
        "db_query_complex": 200,    # 200ms

        # Batch Operations
        "batch_10_cards": 500,      # 500ms
        "batch_100_cards": 2000,    # 2 seconds
    }


# ============================================================================
# Timer Utilities
# ============================================================================

@dataclass
class TimerResult:
    """Result from a timing operation."""
    start: float
    end: float
    duration_ms: float


class Timer:
    """Context manager for timing operations."""

    def __init__(self):
        self.result: TimerResult = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        end = time.perf_counter()
        self.result = TimerResult(
            start=self._start,
            end=end,
            duration_ms=(end - self._start) * 1000
        )

    @property
    def duration_ms(self) -> float:
        return self.result.duration_ms if self.result else 0


@pytest.fixture
def timer():
    """Timer context manager for benchmarking."""
    return Timer


# ============================================================================
# Benchmark Utilities
# ============================================================================

@pytest.fixture
def run_benchmark():
    """
    Fixture that returns a benchmark runner function.

    Usage:
        def test_something(run_benchmark):
            stats = run_benchmark(my_function, iterations=100)
            assert stats["avg"] < 50  # Average under 50ms
    """
    def _run(func: Callable, iterations: int = 100) -> Dict[str, float]:
        """
        Run a function multiple times and collect timing statistics.

        Args:
            func: Function to benchmark (no arguments)
            iterations: Number of iterations

        Returns:
            Dict with min, max, avg, p50, p95, p99 times in ms
        """
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        times.sort()
        return {
            "min": times[0],
            "max": times[-1],
            "avg": sum(times) / len(times),
            "p50": times[len(times) // 2],
            "p95": times[int(len(times) * 0.95)],
            "p99": times[int(len(times) * 0.99)],
            "iterations": iterations
        }

    return _run


@pytest.fixture
def assert_performance():
    """
    Fixture for asserting performance within thresholds.

    Usage:
        def test_something(assert_performance, timer):
            with timer() as t:
                do_something()
            assert_performance(t.duration_ms, 100, "operation_name")
    """
    def _assert(duration_ms: float, threshold_ms: float, operation: str):
        assert duration_ms < threshold_ms, (
            f"{operation} took {duration_ms:.2f}ms, "
            f"threshold is {threshold_ms}ms"
        )
        print(f"  {operation}: {duration_ms:.2f}ms (threshold: {threshold_ms}ms)")

    return _assert
