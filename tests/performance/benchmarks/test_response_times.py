"""
Response time benchmark tests.

These tests measure and validate response times for critical operations.
Run with: pytest tests/performance/benchmarks/ -v --benchmark
"""

import pytest
import time
from typing import Callable


# ============================================================================
# Benchmark Fixtures
# ============================================================================

@pytest.fixture
def benchmark_threshold():
    """Default performance thresholds in milliseconds."""
    return {
        "session_start": 1000,      # 1 second
        "card_answer": 500,         # 500ms
        "card_load": 200,           # 200ms
        "health_check": 100,        # 100ms
        "websocket_latency": 50     # 50ms
    }


@pytest.fixture
def timer():
    """Simple timer context manager for benchmarking."""
    class Timer:
        def __init__(self):
            self.start = None
            self.end = None
            self.duration_ms = 0

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end = time.perf_counter()
            self.duration_ms = (self.end - self.start) * 1000

    return Timer


# ============================================================================
# Benchmark Tests (Placeholders)
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.skip(reason="Requires services running")
async def test_session_start_benchmark(timer, benchmark_threshold):
    """Benchmark session start time."""
    # TODO: Implement when services are running
    pass


@pytest.mark.benchmark
@pytest.mark.skip(reason="Requires services running")
async def test_card_answer_benchmark(timer, benchmark_threshold):
    """Benchmark card answer processing time."""
    # TODO: Implement when services are running
    pass


@pytest.mark.benchmark
def test_local_computation_benchmark(timer):
    """Benchmark local computation (example)."""
    with timer() as t:
        # Simulate some computation
        result = sum(range(10000))

    assert t.duration_ms < 100, f"Computation took {t.duration_ms:.2f}ms"
    print(f"Local computation: {t.duration_ms:.2f}ms")


# ============================================================================
# Benchmark Utilities
# ============================================================================

def run_benchmark(func: Callable, iterations: int = 100) -> dict:
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
