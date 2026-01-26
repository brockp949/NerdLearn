"""
Student Simulator and Benchmarking Framework

Provides tools for:
1. Simulating student learning with realistic cognitive dynamics
2. Benchmarking CRL policies against baselines
3. Generating synthetic training data

Key Components:
- StudentSimulator: Simulates learning, forgetting, and response generation
- Benchmark: Compares policies using standardized metrics
- Metrics: Day-30 retention, learning efficiency, interleaving score

Usage:
    simulator = StudentSimulator(config)
    benchmark = Benchmark(simulators=[simulator for _ in range(1000)])
    results = benchmark.compare_policies([crl_policy, baseline_policy])
"""

from .student_simulator import (
    StudentSimulatorConfig,
    StudentSimulator,
    SimulatedInteraction,
)

from .benchmark import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkResults,
)

__all__ = [
    "StudentSimulatorConfig",
    "StudentSimulator",
    "SimulatedInteraction",
    "Benchmark",
    "BenchmarkConfig",
    "BenchmarkResults",
]
