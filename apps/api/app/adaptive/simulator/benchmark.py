"""
Benchmarking Framework for Curriculum RL Policies

Provides standardized evaluation of CRL policies against baselines:
1. Day-30 Retention: Long-term memory retention probability
2. Learning Efficiency: Mastery gained per practice session
3. Interleaving Score: Diversity of concept coverage

Baselines:
- Random: Uniform random concept selection
- Round-Robin: Sequential cycling through concepts
- Mastery-Threshold: Practice until 0.75 mastery (traditional ITS)
- Spaced-Only: Pure spaced repetition (no interleaving)

Usage:
    benchmark = Benchmark(num_simulated_students=1000)
    results = benchmark.compare_policies(
        policies=[crl_policy, baseline_policy],
        num_sessions=50
    )
    results.plot_comparison()

Based on evaluation methodology from:
- "Optimizing Spaced Interleaving with RL" paper
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import json
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from enum import Enum

from .student_simulator import StudentSimulator, StudentSimulatorConfig, SimulatedInteraction

logger = logging.getLogger(__name__)


class BaselinePolicy(Enum):
    """Baseline policies for comparison"""
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    MASTERY_THRESHOLD = "mastery_threshold"
    SPACED_ONLY = "spaced_only"
    RECENCY_BASED = "recency_based"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking"""

    # Simulation parameters
    num_simulated_students: int = 100
    num_sessions_per_student: int = 50
    items_per_session: int = 10
    session_interval_hours: float = 24.0  # Time between sessions

    # Evaluation parameters
    retention_delay_days: float = 30.0  # For Day-30 retention metric

    # Baseline parameters
    mastery_threshold: float = 0.75  # For threshold baseline

    # Parallelization
    num_workers: int = 4
    use_multiprocessing: bool = False  # Use threading by default

    # Random seed for reproducibility
    seed: Optional[int] = 42

    def to_dict(self) -> Dict:
        return {
            "num_simulated_students": self.num_simulated_students,
            "num_sessions_per_student": self.num_sessions_per_student,
            "items_per_session": self.items_per_session,
            "session_interval_hours": self.session_interval_hours,
            "retention_delay_days": self.retention_delay_days,
            "mastery_threshold": self.mastery_threshold,
            "num_workers": self.num_workers,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "BenchmarkConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PolicyEvaluation:
    """Results from evaluating a single policy"""

    policy_name: str

    # Primary metrics
    day30_retention_mean: float
    day30_retention_std: float
    day30_retention_scores: List[float] = field(default_factory=list)

    # Secondary metrics
    learning_efficiency_mean: float = 0.0
    learning_efficiency_std: float = 0.0

    final_mastery_mean: float = 0.0
    final_mastery_std: float = 0.0

    accuracy_mean: float = 0.0
    accuracy_std: float = 0.0

    interleaving_score_mean: float = 0.0
    interleaving_score_std: float = 0.0

    # Coverage metrics
    concepts_practiced_mean: float = 0.0
    concepts_practiced_std: float = 0.0

    # Time metrics
    total_interactions: int = 0
    evaluation_time_seconds: float = 0.0

    # Per-student detailed results
    student_results: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "policy_name": self.policy_name,
            "day30_retention_mean": self.day30_retention_mean,
            "day30_retention_std": self.day30_retention_std,
            "learning_efficiency_mean": self.learning_efficiency_mean,
            "learning_efficiency_std": self.learning_efficiency_std,
            "final_mastery_mean": self.final_mastery_mean,
            "final_mastery_std": self.final_mastery_std,
            "accuracy_mean": self.accuracy_mean,
            "accuracy_std": self.accuracy_std,
            "interleaving_score_mean": self.interleaving_score_mean,
            "interleaving_score_std": self.interleaving_score_std,
            "concepts_practiced_mean": self.concepts_practiced_mean,
            "concepts_practiced_std": self.concepts_practiced_std,
            "total_interactions": self.total_interactions,
            "evaluation_time_seconds": self.evaluation_time_seconds,
        }


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results across all policies"""

    config: BenchmarkConfig
    policy_results: Dict[str, PolicyEvaluation] = field(default_factory=dict)

    # Ranking
    ranking_by_retention: List[str] = field(default_factory=list)
    ranking_by_efficiency: List[str] = field(default_factory=list)

    # Statistical comparisons
    significance_tests: Dict[str, Dict] = field(default_factory=dict)

    # Metadata
    timestamp: str = ""
    total_time_seconds: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def get_best_policy(self, metric: str = "day30_retention") -> str:
        """Get name of best performing policy"""
        if metric == "day30_retention":
            return self.ranking_by_retention[0] if self.ranking_by_retention else ""
        elif metric == "learning_efficiency":
            return self.ranking_by_efficiency[0] if self.ranking_by_efficiency else ""
        return ""

    def get_improvement_over_baseline(
        self,
        policy_name: str,
        baseline_name: str = "mastery_threshold"
    ) -> Dict[str, float]:
        """Calculate improvement percentages over baseline"""
        if policy_name not in self.policy_results or baseline_name not in self.policy_results:
            return {}

        policy = self.policy_results[policy_name]
        baseline = self.policy_results[baseline_name]

        def pct_improvement(policy_val, baseline_val):
            if baseline_val == 0:
                return 0.0
            return ((policy_val - baseline_val) / baseline_val) * 100

        return {
            "day30_retention": pct_improvement(
                policy.day30_retention_mean, baseline.day30_retention_mean
            ),
            "learning_efficiency": pct_improvement(
                policy.learning_efficiency_mean, baseline.learning_efficiency_mean
            ),
            "final_mastery": pct_improvement(
                policy.final_mastery_mean, baseline.final_mastery_mean
            ),
        }

    def to_dict(self) -> Dict:
        return {
            "config": self.config.to_dict(),
            "policy_results": {k: v.to_dict() for k, v in self.policy_results.items()},
            "ranking_by_retention": self.ranking_by_retention,
            "ranking_by_efficiency": self.ranking_by_efficiency,
            "significance_tests": self.significance_tests,
            "timestamp": self.timestamp,
            "total_time_seconds": self.total_time_seconds,
        }

    def save(self, path: str):
        """Save results to JSON"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BenchmarkResults":
        """Load results from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)

        config = BenchmarkConfig.from_dict(data["config"])
        results = cls(config=config)

        for name, eval_dict in data["policy_results"].items():
            results.policy_results[name] = PolicyEvaluation(
                policy_name=eval_dict["policy_name"],
                day30_retention_mean=eval_dict["day30_retention_mean"],
                day30_retention_std=eval_dict["day30_retention_std"],
                learning_efficiency_mean=eval_dict.get("learning_efficiency_mean", 0),
                learning_efficiency_std=eval_dict.get("learning_efficiency_std", 0),
                final_mastery_mean=eval_dict.get("final_mastery_mean", 0),
                final_mastery_std=eval_dict.get("final_mastery_std", 0),
                accuracy_mean=eval_dict.get("accuracy_mean", 0),
                accuracy_std=eval_dict.get("accuracy_std", 0),
                interleaving_score_mean=eval_dict.get("interleaving_score_mean", 0),
                interleaving_score_std=eval_dict.get("interleaving_score_std", 0),
            )

        results.ranking_by_retention = data.get("ranking_by_retention", [])
        results.ranking_by_efficiency = data.get("ranking_by_efficiency", [])
        results.significance_tests = data.get("significance_tests", {})
        results.timestamp = data.get("timestamp", "")
        results.total_time_seconds = data.get("total_time_seconds", 0)

        return results

    def summary(self) -> str:
        """Generate human-readable summary"""
        lines = [
            "=" * 60,
            "BENCHMARK RESULTS SUMMARY",
            "=" * 60,
            f"Timestamp: {self.timestamp}",
            f"Students: {self.config.num_simulated_students}",
            f"Sessions: {self.config.num_sessions_per_student}",
            f"Items/Session: {self.config.items_per_session}",
            "",
            "DAY-30 RETENTION (higher is better):",
            "-" * 40,
        ]

        for rank, name in enumerate(self.ranking_by_retention, 1):
            result = self.policy_results[name]
            lines.append(
                f"  {rank}. {name}: {result.day30_retention_mean:.4f} "
                f"(+/- {result.day30_retention_std:.4f})"
            )

        lines.extend([
            "",
            "LEARNING EFFICIENCY (higher is better):",
            "-" * 40,
        ])

        for rank, name in enumerate(self.ranking_by_efficiency, 1):
            result = self.policy_results[name]
            lines.append(
                f"  {rank}. {name}: {result.learning_efficiency_mean:.4f} "
                f"(+/- {result.learning_efficiency_std:.4f})"
            )

        if self.ranking_by_retention and "mastery_threshold" in self.policy_results:
            lines.extend([
                "",
                "IMPROVEMENT OVER MASTERY-THRESHOLD BASELINE:",
                "-" * 40,
            ])
            best = self.ranking_by_retention[0]
            if best != "mastery_threshold":
                improvement = self.get_improvement_over_baseline(best, "mastery_threshold")
                lines.append(f"  Best policy ({best}):")
                lines.append(f"    Day-30 Retention: {improvement.get('day30_retention', 0):+.2f}%")
                lines.append(f"    Learning Efficiency: {improvement.get('learning_efficiency', 0):+.2f}%")

        lines.append("=" * 60)

        return "\n".join(lines)


class Benchmark:
    """
    Benchmarking framework for comparing curriculum policies.

    Evaluates policies using simulated students with realistic
    learning dynamics (forgetting, slip/guess, spacing effects).
    """

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        simulator_config: Optional[StudentSimulatorConfig] = None,
    ):
        """
        Initialize benchmark.

        Args:
            config: Benchmark configuration
            simulator_config: Configuration for student simulators
        """
        self.config = config or BenchmarkConfig()
        self.simulator_config = simulator_config or StudentSimulatorConfig()

        if self.config.seed is not None:
            np.random.seed(self.config.seed)

    def _create_baseline_policy(
        self,
        baseline_type: BaselinePolicy,
        concept_ids: List[str]
    ) -> Callable:
        """
        Create a baseline policy function.

        Args:
            baseline_type: Type of baseline
            concept_ids: List of available concepts

        Returns:
            Policy function: (simulator_state, belief_state) -> concept_id
        """
        num_concepts = len(concept_ids)

        if baseline_type == BaselinePolicy.RANDOM:
            def random_policy(simulator, belief_state=None):
                return np.random.choice(concept_ids)
            return random_policy

        elif baseline_type == BaselinePolicy.ROUND_ROBIN:
            index = [0]  # Mutable to track state
            def round_robin_policy(simulator, belief_state=None):
                concept = concept_ids[index[0] % num_concepts]
                index[0] += 1
                return concept
            return round_robin_policy

        elif baseline_type == BaselinePolicy.MASTERY_THRESHOLD:
            def mastery_threshold_policy(simulator, belief_state=None):
                # Find first concept below threshold
                for cid in concept_ids:
                    if simulator.true_mastery.get(cid, 0) < self.config.mastery_threshold:
                        return cid
                # All mastered, pick randomly
                return np.random.choice(concept_ids)
            return mastery_threshold_policy

        elif baseline_type == BaselinePolicy.SPACED_ONLY:
            def spaced_only_policy(simulator, belief_state=None):
                # Pick concept with oldest last practice
                oldest_time = datetime.max
                oldest_concept = concept_ids[0]

                for cid in concept_ids:
                    if cid not in simulator.last_practice:
                        return cid  # Never practiced, pick this
                    if simulator.last_practice[cid] < oldest_time:
                        oldest_time = simulator.last_practice[cid]
                        oldest_concept = cid

                return oldest_concept
            return spaced_only_policy

        elif baseline_type == BaselinePolicy.RECENCY_BASED:
            def recency_based_policy(simulator, belief_state=None):
                # Weighted by time since last practice (for interleaving)
                weights = []
                for cid in concept_ids:
                    if cid not in simulator.last_practice:
                        weights.append(1000)  # High weight for unpracticed
                    else:
                        hours = (simulator.current_time - simulator.last_practice[cid]).total_seconds() / 3600
                        weights.append(hours)

                weights = np.array(weights)
                probs = weights / weights.sum()
                return np.random.choice(concept_ids, p=probs)
            return recency_based_policy

        else:
            raise ValueError(f"Unknown baseline: {baseline_type}")

    def _compute_interleaving_score(self, interactions: List[SimulatedInteraction]) -> float:
        """
        Compute interleaving score from interaction history.

        Score measures how much the policy switches between concepts.
        Higher score = more interleaving.

        Returns:
            Interleaving score in [0, 1]
        """
        if len(interactions) < 2:
            return 0.0

        switches = 0
        for i in range(1, len(interactions)):
            if interactions[i].concept_id != interactions[i-1].concept_id:
                switches += 1

        return switches / (len(interactions) - 1)

    def _compute_learning_efficiency(
        self,
        interactions: List[SimulatedInteraction]
    ) -> float:
        """
        Compute learning efficiency from interactions.

        Efficiency = total mastery gain / number of interactions
        """
        if not interactions:
            return 0.0

        total_gain = sum(
            max(0, i.true_mastery_after - i.true_mastery_before)
            for i in interactions
        )

        return total_gain / len(interactions)

    def _evaluate_single_student(
        self,
        policy_fn: Callable,
        student_seed: int,
        concept_ids: List[str],
    ) -> Dict:
        """
        Evaluate policy on a single simulated student.

        Args:
            policy_fn: Policy function (simulator, belief_state) -> concept_id
            student_seed: Random seed for this student
            concept_ids: List of concept IDs

        Returns:
            Dictionary with evaluation metrics
        """
        # Create simulator with unique seed
        config = StudentSimulatorConfig(
            num_concepts=len(concept_ids),
            seed=student_seed,
            **{k: v for k, v in self.simulator_config.to_dict().items()
               if k not in ['num_concepts', 'seed']}
        )
        simulator = StudentSimulator(config)
        simulator.reset(concept_ids=concept_ids)

        # Run sessions
        for session_idx in range(self.config.num_sessions_per_student):
            # Items within session
            for item_idx in range(self.config.items_per_session):
                # Get policy's concept selection
                concept_id = policy_fn(simulator, None)

                # Simulate response
                correct, info = simulator.respond(concept_id)

            # Advance time between sessions
            simulator.advance_time(hours=self.config.session_interval_hours)

        # Compute metrics
        stats = simulator.get_statistics()

        return {
            "day30_retention": simulator.compute_day30_retention(),
            "final_mastery": stats["mean_mastery"],
            "accuracy": stats["accuracy"],
            "interleaving_score": self._compute_interleaving_score(simulator.interaction_history),
            "learning_efficiency": self._compute_learning_efficiency(simulator.interaction_history),
            "concepts_practiced": stats["concepts_practiced"],
            "total_interactions": stats["num_interactions"],
        }

    def evaluate_policy(
        self,
        policy_fn: Callable,
        policy_name: str,
        concept_ids: Optional[List[str]] = None,
    ) -> PolicyEvaluation:
        """
        Evaluate a single policy across multiple simulated students.

        Args:
            policy_fn: Policy function (simulator, belief_state) -> concept_id
            policy_name: Name for this policy
            concept_ids: List of concept IDs (auto-generates if None)

        Returns:
            PolicyEvaluation with aggregated metrics
        """
        import time
        start_time = time.time()

        # Generate concept IDs if not provided
        if concept_ids is None:
            concept_ids = [f"concept_{i}" for i in range(self.simulator_config.num_concepts)]

        logger.info(f"Evaluating policy: {policy_name} on {self.config.num_simulated_students} students")

        # Generate student seeds
        base_seed = self.config.seed or 0
        student_seeds = [base_seed + i for i in range(self.config.num_simulated_students)]

        # Evaluate each student
        student_results = []

        if self.config.num_workers > 1:
            # Parallel evaluation
            executor_class = ProcessPoolExecutor if self.config.use_multiprocessing else ThreadPoolExecutor

            # Note: For multiprocessing, policy_fn must be picklable
            # Using threading for safer default behavior
            with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                futures = [
                    executor.submit(
                        self._evaluate_single_student,
                        policy_fn,
                        seed,
                        concept_ids
                    )
                    for seed in student_seeds
                ]

                for future in futures:
                    student_results.append(future.result())
        else:
            # Sequential evaluation
            for seed in student_seeds:
                result = self._evaluate_single_student(policy_fn, seed, concept_ids)
                student_results.append(result)

        # Aggregate metrics
        retention_scores = [r["day30_retention"] for r in student_results]
        efficiency_scores = [r["learning_efficiency"] for r in student_results]
        mastery_scores = [r["final_mastery"] for r in student_results]
        accuracy_scores = [r["accuracy"] for r in student_results]
        interleaving_scores = [r["interleaving_score"] for r in student_results]
        coverage_scores = [r["concepts_practiced"] for r in student_results]

        evaluation = PolicyEvaluation(
            policy_name=policy_name,
            day30_retention_mean=float(np.mean(retention_scores)),
            day30_retention_std=float(np.std(retention_scores)),
            day30_retention_scores=retention_scores,
            learning_efficiency_mean=float(np.mean(efficiency_scores)),
            learning_efficiency_std=float(np.std(efficiency_scores)),
            final_mastery_mean=float(np.mean(mastery_scores)),
            final_mastery_std=float(np.std(mastery_scores)),
            accuracy_mean=float(np.mean(accuracy_scores)),
            accuracy_std=float(np.std(accuracy_scores)),
            interleaving_score_mean=float(np.mean(interleaving_scores)),
            interleaving_score_std=float(np.std(interleaving_scores)),
            concepts_practiced_mean=float(np.mean(coverage_scores)),
            concepts_practiced_std=float(np.std(coverage_scores)),
            total_interactions=sum(r["total_interactions"] for r in student_results),
            evaluation_time_seconds=time.time() - start_time,
            student_results=student_results,
        )

        logger.info(
            f"Policy {policy_name}: Day-30 Retention = {evaluation.day30_retention_mean:.4f} "
            f"(+/- {evaluation.day30_retention_std:.4f})"
        )

        return evaluation

    def _run_significance_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
        alpha: float = 0.05
    ) -> Dict:
        """
        Run statistical significance test (Welch's t-test).

        Args:
            scores_a: Scores from policy A
            scores_b: Scores from policy B
            alpha: Significance level

        Returns:
            Test results dictionary
        """
        from scipy import stats

        t_stat, p_value = stats.ttest_ind(scores_a, scores_b, equal_var=False)

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < alpha,
            "effect_size": float((np.mean(scores_a) - np.mean(scores_b)) /
                                 np.sqrt((np.var(scores_a) + np.var(scores_b)) / 2)),
        }

    def compare_policies(
        self,
        policies: List[Tuple[str, Callable]],
        include_baselines: bool = True,
        concept_ids: Optional[List[str]] = None,
    ) -> BenchmarkResults:
        """
        Compare multiple policies against each other and baselines.

        Args:
            policies: List of (name, policy_fn) tuples
            include_baselines: Whether to include baseline policies
            concept_ids: List of concept IDs

        Returns:
            BenchmarkResults with full comparison
        """
        import time
        start_time = time.time()

        # Generate concept IDs if not provided
        if concept_ids is None:
            concept_ids = [f"concept_{i}" for i in range(self.simulator_config.num_concepts)]

        results = BenchmarkResults(config=self.config)

        # Add baseline policies
        all_policies = list(policies)

        if include_baselines:
            for baseline_type in BaselinePolicy:
                baseline_fn = self._create_baseline_policy(baseline_type, concept_ids)
                all_policies.append((baseline_type.value, baseline_fn))

        # Evaluate each policy
        for name, policy_fn in all_policies:
            evaluation = self.evaluate_policy(policy_fn, name, concept_ids)
            results.policy_results[name] = evaluation

        # Compute rankings
        results.ranking_by_retention = sorted(
            results.policy_results.keys(),
            key=lambda n: results.policy_results[n].day30_retention_mean,
            reverse=True
        )

        results.ranking_by_efficiency = sorted(
            results.policy_results.keys(),
            key=lambda n: results.policy_results[n].learning_efficiency_mean,
            reverse=True
        )

        # Statistical significance tests (optional)
        try:
            for i, name_a in enumerate(results.ranking_by_retention):
                for name_b in results.ranking_by_retention[i+1:]:
                    scores_a = results.policy_results[name_a].day30_retention_scores
                    scores_b = results.policy_results[name_b].day30_retention_scores

                    test_key = f"{name_a}_vs_{name_b}"
                    results.significance_tests[test_key] = self._run_significance_test(
                        scores_a, scores_b
                    )
        except ImportError:
            logger.warning("scipy not available, skipping significance tests")

        results.total_time_seconds = time.time() - start_time

        logger.info(f"Benchmark complete in {results.total_time_seconds:.2f}s")
        logger.info(f"Best by Day-30 Retention: {results.ranking_by_retention[0]}")

        return results

    def quick_benchmark(
        self,
        policy_fn: Callable,
        policy_name: str = "test_policy",
        num_students: int = 100,
    ) -> Dict:
        """
        Quick benchmark for rapid iteration.

        Args:
            policy_fn: Policy to evaluate
            policy_name: Name for the policy
            num_students: Number of students (reduced for speed)

        Returns:
            Dictionary with key metrics
        """
        # Use reduced config
        original_students = self.config.num_simulated_students
        self.config.num_simulated_students = num_students

        try:
            results = self.compare_policies(
                [(policy_name, policy_fn)],
                include_baselines=True
            )

            policy_result = results.policy_results[policy_name]
            baseline_result = results.policy_results.get(
                BaselinePolicy.MASTERY_THRESHOLD.value
            )

            improvement = 0.0
            if baseline_result:
                improvement = (
                    (policy_result.day30_retention_mean - baseline_result.day30_retention_mean) /
                    baseline_result.day30_retention_mean * 100
                )

            return {
                "day30_retention": policy_result.day30_retention_mean,
                "day30_retention_std": policy_result.day30_retention_std,
                "improvement_over_baseline_pct": improvement,
                "rank": results.ranking_by_retention.index(policy_name) + 1,
                "total_policies": len(results.ranking_by_retention),
                "learning_efficiency": policy_result.learning_efficiency_mean,
                "interleaving_score": policy_result.interleaving_score_mean,
            }
        finally:
            self.config.num_simulated_students = original_students


def run_full_benchmark(
    crl_policy_fn: Optional[Callable] = None,
    output_path: str = "benchmark_results.json",
    num_students: int = 1000,
    num_sessions: int = 50,
) -> BenchmarkResults:
    """
    Convenience function to run a full benchmark.

    Args:
        crl_policy_fn: Optional CRL policy to evaluate
        output_path: Path to save results
        num_students: Number of simulated students
        num_sessions: Number of sessions per student

    Returns:
        BenchmarkResults
    """
    config = BenchmarkConfig(
        num_simulated_students=num_students,
        num_sessions_per_student=num_sessions,
    )

    benchmark = Benchmark(config)

    policies = []
    if crl_policy_fn is not None:
        policies.append(("crl_policy", crl_policy_fn))

    results = benchmark.compare_policies(policies)
    results.save(output_path)

    print(results.summary())

    return results
