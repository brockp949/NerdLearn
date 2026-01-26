"""
Joint Causal Inference (JCI) Confidence Updater.

Performs Bayesian updates on causal edge confidence by combining
observational evidence with experimental (A/B testing) results.

The JCI framework allows us to:
1. Start with observational confidence P(A→B) from causal discovery
2. Design A/B experiments to test the causal hypothesis
3. Update P(A→B | experiment) using Bayesian inference
4. Sync updated confidence back to the knowledge graph

References:
- Mooij et al. "Joint Causal Inference" (JMLR 2020)
- Pearl "Causality" Chapter 7: The Interventional Interpretation
"""

import logging
import math
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.models.jci import (
    ExperimentEdge,
    ExperimentEdgeStatus,
    CausalDirection,
    CausalEdgeHistory,
    EdgeValidationQueue,
    MetaAnalysisResult,
)
from app.services.ab_testing import experiment_manager
from app.adaptive.causal_discovery.active_learning import ActiveLearningModule

logger = logging.getLogger(__name__)


@dataclass
class BayesianUpdate:
    """Result of a Bayesian confidence update."""
    prior: float
    posterior: float
    likelihood_ratio: float
    evidence_strength: float
    update_reason: str


class JCIConfidenceUpdater:
    """
    Updates causal edge confidence using Joint Causal Inference.

    Combines observational data (from causal discovery algorithms)
    with experimental data (from A/B tests) using Bayesian updating.
    """

    def __init__(self, db: AsyncSession, graph_service=None):
        """
        Initialize JCI updater.

        Args:
            db: Database session for persisting updates
            graph_service: Optional AsyncGraphService for graph updates
        """
        self.db = db
        self.graph_service = graph_service
        self.active_learning = ActiveLearningModule()

        # Bayesian update parameters
        self.significance_threshold = 0.05  # p-value threshold
        self.min_effect_size = 0.2  # Cohen's d minimum for meaningful effect
        self.prior_weight = 0.3  # Weight given to observational prior

    async def create_experiment_edge_link(
        self,
        experiment_id: str,
        source_concept: str,
        target_concept: str,
        prior_confidence: float,
        course_id: Optional[int] = None,
        observational_data: Optional[Dict[str, Any]] = None,
        treatment_description: Optional[str] = None,
        control_description: Optional[str] = None,
    ) -> ExperimentEdge:
        """
        Create a link between an A/B experiment and a causal edge.

        This is called when designing an experiment to validate
        a hypothesized causal relationship.

        Args:
            experiment_id: ID of the A/B experiment
            source_concept: Source concept in the causal relationship
            target_concept: Target concept (effect)
            prior_confidence: P(A→B) from observational data
            course_id: Optional course context
            observational_data: Optional dict with correlation, sample_size, p_value
            treatment_description: What intervention is being applied
            control_description: What the control condition is

        Returns:
            Created ExperimentEdge instance
        """
        # Create experiment edge record
        experiment_edge = ExperimentEdge(
            experiment_id=experiment_id,
            source_concept=source_concept,
            target_concept=target_concept,
            course_id=course_id,
            prior_confidence=prior_confidence,
            status=ExperimentEdgeStatus.PENDING,
            treatment_description=treatment_description,
            control_description=control_description,
        )

        # Add observational evidence if provided
        if observational_data:
            experiment_edge.observational_correlation = observational_data.get('correlation')
            experiment_edge.observational_sample_size = observational_data.get('sample_size')
            experiment_edge.observational_p_value = observational_data.get('p_value')

        self.db.add(experiment_edge)
        await self.db.flush()

        logger.info(
            f"Created experiment-edge link: {experiment_id} -> {source_concept}→{target_concept} "
            f"(prior={prior_confidence:.3f})"
        )

        return experiment_edge

    async def start_experiment(self, experiment_edge_id: int) -> None:
        """Mark an experiment as running."""
        await self.db.execute(
            update(ExperimentEdge)
            .where(ExperimentEdge.id == experiment_edge_id)
            .values(
                status=ExperimentEdgeStatus.RUNNING,
                experiment_started_at=datetime.utcnow()
            )
        )
        await self.db.commit()

    async def update_edge_confidence(
        self,
        experiment_edge_id: int,
        experiment_results: Dict[str, Any],
    ) -> BayesianUpdate:
        """
        Update edge confidence based on experiment results.

        Uses Bayesian inference to combine prior (observational) confidence
        with experimental evidence (A/B test results).

        Args:
            experiment_edge_id: ID of the ExperimentEdge record
            experiment_results: Dict with effect_size, p_value, sample_size, etc.

        Returns:
            BayesianUpdate with prior, posterior, and details
        """
        # Load experiment edge
        result = await self.db.execute(
            select(ExperimentEdge).where(ExperimentEdge.id == experiment_edge_id)
        )
        experiment_edge = result.scalar_one_or_none()

        if not experiment_edge:
            raise ValueError(f"ExperimentEdge {experiment_edge_id} not found")

        prior = experiment_edge.prior_confidence

        # Extract experiment results
        effect_size = experiment_results.get('effect_size', 0.0)
        p_value = experiment_results.get('p_value', 1.0)
        sample_size = experiment_results.get('sample_size', 0)
        ci_lower = experiment_results.get('ci_lower')
        ci_upper = experiment_results.get('ci_upper')
        lift = experiment_results.get('lift', 0.0)

        # Compute likelihood ratio: P(data | causal) / P(data | no causal)
        # Using effect size and significance as evidence
        likelihood_ratio = self._compute_likelihood_ratio(
            effect_size=effect_size,
            p_value=p_value,
            sample_size=sample_size,
        )

        # Bayesian update: P(causal | data) ∝ P(data | causal) * P(causal)
        # Using logit transform for numerical stability
        posterior = self._bayesian_update(prior, likelihood_ratio)

        # Determine evidence strength and update reason
        evidence_strength = self._compute_evidence_strength(
            effect_size=effect_size,
            p_value=p_value,
            sample_size=sample_size,
        )

        if p_value < self.significance_threshold and effect_size > self.min_effect_size:
            update_reason = "significant_positive_effect"
            # Strong evidence supporting causation
        elif p_value < self.significance_threshold and effect_size < -self.min_effect_size:
            update_reason = "significant_negative_effect"
            # Evidence against this direction (might be reverse causation)
        elif p_value >= self.significance_threshold:
            update_reason = "inconclusive_experiment"
            # Experiment didn't reach significance
        else:
            update_reason = "small_effect_size"
            # Significant but small effect

        # Update experiment edge record
        experiment_edge.posterior_confidence = posterior
        experiment_edge.experiment_effect_size = effect_size
        experiment_edge.experiment_p_value = p_value
        experiment_edge.experiment_sample_size = sample_size
        experiment_edge.experiment_power = experiment_results.get('power')
        experiment_edge.effect_ci_lower = ci_lower
        experiment_edge.effect_ci_upper = ci_upper
        experiment_edge.status = ExperimentEdgeStatus.COMPLETED
        experiment_edge.experiment_completed_at = datetime.utcnow()

        # Determine causal direction from effect
        experiment_edge.inferred_direction = self._infer_direction(
            effect_size=effect_size,
            p_value=p_value,
        )
        experiment_edge.direction_confidence = abs(effect_size) if p_value < self.significance_threshold else 0.0

        # Record history
        history = CausalEdgeHistory(
            experiment_edge_id=experiment_edge.id,
            confidence_before=prior,
            confidence_after=posterior,
            confidence_delta=posterior - prior,
            update_reason=update_reason,
            update_source=experiment_edge.experiment_id,
            evidence_type="experimental",
            evidence_strength=evidence_strength,
            likelihood_ratio=likelihood_ratio,
            prior=prior,
            posterior=posterior,
        )
        self.db.add(history)

        await self.db.commit()

        logger.info(
            f"Updated edge confidence for {experiment_edge.source_concept}→{experiment_edge.target_concept}: "
            f"{prior:.3f} → {posterior:.3f} (reason: {update_reason})"
        )

        return BayesianUpdate(
            prior=prior,
            posterior=posterior,
            likelihood_ratio=likelihood_ratio,
            evidence_strength=evidence_strength,
            update_reason=update_reason,
        )

    def _compute_likelihood_ratio(
        self,
        effect_size: float,
        p_value: float,
        sample_size: int,
    ) -> float:
        """
        Compute likelihood ratio P(data | H1) / P(data | H0).

        H1: Causal relationship exists
        H0: No causal relationship (correlation is spurious)

        Uses a simplified model based on effect size and significance.
        """
        if sample_size == 0:
            return 1.0  # No evidence

        # Transform p-value to Bayes Factor approximation
        # Using the BIC approximation: BF ≈ exp((BIC_H0 - BIC_H1) / 2)
        # Simplified: use -2 * log(p_value) as evidence strength

        if p_value <= 0 or p_value >= 1:
            log_bf = 0.0
        else:
            # Sellke et al. (2001) bound: BF ≤ -1 / (e * p * ln(p))
            # Simplified approximation
            log_bf = -math.log(p_value) * abs(effect_size)

        # Incorporate effect size magnitude
        # Larger effects are stronger evidence
        effect_factor = 1.0 + abs(effect_size)

        # Incorporate sample size (larger N = more reliable)
        n_factor = math.log1p(sample_size) / 5.0  # Normalize

        # Combine factors
        log_likelihood_ratio = log_bf * effect_factor * n_factor

        # Bound the likelihood ratio to prevent extreme updates
        log_likelihood_ratio = max(-5.0, min(5.0, log_likelihood_ratio))

        return math.exp(log_likelihood_ratio)

    def _bayesian_update(self, prior: float, likelihood_ratio: float) -> float:
        """
        Perform Bayesian update on confidence.

        P(H1 | data) = P(data | H1) * P(H1) / P(data)

        Using odds form:
        posterior_odds = likelihood_ratio * prior_odds
        """
        # Bound prior away from 0 and 1 for numerical stability
        prior = max(0.01, min(0.99, prior))

        # Convert to odds
        prior_odds = prior / (1 - prior)

        # Update odds
        posterior_odds = likelihood_ratio * prior_odds

        # Convert back to probability
        posterior = posterior_odds / (1 + posterior_odds)

        # Bound result
        posterior = max(0.01, min(0.99, posterior))

        return posterior

    def _compute_evidence_strength(
        self,
        effect_size: float,
        p_value: float,
        sample_size: int,
    ) -> float:
        """
        Compute overall evidence strength on 0-1 scale.

        Combines statistical significance, effect magnitude, and sample size.
        """
        # Significance component (higher when p < 0.05)
        if p_value <= 0:
            sig_component = 1.0
        elif p_value >= 1:
            sig_component = 0.0
        else:
            sig_component = max(0, 1 - p_value / self.significance_threshold)

        # Effect size component (Cohen's d interpretation)
        # Small: 0.2, Medium: 0.5, Large: 0.8
        effect_component = min(1.0, abs(effect_size) / 0.8)

        # Sample size component (diminishing returns)
        n_component = min(1.0, math.log1p(sample_size) / 7.0)  # ~1.0 at N=1000

        # Weighted combination
        strength = 0.4 * sig_component + 0.4 * effect_component + 0.2 * n_component

        return strength

    def _infer_direction(
        self,
        effect_size: float,
        p_value: float,
    ) -> CausalDirection:
        """Infer causal direction from experiment results."""
        if p_value >= self.significance_threshold:
            return CausalDirection.NONE

        if effect_size > self.min_effect_size:
            return CausalDirection.FORWARD
        elif effect_size < -self.min_effect_size:
            return CausalDirection.REVERSE
        else:
            return CausalDirection.NONE

    async def sync_with_graph(
        self,
        experiment_edge_id: int,
        confidence_threshold: float = 0.5,
    ) -> bool:
        """
        Sync updated confidence to the knowledge graph.

        Updates the CAUSAL_PREREQUISITE edge in Apache AGE with
        the new posterior confidence from experimental validation.

        Args:
            experiment_edge_id: ID of the ExperimentEdge record
            confidence_threshold: Minimum confidence to mark as 'verified'

        Returns:
            True if graph was updated successfully
        """
        if not self.graph_service:
            logger.warning("No graph service configured, skipping sync")
            return False

        # Load experiment edge
        result = await self.db.execute(
            select(ExperimentEdge).where(ExperimentEdge.id == experiment_edge_id)
        )
        experiment_edge = result.scalar_one_or_none()

        if not experiment_edge or experiment_edge.posterior_confidence is None:
            return False

        # Prepare edge update for graph
        confidence = experiment_edge.posterior_confidence
        status = 'verified' if confidence > 0.85 else 'hypothetical' if confidence >= confidence_threshold else 'weak'

        edge_data = {
            'source': experiment_edge.source_concept,
            'target': experiment_edge.target_concept,
            'weight': confidence,
            'confidence': confidence,
            'source_algo': 'jci_experimental',
            'type': 'directed' if experiment_edge.inferred_direction == CausalDirection.FORWARD else 'undirected',
        }

        # Update graph
        try:
            result = await self.graph_service.update_causal_edges(
                edges=[edge_data],
                course_id=experiment_edge.course_id,
            )

            # Mark as validated
            experiment_edge.status = ExperimentEdgeStatus.VALIDATED
            await self.db.commit()

            logger.info(
                f"Synced edge to graph: {experiment_edge.source_concept}→{experiment_edge.target_concept} "
                f"(conf={confidence:.3f}, status={status})"
            )

            return result.get('persisted', 0) > 0

        except Exception as e:
            logger.error(f"Failed to sync edge to graph: {e}")
            return False

    async def process_experiment_completion(
        self,
        experiment_id: str,
    ) -> List[BayesianUpdate]:
        """
        Process completion of an A/B experiment.

        Finds all ExperimentEdge records linked to this experiment
        and updates their confidence based on the experiment results.

        Args:
            experiment_id: ID of the completed experiment

        Returns:
            List of BayesianUpdate results for each affected edge
        """
        # Get experiment results from A/B testing framework
        results = experiment_manager.get_results(experiment_id)

        if results.get('error'):
            logger.error(f"Could not get results for experiment {experiment_id}: {results['error']}")
            return []

        # Find linked experiment edges
        query = select(ExperimentEdge).where(
            ExperimentEdge.experiment_id == experiment_id,
            ExperimentEdge.status.in_([ExperimentEdgeStatus.PENDING, ExperimentEdgeStatus.RUNNING])
        )
        result = await self.db.execute(query)
        experiment_edges = result.scalars().all()

        if not experiment_edges:
            logger.warning(f"No pending experiment edges found for {experiment_id}")
            return []

        updates = []

        # Extract effect size and p-value from results
        # Look for treatment variant results
        treatment_variant = None
        for variant in results.get('variants', []):
            if variant.get('type') == 'treatment':
                treatment_variant = variant
                break

        if not treatment_variant:
            logger.warning(f"No treatment variant found in results for {experiment_id}")
            return []

        experiment_results = {
            'effect_size': self._lift_to_cohens_d(
                treatment_variant.get('lift', 0),
                treatment_variant.get('std_dev', 1.0),
            ),
            'p_value': treatment_variant.get('p_value', 1.0),
            'sample_size': treatment_variant.get('sample_size', 0),
            'lift': treatment_variant.get('lift', 0),
            'ci_lower': treatment_variant.get('confidence_interval', (0, 0))[0],
            'ci_upper': treatment_variant.get('confidence_interval', (0, 0))[1],
        }

        # Update each linked edge
        for edge in experiment_edges:
            try:
                update = await self.update_edge_confidence(
                    experiment_edge_id=edge.id,
                    experiment_results=experiment_results,
                )
                updates.append(update)

                # Sync to graph
                await self.sync_with_graph(edge.id)

            except Exception as e:
                logger.error(f"Failed to update edge {edge.id}: {e}")

        return updates

    def _lift_to_cohens_d(self, lift_percent: float, std_dev: float) -> float:
        """Convert percentage lift to Cohen's d effect size."""
        if std_dev <= 0:
            return 0.0
        # Approximate: lift% / 100 / std_dev
        return (lift_percent / 100.0) / std_dev

    async def recommend_experiments(
        self,
        course_id: int,
        max_recommendations: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Recommend causal edges that would benefit from experimental validation.

        Uses the ActiveLearningModule to identify high-entropy edges
        that would provide the most information gain if validated.

        Args:
            course_id: Course to find experiments for
            max_recommendations: Maximum number of recommendations

        Returns:
            List of recommended experiment targets
        """
        if not self.graph_service:
            logger.warning("No graph service configured")
            return []

        # Get current causal edges
        edges = await self.graph_service.get_causal_edges(
            course_id=course_id,
            status='hypothetical'  # Only hypothetical edges need validation
        )

        # Format for active learning module
        candidate_edges = [
            {
                'source': e['source'],
                'target': e['target'],
                'confidence': e.get('confidence', 0.5),
            }
            for e in edges
        ]

        # Get recommendations
        recommendations = self.active_learning.recommend_experiments(
            candidate_edges=candidate_edges,
            max_experiments=max_recommendations,
        )

        # Add to validation queue
        for rec in recommendations:
            queue_entry = EdgeValidationQueue(
                source_concept=rec['source'],
                target_concept=rec['target'],
                course_id=course_id,
                priority_score=rec['priority'],
                information_gain=rec['entropy'],
                uncertainty=rec['entropy'],
                current_confidence=rec['current_confidence'],
                estimated_sample_size=self._estimate_sample_size(rec['current_confidence']),
                feasibility_score=0.7,  # Default feasibility
                status='pending',
            )
            self.db.add(queue_entry)

        await self.db.commit()

        return recommendations

    def _estimate_sample_size(self, current_confidence: float) -> int:
        """
        Estimate required sample size for significant result.

        Uses power analysis approximation for detecting
        departure from current confidence.
        """
        # Effect size we want to detect
        effect = abs(0.5 - current_confidence) + 0.2

        # Simplified power calculation for 80% power, alpha=0.05
        # N ≈ 2 * ((z_alpha + z_beta) / effect)^2
        z_alpha = 1.96  # Two-tailed, alpha=0.05
        z_beta = 0.84   # Power=0.80

        n_per_group = 2 * ((z_alpha + z_beta) / effect) ** 2
        n_per_group = max(50, min(5000, int(n_per_group)))

        return n_per_group * 2  # Total across both groups

    async def run_meta_analysis(
        self,
        source_concept: str,
        target_concept: str,
    ) -> Optional[MetaAnalysisResult]:
        """
        Run meta-analysis combining multiple experiments for the same edge.

        When multiple experiments test the same causal relationship,
        we can combine their evidence through meta-analysis.

        Args:
            source_concept: Source concept
            target_concept: Target concept

        Returns:
            MetaAnalysisResult if sufficient experiments exist
        """
        # Find completed experiments for this edge
        query = select(ExperimentEdge).where(
            ExperimentEdge.source_concept == source_concept,
            ExperimentEdge.target_concept == target_concept,
            ExperimentEdge.status == ExperimentEdgeStatus.COMPLETED,
        )
        result = await self.db.execute(query)
        experiments = result.scalars().all()

        if len(experiments) < 2:
            logger.info("Not enough experiments for meta-analysis (need >= 2)")
            return None

        # Collect effect sizes and variances
        effects = []
        variances = []
        sample_sizes = []

        for exp in experiments:
            if exp.experiment_effect_size is not None and exp.experiment_sample_size:
                effects.append(exp.experiment_effect_size)
                # Approximate variance from sample size
                variance = 4.0 / exp.experiment_sample_size  # Variance of Cohen's d
                variances.append(variance)
                sample_sizes.append(exp.experiment_sample_size)

        if len(effects) < 2:
            return None

        # Random effects meta-analysis
        pooled_effect, pooled_se, i_squared, q_stat = self._random_effects_meta(
            effects=effects,
            variances=variances,
        )

        # Compute p-value
        z = pooled_effect / pooled_se if pooled_se > 0 else 0
        pooled_p_value = 2 * (1 - self._normal_cdf(abs(z)))

        # Compute 95% CI
        ci_lower = pooled_effect - 1.96 * pooled_se
        ci_upper = pooled_effect + 1.96 * pooled_se

        # Determine conclusion
        if pooled_p_value < 0.05 and pooled_effect > 0.2:
            conclusion = 'supported'
        elif pooled_p_value < 0.05 and pooled_effect < -0.2:
            conclusion = 'refuted'
        else:
            conclusion = 'inconclusive'

        # Compute combined confidence
        combined_confidence = self._effect_to_confidence(pooled_effect, pooled_p_value)

        # Create result
        meta_result = MetaAnalysisResult(
            source_concept=source_concept,
            target_concept=target_concept,
            experiment_ids=[exp.id for exp in experiments],
            num_experiments=len(experiments),
            total_sample_size=sum(sample_sizes),
            pooled_effect_size=pooled_effect,
            pooled_effect_ci_lower=ci_lower,
            pooled_effect_ci_upper=ci_upper,
            pooled_p_value=pooled_p_value,
            heterogeneity_i2=i_squared,
            heterogeneity_q=q_stat,
            combined_confidence=combined_confidence,
            conclusion=conclusion,
            analysis_method='random_effects',
        )

        self.db.add(meta_result)
        await self.db.commit()

        logger.info(
            f"Meta-analysis for {source_concept}→{target_concept}: "
            f"effect={pooled_effect:.3f}, p={pooled_p_value:.4f}, conclusion={conclusion}"
        )

        return meta_result

    def _random_effects_meta(
        self,
        effects: List[float],
        variances: List[float],
    ) -> Tuple[float, float, float, float]:
        """
        Perform random effects meta-analysis.

        Returns:
            (pooled_effect, pooled_se, i_squared, q_statistic)
        """
        k = len(effects)

        # Fixed effects weights
        weights = [1.0 / v for v in variances]
        total_weight = sum(weights)

        # Fixed effects estimate
        fixed_effect = sum(w * e for w, e in zip(weights, effects)) / total_weight

        # Q statistic (heterogeneity)
        q_stat = sum(w * (e - fixed_effect) ** 2 for w, e in zip(weights, effects))

        # Degrees of freedom
        df = k - 1

        # I² statistic
        i_squared = max(0, (q_stat - df) / q_stat * 100) if q_stat > 0 else 0

        # Tau² (between-study variance) using DerSimonian-Laird
        c = total_weight - sum(w ** 2 for w in weights) / total_weight
        tau_squared = max(0, (q_stat - df) / c) if c > 0 else 0

        # Random effects weights
        re_weights = [1.0 / (v + tau_squared) for v in variances]
        re_total_weight = sum(re_weights)

        # Random effects estimate
        pooled_effect = sum(w * e for w, e in zip(re_weights, effects)) / re_total_weight

        # Standard error
        pooled_se = math.sqrt(1.0 / re_total_weight) if re_total_weight > 0 else 1.0

        return pooled_effect, pooled_se, i_squared, q_stat

    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _effect_to_confidence(self, effect: float, p_value: float) -> float:
        """Convert effect size and p-value to confidence score."""
        if p_value >= 0.05:
            # Not significant - confidence stays near prior
            return 0.5

        # Scale effect to confidence
        # Large positive effect -> high confidence
        # Large negative effect -> low confidence (reverse causation)
        sigmoid = 1.0 / (1.0 + math.exp(-2 * effect))

        return sigmoid
