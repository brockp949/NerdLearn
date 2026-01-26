"""
Unit tests for JCI (Joint Causal Inference) Confidence Updater.

Tests Bayesian updates combining observational and experimental evidence.
"""

import pytest
import math
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.adaptive.causal_discovery.jci_updater import (
    JCIConfidenceUpdater,
    BayesianUpdate,
)
from app.models.jci import (
    ExperimentEdge,
    ExperimentEdgeStatus,
    CausalDirection,
)


class TestBayesianUpdate:
    """Tests for Bayesian update logic."""

    def test_likelihood_ratio_significant_positive(self):
        """Significant positive effect should have high likelihood ratio."""
        updater = JCIConfidenceUpdater(db=MagicMock())

        lr = updater._compute_likelihood_ratio(
            effect_size=0.5,  # Medium positive effect
            p_value=0.01,     # Significant
            sample_size=200,
        )

        assert lr > 1.0  # Evidence favors causation

    def test_likelihood_ratio_not_significant(self):
        """Non-significant result should have likelihood ratio near 1."""
        updater = JCIConfidenceUpdater(db=MagicMock())

        lr = updater._compute_likelihood_ratio(
            effect_size=0.1,   # Small effect
            p_value=0.5,       # Not significant
            sample_size=50,
        )

        # Should be close to 1 (no strong evidence either way)
        assert 0.5 < lr < 2.0

    def test_likelihood_ratio_significant_negative(self):
        """Significant negative effect should affect likelihood ratio."""
        updater = JCIConfidenceUpdater(db=MagicMock())

        lr = updater._compute_likelihood_ratio(
            effect_size=-0.5,  # Negative effect
            p_value=0.01,      # Significant
            sample_size=200,
        )

        # Negative effect is still evidence, just against this direction
        assert lr != 1.0

    def test_bayesian_update_increases_with_positive_evidence(self):
        """Positive evidence should increase posterior confidence."""
        updater = JCIConfidenceUpdater(db=MagicMock())

        prior = 0.5
        lr = 3.0  # Strong evidence for causation

        posterior = updater._bayesian_update(prior, lr)

        assert posterior > prior

    def test_bayesian_update_decreases_with_negative_evidence(self):
        """Negative evidence should decrease posterior confidence."""
        updater = JCIConfidenceUpdater(db=MagicMock())

        prior = 0.5
        lr = 0.3  # Evidence against causation

        posterior = updater._bayesian_update(prior, lr)

        assert posterior < prior

    def test_bayesian_update_bounded(self):
        """Posterior should always be in (0, 1)."""
        updater = JCIConfidenceUpdater(db=MagicMock())

        # Extreme high likelihood ratio
        posterior_high = updater._bayesian_update(0.99, 100.0)
        assert 0 < posterior_high < 1

        # Extreme low likelihood ratio
        posterior_low = updater._bayesian_update(0.01, 0.01)
        assert 0 < posterior_low < 1


class TestEvidenceStrength:
    """Tests for evidence strength calculation."""

    def test_evidence_strength_high_for_significant_large_effect(self):
        """High evidence strength for significant, large effect, large N."""
        updater = JCIConfidenceUpdater(db=MagicMock())

        strength = updater._compute_evidence_strength(
            effect_size=0.8,   # Large
            p_value=0.001,     # Very significant
            sample_size=500,   # Large
        )

        assert strength > 0.7  # High strength

    def test_evidence_strength_low_for_non_significant(self):
        """Low evidence strength for non-significant results."""
        updater = JCIConfidenceUpdater(db=MagicMock())

        strength = updater._compute_evidence_strength(
            effect_size=0.3,
            p_value=0.2,   # Not significant
            sample_size=50,
        )

        assert strength < 0.5  # Low strength

    def test_evidence_strength_bounded(self):
        """Evidence strength should be in [0, 1]."""
        updater = JCIConfidenceUpdater(db=MagicMock())

        # Test various combinations
        test_cases = [
            (0.0, 0.99, 10),
            (1.0, 0.001, 1000),
            (0.5, 0.05, 100),
        ]

        for effect, p, n in test_cases:
            strength = updater._compute_evidence_strength(effect, p, n)
            assert 0 <= strength <= 1


class TestDirectionInference:
    """Tests for causal direction inference."""

    def test_forward_direction_positive_effect(self):
        """Positive significant effect implies forward direction."""
        updater = JCIConfidenceUpdater(db=MagicMock())

        direction = updater._infer_direction(
            effect_size=0.5,
            p_value=0.01,
        )

        assert direction == CausalDirection.FORWARD

    def test_reverse_direction_negative_effect(self):
        """Negative significant effect implies reverse direction."""
        updater = JCIConfidenceUpdater(db=MagicMock())

        direction = updater._infer_direction(
            effect_size=-0.5,
            p_value=0.01,
        )

        assert direction == CausalDirection.REVERSE

    def test_no_direction_non_significant(self):
        """Non-significant result implies no clear direction."""
        updater = JCIConfidenceUpdater(db=MagicMock())

        direction = updater._infer_direction(
            effect_size=0.5,
            p_value=0.1,  # Not significant
        )

        assert direction == CausalDirection.NONE

    def test_no_direction_small_effect(self):
        """Small effect (even if significant) implies no clear direction."""
        updater = JCIConfidenceUpdater(db=MagicMock())

        direction = updater._infer_direction(
            effect_size=0.1,  # Smaller than min_effect_size
            p_value=0.01,
        )

        assert direction == CausalDirection.NONE


class TestSampleSizeEstimation:
    """Tests for required sample size estimation."""

    def test_sample_size_higher_for_uncertain_edges(self):
        """Edges with confidence near 0.5 need more samples."""
        updater = JCIConfidenceUpdater(db=MagicMock())

        n_uncertain = updater._estimate_sample_size(0.5)  # Most uncertain
        n_confident = updater._estimate_sample_size(0.2)  # More certain

        # More uncertainty requires larger sample size
        assert n_uncertain >= n_confident

    def test_sample_size_bounded(self):
        """Sample size should be within reasonable bounds."""
        updater = JCIConfidenceUpdater(db=MagicMock())

        for conf in [0.1, 0.3, 0.5, 0.7, 0.9]:
            n = updater._estimate_sample_size(conf)
            assert 100 <= n <= 10000  # Reasonable bounds


class TestMetaAnalysis:
    """Tests for meta-analysis calculations."""

    def test_random_effects_meta_single_study(self):
        """Single study meta-analysis should return study values."""
        updater = JCIConfidenceUpdater(db=MagicMock())

        # With just one study
        effects = [0.5]
        variances = [0.1]

        pooled, se, i2, q = updater._random_effects_meta(effects, variances)

        assert pooled == 0.5  # Same as single study
        assert se > 0

    def test_random_effects_meta_multiple_studies(self):
        """Multiple studies should be combined."""
        updater = JCIConfidenceUpdater(db=MagicMock())

        # Two studies with similar effects
        effects = [0.4, 0.6]
        variances = [0.1, 0.1]

        pooled, se, i2, q = updater._random_effects_meta(effects, variances)

        # Pooled should be between individual effects
        assert 0.4 <= pooled <= 0.6
        assert se > 0
        assert 0 <= i2 <= 100

    def test_random_effects_meta_heterogeneous(self):
        """Heterogeneous studies should have high I²."""
        updater = JCIConfidenceUpdater(db=MagicMock())

        # Very different effects
        effects = [0.1, 0.9]
        variances = [0.01, 0.01]  # Low within-study variance

        pooled, se, i2, q = updater._random_effects_meta(effects, variances)

        # Should have high heterogeneity
        assert i2 > 50  # Substantial heterogeneity


class TestEffectToConfidence:
    """Tests for converting effect size to confidence."""

    def test_positive_effect_high_confidence(self):
        """Large positive effect should give high confidence."""
        updater = JCIConfidenceUpdater(db=MagicMock())

        conf = updater._effect_to_confidence(0.8, 0.01)

        assert conf > 0.7

    def test_negative_effect_low_confidence(self):
        """Large negative effect should give low confidence."""
        updater = JCIConfidenceUpdater(db=MagicMock())

        conf = updater._effect_to_confidence(-0.8, 0.01)

        assert conf < 0.3

    def test_non_significant_near_half(self):
        """Non-significant result should give confidence near 0.5."""
        updater = JCIConfidenceUpdater(db=MagicMock())

        conf = updater._effect_to_confidence(0.5, 0.1)

        assert 0.4 <= conf <= 0.6


class TestLiftToCohenD:
    """Tests for lift percentage to Cohen's d conversion."""

    def test_lift_to_cohens_d_positive(self):
        """Positive lift should give positive Cohen's d."""
        updater = JCIConfidenceUpdater(db=MagicMock())

        d = updater._lift_to_cohens_d(20.0, 0.5)  # 20% lift, std=0.5

        assert d > 0
        assert d == pytest.approx(0.4)  # 0.2 / 0.5 = 0.4

    def test_lift_to_cohens_d_zero_std(self):
        """Zero std deviation should return 0."""
        updater = JCIConfidenceUpdater(db=MagicMock())

        d = updater._lift_to_cohens_d(20.0, 0.0)

        assert d == 0.0


class TestAsyncMethods:
    """Tests for async methods (mocked DB)."""

    @pytest.mark.asyncio
    async def test_create_experiment_edge_link(self):
        """Test creating experiment-edge link."""
        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()

        updater = JCIConfidenceUpdater(db=mock_db)

        edge = await updater.create_experiment_edge_link(
            experiment_id="exp_123",
            source_concept="Concept A",
            target_concept="Concept B",
            prior_confidence=0.6,
            course_id=1,
        )

        assert mock_db.add.called
        assert mock_db.flush.called

    @pytest.mark.asyncio
    async def test_recommend_experiments(self):
        """Test experiment recommendations."""
        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()

        mock_graph_service = AsyncMock()
        mock_graph_service.get_causal_edges = AsyncMock(return_value=[
            {'source': 'A', 'target': 'B', 'confidence': 0.5},
            {'source': 'B', 'target': 'C', 'confidence': 0.8},
            {'source': 'C', 'target': 'D', 'confidence': 0.4},
        ])

        updater = JCIConfidenceUpdater(db=mock_db, graph_service=mock_graph_service)

        recommendations = await updater.recommend_experiments(
            course_id=1,
            max_recommendations=3,
        )

        assert len(recommendations) <= 3
        # Should prioritize edges with confidence near 0.5
        if recommendations:
            # First recommendation should be most uncertain
            assert recommendations[0]['current_confidence'] in [0.4, 0.5]


class TestIntegrationScenarios:
    """Integration-style tests for full workflows."""

    def test_full_bayesian_update_workflow(self):
        """Test complete prior -> experiment -> posterior workflow."""
        updater = JCIConfidenceUpdater(db=MagicMock())

        # Start with uncertain prior
        prior = 0.5

        # Simulate significant positive experiment
        experiment_results = {
            'effect_size': 0.6,
            'p_value': 0.01,
            'sample_size': 200,
        }

        # Compute likelihood ratio
        lr = updater._compute_likelihood_ratio(
            effect_size=experiment_results['effect_size'],
            p_value=experiment_results['p_value'],
            sample_size=experiment_results['sample_size'],
        )

        # Bayesian update
        posterior = updater._bayesian_update(prior, lr)

        # Check update direction and magnitude
        assert posterior > prior
        assert posterior > 0.6  # Should be fairly confident now

        # Evidence strength should be high
        strength = updater._compute_evidence_strength(
            experiment_results['effect_size'],
            experiment_results['p_value'],
            experiment_results['sample_size'],
        )
        assert strength > 0.5

        # Direction should be forward
        direction = updater._infer_direction(
            experiment_results['effect_size'],
            experiment_results['p_value'],
        )
        assert direction == CausalDirection.FORWARD

    def test_inconclusive_experiment_workflow(self):
        """Test workflow when experiment is inconclusive."""
        updater = JCIConfidenceUpdater(db=MagicMock())

        prior = 0.5

        # Simulate non-significant experiment
        experiment_results = {
            'effect_size': 0.1,
            'p_value': 0.3,
            'sample_size': 50,
        }

        lr = updater._compute_likelihood_ratio(
            effect_size=experiment_results['effect_size'],
            p_value=experiment_results['p_value'],
            sample_size=experiment_results['sample_size'],
        )

        posterior = updater._bayesian_update(prior, lr)

        # Posterior should be close to prior (weak evidence)
        assert abs(posterior - prior) < 0.2

        # Evidence strength should be low
        strength = updater._compute_evidence_strength(
            experiment_results['effect_size'],
            experiment_results['p_value'],
            experiment_results['sample_size'],
        )
        assert strength < 0.5

        # Direction should be none
        direction = updater._infer_direction(
            experiment_results['effect_size'],
            experiment_results['p_value'],
        )
        assert direction == CausalDirection.NONE
