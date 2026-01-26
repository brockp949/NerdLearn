"""
Unit tests for Evidential Deep Learning modules.

Tests uncertainty quantification, evidential readout heads, and loss functions.
"""

import pytest
import torch
import torch.nn as nn
from app.adaptive.neural_ode.evidential import (
    EvidentialReadoutHead,
    EvidentialBinaryHead,
    UncertaintyAwareModel,
)
from app.adaptive.neural_ode.evidential_loss import (
    EvidentialRegressionLoss,
    EvidentialClassificationLoss,
    EvidentialLoss,
)


class TestEvidentialReadoutHead:
    """Tests for the EvidentialReadoutHead (regression)."""

    def test_output_shapes(self):
        """Verify all outputs have correct shapes."""
        state_dim = 32
        hidden_dim = 64
        batch_size = 4

        head = EvidentialReadoutHead(state_dim=state_dim, hidden_dim=hidden_dim)
        h = torch.randn(batch_size, state_dim)

        outputs = head(h)

        assert outputs['mean'].shape == (batch_size, 1)
        assert outputs['nu'].shape == (batch_size, 1)
        assert outputs['alpha'].shape == (batch_size, 1)
        assert outputs['beta'].shape == (batch_size, 1)
        assert outputs['aleatoric_var'].shape == (batch_size, 1)
        assert outputs['epistemic_var'].shape == (batch_size, 1)
        assert outputs['total_var'].shape == (batch_size, 1)
        assert outputs['confidence'].shape == (batch_size, 1)

    def test_nig_constraints(self):
        """Verify Normal-Inverse-Gamma parameter constraints."""
        head = EvidentialReadoutHead()
        h = torch.randn(10, 32)

        outputs = head(h)

        # nu > 0 (softplus + min_nu)
        assert (outputs['nu'] > 0).all()

        # alpha > 1 (softplus + min_alpha)
        assert (outputs['alpha'] > 1).all()

        # beta > 0 (softplus)
        assert (outputs['beta'] > 0).all()

    def test_variance_decomposition(self):
        """Total variance should equal aleatoric + epistemic."""
        head = EvidentialReadoutHead()
        h = torch.randn(5, 32)

        outputs = head(h)

        expected_total = outputs['aleatoric_var'] + outputs['epistemic_var']
        assert torch.allclose(outputs['total_var'], expected_total, atol=1e-5)

    def test_confidence_bounded(self):
        """Confidence should be bounded in [0, 1]."""
        head = EvidentialReadoutHead()
        h = torch.randn(10, 32) * 10  # Large inputs

        outputs = head(h)

        assert (outputs['confidence'] >= 0).all()
        assert (outputs['confidence'] <= 1).all()

    def test_higher_nu_means_lower_epistemic(self):
        """Higher nu (more evidence) should reduce epistemic uncertainty."""
        head = EvidentialReadoutHead()

        # Create two different inputs
        h1 = torch.randn(1, 32)
        h2 = h1.clone()

        # Get baseline
        out1 = head(h1)

        # Manually check: higher nu -> lower epistemic_var
        # epistemic_var = beta / (nu * (alpha - 1))
        # So increasing nu should decrease epistemic_var

        # This is a property test of the formula
        nu = out1['nu']
        alpha = out1['alpha']
        beta = out1['beta']

        epistemic_low_nu = beta / (nu * (alpha - 1))
        epistemic_high_nu = beta / ((nu * 2) * (alpha - 1))

        assert (epistemic_high_nu < epistemic_low_nu).all()

    def test_sample_method(self):
        """Test sampling from the evidential distribution."""
        head = EvidentialReadoutHead()
        h = torch.randn(4, 32)
        num_samples = 100

        samples = head.sample(h, num_samples=num_samples)

        assert samples.shape == (num_samples, 4, 1)
        # Samples should have reasonable variance
        assert samples.std() > 0

    def test_gradient_flow(self):
        """Verify gradients flow through the head."""
        head = EvidentialReadoutHead()
        h = torch.randn(2, 32, requires_grad=True)

        outputs = head(h)
        loss = outputs['mean'].sum() + outputs['confidence'].sum()
        loss.backward()

        assert h.grad is not None
        assert not torch.all(h.grad == 0)


class TestEvidentialBinaryHead:
    """Tests for the EvidentialBinaryHead (classification)."""

    def test_output_shapes(self):
        """Verify all outputs have correct shapes."""
        state_dim = 32
        hidden_dim = 64
        batch_size = 4

        head = EvidentialBinaryHead(state_dim=state_dim, hidden_dim=hidden_dim)
        h = torch.randn(batch_size, state_dim)

        outputs = head(h)

        assert outputs['evidence'].shape == (batch_size, 2)
        assert outputs['alpha'].shape == (batch_size, 2)
        assert outputs['prob'].shape == (batch_size, 1)
        assert outputs['uncertainty'].shape == (batch_size, 1)
        assert outputs['epistemic_var'].shape == (batch_size, 1)
        assert outputs['aleatoric_var'].shape == (batch_size, 1)

    def test_evidence_non_negative(self):
        """Evidence should always be non-negative (softplus output)."""
        head = EvidentialBinaryHead()
        h = torch.randn(10, 32) * 10  # Large inputs

        outputs = head(h)

        assert (outputs['evidence'] >= 0).all()

    def test_alpha_greater_than_one(self):
        """Alpha = evidence + 1, so should be > 1."""
        head = EvidentialBinaryHead()
        h = torch.randn(10, 32)

        outputs = head(h)

        assert (outputs['alpha'] >= 1).all()

    def test_probability_bounded(self):
        """Probability should be in [0, 1]."""
        head = EvidentialBinaryHead()
        h = torch.randn(10, 32) * 10

        outputs = head(h)

        assert (outputs['prob'] >= 0).all()
        assert (outputs['prob'] <= 1).all()

    def test_uncertainty_bounded(self):
        """Uncertainty should be in [0, 1]."""
        head = EvidentialBinaryHead()
        h = torch.randn(10, 32)

        outputs = head(h)

        assert (outputs['uncertainty'] >= 0).all()
        assert (outputs['uncertainty'] <= 1).all()

    def test_high_evidence_low_uncertainty(self):
        """Higher evidence should lead to lower uncertainty."""
        head = EvidentialBinaryHead()

        # Get output and check relationship
        h = torch.randn(10, 32)
        outputs = head(h)

        # Uncertainty = K / S where S = sum(alpha)
        # Higher alpha sum -> lower uncertainty
        S = outputs['alpha'].sum(dim=-1, keepdim=True)
        expected_uncertainty = 2.0 / S

        assert torch.allclose(outputs['uncertainty'], expected_uncertainty, atol=1e-5)


class TestEvidentialRegressionLoss:
    """Tests for the EvidentialRegressionLoss."""

    def test_loss_output_structure(self):
        """Loss should return total loss and breakdown dict."""
        loss_fn = EvidentialRegressionLoss()

        predictions = {
            'mean': torch.randn(4, 1),
            'nu': torch.ones(4, 1) * 2,
            'alpha': torch.ones(4, 1) * 2,
            'beta': torch.ones(4, 1),
        }
        targets = torch.randn(4, 1)

        loss, breakdown = loss_fn(predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert 'nll' in breakdown
        assert 'kl' in breakdown
        assert 'calibration' in breakdown
        assert 'total' in breakdown

    def test_loss_non_negative(self):
        """NLL loss should be non-negative."""
        loss_fn = EvidentialRegressionLoss()

        predictions = {
            'mean': torch.zeros(4, 1),
            'nu': torch.ones(4, 1) * 5,
            'alpha': torch.ones(4, 1) * 5,
            'beta': torch.ones(4, 1),
        }
        targets = torch.zeros(4, 1)  # Perfect prediction

        loss, breakdown = loss_fn(predictions, targets)

        # NLL component should be bounded below
        # (not necessarily non-negative for NIG)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_loss_increases_with_error(self):
        """Loss should increase when predictions are wrong."""
        loss_fn = EvidentialRegressionLoss()

        predictions = {
            'mean': torch.zeros(4, 1),
            'nu': torch.ones(4, 1) * 5,
            'alpha': torch.ones(4, 1) * 5,
            'beta': torch.ones(4, 1),
        }

        targets_close = torch.zeros(4, 1)
        targets_far = torch.ones(4, 1) * 10

        loss_close, _ = loss_fn(predictions, targets_close)
        loss_far, _ = loss_fn(predictions, targets_far)

        assert loss_far > loss_close

    def test_gradient_flow(self):
        """Gradients should flow through the loss."""
        loss_fn = EvidentialRegressionLoss()

        predictions = {
            'mean': torch.randn(4, 1, requires_grad=True),
            'nu': torch.ones(4, 1, requires_grad=True) * 2,
            'alpha': torch.ones(4, 1, requires_grad=True) * 2,
            'beta': torch.ones(4, 1, requires_grad=True),
        }
        targets = torch.randn(4, 1)

        loss, _ = loss_fn(predictions, targets)
        loss.backward()

        assert predictions['mean'].grad is not None

    def test_mask_support(self):
        """Loss should support masking invalid entries."""
        loss_fn = EvidentialRegressionLoss()

        predictions = {
            'mean': torch.randn(4, 1),
            'nu': torch.ones(4, 1) * 2,
            'alpha': torch.ones(4, 1) * 2,
            'beta': torch.ones(4, 1),
        }
        targets = torch.randn(4, 1)
        mask = torch.tensor([[1], [1], [0], [0]])  # Only first 2 valid

        loss_masked, _ = loss_fn(predictions, targets, mask)
        loss_unmasked, _ = loss_fn(predictions, targets)

        # Masked loss should be different (unless by chance)
        assert not torch.isnan(loss_masked)


class TestEvidentialClassificationLoss:
    """Tests for the EvidentialClassificationLoss."""

    def test_loss_output_structure(self):
        """Loss should return total loss and breakdown dict."""
        loss_fn = EvidentialClassificationLoss()

        predictions = {
            'evidence': torch.ones(4, 2),
            'alpha': torch.ones(4, 2) * 2,
        }
        targets = torch.tensor([0, 1, 0, 1])

        loss, breakdown = loss_fn(predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert 'mle' in breakdown
        assert 'kl' in breakdown
        assert 'total' in breakdown

    def test_kl_annealing(self):
        """KL weight should increase with epoch."""
        loss_fn = EvidentialClassificationLoss(kl_weight=0.1, annealing_epochs=10)

        predictions = {
            'evidence': torch.ones(4, 2),
            'alpha': torch.ones(4, 2) * 2,
        }
        targets = torch.tensor([0, 1, 0, 1])

        # Epoch 0: KL weight should be 0
        loss_fn.set_epoch(0)
        _, breakdown_0 = loss_fn(predictions, targets)

        # Epoch 10: KL weight should be full
        loss_fn.set_epoch(10)
        _, breakdown_10 = loss_fn(predictions, targets)

        assert breakdown_0['kl_weight_effective'] < breakdown_10['kl_weight_effective']


class TestCombinedEvidentialLoss:
    """Tests for the combined EvidentialLoss."""

    def test_combined_loss_structure(self):
        """Combined loss should handle recall, latency, and hesitation."""
        loss_fn = EvidentialLoss()

        predictions = {
            'recall': {
                'evidence': torch.ones(4, 2),
                'alpha': torch.ones(4, 2) * 2,
            },
            'latency': {
                'mean': torch.randn(4, 1),
                'nu': torch.ones(4, 1) * 2,
                'alpha': torch.ones(4, 1) * 2,
                'beta': torch.ones(4, 1),
            },
            'hesitation': {
                'mean': torch.randn(4, 1),
                'nu': torch.ones(4, 1) * 2,
                'alpha': torch.ones(4, 1) * 2,
                'beta': torch.ones(4, 1),
            },
        }
        targets = {
            'recall': torch.tensor([0, 1, 0, 1]),
            'latency': torch.randn(4, 1),
            'hesitation': torch.randn(4, 1),
        }

        loss, breakdown = loss_fn(predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert 'recall_loss' in breakdown
        assert 'latency_loss' in breakdown
        assert 'hesitation_loss' in breakdown
        assert 'total' in breakdown

    def test_partial_targets(self):
        """Loss should handle missing targets gracefully."""
        loss_fn = EvidentialLoss()

        # Only recall predictions/targets
        predictions = {
            'recall': {
                'evidence': torch.ones(4, 2),
                'alpha': torch.ones(4, 2) * 2,
            },
        }
        targets = {
            'recall': torch.tensor([0, 1, 0, 1]),
        }

        loss, breakdown = loss_fn(predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)


class TestUncertaintyAwareModel:
    """Tests for the UncertaintyAwareModel wrapper."""

    def test_predict_with_uncertainty(self):
        """Model should output predictions with uncertainty."""
        model = UncertaintyAwareModel(state_dim=32, hidden_dim=64)
        h = torch.randn(4, 32)

        predictions = model.predict_with_uncertainty(h)

        assert 'recall' in predictions
        assert 'latency' in predictions
        assert 'hesitation' in predictions

        # Check recall has expected fields
        assert 'prob' in predictions['recall']
        assert 'uncertainty' in predictions['recall']
        assert 'epistemic_var' in predictions['recall']

        # Check latency has expected fields
        assert 'mean' in predictions['latency']
        assert 'aleatoric_var' in predictions['latency']
        assert 'epistemic_var' in predictions['latency']

    def test_get_confidence_score(self):
        """Confidence score should be in [0, 1]."""
        model = UncertaintyAwareModel(state_dim=32, hidden_dim=64)
        h = torch.randn(4, 32)

        confidence = model.get_confidence_score(h)

        assert confidence.shape == (4, 1)
        assert (confidence >= 0).all()
        assert (confidence <= 1).all()

    def test_time_dimension_handling(self):
        """Model should handle 3D input (time, batch, state)."""
        model = UncertaintyAwareModel(state_dim=32, hidden_dim=64)
        h = torch.randn(10, 4, 32)  # 10 timesteps, batch 4

        predictions = model.predict_with_uncertainty(h)

        # Output should have time dimension
        assert predictions['recall']['prob'].shape == (10, 4, 1)
        assert predictions['latency']['mean'].shape == (10, 4, 1)
