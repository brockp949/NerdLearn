"""
Evidential Deep Learning for Neural ODE Uncertainty Quantification.

Implements uncertainty-aware readout heads that output both predictions
and confidence estimates (epistemic and aleatoric uncertainty).

Based on:
- Amini et al. "Deep Evidential Regression" (NeurIPS 2020)
- Sensoy et al. "Evidential Deep Learning" (NeurIPS 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math

from .model import NeuralMemoryODE


class EvidentialReadoutHead(nn.Module):
    """
    Evidential readout head for uncertainty quantification.

    Instead of outputting a single point estimate, outputs parameters of a
    Normal-Inverse-Gamma (NIG) distribution:
    - γ (gamma): Mean prediction
    - ν (nu): Confidence in the mean (pseudo-observations)
    - α (alpha): Shape parameter (> 1)
    - β (beta): Scale parameter (> 0)

    From these, we derive:
    - Aleatoric uncertainty: β / (α - 1) - inherent data noise
    - Epistemic uncertainty: β / (ν * (α - 1)) - model uncertainty
    """

    def __init__(
        self,
        state_dim: int = 32,
        hidden_dim: int = 64,
        min_nu: float = 0.1,
        min_alpha: float = 1.01,
    ):
        """
        Initialize evidential readout head.

        Args:
            state_dim: Dimension of input latent state
            hidden_dim: Hidden layer dimension
            min_nu: Minimum value for ν (prevents division by zero)
            min_alpha: Minimum value for α (must be > 1 for valid variance)
        """
        super().__init__()

        self.min_nu = min_nu
        self.min_alpha = min_alpha

        # Shared feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Output heads for NIG parameters
        self.gamma_head = nn.Linear(hidden_dim, 1)  # Mean
        self.nu_head = nn.Linear(hidden_dim, 1)  # Confidence
        self.alpha_head = nn.Linear(hidden_dim, 1)  # Shape
        self.beta_head = nn.Linear(hidden_dim, 1)  # Scale

    def forward(
        self,
        h: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass outputting NIG parameters and derived uncertainties.

        Args:
            h: Latent state [batch, state_dim] or [time, batch, state_dim]

        Returns:
            Dictionary with:
                - 'mean': Predicted mean [batch, 1]
                - 'nu': Confidence parameter [batch, 1]
                - 'alpha': Shape parameter [batch, 1]
                - 'beta': Scale parameter [batch, 1]
                - 'aleatoric_var': Aleatoric variance [batch, 1]
                - 'epistemic_var': Epistemic variance [batch, 1]
                - 'total_var': Total variance [batch, 1]
                - 'confidence': Derived confidence score [batch, 1]
        """
        # Extract features
        features = self.feature_net(h)

        # Get NIG parameters with appropriate constraints
        gamma = self.gamma_head(features)  # Unbounded mean

        # ν > 0 (use softplus)
        nu = F.softplus(self.nu_head(features)) + self.min_nu

        # α > 1 (use softplus + offset)
        alpha = F.softplus(self.alpha_head(features)) + self.min_alpha

        # β > 0 (use softplus)
        beta = F.softplus(self.beta_head(features)) + 1e-6

        # Compute derived quantities
        # Aleatoric variance: E[σ²] = β / (α - 1)
        aleatoric_var = beta / (alpha - 1)

        # Epistemic variance: Var[μ] = β / (ν * (α - 1))
        epistemic_var = beta / (nu * (alpha - 1))

        # Total variance
        total_var = aleatoric_var + epistemic_var

        # Confidence score: Higher ν and α mean more confident
        # Normalized to roughly [0, 1]
        confidence = 1.0 / (1.0 + epistemic_var)

        return {
            'mean': gamma,
            'nu': nu,
            'alpha': alpha,
            'beta': beta,
            'aleatoric_var': aleatoric_var,
            'epistemic_var': epistemic_var,
            'total_var': total_var,
            'confidence': confidence,
        }

    def sample(
        self,
        h: torch.Tensor,
        num_samples: int = 100,
    ) -> torch.Tensor:
        """
        Sample predictions from the evidential distribution.

        Useful for generating prediction intervals.

        Args:
            h: Latent state [batch, state_dim]
            num_samples: Number of samples to draw

        Returns:
            Samples [num_samples, batch, 1]
        """
        outputs = self.forward(h)
        gamma = outputs['mean']
        nu = outputs['nu']
        alpha = outputs['alpha']
        beta = outputs['beta']

        batch_size = h.size(0)
        samples = []

        for _ in range(num_samples):
            # Sample σ² from Inverse-Gamma(α, β)
            # Using the relationship: if X ~ Gamma(α, β), then 1/X ~ InvGamma(α, β)
            gamma_samples = torch.distributions.Gamma(alpha, beta).sample()
            sigma_sq = 1.0 / gamma_samples

            # Sample μ from Normal(γ, σ²/ν)
            mu_var = sigma_sq / nu
            mu = torch.normal(gamma, torch.sqrt(mu_var))

            samples.append(mu)

        return torch.stack(samples)


class EvidentialBinaryHead(nn.Module):
    """
    Evidential head for binary classification (recall prediction).

    Uses Dirichlet distribution to model uncertainty over class probabilities.
    Outputs evidence for each class, from which we derive probabilities
    and uncertainty.
    """

    def __init__(
        self,
        state_dim: int = 32,
        hidden_dim: int = 64,
    ):
        """
        Initialize evidential binary classification head.

        Args:
            state_dim: Dimension of input latent state
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # Evidence for [not_recall, recall]
        )

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass outputting Dirichlet parameters.

        Args:
            h: Latent state [batch, state_dim]

        Returns:
            Dictionary with:
                - 'evidence': Evidence for each class [batch, 2]
                - 'alpha': Dirichlet concentration [batch, 2]
                - 'prob': Expected probability of recall [batch, 1]
                - 'uncertainty': Uncertainty score [batch, 1]
                - 'epistemic_var': Epistemic uncertainty [batch, 1]
                - 'aleatoric_var': Aleatoric uncertainty [batch, 1]
        """
        # Get evidence (non-negative)
        evidence = F.softplus(self.net(h))

        # Dirichlet concentration parameters: α = evidence + 1
        alpha = evidence + 1

        # Dirichlet strength: S = sum(α)
        S = alpha.sum(dim=-1, keepdim=True)

        # Expected probability of recall (class 1)
        prob = alpha[:, 1:2] / S

        # Dirichlet uncertainty: K / S where K = num classes
        # Higher S → lower uncertainty
        K = 2
        uncertainty = K / S

        # Decompose uncertainty
        # Epistemic: uncertainty in the probability estimate
        epistemic_var = prob * (1 - prob) / (S + 1)

        # Aleatoric: irreducible uncertainty (class overlap)
        aleatoric_var = prob * (1 - prob) * S / (S + 1)

        return {
            'evidence': evidence,
            'alpha': alpha,
            'prob': prob,
            'uncertainty': uncertainty,
            'epistemic_var': epistemic_var,
            'aleatoric_var': aleatoric_var,
        }


class UncertaintyAwareModel(NeuralMemoryODE):
    """
    Extended Neural ODE model with uncertainty quantification.

    Replaces point-estimate readout heads with evidential heads
    that output predictions with epistemic and aleatoric uncertainty.
    """

    def __init__(
        self,
        state_dim: int = 32,
        hidden_dim: int = 64,
        card_feat_dim: int = 64,
        user_feat_dim: int = 16,
        use_gated_jump: bool = False,
    ):
        """Initialize uncertainty-aware model."""
        super().__init__(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            card_feat_dim=card_feat_dim,
            user_feat_dim=user_feat_dim,
            use_gated_jump=use_gated_jump,
        )

        # Replace readout heads with evidential versions
        self.evidential_recall = EvidentialBinaryHead(state_dim, hidden_dim)
        self.evidential_latency = EvidentialReadoutHead(state_dim, hidden_dim)
        self.evidential_hesitation = EvidentialReadoutHead(state_dim, hidden_dim)

    def predict_with_uncertainty(
        self,
        h: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Get predictions with uncertainty estimates.

        Args:
            h: Latent state [batch, state_dim] or [time, batch, state_dim]

        Returns:
            Dictionary with predictions and uncertainty for each output
        """
        # Handle time dimension if present
        original_shape = h.shape
        if h.dim() == 3:
            T, B, D = h.shape
            h = h.reshape(T * B, D)
            reshape_output = True
        else:
            reshape_output = False

        # Get evidential outputs
        recall_out = self.evidential_recall(h)
        latency_out = self.evidential_latency(h)
        hesitation_out = self.evidential_hesitation(h)

        # Reshape if needed
        if reshape_output:
            for key in recall_out:
                recall_out[key] = recall_out[key].reshape(T, B, -1)
            for key in latency_out:
                latency_out[key] = latency_out[key].reshape(T, B, -1)
            for key in hesitation_out:
                hesitation_out[key] = hesitation_out[key].reshape(T, B, -1)

        return {
            'recall': recall_out,
            'latency': latency_out,
            'hesitation': hesitation_out,
        }

    def predict_all_with_uncertainty(
        self,
        h0: torch.Tensor,
        t_eval: torch.Tensor,
        sleep_schedule_fn=None,
        stress_level: float = 0.0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Predict trajectories with uncertainty at each time point.

        Args:
            h0: Initial state [batch, state_dim]
            t_eval: Time points to evaluate [num_steps]
            sleep_schedule_fn: Optional sleep schedule function
            stress_level: Current stress level

        Returns:
            Dictionary with uncertainty-aware predictions at each time
        """
        # Get trajectory
        trajectory = self.forward(h0, t_eval, sleep_schedule_fn, stress_level)

        # Get predictions with uncertainty at each point
        return self.predict_with_uncertainty(trajectory)

    def get_confidence_score(
        self,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get overall confidence score for current state.

        Combines epistemic uncertainties from all heads.

        Args:
            h: Latent state [batch, state_dim]

        Returns:
            Confidence score [batch, 1] in [0, 1]
        """
        predictions = self.predict_with_uncertainty(h)

        # Get epistemic variances
        recall_epistemic = predictions['recall']['epistemic_var']
        latency_epistemic = predictions['latency']['epistemic_var']
        hesitation_epistemic = predictions['hesitation']['epistemic_var']

        # Normalize and combine (weighted by importance)
        # Recall is most important for scheduling
        recall_conf = 1.0 / (1.0 + recall_epistemic)
        latency_conf = 1.0 / (1.0 + latency_epistemic)
        hesitation_conf = 1.0 / (1.0 + hesitation_epistemic)

        # Weighted average
        confidence = 0.6 * recall_conf + 0.25 * latency_conf + 0.15 * hesitation_conf

        return confidence

    def get_prediction_interval(
        self,
        h: torch.Tensor,
        confidence_level: float = 0.95,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get prediction intervals for all outputs.

        Args:
            h: Latent state [batch, state_dim]
            confidence_level: Confidence level for interval (e.g., 0.95 for 95%)

        Returns:
            Dictionary mapping output name to (lower_bound, upper_bound)
        """
        predictions = self.predict_with_uncertainty(h)

        intervals = {}

        # Z-score for confidence level
        from scipy import stats
        z = stats.norm.ppf((1 + confidence_level) / 2)

        # Recall interval (using beta distribution approximation)
        recall_prob = predictions['recall']['prob']
        recall_var = predictions['recall']['epistemic_var'] + predictions['recall']['aleatoric_var']
        recall_std = torch.sqrt(recall_var)
        intervals['recall'] = (
            torch.clamp(recall_prob - z * recall_std, 0, 1),
            torch.clamp(recall_prob + z * recall_std, 0, 1),
        )

        # Latency interval
        latency_mean = predictions['latency']['mean']
        latency_std = torch.sqrt(predictions['latency']['total_var'])
        intervals['latency'] = (
            latency_mean - z * latency_std,
            latency_mean + z * latency_std,
        )

        # Hesitation interval
        hesitation_mean = predictions['hesitation']['mean']
        hesitation_std = torch.sqrt(predictions['hesitation']['total_var'])
        intervals['hesitation'] = (
            hesitation_mean - z * hesitation_std,
            hesitation_mean + z * hesitation_std,
        )

        return intervals
