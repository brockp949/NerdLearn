"""
Evidential Loss Functions for Uncertainty-Aware Training.

Implements loss functions for evidential deep learning that jointly
optimize predictions and uncertainty estimates.

References:
- Amini et al. "Deep Evidential Regression" (NeurIPS 2020)
- Sensoy et al. "Evidential Deep Learning" (NeurIPS 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class EvidentialRegressionLoss(nn.Module):
    """
    Evidential loss for regression with Normal-Inverse-Gamma prior.

    Total loss = NLL + λ_kl * KL_regularization + λ_calib * Calibration

    The NLL encourages accurate predictions while the KL term prevents
    the model from being overconfident (assigning too much evidence
    to incorrect predictions).
    """

    def __init__(
        self,
        kl_weight: float = 0.1,
        calibration_weight: float = 0.05,
        epsilon: float = 1e-6,
    ):
        """
        Initialize evidential regression loss.

        Args:
            kl_weight: Weight for KL divergence regularization
            calibration_weight: Weight for calibration penalty
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.kl_weight = kl_weight
        self.calibration_weight = calibration_weight
        self.epsilon = epsilon

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute evidential regression loss.

        Args:
            predictions: Dict with 'mean', 'nu', 'alpha', 'beta'
            targets: Ground truth values [batch, 1]
            mask: Optional mask for valid entries [batch, 1]

        Returns:
            (total_loss, breakdown_dict)
        """
        gamma = predictions['mean']  # [batch, 1]
        nu = predictions['nu']  # [batch, 1]
        alpha = predictions['alpha']  # [batch, 1]
        beta = predictions['beta']  # [batch, 1]

        # Ensure shapes match
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)

        # NLL of Normal-Inverse-Gamma
        # -log p(y | γ, ν, α, β)
        error = (targets - gamma) ** 2
        nll = 0.5 * torch.log(math.pi / nu) \
            - alpha * torch.log(beta) \
            + (alpha + 0.5) * torch.log(beta + 0.5 * nu * error) \
            + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)

        # KL Divergence regularization
        # Penalizes evidence on wrong predictions
        kl_div = self._kl_divergence(gamma, nu, alpha, beta, targets)

        # Calibration penalty
        # Encourages predicted variance to match empirical error
        aleatoric_var = beta / (alpha - 1 + self.epsilon)
        calib_penalty = (error - aleatoric_var).abs()

        # Apply mask if provided
        if mask is not None:
            nll = nll * mask
            kl_div = kl_div * mask
            calib_penalty = calib_penalty * mask
            n_valid = mask.sum().clamp(min=1)
        else:
            n_valid = torch.tensor(targets.numel(), device=targets.device)

        # Aggregate losses
        nll_loss = nll.sum() / n_valid
        kl_loss = kl_div.sum() / n_valid
        calib_loss = calib_penalty.sum() / n_valid

        total_loss = nll_loss + self.kl_weight * kl_loss + self.calibration_weight * calib_loss

        return total_loss, {
            'nll': nll_loss.item(),
            'kl': kl_loss.item(),
            'calibration': calib_loss.item(),
            'total': total_loss.item(),
        }

    def _kl_divergence(
        self,
        gamma: torch.Tensor,
        nu: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence regularization term.

        Encourages the model to reduce evidence when predictions are wrong.
        """
        error = torch.abs(targets - gamma)

        # Scale evidence reduction by prediction error
        # More error → more penalty for high evidence
        kl = error * (2 * nu + alpha)

        return kl


class EvidentialClassificationLoss(nn.Module):
    """
    Evidential loss for binary classification with Dirichlet prior.

    Uses the Dirichlet distribution to model uncertainty over class
    probabilities. The loss includes:
    - Type II Maximum Likelihood
    - KL divergence to uniform Dirichlet (regularization)
    """

    def __init__(
        self,
        kl_weight: float = 0.1,
        annealing_epochs: int = 10,
    ):
        """
        Initialize evidential classification loss.

        Args:
            kl_weight: Weight for KL divergence regularization
            annealing_epochs: Number of epochs for KL annealing
        """
        super().__init__()
        self.kl_weight = kl_weight
        self.annealing_epochs = annealing_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """Update current epoch for KL annealing."""
        self.current_epoch = epoch

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute evidential classification loss.

        Args:
            predictions: Dict with 'evidence', 'alpha'
            targets: Ground truth labels {0, 1} [batch] or [batch, 1]
            mask: Optional mask for valid entries [batch] or [batch, 1]

        Returns:
            (total_loss, breakdown_dict)
        """
        evidence = predictions['evidence']  # [batch, 2]
        alpha = predictions['alpha']  # [batch, 2]

        # Ensure target shape
        if targets.dim() == 2:
            targets = targets.squeeze(-1)

        # Convert targets to one-hot
        y = F.one_hot(targets.long(), num_classes=2).float()  # [batch, 2]

        # Dirichlet strength
        S = alpha.sum(dim=-1, keepdim=True)  # [batch, 1]

        # Type II Maximum Likelihood loss
        # -E[log p(y | θ)] where θ ~ Dir(α)
        # = sum_k y_k * (digamma(S) - digamma(α_k))
        digamma_S = torch.digamma(S)
        digamma_alpha = torch.digamma(alpha)
        mle_loss = torch.sum(y * (digamma_S - digamma_alpha), dim=-1)  # [batch]

        # KL divergence to uniform Dirichlet Dir(1, 1)
        # Encourages uniform (uncertain) predictions for misclassified samples
        # Remove evidence for correct class to avoid penalizing correct confidence
        alpha_tilde = y + (1 - y) * alpha  # Keep evidence only for incorrect class
        kl_div = self._kl_divergence_dirichlet(alpha_tilde)  # [batch]

        # KL annealing: gradually increase KL weight
        annealing_factor = min(1.0, self.current_epoch / self.annealing_epochs)
        effective_kl_weight = self.kl_weight * annealing_factor

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.squeeze(-1)
            mle_loss = mle_loss * mask
            kl_div = kl_div * mask
            n_valid = mask.sum().clamp(min=1)
        else:
            n_valid = torch.tensor(targets.numel(), device=targets.device)

        # Aggregate losses
        mle_mean = mle_loss.sum() / n_valid
        kl_mean = kl_div.sum() / n_valid

        total_loss = mle_mean + effective_kl_weight * kl_mean

        return total_loss, {
            'mle': mle_mean.item(),
            'kl': kl_mean.item(),
            'kl_weight_effective': effective_kl_weight,
            'total': total_loss.item(),
        }

    def _kl_divergence_dirichlet(
        self,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL divergence from Dirichlet(α) to Dirichlet(1, 1, ...).

        KL[Dir(α) || Dir(1)] = log Γ(S) - sum log Γ(α_k) + sum (α_k - 1)(ψ(α_k) - ψ(S))
        where S = sum(α_k)
        """
        K = alpha.size(-1)
        ones = torch.ones_like(alpha)

        S_alpha = alpha.sum(dim=-1, keepdim=True)
        S_ones = ones.sum(dim=-1, keepdim=True)

        # log Γ(S_α) - log Γ(S_1)
        term1 = torch.lgamma(S_alpha) - torch.lgamma(S_ones)

        # - sum log Γ(α_k) + sum log Γ(1)
        term2 = torch.lgamma(ones).sum(dim=-1, keepdim=True) - torch.lgamma(alpha).sum(dim=-1, keepdim=True)

        # sum (α_k - 1)(ψ(α_k) - ψ(S_α))
        digamma_alpha = torch.digamma(alpha)
        digamma_S = torch.digamma(S_alpha)
        term3 = ((alpha - 1) * (digamma_alpha - digamma_S)).sum(dim=-1, keepdim=True)

        kl = term1 + term2 + term3
        return kl.squeeze(-1)


class EvidentialLoss(nn.Module):
    """
    Combined evidential loss for uncertainty-aware memory model.

    Handles both binary classification (recall) and regression (latency, hesitation).
    """

    def __init__(
        self,
        recall_weight: float = 1.0,
        latency_weight: float = 0.5,
        hesitation_weight: float = 0.3,
        kl_weight: float = 0.1,
        calibration_weight: float = 0.05,
    ):
        """
        Initialize combined evidential loss.

        Args:
            recall_weight: Weight for recall classification loss
            latency_weight: Weight for latency regression loss
            hesitation_weight: Weight for hesitation regression loss
            kl_weight: Weight for KL regularization
            calibration_weight: Weight for calibration penalty
        """
        super().__init__()

        self.recall_weight = recall_weight
        self.latency_weight = latency_weight
        self.hesitation_weight = hesitation_weight

        # Individual loss functions
        self.recall_loss_fn = EvidentialClassificationLoss(kl_weight=kl_weight)
        self.latency_loss_fn = EvidentialRegressionLoss(
            kl_weight=kl_weight,
            calibration_weight=calibration_weight,
        )
        self.hesitation_loss_fn = EvidentialRegressionLoss(
            kl_weight=kl_weight,
            calibration_weight=calibration_weight,
        )

    def set_epoch(self, epoch: int):
        """Update epoch for KL annealing."""
        self.recall_loss_fn.set_epoch(epoch)

    def forward(
        self,
        predictions: Dict[str, Dict[str, torch.Tensor]],
        targets: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined evidential loss.

        Args:
            predictions: Dict with 'recall', 'latency', 'hesitation' outputs
                Each containing evidential parameters
            targets: Dict with 'recall', 'latency', 'hesitation' ground truth
            mask: Optional mask for valid entries

        Returns:
            (total_loss, breakdown_dict)
        """
        breakdown = {}

        # Recall loss (classification)
        if 'recall' in predictions and 'recall' in targets:
            recall_loss, recall_breakdown = self.recall_loss_fn(
                predictions['recall'],
                targets['recall'],
                mask,
            )
            breakdown['recall_loss'] = recall_breakdown['total']
            breakdown['recall_kl'] = recall_breakdown['kl']
        else:
            recall_loss = torch.tensor(0.0)

        # Latency loss (regression)
        if 'latency' in predictions and 'latency' in targets:
            latency_loss, latency_breakdown = self.latency_loss_fn(
                predictions['latency'],
                targets['latency'],
                mask,
            )
            breakdown['latency_loss'] = latency_breakdown['total']
            breakdown['latency_nll'] = latency_breakdown['nll']
            breakdown['latency_calib'] = latency_breakdown['calibration']
        else:
            latency_loss = torch.tensor(0.0)

        # Hesitation loss (regression)
        if 'hesitation' in predictions and 'hesitation' in targets:
            hesitation_loss, hesitation_breakdown = self.hesitation_loss_fn(
                predictions['hesitation'],
                targets['hesitation'],
                mask,
            )
            breakdown['hesitation_loss'] = hesitation_breakdown['total']
        else:
            hesitation_loss = torch.tensor(0.0)

        # Weighted total
        total_loss = (
            self.recall_weight * recall_loss +
            self.latency_weight * latency_loss +
            self.hesitation_weight * hesitation_loss
        )

        breakdown['total'] = total_loss.item()

        return total_loss, breakdown
