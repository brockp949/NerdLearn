"""
Neural ODE Trainer for memory model training.

Provides training loop, validation, checkpointing, and metrics tracking.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import logging
from datetime import datetime
import numpy as np
from sklearn.metrics import roc_auc_score

from .model import NeuralMemoryODE
from .loss import ImplicitTelemetryLoss
from .dataset import ReviewSequenceDataset, collate_review_sequences

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for Neural ODE training."""
    # Data
    batch_size: int = 32
    validation_split: float = 0.15
    min_reviews_per_sequence: int = 5
    max_sequence_length: int = 100

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 50
    gradient_clip_norm: float = 1.0

    # Scheduler
    lr_scheduler: str = "cosine"  # cosine, plateau, step
    warmup_epochs: int = 3

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"  # min or max

    # Checkpointing
    checkpoint_dir: str = "./models/neural_ode"
    save_best_only: bool = True
    save_every_n_epochs: int = 5

    # Logging
    log_interval: int = 100  # Log every N batches
    tensorboard_dir: Optional[str] = None

    # Model
    state_dim: int = 32
    hidden_dim: int = 64
    card_feat_dim: int = 64
    user_feat_dim: int = 16

    # Loss weights
    recall_weight: float = 1.0
    latency_weight: float = 0.5
    hesitation_weight: float = 0.3


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_recall_loss: float = 0.0
    train_latency_loss: float = 0.0
    val_recall_loss: float = 0.0
    val_latency_loss: float = 0.0
    val_auc: float = 0.0
    val_calibration_error: float = 0.0
    val_retention_accuracy: float = 0.0
    learning_rate: float = 0.0
    best_val_loss: float = float('inf')
    epochs_without_improvement: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_recall_loss": self.train_recall_loss,
            "train_latency_loss": self.train_latency_loss,
            "val_recall_loss": self.val_recall_loss,
            "val_latency_loss": self.val_latency_loss,
            "val_auc": self.val_auc,
            "val_calibration_error": self.val_calibration_error,
            "val_retention_accuracy": self.val_retention_accuracy,
            "learning_rate": self.learning_rate,
        }


class NeuralODETrainer:
    """
    Trainer class for Neural ODE memory model.

    Handles training loop, validation, checkpointing, and metrics.
    """

    def __init__(
        self,
        model: NeuralMemoryODE,
        config: TrainingConfig,
        device: str = "cpu",
    ):
        """
        Initialize trainer.

        Args:
            model: NeuralMemoryODE model instance
            config: TrainingConfig with hyperparameters
            device: Device to train on ("cpu" or "cuda")
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Loss function
        self.loss_fn = ImplicitTelemetryLoss(
            recall_weight=config.recall_weight,
            latency_weight=config.latency_weight,
            hesitation_weight=config.hesitation_weight,
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Metrics
        self.metrics = TrainingMetrics()
        self.history: List[Dict[str, float]] = []

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if self.config.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.learning_rate * 0.01,
            )
        elif self.config.lr_scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                verbose=True,
            )
        elif self.config.lr_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5,
            )
        return None

    def train(
        self,
        dataset: ReviewSequenceDataset,
        val_dataset: Optional[ReviewSequenceDataset] = None,
    ) -> TrainingMetrics:
        """
        Run full training loop.

        Args:
            dataset: Training dataset
            val_dataset: Optional validation dataset (will split from train if not provided)

        Returns:
            Final training metrics
        """
        # Split dataset if no validation provided
        if val_dataset is None:
            val_size = int(len(dataset) * self.config.validation_split)
            train_size = len(dataset) - val_size
            dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create data loaders
        train_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_review_sequences,
            num_workers=0,  # Windows compatibility
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_review_sequences,
            num_workers=0,
        )

        logger.info(f"Training on {len(dataset)} sequences, validating on {len(val_dataset)}")

        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(1, self.config.num_epochs + 1):
            self.metrics.epoch = epoch

            # Training epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            self.metrics.train_loss = train_metrics['loss']
            self.metrics.train_recall_loss = train_metrics.get('recall_loss', 0)
            self.metrics.train_latency_loss = train_metrics.get('latency_loss', 0)

            # Validation
            val_metrics = self.validate(val_loader)
            self.metrics.val_loss = val_metrics['loss']
            self.metrics.val_recall_loss = val_metrics.get('recall_loss', 0)
            self.metrics.val_latency_loss = val_metrics.get('latency_loss', 0)
            self.metrics.val_auc = val_metrics.get('auc', 0)
            self.metrics.val_calibration_error = val_metrics.get('calibration_error', 0)
            self.metrics.val_retention_accuracy = val_metrics.get('retention_accuracy', 0)

            # Learning rate
            self.metrics.learning_rate = self.optimizer.param_groups[0]['lr']

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(self.metrics.val_loss)
                else:
                    self.scheduler.step()

            # Log metrics
            self.history.append(self.metrics.to_dict())
            logger.info(
                f"Epoch {epoch}/{self.config.num_epochs} - "
                f"train_loss: {self.metrics.train_loss:.4f}, "
                f"val_loss: {self.metrics.val_loss:.4f}, "
                f"val_auc: {self.metrics.val_auc:.4f}"
            )

            # Early stopping check
            current_metric = self.metrics.val_loss
            if current_metric < best_val_loss:
                best_val_loss = current_metric
                epochs_without_improvement = 0
                self.metrics.best_val_loss = best_val_loss

                # Save best model
                if self.config.save_best_only:
                    self.save_checkpoint("best_model.pt", epoch, self.metrics.to_dict())
            else:
                epochs_without_improvement += 1
                self.metrics.epochs_without_improvement = epochs_without_improvement

            # Early stopping
            if epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(f"Early stopping after {epoch} epochs")
                break

            # Periodic checkpoint
            if epoch % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt", epoch, self.metrics.to_dict())

        # Save final model
        self.save_checkpoint("final_model.pt", epoch, self.metrics.to_dict())

        # Save training history
        self._save_history()

        return self.metrics

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Run single training epoch.

        Args:
            dataloader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_recall_loss = 0.0
        total_latency_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward pass
            self.optimizer.zero_grad()

            # Process each sequence in batch
            loss, loss_breakdown = self._compute_batch_loss(batch)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )

            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_recall_loss += loss_breakdown.get('recall', 0)
            total_latency_loss += loss_breakdown.get('latency', 0)
            num_batches += 1

            # Log progress
            if batch_idx % self.config.log_interval == 0:
                logger.debug(
                    f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] - "
                    f"loss: {loss.item():.4f}"
                )

        return {
            'loss': total_loss / num_batches,
            'recall_loss': total_recall_loss / num_batches,
            'latency_loss': total_latency_loss / num_batches,
        }

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Run validation.

        Args:
            dataloader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_recall_loss = 0.0
        total_latency_loss = 0.0
        num_batches = 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                loss, loss_breakdown = self._compute_batch_loss(batch)

                total_loss += loss.item()
                total_recall_loss += loss_breakdown.get('recall', 0)
                total_latency_loss += loss_breakdown.get('latency', 0)
                num_batches += 1

                # Collect predictions for AUC
                preds, targets = self._get_predictions(batch)
                all_predictions.extend(preds)
                all_targets.extend(targets)

        # Compute validation metrics
        metrics = {
            'loss': total_loss / num_batches,
            'recall_loss': total_recall_loss / num_batches,
            'latency_loss': total_latency_loss / num_batches,
        }

        # AUC
        if len(all_predictions) > 0 and len(set(all_targets)) > 1:
            try:
                metrics['auc'] = roc_auc_score(all_targets, all_predictions)
            except ValueError:
                metrics['auc'] = 0.5

        # Calibration error
        metrics['calibration_error'] = self._compute_calibration_error(
            all_predictions, all_targets
        )

        # Retention accuracy at threshold
        metrics['retention_accuracy'] = self._compute_retention_accuracy(
            all_predictions, all_targets, threshold=0.9
        )

        return metrics

    def _compute_batch_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for a batch.

        Args:
            batch: Batched tensors from collate function

        Returns:
            (loss_tensor, loss_breakdown_dict)
        """
        card_features = batch['card_features']  # [B, 64]
        user_features = batch['user_features']  # [B, 16]
        times = batch['times']  # [B, max_len]
        grades = batch['grades']  # [B, max_len]
        telemetry = batch['telemetry']  # [B, max_len, 4]
        recalls = batch['recalls']  # [B, max_len]
        latencies = batch['latencies']  # [B, max_len]
        mask = batch['mask']  # [B, max_len]
        seq_lengths = batch['sequence_lengths']  # [B]

        batch_size = card_features.size(0)
        total_loss = torch.tensor(0.0, device=self.device)
        breakdown = {'recall': 0.0, 'latency': 0.0, 'hesitation': 0.0}

        for i in range(batch_size):
            seq_len = seq_lengths[i].item()
            if seq_len == 0:
                continue

            # Build review events for this sequence
            events = []
            for t in range(seq_len):
                events.append((
                    times[i, t].item(),
                    grades[i, t].item(),
                    telemetry[i, t],
                ))

            # Process through model
            result = self.model.process_review_sequence(
                card_features[i:i+1],
                events,
                user_features=user_features[i:i+1],
                return_trajectory=False,
            )

            # Get predictions
            pred_prob = result['final_prob']  # [1, 1]
            pred_latency = result['final_latency']  # [1, 1]
            pred_hesitation = result['final_hesitation']  # [1, 1]

            # Get targets (last event in sequence)
            target_recall = recalls[i, seq_len-1:seq_len].unsqueeze(-1)  # [1, 1]
            target_latency = latencies[i, seq_len-1:seq_len].unsqueeze(-1)  # [1, 1]

            # Compute losses
            recall_loss = nn.functional.binary_cross_entropy(
                pred_prob, target_recall
            )
            latency_loss = nn.functional.mse_loss(pred_latency, target_latency)
            hesitation_loss = nn.functional.mse_loss(
                pred_hesitation,
                telemetry[i, seq_len-1, 1:2].unsqueeze(0)  # Hesitation from telemetry
            )

            # Weighted sum
            seq_loss = (
                self.config.recall_weight * recall_loss +
                self.config.latency_weight * latency_loss +
                self.config.hesitation_weight * hesitation_loss
            )

            total_loss = total_loss + seq_loss
            breakdown['recall'] += recall_loss.item()
            breakdown['latency'] += latency_loss.item()
            breakdown['hesitation'] += hesitation_loss.item()

        # Average over batch
        total_loss = total_loss / batch_size
        breakdown = {k: v / batch_size for k, v in breakdown.items()}

        return total_loss, breakdown

    def _get_predictions(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[List[float], List[int]]:
        """Extract predictions and targets from batch."""
        predictions = []
        targets = []

        card_features = batch['card_features']
        user_features = batch['user_features']
        times = batch['times']
        grades = batch['grades']
        telemetry = batch['telemetry']
        recalls = batch['recalls']
        seq_lengths = batch['sequence_lengths']

        batch_size = card_features.size(0)

        for i in range(batch_size):
            seq_len = seq_lengths[i].item()
            if seq_len == 0:
                continue

            events = []
            for t in range(seq_len):
                events.append((
                    times[i, t].item(),
                    grades[i, t].item(),
                    telemetry[i, t],
                ))

            result = self.model.process_review_sequence(
                card_features[i:i+1],
                events,
                user_features=user_features[i:i+1],
                return_trajectory=False,
            )

            pred = result['final_prob'].item()
            target = int(recalls[i, seq_len-1].item())

            predictions.append(pred)
            targets.append(target)

        return predictions, targets

    def _compute_calibration_error(
        self,
        predictions: List[float],
        targets: List[int],
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error."""
        if not predictions:
            return 0.0

        predictions = np.array(predictions)
        targets = np.array(targets)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            in_bin = (predictions > bin_boundaries[i]) & (predictions <= bin_boundaries[i+1])
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                avg_confidence = predictions[in_bin].mean()
                avg_accuracy = targets[in_bin].mean()
                ece += prop_in_bin * abs(avg_accuracy - avg_confidence)

        return ece

    def _compute_retention_accuracy(
        self,
        predictions: List[float],
        targets: List[int],
        threshold: float = 0.9,
    ) -> float:
        """Compute accuracy for high-confidence predictions."""
        if not predictions:
            return 0.0

        predictions = np.array(predictions)
        targets = np.array(targets)

        high_conf = predictions >= threshold
        if high_conf.sum() == 0:
            return 0.0

        return targets[high_conf].mean()

    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        metrics: Dict[str, Any],
    ):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': {
                'state_dim': self.config.state_dim,
                'hidden_dim': self.config.hidden_dim,
                'card_feat_dim': self.config.card_feat_dim,
                'user_feat_dim': self.config.user_feat_dim,
            },
            'timestamp': datetime.now().isoformat(),
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint

    def _save_history(self):
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")
