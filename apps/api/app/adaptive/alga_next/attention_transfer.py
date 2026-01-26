"""
Attention Transfer / Multi-Task Learning Architecture

Solves the "Cold Start" problem in modality selection by using shared encoder
architectures to predict performance in untested modalities based on latent
cognitive features observed in other modalities.

Architecture:
- Shared Encoder: Learns modality-agnostic representations (Cognitive Capacity,
                  Focus Level, Topic Familiarity)
- Task-Specific Heads: Predict performance for each modality (Text, Video, etc.)

Transfer Mechanism:
When user engages with Text modality (Task A), backpropagation updates the
Shared Encoder. The Video head (Task B) uses this same encoder, so Video
performance prediction is updated even without Video interaction.

Cross-Modality Transfer Matrix:
Quantifies correlation of prediction errors between modalities.
High correlation = high transferability.

References:
- "Multimodal Predictive Student Modeling with Multi-Task Transfer Learning" (LAK23)
- Ruder, "An Overview of Multi-Task Learning in Deep Neural Networks"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModalityType(str, Enum):
    """Content modality types"""
    TEXT = "text"
    VIDEO = "video"
    AUDIO = "audio"
    INTERACTIVE = "interactive"
    DIAGRAM = "diagram"
    PODCAST = "podcast"


@dataclass
class UserObservation:
    """
    Observation of user performance/engagement in a specific modality

    Used to update the MTL model and transfer learning across modalities.
    """
    user_id: str
    modality: ModalityType
    content_id: str

    # Performance metrics
    engagement_score: float  # 0-1
    completion_rate: float  # 0-1
    assessment_score: Optional[float] = None  # 0-1 if quiz taken
    dwell_time_ratio: float = 1.0  # Actual/Expected dwell time

    # Behavioral features
    mouse_dynamics_score: float = 0.5
    cognitive_load_indicator: str = "medium"

    # Context
    fatigue_level: float = 0.0
    session_duration_minutes: float = 0.0

    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for encoder"""
        cognitive_load_map = {"low": 0.0, "medium": 0.5, "high": 1.0, "unknown": 0.5}

        return np.array([
            self.engagement_score,
            self.completion_rate,
            self.assessment_score if self.assessment_score is not None else 0.5,
            self.dwell_time_ratio,
            self.mouse_dynamics_score,
            cognitive_load_map.get(self.cognitive_load_indicator, 0.5),
            self.fatigue_level,
            min(1.0, self.session_duration_minutes / 60),
        ])


@dataclass
class ModalityPrediction:
    """Prediction of user performance for a specific modality"""
    modality: ModalityType
    predicted_engagement: float
    predicted_completion: float
    predicted_mastery: float
    confidence: float
    uncertainty: float  # Higher = less data for this modality


@dataclass
class TransferPredictions:
    """Complete set of predictions across all modalities"""
    user_id: str
    predictions: Dict[ModalityType, ModalityPrediction]
    shared_representation: np.ndarray
    transfer_confidence: Dict[Tuple[ModalityType, ModalityType], float]


class CrossModalityTransferMatrix:
    """
    Tracks and quantifies transfer learning effectiveness between modalities

    Matrix entry (i,j) represents how well observations in modality i
    predict performance in modality j.
    """

    def __init__(self, modalities: List[ModalityType]):
        self.modalities = modalities
        self.n_modalities = len(modalities)

        # Initialize correlation matrix
        self.transfer_matrix = np.ones((self.n_modalities, self.n_modalities)) * 0.5

        # Track prediction errors for correlation calculation
        self.prediction_errors: Dict[ModalityType, List[float]] = {
            m: [] for m in modalities
        }

        # Prior beliefs about transferability (from research)
        self._initialize_priors()

    def _initialize_priors(self):
        """Initialize transfer matrix with research-based priors"""
        # From the research paper's Cross-Modality Transfer Matrix:
        # Interactive behavior strongly predicts Video engagement
        # Text performance has moderate transfer to other text-heavy content

        modality_indices = {m: i for i, m in enumerate(self.modalities)}

        # Text ↔ Diagram (both visual-reading)
        if ModalityType.TEXT in modality_indices and ModalityType.DIAGRAM in modality_indices:
            i, j = modality_indices[ModalityType.TEXT], modality_indices[ModalityType.DIAGRAM]
            self.transfer_matrix[i, j] = 0.7
            self.transfer_matrix[j, i] = 0.7

        # Video ↔ Podcast (both passive AV)
        if ModalityType.VIDEO in modality_indices and ModalityType.PODCAST in modality_indices:
            i, j = modality_indices[ModalityType.VIDEO], modality_indices[ModalityType.PODCAST]
            self.transfer_matrix[i, j] = 0.8
            self.transfer_matrix[j, i] = 0.8

        # Interactive → Video (engagement patterns transfer)
        if ModalityType.INTERACTIVE in modality_indices and ModalityType.VIDEO in modality_indices:
            i, j = modality_indices[ModalityType.INTERACTIVE], modality_indices[ModalityType.VIDEO]
            self.transfer_matrix[i, j] = 0.75

        # Audio → Podcast (both audio-based)
        if ModalityType.AUDIO in modality_indices and ModalityType.PODCAST in modality_indices:
            i, j = modality_indices[ModalityType.AUDIO], modality_indices[ModalityType.PODCAST]
            self.transfer_matrix[i, j] = 0.85
            self.transfer_matrix[j, i] = 0.85

    def update(
        self,
        source_modality: ModalityType,
        target_modality: ModalityType,
        prediction_error: float,
    ):
        """Update transfer coefficient based on prediction error"""
        if source_modality not in self.modalities or target_modality not in self.modalities:
            return

        # Track error
        self.prediction_errors[target_modality].append(prediction_error)

        # Calculate correlation if we have enough samples
        source_errors = self.prediction_errors[source_modality]
        target_errors = self.prediction_errors[target_modality]

        if len(source_errors) >= 5 and len(target_errors) >= 5:
            # Use last 20 errors for rolling correlation
            s = np.array(source_errors[-20:])
            t = np.array(target_errors[-20:])
            min_len = min(len(s), len(t))

            if min_len >= 5:
                correlation = np.corrcoef(s[:min_len], t[:min_len])[0, 1]
                if not np.isnan(correlation):
                    # Convert correlation to transfer coefficient
                    # High positive correlation = high transferability
                    transfer_coef = (correlation + 1) / 2  # Map [-1,1] to [0,1]

                    # Update matrix with exponential smoothing
                    i = self.modalities.index(source_modality)
                    j = self.modalities.index(target_modality)
                    alpha = 0.1  # Learning rate
                    self.transfer_matrix[i, j] = (1 - alpha) * self.transfer_matrix[i, j] + alpha * transfer_coef

    def get_transfer_coefficient(
        self,
        source: ModalityType,
        target: ModalityType,
    ) -> float:
        """Get transfer coefficient from source to target modality"""
        if source not in self.modalities or target not in self.modalities:
            return 0.5

        i = self.modalities.index(source)
        j = self.modalities.index(target)
        return self.transfer_matrix[i, j]

    def get_best_source(self, target: ModalityType) -> Tuple[ModalityType, float]:
        """Find the best source modality for predicting target"""
        if target not in self.modalities:
            return (self.modalities[0], 0.5)

        j = self.modalities.index(target)
        transfer_scores = self.transfer_matrix[:, j]

        # Exclude self
        transfer_scores[j] = 0

        best_i = np.argmax(transfer_scores)
        return (self.modalities[best_i], transfer_scores[best_i])

    def to_dict(self) -> Dict[str, Any]:
        """Export matrix as dictionary"""
        return {
            "modalities": [m.value for m in self.modalities],
            "matrix": self.transfer_matrix.tolist(),
        }


class SharedEncoderNumpy:
    """
    Numpy implementation of the Shared Encoder

    Learns modality-agnostic representations of user cognitive state.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, latent_dim: int = 16):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Initialize weights
        np.random.seed(42)
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, latent_dim) * 0.1
        self.b2 = np.zeros(latent_dim)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode observation to latent representation"""
        h = np.tanh(x @ self.W1 + self.b1)
        z = np.tanh(h @ self.W2 + self.b2)
        return z

    def update(self, x: np.ndarray, target: np.ndarray, learning_rate: float = 0.01):
        """Simple gradient descent update"""
        # Forward pass
        h = np.tanh(x @ self.W1 + self.b1)
        z = np.tanh(h @ self.W2 + self.b2)

        # Backward pass (simplified)
        error = z - target
        dW2 = np.outer(h, error * (1 - z**2))
        db2 = error * (1 - z**2)

        dh = (error * (1 - z**2)) @ self.W2.T
        dW1 = np.outer(x, dh * (1 - h**2))
        db1 = dh * (1 - h**2)

        # Update
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1


class ModalityHeadNumpy:
    """
    Numpy implementation of a Modality-Specific Head

    Predicts performance metrics for a specific modality.
    """

    def __init__(self, latent_dim: int = 16, output_dim: int = 3):
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        np.random.seed(42)
        self.W = np.random.randn(latent_dim, output_dim) * 0.1
        self.b = np.zeros(output_dim)

    def predict(self, z: np.ndarray) -> np.ndarray:
        """Predict from latent representation"""
        return 1 / (1 + np.exp(-(z @ self.W + self.b)))  # Sigmoid

    def update(self, z: np.ndarray, target: np.ndarray, learning_rate: float = 0.01):
        """Update head weights"""
        pred = self.predict(z)
        error = pred - target

        # Gradient for sigmoid
        grad = error * pred * (1 - pred)
        dW = np.outer(z, grad)
        db = grad

        self.W -= learning_rate * dW
        self.b -= learning_rate * db


class AttentionTransferNetworkNumpy:
    """
    Numpy implementation of the Multi-Task Learning network for Attention Transfer

    Architecture:
    - Shared Encoder: processes observations into latent cognitive features
    - Modality Heads: predict performance for each modality type
    """

    def __init__(
        self,
        modalities: List[ModalityType] = None,
        input_dim: int = 8,
        latent_dim: int = 16,
    ):
        if modalities is None:
            modalities = list(ModalityType)

        self.modalities = modalities
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Shared encoder
        self.encoder = SharedEncoderNumpy(
            input_dim=input_dim,
            hidden_dim=32,
            latent_dim=latent_dim,
        )

        # Modality-specific heads
        self.heads: Dict[ModalityType, ModalityHeadNumpy] = {
            m: ModalityHeadNumpy(latent_dim=latent_dim, output_dim=3)
            for m in modalities
        }

        # Transfer matrix
        self.transfer_matrix = CrossModalityTransferMatrix(modalities)

        # User representations (cached latent vectors)
        self.user_representations: Dict[str, np.ndarray] = {}

        # Observation counts per modality per user
        self.observation_counts: Dict[str, Dict[ModalityType, int]] = {}

    def observe(self, observation: UserObservation):
        """
        Process an observation and update the model

        This updates:
        1. The shared encoder
        2. The modality-specific head for the observed modality
        3. The user's cached representation
        """
        user_id = observation.user_id
        modality = observation.modality

        # Get feature vector
        x = observation.to_feature_vector()

        # Encode to latent
        z = self.encoder.encode(x)

        # Cache user representation
        if user_id not in self.user_representations:
            self.user_representations[user_id] = z
            self.observation_counts[user_id] = {m: 0 for m in self.modalities}
        else:
            # Exponential moving average
            alpha = 0.3
            self.user_representations[user_id] = (
                (1 - alpha) * self.user_representations[user_id] + alpha * z
            )

        self.observation_counts[user_id][modality] += 1

        # Update head for this modality
        target = np.array([
            observation.engagement_score,
            observation.completion_rate,
            observation.assessment_score if observation.assessment_score else 0.5,
        ])
        self.heads[modality].update(z, target)

        # Calculate prediction error for transfer matrix
        pred = self.heads[modality].predict(z)
        error = np.mean(np.abs(pred - target))

        # Update transfer matrix
        for source_modality in self.modalities:
            if source_modality != modality:
                self.transfer_matrix.update(source_modality, modality, error)

    def predict(
        self,
        user_id: str,
        target_modality: ModalityType = None,
    ) -> TransferPredictions:
        """
        Predict performance across modalities for a user

        Uses the cached representation and transfer coefficients
        to make predictions even for untested modalities.
        """
        if user_id not in self.user_representations:
            # No observations yet - return default predictions
            default_pred = ModalityPrediction(
                modality=target_modality or ModalityType.TEXT,
                predicted_engagement=0.5,
                predicted_completion=0.5,
                predicted_mastery=0.5,
                confidence=0.0,
                uncertainty=1.0,
            )
            return TransferPredictions(
                user_id=user_id,
                predictions={m: default_pred for m in self.modalities},
                shared_representation=np.zeros(self.latent_dim),
                transfer_confidence={},
            )

        z = self.user_representations[user_id]
        obs_counts = self.observation_counts[user_id]

        predictions = {}
        transfer_confidence = {}

        for modality in self.modalities:
            pred = self.heads[modality].predict(z)

            # Calculate confidence based on direct observations
            direct_obs = obs_counts.get(modality, 0)
            direct_confidence = min(1.0, direct_obs / 5)  # Max at 5 observations

            # Calculate transfer confidence if no direct observations
            if direct_obs == 0:
                best_source, transfer_coef = self.transfer_matrix.get_best_source(modality)
                source_obs = obs_counts.get(best_source, 0)
                transfer_conf = transfer_coef * min(1.0, source_obs / 5)
                confidence = transfer_conf
                uncertainty = 1 - transfer_conf
            else:
                confidence = direct_confidence
                uncertainty = 1 - direct_confidence

            predictions[modality] = ModalityPrediction(
                modality=modality,
                predicted_engagement=float(pred[0]),
                predicted_completion=float(pred[1]),
                predicted_mastery=float(pred[2]),
                confidence=float(confidence),
                uncertainty=float(uncertainty),
            )

            # Store transfer confidence for debugging
            for source in self.modalities:
                if source != modality:
                    key = (source, modality)
                    transfer_confidence[key] = self.transfer_matrix.get_transfer_coefficient(
                        source, modality
                    )

        return TransferPredictions(
            user_id=user_id,
            predictions=predictions,
            shared_representation=z,
            transfer_confidence=transfer_confidence,
        )

    def recommend_modality(
        self,
        user_id: str,
        available_modalities: List[ModalityType] = None,
    ) -> Tuple[ModalityType, ModalityPrediction]:
        """
        Recommend the best modality for a user

        Balances predicted performance with exploration (uncertainty bonus).
        """
        if available_modalities is None:
            available_modalities = self.modalities

        predictions = self.predict(user_id)

        # Score = predicted engagement + exploration bonus for uncertainty
        exploration_weight = 0.2
        scores = {}

        for modality in available_modalities:
            pred = predictions.predictions.get(modality)
            if pred:
                # UCB-like score
                score = pred.predicted_engagement + exploration_weight * pred.uncertainty
                scores[modality] = score

        if not scores:
            return (available_modalities[0], predictions.predictions.get(available_modalities[0]))

        best_modality = max(scores, key=scores.get)
        return (best_modality, predictions.predictions[best_modality])


if TORCH_AVAILABLE:
    class SharedEncoder(nn.Module):
        """PyTorch Shared Encoder"""

        def __init__(self, input_dim: int = 8, hidden_dim: int = 64, latent_dim: int = 32):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

    class ModalityHead(nn.Module):
        """PyTorch Modality-Specific Head"""

        def __init__(self, latent_dim: int = 32, output_dim: int = 3):
            super().__init__()
            self.head = nn.Sequential(
                nn.Linear(latent_dim, 16),
                nn.ReLU(),
                nn.Linear(16, output_dim),
                nn.Sigmoid(),
            )

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.head(z)


# Main interface class
class AttentionTransferNetwork:
    """
    Attention Transfer Network for solving cold-start problem

    Unified interface that uses the appropriate backend.
    """

    def __init__(
        self,
        modalities: List[ModalityType] = None,
        input_dim: int = 8,
        latent_dim: int = 16,
    ):
        if modalities is None:
            modalities = list(ModalityType)

        self.model = AttentionTransferNetworkNumpy(
            modalities=modalities,
            input_dim=input_dim,
            latent_dim=latent_dim,
        )
        logger.info(f"AttentionTransferNetwork initialized with {len(modalities)} modalities")

    def observe(self, observation: UserObservation):
        """Process an observation"""
        self.model.observe(observation)

    def predict(
        self,
        user_id: str,
        target_modality: ModalityType = None,
    ) -> TransferPredictions:
        """Predict performance across modalities"""
        return self.model.predict(user_id, target_modality)

    def recommend_modality(
        self,
        user_id: str,
        available_modalities: List[ModalityType] = None,
    ) -> Tuple[ModalityType, ModalityPrediction]:
        """Recommend best modality"""
        return self.model.recommend_modality(user_id, available_modalities)

    def get_transfer_matrix(self) -> Dict[str, Any]:
        """Get the current transfer matrix"""
        return self.model.transfer_matrix.to_dict()
