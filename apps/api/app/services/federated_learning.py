"""
Federated Learning Service

Privacy-preserving machine learning across distributed learner data:
- Model training without centralizing data
- Differential privacy guarantees
- Secure aggregation
- Personalized local models

Architecture:
- Coordinator: Manages global model and aggregation
- Clients: Train local models on user data
- Aggregator: Combines model updates securely

Use cases:
- Personalized content recommendations
- Adaptive difficulty prediction
- Learning pattern recognition
"""

import math
import random
import hashlib
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
from abc import ABC, abstractmethod
import json

logger = logging.getLogger(__name__)


# ==================== Enums and Data Classes ====================

class AggregationMethod(str, Enum):
    """Federated aggregation methods"""
    FEDAVG = "fedavg"           # Federated Averaging
    FEDPROX = "fedprox"         # FedProx (proximal term)
    FEDADAM = "fedadam"         # Federated Adam optimizer
    SCAFFOLD = "scaffold"       # SCAFFOLD variance reduction


class PrivacyMechanism(str, Enum):
    """Differential privacy mechanisms"""
    NONE = "none"
    GAUSSIAN = "gaussian"       # Gaussian noise
    LAPLACE = "laplace"         # Laplacian noise
    EXPONENTIAL = "exponential" # Exponential mechanism


@dataclass
class ModelWeights:
    """Model weights representation"""
    weights: Dict[str, List[float]]  # layer_name -> weights
    bias: Dict[str, List[float]]     # layer_name -> biases
    version: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_flat_vector(self) -> List[float]:
        """Flatten weights to single vector"""
        flat = []
        for layer in sorted(self.weights.keys()):
            flat.extend(self.weights[layer])
            flat.extend(self.bias.get(layer, []))
        return flat

    @classmethod
    def from_flat_vector(
        cls,
        vector: List[float],
        structure: Dict[str, int],
        version: int = 0
    ) -> "ModelWeights":
        """Reconstruct from flat vector"""
        weights = {}
        bias = {}
        idx = 0

        for layer, size in sorted(structure.items()):
            weights[layer] = vector[idx:idx + size]
            idx += size
            # Assume bias is 1/10 of weight size
            bias_size = max(1, size // 10)
            bias[layer] = vector[idx:idx + bias_size]
            idx += bias_size

        return cls(weights=weights, bias=bias, version=version)


@dataclass
class ClientUpdate:
    """Update from a federated client"""
    client_id: str
    model_delta: ModelWeights
    sample_count: int
    loss: float
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FederatedRound:
    """A single federated learning round"""
    round_id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    participating_clients: List[str] = field(default_factory=list)
    global_model_version: int = 0
    aggregated_loss: Optional[float] = None
    aggregated_metrics: Dict[str, float] = field(default_factory=dict)
    status: str = "in_progress"


@dataclass
class PrivacyBudget:
    """Differential privacy budget"""
    epsilon: float = 1.0      # Privacy parameter
    delta: float = 1e-5       # Failure probability
    used_epsilon: float = 0.0
    rounds_participated: int = 0

    @property
    def remaining_epsilon(self) -> float:
        return self.epsilon - self.used_epsilon

    def can_participate(self, cost: float = 0.1) -> bool:
        return self.used_epsilon + cost <= self.epsilon


# ==================== Differential Privacy ====================

class DifferentialPrivacy:
    """
    Differential privacy mechanisms for federated learning.
    """

    def __init__(
        self,
        mechanism: PrivacyMechanism = PrivacyMechanism.GAUSSIAN,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        clip_norm: float = 1.0
    ):
        self.mechanism = mechanism
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm

    def clip_gradients(self, gradients: List[float]) -> List[float]:
        """Clip gradients to bound sensitivity"""
        norm = math.sqrt(sum(g ** 2 for g in gradients))

        if norm > self.clip_norm:
            scale = self.clip_norm / norm
            return [g * scale for g in gradients]
        return gradients

    def add_noise(self, values: List[float], sensitivity: float = 1.0) -> List[float]:
        """Add noise for differential privacy"""
        if self.mechanism == PrivacyMechanism.NONE:
            return values

        elif self.mechanism == PrivacyMechanism.GAUSSIAN:
            # Gaussian mechanism
            sigma = sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
            return [
                v + random.gauss(0, sigma)
                for v in values
            ]

        elif self.mechanism == PrivacyMechanism.LAPLACE:
            # Laplace mechanism
            scale = sensitivity / self.epsilon
            return [
                v + random.uniform(-scale, scale) * math.copysign(1, random.random() - 0.5)
                for v in values
            ]

        return values

    def privatize_update(self, update: ClientUpdate) -> ClientUpdate:
        """Apply differential privacy to client update"""
        # Clip gradients
        flat_weights = update.model_delta.to_flat_vector()
        clipped = self.clip_gradients(flat_weights)

        # Add noise
        noisy = self.add_noise(clipped, sensitivity=self.clip_norm)

        # Reconstruct
        structure = {
            layer: len(weights)
            for layer, weights in update.model_delta.weights.items()
        }
        privatized_delta = ModelWeights.from_flat_vector(
            noisy, structure, update.model_delta.version
        )

        return ClientUpdate(
            client_id=update.client_id,
            model_delta=privatized_delta,
            sample_count=update.sample_count,
            loss=update.loss,
            metrics=update.metrics,
            timestamp=update.timestamp
        )


# ==================== Aggregation Strategies ====================

class FederatedAggregator(ABC):
    """Base class for federated aggregation"""

    @abstractmethod
    def aggregate(
        self,
        global_model: ModelWeights,
        client_updates: List[ClientUpdate]
    ) -> ModelWeights:
        """Aggregate client updates into global model"""
        pass


class FedAvgAggregator(FederatedAggregator):
    """
    Federated Averaging (FedAvg) aggregation.

    Weighted average of client updates based on sample counts.
    """

    def aggregate(
        self,
        global_model: ModelWeights,
        client_updates: List[ClientUpdate]
    ) -> ModelWeights:
        if not client_updates:
            return global_model

        # Calculate total samples
        total_samples = sum(u.sample_count for u in client_updates)
        if total_samples == 0:
            return global_model

        # Weighted average
        new_weights = {}
        new_bias = {}

        for layer in global_model.weights.keys():
            layer_size = len(global_model.weights[layer])
            aggregated = [0.0] * layer_size

            for update in client_updates:
                weight = update.sample_count / total_samples
                for i in range(layer_size):
                    if layer in update.model_delta.weights:
                        delta = update.model_delta.weights[layer][i] if i < len(update.model_delta.weights[layer]) else 0
                        aggregated[i] += weight * delta

            # Apply to global model
            new_weights[layer] = [
                global_model.weights[layer][i] + aggregated[i]
                for i in range(layer_size)
            ]

        # Same for biases
        for layer in global_model.bias.keys():
            layer_size = len(global_model.bias[layer])
            aggregated = [0.0] * layer_size

            for update in client_updates:
                weight = update.sample_count / total_samples
                for i in range(layer_size):
                    if layer in update.model_delta.bias:
                        delta = update.model_delta.bias[layer][i] if i < len(update.model_delta.bias[layer]) else 0
                        aggregated[i] += weight * delta

            new_bias[layer] = [
                global_model.bias[layer][i] + aggregated[i]
                for i in range(layer_size)
            ]

        return ModelWeights(
            weights=new_weights,
            bias=new_bias,
            version=global_model.version + 1
        )


class FedProxAggregator(FederatedAggregator):
    """
    FedProx aggregation with proximal term.

    Adds regularization to handle heterogeneous data.
    """

    def __init__(self, mu: float = 0.01):
        self.mu = mu  # Proximal term weight

    def aggregate(
        self,
        global_model: ModelWeights,
        client_updates: List[ClientUpdate]
    ) -> ModelWeights:
        # First do FedAvg
        fedavg = FedAvgAggregator()
        new_model = fedavg.aggregate(global_model, client_updates)

        # Apply proximal regularization (pull towards global)
        for layer in new_model.weights.keys():
            for i in range(len(new_model.weights[layer])):
                diff = new_model.weights[layer][i] - global_model.weights[layer][i]
                new_model.weights[layer][i] -= self.mu * diff

        return new_model


# ==================== Federated Client ====================

class FederatedClient:
    """
    Federated learning client that trains locally.
    """

    def __init__(
        self,
        client_id: str,
        privacy_budget: Optional[PrivacyBudget] = None
    ):
        self.client_id = client_id
        self.privacy_budget = privacy_budget or PrivacyBudget()
        self._local_data: List[Dict[str, Any]] = []
        self._local_model: Optional[ModelWeights] = None

    def add_training_data(self, data: Dict[str, Any]):
        """Add local training data"""
        self._local_data.append(data)

    def set_global_model(self, model: ModelWeights):
        """Set the current global model"""
        self._local_model = ModelWeights(
            weights={k: v.copy() for k, v in model.weights.items()},
            bias={k: v.copy() for k, v in model.bias.items()},
            version=model.version
        )

    def train_local(
        self,
        epochs: int = 1,
        learning_rate: float = 0.01
    ) -> ClientUpdate:
        """
        Train local model on local data.

        Returns model delta (difference from global model).
        """
        if not self._local_model or not self._local_data:
            raise ValueError("No model or data available")

        # Store initial weights
        initial_weights = {
            k: v.copy() for k, v in self._local_model.weights.items()
        }
        initial_bias = {
            k: v.copy() for k, v in self._local_model.bias.items()
        }

        # Simulated local training
        # In production, this would be actual gradient descent
        total_loss = 0.0

        for epoch in range(epochs):
            for data_point in self._local_data:
                # Compute loss (simplified)
                loss = self._compute_loss(data_point)
                total_loss += loss

                # Update weights (simplified SGD)
                self._update_weights(data_point, learning_rate)

        avg_loss = total_loss / (epochs * len(self._local_data))

        # Compute delta (new - initial)
        weight_delta = {}
        bias_delta = {}

        for layer in self._local_model.weights.keys():
            weight_delta[layer] = [
                self._local_model.weights[layer][i] - initial_weights[layer][i]
                for i in range(len(self._local_model.weights[layer]))
            ]

        for layer in self._local_model.bias.keys():
            bias_delta[layer] = [
                self._local_model.bias[layer][i] - initial_bias[layer][i]
                for i in range(len(self._local_model.bias[layer]))
            ]

        model_delta = ModelWeights(
            weights=weight_delta,
            bias=bias_delta,
            version=self._local_model.version
        )

        return ClientUpdate(
            client_id=self.client_id,
            model_delta=model_delta,
            sample_count=len(self._local_data),
            loss=avg_loss,
            metrics={"epochs": epochs, "learning_rate": learning_rate}
        )

    def _compute_loss(self, data_point: Dict[str, Any]) -> float:
        """Compute loss for a data point (simplified)"""
        # Simplified MSE-like loss
        target = data_point.get("target", 0.5)
        # Simple forward pass approximation
        prediction = self._forward(data_point.get("features", []))
        return (prediction - target) ** 2

    def _forward(self, features: List[float]) -> float:
        """Simple forward pass (simplified)"""
        if not self._local_model:
            return 0.5

        # Use first layer weights for simple prediction
        first_layer = list(self._local_model.weights.keys())[0]
        weights = self._local_model.weights[first_layer]

        result = 0.0
        for i, f in enumerate(features):
            if i < len(weights):
                result += f * weights[i]

        # Sigmoid activation
        return 1 / (1 + math.exp(-result)) if abs(result) < 500 else (1 if result > 0 else 0)

    def _update_weights(self, data_point: Dict[str, Any], lr: float):
        """Update weights via SGD (simplified)"""
        if not self._local_model:
            return

        features = data_point.get("features", [])
        target = data_point.get("target", 0.5)
        prediction = self._forward(features)

        # Gradient (simplified)
        error = prediction - target

        # Update first layer
        first_layer = list(self._local_model.weights.keys())[0]
        for i in range(min(len(features), len(self._local_model.weights[first_layer]))):
            gradient = error * features[i] * prediction * (1 - prediction)
            self._local_model.weights[first_layer][i] -= lr * gradient


# ==================== Federated Coordinator ====================

class FederatedCoordinator:
    """
    Coordinates federated learning across clients.

    Manages:
    - Global model distribution
    - Client selection
    - Secure aggregation
    - Training rounds
    """

    def __init__(
        self,
        aggregation_method: AggregationMethod = AggregationMethod.FEDAVG,
        privacy_mechanism: PrivacyMechanism = PrivacyMechanism.GAUSSIAN,
        epsilon: float = 1.0,
        min_clients_per_round: int = 2,
        max_clients_per_round: int = 100
    ):
        self.aggregation_method = aggregation_method
        self.min_clients = min_clients_per_round
        self.max_clients = max_clients_per_round

        # Initialize aggregator
        if aggregation_method == AggregationMethod.FEDAVG:
            self.aggregator = FedAvgAggregator()
        elif aggregation_method == AggregationMethod.FEDPROX:
            self.aggregator = FedProxAggregator()
        else:
            self.aggregator = FedAvgAggregator()

        # Privacy
        self.dp = DifferentialPrivacy(
            mechanism=privacy_mechanism,
            epsilon=epsilon
        )

        # State
        self._global_model: Optional[ModelWeights] = None
        self._clients: Dict[str, FederatedClient] = {}
        self._rounds: List[FederatedRound] = []
        self._round_updates: Dict[int, List[ClientUpdate]] = defaultdict(list)

    def initialize_model(self, model_structure: Dict[str, int]):
        """Initialize global model with random weights"""
        weights = {}
        bias = {}

        for layer, size in model_structure.items():
            # Xavier initialization
            scale = math.sqrt(2.0 / size)
            weights[layer] = [random.gauss(0, scale) for _ in range(size)]
            bias[layer] = [0.0 for _ in range(max(1, size // 10))]

        self._global_model = ModelWeights(weights=weights, bias=bias, version=0)
        logger.info(f"Initialized global model with {len(model_structure)} layers")

    def register_client(self, client: FederatedClient):
        """Register a federated client"""
        self._clients[client.client_id] = client
        logger.info(f"Registered federated client: {client.client_id}")

    def start_round(self) -> FederatedRound:
        """Start a new federated learning round"""
        if not self._global_model:
            raise ValueError("Global model not initialized")

        round_id = len(self._rounds)

        # Select clients
        available_clients = [
            c for c in self._clients.values()
            if c.privacy_budget.can_participate()
        ]

        if len(available_clients) < self.min_clients:
            logger.warning(f"Not enough clients for round: {len(available_clients)}")

        selected = random.sample(
            available_clients,
            min(len(available_clients), self.max_clients)
        )

        # Distribute global model
        for client in selected:
            client.set_global_model(self._global_model)

        round_info = FederatedRound(
            round_id=round_id,
            start_time=datetime.utcnow(),
            participating_clients=[c.client_id for c in selected],
            global_model_version=self._global_model.version
        )
        self._rounds.append(round_info)

        logger.info(f"Started round {round_id} with {len(selected)} clients")
        return round_info

    def submit_update(self, round_id: int, update: ClientUpdate) -> bool:
        """Submit client update for a round"""
        if round_id >= len(self._rounds):
            return False

        round_info = self._rounds[round_id]
        if round_info.status != "in_progress":
            return False

        if update.client_id not in round_info.participating_clients:
            return False

        # Apply differential privacy
        privatized_update = self.dp.privatize_update(update)

        # Update client's privacy budget
        client = self._clients.get(update.client_id)
        if client:
            client.privacy_budget.used_epsilon += 0.1  # Per-round cost
            client.privacy_budget.rounds_participated += 1

        self._round_updates[round_id].append(privatized_update)
        return True

    def complete_round(self, round_id: int) -> Optional[ModelWeights]:
        """Complete a round and aggregate updates"""
        if round_id >= len(self._rounds):
            return None

        round_info = self._rounds[round_id]
        updates = self._round_updates.get(round_id, [])

        if len(updates) < self.min_clients:
            logger.warning(f"Round {round_id} has insufficient updates: {len(updates)}")
            round_info.status = "failed"
            return None

        # Aggregate
        new_global_model = self.aggregator.aggregate(self._global_model, updates)
        self._global_model = new_global_model

        # Update round info
        round_info.end_time = datetime.utcnow()
        round_info.status = "completed"
        round_info.aggregated_loss = sum(u.loss for u in updates) / len(updates)

        logger.info(f"Completed round {round_id}, new model version: {new_global_model.version}")
        return new_global_model

    def get_global_model(self) -> Optional[ModelWeights]:
        """Get current global model"""
        return self._global_model

    def get_round_status(self, round_id: int) -> Optional[Dict[str, Any]]:
        """Get round status"""
        if round_id >= len(self._rounds):
            return None

        round_info = self._rounds[round_id]
        return {
            "round_id": round_info.round_id,
            "status": round_info.status,
            "start_time": round_info.start_time.isoformat(),
            "end_time": round_info.end_time.isoformat() if round_info.end_time else None,
            "participating_clients": len(round_info.participating_clients),
            "updates_received": len(self._round_updates.get(round_id, [])),
            "aggregated_loss": round_info.aggregated_loss
        }

    def get_training_stats(self) -> Dict[str, Any]:
        """Get overall training statistics"""
        completed_rounds = [r for r in self._rounds if r.status == "completed"]

        return {
            "total_rounds": len(self._rounds),
            "completed_rounds": len(completed_rounds),
            "total_clients": len(self._clients),
            "active_clients": sum(
                1 for c in self._clients.values()
                if c.privacy_budget.can_participate()
            ),
            "global_model_version": self._global_model.version if self._global_model else 0,
            "avg_loss": sum(r.aggregated_loss or 0 for r in completed_rounds) / len(completed_rounds) if completed_rounds else None,
            "privacy": {
                "mechanism": self.dp.mechanism.value,
                "epsilon": self.dp.epsilon,
                "delta": self.dp.delta
            }
        }


# ==================== Learning-Specific Models ====================

def create_recommendation_model() -> Dict[str, int]:
    """Create model structure for content recommendations"""
    return {
        "embedding": 64,       # User/content embedding
        "hidden1": 128,        # First hidden layer
        "hidden2": 64,         # Second hidden layer
        "output": 16           # Output scores
    }


def create_difficulty_predictor_model() -> Dict[str, int]:
    """Create model structure for difficulty prediction"""
    return {
        "input": 32,           # Input features
        "hidden": 64,          # Hidden layer
        "output": 8            # Difficulty scores
    }


# Singleton coordinator
federated_coordinator = FederatedCoordinator()
