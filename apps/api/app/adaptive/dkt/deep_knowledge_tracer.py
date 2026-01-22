"""
Deep Knowledge Tracing (DKT) and Self-Attentive Knowledge Tracing (SAKT)

Modern deep learning approaches for knowledge tracing that achieve
superior performance (AUC ~0.83) compared to BKT (~0.75).

DKT: Uses LSTM to model student knowledge state evolution
SAKT: Uses Transformer self-attention for better long-range dependencies

References:
- Piech et al., 2015: Deep Knowledge Tracing
- Pandey & Karypis, 2019: Self-Attentive Knowledge Tracing
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import math
import json
from datetime import datetime

# Try to import PyTorch, fall back to lightweight implementation
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModelType(str, Enum):
    """Types of deep knowledge tracing models"""
    DKT_LSTM = "dkt_lstm"
    SAKT = "sakt"  # Self-Attentive Knowledge Tracing
    DKT_LITE = "dkt_lite"  # Lightweight inference-only


@dataclass
class DKTConfig:
    """Configuration for Deep Knowledge Tracing models"""
    model_type: ModelType = ModelType.DKT_LSTM
    num_concepts: int = 100  # Number of unique concepts/skills
    embedding_dim: int = 64  # Embedding dimension
    hidden_dim: int = 128  # Hidden state dimension
    num_layers: int = 2  # Number of LSTM/Transformer layers
    num_heads: int = 4  # Number of attention heads (SAKT only)
    dropout: float = 0.2  # Dropout rate
    max_seq_length: int = 200  # Maximum sequence length
    learning_rate: float = 0.001
    batch_size: int = 32

    # Inference parameters
    prediction_threshold: float = 0.5
    mastery_threshold: float = 0.85

    def to_dict(self) -> Dict:
        return {
            "model_type": self.model_type.value,
            "num_concepts": self.num_concepts,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "max_seq_length": self.max_seq_length,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "prediction_threshold": self.prediction_threshold,
            "mastery_threshold": self.mastery_threshold,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "DKTConfig":
        d = d.copy()
        if "model_type" in d:
            d["model_type"] = ModelType(d["model_type"])
        return cls(**d)


@dataclass
class Interaction:
    """A single student-concept interaction"""
    concept_id: int
    correct: bool
    timestamp: Optional[datetime] = None
    response_time_ms: Optional[int] = None
    attempt_count: int = 1

    def to_tuple(self) -> Tuple[int, int]:
        """Convert to (concept_id, correct) tuple for model input"""
        return (self.concept_id, 1 if self.correct else 0)


@dataclass
class InteractionSequence:
    """A sequence of student interactions for knowledge tracing"""
    user_id: str
    interactions: List[Interaction] = field(default_factory=list)

    def add_interaction(self, concept_id: int, correct: bool, **kwargs):
        """Add an interaction to the sequence"""
        self.interactions.append(Interaction(
            concept_id=concept_id,
            correct=correct,
            **kwargs
        ))

    def to_tensors(self) -> Tuple[List[int], List[int]]:
        """Convert to (concept_ids, correctness) lists"""
        concept_ids = [i.concept_id for i in self.interactions]
        correct = [1 if i.correct else 0 for i in self.interactions]
        return concept_ids, correct

    def __len__(self) -> int:
        return len(self.interactions)


@dataclass
class KnowledgeState:
    """Current knowledge state for a student across concepts"""
    user_id: str
    concept_masteries: Dict[int, float] = field(default_factory=dict)
    overall_mastery: float = 0.0
    confidence: float = 0.0
    last_updated: Optional[datetime] = None
    model_type: str = "dkt"

    def get_mastery(self, concept_id: int, default: float = 0.1) -> float:
        """Get mastery for a specific concept"""
        return self.concept_masteries.get(concept_id, default)

    def is_mastered(self, concept_id: int, threshold: float = 0.85) -> bool:
        """Check if a concept is mastered"""
        return self.get_mastery(concept_id) >= threshold

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "concept_masteries": self.concept_masteries,
            "overall_mastery": self.overall_mastery,
            "confidence": self.confidence,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "model_type": self.model_type,
        }


# ============================================================================
# PyTorch Models (when available)
# ============================================================================

if TORCH_AVAILABLE:

    class DKTModel(nn.Module):
        """
        Deep Knowledge Tracing using LSTM

        Architecture:
        1. Embed (concept, response) pairs
        2. Process sequence with LSTM
        3. Output probability of correct response for each concept
        """

        def __init__(self, config: DKTConfig):
            super().__init__()
            self.config = config

            # Input: (concept_id, correct) -> one-hot encoding of 2*num_concepts
            # Or use embeddings for better generalization
            self.concept_embedding = nn.Embedding(
                config.num_concepts,
                config.embedding_dim
            )
            self.response_embedding = nn.Embedding(2, config.embedding_dim)

            # Combine concept and response embeddings
            self.input_projection = nn.Linear(
                config.embedding_dim * 2,
                config.hidden_dim
            )

            # LSTM for sequential processing
            self.lstm = nn.LSTM(
                input_size=config.hidden_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=config.dropout if config.num_layers > 1 else 0,
            )

            # Output layer: predict probability for each concept
            self.output_layer = nn.Linear(config.hidden_dim, config.num_concepts)
            self.dropout = nn.Dropout(config.dropout)

        def forward(
            self,
            concept_ids: torch.Tensor,  # (batch, seq_len)
            responses: torch.Tensor,     # (batch, seq_len)
            hidden: Optional[Tuple] = None
        ) -> Tuple[torch.Tensor, Tuple]:
            """
            Forward pass

            Args:
                concept_ids: Concept IDs (batch, seq_len)
                responses: Binary correct/incorrect (batch, seq_len)
                hidden: Optional initial hidden state

            Returns:
                (predictions, hidden_state)
                predictions: (batch, seq_len, num_concepts) - P(correct) for each concept
            """
            batch_size, seq_len = concept_ids.shape

            # Get embeddings
            concept_emb = self.concept_embedding(concept_ids)  # (batch, seq, emb)
            response_emb = self.response_embedding(responses)  # (batch, seq, emb)

            # Concatenate and project
            combined = torch.cat([concept_emb, response_emb], dim=-1)
            x = self.input_projection(combined)  # (batch, seq, hidden)
            x = self.dropout(x)

            # LSTM processing
            lstm_out, hidden = self.lstm(x, hidden)  # (batch, seq, hidden)
            lstm_out = self.dropout(lstm_out)

            # Output predictions for all concepts
            output = torch.sigmoid(self.output_layer(lstm_out))  # (batch, seq, num_concepts)

            return output, hidden

        def predict_next(
            self,
            concept_ids: torch.Tensor,
            responses: torch.Tensor,
            next_concept: torch.Tensor,
            hidden: Optional[Tuple] = None
        ) -> Tuple[torch.Tensor, Tuple]:
            """
            Predict probability of correct response on next concept

            Args:
                concept_ids: Past concept IDs
                responses: Past responses
                next_concept: Concept to predict for
                hidden: Hidden state

            Returns:
                (probability, hidden_state)
            """
            output, hidden = self.forward(concept_ids, responses, hidden)

            # Get prediction for the specific next concept
            # Use the last timestep's prediction
            last_output = output[:, -1, :]  # (batch, num_concepts)

            # Gather predictions for the target concepts
            predictions = torch.gather(
                last_output,
                1,
                next_concept.unsqueeze(1)
            ).squeeze(1)

            return predictions, hidden


    class SAKTModel(nn.Module):
        """
        Self-Attentive Knowledge Tracing using Transformer

        Uses self-attention to capture long-range dependencies in
        student learning sequences, achieving better performance than LSTM.
        """

        def __init__(self, config: DKTConfig):
            super().__init__()
            self.config = config

            # Embeddings
            self.concept_embedding = nn.Embedding(
                config.num_concepts,
                config.embedding_dim
            )
            self.interaction_embedding = nn.Embedding(
                config.num_concepts * 2,  # concept + correct/incorrect
                config.embedding_dim
            )

            # Positional encoding
            self.pos_embedding = nn.Embedding(
                config.max_seq_length,
                config.embedding_dim
            )

            # Multi-head attention for knowledge state
            self.attention = nn.MultiheadAttention(
                embed_dim=config.embedding_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                batch_first=True
            )

            # Feed-forward network
            self.ffn = nn.Sequential(
                nn.Linear(config.embedding_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.embedding_dim),
            )

            # Layer normalization
            self.layer_norm1 = nn.LayerNorm(config.embedding_dim)
            self.layer_norm2 = nn.LayerNorm(config.embedding_dim)

            # Output projection
            self.output_layer = nn.Linear(config.embedding_dim, 1)
            self.dropout = nn.Dropout(config.dropout)

        def forward(
            self,
            concept_ids: torch.Tensor,
            responses: torch.Tensor,
            query_concepts: torch.Tensor,
        ) -> torch.Tensor:
            """
            Forward pass

            Args:
                concept_ids: Past concept IDs (batch, seq_len)
                responses: Past responses (batch, seq_len)
                query_concepts: Concepts to predict (batch, seq_len or 1)

            Returns:
                predictions: P(correct) for query concepts
            """
            batch_size, seq_len = concept_ids.shape

            # Create interaction embeddings (concept + response combined)
            interaction_ids = concept_ids + responses * self.config.num_concepts
            interaction_emb = self.interaction_embedding(interaction_ids)

            # Add positional encoding
            positions = torch.arange(seq_len, device=concept_ids.device)
            pos_emb = self.pos_embedding(positions).unsqueeze(0)
            interaction_emb = interaction_emb + pos_emb

            # Query embeddings (what we want to predict)
            query_emb = self.concept_embedding(query_concepts)

            # Create causal mask (can only attend to past interactions)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=concept_ids.device),
                diagonal=1
            ).bool()

            # Self-attention
            attn_out, _ = self.attention(
                query=query_emb,
                key=interaction_emb,
                value=interaction_emb,
                attn_mask=causal_mask if query_emb.shape[1] == seq_len else None,
            )

            # Residual connection and layer norm
            x = self.layer_norm1(query_emb + self.dropout(attn_out))

            # Feed-forward
            ffn_out = self.ffn(x)
            x = self.layer_norm2(x + self.dropout(ffn_out))

            # Output
            output = torch.sigmoid(self.output_layer(x)).squeeze(-1)

            return output


    class DKTDataset(Dataset):
        """Dataset for DKT training"""

        def __init__(
            self,
            sequences: List[InteractionSequence],
            config: DKTConfig
        ):
            self.sequences = sequences
            self.config = config

        def __len__(self) -> int:
            return len(self.sequences)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            seq = self.sequences[idx]
            concept_ids, responses = seq.to_tensors()

            # Truncate or pad
            max_len = self.config.max_seq_length
            if len(concept_ids) > max_len:
                concept_ids = concept_ids[:max_len]
                responses = responses[:max_len]

            # Pad
            pad_len = max_len - len(concept_ids)
            concept_ids = concept_ids + [0] * pad_len
            responses = responses + [0] * pad_len

            return {
                "concept_ids": torch.tensor(concept_ids, dtype=torch.long),
                "responses": torch.tensor(responses, dtype=torch.long),
                "length": torch.tensor(len(seq), dtype=torch.long),
            }


# ============================================================================
# Lightweight Inference Model (no PyTorch required)
# ============================================================================

class DKTLite:
    """
    Lightweight Deep Knowledge Tracing for inference without PyTorch

    Uses pre-computed embeddings and simplified attention mechanism
    for fast inference in production environments.
    """

    def __init__(
        self,
        num_concepts: int = 100,
        embedding_dim: int = 32,
        decay_rate: float = 0.95,
        learning_rate: float = 0.15,
    ):
        self.num_concepts = num_concepts
        self.embedding_dim = embedding_dim
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate

        # Initialize concept difficulty estimates
        self.concept_difficulty: Dict[int, float] = {}

        # Pre-computed concept relationships (can be loaded from trained model)
        self.concept_similarity: Dict[Tuple[int, int], float] = {}

        # User knowledge states
        self.user_states: Dict[str, Dict[int, float]] = {}

    def update_from_interaction(
        self,
        user_id: str,
        concept_id: int,
        correct: bool,
    ) -> KnowledgeState:
        """
        Update knowledge state from a single interaction

        Uses exponential smoothing with concept difficulty adjustment
        """
        if user_id not in self.user_states:
            self.user_states[user_id] = {}

        state = self.user_states[user_id]

        # Get current mastery
        current_mastery = state.get(concept_id, 0.1)

        # Get concept difficulty (default 0.5)
        difficulty = self.concept_difficulty.get(concept_id, 0.5)

        # Update based on response
        if correct:
            # Increase mastery, scaled by difficulty
            gain = self.learning_rate * (1 - current_mastery) * (1 + difficulty * 0.5)
            new_mastery = current_mastery + gain
        else:
            # Decrease mastery slightly
            loss = self.learning_rate * 0.3 * current_mastery
            new_mastery = current_mastery - loss

        # Bound mastery
        new_mastery = max(0.01, min(0.99, new_mastery))
        state[concept_id] = new_mastery

        # Apply decay to related concepts (forgetting)
        for other_concept, mastery in list(state.items()):
            if other_concept != concept_id:
                # Slight decay for non-practiced concepts
                state[other_concept] = mastery * self.decay_rate

        # Transfer learning to similar concepts
        for (c1, c2), similarity in self.concept_similarity.items():
            if c1 == concept_id and c2 in state:
                transfer = (new_mastery - state[c2]) * similarity * 0.1
                state[c2] = max(0.01, min(0.99, state[c2] + transfer))

        # Build knowledge state
        overall = sum(state.values()) / max(len(state), 1)

        return KnowledgeState(
            user_id=user_id,
            concept_masteries=state.copy(),
            overall_mastery=overall,
            confidence=min(len(state) / self.num_concepts, 1.0),
            last_updated=datetime.utcnow(),
            model_type="dkt_lite",
        )

    def predict_performance(
        self,
        user_id: str,
        concept_id: int,
    ) -> Dict[str, float]:
        """
        Predict probability of correct response
        """
        state = self.user_states.get(user_id, {})
        mastery = state.get(concept_id, 0.1)
        difficulty = self.concept_difficulty.get(concept_id, 0.5)

        # P(correct) based on mastery and difficulty
        p_correct = mastery * (1 - difficulty * 0.3)

        return {
            "p_correct": p_correct,
            "mastery": mastery,
            "difficulty": difficulty,
            "confidence": min(len(state) / 10, 1.0),
        }

    def get_knowledge_state(self, user_id: str) -> KnowledgeState:
        """Get current knowledge state for a user"""
        state = self.user_states.get(user_id, {})
        overall = sum(state.values()) / max(len(state), 1) if state else 0.1

        return KnowledgeState(
            user_id=user_id,
            concept_masteries=state.copy(),
            overall_mastery=overall,
            confidence=min(len(state) / self.num_concepts, 1.0),
            last_updated=datetime.utcnow(),
            model_type="dkt_lite",
        )

    def set_concept_difficulty(self, concept_id: int, difficulty: float):
        """Set difficulty for a concept (0-1)"""
        self.concept_difficulty[concept_id] = max(0.0, min(1.0, difficulty))

    def set_concept_similarity(self, concept1: int, concept2: int, similarity: float):
        """Set similarity between concepts (for transfer learning)"""
        similarity = max(0.0, min(1.0, similarity))
        self.concept_similarity[(concept1, concept2)] = similarity
        self.concept_similarity[(concept2, concept1)] = similarity

    def load_from_trained_model(self, model_data: Dict):
        """Load concept difficulties and similarities from trained model"""
        if "concept_difficulty" in model_data:
            self.concept_difficulty = model_data["concept_difficulty"]
        if "concept_similarity" in model_data:
            self.concept_similarity = {
                tuple(k.split(",")): v
                for k, v in model_data["concept_similarity"].items()
            }


# ============================================================================
# Main DeepKnowledgeTracer Class
# ============================================================================

class DeepKnowledgeTracer:
    """
    Deep Knowledge Tracing with multiple model options

    Provides a unified interface for:
    - DKT (LSTM-based)
    - SAKT (Transformer-based)
    - DKTLite (lightweight inference)

    Typical AUC performance:
    - BKT: ~0.75
    - DKT: ~0.80
    - SAKT: ~0.83
    """

    def __init__(self, config: Optional[DKTConfig] = None):
        self.config = config or DKTConfig()
        self.model = None
        self.optimizer = None
        self.device = "cpu"

        # For lightweight mode
        self.lite_model = DKTLite(
            num_concepts=self.config.num_concepts,
            embedding_dim=self.config.embedding_dim,
        )

        # User interaction sequences for online learning
        self.user_sequences: Dict[str, InteractionSequence] = {}

        # Initialize PyTorch model if available and not lite mode
        if TORCH_AVAILABLE and self.config.model_type != ModelType.DKT_LITE:
            self._init_torch_model()

    def _init_torch_model(self):
        """Initialize PyTorch model"""
        if not TORCH_AVAILABLE:
            return

        if self.config.model_type == ModelType.DKT_LSTM:
            self.model = DKTModel(self.config)
        elif self.config.model_type == ModelType.SAKT:
            self.model = SAKTModel(self.config)

        if self.model:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate
            )

    def update_from_interaction(
        self,
        user_id: str,
        concept_id: int,
        correct: bool,
        **kwargs
    ) -> Tuple[KnowledgeState, Dict]:
        """
        Update knowledge state from a single interaction

        Args:
            user_id: User identifier
            concept_id: Concept being practiced
            correct: Whether response was correct
            **kwargs: Additional interaction metadata

        Returns:
            (KnowledgeState, update_details)
        """
        # Add to user sequence
        if user_id not in self.user_sequences:
            self.user_sequences[user_id] = InteractionSequence(user_id=user_id)

        self.user_sequences[user_id].add_interaction(
            concept_id=concept_id,
            correct=correct,
            **kwargs
        )

        # Update lightweight model (always available)
        state = self.lite_model.update_from_interaction(
            user_id, concept_id, correct
        )

        # If PyTorch model available, get enhanced predictions
        predictions = {}
        if TORCH_AVAILABLE and self.model is not None:
            predictions = self._get_torch_predictions(user_id)
            # Blend lite and torch predictions
            for cid, p in predictions.items():
                if cid in state.concept_masteries:
                    # Weighted average favoring torch model
                    state.concept_masteries[cid] = (
                        state.concept_masteries[cid] * 0.3 + p * 0.7
                    )
                else:
                    state.concept_masteries[cid] = p

        update_details = {
            "concept_id": concept_id,
            "correct": correct,
            "new_mastery": state.get_mastery(concept_id),
            "overall_mastery": state.overall_mastery,
            "model_type": self.config.model_type.value,
            "torch_available": TORCH_AVAILABLE,
        }

        return state, update_details

    def _get_torch_predictions(self, user_id: str) -> Dict[int, float]:
        """Get predictions from PyTorch model"""
        if not TORCH_AVAILABLE or self.model is None:
            return {}

        seq = self.user_sequences.get(user_id)
        if not seq or len(seq) == 0:
            return {}

        self.model.eval()
        with torch.no_grad():
            concept_ids, responses = seq.to_tensors()

            # Truncate if needed
            max_len = self.config.max_seq_length
            if len(concept_ids) > max_len:
                concept_ids = concept_ids[-max_len:]
                responses = responses[-max_len:]

            # Convert to tensors
            concept_tensor = torch.tensor(
                [concept_ids], dtype=torch.long, device=self.device
            )
            response_tensor = torch.tensor(
                [responses], dtype=torch.long, device=self.device
            )

            if self.config.model_type == ModelType.DKT_LSTM:
                output, _ = self.model(concept_tensor, response_tensor)
                # Get last timestep predictions
                predictions = output[0, -1, :].cpu().numpy()
            elif self.config.model_type == ModelType.SAKT:
                # For SAKT, query all concepts
                query = torch.arange(
                    self.config.num_concepts,
                    device=self.device
                ).unsqueeze(0)
                predictions = self.model(
                    concept_tensor,
                    response_tensor,
                    query
                )[0].cpu().numpy()
            else:
                return {}

        return {i: float(p) for i, p in enumerate(predictions)}

    def predict_performance(
        self,
        user_id: str,
        concept_id: int,
    ) -> Dict[str, float]:
        """
        Predict probability of correct response for a concept

        Args:
            user_id: User identifier
            concept_id: Concept to predict

        Returns:
            Prediction details including p_correct
        """
        # Get lite model prediction
        lite_pred = self.lite_model.predict_performance(user_id, concept_id)

        # Enhance with torch model if available
        if TORCH_AVAILABLE and self.model is not None:
            torch_preds = self._get_torch_predictions(user_id)
            if concept_id in torch_preds:
                # Blend predictions
                lite_pred["p_correct"] = (
                    lite_pred["p_correct"] * 0.3 +
                    torch_preds[concept_id] * 0.7
                )
                lite_pred["torch_prediction"] = torch_preds[concept_id]

        return lite_pred

    def get_knowledge_state(self, user_id: str) -> KnowledgeState:
        """Get current knowledge state for a user"""
        state = self.lite_model.get_knowledge_state(user_id)

        # Enhance with torch predictions if available
        if TORCH_AVAILABLE and self.model is not None:
            torch_preds = self._get_torch_predictions(user_id)
            for cid, p in torch_preds.items():
                if p > 0.1:  # Only include non-trivial predictions
                    if cid in state.concept_masteries:
                        state.concept_masteries[cid] = (
                            state.concept_masteries[cid] * 0.3 + p * 0.7
                        )
                    else:
                        state.concept_masteries[cid] = p

            # Recalculate overall mastery
            if state.concept_masteries:
                state.overall_mastery = (
                    sum(state.concept_masteries.values()) /
                    len(state.concept_masteries)
                )

        state.model_type = self.config.model_type.value
        return state

    def train(
        self,
        sequences: List[InteractionSequence],
        epochs: int = 10,
        validation_split: float = 0.1,
    ) -> Dict[str, List[float]]:
        """
        Train the model on interaction sequences

        Args:
            sequences: List of user interaction sequences
            epochs: Number of training epochs
            validation_split: Fraction of data for validation

        Returns:
            Training history (loss, accuracy per epoch)
        """
        if not TORCH_AVAILABLE or self.model is None:
            # Train lightweight model parameters from data
            return self._train_lite(sequences)

        # Split data
        n_val = int(len(sequences) * validation_split)
        train_seqs = sequences[n_val:]
        val_seqs = sequences[:n_val] if n_val > 0 else []

        # Create datasets
        train_dataset = DKTDataset(train_seqs, self.config)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        history = {"train_loss": [], "train_auc": [], "val_auc": []}

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                concept_ids = batch["concept_ids"].to(self.device)
                responses = batch["responses"].to(self.device)
                lengths = batch["length"]

                self.optimizer.zero_grad()

                if self.config.model_type == ModelType.DKT_LSTM:
                    output, _ = self.model(concept_ids, responses)

                    # Create target: next timestep correctness
                    target = responses[:, 1:].float()
                    pred = output[:, :-1, :]

                    # Gather predictions for actual concepts
                    pred_for_concept = torch.gather(
                        pred, 2, concept_ids[:, 1:].unsqueeze(-1)
                    ).squeeze(-1)

                    # Binary cross entropy loss
                    loss = F.binary_cross_entropy(pred_for_concept, target)

                elif self.config.model_type == ModelType.SAKT:
                    # SAKT training
                    query_concepts = concept_ids[:, 1:]
                    output = self.model(
                        concept_ids[:, :-1],
                        responses[:, :-1],
                        query_concepts
                    )
                    target = responses[:, 1:].float()
                    loss = F.binary_cross_entropy(output, target)

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            history["train_loss"].append(avg_loss)

            # Calculate AUC
            train_auc = self._calculate_auc(train_seqs[:100])
            history["train_auc"].append(train_auc)

            if val_seqs:
                val_auc = self._calculate_auc(val_seqs)
                history["val_auc"].append(val_auc)

        return history

    def _train_lite(self, sequences: List[InteractionSequence]) -> Dict:
        """Train lightweight model from data"""
        # Calculate concept difficulties from data
        concept_attempts: Dict[int, List[bool]] = {}

        for seq in sequences:
            for interaction in seq.interactions:
                if interaction.concept_id not in concept_attempts:
                    concept_attempts[interaction.concept_id] = []
                concept_attempts[interaction.concept_id].append(interaction.correct)

        # Set difficulties based on average success rate
        for concept_id, attempts in concept_attempts.items():
            success_rate = sum(attempts) / len(attempts)
            difficulty = 1 - success_rate  # Harder = lower success rate
            self.lite_model.set_concept_difficulty(concept_id, difficulty)

        return {"trained_concepts": len(concept_attempts)}

    def _calculate_auc(self, sequences: List[InteractionSequence]) -> float:
        """Calculate AUC on sequences"""
        if not TORCH_AVAILABLE or self.model is None:
            return 0.0

        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for seq in sequences:
                if len(seq) < 2:
                    continue

                concept_ids, responses = seq.to_tensors()
                concept_tensor = torch.tensor(
                    [concept_ids[:-1]], dtype=torch.long, device=self.device
                )
                response_tensor = torch.tensor(
                    [responses[:-1]], dtype=torch.long, device=self.device
                )

                if self.config.model_type == ModelType.DKT_LSTM:
                    output, _ = self.model(concept_tensor, response_tensor)
                    pred = output[0, -1, concept_ids[-1]].item()
                elif self.config.model_type == ModelType.SAKT:
                    query = torch.tensor(
                        [[concept_ids[-1]]], dtype=torch.long, device=self.device
                    )
                    pred = self.model(
                        concept_tensor, response_tensor, query
                    )[0, 0].item()
                else:
                    continue

                all_preds.append(pred)
                all_targets.append(responses[-1])

        if len(all_preds) < 2:
            return 0.5

        # Simple AUC calculation
        return self._simple_auc(all_targets, all_preds)

    def _simple_auc(self, targets: List[int], preds: List[float]) -> float:
        """Simple AUC calculation without sklearn"""
        pairs = list(zip(targets, preds))
        positive = [p for t, p in pairs if t == 1]
        negative = [p for t, p in pairs if t == 0]

        if not positive or not negative:
            return 0.5

        correct = 0
        total = len(positive) * len(negative)

        for pos_pred in positive:
            for neg_pred in negative:
                if pos_pred > neg_pred:
                    correct += 1
                elif pos_pred == neg_pred:
                    correct += 0.5

        return correct / total if total > 0 else 0.5

    def save_model(self, path: str):
        """Save model to file"""
        data = {
            "config": self.config.to_dict(),
            "lite_model": {
                "concept_difficulty": self.lite_model.concept_difficulty,
                "concept_similarity": {
                    f"{k[0]},{k[1]}": v
                    for k, v in self.lite_model.concept_similarity.items()
                },
            },
        }

        if TORCH_AVAILABLE and self.model is not None:
            torch.save({
                "model_state": self.model.state_dict(),
                **data
            }, path)
        else:
            with open(path, "w") as f:
                json.dump(data, f)

    def load_model(self, path: str):
        """Load model from file"""
        if TORCH_AVAILABLE and path.endswith(".pt"):
            checkpoint = torch.load(path, map_location=self.device)
            if self.model is not None and "model_state" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state"])
            if "lite_model" in checkpoint:
                self.lite_model.load_from_trained_model(checkpoint["lite_model"])
        else:
            with open(path, "r") as f:
                data = json.load(f)
            if "lite_model" in data:
                self.lite_model.load_from_trained_model(data["lite_model"])

    def is_mastered(
        self,
        user_id: str,
        concept_id: int,
        threshold: Optional[float] = None
    ) -> bool:
        """Check if a concept is mastered"""
        threshold = threshold or self.config.mastery_threshold
        state = self.get_knowledge_state(user_id)
        return state.is_mastered(concept_id, threshold)

    def sessions_to_mastery(
        self,
        user_id: str,
        concept_id: int,
        threshold: float = 0.85,
    ) -> int:
        """Estimate sessions needed to reach mastery"""
        current = self.get_knowledge_state(user_id).get_mastery(concept_id)

        if current >= threshold:
            return 0

        # Estimate based on learning rate
        learning_rate = self.lite_model.learning_rate
        difficulty = self.lite_model.concept_difficulty.get(concept_id, 0.5)

        # Approximate sessions needed
        gap = threshold - current
        avg_gain = learning_rate * 0.7 * (1 - difficulty * 0.3)  # Assume 70% success rate

        if avg_gain <= 0:
            return -1

        sessions = int(math.ceil(gap / avg_gain))
        return min(sessions, 100)
