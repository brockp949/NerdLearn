"""
Deep Knowledge Tracing (DKT) Implementation
Uses LSTM/Transformer to model learner knowledge state over time

Reference: "Deep Knowledge Tracing" (Piech et al., 2015)
Enhanced with SAINT+ architecture for improved performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class DKTModel(nn.Module):
    """
    Deep Knowledge Tracing using LSTM

    Architecture:
    Input: (question_id, answer_correctness) pairs
    Hidden: LSTM layers modeling knowledge evolution
    Output: Probability of correctness for next question
    """

    def __init__(
        self,
        num_concepts: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(DKTModel, self).__init__()

        self.num_concepts = num_concepts
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Input is (concept_id, correctness) - we encode as 2*num_concepts
        # Index: concept_id for correct, concept_id + num_concepts for incorrect
        self.input_dim = 2 * num_concepts

        # Embedding layer for input interactions
        self.embedding = nn.Embedding(self.input_dim, embedding_dim)

        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layer: predict P(correct) for each concept
        self.fc = nn.Linear(hidden_dim, num_concepts)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        interactions: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass

        Args:
            interactions: (batch_size, seq_len) - encoded interactions
            hidden: Optional previous hidden state

        Returns:
            predictions: (batch_size, seq_len, num_concepts)
            hidden: Updated hidden state
        """
        # Embed interactions
        embedded = self.embedding(interactions)  # (batch, seq, embedding_dim)

        # LSTM forward
        lstm_out, hidden = self.lstm(embedded, hidden)  # (batch, seq, hidden_dim)

        # Dropout
        lstm_out = self.dropout(lstm_out)

        # Predict probabilities for each concept
        predictions = self.fc(lstm_out)  # (batch, seq, num_concepts)
        predictions = torch.sigmoid(predictions)

        return predictions, hidden

    def encode_interaction(self, concept_id: int, is_correct: bool) -> int:
        """
        Encode (concept, correctness) as single integer

        Encoding:
        - Correct answer to concept i: i
        - Incorrect answer to concept i: i + num_concepts
        """
        if is_correct:
            return concept_id
        else:
            return concept_id + self.num_concepts

    def predict_next(
        self,
        interaction_history: List[Tuple[int, bool]],
        target_concept: int
    ) -> float:
        """
        Predict probability of correct answer for target concept

        Args:
            interaction_history: List of (concept_id, is_correct) tuples
            target_concept: Concept ID to predict

        Returns:
            Probability of correct answer (0-1)
        """
        self.eval()
        with torch.no_grad():
            # Encode interactions
            encoded = [
                self.encode_interaction(c, correct)
                for c, correct in interaction_history
            ]
            interactions = torch.tensor([encoded], dtype=torch.long)

            # Forward pass
            predictions, _ = self.forward(interactions)

            # Get prediction for target concept at last timestep
            prob = predictions[0, -1, target_concept].item()

            return prob


class SAINTModel(nn.Module):
    """
    SAINT+ (Separated Self-Attention for Knowledge Tracing)

    Improvement over DKT:
    - Uses Transformer architecture instead of LSTM
    - Separates exercise and response embeddings
    - Better handles long sequences
    - State-of-the-art performance

    Reference: "SAINT+: Integrating Temporal Features for EdNet Correctness Prediction" (Shin et al., 2021)
    """

    def __init__(
        self,
        num_concepts: int,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super(SAINTModel, self).__init__()

        self.num_concepts = num_concepts
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        # Exercise (question) embeddings
        self.exercise_embedding = nn.Embedding(num_concepts, embedding_dim)

        # Response embeddings (correct/incorrect)
        self.response_embedding = nn.Embedding(2, embedding_dim)

        # Positional encoding
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim * 2,  # Exercise + Response
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.fc = nn.Linear(embedding_dim * 2, num_concepts)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        exercises: torch.Tensor,
        responses: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            exercises: (batch_size, seq_len) - concept IDs
            responses: (batch_size, seq_len) - correctness (0/1)
            attention_mask: Optional attention mask

        Returns:
            predictions: (batch_size, seq_len, num_concepts)
        """
        batch_size, seq_len = exercises.shape

        # Embed exercises and responses
        exercise_emb = self.exercise_embedding(exercises)  # (batch, seq, emb_dim)
        response_emb = self.response_embedding(responses)  # (batch, seq, emb_dim)

        # Position embeddings
        positions = torch.arange(seq_len, device=exercises.device).unsqueeze(0)
        position_emb = self.position_embedding(positions)  # (1, seq, emb_dim)

        # Combine: [exercise + position, response]
        exercise_emb = exercise_emb + position_emb
        combined = torch.cat([exercise_emb, response_emb], dim=-1)  # (batch, seq, 2*emb_dim)

        # Transformer
        if attention_mask is not None:
            # Create causal mask (can't attend to future)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=exercises.device),
                diagonal=1
            ).bool()
        else:
            causal_mask = None

        transformer_out = self.transformer(
            combined,
            mask=causal_mask
        )  # (batch, seq, 2*emb_dim)

        # Dropout and predict
        transformer_out = self.dropout(transformer_out)
        predictions = self.fc(transformer_out)  # (batch, seq, num_concepts)
        predictions = torch.sigmoid(predictions)

        return predictions


class KnowledgeStateTracker:
    """
    High-level wrapper for knowledge tracing models
    Manages model selection, training, and inference
    """

    def __init__(
        self,
        num_concepts: int,
        model_type: str = "saint",  # "dkt" or "saint"
        device: str = "cpu"
    ):
        self.num_concepts = num_concepts
        self.model_type = model_type
        self.device = torch.device(device)

        # Initialize model
        if model_type == "dkt":
            self.model = DKTModel(num_concepts=num_concepts)
        elif model_type == "saint":
            self.model = SAINTModel(num_concepts=num_concepts)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model.to(self.device)

    def predict_performance(
        self,
        exercise_history: List[int],
        response_history: List[bool],
        target_concept: int
    ) -> float:
        """
        Predict probability of success on target concept

        Args:
            exercise_history: List of concept IDs attempted
            response_history: List of correctness (True/False)
            target_concept: Concept to predict

        Returns:
            Probability of correct response (0-1)
        """
        self.model.eval()
        with torch.no_grad():
            if self.model_type == "dkt":
                # DKT prediction
                interaction_history = list(zip(exercise_history, response_history))
                prob = self.model.predict_next(interaction_history, target_concept)
            else:
                # SAINT prediction
                exercises = torch.tensor([exercise_history], dtype=torch.long, device=self.device)
                responses = torch.tensor(
                    [[int(r) for r in response_history]],
                    dtype=torch.long,
                    device=self.device
                )

                predictions = self.model(exercises, responses)
                prob = predictions[0, -1, target_concept].item()

            return prob

    def get_knowledge_state(
        self,
        exercise_history: List[int],
        response_history: List[bool]
    ) -> np.ndarray:
        """
        Get current knowledge state vector across all concepts

        Returns:
            Array of shape (num_concepts,) with predicted mastery for each concept
        """
        self.model.eval()
        with torch.no_grad():
            if self.model_type == "saint":
                exercises = torch.tensor([exercise_history], dtype=torch.long, device=self.device)
                responses = torch.tensor(
                    [[int(r) for r in response_history]],
                    dtype=torch.long,
                    device=self.device
                )

                predictions = self.model(exercises, responses)
                state = predictions[0, -1, :].cpu().numpy()  # (num_concepts,)
            else:
                # For DKT, predict all concepts
                state = np.zeros(self.num_concepts)
                for concept in range(self.num_concepts):
                    state[concept] = self.predict_performance(
                        exercise_history,
                        response_history,
                        concept
                    )

            return state

    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_concepts': self.num_concepts,
            'model_type': self.model_type
        }, path)

    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def train_dkt_model(
    model: nn.Module,
    train_data: List[Tuple[List[int], List[bool]]],
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = "cpu"
):
    """
    Train DKT model on interaction sequences

    Args:
        model: DKT or SAINT model
        train_data: List of (exercise_sequence, response_sequence) tuples
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    device = torch.device(device)
    model.to(device)

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for exercises, responses in train_data:
            if len(exercises) < 2:
                continue

            # Prepare data
            exercises_tensor = torch.tensor([exercises[:-1]], dtype=torch.long, device=device)
            responses_tensor = torch.tensor(
                [[int(r) for r in responses[:-1]]],
                dtype=torch.long,
                device=device
            )
            target_concept = exercises[-1]
            target_response = float(responses[-1])

            # Forward
            if isinstance(model, SAINTModel):
                predictions = model(exercises_tensor, responses_tensor)
            else:
                # DKT
                encoded = [
                    model.encode_interaction(e, r)
                    for e, r in zip(exercises[:-1], responses[:-1])
                ]
                interactions = torch.tensor([encoded], dtype=torch.long, device=device)
                predictions, _ = model(interactions)

            # Loss
            pred_prob = predictions[0, -1, target_concept]
            target = torch.tensor([target_response], device=device)
            loss = criterion(pred_prob.unsqueeze(0), target)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model
