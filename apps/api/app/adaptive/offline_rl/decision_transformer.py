"""
Decision Transformer for Curriculum Reinforcement Learning

A GPT-style Transformer that treats RL as sequence modeling.
Instead of learning value functions, it learns to predict actions
conditioned on desired return-to-go.

Architecture:
- Input: (R̂_t, s_t, a_t) triplets for each timestep
- Model: Causal Transformer with self-attention
- Output: Predicted action at each timestep

Key Innovation for Spacing:
The self-attention mechanism can look back at the entire history,
capturing long-term temporal dependencies like:
"If concept A was struggled with 20 steps ago, schedule it now"

This is superior to value-based methods that compress history into a state.

Training:
- Supervised learning on offline trajectories
- Cross-entropy loss for action prediction
- Condition on Return-to-Go to control quality of generated sequences

Inference:
- Prompt with high R̂ to get optimal actions
- Apply action mask before sampling

References:
- Chen et al. (2021): Decision Transformer
- Janner et al. (2021): Trajectory Transformer
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. DecisionTransformer will not work.")


@dataclass
class DecisionTransformerConfig:
    """Configuration for Decision Transformer"""

    # Model dimensions
    state_dim: int = 60              # Belief state dimension (3K for K concepts)
    action_dim: int = 20             # Number of concepts (action space)
    hidden_dim: int = 128            # Transformer hidden dimension
    num_layers: int = 4              # Number of transformer layers
    num_heads: int = 4               # Number of attention heads
    dropout: float = 0.1             # Dropout rate

    # Sequence parameters
    max_length: int = 50             # Maximum context length
    max_episode_length: int = 1000   # Maximum episode length (for positional encoding)

    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    warmup_steps: int = 1000
    max_steps: int = 100000

    # Inference parameters
    temperature: float = 0.1         # Sampling temperature (low = more deterministic)
    top_k: int = 5                   # Top-k sampling
    target_return: float = 10.0      # Default target return for inference

    def to_dict(self) -> Dict:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "max_length": self.max_length,
            "max_episode_length": self.max_episode_length,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "target_return": self.target_return,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "DecisionTransformerConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DecisionTransformerConfig":
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


if TORCH_AVAILABLE:

    class PositionalEncoding(nn.Module):
        """
        Positional encoding for the transformer.

        Uses learnable embeddings for timesteps (not sinusoidal)
        since educational sequences have meaningful temporal structure.
        """

        def __init__(self, hidden_dim: int, max_length: int, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            self.position_embedding = nn.Embedding(max_length, hidden_dim)

        def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
            """
            Add positional encoding to input.

            Args:
                x: Input tensor (batch, seq_len, hidden_dim)
                timesteps: Timestep indices (batch, seq_len)

            Returns:
                Input with positional encoding added
            """
            pos_emb = self.position_embedding(timesteps)
            return self.dropout(x + pos_emb)

    class TransformerBlock(nn.Module):
        """
        Single transformer block with causal self-attention.
        """

        def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
            super().__init__()

            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )

            self.feed_forward = nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Linear(4 * hidden_dim, hidden_dim),
                nn.Dropout(dropout),
            )

            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            causal_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Forward pass with causal self-attention.

            Args:
                x: Input tensor (batch, seq_len, hidden_dim)
                attention_mask: Padding mask (batch, seq_len)
                causal_mask: Causal attention mask (seq_len, seq_len)

            Returns:
                Output tensor (batch, seq_len, hidden_dim)
            """
            # Self-attention with residual
            normed = self.norm1(x)
            attended, _ = self.attention(
                normed, normed, normed,
                key_padding_mask=attention_mask,
                attn_mask=causal_mask,
                need_weights=False
            )
            x = x + self.dropout(attended)

            # Feed-forward with residual
            x = x + self.feed_forward(self.norm2(x))

            return x

    class DecisionTransformer(nn.Module):
        """
        Decision Transformer for curriculum sequencing.

        Input sequence format:
        [R̂_1, s_1, a_1, R̂_2, s_2, a_2, ..., R̂_T, s_T, a_T]

        Each triplet (R̂_t, s_t, a_t) represents:
        - R̂_t: Return-to-go (desired future reward)
        - s_t: Belief state
        - a_t: Action taken (concept practiced)

        The model predicts actions conditioned on desired returns.
        """

        def __init__(self, config: DecisionTransformerConfig):
            super().__init__()
            self.config = config

            # Input embeddings
            self.return_embedding = nn.Linear(1, config.hidden_dim)
            self.state_embedding = nn.Linear(config.state_dim, config.hidden_dim)
            self.action_embedding = nn.Embedding(config.action_dim, config.hidden_dim)

            # Positional encoding (for global timesteps)
            self.pos_encoding = PositionalEncoding(
                config.hidden_dim,
                config.max_episode_length,
                config.dropout
            )

            # Transformer blocks
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(config.hidden_dim, config.num_heads, config.dropout)
                for _ in range(config.num_layers)
            ])

            # Output heads
            self.action_head = nn.Linear(config.hidden_dim, config.action_dim)

            # Layer norm
            self.final_norm = nn.LayerNorm(config.hidden_dim)

            # Initialize weights
            self._init_weights()

        def _init_weights(self):
            """Initialize weights with small values"""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
            """
            Create causal attention mask.

            For the decision transformer, we use block-wise causality:
            - Within a timestep, R̂ → s → a (can only attend left)
            - Across timesteps, standard causal masking

            Returns:
                Upper triangular mask (seq_len, seq_len)
            """
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device),
                diagonal=1
            ).bool()
            return mask

        def forward(
            self,
            returns_to_go: torch.Tensor,      # (batch, seq_len, 1)
            states: torch.Tensor,              # (batch, seq_len, state_dim)
            actions: torch.Tensor,             # (batch, seq_len)
            timesteps: torch.Tensor,           # (batch, seq_len)
            attention_mask: Optional[torch.Tensor] = None  # (batch, seq_len)
        ) -> torch.Tensor:
            """
            Forward pass.

            Args:
                returns_to_go: Target returns (batch, seq_len, 1)
                states: Belief states (batch, seq_len, state_dim)
                actions: Actions taken (batch, seq_len)
                timesteps: Global timestep indices (batch, seq_len)
                attention_mask: Padding mask (1=valid, 0=pad)

            Returns:
                Action logits (batch, seq_len, action_dim)
            """
            batch_size, seq_len = states.shape[:2]
            device = states.device

            # Embed inputs
            rtg_emb = self.return_embedding(returns_to_go)      # (batch, seq, hidden)
            state_emb = self.state_embedding(states)            # (batch, seq, hidden)
            action_emb = self.action_embedding(actions)         # (batch, seq, hidden)

            # Interleave embeddings: [R̂_1, s_1, a_1, R̂_2, s_2, a_2, ...]
            # This creates a sequence of length 3 * seq_len
            stacked = torch.stack([rtg_emb, state_emb, action_emb], dim=2)
            # (batch, seq_len, 3, hidden) -> (batch, 3*seq_len, hidden)
            x = stacked.reshape(batch_size, 3 * seq_len, -1)

            # Expand timesteps for the interleaved sequence
            timesteps_expanded = timesteps.unsqueeze(2).repeat(1, 1, 3)
            timesteps_expanded = timesteps_expanded.reshape(batch_size, 3 * seq_len)

            # Add positional encoding
            x = self.pos_encoding(x, timesteps_expanded)

            # Create causal mask
            causal_mask = self._create_causal_mask(3 * seq_len, device)

            # Expand attention mask for interleaved sequence
            if attention_mask is not None:
                # Convert 1=valid to True=ignore for key_padding_mask
                expanded_mask = attention_mask.unsqueeze(2).repeat(1, 1, 3)
                expanded_mask = expanded_mask.reshape(batch_size, 3 * seq_len)
                padding_mask = (expanded_mask == 0)  # True where padding
            else:
                padding_mask = None

            # Apply transformer blocks
            for block in self.transformer_blocks:
                x = block(x, attention_mask=padding_mask, causal_mask=causal_mask)

            # Final layer norm
            x = self.final_norm(x)

            # Extract state positions for action prediction
            # States are at positions 1, 4, 7, ... (index 1 mod 3)
            state_positions = torch.arange(1, 3 * seq_len, 3, device=device)
            state_outputs = x[:, state_positions, :]  # (batch, seq_len, hidden)

            # Predict actions
            action_logits = self.action_head(state_outputs)  # (batch, seq_len, action_dim)

            return action_logits

        def get_action(
            self,
            returns_to_go: torch.Tensor,
            states: torch.Tensor,
            actions: torch.Tensor,
            timesteps: torch.Tensor,
            action_mask: Optional[torch.Tensor] = None,
            temperature: Optional[float] = None,
            deterministic: bool = False
        ) -> Tuple[int, torch.Tensor]:
            """
            Get action for the current timestep.

            Args:
                returns_to_go: Returns-to-go history (1, context_len, 1)
                states: State history (1, context_len, state_dim)
                actions: Action history (1, context_len)
                timesteps: Timestep history (1, context_len)
                action_mask: Valid action mask (action_dim,)
                temperature: Sampling temperature
                deterministic: If True, return argmax

            Returns:
                (selected_action, action_probabilities)
            """
            temperature = temperature or self.config.temperature

            # Forward pass
            with torch.no_grad():
                logits = self.forward(returns_to_go, states, actions, timesteps)

            # Get logits for the last timestep
            last_logits = logits[0, -1, :]  # (action_dim,)

            # Apply action mask
            if action_mask is not None:
                mask_tensor = torch.tensor(action_mask, device=last_logits.device)
                # Add large negative value to invalid actions
                last_logits = last_logits + (1 - mask_tensor) * (-1e9)

            if deterministic:
                action = int(torch.argmax(last_logits).item())
                probs = F.softmax(last_logits, dim=-1)
            else:
                # Apply temperature
                scaled_logits = last_logits / temperature

                # Optional top-k filtering
                if self.config.top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(
                        scaled_logits, self.config.top_k
                    )
                    probs = F.softmax(top_k_logits, dim=-1)
                    sampled_idx = torch.multinomial(probs, 1).item()
                    action = int(top_k_indices[sampled_idx].item())
                    probs = F.softmax(scaled_logits, dim=-1)
                else:
                    probs = F.softmax(scaled_logits, dim=-1)
                    action = int(torch.multinomial(probs, 1).item())

            return action, probs

        def compute_loss(
            self,
            returns_to_go: torch.Tensor,
            states: torch.Tensor,
            actions: torch.Tensor,
            timesteps: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Compute cross-entropy loss for action prediction.

            Args:
                returns_to_go: Target returns (batch, seq_len, 1)
                states: Belief states (batch, seq_len, state_dim)
                actions: Ground truth actions (batch, seq_len)
                timesteps: Timestep indices (batch, seq_len)
                attention_mask: Padding mask

            Returns:
                Scalar loss
            """
            # Forward pass
            logits = self.forward(returns_to_go, states, actions, timesteps, attention_mask)

            # Flatten for cross-entropy
            batch_size, seq_len, action_dim = logits.shape
            logits_flat = logits.reshape(-1, action_dim)
            targets_flat = actions.reshape(-1)

            # Compute loss
            loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
            loss = loss.reshape(batch_size, seq_len)

            # Apply attention mask
            if attention_mask is not None:
                loss = loss * attention_mask
                loss = loss.sum() / attention_mask.sum()
            else:
                loss = loss.mean()

            return loss

        def save(self, path: str):
            """Save model checkpoint"""
            checkpoint = {
                'config': self.config.to_dict(),
                'state_dict': self.state_dict(),
            }
            torch.save(checkpoint, path)
            logger.info(f"Saved model to {path}")

        @classmethod
        def load(cls, path: str, device: str = 'cpu') -> "DecisionTransformer":
            """Load model from checkpoint"""
            checkpoint = torch.load(path, map_location=device)
            config = DecisionTransformerConfig.from_dict(checkpoint['config'])
            model = cls(config)
            model.load_state_dict(checkpoint['state_dict'])
            model.to(device)
            logger.info(f"Loaded model from {path}")
            return model

        def export_weights(self, path: str):
            """
            Export weights to numpy format for DT-Lite.

            Saves weights in a format that can be loaded without PyTorch.
            """
            weights = {}
            for name, param in self.named_parameters():
                weights[name] = param.detach().cpu().numpy()

            np.savez(path, **weights)
            logger.info(f"Exported weights to {path}")

    class DecisionTransformerTrainer:
        """
        Trainer for Decision Transformer.

        Handles:
        - Training loop with gradient accumulation
        - Learning rate scheduling
        - Validation and logging
        - Checkpointing
        """

        def __init__(
            self,
            model: DecisionTransformer,
            config: DecisionTransformerConfig,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        ):
            self.model = model.to(device)
            self.config = config
            self.device = device

            # Optimizer
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )

            # Learning rate scheduler with warmup
            self.scheduler = self._create_scheduler()

            # Training state
            self.step = 0
            self.best_loss = float('inf')

        def _create_scheduler(self):
            """Create learning rate scheduler with linear warmup and cosine decay"""
            def lr_lambda(step):
                if step < self.config.warmup_steps:
                    return step / self.config.warmup_steps
                else:
                    progress = (step - self.config.warmup_steps) / (
                        self.config.max_steps - self.config.warmup_steps
                    )
                    return 0.5 * (1 + math.cos(math.pi * progress))

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        def train_step(
            self,
            returns_to_go: torch.Tensor,
            states: torch.Tensor,
            actions: torch.Tensor,
            timesteps: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
        ) -> float:
            """
            Single training step.

            Returns:
                Loss value
            """
            self.model.train()
            self.optimizer.zero_grad()

            # Move to device
            returns_to_go = returns_to_go.to(self.device)
            states = states.to(self.device)
            actions = actions.to(self.device)
            timesteps = timesteps.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # Compute loss
            loss = self.model.compute_loss(
                returns_to_go, states, actions, timesteps, attention_mask
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            self.step += 1
            return loss.item()

        def train(
            self,
            dataset,  # TrajectoryDataset
            num_steps: int,
            eval_interval: int = 1000,
            save_interval: int = 5000,
            save_path: str = "dt_checkpoint.pt"
        ) -> Dict[str, List[float]]:
            """
            Full training loop.

            Args:
                dataset: TrajectoryDataset with training data
                num_steps: Number of training steps
                eval_interval: Steps between evaluations
                save_interval: Steps between checkpoints
                save_path: Path for saving checkpoints

            Returns:
                Dictionary with training metrics
            """
            history = {"loss": [], "lr": []}

            for step in range(num_steps):
                # Get batch
                rtg, states, actions, timesteps, mask = dataset.get_trajectory_batch(
                    batch_size=self.config.batch_size,
                    max_length=self.config.max_length
                )

                # Convert to tensors
                rtg = torch.tensor(rtg, dtype=torch.float32)
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.long)
                timesteps = torch.tensor(timesteps, dtype=torch.long)
                mask = torch.tensor(mask, dtype=torch.float32)

                # Training step
                loss = self.train_step(rtg, states, actions, timesteps, mask)
                history["loss"].append(loss)
                history["lr"].append(self.scheduler.get_last_lr()[0])

                # Logging
                if step % 100 == 0:
                    logger.info(
                        f"Step {step}/{num_steps} | Loss: {loss:.4f} | "
                        f"LR: {self.scheduler.get_last_lr()[0]:.6f}"
                    )

                # Save checkpoint
                if step > 0 and step % save_interval == 0:
                    self.model.save(save_path.replace('.pt', f'_step{step}.pt'))

            # Save final model
            self.model.save(save_path)

            return history

else:
    # Stub classes when PyTorch not available
    class DecisionTransformer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for DecisionTransformer")

    class DecisionTransformerTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for DecisionTransformerTrainer")
