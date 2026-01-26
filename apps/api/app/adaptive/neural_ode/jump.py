"""
Jump Network for Neural ODE Memory Model.

Implements discrete state updates when review events occur.
Per Section 2.3 of CT-MCN specification:
    h(t_i+) = h(t_i-) + g_φ(h(t_i-), y_i, u_i)

The jump network captures encoding dynamics (learning) separate from
the continuous decay dynamics (forgetting) handled by the drift network.
"""

import torch
import torch.nn as nn
from typing import Optional


class JumpNetwork(nn.Module):
    """
    Computes instantaneous state update when a review event occurs.

    The jump magnitude depends on:
    - Current memory state (h_pre)
    - Review grade/rating (1-4 scale: Again, Hard, Good, Easy)
    - Telemetry signals (response time, hesitation, tortuosity, fluency)

    Architecture:
    - Grade embedding: Maps discrete rating to learned vector
    - Telemetry encoder: Processes continuous behavioral signals
    - Jump network: Computes state delta with residual connection
    """

    def __init__(
        self,
        state_dim: int = 32,
        grade_embedding_dim: int = 8,
        telemetry_dim: int = 4,
        hidden_dim: int = 64,
    ):
        """
        Initialize the Jump Network.

        Args:
            state_dim: Dimension of the memory state vector
            grade_embedding_dim: Dimension of grade embedding
            telemetry_dim: Number of telemetry features (RT, hesitation, tortuosity, fluency)
            hidden_dim: Hidden layer dimension for jump computation
        """
        super().__init__()

        self.state_dim = state_dim
        self.grade_embedding_dim = grade_embedding_dim
        self.telemetry_dim = telemetry_dim

        # Grade embedding: maps rating (0=pad, 1-4=grades) to learned vector
        # Grade semantics: 1=Again (fail), 2=Hard, 3=Good, 4=Easy
        self.grade_embed = nn.Embedding(5, grade_embedding_dim, padding_idx=0)

        # Initialize grade embeddings with semantic structure
        # Higher grades should start with larger positive values
        with torch.no_grad():
            self.grade_embed.weight[1] = torch.randn(grade_embedding_dim) * 0.1 - 0.3  # Again: negative
            self.grade_embed.weight[2] = torch.randn(grade_embedding_dim) * 0.1 - 0.1  # Hard: slight negative
            self.grade_embed.weight[3] = torch.randn(grade_embedding_dim) * 0.1 + 0.1  # Good: slight positive
            self.grade_embed.weight[4] = torch.randn(grade_embedding_dim) * 0.1 + 0.3  # Easy: positive

        # Telemetry encoder: processes behavioral signals
        # Input: [response_time_norm, hesitation, tortuosity, fluency]
        self.telemetry_net = nn.Sequential(
            nn.Linear(telemetry_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 16),
        )

        # Jump computation network
        # Input: concatenation of [h_pre, grade_emb, telemetry_emb]
        combined_dim = state_dim + grade_embedding_dim + 16
        self.jump_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Learnable scaling factor for jump magnitude
        # Starts small to ensure stable training
        self.jump_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        h_pre: torch.Tensor,
        grade: torch.Tensor,
        telemetry: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute post-review memory state.

        Args:
            h_pre: Pre-review memory state [batch, state_dim]
            grade: Review grade 1-4 [batch] (long tensor)
            telemetry: Behavioral signals [batch, telemetry_dim] or None
                       Expected format: [response_time_norm, hesitation, tortuosity, fluency]
                       All values should be normalized to roughly [0, 1]

        Returns:
            h_post: Post-review memory state [batch, state_dim]
        """
        batch_size = h_pre.size(0)
        device = h_pre.device

        # Embed grade
        grade_emb = self.grade_embed(grade)  # [batch, grade_embedding_dim]

        # Process telemetry or use default neutral values
        if telemetry is not None:
            telem_emb = self.telemetry_net(telemetry)  # [batch, 16]
        else:
            # Default telemetry: neutral values (0.5 for normalized signals)
            default_telemetry = torch.full(
                (batch_size, self.telemetry_dim),
                0.5,
                device=device,
                dtype=h_pre.dtype
            )
            telem_emb = self.telemetry_net(default_telemetry)

        # Concatenate all inputs
        combined = torch.cat([h_pre, grade_emb, telem_emb], dim=-1)

        # Compute state delta
        delta_h = self.jump_net(combined)

        # Apply scaled residual connection
        # h_post = h_pre + scale * delta_h
        h_post = h_pre + self.jump_scale * delta_h

        return h_post

    def get_jump_magnitude(
        self,
        h_pre: torch.Tensor,
        grade: torch.Tensor,
        telemetry: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the magnitude of the state jump (for analysis/logging).

        Returns:
            magnitude: L2 norm of the state change [batch]
        """
        h_post = self.forward(h_pre, grade, telemetry)
        delta = h_post - h_pre
        return torch.norm(delta, dim=-1)


class GatedJumpNetwork(JumpNetwork):
    """
    Extended Jump Network with gating mechanism.

    Uses a learned gate to control how much of the update is applied,
    allowing the model to learn when updates should be small vs large.
    This can help with stability during early training.
    """

    def __init__(
        self,
        state_dim: int = 32,
        grade_embedding_dim: int = 8,
        telemetry_dim: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__(state_dim, grade_embedding_dim, telemetry_dim, hidden_dim)

        # Gate network: outputs per-dimension gate values in [0, 1]
        combined_dim = state_dim + grade_embedding_dim + 16
        self.gate_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h_pre: torch.Tensor,
        grade: torch.Tensor,
        telemetry: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute gated post-review memory state.

        The gate controls per-dimension how much of the update is applied:
            h_post = h_pre + gate * (scale * delta_h)
        """
        batch_size = h_pre.size(0)
        device = h_pre.device

        # Embed grade
        grade_emb = self.grade_embed(grade)

        # Process telemetry
        if telemetry is not None:
            telem_emb = self.telemetry_net(telemetry)
        else:
            default_telemetry = torch.full(
                (batch_size, self.telemetry_dim),
                0.5,
                device=device,
                dtype=h_pre.dtype
            )
            telem_emb = self.telemetry_net(default_telemetry)

        # Concatenate all inputs
        combined = torch.cat([h_pre, grade_emb, telem_emb], dim=-1)

        # Compute state delta and gate
        delta_h = self.jump_net(combined)
        gate = self.gate_net(combined)

        # Apply gated residual connection
        h_post = h_pre + gate * (self.jump_scale * delta_h)

        return h_post
