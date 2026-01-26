"""
State Encoder for Neural ODE Memory Model.

Maps card features and user features to the initial latent memory state h(t₀).
Per Section 7.1 of CT-MCN specification:
    h(t_0) = E_ψ(x_static)

Supports cold-start initialization via learner phenotypes for new users
who don't have historical data yet.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Union


# Phenotype definitions matching the database schema
PHENOTYPE_MAP = {
    'unknown': 0,
    'fast_forgetter': 1,
    'steady_learner': 2,
    'cramper': 3,
    'deep_processor': 4,
    'night_owl': 5,
    'morning_lark': 6,
    'variable': 7,
}

# Phenotype characteristics for initialization priors
# Each phenotype has distinct characteristics that affect memory dynamics
PHENOTYPE_CHARACTERISTICS = {
    'fast_forgetter': {'decay_rate': 0.8, 'encoding_strength': 0.6, 'consolidation': 0.4, 'circadian_phase': 0.0},
    'steady_learner': {'decay_rate': 0.4, 'encoding_strength': 0.7, 'consolidation': 0.7, 'circadian_phase': 0.0},
    'cramper': {'decay_rate': 0.6, 'encoding_strength': 0.9, 'consolidation': 0.3, 'circadian_phase': 0.0},
    'deep_processor': {'decay_rate': 0.3, 'encoding_strength': 0.8, 'consolidation': 0.8, 'circadian_phase': 0.0},
    'night_owl': {'decay_rate': 0.5, 'encoding_strength': 0.7, 'consolidation': 0.6, 'circadian_phase': 0.8},  # Late peak
    'morning_lark': {'decay_rate': 0.5, 'encoding_strength': 0.7, 'consolidation': 0.6, 'circadian_phase': 0.2},  # Early peak
    'variable': {'decay_rate': 0.5, 'encoding_strength': 0.5, 'consolidation': 0.5, 'circadian_phase': 0.5},
}


class StateEncoder(nn.Module):
    """
    Encodes card features and user features into initial memory state h₀.

    Architecture:
    - Card encoder: Processes card difficulty, complexity, semantic embedding
    - User encoder: Processes user historical statistics
    - Phenotype embeddings: For cold-start when user history unavailable
    - State projection: Combines encodings into final latent state

    Cold-start Strategy:
    When a new user has no review history, we use their learner phenotype
    (determined via initial assessment or default) to initialize the state.
    """

    def __init__(
        self,
        card_feat_dim: int = 64,
        user_feat_dim: int = 16,
        state_dim: int = 32,
        hidden_dim: int = 64,
        num_phenotypes: int = 8,
    ):
        """
        Initialize the State Encoder.

        Args:
            card_feat_dim: Dimension of card feature vector
                           (includes difficulty, complexity, semantic embedding)
            user_feat_dim: Dimension of user feature vector
                           (includes avg_retention, review_count, avg_interval, etc.)
            state_dim: Dimension of the output memory state
            hidden_dim: Hidden layer dimension
            num_phenotypes: Number of learner phenotypes (including 'unknown')
        """
        super().__init__()

        self.card_feat_dim = card_feat_dim
        self.user_feat_dim = user_feat_dim
        self.state_dim = state_dim

        # Card encoder: processes static card features
        # Input features: [difficulty, complexity, ...semantic_embedding...]
        self.card_encoder = nn.Sequential(
            nn.Linear(card_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # User encoder: processes user historical statistics
        # Input features: [avg_retention, review_count, avg_interval, variance, ...]
        self.user_encoder = nn.Sequential(
            nn.Linear(user_feat_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
        )

        # Phenotype embeddings for cold-start initialization
        # 8 phenotypes: 0=unknown, 1-7=defined phenotypes
        phenotype_embed_dim = hidden_dim // 4
        self.phenotype_embed = nn.Embedding(num_phenotypes, phenotype_embed_dim)

        # Initialize phenotype embeddings with semantic structure
        self._init_phenotype_embeddings(phenotype_embed_dim)

        # State projection: combines card and user/phenotype encodings
        # Card encoding (hidden_dim // 2) + User/Phenotype encoding (hidden_dim // 4)
        combined_dim = hidden_dim // 2 + hidden_dim // 4
        self.state_proj = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.Tanh(),  # Bound intermediate representation
            nn.Linear(hidden_dim, state_dim),
        )

        # Optional: difficulty-aware scaling
        # Cards with higher difficulty start with lower initial memory strength
        self.difficulty_scale = nn.Linear(1, state_dim, bias=False)
        with torch.no_grad():
            self.difficulty_scale.weight.fill_(-0.1)  # Small negative scaling

    def _init_phenotype_embeddings(self, embed_dim: int):
        """Initialize phenotype embeddings with semantic structure."""
        with torch.no_grad():
            # Unknown phenotype: neutral initialization
            self.phenotype_embed.weight.data[0] = torch.zeros(embed_dim)

            # Initialize based on phenotype characteristics
            for name, idx in PHENOTYPE_MAP.items():
                if idx == 0:
                    continue
                chars = PHENOTYPE_CHARACTERISTICS.get(name, {})
                # Create embedding that encodes decay/encoding/consolidation/circadian tendencies
                embed = torch.zeros(embed_dim)
                quarter = embed_dim // 4
                # First quarter encodes decay tendency (higher = faster decay)
                embed[:quarter] = chars.get('decay_rate', 0.5) - 0.5
                # Second quarter encodes encoding strength
                embed[quarter:2 * quarter] = chars.get('encoding_strength', 0.5) - 0.5
                # Third quarter encodes consolidation
                embed[2 * quarter:3 * quarter] = chars.get('consolidation', 0.5) - 0.5
                # Fourth quarter encodes circadian phase preference
                embed[3 * quarter:] = chars.get('circadian_phase', 0.5) - 0.5
                self.phenotype_embed.weight.data[idx] = embed

    def forward(
        self,
        card_features: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
        phenotype_id: Optional[torch.Tensor] = None,
        difficulty: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode features into initial memory state.

        Args:
            card_features: Card feature vector [batch, card_feat_dim]
                           Includes difficulty, complexity, semantic embedding
            user_features: User statistics [batch, user_feat_dim] or None
                           If provided, used instead of phenotype
            phenotype_id: Learner phenotype ID [batch] (long tensor) or None
                          Used for cold-start when user_features unavailable
            difficulty: Card difficulty scalar [batch, 1] for additional scaling

        Returns:
            h0: Initial memory state [batch, state_dim]
        """
        batch_size = card_features.size(0)
        device = card_features.device

        # Encode card features
        card_enc = self.card_encoder(card_features)  # [batch, hidden_dim // 2]

        # Encode user features or use phenotype embedding
        if user_features is not None:
            user_enc = self.user_encoder(user_features)  # [batch, hidden_dim // 4]
        elif phenotype_id is not None:
            user_enc = self.phenotype_embed(phenotype_id)  # [batch, hidden_dim // 4]
        else:
            # Default: unknown phenotype (index 0)
            default_phenotype = torch.zeros(batch_size, dtype=torch.long, device=device)
            user_enc = self.phenotype_embed(default_phenotype)

        # Combine encodings
        combined = torch.cat([card_enc, user_enc], dim=-1)

        # Project to state space
        h0 = self.state_proj(combined)

        # Apply difficulty-aware scaling if provided
        if difficulty is not None:
            diff_scale = self.difficulty_scale(difficulty)  # [batch, state_dim]
            h0 = h0 + diff_scale

        return h0

    def cold_start_init(
        self,
        card_features: torch.Tensor,
        phenotype: Union[str, int],
        difficulty: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Initialize state for a new user using phenotype-based cold-start.

        Args:
            card_features: Card feature vector [batch, card_feat_dim]
            phenotype: Phenotype name (str) or ID (int)
            difficulty: Optional card difficulty [batch, 1]

        Returns:
            h0: Initial memory state [batch, state_dim]
        """
        batch_size = card_features.size(0)
        device = card_features.device

        # Convert phenotype to ID if string
        if isinstance(phenotype, str):
            phenotype_id = PHENOTYPE_MAP.get(phenotype.lower(), 0)
        else:
            phenotype_id = phenotype

        # Create phenotype tensor
        phenotype_tensor = torch.full(
            (batch_size,), phenotype_id, dtype=torch.long, device=device
        )

        return self.forward(
            card_features,
            user_features=None,
            phenotype_id=phenotype_tensor,
            difficulty=difficulty,
        )

    def get_phenotype_embedding(self, phenotype: Union[str, int]) -> torch.Tensor:
        """
        Get the raw embedding vector for a phenotype.

        Useful for analysis and visualization.
        """
        if isinstance(phenotype, str):
            phenotype_id = PHENOTYPE_MAP.get(phenotype.lower(), 0)
        else:
            phenotype_id = phenotype

        return self.phenotype_embed.weight[phenotype_id].detach()


class HierarchicalStateEncoder(StateEncoder):
    """
    Extended encoder with hierarchical feature processing.

    Processes card features at multiple granularities:
    - Low-level: Raw difficulty, complexity scores
    - Mid-level: Semantic embedding of card content
    - High-level: Topic/domain information

    This allows the model to capture different aspects of what makes
    a card easy or hard to remember.
    """

    def __init__(
        self,
        low_level_dim: int = 8,
        mid_level_dim: int = 48,
        high_level_dim: int = 8,
        user_feat_dim: int = 16,
        state_dim: int = 32,
        hidden_dim: int = 64,
    ):
        # Total card features = low + mid + high
        card_feat_dim = low_level_dim + mid_level_dim + high_level_dim
        super().__init__(card_feat_dim, user_feat_dim, state_dim, hidden_dim)

        self.low_level_dim = low_level_dim
        self.mid_level_dim = mid_level_dim
        self.high_level_dim = high_level_dim

        # Separate encoders for each feature level
        self.low_encoder = nn.Sequential(
            nn.Linear(low_level_dim, 16),
            nn.ReLU(),
        )
        self.mid_encoder = nn.Sequential(
            nn.Linear(mid_level_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
        )
        self.high_encoder = nn.Sequential(
            nn.Linear(high_level_dim, 16),
            nn.ReLU(),
        )

        # Attention-based combination
        self.level_attention = nn.Sequential(
            nn.Linear(16 + 32 + 16, 3),
            nn.Softmax(dim=-1),
        )

        # Override card encoder to use hierarchical features
        self.card_encoder = nn.Sequential(
            nn.Linear(16 + 32 + 16, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

    def forward(
        self,
        card_features: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
        phenotype_id: Optional[torch.Tensor] = None,
        difficulty: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode hierarchical card features into initial memory state.

        Expects card_features to be concatenation of [low, mid, high] level features.
        """
        # Split card features into levels
        low_feat = card_features[:, :self.low_level_dim]
        mid_feat = card_features[:, self.low_level_dim:self.low_level_dim + self.mid_level_dim]
        high_feat = card_features[:, self.low_level_dim + self.mid_level_dim:]

        # Encode each level
        low_enc = self.low_encoder(low_feat)
        mid_enc = self.mid_encoder(mid_feat)
        high_enc = self.high_encoder(high_feat)

        # Concatenate for attention
        combined_levels = torch.cat([low_enc, mid_enc, high_enc], dim=-1)

        # Compute attention weights
        attention = self.level_attention(combined_levels)  # [batch, 3]

        # Apply attention-weighted combination
        # Note: For simplicity, we just concatenate; attention weights can be used for analysis
        # A more sophisticated version would use the attention to weight the encodings

        # Use parent class forward with combined hierarchical encoding
        batch_size = card_features.size(0)
        device = card_features.device

        card_enc = self.card_encoder(combined_levels)

        # Get user encoding (same as parent)
        if user_features is not None:
            user_enc = self.user_encoder(user_features)
        elif phenotype_id is not None:
            user_enc = self.phenotype_embed(phenotype_id)
        else:
            default_phenotype = torch.zeros(batch_size, dtype=torch.long, device=device)
            user_enc = self.phenotype_embed(default_phenotype)

        # Combine and project
        combined = torch.cat([card_enc, user_enc], dim=-1)
        h0 = self.state_proj(combined)

        if difficulty is not None:
            diff_scale = self.difficulty_scale(difficulty)
            h0 = h0 + diff_scale

        return h0
