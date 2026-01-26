"""
MMSAF-Net: Multi-Modal Self-Attention Fusion Network

Deep learning module for fusing heterogeneous telemetry data streams
to infer high-fidelity User State Vectors representing latent cognitive status.

Input Sub-vectors:
1. Behavioral (Physiological): Mouse velocity, jitter, click rate, idle time
2. Contextual: Time of day, device type, bandwidth
3. Content: Complexity score, modality type, length

The self-attention mechanism dynamically weighs feature importance based on
current context. During video playback, mouse signals are down-weighted;
during interactive simulations, mouse jitter gains higher attention.

Output:
- User State Vector (u_t) representing cognitive state
  e.g., "Highly Engaged but Fatigued"

References:
- "Design of an integrated multi-modal machine learning framework for
   real-time student engagement evaluation" (PMC)
- Vaswani et al., "Attention Is All You Need" (Transformer architecture)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


# Try to import PyTorch for deep learning version
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - using numpy fallback for MMSAF-Net")


@dataclass
class BehavioralFeatures:
    """
    Behavioral/Physiological features from mouse dynamics

    Captures the physical manifestation of cognitive state.
    """
    # Mouse dynamics
    mouse_velocity: float = 0.0  # Average velocity (px/s)
    mouse_jitter: float = 0.0  # High-frequency movement variance
    velocity_std: float = 0.0  # Velocity standard deviation
    click_rate: float = 0.0  # Clicks per minute

    # Temporal patterns
    idle_time_ratio: float = 0.0  # Fraction of time idle
    micro_hesitation_rate: float = 0.0  # Hesitations per minute
    dwell_time_factor: float = 0.5  # Normalized dwell time

    # Movement quality
    straightness_ratio: float = 0.5  # Path efficiency
    curvature_entropy: float = 0.0  # Movement unpredictability

    # Engagement indicators
    scroll_velocity: float = 0.0  # Scroll speed
    scroll_depth: float = 0.0  # Maximum scroll depth reached

    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector"""
        return np.array([
            self.mouse_velocity / 500,  # Normalize to ~0-1
            self.mouse_jitter,
            self.velocity_std / 200,
            self.click_rate / 10,
            self.idle_time_ratio,
            self.micro_hesitation_rate / 5,
            self.dwell_time_factor,
            self.straightness_ratio,
            self.curvature_entropy,
            self.scroll_velocity / 1000,
            self.scroll_depth,
        ])


@dataclass
class ContextualFeatures:
    """
    Environmental and session context features

    Captures external factors affecting learning capacity.
    """
    # Time context
    time_of_day: float = 0.5  # 0=morning, 0.5=afternoon, 1=evening
    day_of_week: int = 0  # 0=Monday, 6=Sunday
    session_duration_minutes: float = 0.0

    # Device context
    device_type: str = "desktop"  # "desktop", "mobile", "tablet"
    screen_width: int = 1920
    screen_height: int = 1080

    # Network context
    bandwidth_quality: float = 1.0  # 0-1, 1=excellent

    # Session context
    session_card_count: int = 0  # Cards seen this session
    session_success_rate: float = 0.5  # Recent performance
    consecutive_failures: int = 0

    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector"""
        # Encode device type
        device_encoding = {
            "desktop": [1, 0, 0],
            "mobile": [0, 1, 0],
            "tablet": [0, 0, 1],
        }.get(self.device_type, [0.33, 0.33, 0.33])

        # Encode day of week (cyclical)
        day_sin = np.sin(2 * np.pi * self.day_of_week / 7)
        day_cos = np.cos(2 * np.pi * self.day_of_week / 7)

        return np.array([
            self.time_of_day,
            day_sin,
            day_cos,
            min(1.0, self.session_duration_minutes / 60),  # Normalize to 1 hour
            *device_encoding,
            self.screen_width / 2560,  # Normalize
            self.screen_height / 1440,
            self.bandwidth_quality,
            min(1.0, self.session_card_count / 20),  # Normalize to 20 cards
            self.session_success_rate,
            min(1.0, self.consecutive_failures / 5),  # Normalize
        ])


@dataclass
class ContentFeatures:
    """
    Content-specific features

    Describes the current learning material being presented.
    """
    # Complexity
    complexity_score: float = 0.5  # 0-1
    reading_level: float = 0.5  # Grade level normalized
    concept_novelty: float = 0.5  # How new this concept is to user

    # Modality
    modality_type: str = "text"  # "text", "video", "audio", "interactive"
    content_length_minutes: float = 5.0  # Expected duration

    # Semantic
    topic_familiarity: float = 0.5  # User's familiarity with topic
    prerequisite_mastery: float = 0.5  # Mastery of prerequisites

    # Metadata
    interactivity_level: float = 0.0  # 0=passive, 1=highly interactive
    visual_density: float = 0.5  # Amount of visual information

    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector"""
        # Encode modality
        modality_encoding = {
            "text": [1, 0, 0, 0],
            "video": [0, 1, 0, 0],
            "audio": [0, 0, 1, 0],
            "interactive": [0, 0, 0, 1],
        }.get(self.modality_type, [0.25, 0.25, 0.25, 0.25])

        return np.array([
            self.complexity_score,
            self.reading_level,
            self.concept_novelty,
            *modality_encoding,
            min(1.0, self.content_length_minutes / 30),  # Normalize
            self.topic_familiarity,
            self.prerequisite_mastery,
            self.interactivity_level,
            self.visual_density,
        ])


@dataclass
class UserStateVector:
    """
    Output User State Vector representing latent cognitive status

    This is the unified representation of the user's current state,
    suitable for input to the bandit decision engine.
    """
    # Core cognitive dimensions
    cognitive_capacity: float = 0.5  # Available mental resources
    fatigue_level: float = 0.0  # Accumulated fatigue
    focus_level: float = 0.5  # Current attention/focus
    engagement: float = 0.5  # Overall engagement

    # Affective dimensions
    frustration: float = 0.0  # Frustration indicator
    confidence: float = 0.5  # Self-efficacy indicator
    flow_state: float = 0.0  # Flow state indicator

    # Learning state
    confusion_indicator: float = 0.0  # Confusion level
    struggle_indicator: float = 0.0  # Productive struggle vs frustration

    # Meta features
    state_confidence: float = 0.5  # Confidence in this state estimate
    dominant_state: str = "neutral"  # "engaged", "fatigued", "confused", etc.

    def to_vector(self) -> np.ndarray:
        """Convert to vector for bandit input"""
        return np.array([
            self.cognitive_capacity,
            1 - self.fatigue_level,  # Invert so higher = better
            self.focus_level,
            self.engagement,
            1 - self.frustration,  # Invert
            self.confidence,
            self.flow_state,
            1 - self.confusion_indicator,  # Invert
            self.struggle_indicator,
        ])

    def get_summary(self) -> str:
        """Human-readable state summary"""
        if self.flow_state > 0.7:
            return "In Flow State - highly engaged and productive"
        elif self.fatigue_level > 0.7:
            return "Fatigued - consider break or lighter content"
        elif self.confusion_indicator > 0.7:
            return "Confused - scaffolding needed"
        elif self.frustration > 0.7:
            return "Frustrated - reduce difficulty or change approach"
        elif self.engagement > 0.7:
            return "Highly Engaged - maintain current approach"
        elif self.engagement < 0.3:
            return "Disengaged - need intervention"
        else:
            return f"Moderate engagement (focus={self.focus_level:.2f})"


class MMSAFNetNumpy:
    """
    Numpy implementation of MMSAF-Net for environments without PyTorch

    Uses learned attention weights and simple feedforward networks
    approximated with numpy operations.
    """

    def __init__(
        self,
        behavioral_dim: int = 11,
        contextual_dim: int = 13,
        content_dim: int = 12,
        hidden_dim: int = 32,
        output_dim: int = 9,
    ):
        self.behavioral_dim = behavioral_dim
        self.contextual_dim = contextual_dim
        self.content_dim = content_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Total input dimension
        self.total_input_dim = behavioral_dim + contextual_dim + content_dim

        # Initialize weights (would be learned in real training)
        np.random.seed(42)

        # Projection weights for each modality
        self.W_behavioral = np.random.randn(behavioral_dim, hidden_dim) * 0.1
        self.W_contextual = np.random.randn(contextual_dim, hidden_dim) * 0.1
        self.W_content = np.random.randn(content_dim, hidden_dim) * 0.1

        # Attention weights
        self.W_query = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.W_key = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.W_value = np.random.randn(hidden_dim, hidden_dim) * 0.1

        # Output projection
        self.W_output = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b_output = np.zeros(output_dim)

        # Learned modality importance weights (adjust based on context)
        self.modality_weights = {
            "text": np.array([1.0, 0.3, 0.5]),  # [behavioral, contextual, content]
            "video": np.array([0.3, 0.5, 0.7]),  # Lower behavioral for passive
            "audio": np.array([0.2, 0.5, 0.7]),
            "interactive": np.array([1.0, 0.4, 0.6]),  # High behavioral for interactive
        }

        logger.info("MMSAFNetNumpy initialized")

    def forward(
        self,
        behavioral: BehavioralFeatures,
        contextual: ContextualFeatures,
        content: ContentFeatures,
    ) -> UserStateVector:
        """
        Forward pass to compute User State Vector

        1. Project each feature set to hidden dimension
        2. Apply self-attention to learn feature importance
        3. Fuse attended features
        4. Project to output state dimensions
        """
        # Get raw vectors
        b_vec = behavioral.to_vector()
        c_vec = contextual.to_vector()
        ct_vec = content.to_vector()

        # Project to hidden dimension
        h_behavioral = np.tanh(b_vec @ self.W_behavioral)
        h_contextual = np.tanh(c_vec @ self.W_contextual)
        h_content = np.tanh(ct_vec @ self.W_content)

        # Stack for attention
        H = np.stack([h_behavioral, h_contextual, h_content])  # (3, hidden_dim)

        # Self-attention
        Q = H @ self.W_query  # (3, hidden_dim)
        K = H @ self.W_key
        V = H @ self.W_value

        # Scaled dot-product attention
        scale = np.sqrt(self.hidden_dim)
        attention_scores = (Q @ K.T) / scale  # (3, 3)
        attention_weights = self._softmax(attention_scores)  # (3, 3)

        # Apply attention
        attended = attention_weights @ V  # (3, hidden_dim)

        # Get modality-specific weights based on content type
        modality_w = self.modality_weights.get(
            content.modality_type,
            np.array([0.5, 0.5, 0.5])
        )

        # Weighted fusion
        fused = (
            modality_w[0] * attended[0] +
            modality_w[1] * attended[1] +
            modality_w[2] * attended[2]
        )

        # Output projection
        output = self._sigmoid(fused @ self.W_output + self.b_output)

        # Map to UserStateVector
        return UserStateVector(
            cognitive_capacity=float(output[0]),
            fatigue_level=float(1 - output[1]),  # Invert back
            focus_level=float(output[2]),
            engagement=float(output[3]),
            frustration=float(1 - output[4]),  # Invert back
            confidence=float(output[5]),
            flow_state=float(output[6]),
            confusion_indicator=float(1 - output[7]),  # Invert back
            struggle_indicator=float(output[8]),
            state_confidence=float(np.mean(np.max(attention_weights, axis=1))),
            dominant_state=self._determine_dominant_state(output),
        )

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _determine_dominant_state(self, output: np.ndarray) -> str:
        """Determine the dominant cognitive state"""
        # output indices: [capacity, energy, focus, engagement,
        #                  calm, confidence, flow, clarity, struggle]

        engagement = output[3]
        flow = output[6]
        fatigue = 1 - output[1]
        confusion = 1 - output[7]
        frustration = 1 - output[4]

        if flow > 0.7:
            return "flow"
        elif fatigue > 0.7:
            return "fatigued"
        elif confusion > 0.7:
            return "confused"
        elif frustration > 0.7:
            return "frustrated"
        elif engagement > 0.7:
            return "engaged"
        elif engagement < 0.3:
            return "disengaged"
        else:
            return "neutral"


if TORCH_AVAILABLE:
    class MultiHeadSelfAttention(nn.Module):
        """Multi-head self-attention for feature fusion"""

        def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads

            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)

            self.dropout = nn.Dropout(dropout)
            self.scale = self.head_dim ** -0.5

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            batch_size, seq_len, _ = x.shape

            # Project to Q, K, V
            Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Attention scores
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Apply attention
            attn_output = torch.matmul(attn_weights, V)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

            output = self.out_proj(attn_output)
            return output, attn_weights.mean(dim=1)  # Average attention across heads

    class MMSAFNetTorch(nn.Module):
        """
        PyTorch implementation of MMSAF-Net

        Full deep learning version with:
        - Separate encoders for each modality
        - Multi-head self-attention fusion
        - Residual connections
        - Layer normalization
        """

        def __init__(
            self,
            behavioral_dim: int = 11,
            contextual_dim: int = 13,
            content_dim: int = 12,
            hidden_dim: int = 64,
            output_dim: int = 9,
            num_heads: int = 4,
            dropout: float = 0.1,
        ):
            super().__init__()

            # Modality encoders
            self.behavioral_encoder = nn.Sequential(
                nn.Linear(behavioral_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.contextual_encoder = nn.Sequential(
                nn.Linear(contextual_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.content_encoder = nn.Sequential(
                nn.Linear(content_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )

            # Self-attention fusion
            self.self_attention = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
            self.attention_norm = nn.LayerNorm(hidden_dim)

            # Fusion MLP
            self.fusion_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )

            # Output projection
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.Sigmoid(),
            )

            # Modality importance predictor (context-dependent weights)
            self.importance_predictor = nn.Sequential(
                nn.Linear(content_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 3),
                nn.Softmax(dim=-1),
            )

        def forward(
            self,
            behavioral: torch.Tensor,
            contextual: torch.Tensor,
            content: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass

            Args:
                behavioral: (batch, behavioral_dim)
                contextual: (batch, contextual_dim)
                content: (batch, content_dim)

            Returns:
                output: (batch, output_dim) - User State Vector
                attention_weights: (batch, 3, 3) - Inter-modality attention
            """
            # Encode each modality
            h_behavioral = self.behavioral_encoder(behavioral)
            h_contextual = self.contextual_encoder(contextual)
            h_content = self.content_encoder(content)

            # Stack for attention (batch, 3, hidden_dim)
            H = torch.stack([h_behavioral, h_contextual, h_content], dim=1)

            # Self-attention with residual
            attended, attn_weights = self.self_attention(H)
            H = self.attention_norm(H + attended)

            # Predict modality importance based on content
            importance = self.importance_predictor(content)  # (batch, 3)

            # Weighted fusion
            weighted = H * importance.unsqueeze(-1)  # (batch, 3, hidden_dim)
            fused = weighted.view(weighted.shape[0], -1)  # (batch, 3*hidden_dim)

            # MLP fusion
            fused = self.fusion_mlp(fused)

            # Output
            output = self.output_proj(fused)

            return output, attn_weights


# Main interface class
class MMSAFNet:
    """
    MMSAF-Net: Multi-Modal Self-Attention Fusion Network

    Unified interface that uses PyTorch if available, otherwise numpy fallback.
    """

    def __init__(
        self,
        behavioral_dim: int = 11,
        contextual_dim: int = 13,
        content_dim: int = 12,
        hidden_dim: int = 64,
        output_dim: int = 9,
        use_torch: bool = True,
    ):
        self.use_torch = use_torch and TORCH_AVAILABLE

        if self.use_torch:
            self.model = MMSAFNetTorch(
                behavioral_dim=behavioral_dim,
                contextual_dim=contextual_dim,
                content_dim=content_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
            )
            self.model.eval()  # Default to inference mode
            logger.info("MMSAFNet initialized with PyTorch backend")
        else:
            self.model = MMSAFNetNumpy(
                behavioral_dim=behavioral_dim,
                contextual_dim=contextual_dim,
                content_dim=content_dim,
                hidden_dim=hidden_dim // 2,  # Smaller for numpy
                output_dim=output_dim,
            )
            logger.info("MMSAFNet initialized with numpy backend")

    def infer_state(
        self,
        behavioral: BehavioralFeatures,
        contextual: ContextualFeatures,
        content: ContentFeatures,
    ) -> UserStateVector:
        """
        Infer User State Vector from multi-modal features

        Args:
            behavioral: Mouse dynamics and engagement signals
            contextual: Environmental and session context
            content: Current content features

        Returns:
            UserStateVector with cognitive state estimates
        """
        if isinstance(self.model, MMSAFNetNumpy):
            return self.model.forward(behavioral, contextual, content)
        else:
            # PyTorch path
            with torch.no_grad():
                b_tensor = torch.tensor(behavioral.to_vector(), dtype=torch.float32).unsqueeze(0)
                c_tensor = torch.tensor(contextual.to_vector(), dtype=torch.float32).unsqueeze(0)
                ct_tensor = torch.tensor(content.to_vector(), dtype=torch.float32).unsqueeze(0)

                output, _ = self.model(b_tensor, c_tensor, ct_tensor)
                output = output.squeeze(0).numpy()

                return UserStateVector(
                    cognitive_capacity=float(output[0]),
                    fatigue_level=float(1 - output[1]),
                    focus_level=float(output[2]),
                    engagement=float(output[3]),
                    frustration=float(1 - output[4]),
                    confidence=float(output[5]),
                    flow_state=float(output[6]),
                    confusion_indicator=float(1 - output[7]),
                    struggle_indicator=float(output[8]),
                    state_confidence=0.7,  # Would come from attention weights
                    dominant_state=self._determine_dominant_state(output),
                )

    def _determine_dominant_state(self, output: np.ndarray) -> str:
        """Determine dominant state from output vector"""
        engagement = output[3]
        flow = output[6]
        fatigue = 1 - output[1]
        confusion = 1 - output[7]
        frustration = 1 - output[4]

        if flow > 0.7:
            return "flow"
        elif fatigue > 0.7:
            return "fatigued"
        elif confusion > 0.7:
            return "confused"
        elif frustration > 0.7:
            return "frustrated"
        elif engagement > 0.7:
            return "engaged"
        elif engagement < 0.3:
            return "disengaged"
        return "neutral"
