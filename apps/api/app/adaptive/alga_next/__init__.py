"""
ALGA-Next: Adaptive Learning via Generative Allocation

Research-aligned implementation of Adaptive Content Modality Selection System (ACMSS)
based on the Engagement-Mediated Learning Hypothesis.

Key Components:
1. Hybrid LinUCB - Contextual bandit with interaction terms for modality selection
2. MMSAF-Net - Multi-Modal Self-Attention Fusion for telemetry processing
3. Attention Transfer - MTL architecture for cold-start problem
4. Composite Reward - Engagement + Mastery with fatigue penalty
5. Generative UI Registry - SDUI schema generation for adaptive interfaces

References:
- "Adaptive Multimodal Orchestration: A Context-Aware Framework"
- LinUCB: Li et al., "A Contextual-Bandit Approach to Personalized News Article Recommendation"
- MouStress: Sun et al., "MouStress: Detecting Stress from Mouse Motion"
"""

from .hybrid_linucb import (
    HybridLinUCB,
    ModalityArm,
    ContextVector,
    ModalityPolicy,
    InteractionFeatureBuilder,
    Modality,
    create_modality_arms,
)
from .mmsaf_net import (
    MMSAFNet,
    UserStateVector,
    BehavioralFeatures,
    ContextualFeatures,
    ContentFeatures,
)
from .attention_transfer import (
    AttentionTransferNetwork,
    CrossModalityTransferMatrix,
    UserObservation,
    ModalityType,
    ModalityPrediction,
    TransferPredictions,
)
from .reward_function import (
    CompositeRewardFunction,
    RewardComponents,
    RewardConfig,
    FatiguePenalty,
    RewardObjective,
)
from .generative_ui import (
    GenerativeUIRegistry,
    AtomicContentUnit,
    SDUISchema,
    AdaptiveCard,
    ScaffoldingLevel,
    ContentUnitType,
    AdaptivityType,
)
from .mouse_stress import (
    MouStressAnalyzer,
    KinematicStiffness,
    TrajectoryAnalysis,
    LearnerState,
    MouStressResult,
)

__all__ = [
    # LinUCB
    "HybridLinUCB",
    "ModalityArm",
    "ContextVector",
    "ModalityPolicy",
    "InteractionFeatureBuilder",
    "Modality",
    "create_modality_arms",
    # MMSAF
    "MMSAFNet",
    "UserStateVector",
    "BehavioralFeatures",
    "ContextualFeatures",
    "ContentFeatures",
    # Attention Transfer
    "AttentionTransferNetwork",
    "CrossModalityTransferMatrix",
    "UserObservation",
    "ModalityType",
    "ModalityPrediction",
    "TransferPredictions",
    # Reward
    "CompositeRewardFunction",
    "RewardComponents",
    "RewardConfig",
    "FatiguePenalty",
    "RewardObjective",
    # Generative UI
    "GenerativeUIRegistry",
    "AtomicContentUnit",
    "SDUISchema",
    "AdaptiveCard",
    "ScaffoldingLevel",
    "ContentUnitType",
    "AdaptivityType",
    # MouStress
    "MouStressAnalyzer",
    "KinematicStiffness",
    "TrajectoryAnalysis",
    "LearnerState",
    "MouStressResult",
]
