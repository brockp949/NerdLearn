"""
Offline Reinforcement Learning for Curriculum Optimization

This module implements offline RL algorithms for learning optimal
curriculum sequencing policies from historical student interaction data.

Components:
1. Data Pipeline: Extract and process trajectories from ReviewLog
2. Decision Transformer: Sequence modeling approach to RL
3. Conservative Q-Learning (CQL): Value-based approach with conservatism
4. DT-Lite: Lightweight inference without PyTorch

Key Concepts:
- Offline RL learns from static datasets without online interaction
- Avoids ethical concerns of experimenting on real students
- Can "stitch" together optimal subsequences from suboptimal data

Algorithms:
- Decision Transformer: Treats RL as sequence prediction
- CQL: Penalizes Q-values for out-of-distribution actions

References:
- Chen et al. (2021): Decision Transformer
- Kumar et al. (2020): Conservative Q-Learning
"""

from .data_pipeline import (
    TrajectoryDataset,
    DataPipeline,
    Transition,
    Trajectory,
)

# Conditional imports for PyTorch-dependent modules
try:
    from .decision_transformer import (
        DecisionTransformerConfig,
        DecisionTransformer,
    )
    from .cql_agent import (
        CQLConfig,
        CQLAgent,
    )
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from .dt_lite import DTLite, DTLiteConfig

__all__ = [
    "TrajectoryDataset",
    "DataPipeline",
    "Transition",
    "Trajectory",
    "DTLite",
    "DTLiteConfig",
    "PYTORCH_AVAILABLE",
]

if PYTORCH_AVAILABLE:
    __all__.extend([
        "DecisionTransformerConfig",
        "DecisionTransformer",
        "CQLConfig",
        "CQLAgent",
    ])
