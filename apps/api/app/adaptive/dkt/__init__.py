"""
Deep Knowledge Tracing (DKT) Module

Modern deep learning approaches for knowledge tracing that outperform
traditional Bayesian Knowledge Tracing (BKT).

Includes:
- DKT: LSTM-based deep knowledge tracing
- SAKT: Self-Attentive Knowledge Tracing (Transformer-based)
- DKTLite: Lightweight inference-only version
"""

from .deep_knowledge_tracer import (
    DeepKnowledgeTracer,
    DKTLite,
    DKTConfig,
    InteractionSequence,
    KnowledgeState,
)

__all__ = [
    "DeepKnowledgeTracer",
    "DKTLite",
    "DKTConfig",
    "InteractionSequence",
    "KnowledgeState",
]
