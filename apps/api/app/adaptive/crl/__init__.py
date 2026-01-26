"""
Curriculum Reinforcement Learning Policy Service

Unified interface for curriculum sequencing using offline RL.

This module provides:
1. CurriculumRLPolicy: Main policy class for concept selection
2. Integration with TD-BKT, HLR, and Action Masking
3. Support for both Decision Transformer and CQL backends
4. Production-ready inference with DT-Lite

Usage:
    policy = CurriculumRLPolicy.load("model_path")
    concept_id = policy.select_next_concept(
        belief_state=belief,
        valid_actions=mask
    )
"""

from .curriculum_rl_policy import (
    CurriculumRLPolicy,
    PolicyConfig,
    PolicyType,
)

__all__ = [
    "CurriculumRLPolicy",
    "PolicyConfig",
    "PolicyType",
]
