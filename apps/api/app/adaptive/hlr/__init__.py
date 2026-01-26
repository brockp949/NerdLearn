"""
Half-Life Regression (HLR) Reward Model

A psychometric model for estimating memory half-life and providing
reward signals for Curriculum Reinforcement Learning.

The HLR model estimates:
1. Memory half-life (h): How long until recall probability drops to 50%
2. Recall probability: p = 2^(-Δ/h) where Δ is time since last review
3. Memory strength (S): log_2(h) - used as the optimization target

The reward signal for RL is the change in total system memory strength:
r_t = Σ_k(S_t^k - S_{t-1}^k)

This captures both:
- Active gain: Successful review increases half-life
- Passive decay: Neglected concepts decay over time

References:
- Settles & Meeder (2016): A Trainable Spaced Repetition Model
- Duolingo Half-Life Regression
"""

from .half_life_regression import (
    HLRConfig,
    HLRModel,
    ConceptMemory,
    RewardCalculator,
)

__all__ = [
    "HLRConfig",
    "HLRModel",
    "ConceptMemory",
    "RewardCalculator",
]
