"""
Temporal Difference Bayesian Knowledge Tracing (TD-BKT)

An advanced knowledge tracing model that extends standard BKT with:
1. Temporal decay modeling (forgetting curves)
2. Time-dependent slip/guess probabilities
3. Belief state representation for POMDP integration

This module provides the state estimator for the Curriculum RL system,
outputting belief states that capture both mastery and recency information.

References:
- Integrating Temporal Information Into Knowledge Tracing (IEEE, 2018)
- TD-BKT: A Temporal Difference Approach
"""

from .temporal_difference_bkt import (
    TDBKTConfig,
    BeliefState,
    ConceptState,
    TemporalDifferenceBKT,
)

__all__ = [
    "TDBKTConfig",
    "BeliefState",
    "ConceptState",
    "TemporalDifferenceBKT",
]
