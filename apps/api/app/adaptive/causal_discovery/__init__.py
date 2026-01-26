"""
Causal Discovery Module for Adaptive Learning

This module implements algorithms for discovering causal relationships between educational concepts
using student performance data. Based on "Causal Discovery for Educational Graphs" specification.

Algorithms:
- NOTEARS (Score-based DAG discovery with continuous optimization)
- FCI (Constraint-based causal inference with latent confounders)
- Leiden (Community detection for graph clustering)
- Bootstrap Stability Selection (Confidence scoring via resampling)

Pipeline:
1. Data Preprocessing: Convert mastery logs to User x Concept matrix
2. Global Discovery: NOTEARS for DAG skeleton
3. Community Detection: Leiden for clustering
4. Local Refinement: FCI on dense subcommunities
5. Confidence Scoring: Bootstrap stability selection
6. Graph Persistence: MERGE to Apache AGE
"""

from app.adaptive.causal_discovery.manager import causal_manager, CausalDiscoveryManager
from app.adaptive.causal_discovery.bootstrap import bootstrap_selector, BootstrapStabilitySelector

__all__ = [
    "causal_manager",
    "CausalDiscoveryManager",
    "bootstrap_selector",
    "BootstrapStabilitySelector",
]
