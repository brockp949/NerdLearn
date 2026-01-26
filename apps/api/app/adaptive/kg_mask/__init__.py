"""
Knowledge Graph Action Masking

Enforces pedagogical constraints on the RL policy by masking
invalid actions based on prerequisite relationships.

Key Features:
1. Dynamic action mask computation based on belief state
2. Prerequisite graph loading from Apache AGE/Neo4j
3. Logits masking for deep RL integration
4. Zone of Proximal Development awareness

The action masker ensures:
- Students never see concepts before prerequisites are mastered
- RL agent explores only within pedagogically valid actions
- 0% prerequisite violations guaranteed

Integration:
- Works with TD-BKT belief states for mastery checks
- Applies -∞ penalty to invalid action logits before softmax
- Ensures 100% probability mass on valid actions only
"""

from .action_masker import (
    ActionMaskerConfig,
    ActionMasker,
    PrerequisiteGraph,
)

__all__ = [
    "ActionMaskerConfig",
    "ActionMasker",
    "PrerequisiteGraph",
]
