"""
Knowledge Graph Action Masking for Curriculum RL

Ensures the RL policy respects prerequisite constraints by computing
dynamic action masks based on the student's current knowledge state.

Action Masking Process:
1. Load prerequisite graph from knowledge graph database
2. For each concept, check if all prerequisites are mastered
3. Generate binary mask vector M ∈ {0,1}^K
4. Apply mask to RL logits: z̃ = z + (1-M) * (-∞)
5. Softmax over masked logits ensures valid actions only

This transforms RL exploration from "random chaos" to "structured discovery"
within the student's Zone of Proximal Development.

Integration with Deep RL:
- CQL: Mask applied before Q-value maximization
- Decision Transformer: Mask applied before action sampling
- Ensures 0% prerequisite violations by construction
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime
import numpy as np
from enum import Enum


class MaskingMode(str, Enum):
    """How strictly to enforce prerequisites"""
    STRICT = "strict"           # Must fully master prerequisites
    SOFT = "soft"               # Weighted by prerequisite mastery
    PROGRESSIVE = "progressive"  # Gradually unlock as prerequisites improve


@dataclass
class ActionMaskerConfig:
    """Configuration for action masking"""

    # Prerequisite satisfaction threshold
    prerequisite_threshold: float = 0.85  # Must have 85% mastery of prerequisites

    # Masking mode
    masking_mode: MaskingMode = MaskingMode.STRICT

    # Soft masking parameters (for SOFT mode)
    soft_mask_temperature: float = 1.0  # Temperature for soft masking
    min_mask_value: float = 0.01        # Minimum mask value (never fully 0)

    # Progressive unlocking (for PROGRESSIVE mode)
    unlock_start_threshold: float = 0.5   # Start unlocking at 50% prerequisite mastery
    unlock_full_threshold: float = 0.9    # Fully unlock at 90% prerequisite mastery

    # Root concepts (no prerequisites) handling
    always_allow_roots: bool = True

    # Mastered concept handling
    allow_mastered_review: bool = True      # Allow reviewing already mastered concepts
    mastered_threshold: float = 0.95        # Consider mastered above this

    # Logits masking
    invalid_logit_penalty: float = -1e9     # Large negative value (effectively -∞)

    def to_dict(self) -> Dict:
        return {
            "prerequisite_threshold": self.prerequisite_threshold,
            "masking_mode": self.masking_mode.value,
            "soft_mask_temperature": self.soft_mask_temperature,
            "min_mask_value": self.min_mask_value,
            "unlock_start_threshold": self.unlock_start_threshold,
            "unlock_full_threshold": self.unlock_full_threshold,
            "always_allow_roots": self.always_allow_roots,
            "allow_mastered_review": self.allow_mastered_review,
            "mastered_threshold": self.mastered_threshold,
            "invalid_logit_penalty": self.invalid_logit_penalty,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ActionMaskerConfig":
        d = d.copy()
        if "masking_mode" in d:
            d["masking_mode"] = MaskingMode(d["masking_mode"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class PrerequisiteGraph:
    """
    Represents the prerequisite relationships between concepts.

    Structure: Directed Acyclic Graph (DAG)
    - Nodes: Concepts
    - Edges: (u, v) means u is a prerequisite for v

    Provides efficient queries for:
    - Getting prerequisites for a concept
    - Getting dependents of a concept
    - Checking if prerequisites are satisfied
    - Topological ordering for curriculum
    """

    def __init__(self):
        """Initialize empty prerequisite graph"""
        # Adjacency lists
        self._prerequisites: Dict[str, Set[str]] = {}  # concept -> set of prerequisites
        self._dependents: Dict[str, Set[str]] = {}     # concept -> set of dependents
        self._all_concepts: Set[str] = set()

        # Cached topological order
        self._topo_order: Optional[List[str]] = None
        self._order_dirty: bool = True

        # Concept metadata
        self._concept_metadata: Dict[str, Dict] = {}

    def add_concept(
        self,
        concept_id: str,
        metadata: Optional[Dict] = None
    ):
        """
        Add a concept to the graph.

        Args:
            concept_id: Concept identifier
            metadata: Optional metadata (difficulty, importance, etc.)
        """
        self._all_concepts.add(concept_id)
        if concept_id not in self._prerequisites:
            self._prerequisites[concept_id] = set()
        if concept_id not in self._dependents:
            self._dependents[concept_id] = set()
        if metadata:
            self._concept_metadata[concept_id] = metadata
        self._order_dirty = True

    def add_prerequisite(
        self,
        prerequisite_id: str,
        dependent_id: str,
        confidence: float = 1.0
    ):
        """
        Add a prerequisite relationship.

        Args:
            prerequisite_id: The prerequisite concept
            dependent_id: The concept that requires the prerequisite
            confidence: Confidence in this relationship (for soft constraints)
        """
        # Ensure both concepts exist
        self.add_concept(prerequisite_id)
        self.add_concept(dependent_id)

        # Add edge
        self._prerequisites[dependent_id].add(prerequisite_id)
        self._dependents[prerequisite_id].add(dependent_id)
        self._order_dirty = True

    def get_prerequisites(self, concept_id: str) -> Set[str]:
        """
        Get immediate prerequisites for a concept.

        Args:
            concept_id: Concept to query

        Returns:
            Set of prerequisite concept IDs
        """
        return self._prerequisites.get(concept_id, set()).copy()

    def get_all_prerequisites(self, concept_id: str) -> Set[str]:
        """
        Get all prerequisites (transitive closure).

        Args:
            concept_id: Concept to query

        Returns:
            Set of all prerequisite concept IDs (recursive)
        """
        all_prereqs = set()
        to_process = list(self.get_prerequisites(concept_id))

        while to_process:
            prereq = to_process.pop()
            if prereq not in all_prereqs:
                all_prereqs.add(prereq)
                to_process.extend(self.get_prerequisites(prereq))

        return all_prereqs

    def get_dependents(self, concept_id: str) -> Set[str]:
        """
        Get concepts that depend on this one.

        Args:
            concept_id: Concept to query

        Returns:
            Set of dependent concept IDs
        """
        return self._dependents.get(concept_id, set()).copy()

    def is_root(self, concept_id: str) -> bool:
        """Check if concept has no prerequisites (root node)"""
        return len(self._prerequisites.get(concept_id, set())) == 0

    def get_roots(self) -> Set[str]:
        """Get all root concepts (no prerequisites)"""
        return {c for c in self._all_concepts if self.is_root(c)}

    def get_topological_order(self) -> List[str]:
        """
        Get concepts in topological order (prerequisites before dependents).

        Returns:
            List of concept IDs in valid learning order
        """
        if not self._order_dirty and self._topo_order is not None:
            return self._topo_order.copy()

        # Kahn's algorithm for topological sort
        in_degree = {c: len(self._prerequisites.get(c, set())) for c in self._all_concepts}
        queue = [c for c, d in in_degree.items() if d == 0]
        order = []

        while queue:
            concept = queue.pop(0)
            order.append(concept)

            for dependent in self.get_dependents(concept):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Check for cycles
        if len(order) != len(self._all_concepts):
            raise ValueError("Cycle detected in prerequisite graph!")

        self._topo_order = order
        self._order_dirty = False
        return order.copy()

    def get_concept_order(self) -> List[str]:
        """Get ordered list of all concepts"""
        return self.get_topological_order()

    def num_concepts(self) -> int:
        """Get number of concepts"""
        return len(self._all_concepts)

    @classmethod
    def from_edges(
        cls,
        edges: List[Tuple[str, str]],
        concepts: Optional[List[str]] = None
    ) -> "PrerequisiteGraph":
        """
        Create graph from edge list.

        Args:
            edges: List of (prerequisite_id, dependent_id) tuples
            concepts: Optional list of all concepts

        Returns:
            PrerequisiteGraph instance
        """
        graph = cls()

        # Add all concepts first
        if concepts:
            for concept in concepts:
                graph.add_concept(concept)

        # Add edges
        for prereq, dependent in edges:
            graph.add_prerequisite(prereq, dependent)

        return graph

    @classmethod
    def from_adjacency_dict(
        cls,
        adj_dict: Dict[str, List[str]]
    ) -> "PrerequisiteGraph":
        """
        Create graph from adjacency dictionary.

        Args:
            adj_dict: Dict mapping concept -> list of prerequisites

        Returns:
            PrerequisiteGraph instance
        """
        graph = cls()

        for concept, prerequisites in adj_dict.items():
            graph.add_concept(concept)
            for prereq in prerequisites:
                graph.add_prerequisite(prereq, concept)

        return graph

    def to_dict(self) -> Dict:
        """Serialize graph to dictionary"""
        return {
            "concepts": list(self._all_concepts),
            "prerequisites": {k: list(v) for k, v in self._prerequisites.items()},
            "metadata": self._concept_metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PrerequisiteGraph":
        """Deserialize graph from dictionary"""
        graph = cls()

        for concept in d.get("concepts", []):
            metadata = d.get("metadata", {}).get(concept)
            graph.add_concept(concept, metadata)

        for concept, prereqs in d.get("prerequisites", {}).items():
            for prereq in prereqs:
                graph.add_prerequisite(prereq, concept)

        return graph


class ActionMasker:
    """
    Computes dynamic action masks for Curriculum RL.

    Given:
    - Prerequisite graph G
    - Current mastery beliefs b_t

    Computes mask M_t ∈ {0,1}^K where:
    - M_t[k] = 1 if concept k is valid to practice
    - M_t[k] = 0 if prerequisites not met

    Usage:
        masker = ActionMasker(config, graph)
        mask = masker.compute_mask(mastery_vector)
        masked_logits = masker.apply_mask_to_logits(logits, mask)
    """

    def __init__(
        self,
        config: Optional[ActionMaskerConfig] = None,
        prerequisite_graph: Optional[PrerequisiteGraph] = None
    ):
        """
        Initialize action masker.

        Args:
            config: Masking configuration
            prerequisite_graph: Graph of prerequisite relationships
        """
        self.config = config or ActionMaskerConfig()
        self.graph = prerequisite_graph or PrerequisiteGraph()

    def set_prerequisite_graph(self, graph: PrerequisiteGraph):
        """Set or update the prerequisite graph"""
        self.graph = graph

    def _check_prerequisites_satisfied(
        self,
        concept_id: str,
        masteries: Dict[str, float]
    ) -> Tuple[bool, float]:
        """
        Check if prerequisites for a concept are satisfied.

        Args:
            concept_id: Concept to check
            masteries: Dict mapping concept_id to mastery probability

        Returns:
            (is_satisfied, satisfaction_score)
        """
        prerequisites = self.graph.get_prerequisites(concept_id)

        if not prerequisites:
            # Root concept - always satisfied
            return True, 1.0

        # Get mastery values for prerequisites
        prereq_masteries = [
            masteries.get(prereq, 0.0)
            for prereq in prerequisites
        ]

        if not prereq_masteries:
            return True, 1.0

        # Minimum prerequisite mastery
        min_mastery = min(prereq_masteries)
        avg_mastery = sum(prereq_masteries) / len(prereq_masteries)

        # Check based on mode
        if self.config.masking_mode == MaskingMode.STRICT:
            is_satisfied = min_mastery >= self.config.prerequisite_threshold
            score = 1.0 if is_satisfied else 0.0

        elif self.config.masking_mode == MaskingMode.SOFT:
            # Soft threshold - satisfaction score based on how close to threshold
            score = min(1.0, avg_mastery / self.config.prerequisite_threshold)
            is_satisfied = score >= self.config.min_mask_value

        else:  # PROGRESSIVE
            # Gradually unlock as prerequisites improve
            if avg_mastery >= self.config.unlock_full_threshold:
                score = 1.0
            elif avg_mastery >= self.config.unlock_start_threshold:
                # Linear interpolation between start and full thresholds
                progress = (avg_mastery - self.config.unlock_start_threshold) / (
                    self.config.unlock_full_threshold - self.config.unlock_start_threshold
                )
                score = progress
            else:
                score = 0.0
            is_satisfied = score > 0

        return is_satisfied, score

    def compute_mask(
        self,
        masteries: Dict[str, float],
        concept_order: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Compute binary action mask based on current masteries.

        Args:
            masteries: Dict mapping concept_id to mastery probability
            concept_order: Optional ordered list of concepts (uses graph order if None)

        Returns:
            Binary mask array M ∈ {0,1}^K
        """
        if concept_order is None:
            concept_order = self.graph.get_concept_order()

        K = len(concept_order)
        mask = np.zeros(K, dtype=np.float32)

        for i, concept_id in enumerate(concept_order):
            is_valid, score = self._check_prerequisites_satisfied(concept_id, masteries)

            if self.config.masking_mode == MaskingMode.STRICT:
                mask[i] = 1.0 if is_valid else 0.0

            elif self.config.masking_mode == MaskingMode.SOFT:
                # Soft mask value
                mask[i] = max(self.config.min_mask_value, score)

            else:  # PROGRESSIVE
                mask[i] = score

            # Handle mastered concepts
            if not self.config.allow_mastered_review:
                current_mastery = masteries.get(concept_id, 0.0)
                if current_mastery >= self.config.mastered_threshold:
                    mask[i] = 0.0  # Don't review mastered concepts

        return mask

    def compute_mask_from_belief_vector(
        self,
        mastery_vector: np.ndarray,
        concept_order: List[str]
    ) -> np.ndarray:
        """
        Compute mask from numpy mastery vector.

        Args:
            mastery_vector: Array of mastery probabilities
            concept_order: Ordered list of concept IDs

        Returns:
            Binary mask array
        """
        masteries = dict(zip(concept_order, mastery_vector))
        return self.compute_mask(masteries, concept_order)

    def apply_mask_to_logits(
        self,
        logits: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply mask to logits for softmax.

        Adds large negative penalty to invalid actions:
        z̃ = z + (1 - M) * penalty

        Args:
            logits: Raw logits from RL model (K,) or (batch, K)
            mask: Binary mask (K,) or (batch, K)

        Returns:
            Masked logits
        """
        # Compute penalty for invalid actions
        penalty = (1 - mask) * self.config.invalid_logit_penalty

        return logits + penalty

    def get_valid_actions(
        self,
        masteries: Dict[str, float],
        concept_order: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get list of valid actions (concepts that can be practiced).

        Args:
            masteries: Dict mapping concept_id to mastery probability
            concept_order: Optional ordered list of concepts

        Returns:
            List of valid concept IDs
        """
        if concept_order is None:
            concept_order = self.graph.get_concept_order()

        mask = self.compute_mask(masteries, concept_order)
        valid = [
            concept_order[i]
            for i in range(len(concept_order))
            if mask[i] > 0.5  # For soft masks, use 0.5 threshold
        ]
        return valid

    def get_invalid_actions(
        self,
        masteries: Dict[str, float],
        concept_order: Optional[List[str]] = None
    ) -> List[Tuple[str, Set[str]]]:
        """
        Get invalid actions with reasons (unmet prerequisites).

        Args:
            masteries: Dict mapping concept_id to mastery probability
            concept_order: Optional ordered list of concepts

        Returns:
            List of (concept_id, unmet_prerequisites) tuples
        """
        if concept_order is None:
            concept_order = self.graph.get_concept_order()

        invalid = []
        for concept_id in concept_order:
            is_satisfied, _ = self._check_prerequisites_satisfied(concept_id, masteries)
            if not is_satisfied:
                # Find which prerequisites are unmet
                prerequisites = self.graph.get_prerequisites(concept_id)
                unmet = {
                    p for p in prerequisites
                    if masteries.get(p, 0.0) < self.config.prerequisite_threshold
                }
                invalid.append((concept_id, unmet))

        return invalid

    def sample_valid_action(
        self,
        logits: np.ndarray,
        mask: np.ndarray,
        temperature: float = 1.0,
        deterministic: bool = False
    ) -> int:
        """
        Sample a valid action from masked logits.

        Args:
            logits: Raw logits (K,)
            mask: Binary mask (K,)
            temperature: Softmax temperature
            deterministic: If True, return argmax instead of sampling

        Returns:
            Index of selected action
        """
        # Apply mask
        masked_logits = self.apply_mask_to_logits(logits, mask)

        if deterministic:
            return int(np.argmax(masked_logits))

        # Apply temperature
        scaled_logits = masked_logits / temperature

        # Compute probabilities (stable softmax)
        max_logit = np.max(scaled_logits)
        exp_logits = np.exp(scaled_logits - max_logit)
        probabilities = exp_logits / np.sum(exp_logits)

        # Handle numerical issues
        probabilities = np.clip(probabilities, 0, 1)
        probabilities = probabilities / np.sum(probabilities)

        # Sample
        return int(np.random.choice(len(probabilities), p=probabilities))

    def get_mask_statistics(
        self,
        masteries: Dict[str, float],
        concept_order: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about current mask state.

        Args:
            masteries: Dict mapping concept_id to mastery probability
            concept_order: Optional ordered list of concepts

        Returns:
            Dictionary with mask statistics
        """
        if concept_order is None:
            concept_order = self.graph.get_concept_order()

        mask = self.compute_mask(masteries, concept_order)
        valid_actions = self.get_valid_actions(masteries, concept_order)
        invalid_actions = self.get_invalid_actions(masteries, concept_order)

        return {
            "total_concepts": len(concept_order),
            "valid_count": len(valid_actions),
            "invalid_count": len(invalid_actions),
            "valid_ratio": len(valid_actions) / len(concept_order) if concept_order else 0,
            "valid_concepts": valid_actions,
            "invalid_with_reasons": [
                {"concept": c, "unmet_prerequisites": list(p)}
                for c, p in invalid_actions
            ],
            "mask_sum": float(np.sum(mask)),
            "mask_mean": float(np.mean(mask)),
        }


async def load_prerequisite_graph_from_db(
    graph_service: Any,
    course_id: int
) -> PrerequisiteGraph:
    """
    Load prerequisite graph from database (Apache AGE/Neo4j).

    This is an async helper function to integrate with NerdLearn's
    graph service.

    Args:
        graph_service: AsyncGraphService instance
        course_id: Course to load graph for

    Returns:
        PrerequisiteGraph populated with course prerequisites
    """
    # Get course graph data
    graph_data = await graph_service.get_course_graph(course_id)

    graph = PrerequisiteGraph()

    # Add concepts (nodes)
    for node in graph_data.get("nodes", []):
        graph.add_concept(
            concept_id=str(node["id"]),
            metadata={
                "name": node.get("label", ""),
                "difficulty": node.get("difficulty", 5),
                "importance": node.get("importance", 1.0),
            }
        )

    # Add prerequisites (edges)
    for edge in graph_data.get("edges", []):
        if edge.get("type") == "PREREQUISITE_FOR":
            graph.add_prerequisite(
                prerequisite_id=str(edge["source"]),
                dependent_id=str(edge["target"]),
                confidence=edge.get("confidence", 1.0)
            )

    return graph
