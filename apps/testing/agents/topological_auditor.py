"""
Topological Auditor - DAG Validation for Curriculum Graphs

Implements topological validation from "Building Code Testing Agents" PDF.

From PDF:
"The Architect - Goal Vector: 'Topological Continuity'
Key Metric: Graph Connectivity (DAG)
Failure Mode: Cycles / Orphans"

This agent validates:
- Curriculum graphs are valid DAGs (no cycles)
- No orphaned nodes (unreachable content)
- Prerequisite chains are complete
- Learning paths follow proper topology
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class TopologyViolation(Enum):
    """Types of topological violations"""
    CYCLE = "cycle"                    # Circular dependency
    ORPHAN = "orphan"                  # Unreachable node
    MISSING_PREREQUISITE = "missing_prerequisite"
    ISOLATED_COMPONENT = "isolated"    # Disconnected subgraph
    INVALID_EDGE = "invalid_edge"      # Edge to non-existent node


@dataclass
class TopologyNode:
    """A node in the curriculum graph"""
    id: str
    name: str
    prerequisites: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopologyViolationReport:
    """Report of a topology violation"""
    violation_type: TopologyViolation
    nodes_involved: List[str]
    description: str
    severity: str = "high"  # low, medium, high, critical
    suggested_fix: Optional[str] = None


@dataclass
class TopologyAuditResult:
    """Result of a topological audit"""
    is_valid_dag: bool
    violations: List[TopologyViolationReport] = field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0
    max_depth: int = 0
    orphan_nodes: List[str] = field(default_factory=list)
    cycle_nodes: List[List[str]] = field(default_factory=list)
    connectivity_score: float = 1.0  # 1.0 = fully connected, 0.0 = fragmented
    
    def summary(self) -> str:
        """Human-readable summary"""
        status = "✅ Valid" if self.is_valid_dag else "❌ Invalid"
        return (
            f"{status} DAG | "
            f"{self.node_count} nodes, {self.edge_count} edges | "
            f"Depth: {self.max_depth} | "
            f"Violations: {len(self.violations)} | "
            f"Connectivity: {self.connectivity_score:.0%}"
        )


class TopologicalAuditor:
    """
    Validates curriculum/knowledge graphs for topological integrity.
    
    Ensures:
    1. No cycles (valid DAG)
    2. No orphaned nodes
    3. Complete prerequisite chains
    4. Proper learning path structure
    
    From PDF:
    "No PR should merge without a 'Topological Audit.'"
    """
    
    def __init__(self):
        self.nodes: Dict[str, TopologyNode] = {}
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)  # node -> dependents
        self.reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)  # node -> prerequisites
    
    def load_graph(self, nodes: List[Dict[str, Any]]):
        """
        Load a graph from node definitions.
        
        Args:
            nodes: List of node dicts with 'id', 'name', 'prerequisites'
        """
        self.nodes.clear()
        self.adjacency.clear()
        self.reverse_adjacency.clear()
        
        for node_data in nodes:
            node = TopologyNode(
                id=node_data['id'],
                name=node_data.get('name', node_data['id']),
                prerequisites=node_data.get('prerequisites', []),
                metadata=node_data.get('metadata', {})
            )
            self.nodes[node.id] = node
            
            # Build adjacency
            for prereq in node.prerequisites:
                self.adjacency[prereq].add(node.id)
                self.reverse_adjacency[node.id].add(prereq)
    
    def audit(self) -> TopologyAuditResult:
        """
        Perform comprehensive topological audit.
        
        Returns:
            TopologyAuditResult with all findings
        """
        logger.info(f"Starting topological audit of {len(self.nodes)} nodes")
        
        violations = []
        
        # Check for cycles
        cycles = self._find_cycles()
        for cycle in cycles:
            violations.append(TopologyViolationReport(
                violation_type=TopologyViolation.CYCLE,
                nodes_involved=cycle,
                description=f"Circular dependency: {' -> '.join(cycle)} -> {cycle[0]}",
                severity="critical",
                suggested_fix="Remove one edge to break the cycle"
            ))
        
        # Check for orphans (nodes with no path from roots)
        orphans = self._find_orphans()
        for orphan in orphans:
            violations.append(TopologyViolationReport(
                violation_type=TopologyViolation.ORPHAN,
                nodes_involved=[orphan],
                description=f"Orphaned node '{orphan}' is unreachable from any root",
                severity="high",
                suggested_fix="Add prerequisite connection or mark as root"
            ))
        
        # Check for missing prerequisites
        missing = self._find_missing_prerequisites()
        for node_id, missing_prereq in missing:
            violations.append(TopologyViolationReport(
                violation_type=TopologyViolation.MISSING_PREREQUISITE,
                nodes_involved=[node_id, missing_prereq],
                description=f"Node '{node_id}' references non-existent prerequisite '{missing_prereq}'",
                severity="critical",
                suggested_fix=f"Create node '{missing_prereq}' or remove the reference"
            ))
        
        # Calculate metrics
        edge_count = sum(len(deps) for deps in self.adjacency.values())
        max_depth = self._calculate_max_depth()
        connectivity = self._calculate_connectivity()
        
        is_valid = len(cycles) == 0 and len(missing) == 0
        
        result = TopologyAuditResult(
            is_valid_dag=is_valid,
            violations=violations,
            node_count=len(self.nodes),
            edge_count=edge_count,
            max_depth=max_depth,
            orphan_nodes=orphans,
            cycle_nodes=cycles,
            connectivity_score=connectivity
        )
        
        logger.info(f"Audit complete: {result.summary()}")
        return result
    
    def _find_cycles(self) -> List[List[str]]:
        """Find all cycles using DFS"""
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.adjacency.get(node, set()):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:].copy())
                    return True
            
            path.pop()
            rec_stack.remove(node)
            return False
        
        for node in self.nodes:
            if node not in visited:
                dfs(node)
        
        return cycles
    
    def _find_orphans(self) -> List[str]:
        """Find nodes unreachable from any root"""
        # Roots are nodes with no prerequisites
        roots = [n for n, node in self.nodes.items() if not node.prerequisites]
        
        if not roots:
            # If no roots, all nodes are potentially orphaned
            return list(self.nodes.keys())
        
        # BFS from all roots
        reachable = set()
        queue = deque(roots)
        
        while queue:
            node = queue.popleft()
            if node in reachable:
                continue
            reachable.add(node)
            
            for dependent in self.adjacency.get(node, set()):
                if dependent not in reachable:
                    queue.append(dependent)
        
        return [n for n in self.nodes if n not in reachable]
    
    def _find_missing_prerequisites(self) -> List[Tuple[str, str]]:
        """Find references to non-existent nodes"""
        missing = []
        for node_id, node in self.nodes.items():
            for prereq in node.prerequisites:
                if prereq not in self.nodes:
                    missing.append((node_id, prereq))
        return missing
    
    def _calculate_max_depth(self) -> int:
        """Calculate maximum depth of the DAG"""
        if not self.nodes:
            return 0
            
        depths = {}
        visiting = set()  # To detect cycles during depth calculation
        
        def get_depth(node: str) -> int:
            if node in depths:
                return depths[node]
            
            if node in visiting:
                # Cycle detected during depth calculation - return 0 or handle error
                # Since we already check for cycles separately, we can just break here
                return 0
                
            visiting.add(node)
            
            prereqs = self.nodes[node].prerequisites if node in self.nodes else []
            valid_prereqs = [p for p in prereqs if p in self.nodes]
            
            if not valid_prereqs:
                depths[node] = 0
            else:
                depths[node] = 1 + max(get_depth(p) for p in valid_prereqs)
            
            visiting.remove(node)
            return depths[node]
        
        try:
            return max(get_depth(n) for n in self.nodes)
        except RecursionError:
            # Fallback if graph is extremely deep or cyclic
            return -1
    
    def _calculate_connectivity(self) -> float:
        """Calculate how well-connected the graph is"""
        if len(self.nodes) <= 1:
            return 1.0
        
        # Count connected components
        visited = set()
        components = 0
        
        def dfs(node: str):
            visited.add(node)
            # Visit both directions
            for neighbor in self.adjacency.get(node, set()):
                if neighbor not in visited and neighbor in self.nodes:
                    dfs(neighbor)
            for neighbor in self.reverse_adjacency.get(node, set()):
                if neighbor not in visited and neighbor in self.nodes:
                    dfs(neighbor)
        
        for node in self.nodes:
            if node not in visited:
                dfs(node)
                components += 1
        
        # Connectivity score: 1 component = 1.0, more components = lower score
        return 1.0 / components if components > 0 else 0.0
    
    def validate_learning_path(
        self,
        path: List[str]
    ) -> Tuple[bool, List[TopologyViolationReport]]:
        """
        Validate a specific learning path follows DAG order.
        
        Args:
            path: List of node IDs in proposed order
        
        Returns:
            (is_valid, violations)
        """
        violations = []
        completed = set()
        
        for i, node_id in enumerate(path):
            if node_id not in self.nodes:
                violations.append(TopologyViolationReport(
                    violation_type=TopologyViolation.INVALID_EDGE,
                    nodes_involved=[node_id],
                    description=f"Node '{node_id}' at position {i} does not exist",
                    severity="critical"
                ))
                continue
            
            node = self.nodes[node_id]
            for prereq in node.prerequisites:
                if prereq not in completed and prereq in self.nodes:
                    violations.append(TopologyViolationReport(
                        violation_type=TopologyViolation.MISSING_PREREQUISITE,
                        nodes_involved=[node_id, prereq],
                        description=f"'{node_id}' at position {i} requires '{prereq}' which comes later or is missing",
                        severity="high",
                        suggested_fix=f"Move '{prereq}' before '{node_id}'"
                    ))
            
            completed.add(node_id)
        
        return len(violations) == 0, violations
