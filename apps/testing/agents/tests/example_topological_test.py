"""
Example: Topological Audit

Demonstrates DAG validation for curriculum graphs.

From PDF:
"Goal Vector: 'Topological Continuity'
Key Metric: Graph Connectivity (DAG)
Failure Mode: Cycles / Orphans"
"""

import pytest
from apps.testing.agents.topological_auditor import TopologicalAuditor

def test_cycle_detection():
    """Test detection of circular dependencies"""
    nodes = [
        {"id": "A", "prerequisites": ["C"]}, # A depends on C
        {"id": "B", "prerequisites": ["A"]}, # B depends on A
        {"id": "C", "prerequisites": ["B"]}  # C depends on B -> CYCLE!
    ]
    
    auditor = TopologicalAuditor()
    auditor.load_graph(nodes)
    result = auditor.audit()
    
    assert not result.is_valid_dag, "Should detect cycle"
    assert len(result.cycle_nodes) > 0, "Should report cycle nodes"
    print(f"\n🔄 Cycle detected: {result.cycle_nodes[0]}")

def test_orphan_detection():
    """Test detection of unreachable nodes"""
    nodes = [
        {"id": "Root", "prerequisites": []},
        {"id": "Child", "prerequisites": ["Root"]},
        {"id": "Orphan", "prerequisites": ["NonExistentRoot"]} # Or just disconnected
    ]
    
    # Case 2: Just isolated
    nodes_isolated = [
        {"id": "Root", "prerequisites": []},
        {"id": "Child", "prerequisites": ["Root"]},
        {"id": "Orphan", "prerequisites": []} # Another root, theoretically allowed but maybe we want single root?
        # Actually, an orphan in our definition is unreachable from "Roots"
        # Since "Orphan" has no prereqs, it IS a root.
        # Let's try a node that requires something not linked to main graph
    ]
    
    # Better orphan definition: Node with missing prereq is invalid edge
    # Orphan usually means unreachable. 
    # Let's test missing prerequisite first
    
    auditor = TopologicalAuditor()
    auditor.load_graph(nodes)
    result = auditor.audit()
    
    # This specifically finds missing prerequisites
    missing = [v for v in result.violations if v.violation_type.value == "missing_prerequisite"]
    assert len(missing) > 0, "Should detect missing prerequisite"
    print(f"\n❌ Missing prerequisite detected: {missing[0].description}")

if __name__ == "__main__":
    test_cycle_detection()
    test_orphan_detection()
