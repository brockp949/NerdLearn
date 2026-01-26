"""
Test Causal Discovery Pipeline

Runs the full CausalDiscoveryManager pipeline on synthetic data.
Mocks the GraphService persistence layer.
"""

import asyncio
import logging
import sys
import os

# Add api directory to path
sys.path.append(os.path.join(os.getcwd(), "apps/api"))

from app.adaptive.causal_discovery.manager import CausalDiscoveryManager
from app.adaptive.causal_discovery.generate_synthetic_data import CausalDataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockGraphService:
    """Mock for graph persistence"""
    async def update_causal_edges(self, edges):
        logger.info(f"Mock Persist: {len(edges)} edges")
        for e in edges:
            logger.info(f"  {e['source']} -> {e['target']} ({e['type']}) conf={e.get('confidence', 0):.2f}")
        return {"persisted": len(edges), "skipped": 0}

async def run_test():
    # 1. Generate Data
    logger.info("Generating synthetic data...")
    gen = CausalDataGenerator(num_concepts=8, num_users=1000, seed=42)
    G = gen.generate_random_dag(edge_prob=0.3)
    logger.info(f"Ground Truth Edges: {list(G.edges())}")
    
    df = gen.generate_data_from_graph(G)
    mastery_data = gen.to_mastery_list(df)
    
    # 2. Run Pipeline
    manager = CausalDiscoveryManager(use_bootstrap=False)
    mock_service = MockGraphService()
    
    logger.info("Running pipeline...")
    await manager.run_discovery_pipeline(
        mastery_data=mastery_data,
        graph_service_instance=mock_service
    )
    
    # 3. Verify Results
    discovered_edges = set()
    for e in manager._last_edges:
        # Map C0 -> 0 for comparison if needed, but our generator uses C0, C1...
        # Wait, generator uses "C0", "C1". Manager output depends on algorithm.
        # NOTEARS usually returns indices if not careful, but manager preprocesses DF columns.
        # DF columns are "C0", "C1"... so NOTEARS should return those.
        src = e['source']
        tgt = e['target']
        discovered_edges.add((int(src[1:]), int(tgt[1:]))) # Parse "C0" -> 0
    
    true_edges = set(G.edges())
    
    # Calculate precision/recall
    tp = len(discovered_edges.intersection(true_edges))
    fp = len(discovered_edges - true_edges)
    fn = len(true_edges - discovered_edges)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    logger.info("="*40)
    logger.info(f"RESULTS:")
    logger.info(f"Ground Truth: {len(true_edges)} edges")
    logger.info(f"Discovered:   {len(discovered_edges)} edges")
    logger.info(f"True Positives: {tp}")
    logger.info(f"False Positives: {fp}")
    logger.info(f"Precision: {precision:.2f}")
    logger.info(f"Recall:    {recall:.2f}")
    logger.info("="*40)

if __name__ == "__main__":
    asyncio.run(run_test())
