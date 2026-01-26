import logging
import igraph as ig
import leidenalg
import pandas as pd
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class LeidenAlgorithm:
    """
    Wrapper for the Leiden community detection algorithm.
    """
    
    def __init__(self, resolution_parameter: float = 1.0, n_iterations: int = 2):
        self.resolution_parameter = resolution_parameter
        self.n_iterations = n_iterations

    def detect_communities(self, edges: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Detect communities from a list of edges.
        
        Args:
            edges: List of dicts with 'source', 'target', and optional 'weight'.
            
        Returns:
            Dict mapping node names to community IDs.
        """
        if not edges:
            return {}

        # Build graph
        sources = [e['source'] for e in edges]
        targets = [e['target'] for e in edges]
        weights = [e.get('weight', 1.0) for e in edges]
        
        # Create unique node list to map to indices
        unique_nodes = list(set(sources + targets))
        node_to_idx = {node: i for i, node in enumerate(unique_nodes)}
        
        # Create igraph
        g = ig.Graph(directed=True)
        g.add_vertices(len(unique_nodes))
        
        edge_list = [(node_to_idx[s], node_to_idx[t]) for s, t in zip(sources, targets)]
        g.add_edges(edge_list)
        g.es['weight'] = weights
        
        # multiple edges or loops could be an issue, simplify?
        # Leiden generally handles it, but self-loops might be ignored.
        
        logger.info(f"Running Leiden algorithm on graph with {len(unique_nodes)} nodes and {len(edges)} edges")
        
        try:
            # Use CPMVertexPartition or ModularityVertexPartition
            # "Causal Discovery for Educational Graphs.pdf" mentions Leiden but not specific partition type.
            # Modularity is standard.
            partition = leidenalg.find_partition(
                g, 
                leidenalg.ModularityVertexPartition,
                weights=g.es['weight'],
                n_iterations=self.n_iterations
            )
            
            results = {}
            for i, cluster_idx in enumerate(partition.membership):
                node_name = unique_nodes[i]
                results[node_name] = cluster_idx
                
            num_communities = len(set(partition.membership))
            logger.info(f"Detected {num_communities} communities")
            
            return results
            
        except Exception as e:
            logger.error(f"Leiden detection failed: {e}")
            return {}
