import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional

try:
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.GraphUtils import GraphUtils
except ImportError:
    fci = None

logger = logging.getLogger(__name__)

class FciAlgorithm:
    """
    Wrapper for the FCI (Fast Causal Inference) algorithm.
    Used for discovering causal structures in the presence of latent confounders.
    """
    
    def __init__(self, alpha: float = 0.05, independence_test: str = 'fisherz'):
        """
        Initialize FCI.
        
        Args:
            alpha: Significance level for independence tests.
            independence_test: Name of the test ('fisherz', 'chisq', etc.)
        """
        self.alpha = alpha
        self.independence_test = independence_test
        
        if fci is None:
            logger.warning("causal-learn not installed. FCI will fail if run.")

    def run(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Run FCI on the provided data.
        
        Args:
            data: DataFrame with concepts as columns and user mastery as rows.
            
        Returns:
            List of edges with their types (directed, bi-directed, etc.)
            
            Edge types in PAG (Partial Ancestral Graph):
            0: No edge
            1: Circle (o) - Uncertain orientation
            2: Arrowhead (>) - Ancestral relationship
            3: Tail (-)
            
            X o-o Y : 1, 1
            X o-> Y : 1, 2
            X <-> Y : 2, 2 (Bi-directed, confounding)
            X --> Y : 3, 2 (Directed)
            X --- Y : 3, 3 (Undirected, selection bias)
        """
        if fci is None:
            raise ImportError("causal-learn is required for FCI")

        if data.empty:
            logger.warning("Empty data provided to FCI")
            return []

        # Convert to numpy array
        data_np = data.values
        labels = data.columns.tolist()
        
        logger.info(f"Running FCI on {data_np.shape[0]} samples with {data_np.shape[1]} variables")
        
        # Run FCI
        # G is a GeneralGraph object
        # edges is a list of Edge objects
        G, edges = fci(data_np, alpha=self.alpha, independence_test_method=self.independence_test)
        
        results = []
        
        # Process edges using the graph object nodes to map back to labels
        nodes = G.nodes
        
        for edge in edges:
            # node1 and node2 are Node objects, strictly speaking we rely on indices or names if mapped
            # causal-learn nodes usuall have name 'X1', 'X2' etc corresponding to column index
            
            # We can map back by index if the nodes are ordered 0..N-1
            # Let's verify node names. usually they are like "X1".
            # We will use the node indices.
            
            # The Edge object endpoints are Nodes.
            
            # We need to find the index of the nodes in the list
            # G.nodes is a list of nodes.
            try:
                i = nodes.index(edge.get_node1())
                j = nodes.index(edge.get_node2())
            except ValueError:
                continue
                
            source_label = labels[i]
            target_label = labels[j]
            
            # Determine edge type
            # endpoint1 is shape at node1, endpoint2 is shape at node2
            # shapes: 0=NULL, 1=CIRCLE, 2=ARROW, 3=TAIL
            
            # We want to identify specific structures
            
            edge_info = {
                "source_index": i,
                "target_index": j,
                "source": source_label,
                "target": target_label,
                "endpoint1": self._shape_to_str(edge.get_endpoint1()),
                "endpoint2": self._shape_to_str(edge.get_endpoint2()),
                "type": "unknown"
            }
            
            # Interpret
            e1 = edge.get_endpoint1() # at source
            e2 = edge.get_endpoint2() # at target
            
            if e1 == 3 and e2 == 2: # Tail at source, Arrow at target: Source -> Target
                edge_info["type"] = "directed"
                edge_info["relation"] = f"{source_label} -> {target_label}"
                
            elif e1 == 2 and e2 == 3: # Arrow at source, Tail at target: Target -> Source
                edge_info["type"] = "directed"
                edge_info["relation"] = f"{target_label} -> {source_label}"
                # Swap for consistency
                edge_info["source"], edge_info["target"] = target_label, source_label
                edge_info["source_index"], edge_info["target_index"] = j, i
                edge_info["endpoint1"], edge_info["endpoint2"] = "TAIL", "ARROW"
                
            elif e1 == 2 and e2 == 2: # Arrow at both: Source <-> Target (Confounded)
                edge_info["type"] = "bi-directed"
                edge_info["relation"] = f"{source_label} <-> {target_label}"
                edge_info["is_confounded"] = True
                
            elif e1 == 1 or e2 == 1: # Circle involved
                edge_info["type"] = "uncertain"
                edge_info["relation"] = f"{source_label} o-o {target_label}" # Simplified
                
            results.append(edge_info)
            
        logger.info(f"FCI discovered {len(results)} edges")
        return results

    def _shape_to_str(self, shape_id: int) -> str:
        mapping = {0: "NULL", 1: "CIRCLE", 2: "ARROW", 3: "TAIL"}
        return mapping.get(shape_id, "UNKNOWN")
