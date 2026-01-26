import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional

try:
    from causallearn.search.ScoreBased.NOTEARS import notears_linear
except ImportError:
    notears_linear = None

logger = logging.getLogger(__name__)

class NotearsAlgorithm:
    """
    Wrapper for the NOTEARS (Non-combinatorial Optimization via Trace Exponential
    and Augmented lagRangian Structure Learning) algorithm.
    """
    
    def __init__(self, lambda1: float = 0.1, loss_type: str = 'l2'):
        """
        Initialize NOTEARS.
        
        Args:
            lambda1: L1 penalty parameter (sparsity)
            loss_type: Loss function type ('l2', 'logistic', 'poisson')
        """
        self.lambda1 = lambda1
        self.loss_type = loss_type
        
        if notears_linear is None:
            logger.warning("causal-learn not installed. NOTEARS will fail if run.")

    def run(self, data: pd.DataFrame, threshold: float = 0.3) -> List[Tuple[str, str, float]]:
        """
        Run NOTEARS on the provided data.
        
        Args:
            data: DataFrame with concepts as columns and user mastery as rows.
                  Must be numeric.
            threshold: Minimum edge weight to consider a causal link.
            
        Returns:
            List of (source, target, weight) tuples.
        """
        if notears_linear is None:
            raise ImportError("causal-learn is required for NOTEARS")

        if data.empty:
            logger.warning("Empty data provided to NOTEARS")
            return []

        # Convert to numpy array
        X = data.values
        labels = data.columns.tolist()
        
        logger.info(f"Running NOTEARS on {X.shape[0]} samples with {X.shape[1]} variables")
        
        # Run NOTEARS
        # W is the weighted adjacency matrix
        W = notears_linear(X, lambda1=self.lambda1, loss_type=self.loss_type)
        
        # Extract edges
        edges = []
        rows, cols = W.shape
        
        for i in range(rows):
            for j in range(cols):
                weight = W[i, j]
                if abs(weight) >= threshold and i != j:
                    source = labels[i]
                    target = labels[j]
                    edges.append((source, target, float(weight)))
                    
        logger.info(f"NOTEARS discovered {len(edges)} edges")
        return edges
