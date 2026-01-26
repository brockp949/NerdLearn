import logging
import math
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class ActiveLearningModule:
    """
    Selects optimal A/B tests (interventions) to resolve causal ambiguity in the prerequisite graph.
    Target ambiguous edges (Markov Equivalence Class) where direction is uncertain.
    """
    
    def __init__(self):
        pass

    def calculate_edge_entropy(self, confidence: float) -> float:
        """
        Calculate entropy (uncertainty) of an edge direction.
        Confidence = P(A->B).
        Entropy is maximized when P=0.5 (Maximum ambiguity).
        
        H(p) = -p*log2(p) - (1-p)*log2(1-p)
        
        Args:
            confidence: Probability of edge existing in a specific direction.
                        If correlation exists but direction is unknown, confidence might be 0.5.
        """
        if confidence <= 0 or confidence >= 1:
            return 0.0
            
        return -confidence * math.log2(confidence) - (1 - confidence) * math.log2(1 - confidence)

    def recommend_experiments(self, candidate_edges: List[Dict[str, Any]], max_experiments: int = 5) -> List[Dict[str, Any]]:
        """
        Rank potential experiments based on Information Gain (Entropy Reduction) vs Cost.
        
        Args:
            candidate_edges: List of edges with 'confidence' score (from Bootstrap).
                             Focus on edges with 'hypothetical' status or mid-range confidence (0.4-0.6).
                             
        Returns:
            List of recommended experiments: {target_edge, reason, expected_gain}
        """
        recommendations = []
        
        for edge in candidate_edges:
            # We are interested in edges where direction is ambiguous.
            # If we have A->B with conf 0.5, it likely means A-B correlates but direction is unclear.
            # (In bootstrap, 50% times found A->B).
            
            conf = edge.get('confidence', 0.5)
            
            # Filter for ambiguity
            if 0.3 <= conf <= 0.7:
                entropy = self.calculate_edge_entropy(conf)
                
                # Cost heuristic: 
                # Intervening on "Fundamental" nodes (low ID/early in curriculum) is high cost/high impact.
                # Intervening on leaf nodes is low cost.
                # For now, uniform cost. 
                
                # Expected Gain = Entropy (approximate). Resolving it brings entropy to 0.
                
                recommendations.append({
                    "source": edge['source'],
                    "target": edge['target'],
                    "current_confidence": conf,
                    "entropy": entropy,
                    "priority": entropy  # Rank by uncertainty
                })
                
        # Sort by priority (Entropy desc)
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        
        return recommendations[:max_experiments]
