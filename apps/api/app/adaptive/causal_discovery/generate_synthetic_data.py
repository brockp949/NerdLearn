"""
Synthetic Data Generator for Causal Discovery

Generates synthetic user mastery data based on a known causal graph (DAG).
Used for testing and verifying the Causal Discovery pipeline.

Generates:
1. Ground truth graph (adjacency matrix)
2. User mastery matrix (samples from the graph distribution)
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Optional

class CausalDataGenerator:
    def __init__(self, num_concepts: int = 10, num_users: int = 1000, seed: int = 42):
        self.num_concepts = num_concepts
        self.num_users = num_users
        self.seed = seed
        np.random.seed(seed)
        
    def generate_random_dag(self, edge_prob: float = 0.3) -> nx.DiGraph:
        """Generate a random Directed Acyclic Graph"""
        G = nx.gnp_random_graph(self.num_concepts, edge_prob, directed=True, seed=self.seed)
        G = nx.DiGraph([(u, v) for (u, v) in G.edges() if u < v]) # Enforce DAG by only allowing edges i->j where i<j
        return G
    
    def generate_data_from_graph(self, G: nx.DiGraph) -> pd.DataFrame:
        """
        Generate mastery data respecting the graph structure.
        
        Model:
        - Root nodes: Bernoulli(0.5)
        - Child nodes: P(Child=1 | Parents)
          - If all parents mastered: High prob of mastery
          - If any parent not mastered: Low prob of mastery (Prerequisite constraint)
        """
        data = np.zeros((self.num_users, self.num_concepts))
        
        # Topological sort ensures we generate parents before children
        order = list(nx.topological_sort(G))
        
        # Concept names: "C0", "C1", ...
        concepts = [f"C{i}" for i in range(self.num_concepts)]
        
        for user_idx in range(self.num_users):
            user_mastery = {}
            
            # Base aptitude for this user (some learn faster)
            aptitude = np.random.beta(2, 2) 
            
            for node in order:
                parents = list(G.predecessors(node))
                
                if not parents:
                    # Root node: Base difficulty + User aptitude
                    # E.g., easy concepts (low index) easier than hard ones? 
                    # Let's just use aptitude + noise
                    prob = aptitude
                else:
                    # Check parent mastery
                    parents_mastered = all(user_mastery[p] for p in parents)
                    
                    if parents_mastered:
                        # Prereqs met: Standard learning probability
                        prob = aptitude
                    else:
                        # Prereqs NOT met: Very low probability (Prerequisite constraint)
                        prob = 0.1 * aptitude
                
                # Sample
                user_mastery[node] = 1 if np.random.random() < prob else 0
                
            # Store row
            # Sort by index to match column order
            for i in range(self.num_concepts):
                data[user_idx, i] = user_mastery[i]
                
        df = pd.DataFrame(data, columns=concepts)
        df['user_id'] = [f"U{i}" for i in range(self.num_users)]
        
        return df

    def to_mastery_list(self, df: pd.DataFrame) -> List[Dict]:
        """Convert DataFrame to List[Dict] format expected by manager"""
        melted = df.melt(id_vars=['user_id'], var_name='concept_id', value_name='mastery')
        return melted.to_dict('records')

def main():
    gen = CausalDataGenerator(num_concepts=5, num_users=500)
    G = gen.generate_random_dag()
    print("Ground Truth Edges:", G.edges())
    
    df = gen.generate_data_from_graph(G)
    print(df.head())
    
    mastery_list = gen.to_mastery_list(df)
    print(f"Generated {len(mastery_list)} records")

if __name__ == "__main__":
    main()
