import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import timedelta

class TemporalPreprocessor:
    """
    Handles temporal unrolling of student mastery data to detect time-lagged causal relationships.
    Converts static (User, Concept, Mastery) data into time-series features.
    """
    
    def __init__(self, time_window_days: int = 7):
        self.time_window = timedelta(days=time_window_days)

    def pivot_temporal_features(self, mastery_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert log data into a dataset with lagged features.
        Concept_A_t0, Concept_B_t1
        
        Simple approach:
        For each user, sort by time.
        Create shift features.
        
        Refined approach (DBN style):
        We want to see if Mastery(A) at T affects Mastery(B) at T+1.
        
        We will organize data into 'Learning Sessions' or 'Weeks'.
        Then pivot:
        User | Week | Concept_A | Concept_B
        
        Then create lagged columns:
        Concept_A_lag1, Concept_B_lag1
        """
        if not mastery_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(mastery_data)
        
        # Ensure timestamp is datetime
        if 'timestamp' not in df.columns:
            # Fallback if no timestamp: assume sequential order in list? 
            # Or invalid for temporal analysis.
            return pd.DataFrame()
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Bin by time period (e.g. Week)
        # We need a 'period' identifier for each user
        df['period'] = df['timestamp'].dt.to_period('W')
        
        # Aggregate Max Mastery per Week
        pivot_df = df.pivot_table(
            index=['user_id', 'period'],
            columns='concept_id',
            values='mastery',
            aggfunc='max'
        ).fillna(0.0)
        
        # Now we have User-Week rows.
        # We need to create Lagged DataFrame for Causal Discovery.
        # Structure: [Concept_A_t, Concept_B_t, Concept_A_t-1, Concept_B_t-1]
        
        shifted_df = pivot_df.groupby(level=0).shift(1)
        shifted_df.columns = [f"{c}_lag1" for c in shifted_df.columns]
        
        # Combine
        # Inner join to only keep rows where we have history
        combined_df = pd.concat([pivot_df, shifted_df], axis=1).dropna()
        
        # Flatten index
        combined_df.reset_index(drop=True, inplace=True)
        
        # Ensure string columns for consistency with other modules
        combined_df.columns = combined_df.columns.astype(str)
        
        return combined_df

class SummaryGraphGenerator:
    """
    Condenses a Dynamic Bayesian Network (DBN) into a Summary Causal Graph.
    Resolves cycles by mapping temporal loops (A_t0 -> B_t1, B_t0 -> A_t1) to 
    cyclic graph edges (A <-> B or A -> B and B -> A).
    """
    
    def condense_temporal_graph(self, temporal_edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Input: edges with names like '101', '101_lag1'.
        Output: edges between '101' and '102' with lag properties.
        """
        summary_edges = []
        
        for edge in temporal_edges:
            src = edge['source'] # e.g. "101_lag1" or "101"
            tgt = edge['target'] # e.g. "102"
            
            # Parse names
            src_clean, src_lag = self._parse_node_name(src)
            tgt_clean, tgt_lag = self._parse_node_name(tgt)
            
            # We are interested in causal flow.
            # 1. Instantaneous: A_t0 -> B_t0 (lag=0)
            # 2. Lagged: A_t-1 -> B_t0 (lag=1)
            
            lag = src_lag - tgt_lag # This is tricky depending on how we named them.
            # If src is lag1 and tgt is current (lag0), then flow is t-1 -> t. This is lag 1.
            
            # Simpler logic based on names suffix
            is_src_lagged = "_lag1" in src
            is_tgt_lagged = "_lag1" in tgt
            
            # Filter out edges pointing TO the past (t -> t-1). Causal arrows go forward in time.
            if not is_src_lagged and is_tgt_lagged:
                continue
                
            real_src = src.replace("_lag1", "")
            real_tgt = tgt.replace("_lag1", "")
            
            if real_src == real_tgt:
                 # Auto-regressive edge (A_t-1 -> A_t). 
                 # Important for model fit, but maybe not for the prerequisite graph?
                 # It means "Knowing A helps you keep knowing A". Trivial.
                 continue
                 
            edge_type = "instantaneous"
            if is_src_lagged and not is_tgt_lagged:
                edge_type = "temporal"
                
            summary_edges.append({
                "source": real_src,
                "target": real_tgt,
                "type": "directed", # In summary graph, it's just a directed edge
                "weight": edge['weight'],
                "temporal_type": edge_type,
                "source_algo": "temporal_dbn"
            })
            
        return summary_edges

    def _parse_node_name(self, name: str) -> Tuple[str, int]:
        if "_lag1" in name:
            return name.replace("_lag1", ""), 1
        return name, 0
