"""
Causal Discovery Manager - Orchestrates the automated discovery pipeline

Based on "Causal Discovery for Educational Graphs" PDF specification.

Pipeline Steps:
1. Data Preprocessing: Convert mastery logs to User x Concept matrix
2. Global Discovery: NOTEARS for DAG skeleton (with optional Bootstrap)
3. Community Detection: Leiden for clustering concepts into modules
4. Local Refinement: FCI on dense subcommunities to detect confounders
5. Confidence Scoring: Bootstrap stability selection for edge confidence
6. Graph Persistence: MERGE to Apache AGE with status classification

Research basis:
- NOTEARS: Continuous optimization for structure learning (Zheng et al., 2018)
- FCI: Fast Causal Inference for latent confounders (Spirtes et al., 2000)
- Leiden: Guaranteed well-connected communities (Traag et al., 2019)
- Bootstrap Stability: Robust edge confidence (Meinshausen & Buhlmann, 2010)
"""

import logging
import asyncio
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.adaptive.causal_discovery.algorithms.notears import NotearsAlgorithm
from app.adaptive.causal_discovery.algorithms.fci import FciAlgorithm
from app.adaptive.causal_discovery.algorithms.leiden import LeidenAlgorithm
from app.adaptive.causal_discovery.temporal import TemporalPreprocessor, SummaryGraphGenerator
from app.adaptive.causal_discovery.bootstrap import BootstrapStabilitySelector

logger = logging.getLogger(__name__)


class CausalDiscoveryManager:
    """
    Orchestrates the automated causal discovery pipeline.

    Pipeline Steps:
    1. Data Preprocessing: Convert mastery logs to user-concept matrix.
    2. Global Discovery: Run NOTEARS to find skeleton DAG.
    3. Community Detection: Run Leiden to identify clusters.
    4. Local Refinement: Run FCI on dense/suspect clusters to find confounders.
    5. Graph Update: Persist discovered causal edges.

    Attributes:
        _last_edges: List of edges from the last discovery run
        _last_communities: Dict of concept -> community_id from Leiden
        _last_persist_result: Dict with persistence statistics
        _last_confidence_dist: Dict with confidence distribution
    """

    def __init__(self, use_bootstrap: bool = False, bootstrap_samples: int = 100):
        """
        Initialize the discovery manager.

        Args:
            use_bootstrap: Whether to use bootstrap stability selection by default
            bootstrap_samples: Number of bootstrap samples
        """
        self.notears = NotearsAlgorithm(lambda1=0.1)
        self.fci = FciAlgorithm(alpha=0.05)
        self.leiden = LeidenAlgorithm()
        self.temporal_preprocessor = TemporalPreprocessor()
        self.summary_generator = SummaryGraphGenerator()
        self.bootstrap = BootstrapStabilitySelector(n_bootstrap=bootstrap_samples)
        self.use_bootstrap = use_bootstrap

        # Store results for inspection/reporting (used by router endpoints)
        self._last_edges: List[Dict[str, Any]] = []
        self._last_communities: Dict[str, int] = {}
        self._last_persist_result: Dict[str, int] = {"persisted": 0, "skipped": 0}
        self._last_confidence_dist: Dict[str, int] = {"verified": 0, "hypothetical": 0, "discarded": 0}

    async def run_discovery_pipeline(
        self,
        mastery_data: List[Dict[str, Any]],
        graph_service_instance: Any,
        use_bootstrap: bool = None
    ):
        """
        Run the full discovery pipeline.

        Args:
            mastery_data: List of dicts containing 'user_id', 'concept_id', 'mastery'
            graph_service_instance: Instance of AsyncGraphService
            use_bootstrap: Override default bootstrap setting for this run
        """
        logger.info("Starting Causal Discovery Pipeline")

        # Reset results
        self._last_edges = []
        self._last_communities = {}
        self._last_persist_result = {"persisted": 0, "skipped": 0}
        self._last_confidence_dist = {"verified": 0, "hypothetical": 0, "discarded": 0}

        # 1. Preprocess Data
        df = self._preprocess_data(mastery_data)
        if df.empty or df.shape[1] < 2:
            logger.warning("Insufficient data for causal discovery")
            return

        # Determine if we should use bootstrap
        should_bootstrap = use_bootstrap if use_bootstrap is not None else self.use_bootstrap

        if should_bootstrap:
            # Run bootstrap stability selection for confidence scoring
            logger.info("Step 2: Global Discovery with Bootstrap Stability Selection")
            edges_with_confidence, confidence_dist = self.bootstrap.run(df)
            self._last_confidence_dist = confidence_dist

            # Convert to skeletal edges format
            skeletal_edges = edges_with_confidence
        else:
            # 2. Global Discovery (NOTEARS) - original approach
            logger.info("Step 2: Global Discovery (NOTEARS)")
            notears_edges = self.notears.run(df)

            # Convert to standard format
            skeletal_edges = []
            for src, tgt, weight in notears_edges:
                skeletal_edges.append({
                    "source": src,
                    "target": tgt,
                    "weight": weight,
                    "confidence": 0.7,  # Default confidence without bootstrap
                    "type": "directed",
                    "source_algo": "notears"
                })

        # 3. Community Detection (Leiden)
        logger.info("Step 3: Community Detection (Leiden)")
        communities = self.leiden.detect_communities(skeletal_edges)
        self._last_communities = communities

        # Group concepts by community
        community_concepts = {}
        for concept, comm_id in communities.items():
            if comm_id not in community_concepts:
                community_concepts[comm_id] = []
            community_concepts[comm_id].append(concept)

        # 4. Local Refinement (FCI) on dense communities
        logger.info("Step 4: Local Refinement (FCI)")
        refined_edges = []

        for comm_id, concepts in community_concepts.items():
            if len(concepts) < 3:
                continue

            # Filter data for this community - handle missing columns gracefully
            valid_concepts = [c for c in concepts if c in df.columns]
            if len(valid_concepts) < 3:
                continue

            comm_df = df[valid_concepts]

            logger.info(f"Running FCI on community {comm_id} with {len(valid_concepts)} concepts")
            fci_results = self.fci.run(comm_df)

            for res in fci_results:
                # We prioritize FCI for finding confounders (bi-directed)
                if res.get("is_confounded"):
                    refined_edges.append({
                        "source": res["source"],
                        "target": res["target"],
                        "type": "bi-directed",
                        "weight": 1.0,
                        "confidence": 0.8,  # FCI-detected confounders get high confidence
                        "source_algo": "fci"
                    })
                elif res["type"] == "directed":
                    refined_edges.append({
                        "source": res["source"],
                        "target": res["target"],
                        "type": "directed_fci",
                        "weight": 1.0,
                        "confidence": 0.75,
                        "source_algo": "fci"
                    })

        # 5. Merge and Persist
        logger.info("Step 5: Merge and Persist")
        final_edges = self._merge_results(skeletal_edges, refined_edges)
        self._last_edges = final_edges

        persist_result = await self._persist_graph(final_edges, graph_service_instance)
        self._last_persist_result = persist_result

        logger.info(
            f"Causal Discovery Pipeline Completed: "
            f"{len(final_edges)} edges discovered, "
            f"{persist_result.get('persisted', 0)} persisted, "
            f"{len(communities)} communities detected"
        )

    async def run_bootstrap_pipeline(
        self,
        mastery_data: List[Dict[str, Any]],
        graph_service_instance: Any,
        n_bootstraps: int = 100
    ):
        """
        Run the discovery pipeline with bootstrap stability selection.

        This is an alternative entry point that uses the BootstrapStabilitySelector
        for confidence scoring. Prefer using run_discovery_pipeline with use_bootstrap=True.

        Args:
            mastery_data: List of dicts containing 'user_id', 'concept_id', 'mastery'
            graph_service_instance: Instance of AsyncGraphService
            n_bootstraps: Number of bootstrap samples
        """
        logger.info(f"Starting Bootstrap Pipeline with {n_bootstraps} iterations")

        # Reset results
        self._last_edges = []
        self._last_communities = {}
        self._last_persist_result = {"persisted": 0, "skipped": 0}
        self._last_confidence_dist = {"verified": 0, "hypothetical": 0, "discarded": 0}

        df = self._preprocess_data(mastery_data)
        if df.empty or df.shape[1] < 2:
            logger.warning("Insufficient data for bootstrap discovery")
            return

        # Use the BootstrapStabilitySelector for cleaner implementation
        bootstrap_selector = BootstrapStabilitySelector(n_bootstrap=n_bootstraps)
        edges_with_confidence, confidence_dist = bootstrap_selector.run(df)

        # Store results
        self._last_edges = edges_with_confidence
        self._last_confidence_dist = confidence_dist

        # Run Leiden on the discovered edges for community detection
        communities = self.leiden.detect_communities(edges_with_confidence)
        self._last_communities = communities

        logger.info(f"Bootstrap Analysis complete. Found {len(edges_with_confidence)} stable edges.")

        # Persist
        persist_result = await self._persist_graph(edges_with_confidence, graph_service_instance)
        self._last_persist_result = persist_result

    async def detect_cycles_and_temporal_flow(self, mastery_data: List[Dict[str, Any]], graph_service_instance: Any):
        """
        Run temporal causal discovery (DBN style) to resolve cycles.
        """
        logger.info("Starting Temporal Cycle Detection")
        
        # 1. Temporal Preprocessing
        df_lagged = self.temporal_preprocessor.pivot_temporal_features(mastery_data)
        
        if df_lagged.empty or df_lagged.shape[1] < 4: # Need at least 2 vars * 2 timesteps
             logger.warning("Insufficient temporal data")
             return
             
        # 2. Run Structure Learning on Lagged Data
        # We assume causal sufficiency for simplicity here, so using NOTEARS on the larger feature set
        edges_lagged = self.notears.run(df_lagged, threshold=0.2)
        
        # 3. Condense to Summary Graph
        # Convert edge list (A_lag1 -> B) to dicts
        raw_edges = []
        for src, tgt, w in edges_lagged:
            raw_edges.append({"source": src, "target": tgt, "weight": w})
            
        summary_edges = self.summary_generator.condense_temporal_graph(raw_edges)
        
        logger.info(f"Temporal Analysis complete. Found {len(summary_edges)} edges.")
        
        # 4. Persist
        await self._persist_graph(summary_edges, graph_service_instance)

    def _preprocess_data(self, mastery_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert telemetry logs to a User x Concept mastery dataframe.
        """
        if not mastery_data:
            return pd.DataFrame()
            
        # We expect data to be flattened: user_id, concept_id, mastery
        # If multiple entries exist, take the max mastery/latest?
        # For 'prerequisite' detection, we usually want final state or average.
        # Let's take max to represent "has mastered".
        
        # Convert to DF
        raw_df = pd.DataFrame(mastery_data)
        
        # ensure columns exist
        if not {'user_id', 'concept_id', 'mastery'}.issubset(raw_df.columns):
            logger.error("Data missing required columns")
            return pd.DataFrame()

        # Pivot
        # index=user_id, columns=concept_id, values=mastery
        pivot_df = raw_df.pivot_table(
            index='user_id', 
            columns='concept_id', 
            values='mastery', 
            aggfunc='max'
        )
        
        # Fill NaNs?
        # If a user hasn't seen a concept, is it 0 mastery or missing?
        # For causal discovery, missing might be better handled, but NOTEARS usually expects complete data.
        # We will fill with 0 (unmastered).
        pivot_df = pivot_df.fillna(0.0)
        
        # Drop columns with 0 variance (everyone has 0 or everyone has 1)
        pivot_df = pivot_df.loc[:, pivot_df.var() > 0.01]
        
        # Ensure column names are strings for consistency
        pivot_df.columns = pivot_df.columns.astype(str)
        
        return pivot_df

    def _merge_results(self, skeletons: List[Dict], refinements: List[Dict]) -> List[Dict]:
        """
        Merge global skeleton with local refinements.
        FCI results (especially confounders) override NOTEARS.
        """
        # Key by source-target pair
        edge_map = {}
        
        # Add skeleton first
        for e in skeletons:
            # Sort keys for undirected comparison? Directed matters.
            key = (e['source'], e['target'])
            edge_map[key] = e
            
        # Overwrite/Add refinements
        for e in refinements:
            if e['type'] == 'bi-directed':
                # Remove directed edges between these two if they exist, replace with bi-directed
                k1 = (e['source'], e['target'])
                k2 = (e['target'], e['source'])
                
                # Bi-directed implies common cause, not direct A->B
                # So we replace A->B or B->A with A<->B
                edge_map.pop(k1, None)
                edge_map.pop(k2, None)
                
                edge_map[k1] = e # Store one way, mark as bi-directed
                
            elif e['type'] == 'directed_fci':
                # Confirm or add directed edge
                k = (e['source'], e['target'])
                if k in edge_map:
                    edge_map[k]['confidence_boost'] = True
                else:
                    edge_map[k] = e
                    
        return list(edge_map.values())

    async def _persist_graph(
        self,
        edges: List[Dict[str, Any]],
        graph_service_instance: Any
    ) -> Dict[str, int]:
        """
        Send edges to GraphService.

        Args:
            edges: List of edge dicts to persist
            graph_service_instance: Instance of AsyncGraphService

        Returns:
            Dict with 'persisted' and 'skipped' counts
        """
        try:
            result = await graph_service_instance.update_causal_edges(edges)
            return result if result else {"persisted": 0, "skipped": 0}
        except Exception as e:
            logger.error(f"Failed to persist edges: {e}")
            return {"persisted": 0, "skipped": 0, "error": str(e)}

causal_manager = CausalDiscoveryManager()
