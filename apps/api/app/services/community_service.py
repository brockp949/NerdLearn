"""
Community Detection Service
Manage community detection and summarization for GraphRAG

Research alignment:
- Leiden Algorithm: Guarantees well-connected communities (improvement over Louvain)
- Hierarchical community detection for multi-scale analysis
- Map-Reduce summarization pattern for large community sets
"""
import networkx as nx
import community as community_louvain
import igraph as ig
import leidenalg
from typing import List, Dict, Any
import logging
import dspy
from app.core.config import settings
from app.services.graph_service import AsyncGraphService
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

# Define DSPy Signature
class CommunitySummarizer(dspy.Signature):
    """
    Analyze the concept list and generate a concise summary of the central topic.
    """
    context = dspy.InputField(desc="List of concepts and their descriptions belonging to a community")
    summary = dspy.OutputField(desc="1-2 paragraph summary of the topic and relationships")

class CommunityDetectionService:
    def __init__(self, db=None):
        self.db = db
        self.graph_service = AsyncGraphService(db=self.db)
        self.vector_store = VectorStoreService(db=self.db)
        
        # Configure DSPy
        if settings.OPENAI_API_KEY:
            lm = dspy.OpenAI(
                model='gpt-3.5-turbo',
                api_key=settings.OPENAI_API_KEY,
                max_tokens=500
            )
            dspy.settings.configure(lm=lm)
            self.summarize_module = dspy.Predict(CommunitySummarizer)
        else:
            logger.warning("OPENAI_API_KEY not set. DSPy summarization disabled.")
            self.summarize_module = None

    async def run_detection(self, course_id: int, algorithm: str = "leiden") -> int:
        """
        Run community detection on the course graph using Leiden algorithm (preferred)
        or Louvain as fallback.

        Research basis:
        - Leiden algorithm guarantees well-connected communities
        - Better modularity optimization than Louvain
        - Supports hierarchical/multi-resolution detection

        Args:
            course_id: Course to detect communities for
            algorithm: "leiden" (default) or "louvain" for fallback
        """
        logger.info(f"Starting community detection for course {course_id} using {algorithm}")

        # 1. Fetch Graph
        graph_data = await self.graph_service.get_course_graph(course_id)

        if not graph_data["nodes"]:
            logger.warning("No nodes found for community detection")
            return 0

        # 2. Build NetworkX Graph
        G = nx.Graph()
        for node in graph_data["nodes"]:
            G.add_node(node["id"])

        for edge in graph_data["edges"]:
            G.add_edge(edge["source"], edge["target"], weight=edge.get("confidence", 1.0))

        # 3. Run Community Detection Algorithm
        if algorithm == "leiden":
            partition = self._run_leiden(G)
        else:
            # Fallback to Louvain
            partition = community_louvain.best_partition(G, weight='weight', resolution=1.0)

        # 4. Update Neo4j
        count = await self.graph_service.update_community_structure(course_id, partition)

        logger.info(f"Community detection complete. Updated {count} concepts.")
        return count

    def _run_leiden(self, G: nx.Graph, resolution: float = 1.0) -> Dict[Any, int]:
        """
        Run Leiden algorithm for community detection.

        Research alignment: Leiden algorithm (Traag et al., 2019) provides:
        - Guaranteed well-connected communities (unlike Louvain)
        - Better modularity optimization
        - Faster convergence on large graphs

        Args:
            G: NetworkX graph
            resolution: Resolution parameter (higher = smaller communities)

        Returns:
            Dict mapping node to community_id
        """
        if len(G.nodes()) == 0:
            return {}

        # Convert NetworkX to igraph
        node_list = list(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        # Create igraph graph
        ig_graph = ig.Graph()
        ig_graph.add_vertices(len(node_list))

        # Add edges with weights
        edges = []
        weights = []
        for u, v, data in G.edges(data=True):
            edges.append((node_to_idx[u], node_to_idx[v]))
            weights.append(data.get('weight', 1.0))

        ig_graph.add_edges(edges)
        ig_graph.es['weight'] = weights

        # Run Leiden algorithm with Modularity optimization
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.ModularityVertexPartition,
            weights='weight',
            n_iterations=-1,  # Run until convergence
            seed=42  # For reproducibility
        )

        # Convert back to node -> community_id dict
        result = {}
        for community_id, members in enumerate(partition):
            for idx in members:
                result[node_list[idx]] = community_id

        return result

    async def run_hierarchical_detection(self, course_id: int, levels: int = 3) -> Dict[str, Any]:
        """
        Run hierarchical community detection at multiple resolution levels.

        Research basis: Multi-scale community structure reveals:
        - Fine-grained topics at high resolution
        - Broad themes at low resolution
        - Optimal learning path granularity

        Args:
            course_id: Course to analyze
            levels: Number of hierarchy levels

        Returns:
            Hierarchical community structure
        """
        logger.info(f"Running hierarchical detection for course {course_id}")

        graph_data = await self.graph_service.get_course_graph(course_id)
        if not graph_data["nodes"]:
            return {"levels": [], "total_communities": 0}

        # Build graph
        G = nx.Graph()
        for node in graph_data["nodes"]:
            G.add_node(node["id"])
        for edge in graph_data["edges"]:
            G.add_edge(edge["source"], edge["target"], weight=edge.get("confidence", 1.0))

        # Run detection at different resolutions
        resolutions = [0.5, 1.0, 2.0][:levels]
        hierarchy = {"levels": [], "total_communities": 0}

        for level, resolution in enumerate(resolutions):
            partition = self._run_leiden(G, resolution=resolution)

            # Group by community
            communities = {}
            for node, comm_id in partition.items():
                if comm_id not in communities:
                    communities[comm_id] = []
                communities[comm_id].append(node)

            hierarchy["levels"].append({
                "level": level,
                "resolution": resolution,
                "num_communities": len(communities),
                "communities": communities
            })
            hierarchy["total_communities"] += len(communities)

        return hierarchy

    async def summarize_communities(self, course_id: int) -> int:
        """
        Generate summaries for each detected community and index them.
        """
        # 1. Get all communities
        community_ids = await self.graph_service.get_all_communities(course_id)
        
        processed = 0
        for info_id in community_ids:
            # 2. Get members
            members = await self.graph_service.get_community_members(course_id, info_id)
            if not members:
                continue
                
            # 3. Generate Context
            # Concatenate concept names and descriptions
            context_parts = []
            for m in members:
                desc = m.get("description") or ""
                context_parts.append(f"Concept: {m['name']}\nDescription: {desc}")
                
            context_text = "\n\n".join(context_parts)
            
            # 4. Generate Summary via DSPy
            summary = await self._generate_summary(context_text)
            
            # 5. Index in Vector Store
            doc = {
                "text": summary,
                "course_id": course_id,
                "module_type": "community_summary",
                "page_number": 0,
                "metadata": {
                    "community_id": info_id,
                    "concept_count": len(members),
                    "concepts": [m["name"] for m in members[:10]] # Store top 10 concepts in metadata
                }
            }
            
            await self.vector_store.upsert_documents([doc])
            processed += 1
            
        return processed

    async def _generate_summary(self, context: str) -> str:
        """Generate a summary using DSPy"""
        if not self.summarize_module:
            return "Summarization unavailable (No API Key)"

        # Truncate context to avoid token limits (rough heuristic)
        max_chars = 12000 
        if len(context) > max_chars:
            context = context[:max_chars] + "...(truncated)"
            
        try:
            # DSPy call
            result = self.summarize_module(context=context)
            return result.summary
        except Exception as e:
            logger.error(f"Failed to generate summary with DSPy: {e}")
            return "Summary generation failed."
