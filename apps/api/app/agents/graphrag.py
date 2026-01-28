"""
GraphRAG Integration - Graph-Augmented Retrieval for Curriculum Generation

Research alignment:
- Microsoft GraphRAG: Community detection and hierarchical summarization
- Leiden Algorithm: Superior community detection (Traag et al., 2019)
- Global vs Local Queries: Community summaries for broad topics, vector search for specific QA
- Map-Reduce Summarization: Scalable community summarization pattern

Key Features:
1. Community Detection: Group related concepts using Leiden algorithm (preferred over Louvain)
2. Community Summarization: Map-Reduce LLM-generated summaries for each concept cluster
3. Graph Traversal: Navigate prerequisite chains for curriculum ordering
4. Hybrid Retrieval: Combine graph structure with semantic similarity
"""
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import networkx as nx
from community import community_louvain  # python-louvain (fallback)
import igraph as ig
import leidenalg
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import asyncio
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConceptCommunity:
    """Represents a community of related concepts"""
    community_id: int
    concepts: List[str]
    central_concept: str  # Most connected concept in community
    difficulty_range: Tuple[float, float]  # (min, max) difficulty
    summary: Optional[str] = None
    keywords: List[str] = field(default_factory=list)


@dataclass
class GraphRAGResult:
    """Result from GraphRAG query"""
    communities: List[ConceptCommunity]
    global_summary: str
    prerequisite_chains: List[List[str]]
    concept_hierarchy: Dict[str, List[str]]
    recommendations: List[str]


class GraphRAGService:
    """
    GraphRAG Service for Knowledge Graph-Augmented Generation

    Implements Microsoft GraphRAG methodology:
    1. Build NetworkX graph from Neo4j data
    2. Detect communities using Louvain algorithm
    3. Generate community summaries with LLM
    4. Use community context for curriculum generation

    This provides the "global context" that standard RAG lacks,
    enabling synthesis of broad topics like "Overview of Machine Learning"
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        resolution: float = 1.0,  # Leiden resolution parameter
        algorithm: str = "leiden"  # "leiden" or "louvain"
    ):
        """
        Initialize GraphRAG service

        Args:
            llm: LLM for generating summaries
            resolution: Resolution parameter (higher = more communities)
            algorithm: Community detection algorithm ("leiden" preferred, "louvain" fallback)
        """
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.resolution = resolution
        self.algorithm = algorithm
        self._community_cache: Dict[int, Dict[int, ConceptCommunity]] = {}

    def build_networkx_graph(
        self,
        graph_data: Dict[str, Any]
    ) -> nx.DiGraph:
        """
        Convert Neo4j graph data to NetworkX DiGraph

        Args:
            graph_data: Output from graph_service.get_course_graph()

        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()

        # Add nodes with attributes
        for node in graph_data.get("nodes", []):
            G.add_node(
                node["id"],
                label=node.get("label", node["id"]),
                module=node.get("module"),
                module_order=node.get("module_order", 0),
                difficulty=node.get("difficulty", 5.0),
                importance=node.get("importance", 0.5)
            )

        # Add edges (prerequisite relationships)
        for edge in graph_data.get("edges", []):
            G.add_edge(
                edge["source"],
                edge["target"],
                weight=edge.get("confidence", 0.5),
                type=edge.get("type", "prerequisite")
            )

        return G

    def detect_communities(
        self,
        G: nx.Graph,
        course_id: int
    ) -> Dict[int, ConceptCommunity]:
        """
        Detect concept communities using Leiden algorithm (preferred) or Louvain.

        Research basis: Leiden algorithm (Traag et al., 2019) provides:
        - Guaranteed well-connected communities (improvement over Louvain)
        - Better modularity optimization
        - O(n log n) complexity

        Args:
            G: NetworkX graph
            course_id: Course ID for caching

        Returns:
            Dictionary mapping community_id to ConceptCommunity
        """
        # Check cache
        if course_id in self._community_cache:
            logger.debug(f"Using cached communities for course {course_id}")
            return self._community_cache[course_id]

        # Convert to undirected for community detection
        G_undirected = G.to_undirected()

        if len(G_undirected.nodes()) == 0:
            logger.warning("Empty graph - no communities to detect")
            return {}

        # Detect communities using configured algorithm
        try:
            if self.algorithm == "leiden":
                partition = self._run_leiden(G_undirected)
            else:
                partition = community_louvain.best_partition(
                    G_undirected,
                    resolution=self.resolution,
                    random_state=42
                )
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            # Fallback: treat each node as its own community
            partition = {node: i for i, node in enumerate(G_undirected.nodes())}

        # Group concepts by community
        community_concepts: Dict[int, List[str]] = defaultdict(list)
        for concept, community_id in partition.items():
            community_concepts[community_id].append(concept)

        # Build ConceptCommunity objects
        communities: Dict[int, ConceptCommunity] = {}

        for community_id, concepts in community_concepts.items():
            # Find central concept (highest degree)
            subgraph = G_undirected.subgraph(concepts)
            degrees = dict(subgraph.degree())
            central_concept = max(degrees, key=degrees.get) if degrees else concepts[0]

            # Calculate difficulty range
            difficulties = [
                G.nodes[c].get("difficulty", 5.0)
                for c in concepts
                if c in G.nodes
            ]
            difficulty_range = (
                min(difficulties) if difficulties else 5.0,
                max(difficulties) if difficulties else 5.0
            )

            communities[community_id] = ConceptCommunity(
                community_id=community_id,
                concepts=concepts,
                central_concept=central_concept,
                difficulty_range=difficulty_range
            )

        # Cache results
        self._community_cache[course_id] = communities
        logger.info(f"Detected {len(communities)} communities for course {course_id}")

        return communities

    def _run_leiden(self, G: nx.Graph) -> Dict[Any, int]:
        """
        Run Leiden algorithm for community detection.

        Research alignment: Leiden algorithm provides superior community detection:
        - Guarantees all communities are connected (Louvain does not)
        - Better modularity scores on benchmark datasets
        - Faster convergence through smart local moving

        Args:
            G: NetworkX undirected graph

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

        if edges:
            ig_graph.add_edges(edges)
            ig_graph.es['weight'] = weights

        # Run Leiden algorithm with CPM (Constant Potts Model) for resolution support
        # or use ModularityVertexPartition for standard modularity
        try:
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.RBConfigurationVertexPartition,
                weights='weight' if edges else None,
                resolution_parameter=self.resolution,
                n_iterations=-1,  # Run until convergence
                seed=42  # For reproducibility
            )
        except Exception as e:
            logger.warning(f"Leiden with resolution failed, using Modularity: {e}")
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.ModularityVertexPartition,
                weights='weight' if edges else None,
                n_iterations=-1,
                seed=42
            )

        # Convert back to node -> community_id dict
        result = {}
        for community_id, members in enumerate(partition):
            for idx in members:
                result[node_list[idx]] = community_id

        logger.debug(f"Leiden found {len(partition)} communities")
        return result

    async def summarize_community(
        self,
        community: ConceptCommunity,
        topic: str
    ) -> str:
        """
        Generate LLM summary for a concept community

        Args:
            community: The community to summarize
            topic: Overall course topic for context

        Returns:
            Summary string
        """
        if community.summary:
            return community.summary

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert educational content organizer.
Your task is to summarize a cluster of related concepts into a coherent learning theme."""),
            ("human", """Course Topic: {topic}

Concept Cluster (Community {community_id}):
- Concepts: {concepts}
- Central Concept: {central_concept}
- Difficulty Range: {difficulty_min:.1f} - {difficulty_max:.1f}

Generate a 2-3 sentence summary that:
1. Explains what unifies these concepts
2. Identifies the learning theme
3. Suggests the role this cluster plays in the overall curriculum

Also provide 3-5 keywords that capture this cluster's focus.""")
        ])

        try:
            messages = prompt.format_messages(
                topic=topic,
                community_id=community.community_id,
                concepts=", ".join(community.concepts[:15]),  # Limit for context
                central_concept=community.central_concept,
                difficulty_min=community.difficulty_range[0],
                difficulty_max=community.difficulty_range[1]
            )

            response = await self.llm.ainvoke(messages)
            community.summary = response.content

            # Extract keywords (simple heuristic)
            community.keywords = community.concepts[:5]

            return community.summary

        except Exception as e:
            logger.error(f"Error summarizing community: {e}")
            return f"Cluster of concepts related to {community.central_concept}"

    def find_prerequisite_chains(
        self,
        G: nx.DiGraph,
        target_concepts: Optional[List[str]] = None
    ) -> List[List[str]]:
        """
        Find all prerequisite chains in the graph

        Args:
            G: NetworkX directed graph
            target_concepts: Optional targets to find paths to

        Returns:
            List of prerequisite chains (from foundation to target)
        """
        chains = []

        # Find root concepts (no incoming edges)
        roots = [n for n in G.nodes() if G.in_degree(n) == 0]

        # Find terminal concepts (no outgoing edges)
        terminals = target_concepts or [n for n in G.nodes() if G.out_degree(n) == 0]

        for root in roots:
            for terminal in terminals:
                if root == terminal:
                    continue

                try:
                    # Find all simple paths (prerequisite chains)
                    for path in nx.all_simple_paths(G, root, terminal, cutoff=10):
                        if len(path) >= 2:
                            chains.append(path)
                except nx.NetworkXNoPath:
                    continue
                except nx.NodeNotFound:
                    continue

        # Sort by length and deduplicate
        chains.sort(key=len)
        return chains[:50]  # Limit to top 50 chains

    def compute_concept_hierarchy(
        self,
        G: nx.DiGraph,
        communities: Dict[int, ConceptCommunity]
    ) -> Dict[str, List[str]]:
        """
        Compute hierarchical ordering of concepts

        Args:
            G: NetworkX graph
            communities: Detected communities

        Returns:
            Dictionary mapping parent concepts to children
        """
        hierarchy: Dict[str, List[str]] = defaultdict(list)

        for node in G.nodes():
            # Children are concepts that depend on this node
            children = list(G.successors(node))
            if children:
                hierarchy[node] = children

        return dict(hierarchy)

    async def map_reduce_summarize(
        self,
        communities: Dict[int, ConceptCommunity],
        topic: str,
        batch_size: int = 5
    ) -> str:
        """
        Map-Reduce pattern for summarizing large numbers of communities.

        Research alignment: Map-Reduce summarization pattern from GraphRAG paper:
        - MAP: Generate summaries for each community in parallel
        - REDUCE: Combine community summaries into global overview

        This scales better than sequential summarization for courses
        with many concept clusters.

        Args:
            communities: All detected communities
            topic: Course topic
            batch_size: Number of communities to process in parallel

        Returns:
            Combined summary string
        """
        if not communities:
            return f"This course covers {topic} with no established concept structure yet."

        # MAP phase: Summarize communities in parallel batches
        community_list = list(communities.values())
        intermediate_summaries = []

        for i in range(0, len(community_list), batch_size):
            batch = community_list[i:i + batch_size]
            # Process batch in parallel
            tasks = [self.summarize_community(c, topic) for c in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for community, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to summarize community: {result}")
                    intermediate_summaries.append(
                        f"- {community.central_concept}: Concepts related to {', '.join(community.concepts[:3])}"
                    )
                else:
                    intermediate_summaries.append(
                        f"- {community.central_concept}: {result}"
                    )

        # REDUCE phase: Combine into final summary
        reduce_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert curriculum synthesizer.
Combine multiple concept cluster summaries into a coherent course overview."""),
            ("human", """Course Topic: {topic}

Community Summaries:
{summaries}

Synthesize these into a comprehensive 4-6 sentence overview that:
1. Captures the full scope of the curriculum
2. Shows how different concept clusters connect
3. Identifies the progression from foundational to advanced
4. Highlights practical applications and skills developed""")
        ])

        try:
            messages = reduce_prompt.format_messages(
                topic=topic,
                summaries="\n".join(intermediate_summaries[:15])  # Limit for context
            )
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Map-Reduce reduce phase failed: {e}")
            return f"This course covers {topic} through {len(communities)} interconnected concept clusters."

    async def generate_global_summary(
        self,
        communities: Dict[int, ConceptCommunity],
        topic: str
    ) -> str:
        """
        Generate a global summary across all communities

        This is the key GraphRAG capability: synthesizing broad overviews
        that standard RAG cannot provide.

        Uses Map-Reduce pattern for scalability with large community counts.

        Args:
            communities: All detected communities
            topic: Course topic

        Returns:
            Global summary string
        """
        if not communities:
            return f"This course covers {topic} with no established concept structure yet."

        # Use Map-Reduce for large community sets
        if len(communities) > 8:
            return await self.map_reduce_summarize(communities, topic)

        # Summarize each community first
        community_summaries = []
        for community in communities.values():
            if not community.summary:
                await self.summarize_community(community, topic)
            community_summaries.append(
                f"- {community.central_concept}: {community.summary or 'Related concepts'}"
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert curriculum designer.
Synthesize multiple concept clusters into a coherent course overview."""),
            ("human", """Course Topic: {topic}

Concept Clusters:
{community_summaries}

Generate a comprehensive 3-5 sentence overview that:
1. Describes the overall scope of the curriculum
2. Explains how the concept clusters relate to each other
3. Identifies the learning progression from foundational to advanced
4. Highlights key themes and skills students will develop""")
        ])

        try:
            messages = prompt.format_messages(
                topic=topic,
                community_summaries="\n".join(community_summaries[:10])
            )

            response = await self.llm.ainvoke(messages)
            return response.content

        except Exception as e:
            logger.error(f"Error generating global summary: {e}")
            return f"This course covers {topic} through {len(communities)} interconnected concept clusters."

    async def query(
        self,
        graph_data: Dict[str, Any],
        course_id: int,
        topic: str,
        question: Optional[str] = None
    ) -> GraphRAGResult:
        """
        Main GraphRAG query interface

        Args:
            graph_data: Neo4j graph data
            course_id: Course ID
            topic: Course topic
            question: Optional specific question

        Returns:
            GraphRAGResult with communities, summaries, and chains
        """
        # Build NetworkX graph
        G = self.build_networkx_graph(graph_data)

        if len(G.nodes()) == 0:
            logger.warning("Empty graph provided to GraphRAG")
            return GraphRAGResult(
                communities=[],
                global_summary=f"No concept structure available for {topic}",
                prerequisite_chains=[],
                concept_hierarchy={},
                recommendations=["Add concepts to the knowledge graph first"]
            )

        # Detect communities
        communities = self.detect_communities(G, course_id)

        # Summarize communities
        for community in communities.values():
            await self.summarize_community(community, topic)

        # Generate global summary
        global_summary = await self.generate_global_summary(communities, topic)

        # Find prerequisite chains
        prerequisite_chains = self.find_prerequisite_chains(G)

        # Compute hierarchy
        concept_hierarchy = self.compute_concept_hierarchy(G, communities)

        # Generate recommendations
        recommendations = self._generate_recommendations(G, communities)

        return GraphRAGResult(
            communities=list(communities.values()),
            global_summary=global_summary,
            prerequisite_chains=prerequisite_chains,
            concept_hierarchy=concept_hierarchy,
            recommendations=recommendations
        )

    def _generate_recommendations(
        self,
        G: nx.DiGraph,
        communities: Dict[int, ConceptCommunity]
    ) -> List[str]:
        """Generate curriculum recommendations based on graph analysis"""
        recommendations = []

        # Check for isolated concepts
        isolated = [n for n in G.nodes() if G.degree(n) == 0]
        if isolated:
            recommendations.append(
                f"Consider linking isolated concepts: {', '.join(isolated[:5])}"
            )

        # Check for very large communities (might need splitting)
        large_communities = [c for c in communities.values() if len(c.concepts) > 10]
        if large_communities:
            recommendations.append(
                f"Large concept clusters detected ({len(large_communities)}). "
                "Consider breaking into sub-topics for better learning progression."
            )

        # Check for missing prerequisites
        roots = [n for n in G.nodes() if G.in_degree(n) == 0]
        if len(roots) > len(communities):
            recommendations.append(
                "Many entry-point concepts detected. Ensure learners have clear starting points."
            )

        # Check graph connectivity
        if not nx.is_weakly_connected(G) and len(G.nodes()) > 1:
            recommendations.append(
                "Knowledge graph has disconnected components. "
                "Consider adding relationships between concept clusters."
            )

        return recommendations

    def clear_cache(self, course_id: Optional[int] = None):
        """Clear community cache"""
        if course_id:
            self._community_cache.pop(course_id, None)
        else:
            self._community_cache.clear()


# Lazy-initialized global instance
_graphrag_service: Optional[GraphRAGService] = None


def get_graphrag_service() -> GraphRAGService:
    """Get or create the GraphRAG service singleton (lazy initialization)"""
    global _graphrag_service
    if _graphrag_service is None:
        _graphrag_service = GraphRAGService()
    return _graphrag_service
