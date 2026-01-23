"""
Community Detection Service
Manage community detection and summarization for GraphRAG
"""
import networkx as nx
import community as community_louvain
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

    async def run_detection(self, course_id: int) -> int:
        """
        Run community detection (Louvain) on the course graph
        and update the graph with community assignments.
        """
        logger.info(f"Starting community detection for course {course_id}")
        
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
            
        # 3. Run Louvain Algorithm
        # Resolution 1.0 is standard. Higher = smaller communities, Lower = larger communities.
        partition = community_louvain.best_partition(G, weight='weight', resolution=1.0)
        
        # 4. Update Neo4j
        # partition is Dict[node, community_id]
        count = await self.graph_service.update_community_structure(course_id, partition)
        
        logger.info(f"Community detection complete. Updated {count} concepts.")
        return count

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
