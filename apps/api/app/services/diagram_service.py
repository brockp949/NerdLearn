"""
Diagram Service - Generative Concept Maps

Research alignment:
- Auto-schematization: Converting linear text to graph structures
- Hybrid approach: LLM generates topology, Elkjs handles layout
"""
import logging
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import json

from app.core.config import settings

logger = logging.getLogger(__name__)


class DiagramService:
    def __init__(self, model: str = "gpt-4o"):
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.2,  # Low temperature for structural consistency
            api_key=settings.OPENAI_API_KEY
        )

    async def generate_concept_map(self, content: str, topic: str = "") -> Dict[str, Any]:
        """
        Generate a node-edge structure from text content
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert educational visualizer. 
Your goal is to convert complex text into a concept map.
            
Output strictly in this JSON format:
{
    "nodes": [
        {"id": "node1", "label": "Main Concept", "type": "root"},
        {"id": "node2", "label": "Sub Concept", "type": "concept"}
    ],
    "edges": [
        {"id": "e1", "source": "node1", "target": "node2", "label": "includes"}
    ]
}

Rules:
1. Identify key entities (nodes).
2. Identify relationships (edges) with short, active verbs.
3. Keep labels concise (< 4 words).
4. Limit to 10-15 nodes for readability.
"""),
            ("human", "Create a concept map for the topic '{topic}' based on this content:\n\n{content}")
        ])

        try:
            messages = prompt.format_messages(topic=topic, content=content[:4000]) # Truncate safety
            response = await self.llm.ainvoke(messages)
            
            # Extract JSON
            json_str = self._extract_json(response.content)
            graph_data = json.loads(json_str)
            
            # Validate structure
            if "nodes" not in graph_data or "edges" not in graph_data:
                raise ValueError("Invalid graph structure returned")
                
            return graph_data
            
        except Exception as e:
            logger.error(f"Failed to generate diagram: {e}", exc_info=True)
            raise e

    def _extract_json(self, text: str) -> str:
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            return text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            return text[start:end].strip()
        return text.strip()
