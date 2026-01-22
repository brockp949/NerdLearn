"""
The Architect Agent - Global Curriculum Planner

Research alignment:
- HiPlan: High-level planner outlines global milestones (modules/weeks)
- Knowledge Graph: Queries Neo4j for topological dependencies
- Arc of Learning: Ensures foundational concepts precede advanced topics

The Architect does NOT concern itself with specific content, only with
the structural integrity of the learning path.
"""
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import json
import logging

from .base_agent import BaseAgent, AgentState

logger = logging.getLogger(__name__)


class ArchitectAgent(BaseAgent):
    """
    The Architect Agent - Global Planner
    
    Responsibilities:
    1. Receive user's topic and constraints (e.g., "4-week Quantum Computing course")
    2. Query Knowledge Graph for domain's root concepts and dependencies
    3. Output JSON skeleton defining the "Arc of Learning"
    4. Ensure prerequisite concepts logically precede advanced topics
    
    Example Output:
    {
        "modules": [
            {
                "week": 1,
                "title": "Foundations of Linear Algebra",
                "concepts": ["Vectors", "Matrices", "Eigenvalues"],
                "rationale": "Required prerequisite for quantum state representation"
            },
            {
                "week": 2,
                "title": "Quantum State Representation",
                "concepts": ["Qubits", "Superposition", "Bloch Sphere"],
                "prerequisites": ["Vectors", "Matrices"]
            }
        ]
    }
    """
    
    def __init__(self, graph_service, **kwargs):
        super().__init__(
            name="Architect",
            role_description="""You are an expert instructional designer who creates 
high-level curriculum structures. You analyze domain topology in the knowledge graph 
and design optimal learning sequences that respect prerequisite relationships.""",
            **kwargs
        )
        self.graph_service = graph_service
        
    def create_system_prompt(self) -> str:
        """System prompt for the Architect"""
        return """You are the Architect Agent - a master curriculum designer.

Your role:
1. Analyze the topic and identify its core conceptual structure
2. Query the knowledge graph to understand concept dependencies
3. Design a high-level learning sequence (modules/weeks)
4. Ensure prerequisites are taught before dependent concepts

You output a JSON structure called the "Arc of Learning" with:
- Module sequence (ordered by difficulty/prerequisites)
- Core concepts per module
- Rationale for sequencing
- Estimated time allocation

Principles:
- Foundation before application
- Concrete before abstract
- Simple before complex
- Respect topological ordering of prerequisites

You do NOT generate specific lesson content (that's the Refiner's job).
You focus on STRUCTURE and SEQUENCE."""
        
    async def query_knowledge_graph(self, topic: str, course_id: int) -> Dict[str, Any]:
        """
        Query Neo4j for concept topology related to the topic
        
        Args:
            topic: The subject domain
            course_id: Course ID
            
        Returns:
            Dict with concepts, prerequisites, and metadata
        """
        try:
            # Get root concepts (concepts with no prerequisites)
            root_concepts = await self.graph_service.find_concepts_without_prerequisites(course_id)
            
            # Get terminal concepts (endpoints)
            terminal_concepts = await self.graph_service.find_terminal_concepts(course_id)
            
            # Get complete graph structure
            graph_data = await self.graph_service.get_course_graph(course_id)
            
            # Get graph statistics
            stats = await self.graph_service.get_graph_stats(course_id)
            
            return {
                "root_concepts": root_concepts,
                "terminal_concepts": terminal_concepts,
                "graph": graph_data,
                "stats": stats,
                "topic": topic
            }
        except Exception as e:
            logger.error(f"Error querying knowledge graph: {e}")
            # Return empty structure if graph doesn't exist yet
            return {
                "root_concepts": [],
                "terminal_concepts": [],
                "graph": {"nodes": [], "edges": []},
                "stats": {},
                "topic": topic,
                "warning": "No existing knowledge graph found. Will generate from scratch."
            }
    
    def build_arc_prompt(
        self,
        topic: str,
        kg_data: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> ChatPromptTemplate:
        """
        Build prompt for generating the Arc of Learning
        
        Args:
            topic: Subject to teach
            kg_data: Knowledge graph data
            constraints: Duration, level, etc.
            
        Returns:
            Prompt template
        """
        # Extract constraints
        duration_weeks = constraints.get("duration_weeks", 4)
        difficulty_level = constraints.get("difficulty_level", "intermediate")
        prerequisites_list = constraints.get("prerequisites", [])
        
        # Format knowledge graph info
        kg_summary = self._format_kg_summary(kg_data)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.create_system_prompt()),
            ("human", """Design a {duration_weeks}-week curriculum for: {topic}

Target Level: {difficulty_level}
Student Prerequisites: {prerequisites}

Knowledge Graph Data:
{kg_summary}

Requirements:
1. Create {duration_weeks} modules (one per week)
2. Each module should contain 3-5 core concepts
3. Respect prerequisite ordering (foundational → advanced)
4. Provide rationale for each module's position in sequence
5. Consider the student's existing knowledge: {prerequisites}

Output Format (JSON):
{{
    "modules": [
        {{
            "week": 1,
            "title": "Module Title",
            "concepts": ["Concept1", "Concept2", "Concept3"],
            "difficulty": 1-10,
            "prerequisites": ["Previous concepts needed"],
            "rationale": "Why this module is positioned here"
        }}
    ],
    "overall_arc": "Brief description of the learning journey"
}}

IMPORTANT: 
- Ensure each concept in "concepts" appears as a prerequisite before being used as a dependency
- Check topological validity using the knowledge graph
- If the knowledge graph is empty/minimal, use domain expertise to infer logical dependencies""")
        ])
        
        return prompt.partial(
            topic=topic,
            duration_weeks=duration_weeks,
            difficulty_level=difficulty_level,
            prerequisites=", ".join(prerequisites_list) if prerequisites_list else "None",
            kg_summary=kg_summary
        )
    
    def _format_kg_summary(self, kg_data: Dict[str, Any]) -> str:
        """Format knowledge graph data for prompt"""
        root_concepts = kg_data.get("root_concepts", [])
        terminal_concepts = kg_data.get("terminal_concepts", [])
        stats = kg_data.get("stats", {})
        warning = kg_data.get("warning", "")
        
        summary_parts = []
        
        if warning:
            summary_parts.append(f"⚠️ {warning}")
        
        if root_concepts:
            summary_parts.append(f"Entry Point Concepts: {', '.join(root_concepts)}")
        
        if terminal_concepts:
            summary_parts.append(f"Advanced Endpoint Concepts: {', '.join(terminal_concepts)}")
        
        if stats:
            summary_parts.append(f"Total Concepts in Graph: {stats.get('concept_count', 0)}")
            summary_parts.append(f"Prerequisite Relationships: {stats.get('prerequisite_count', 0)}")
        
        return "\n".join(summary_parts) if summary_parts else "No knowledge graph data available"
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Process curriculum generation request - Architect phase
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with arc_of_learning
        """
        self.log_action(state, "Starting curriculum architecture design")
        
        try:
            topic = state["topic"]
            course_id = state["course_id"]
            constraints = state["constraints"]
            iteration = state.get("iteration_count", 0)
            
            # Query knowledge graph
            self.log_action(state, "Querying knowledge graph")
            kg_data = await self.query_knowledge_graph(topic, course_id)
            state["knowledge_graph_data"] = kg_data
            
            # Check if this is a revision iteration
            verification_results = state.get("verification_results", {})
            is_revision = verification_results.get("needs_revision", False)
            
            if is_revision:
                # Incorporate feedback from Verifier
                issues = verification_results.get("issues", [])
                self.log_action(state, "Incorporating verification feedback", {"issues": issues})
                
                # Add revision instructions to prompt
                revision_context = f"\n\nREVISION REQUIRED:\nThe following issues were found:\n"
                revision_context += "\n".join(f"- {issue}" for issue in issues)
                constraints["revision_context"] = revision_context
            
            # Build prompt
            prompt = self.build_arc_prompt(topic, kg_data, constraints)
            
            # Generate Arc of Learning
            self.log_action(state, "Generating Arc of Learning")
            messages = prompt.format_messages()
            response = await self.llm.ainvoke(messages)
            
            # Parse JSON response
            try:
                arc_json = self._extract_json(response.content)
                arc_of_learning = json.loads(arc_json)
                
                state["arc_of_learning"] = arc_of_learning
                state["messages"].append(HumanMessage(content=f"Generate curriculum for: {topic}"))
                state["messages"].append(response)
                
                self.log_action(
                    state,
                    "Arc of Learning generated",
                    {"modules": len(arc_of_learning.get("modules", []))}
                )
                
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse Architect output as JSON: {e}"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                # Store raw response for debugging
                state["arc_of_learning"] = {"raw_response": response.content, "error": str(e)}
            
            # Increment iteration
            state["iteration_count"] = iteration + 1
            state["current_agent"] = "refiner"
            
        except Exception as e:
            error_msg = f"Error in Architect agent: {str(e)}"
            logger.error(error_msg, exc_info=True)
            state["errors"].append(f"CRITICAL: {error_msg}")
        
        return state
    
    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from LLM response (handles markdown code blocks)
        
        Args:
            text: Raw LLM response
            
        Returns:
            JSON string
        """
        # Try to find JSON in markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            return text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            return text[start:end].strip()
        else:
            # Assume entire response is JSON
            return text.strip()
