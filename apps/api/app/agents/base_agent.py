"""
Base Agent Framework for LangGraph Multi-Agent System

Research alignment:
- HiPlan: Hierarchical Planning for long-horizon tasks
- Agentic Workflows: Distinct AI agents with specialized roles
- State Management: Cyclic workflows with iterative refinement
"""
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """
    Shared state across all agents in the curriculum generation workflow
    
    This state is passed through the agent graph and updated at each step.
    """
    # Input
    topic: str
    constraints: Dict[str, Any]  # duration, level, prerequisites, etc.
    course_id: int
    
    # Intermediate Processing
    messages: Annotated[List[BaseMessage], "Conversation history"]
    knowledge_graph_data: Dict[str, Any]  # Retrieved from Neo4j
    current_agent: str  # Track which agent is active
    iteration_count: int
    
    # Output Stages
    arc_of_learning: Optional[Dict[str, Any]]  # Global structure from Architect
    learning_outcomes: Optional[Dict[str, List[str]]]  # LOs per module from Refiner
    verification_results: Optional[Dict[str, Any]]  # Audit from Verifier
    
    # Final Output
    final_syllabus: Optional[Dict[str, Any]]
    errors: List[str]
    warnings: List[str]


class CurriculumConstraints(BaseModel):
    """Constraints for curriculum generation"""
    duration_weeks: int = Field(default=4, description="Course duration in weeks")
    difficulty_level: str = Field(default="intermediate", description="beginner/intermediate/advanced")
    target_audience: str = Field(default="general", description="Target learner persona")
    prerequisites: List[str] = Field(default_factory=list, description="Required prior knowledge")
    learning_style: str = Field(default="balanced", description="visual/text/interactive/balanced")
    max_modules: Optional[int] = Field(default=None, description="Max number of modules")


class BaseAgent:
    """
    Base class for all curriculum generation agents
    
    Each agent has:
    - A specific role (Architect, Refiner, Verifier)
    - Access to shared state
    - LLM with specialized prompts
    - Ability to update state and pass to next agent
    """
    
    def __init__(
        self,
        name: str,
        role_description: str,
        llm: Optional[ChatOpenAI] = None,
        temperature: float = 0.7,
        model: str = "gpt-4o"
    ):
        """
        Initialize base agent
        
        Args:
            name: Agent name (e.g., "Architect")
            role_description: What this agent does
            llm: Optional pre-configured LLM
            temperature: LLM temperature for creativity/determinism
            model: OpenAI model to use
        """
        self.name = name
        self.role_description = role_description
        self.llm = llm or ChatOpenAI(
            model=model,
            temperature=temperature,
        )
        
    def create_system_prompt(self) -> str:
        """
        Create system prompt for this agent
        
        Override in subclasses to customize agent behavior
        """
        return f"""You are the {self.name} agent in a curriculum design system.

Role: {self.role_description}

Your responses should be structured, pedagogically sound, and aligned with 
educational best practices (Bloom's Taxonomy, scaffolding, etc.).
"""
    
    def process(self, state: AgentState) -> AgentState:
        """
        Process the current state and return updated state
        
        Override in subclasses to implement agent-specific logic
        """
        raise NotImplementedError("Subclasses must implement process()")
    
    def should_continue(self, state: AgentState) -> bool:
        """
        Determine if the workflow should continue to next agent
        
        Returns:
            True to continue, False to end workflow
        """
        # Default: continue if no critical errors and iteration limit not reached
        max_iterations = 5
        has_critical_errors = any(
            "CRITICAL" in error for error in state.get("errors", [])
        )
        
        return (
            state.get("iteration_count", 0) < max_iterations
            and not has_critical_errors
        )
    
    def log_action(self, state: AgentState, action: str, details: Dict[str, Any] = None):
        """
        Log agent action for debugging and monitoring
        """
        logger.info(
            f"[{self.name}] {action}",
            extra={
                "agent": self.name,
                "iteration": state.get("iteration_count", 0),
                "topic": state.get("topic"),
                **(details or {})
            }
        )


class AgentGraph:
    """
    Manages the agent workflow graph using LangGraph
    
    This orchestrates the flow: Architect → Refiner → Verifier → [loop or end]
    """
    
    def __init__(self, graph_service, config: Dict[str, Any] = None):
        """
        Initialize agent graph
        
        Args:
            graph_service: AsyncGraphService for Neo4j operations
            config: Configuration for agents (model, temperature, etc.)
        """
        self.graph_service = graph_service
        self.config = config or {}
        self.workflow = StateGraph(AgentState)
        
    def build_graph(self) -> StateGraph:
        """
        Build the complete agent workflow graph
        
        Returns:
            Compiled StateGraph ready for execution
        """
        # Import agents (will be created in subsequent files)
        from .architect_agent import ArchitectAgent
        from .refiner_agent import RefinerAgent
        from .verifier_agent import VerifierAgent
        
        # Initialize agents
        architect = ArchitectAgent(graph_service=self.graph_service, **self.config)
        refiner = RefinerAgent(graph_service=self.graph_service, **self.config)
        verifier = VerifierAgent(graph_service=self.graph_service, **self.config)
        
        # Add nodes
        self.workflow.add_node("architect", architect.process)
        self.workflow.add_node("refiner", refiner.process)
        self.workflow.add_node("verifier", verifier.process)
        
        # Define edges
        self.workflow.set_entry_point("architect")
        
        # Architect → Refiner
        self.workflow.add_edge("architect", "refiner")
        
        # Refiner → Verifier
        self.workflow.add_edge("refiner", "verifier")
        
        # Verifier → Decision point
        def should_continue_workflow(state: AgentState) -> str:
            """
            Decide if we should loop back or end
            
            Returns:
                "architect" to loop back for revision
                "END" to complete workflow
            """
            verification_results = state.get("verification_results", {})
            needs_revision = verification_results.get("needs_revision", False)
            iteration_count = state.get("iteration_count", 0)
            max_iterations = 3
            
            if needs_revision and iteration_count < max_iterations:
                logger.info(f"Iteration {iteration_count}: Looping back for revision")
                return "architect"
            else:
                logger.info("Verification passed or max iterations reached. Finalizing.")
                return END
        
        self.workflow.add_conditional_edges(
            "verifier",
            should_continue_workflow,
            {
                "architect": "architect",
                END: END
            }
        )
        
        return self.workflow.compile()
    
    async def run(
        self,
        topic: str,
        course_id: int,
        constraints: Optional[CurriculumConstraints] = None
    ) -> AgentState:
        """
        Run the complete curriculum generation workflow
        
        Args:
            topic: The subject to create curriculum for
            course_id: Associated course ID
            constraints: Optional generation constraints
            
        Returns:
            Final state with generated curriculum
        """
        # Initialize state
        initial_state: AgentState = {
            "topic": topic,
            "constraints": (constraints.dict() if constraints else {}),
            "course_id": course_id,
            "messages": [],
            "knowledge_graph_data": {},
            "current_agent": "architect",
            "iteration_count": 0,
            "arc_of_learning": None,
            "learning_outcomes": None,
            "verification_results": None,
            "final_syllabus": None,
            "errors": [],
            "warnings": []
        }
        
        # Build and run graph
        graph = self.build_graph()
        
        try:
            final_state = await graph.ainvoke(initial_state)
            return final_state
        except Exception as e:
            logger.error(f"Error in agent workflow: {e}", exc_info=True)
            initial_state["errors"].append(f"CRITICAL: {str(e)}")
            return initial_state
