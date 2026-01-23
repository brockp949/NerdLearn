"""
HiPlan Orchestrator - Hierarchical Planning Framework for Curriculum Generation

Research alignment:
- HiPlan: Hierarchical planning where high-level planner outlines global milestones
  and lower-level agents execute local planning (modules/lessons)
- Maintains coherence over massive context windows through state management
- Implements agent handoff mechanisms for multi-agent workflows

Key Features:
1. Global milestone planning (course structure, module sequencing)
2. Local content planning (lessons, activities per module)
3. Context window management for large curricula
4. Agent coordination and handoff protocols
5. Revision loop support for iterative refinement
"""
from typing import Dict, Any, List, Optional, Callable, Awaitable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import logging
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


class PlanningPhase(str, Enum):
    """Phases in hierarchical planning"""
    GLOBAL_PLANNING = "global_planning"      # Architect: course structure
    LOCAL_PLANNING = "local_planning"        # Refiner: module details
    VERIFICATION = "verification"            # Verifier: quality check
    REVISION = "revision"                    # Back to earlier phases if needed
    COMPLETE = "complete"                    # Final syllabus ready


class HandoffType(str, Enum):
    """Types of agent handoffs"""
    SEQUENTIAL = "sequential"    # A -> B -> C in order
    CONDITIONAL = "conditional"  # A -> B or C based on condition
    ITERATIVE = "iterative"      # A -> B -> A (revision loop)


@dataclass
class PlanningContext:
    """Context passed between planning phases"""
    topic: str
    constraints: Dict[str, Any]
    course_id: Optional[int] = None

    # Planning artifacts
    global_milestones: Optional[Dict[str, Any]] = None
    local_plans: List[Dict[str, Any]] = field(default_factory=list)
    verification_result: Optional[Dict[str, Any]] = None

    # State tracking
    current_phase: PlanningPhase = PlanningPhase.GLOBAL_PLANNING
    revision_count: int = 0
    max_revisions: int = 3

    # Context window management
    context_summary: str = ""
    processed_modules: List[int] = field(default_factory=list)

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class HandoffProtocol:
    """Protocol for agent handoffs"""
    source_agent: str
    target_agent: str
    handoff_type: HandoffType
    required_artifacts: List[str]  # What must be present before handoff
    condition: Optional[Callable[[PlanningContext], bool]] = None


class HiPlanOrchestrator:
    """
    Hierarchical Planning Orchestrator

    Coordinates the multi-agent curriculum generation process:

    1. Global Planning (Architect):
       - Receives topic and constraints
       - Queries knowledge graph for domain topology
       - Produces high-level "Arc of Learning" with module structure

    2. Local Planning (Refiner):
       - Takes each module from the Arc
       - Generates specific Learning Outcomes
       - Maps to content types and time estimates

    3. Verification (Verifier):
       - Validates against knowledge graph
       - Checks pedagogical quality
       - Triggers revision if needed

    4. Revision Loop:
       - If issues found, returns to relevant phase
       - Provides feedback for correction
       - Limits revision cycles to prevent infinite loops
    """

    def __init__(
        self,
        architect_agent,
        refiner_agent,
        verifier_agent,
        graph_service,
        llm: Optional[ChatOpenAI] = None
    ):
        self.architect = architect_agent
        self.refiner = refiner_agent
        self.verifier = verifier_agent
        self.graph_service = graph_service
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

        # Define handoff protocols
        self.handoffs = self._define_handoff_protocols()

        # Build the state graph
        self.workflow = self._build_workflow()

    def _define_handoff_protocols(self) -> List[HandoffProtocol]:
        """Define the agent handoff protocols"""
        return [
            # Architect -> Refiner (sequential)
            HandoffProtocol(
                source_agent="architect",
                target_agent="refiner",
                handoff_type=HandoffType.SEQUENTIAL,
                required_artifacts=["global_milestones"]
            ),

            # Refiner -> Verifier (sequential)
            HandoffProtocol(
                source_agent="refiner",
                target_agent="verifier",
                handoff_type=HandoffType.SEQUENTIAL,
                required_artifacts=["global_milestones", "local_plans"]
            ),

            # Verifier -> Architect (conditional - on revision needed)
            HandoffProtocol(
                source_agent="verifier",
                target_agent="architect",
                handoff_type=HandoffType.CONDITIONAL,
                required_artifacts=["verification_result"],
                condition=lambda ctx: (
                    ctx.verification_result.get("needs_revision", False) and
                    ctx.revision_count < ctx.max_revisions and
                    any(
                        i.get("severity") == "critical" and
                        i.get("type") in ["PREREQUISITE_VIOLATION", "MISSING_PREREQUISITE"]
                        for i in ctx.verification_result.get("issues", [])
                    )
                )
            ),

            # Verifier -> Refiner (conditional - on LO revision needed)
            HandoffProtocol(
                source_agent="verifier",
                target_agent="refiner",
                handoff_type=HandoffType.CONDITIONAL,
                required_artifacts=["verification_result"],
                condition=lambda ctx: (
                    ctx.verification_result.get("needs_revision", False) and
                    ctx.revision_count < ctx.max_revisions and
                    any(
                        i.get("type") in ["BLOOM_REGRESSION", "TIME_IMBALANCE"]
                        for i in ctx.verification_result.get("issues", [])
                    )
                )
            )
        ]

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for hierarchical planning"""

        # Define the state graph
        workflow = StateGraph(dict)

        # Add nodes for each phase
        workflow.add_node("global_planning", self._global_planning_node)
        workflow.add_node("local_planning", self._local_planning_node)
        workflow.add_node("verification", self._verification_node)
        workflow.add_node("summarize_context", self._summarize_context_node)

        # Add edges
        workflow.add_edge("global_planning", "local_planning")
        workflow.add_edge("local_planning", "verification")

        # Conditional edges from verification
        workflow.add_conditional_edges(
            "verification",
            self._determine_next_phase,
            {
                "global_planning": "global_planning",
                "local_planning": "local_planning",
                "summarize_context": "summarize_context",
                "complete": END
            }
        )

        workflow.add_edge("summarize_context", END)

        # Set entry point
        workflow.set_entry_point("global_planning")

        return workflow.compile()

    async def _global_planning_node(self, state: dict) -> dict:
        """
        Global Planning Phase - Architect Agent

        Produces the high-level Arc of Learning with:
        - Module structure and sequencing
        - Concept to module mapping
        - Prerequisite ordering
        """
        logger.info("HiPlan: Starting global planning phase")

        context = state.get("context", PlanningContext(topic="", constraints={}))

        try:
            # Prepare agent state for architect
            agent_state = {
                "topic": context.topic,
                "constraints": context.constraints,
                "course_id": context.course_id,
                "knowledge_graph_data": None,
                "arc_of_learning": None,
                "learning_outcomes": None,
                "verification_results": None,
                "final_syllabus": None,
                "errors": context.errors.copy(),
                "warnings": context.warnings.copy(),
                "messages": [],
                "current_agent": "architect"
            }

            # If this is a revision, include feedback
            if context.revision_count > 0 and context.verification_result:
                issues = context.verification_result.get("issues", [])
                critical_issues = [i for i in issues if i.get("severity") == "critical"]

                agent_state["revision_feedback"] = {
                    "attempt": context.revision_count + 1,
                    "critical_issues": critical_issues,
                    "summary": context.verification_result.get("summary", "")
                }

            # Run architect agent
            result_state = await self.architect.process(agent_state)

            # Extract results
            context.global_milestones = result_state.get("arc_of_learning")
            context.errors = result_state.get("errors", [])
            context.warnings = result_state.get("warnings", [])
            context.current_phase = PlanningPhase.LOCAL_PLANNING

            state["context"] = context
            state["agent_state"] = result_state

            logger.info(f"HiPlan: Global planning complete, {len(context.global_milestones.get('modules', []))} modules planned")

        except Exception as e:
            logger.error(f"HiPlan: Global planning error: {e}")
            context.errors.append(f"Global planning failed: {str(e)}")
            state["context"] = context

        return state

    async def _local_planning_node(self, state: dict) -> dict:
        """
        Local Planning Phase - Refiner Agent

        For each module, produces:
        - Specific Learning Outcomes with Bloom's verbs
        - Content type mappings
        - Time estimates
        """
        logger.info("HiPlan: Starting local planning phase")

        context = state["context"]
        agent_state = state.get("agent_state", {})

        try:
            # Ensure arc_of_learning is in agent state
            agent_state["arc_of_learning"] = context.global_milestones
            agent_state["current_agent"] = "refiner"

            # If revision, include specific feedback for Refiner
            if context.revision_count > 0 and context.verification_result:
                lo_issues = [
                    i for i in context.verification_result.get("issues", [])
                    if i.get("type") in ["BLOOM_REGRESSION", "TIME_IMBALANCE"]
                ]
                agent_state["revision_feedback"] = {
                    "attempt": context.revision_count + 1,
                    "lo_issues": lo_issues
                }

            # Run refiner agent
            result_state = await self.refiner.process(agent_state)

            # Extract results
            context.local_plans = result_state.get("learning_outcomes", {}).get("modules", [])
            context.errors = result_state.get("errors", [])
            context.warnings = result_state.get("warnings", [])
            context.current_phase = PlanningPhase.VERIFICATION

            state["context"] = context
            state["agent_state"] = result_state

            logger.info(f"HiPlan: Local planning complete, {len(context.local_plans)} modules refined")

        except Exception as e:
            logger.error(f"HiPlan: Local planning error: {e}")
            context.errors.append(f"Local planning failed: {str(e)}")
            state["context"] = context

        return state

    async def _verification_node(self, state: dict) -> dict:
        """
        Verification Phase - Verifier Agent

        Validates the curriculum against:
        - Knowledge graph prerequisites
        - Bloom's taxonomy progression
        - Time allocation reasonableness
        - Overall pedagogical quality
        """
        logger.info("HiPlan: Starting verification phase")

        context = state["context"]
        agent_state = state.get("agent_state", {})

        try:
            # Ensure all artifacts are in agent state
            agent_state["arc_of_learning"] = context.global_milestones
            agent_state["learning_outcomes"] = {"modules": context.local_plans}
            agent_state["current_agent"] = "verifier"

            # Run verifier agent
            result_state = await self.verifier.process(agent_state)

            # Extract results
            context.verification_result = result_state.get("verification_results", {})
            context.errors = result_state.get("errors", [])
            context.warnings = result_state.get("warnings", [])

            # Store final syllabus if passed
            if result_state.get("final_syllabus"):
                state["final_syllabus"] = result_state["final_syllabus"]

            state["context"] = context
            state["agent_state"] = result_state

            quality_score = context.verification_result.get("quality_score", 0)
            passed = context.verification_result.get("passed", False)

            logger.info(f"HiPlan: Verification complete, score={quality_score}, passed={passed}")

        except Exception as e:
            logger.error(f"HiPlan: Verification error: {e}")
            context.errors.append(f"Verification failed: {str(e)}")
            context.verification_result = {"passed": False, "needs_revision": False}
            state["context"] = context

        return state

    def _determine_next_phase(self, state: dict) -> str:
        """
        Determine next phase based on verification results

        Returns:
            Target node name for routing
        """
        context = state.get("context")
        if not context:
            return "complete"

        verification = context.verification_result or {}

        # If passed, we're done
        if verification.get("passed", False):
            logger.info("HiPlan: Verification passed, completing")
            return "summarize_context"

        # If no revision needed or max revisions reached
        if not verification.get("needs_revision", False):
            logger.info("HiPlan: No revision needed, completing")
            return "summarize_context"

        if context.revision_count >= context.max_revisions:
            logger.warning(f"HiPlan: Max revisions ({context.max_revisions}) reached")
            return "summarize_context"

        # Increment revision count
        context.revision_count += 1
        state["context"] = context

        # Determine which phase needs revision based on issues
        issues = verification.get("issues", [])

        # Critical prerequisite issues -> back to Architect
        has_prerequisite_issues = any(
            i.get("severity") == "critical" and
            i.get("type") in ["PREREQUISITE_VIOLATION", "MISSING_PREREQUISITE", "HALLUCINATED_CONCEPT"]
            for i in issues
        )

        if has_prerequisite_issues:
            logger.info(f"HiPlan: Revision {context.revision_count} - returning to global planning")
            return "global_planning"

        # LO-level issues -> back to Refiner
        has_lo_issues = any(
            i.get("type") in ["BLOOM_REGRESSION", "TIME_IMBALANCE"]
            for i in issues
        )

        if has_lo_issues:
            logger.info(f"HiPlan: Revision {context.revision_count} - returning to local planning")
            return "local_planning"

        # Default: complete anyway
        return "summarize_context"

    async def _summarize_context_node(self, state: dict) -> dict:
        """
        Summarize the planning context for final output

        This helps manage context window for large curricula by
        creating a compressed summary of the generation process.
        """
        context = state.get("context")
        if not context:
            return state

        context.completed_at = datetime.utcnow()

        # Build context summary
        summary_parts = [
            f"Topic: {context.topic}",
            f"Modules: {len(context.global_milestones.get('modules', []))}",
            f"Learning Outcomes: {sum(len(m.get('learning_outcomes', [])) for m in context.local_plans)}",
            f"Quality Score: {context.verification_result.get('quality_score', 0)}",
            f"Revisions: {context.revision_count}",
            f"Passed: {context.verification_result.get('passed', False)}"
        ]

        context.context_summary = " | ".join(summary_parts)
        context.current_phase = PlanningPhase.COMPLETE

        state["context"] = context

        logger.info(f"HiPlan: Planning complete - {context.context_summary}")

        return state

    async def run(
        self,
        topic: str,
        constraints: Dict[str, Any],
        course_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run the hierarchical planning process

        Args:
            topic: Course topic
            constraints: Curriculum constraints
            course_id: Optional course ID for KG queries

        Returns:
            Planning result with syllabus and metadata
        """
        logger.info(f"HiPlan: Starting curriculum generation for '{topic}'")

        # Initialize context
        context = PlanningContext(
            topic=topic,
            constraints=constraints,
            course_id=course_id,
            started_at=datetime.utcnow()
        )

        # Initial state
        initial_state = {
            "context": context,
            "agent_state": {},
            "final_syllabus": None
        }

        try:
            # Run the workflow
            final_state = await self.workflow.ainvoke(initial_state)

            # Extract results
            context = final_state.get("context", context)
            final_syllabus = final_state.get("final_syllabus")

            # Build result
            result = {
                "success": context.verification_result.get("passed", False) if context.verification_result else False,
                "topic": topic,
                "syllabus": final_syllabus,
                "quality_score": context.verification_result.get("quality_score", 0) if context.verification_result else 0,
                "metadata": {
                    "revisions": context.revision_count,
                    "phase_completed": context.current_phase.value,
                    "started_at": context.started_at.isoformat() if context.started_at else None,
                    "completed_at": context.completed_at.isoformat() if context.completed_at else None,
                    "summary": context.context_summary
                },
                "errors": context.errors,
                "warnings": context.warnings,
                "verification": context.verification_result
            }

            return result

        except Exception as e:
            logger.error(f"HiPlan: Critical error during planning: {e}", exc_info=True)
            return {
                "success": False,
                "topic": topic,
                "syllabus": None,
                "errors": [f"Critical error: {str(e)}"],
                "warnings": [],
                "metadata": {
                    "phase_completed": context.current_phase.value,
                    "revisions": context.revision_count
                }
            }

    def validate_handoff(
        self,
        protocol: HandoffProtocol,
        context: PlanningContext
    ) -> Tuple[bool, List[str]]:
        """
        Validate that a handoff can proceed

        Args:
            protocol: The handoff protocol to validate
            context: Current planning context

        Returns:
            (can_proceed, missing_artifacts)
        """
        missing = []

        for artifact in protocol.required_artifacts:
            if artifact == "global_milestones" and not context.global_milestones:
                missing.append("global_milestones")
            elif artifact == "local_plans" and not context.local_plans:
                missing.append("local_plans")
            elif artifact == "verification_result" and not context.verification_result:
                missing.append("verification_result")

        # Check condition if present
        if protocol.condition and missing == []:
            if not protocol.condition(context):
                return (False, ["condition_not_met"])

        return (len(missing) == 0, missing)


# Type hint for Tuple
from typing import Tuple


# Lazy-initialized singleton
_hiplan_orchestrator: Optional[HiPlanOrchestrator] = None


def get_hiplan_orchestrator(
    architect_agent,
    refiner_agent,
    verifier_agent,
    graph_service
) -> HiPlanOrchestrator:
    """Get or create the HiPlan orchestrator"""
    global _hiplan_orchestrator
    if _hiplan_orchestrator is None:
        _hiplan_orchestrator = HiPlanOrchestrator(
            architect_agent=architect_agent,
            refiner_agent=refiner_agent,
            verifier_agent=verifier_agent,
            graph_service=graph_service
        )
    return _hiplan_orchestrator
