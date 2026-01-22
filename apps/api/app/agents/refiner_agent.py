"""
The Refiner Agent - Local Planner for Learning Outcomes

Research alignment:
- HiPlan: Lower-level agent executes local planning (specific lessons/content)
- Bloom's Taxonomy: Generates Learning Outcomes using cognitive verbs
- Scaffolding: Ensures progression from lower-order to higher-order thinking

The Refiner takes the Architect's Arc of Learning and generates specific
Learning Outcomes (LOs) for each module.
"""
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import json
import logging

from .base_agent import BaseAgent, AgentState

logger = logging.getLogger(__name__)


# Bloom's Taxonomy verbs for each cognitive level
BLOOMS_TAXONOMY = {
    "remember": ["define", "list", "recall", "identify", "recognize", "name", "state"],
    "understand": ["explain", "describe", "summarize", "paraphrase", "classify", "compare"],
    "apply": ["implement", "execute", "use", "demonstrate", "solve", "calculate"],
    "analyze": ["differentiate", "organize", "attribute", "deconstruct", "examine", "contrast"],
    "evaluate": ["critique", "judge", "assess", "justify", "defend", "argue"],
    "create": ["design", "construct", "develop", "formulate", "synthesize", "compose"]
}


class RefinerAgent(BaseAgent):
    """
    The Refiner Agent - Local Planner

    Responsibilities:
    1. Take each module from the Architect's Arc of Learning
    2. Generate specific Learning Outcomes (LOs) using Bloom's Taxonomy verbs
    3. Ensure cognitive progression within each module
    4. Map LOs to content types (text, video, interactive, etc.)
    5. Estimate time allocations for each LO

    Example Output for a module:
    {
        "module_title": "Quantum State Representation",
        "learning_outcomes": [
            {
                "lo_id": "M2-LO1",
                "verb": "Define",
                "bloom_level": "remember",
                "statement": "Define what a qubit is and how it differs from a classical bit",
                "content_type": "text",
                "estimated_minutes": 15
            },
            {
                "lo_id": "M2-LO2",
                "verb": "Explain",
                "bloom_level": "understand",
                "statement": "Explain the concept of superposition using the Bloch sphere",
                "content_type": "interactive_diagram",
                "estimated_minutes": 25
            }
        ]
    }
    """

    def __init__(self, graph_service, **kwargs):
        super().__init__(
            name="Refiner",
            role_description="""You are an expert learning designer who creates specific,
measurable Learning Outcomes aligned with Bloom's Taxonomy. You ensure each module
progresses from foundational understanding to higher-order thinking skills.""",
            **kwargs
        )
        self.graph_service = graph_service

    def create_system_prompt(self) -> str:
        """System prompt for the Refiner"""
        blooms_examples = self._format_blooms_examples()

        return f"""You are the Refiner Agent - a master learning designer.

Your role:
1. Take each module from the Arc of Learning and expand it into specific Learning Outcomes (LOs)
2. Use Bloom's Taxonomy verbs to ensure measurable outcomes
3. Ensure cognitive progression within each module (remember → create)
4. Map each LO to an appropriate content type
5. Estimate realistic time allocations

Bloom's Taxonomy Verbs (use these):
{blooms_examples}

Content Types Available:
- text: Reading material, explanations
- video: Video lectures, demonstrations
- interactive_diagram: React Flow visualizations, interactive graphs
- podcast: Audio explanations (Podcastfy)
- quiz: Assessment questions
- coding_challenge: Hands-on coding exercises
- discussion: Socratic dialogue with AI tutor

Rules:
1. Each module should have 3-6 Learning Outcomes
2. Start with lower Bloom levels (remember, understand) and progress to higher (apply, analyze, create)
3. Each LO must begin with an action verb from Bloom's Taxonomy
4. LOs must be specific and measurable
5. Estimate realistic time (most LOs: 10-30 minutes)
6. Vary content types to maintain engagement

Output Format (JSON for each module):
{{
    "module_title": "Module Name",
    "module_week": 1,
    "learning_outcomes": [
        {{
            "lo_id": "M1-LO1",
            "verb": "Define",
            "bloom_level": "remember",
            "statement": "Full learning outcome statement starting with verb",
            "content_type": "text",
            "estimated_minutes": 15,
            "prerequisites": ["Any concepts needed before this LO"]
        }}
    ],
    "module_summary": "Brief description of what learners will achieve"
}}"""

    def _format_blooms_examples(self) -> str:
        """Format Bloom's taxonomy for the prompt"""
        lines = []
        for level, verbs in BLOOMS_TAXONOMY.items():
            lines.append(f"  {level.upper()}: {', '.join(verbs[:4])}...")
        return "\n".join(lines)

    def build_refine_prompt(
        self,
        arc_of_learning: Dict[str, Any],
        module: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> ChatPromptTemplate:
        """
        Build prompt for generating Learning Outcomes for a module

        Args:
            arc_of_learning: The complete arc from Architect
            module: Specific module to refine
            constraints: Course constraints

        Returns:
            Prompt template
        """
        # Get context from surrounding modules
        modules = arc_of_learning.get("modules", [])
        module_week = module.get("week", 1)

        # Find previous and next modules for context
        prev_module = None
        next_module = None
        for m in modules:
            if m.get("week") == module_week - 1:
                prev_module = m
            elif m.get("week") == module_week + 1:
                next_module = m

        context = self._build_module_context(prev_module, next_module)

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.create_system_prompt()),
            ("human", """Generate Learning Outcomes for this module:

Module Information:
- Title: {module_title}
- Week: {module_week}
- Concepts to Cover: {concepts}
- Difficulty (1-10): {difficulty}
- Prerequisites: {module_prerequisites}
- Rationale: {rationale}

Course Context:
- Overall Arc: {overall_arc}
- Target Level: {difficulty_level}
- Learning Style Preference: {learning_style}
{module_context}

Requirements:
1. Create 3-6 specific Learning Outcomes for this module
2. Ensure cognitive progression (start with remember/understand, end with apply/analyze/create)
3. Map each LO to an appropriate content type
4. Total estimated time should be reasonable for a week (~2-4 hours of learning)
5. Reference the prerequisite concepts from previous modules where applicable

Output the refined module as a JSON object.""")
        ])

        return prompt.partial(
            module_title=module.get("title", "Unknown Module"),
            module_week=module_week,
            concepts=", ".join(module.get("concepts", [])),
            difficulty=module.get("difficulty", 5),
            module_prerequisites=", ".join(module.get("prerequisites", [])) or "None",
            rationale=module.get("rationale", "Not specified"),
            overall_arc=arc_of_learning.get("overall_arc", "Not specified"),
            difficulty_level=constraints.get("difficulty_level", "intermediate"),
            learning_style=constraints.get("learning_style", "balanced"),
            module_context=context
        )

    def _build_module_context(
        self,
        prev_module: Optional[Dict[str, Any]],
        next_module: Optional[Dict[str, Any]]
    ) -> str:
        """Build context from surrounding modules"""
        context_parts = []

        if prev_module:
            context_parts.append(
                f"- Previous Module (Week {prev_module.get('week')}): "
                f"{prev_module.get('title')} - Concepts: {', '.join(prev_module.get('concepts', []))}"
            )

        if next_module:
            context_parts.append(
                f"- Next Module (Week {next_module.get('week')}): "
                f"{next_module.get('title')} - Concepts: {', '.join(next_module.get('concepts', []))}"
            )

        if context_parts:
            return "\nModule Context:\n" + "\n".join(context_parts)
        return ""

    async def process(self, state: AgentState) -> AgentState:
        """
        Process curriculum generation - Refiner phase

        Args:
            state: Current agent state with arc_of_learning from Architect

        Returns:
            Updated state with learning_outcomes for each module
        """
        self.log_action(state, "Starting Learning Outcome refinement")

        try:
            arc_of_learning = state.get("arc_of_learning")
            constraints = state.get("constraints", {})

            if not arc_of_learning:
                error_msg = "No Arc of Learning found from Architect"
                state["errors"].append(error_msg)
                logger.error(error_msg)
                return state

            # Handle case where arc_of_learning has an error
            if "error" in arc_of_learning:
                error_msg = f"Arc of Learning has errors: {arc_of_learning.get('error')}"
                state["errors"].append(error_msg)
                logger.error(error_msg)
                return state

            modules = arc_of_learning.get("modules", [])
            if not modules:
                error_msg = "No modules found in Arc of Learning"
                state["errors"].append(error_msg)
                logger.error(error_msg)
                return state

            # Process each module
            refined_modules = []

            for module in modules:
                self.log_action(
                    state,
                    f"Refining module: {module.get('title', 'Unknown')}",
                    {"week": module.get("week")}
                )

                try:
                    # Build and execute prompt
                    prompt = self.build_refine_prompt(arc_of_learning, module, constraints)
                    messages = prompt.format_messages()
                    response = await self.llm.ainvoke(messages)

                    # Parse response
                    refined_json = self._extract_json(response.content)
                    refined_module = json.loads(refined_json)

                    # Validate Learning Outcomes
                    los = refined_module.get("learning_outcomes", [])
                    validated_los = self._validate_learning_outcomes(los, module.get("week", 1))
                    refined_module["learning_outcomes"] = validated_los

                    refined_modules.append(refined_module)

                    self.log_action(
                        state,
                        f"Module refined successfully",
                        {"los_count": len(validated_los)}
                    )

                except json.JSONDecodeError as e:
                    warning = f"Failed to parse LOs for module {module.get('title')}: {e}"
                    state["warnings"].append(warning)
                    logger.warning(warning)
                    # Include module with raw response
                    refined_modules.append({
                        "module_title": module.get("title"),
                        "module_week": module.get("week"),
                        "learning_outcomes": [],
                        "raw_response": response.content,
                        "parse_error": str(e)
                    })

                except Exception as e:
                    warning = f"Error refining module {module.get('title')}: {e}"
                    state["warnings"].append(warning)
                    logger.warning(warning, exc_info=True)

            # Store refined modules
            state["learning_outcomes"] = {
                "modules": refined_modules,
                "total_los": sum(len(m.get("learning_outcomes", [])) for m in refined_modules),
                "total_estimated_minutes": sum(
                    sum(lo.get("estimated_minutes", 0) for lo in m.get("learning_outcomes", []))
                    for m in refined_modules
                )
            }

            state["current_agent"] = "verifier"

            self.log_action(
                state,
                "Refinement complete",
                {
                    "modules_refined": len(refined_modules),
                    "total_los": state["learning_outcomes"]["total_los"]
                }
            )

        except Exception as e:
            error_msg = f"Error in Refiner agent: {str(e)}"
            logger.error(error_msg, exc_info=True)
            state["errors"].append(f"CRITICAL: {error_msg}")

        return state

    def _validate_learning_outcomes(
        self,
        los: List[Dict[str, Any]],
        module_week: int
    ) -> List[Dict[str, Any]]:
        """
        Validate and clean up Learning Outcomes

        Args:
            los: List of learning outcomes
            module_week: Week number for ID generation

        Returns:
            Validated list of LOs
        """
        validated = []

        for i, lo in enumerate(los):
            # Ensure required fields
            validated_lo = {
                "lo_id": lo.get("lo_id") or f"M{module_week}-LO{i+1}",
                "verb": lo.get("verb", "Understand"),
                "bloom_level": lo.get("bloom_level", "understand"),
                "statement": lo.get("statement", ""),
                "content_type": lo.get("content_type", "text"),
                "estimated_minutes": lo.get("estimated_minutes", 15),
                "prerequisites": lo.get("prerequisites", [])
            }

            # Validate Bloom's level
            if validated_lo["bloom_level"].lower() not in BLOOMS_TAXONOMY:
                validated_lo["bloom_level"] = "understand"

            # Validate content type
            valid_content_types = [
                "text", "video", "interactive_diagram", "podcast",
                "quiz", "coding_challenge", "discussion"
            ]
            if validated_lo["content_type"] not in valid_content_types:
                validated_lo["content_type"] = "text"

            # Validate time estimate (reasonable bounds)
            if validated_lo["estimated_minutes"] < 5:
                validated_lo["estimated_minutes"] = 5
            elif validated_lo["estimated_minutes"] > 120:
                validated_lo["estimated_minutes"] = 60

            validated.append(validated_lo)

        return validated

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
