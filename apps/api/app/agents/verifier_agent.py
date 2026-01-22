"""
The Verifier Agent - Curriculum Auditor

Research alignment:
- Graph-Augmented CoT: Validates curriculum against Knowledge Graph
- Hallucination Detection: Identifies "hallucinated prerequisites" or logical gaps
- Quality Assurance: Ensures pedagogical integrity before finalization

The Verifier reviews the generated syllabus to ensure it's valid, coherent,
and aligned with the knowledge graph's prerequisite structure.
"""
from typing import Dict, Any, List, Optional, Set, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import json
import logging

from .base_agent import BaseAgent, AgentState

logger = logging.getLogger(__name__)


class VerificationIssue:
    """Represents a verification issue"""

    SEVERITY_CRITICAL = "critical"
    SEVERITY_WARNING = "warning"
    SEVERITY_INFO = "info"

    def __init__(
        self,
        issue_type: str,
        severity: str,
        description: str,
        location: Optional[str] = None,
        suggestion: Optional[str] = None
    ):
        self.issue_type = issue_type
        self.severity = severity
        self.description = description
        self.location = location
        self.suggestion = suggestion

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.issue_type,
            "severity": self.severity,
            "description": self.description,
            "location": self.location,
            "suggestion": self.suggestion
        }


class VerifierAgent(BaseAgent):
    """
    The Verifier Agent - Curriculum Auditor

    Responsibilities:
    1. Validate prerequisite ordering against Knowledge Graph
    2. Detect "hallucinated" concepts not in the domain
    3. Check for logical gaps in the learning path
    4. Verify Bloom's progression within modules
    5. Assess overall pedagogical quality

    Verification Checks:
    - PREREQUISITE_VIOLATION: Concept B depends on A, but A comes after B
    - MISSING_PREREQUISITE: A prerequisite is mentioned but never taught
    - HALLUCINATED_CONCEPT: Concept doesn't exist in the knowledge graph
    - BLOOM_REGRESSION: Higher-order LO before lower-order in same module
    - TIME_IMBALANCE: Module time allocation is unreasonable
    - ORPHAN_CONCEPT: Concept is taught but never used or assessed
    """

    def __init__(self, graph_service, **kwargs):
        super().__init__(
            name="Verifier",
            role_description="""You are a quality assurance expert for educational curricula.
You meticulously verify that learning paths are logically sound, prerequisites are
respected, and content flows from foundational to advanced topics.""",
            **kwargs
        )
        self.graph_service = graph_service

    def create_system_prompt(self) -> str:
        """System prompt for the Verifier"""
        return """You are the Verifier Agent - a curriculum quality auditor.

Your role:
1. Review the generated curriculum for logical consistency
2. Identify prerequisite violations (teaching B before A when B requires A)
3. Detect gaps where important concepts are missing
4. Verify Bloom's Taxonomy progression within modules
5. Assess overall pedagogical quality

Verification Criteria:
- PREREQUISITE_VIOLATION: Concept B depends on A, but A is taught after B
- MISSING_PREREQUISITE: A required prerequisite is never introduced
- HALLUCINATED_CONCEPT: A concept that doesn't belong in this domain
- BLOOM_REGRESSION: Higher Bloom level (e.g., Analyze) before lower (e.g., Remember)
- TIME_IMBALANCE: Unreasonable time allocation (too short/long for complexity)
- ORPHAN_CONCEPT: Concept introduced but never applied or assessed

Severity Levels:
- CRITICAL: Must be fixed before proceeding (e.g., prerequisite violations)
- WARNING: Should be addressed but not blocking (e.g., time imbalance)
- INFO: Suggestions for improvement (e.g., add more practice)

Output Format (JSON):
{
    "passed": true/false,
    "needs_revision": true/false,
    "issues": [
        {
            "type": "PREREQUISITE_VIOLATION",
            "severity": "critical",
            "description": "Detailed description of the issue",
            "location": "Module 2, LO M2-LO3",
            "suggestion": "Move concept X to Module 1 before introducing Y"
        }
    ],
    "quality_score": 0-100,
    "summary": "Overall assessment of the curriculum",
    "recommendations": ["List of improvement suggestions"]
}"""

    async def verify_against_knowledge_graph(
        self,
        arc_of_learning: Dict[str, Any],
        course_id: int
    ) -> List[VerificationIssue]:
        """
        Verify curriculum against the Knowledge Graph

        Args:
            arc_of_learning: The Arc of Learning from Architect
            course_id: Course ID for graph queries

        Returns:
            List of verification issues
        """
        issues = []

        try:
            # Get all concepts taught in order
            taught_concepts: List[Tuple[str, int]] = []  # (concept, week)

            for module in arc_of_learning.get("modules", []):
                week = module.get("week", 0)
                for concept in module.get("concepts", []):
                    taught_concepts.append((concept.lower(), week))

            # Build a map of when each concept is first taught
            concept_week_map: Dict[str, int] = {}
            for concept, week in taught_concepts:
                if concept not in concept_week_map:
                    concept_week_map[concept] = week

            # Check each concept's prerequisites
            for concept, week in taught_concepts:
                try:
                    # Query knowledge graph for prerequisites
                    prereqs = await self.graph_service.get_concept_prerequisites(
                        course_id, concept
                    )

                    for prereq in prereqs:
                        prereq_name = prereq.lower()

                        # Check if prerequisite is taught
                        if prereq_name not in concept_week_map:
                            # Missing prerequisite
                            issues.append(VerificationIssue(
                                issue_type="MISSING_PREREQUISITE",
                                severity=VerificationIssue.SEVERITY_CRITICAL,
                                description=f"Concept '{concept}' requires '{prereq}' which is never taught",
                                location=f"Module (Week {week})",
                                suggestion=f"Add '{prereq}' to an earlier module before teaching '{concept}'"
                            ))
                        elif concept_week_map[prereq_name] > week:
                            # Prerequisite violation - prereq comes after
                            issues.append(VerificationIssue(
                                issue_type="PREREQUISITE_VIOLATION",
                                severity=VerificationIssue.SEVERITY_CRITICAL,
                                description=f"'{concept}' is taught in Week {week} but requires '{prereq}' which is taught in Week {concept_week_map[prereq_name]}",
                                location=f"Module (Week {week})",
                                suggestion=f"Either move '{concept}' to after Week {concept_week_map[prereq_name]} or move '{prereq}' to before Week {week}"
                            ))

                except Exception as e:
                    logger.warning(f"Could not check prerequisites for {concept}: {e}")

            # Check for concepts not in the knowledge graph (hallucinated)
            try:
                known_concepts = await self.graph_service.get_all_concepts(course_id)
                known_set = {c.lower() for c in known_concepts}

                for concept, week in taught_concepts:
                    if concept not in known_set and known_set:  # Only check if graph has data
                        issues.append(VerificationIssue(
                            issue_type="HALLUCINATED_CONCEPT",
                            severity=VerificationIssue.SEVERITY_WARNING,
                            description=f"Concept '{concept}' not found in knowledge graph",
                            location=f"Module (Week {week})",
                            suggestion="Verify this concept exists in the domain or add it to the knowledge graph"
                        ))
            except Exception as e:
                logger.warning(f"Could not check for hallucinated concepts: {e}")

        except Exception as e:
            logger.error(f"Error verifying against knowledge graph: {e}")
            issues.append(VerificationIssue(
                issue_type="VERIFICATION_ERROR",
                severity=VerificationIssue.SEVERITY_WARNING,
                description=f"Could not fully verify against knowledge graph: {str(e)}",
                suggestion="Ensure knowledge graph is populated"
            ))

        return issues

    def verify_blooms_progression(
        self,
        learning_outcomes: Dict[str, Any]
    ) -> List[VerificationIssue]:
        """
        Verify Bloom's Taxonomy progression within modules

        Args:
            learning_outcomes: The refined learning outcomes

        Returns:
            List of verification issues
        """
        issues = []

        # Bloom's levels in order (lower = foundational)
        bloom_order = {
            "remember": 1,
            "understand": 2,
            "apply": 3,
            "analyze": 4,
            "evaluate": 5,
            "create": 6
        }

        for module in learning_outcomes.get("modules", []):
            los = module.get("learning_outcomes", [])

            if not los:
                continue

            # Track progression
            prev_level = 0
            has_regression = False
            regression_details = []

            for i, lo in enumerate(los):
                bloom_level = lo.get("bloom_level", "understand").lower()
                current_level = bloom_order.get(bloom_level, 2)

                # Check for significant regression (more than 2 levels back)
                if i > 0 and current_level < prev_level - 1:
                    has_regression = True
                    regression_details.append(
                        f"{lo.get('lo_id')}: {bloom_level} after previous {los[i-1].get('bloom_level')}"
                    )

                prev_level = current_level

            if has_regression:
                issues.append(VerificationIssue(
                    issue_type="BLOOM_REGRESSION",
                    severity=VerificationIssue.SEVERITY_WARNING,
                    description=f"Bloom's progression has significant regressions: {'; '.join(regression_details)}",
                    location=f"Module: {module.get('module_title')}",
                    suggestion="Reorder LOs to progress from lower to higher cognitive levels"
                ))

        return issues

    def verify_time_allocation(
        self,
        learning_outcomes: Dict[str, Any]
    ) -> List[VerificationIssue]:
        """
        Verify time allocations are reasonable

        Args:
            learning_outcomes: The refined learning outcomes

        Returns:
            List of verification issues
        """
        issues = []

        for module in learning_outcomes.get("modules", []):
            total_minutes = sum(
                lo.get("estimated_minutes", 0)
                for lo in module.get("learning_outcomes", [])
            )

            # Check for extremes (assuming ~3-4 hours per week is reasonable)
            if total_minutes < 60:
                issues.append(VerificationIssue(
                    issue_type="TIME_IMBALANCE",
                    severity=VerificationIssue.SEVERITY_WARNING,
                    description=f"Module has only {total_minutes} minutes of content, which may be too short for a full week",
                    location=f"Module: {module.get('module_title')}",
                    suggestion="Consider adding more Learning Outcomes or extending existing ones"
                ))
            elif total_minutes > 480:  # 8 hours
                issues.append(VerificationIssue(
                    issue_type="TIME_IMBALANCE",
                    severity=VerificationIssue.SEVERITY_WARNING,
                    description=f"Module has {total_minutes} minutes ({total_minutes/60:.1f} hours) of content, which may overwhelm learners",
                    location=f"Module: {module.get('module_title')}",
                    suggestion="Consider splitting into multiple weeks or reducing content"
                ))

        return issues

    async def llm_quality_assessment(
        self,
        arc_of_learning: Dict[str, Any],
        learning_outcomes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use LLM to assess overall pedagogical quality

        Args:
            arc_of_learning: The Arc of Learning
            learning_outcomes: The refined Learning Outcomes

        Returns:
            LLM assessment results
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.create_system_prompt()),
            ("human", """Review this curriculum for pedagogical quality:

ARC OF LEARNING:
{arc_json}

LEARNING OUTCOMES:
{los_json}

Evaluate:
1. Logical flow from foundational to advanced topics
2. Appropriate use of Bloom's Taxonomy verbs
3. Balance of content types (text, video, interactive)
4. Realistic time estimates
5. Clear learning progression
6. Any missing critical concepts
7. Overall pedagogical soundness

Provide your assessment as JSON with:
- quality_score (0-100)
- passed (true if score >= 70)
- issues (any problems found)
- recommendations (suggestions for improvement)
- summary (brief overall assessment)""")
        ])

        try:
            messages = prompt.format_messages(
                arc_json=json.dumps(arc_of_learning, indent=2)[:3000],  # Truncate if too long
                los_json=json.dumps(learning_outcomes, indent=2)[:5000]
            )

            response = await self.llm.ainvoke(messages)

            # Parse response
            response_json = self._extract_json(response.content)
            assessment = json.loads(response_json)

            return assessment

        except Exception as e:
            logger.error(f"Error in LLM quality assessment: {e}")
            return {
                "quality_score": 50,
                "passed": False,
                "issues": [{"type": "ASSESSMENT_ERROR", "description": str(e)}],
                "recommendations": ["Manual review recommended"],
                "summary": "Could not complete automated quality assessment"
            }

    async def process(self, state: AgentState) -> AgentState:
        """
        Process curriculum verification

        Args:
            state: Current agent state

        Returns:
            Updated state with verification results
        """
        self.log_action(state, "Starting curriculum verification")

        try:
            arc_of_learning = state.get("arc_of_learning", {})
            learning_outcomes = state.get("learning_outcomes", {})
            course_id = state.get("course_id")

            all_issues: List[VerificationIssue] = []

            # 1. Verify against Knowledge Graph
            self.log_action(state, "Verifying against knowledge graph")
            kg_issues = await self.verify_against_knowledge_graph(arc_of_learning, course_id)
            all_issues.extend(kg_issues)

            # 2. Verify Bloom's progression
            self.log_action(state, "Verifying Bloom's progression")
            bloom_issues = self.verify_blooms_progression(learning_outcomes)
            all_issues.extend(bloom_issues)

            # 3. Verify time allocations
            self.log_action(state, "Verifying time allocations")
            time_issues = self.verify_time_allocation(learning_outcomes)
            all_issues.extend(time_issues)

            # 4. LLM-based quality assessment
            self.log_action(state, "Running LLM quality assessment")
            llm_assessment = await self.llm_quality_assessment(arc_of_learning, learning_outcomes)

            # Combine all issues
            issues_list = [issue.to_dict() for issue in all_issues]

            # Add LLM-detected issues
            for llm_issue in llm_assessment.get("issues", []):
                if isinstance(llm_issue, dict):
                    issues_list.append({
                        "type": llm_issue.get("type", "LLM_DETECTED"),
                        "severity": llm_issue.get("severity", "warning"),
                        "description": llm_issue.get("description", ""),
                        "location": llm_issue.get("location"),
                        "suggestion": llm_issue.get("suggestion")
                    })

            # Determine if revision is needed
            critical_count = sum(1 for i in issues_list if i.get("severity") == "critical")
            needs_revision = critical_count > 0 or llm_assessment.get("quality_score", 0) < 60

            # Build verification results
            verification_results = {
                "passed": not needs_revision,
                "needs_revision": needs_revision,
                "quality_score": llm_assessment.get("quality_score", 70),
                "issues": issues_list,
                "issue_counts": {
                    "critical": critical_count,
                    "warning": sum(1 for i in issues_list if i.get("severity") == "warning"),
                    "info": sum(1 for i in issues_list if i.get("severity") == "info")
                },
                "recommendations": llm_assessment.get("recommendations", []),
                "summary": llm_assessment.get("summary", "Verification complete")
            }

            state["verification_results"] = verification_results

            # If passed, build final syllabus
            if not needs_revision:
                state["final_syllabus"] = self._build_final_syllabus(
                    arc_of_learning,
                    learning_outcomes,
                    verification_results
                )
                self.log_action(state, "Verification passed - final syllabus generated")
            else:
                self.log_action(
                    state,
                    "Verification found issues - revision needed",
                    {"critical_issues": critical_count}
                )

                # Add issues to state for Architect to address
                state["errors"].extend([
                    f"VERIFICATION: {i['description']}"
                    for i in issues_list
                    if i.get("severity") == "critical"
                ])

        except Exception as e:
            error_msg = f"Error in Verifier agent: {str(e)}"
            logger.error(error_msg, exc_info=True)
            state["errors"].append(f"CRITICAL: {error_msg}")
            state["verification_results"] = {
                "passed": False,
                "needs_revision": True,
                "issues": [{"type": "VERIFIER_ERROR", "description": error_msg}],
                "quality_score": 0
            }

        return state

    def _build_final_syllabus(
        self,
        arc_of_learning: Dict[str, Any],
        learning_outcomes: Dict[str, Any],
        verification_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build the final syllabus from verified components

        Args:
            arc_of_learning: The Arc of Learning
            learning_outcomes: The refined Learning Outcomes
            verification_results: The verification results

        Returns:
            Complete syllabus structure
        """
        modules = []

        arc_modules = {m.get("week"): m for m in arc_of_learning.get("modules", [])}
        lo_modules = {m.get("module_week"): m for m in learning_outcomes.get("modules", [])}

        for week in sorted(set(arc_modules.keys()) | set(lo_modules.keys())):
            arc_module = arc_modules.get(week, {})
            lo_module = lo_modules.get(week, {})

            modules.append({
                "week": week,
                "title": arc_module.get("title") or lo_module.get("module_title"),
                "concepts": arc_module.get("concepts", []),
                "prerequisites": arc_module.get("prerequisites", []),
                "rationale": arc_module.get("rationale", ""),
                "difficulty": arc_module.get("difficulty", 5),
                "learning_outcomes": lo_module.get("learning_outcomes", []),
                "estimated_minutes": sum(
                    lo.get("estimated_minutes", 0)
                    for lo in lo_module.get("learning_outcomes", [])
                ),
                "summary": lo_module.get("module_summary", "")
            })

        return {
            "version": "1.0",
            "generated_at": None,  # Will be set by the service layer
            "overall_arc": arc_of_learning.get("overall_arc", ""),
            "quality_score": verification_results.get("quality_score", 0),
            "total_weeks": len(modules),
            "total_learning_outcomes": learning_outcomes.get("total_los", 0),
            "total_estimated_hours": learning_outcomes.get("total_estimated_minutes", 0) / 60,
            "modules": modules,
            "metadata": {
                "verification_passed": verification_results.get("passed", False),
                "recommendations": verification_results.get("recommendations", [])
            }
        }

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response"""
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            return text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            return text[start:end].strip()
        else:
            return text.strip()
