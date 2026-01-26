"""
Counterfactual Explanations Service

This service orchestrates the Causal-Counterfactual Framework components:
- Structural Causal Model (SCM)
- Counterfactual Engine
- Algorithmic Recourse
- SHAP Attribution
- Socratic Agent

Provides high-level methods for generating explanations, recourse plans,
and Socratic dialogues from student learning data.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import uuid

from app.adaptive.counterfactual import (
    # SCM
    StructuralCausalModel,
    ExogenousVariables,
    EndogenousVariables,
    # Counterfactual
    CounterfactualEngine,
    CounterfactualQuery,
    CounterfactualResult,
    # Recourse
    AlgorithmicRecourse,
    FeatureConstraint,
    RecoursePlan,
    # SHAP
    BanditSHAPExplainer,
    OrdShapExplainer,
    SHAPExplanation,
    # Socratic
    SocraticAgent,
    SocraticContext,
    SocraticDialogue,
    SocraticDialogueBuilder,
    DialogueMode,
)
from app.adaptive.td_bkt import TemporalDifferenceBKT, BeliefState

logger = logging.getLogger(__name__)


@dataclass
class ExplanationContext:
    """Context for generating explanations."""
    student_id: str
    concept_id: str
    concept_name: str
    belief_state: BeliefState
    recent_attempts: List[Dict[str, Any]]
    study_sessions: List[Dict[str, Any]]
    target_mastery: float = 0.85


@dataclass
class CounterfactualExplanation:
    """Complete counterfactual explanation package."""
    query: str
    counterfactual_result: CounterfactualResult
    shap_explanation: Optional[SHAPExplanation]
    recourse_plan: Optional[RecoursePlan]
    natural_language_explanation: str


class CounterfactualService:
    """
    Service for generating counterfactual explanations and Socratic dialogue.

    Orchestrates all components of the Causal-Counterfactual Framework
    to provide actionable insights for students.
    """

    def __init__(
        self,
        td_bkt: Optional[TemporalDifferenceBKT] = None,
        scm: Optional[StructuralCausalModel] = None,
        counterfactual_engine: Optional[CounterfactualEngine] = None,
        recourse_optimizer: Optional[AlgorithmicRecourse] = None,
        shap_explainer: Optional[BanditSHAPExplainer] = None,
        socratic_agent: Optional[SocraticAgent] = None,
    ):
        """
        Initialize the counterfactual service.

        Args:
            td_bkt: TD-BKT instance for mastery tracking
            scm: Structural Causal Model instance
            counterfactual_engine: Counterfactual computation engine
            recourse_optimizer: Algorithmic recourse optimizer
            shap_explainer: SHAP explainer for bandit decisions
            socratic_agent: Socratic dialogue agent
        """
        self.td_bkt = td_bkt or TemporalDifferenceBKT()
        self.scm = scm or StructuralCausalModel(td_bkt=self.td_bkt)
        self.cf_engine = counterfactual_engine or CounterfactualEngine(scm=self.scm)
        self.recourse = recourse_optimizer or AlgorithmicRecourse(scm=self.scm)
        self.shap_explainer = shap_explainer or BanditSHAPExplainer()
        self.socratic_agent = socratic_agent or SocraticAgent()

        # Cache for active dialogues
        self._dialogue_sessions: Dict[str, str] = {}  # student_concept -> session_id

    async def get_counterfactual_explanation(
        self,
        context: ExplanationContext,
        intervention: Dict[str, Any],
        question: Optional[str] = None,
    ) -> CounterfactualExplanation:
        """
        Generate a complete counterfactual explanation.

        Args:
            context: The explanation context with student data
            intervention: The intervention to analyze (e.g., {"study_time": 30})
            question: Optional natural language question

        Returns:
            CounterfactualExplanation with all analysis
        """
        # Build counterfactual query
        if question is None:
            # Generate question from intervention
            intervention_desc = ", ".join([f"{k}={v}" for k, v in intervention.items()])
            question = f"What if {intervention_desc}?"

        query = CounterfactualQuery(
            student_id=context.student_id,
            concept_id=context.concept_id,
            observed_outcome=self._get_latest_outcome(context.recent_attempts),
            intervention=intervention,
            question=question,
        )

        # Compute counterfactual
        cf_result = self.cf_engine.compute_counterfactual(
            query=query,
            belief_state=context.belief_state,
        )

        # Get SHAP explanation for current state
        shap_explanation = None
        try:
            bandit_context = self._build_bandit_context(context)
            shap_explanation = self.shap_explainer.explain_selection(
                context=bandit_context,
                selected_arm=context.concept_id,
            )
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")

        # Get recourse plan if outcome is unfavorable
        recourse_plan = None
        if context.belief_state.mastery < context.target_mastery:
            try:
                recourse_plan = self.recourse.find_minimum_cost_recourse(
                    current_state=context.belief_state,
                    concept_id=context.concept_id,
                    target_success_probability=context.target_mastery,
                )
            except Exception as e:
                logger.warning(f"Recourse computation failed: {e}")

        # Generate natural language explanation
        nl_explanation = self._generate_natural_language_explanation(
            cf_result=cf_result,
            shap_explanation=shap_explanation,
            intervention=intervention,
        )

        return CounterfactualExplanation(
            query=question,
            counterfactual_result=cf_result,
            shap_explanation=shap_explanation,
            recourse_plan=recourse_plan,
            natural_language_explanation=nl_explanation,
        )

    async def get_retrospective_analysis(
        self,
        context: ExplanationContext,
    ) -> Dict[str, Any]:
        """
        Generate retrospective analysis: "What went wrong?"

        Identifies critical decision points and explains contributing factors.

        Args:
            context: The explanation context

        Returns:
            Dict with critical decisions, SHAP explanation, and suggested counterfactuals
        """
        # Get failures from recent attempts
        failures = [a for a in context.recent_attempts if not a.get('correct', True)]

        if not failures:
            return {
                "status": "no_failures",
                "message": "No recent failures to analyze.",
            }

        # Identify critical decisions
        critical_decisions = self.recourse.retrospective_analysis(
            belief_state=context.belief_state,
            attempt_history=context.recent_attempts,
        )

        # Get SHAP explanation
        bandit_context = self._build_bandit_context(context)
        shap_explanation = self.shap_explainer.explain_selection(
            context=bandit_context,
            selected_arm=context.concept_id,
        )

        # Generate suggested counterfactuals based on SHAP
        suggested_counterfactuals = self._suggest_counterfactuals_from_shap(
            shap_explanation=shap_explanation,
            context=context,
        )

        return {
            "status": "analyzed",
            "concept_id": context.concept_id,
            "concept_name": context.concept_name,
            "current_mastery": context.belief_state.mastery,
            "num_failures": len(failures),
            "critical_decisions": [
                {
                    "timestamp": str(d.timestamp) if hasattr(d, 'timestamp') else None,
                    "description": d.description if hasattr(d, 'description') else str(d),
                    "impact_score": d.impact_score if hasattr(d, 'impact_score') else 0,
                    "alternative_outcome": d.alternative_outcome if hasattr(d, 'alternative_outcome') else None,
                }
                for d in (critical_decisions or [])[:5]
            ],
            "shap_explanation": {
                "arm": shap_explanation.arm if shap_explanation else None,
                "base_value": shap_explanation.base_value if shap_explanation else None,
                "feature_contributions": [
                    {
                        "feature": fc.feature,
                        "value": fc.value,
                        "contribution": fc.contribution,
                    }
                    for fc in (shap_explanation.feature_contributions if shap_explanation else [])
                ],
            },
            "suggested_counterfactuals": suggested_counterfactuals,
        }

    async def get_prospective_plan(
        self,
        context: ExplanationContext,
        time_budget_minutes: Optional[float] = None,
        target_probability: float = 0.85,
    ) -> Dict[str, Any]:
        """
        Generate prospective plan: "What to do next?"

        Computes optimal recourse plan to achieve target outcome.

        Args:
            context: The explanation context
            time_budget_minutes: Optional time budget constraint
            target_probability: Target success probability

        Returns:
            Dict with recourse plan and Socratic context
        """
        # Compute recourse plan
        recourse_plan = self.recourse.find_minimum_cost_recourse(
            current_state=context.belief_state,
            concept_id=context.concept_id,
            target_success_probability=target_probability,
            time_budget_minutes=time_budget_minutes,
        )

        # Build prospective SHAP for what would help
        bandit_context = self._build_bandit_context(context)
        shap_explanation = self.shap_explainer.explain_selection(
            context=bandit_context,
            selected_arm=context.concept_id,
        )

        return {
            "status": "planned",
            "concept_id": context.concept_id,
            "concept_name": context.concept_name,
            "current_mastery": context.belief_state.mastery,
            "target_probability": target_probability,
            "time_budget": time_budget_minutes,
            "recourse_plan": {
                "actions": [
                    {
                        "action": a.action_description if hasattr(a, 'action_description') else str(a),
                        "feature": a.feature if hasattr(a, 'feature') else None,
                        "original_value": a.original_value if hasattr(a, 'original_value') else None,
                        "target_value": a.target_value if hasattr(a, 'target_value') else None,
                        "effort": a.effort if hasattr(a, 'effort') else 0,
                        "expected_impact": a.expected_impact if hasattr(a, 'expected_impact') else 0,
                    }
                    for a in (recourse_plan.actions if recourse_plan else [])
                ],
                "total_effort": recourse_plan.total_effort if recourse_plan else 0,
                "time_estimate_minutes": recourse_plan.time_estimate_minutes if recourse_plan else 0,
                "expected_probability": recourse_plan.expected_probability if recourse_plan else context.belief_state.mastery,
            },
            "shap_factors": {
                "positive": [
                    {"feature": fc.feature, "contribution": fc.contribution}
                    for fc in (shap_explanation.feature_contributions if shap_explanation else [])
                    if fc.contribution > 0
                ][:3],
                "negative": [
                    {"feature": fc.feature, "contribution": fc.contribution}
                    for fc in (shap_explanation.feature_contributions if shap_explanation else [])
                    if fc.contribution < 0
                ][:3],
            },
        }

    async def explain_recommendation(
        self,
        context: ExplanationContext,
        recommended_arm: str,
        arm_values: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Explain why a particular learning activity was recommended.

        Uses SHAP to attribute the recommendation to context features.

        Args:
            context: The explanation context
            recommended_arm: The recommended content/activity
            arm_values: Values for all available arms

        Returns:
            Dict with SHAP explanation and natural language summary
        """
        bandit_context = self._build_bandit_context(context)

        shap_explanation = self.shap_explainer.explain_selection(
            context=bandit_context,
            selected_arm=recommended_arm,
        )

        # Generate natural language explanation
        nl_explanation = self._generate_recommendation_explanation(
            shap_explanation=shap_explanation,
            arm_values=arm_values,
            context=context,
        )

        return {
            "recommended_arm": recommended_arm,
            "explanation": {
                "base_value": shap_explanation.base_value if shap_explanation else 0,
                "feature_contributions": [
                    {
                        "feature": fc.feature,
                        "value": fc.value,
                        "contribution": fc.contribution,
                        "interpretation": self._interpret_feature_contribution(fc),
                    }
                    for fc in (shap_explanation.feature_contributions if shap_explanation else [])
                ],
            },
            "natural_language": nl_explanation,
            "alternative_arms": [
                {"arm": arm, "value": value}
                for arm, value in sorted(arm_values.items(), key=lambda x: -x[1])[:5]
            ],
        }

    async def compare_paths(
        self,
        context: ExplanationContext,
        path_a: List[str],
        path_b: List[str],
        num_simulations: int = 100,
    ) -> Dict[str, Any]:
        """
        Compare two alternative learning paths.

        Uses Monte Carlo simulation to estimate outcomes for each path.

        Args:
            context: The explanation context
            path_a: First learning path (list of concept IDs)
            path_b: Second learning path (list of concept IDs)
            num_simulations: Number of Monte Carlo simulations

        Returns:
            Dict comparing expected outcomes of both paths
        """
        comparison = self.cf_engine.simulate_alternative_trajectory(
            belief_state=context.belief_state,
            original_sequence=path_a,
            alternative_sequence=path_b,
            num_simulations=num_simulations,
        )

        # Use OrdShap to explain sequence differences
        ordshap_explainer = OrdShapExplainer(scm=self.scm)
        sequence_impact = ordshap_explainer.explain_sequence_impact(
            sequence=path_a,
            belief_state=context.belief_state,
        )

        return {
            "path_a": {
                "sequence": path_a,
                "expected_mastery": comparison.original_final_mastery if hasattr(comparison, 'original_final_mastery') else 0,
                "success_probability": comparison.original_success_probability if hasattr(comparison, 'original_success_probability') else 0,
            },
            "path_b": {
                "sequence": path_b,
                "expected_mastery": comparison.alternative_final_mastery if hasattr(comparison, 'alternative_final_mastery') else 0,
                "success_probability": comparison.alternative_success_probability if hasattr(comparison, 'alternative_success_probability') else 0,
            },
            "difference": {
                "mastery_delta": (
                    (comparison.alternative_final_mastery if hasattr(comparison, 'alternative_final_mastery') else 0) -
                    (comparison.original_final_mastery if hasattr(comparison, 'original_final_mastery') else 0)
                ),
                "probability_delta": (
                    (comparison.alternative_success_probability if hasattr(comparison, 'alternative_success_probability') else 0) -
                    (comparison.original_success_probability if hasattr(comparison, 'original_success_probability') else 0)
                ),
            },
            "sequence_analysis": {
                "value_contributions": [
                    {"item": c.item, "value_contribution": c.value_contribution, "position_contribution": c.position_contribution}
                    for c in (sequence_impact or [])
                ],
            },
            "recommendation": self._recommend_path(comparison),
        }

    async def start_socratic_dialogue(
        self,
        context: ExplanationContext,
        mode: DialogueMode = DialogueMode.EXPLORATORY,
    ) -> Dict[str, Any]:
        """
        Start a new Socratic dialogue session.

        Args:
            context: The explanation context
            mode: Type of dialogue (retrospective, prospective, exploratory)

        Returns:
            Dict with session info and initial tutor message
        """
        # Build Socratic context
        failures = [a for a in context.recent_attempts if not a.get('correct', True)]

        # Get SHAP and recourse for context
        shap_data = None
        recourse_data = None

        try:
            bandit_context = self._build_bandit_context(context)
            shap_explanation = self.shap_explainer.explain_selection(
                context=bandit_context,
                selected_arm=context.concept_id,
            )
            shap_data = {
                "feature_contributions": [
                    {"feature": fc.feature, "value": fc.value, "contribution": fc.contribution}
                    for fc in (shap_explanation.feature_contributions if shap_explanation else [])
                ]
            }
        except Exception as e:
            logger.warning(f"SHAP for Socratic context failed: {e}")

        try:
            recourse_plan = self.recourse.find_minimum_cost_recourse(
                current_state=context.belief_state,
                concept_id=context.concept_id,
                target_success_probability=context.target_mastery,
            )
            recourse_data = {
                "actions": [
                    {
                        "action": a.action_description if hasattr(a, 'action_description') else str(a),
                        "effort": a.effort if hasattr(a, 'effort') else 0,
                        "expected_impact": a.expected_impact if hasattr(a, 'expected_impact') else 0,
                    }
                    for a in (recourse_plan.actions if recourse_plan else [])
                ],
                "total_effort": recourse_plan.total_effort if recourse_plan else 0,
                "time_estimate_minutes": recourse_plan.time_estimate_minutes if recourse_plan else 0,
                "expected_probability": recourse_plan.expected_probability if recourse_plan else 0,
            }
        except Exception as e:
            logger.warning(f"Recourse for Socratic context failed: {e}")

        socratic_context = (
            SocraticDialogueBuilder(
                student_id=context.student_id,
                concept_id=context.concept_id,
                concept_name=context.concept_name,
            )
            .with_mastery(context.belief_state.mastery, context.target_mastery)
            .with_attempts(context.recent_attempts)
            .with_study_sessions(context.study_sessions)
            .with_shap_explanation(shap_data)
            .with_recourse_plan(recourse_data)
            .build()
        )

        # Generate dialogue
        dialogue = await self.socratic_agent.generate_dialogue(
            context=socratic_context,
            mode=mode,
        )

        # Store session mapping
        session_key = f"{context.student_id}:{context.concept_id}"
        self._dialogue_sessions[session_key] = dialogue.session_id

        return {
            "session_id": dialogue.session_id,
            "mode": mode.value,
            "concept_id": context.concept_id,
            "concept_name": context.concept_name,
            "current_mastery": context.belief_state.mastery,
            "target_mastery": context.target_mastery,
            "turns": [
                {"role": t.role, "content": t.content, "timestamp": str(t.timestamp)}
                for t in dialogue.turns
            ],
        }

    async def continue_socratic_dialogue(
        self,
        session_id: str,
        student_message: str,
    ) -> Dict[str, Any]:
        """
        Continue an existing Socratic dialogue.

        Args:
            session_id: The dialogue session ID
            student_message: The student's response

        Returns:
            Dict with updated dialogue
        """
        dialogue = await self.socratic_agent.respond_to_challenge(
            session_id=session_id,
            student_message=student_message,
        )

        return {
            "session_id": dialogue.session_id,
            "turns": [
                {"role": t.role, "content": t.content, "timestamp": str(t.timestamp)}
                for t in dialogue.turns
            ],
        }

    async def get_dialogue_summary(
        self,
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Get a summary of a dialogue session.

        Args:
            session_id: The dialogue session ID

        Returns:
            Dict with dialogue summary
        """
        summary = await self.socratic_agent.generate_summary(session_id)

        return {
            "session_id": session_id,
            "summary": summary,
        }

    # Private helper methods

    def _get_latest_outcome(self, attempts: List[Dict[str, Any]]) -> bool:
        """Get the outcome of the most recent attempt."""
        if not attempts:
            return True
        return attempts[-1].get('correct', True)

    def _build_bandit_context(self, context: ExplanationContext) -> Dict[str, float]:
        """Build context dictionary for SHAP explainer."""
        # Recent performance
        recent_correct = sum(1 for a in context.recent_attempts[-10:] if a.get('correct', False))
        recent_total = min(len(context.recent_attempts), 10)
        recent_accuracy = recent_correct / max(recent_total, 1)

        # Study time
        total_study_minutes = sum(
            s.get('duration_minutes', 0) for s in context.study_sessions
        )

        # Time since last study
        hours_since_study = 24.0
        if context.study_sessions:
            last_session = max(
                context.study_sessions,
                key=lambda s: s.get('timestamp', datetime.min)
            )
            if 'timestamp' in last_session:
                if isinstance(last_session['timestamp'], datetime):
                    hours_since_study = (datetime.utcnow() - last_session['timestamp']).total_seconds() / 3600
                else:
                    hours_since_study = 24.0

        return {
            "mastery": context.belief_state.mastery,
            "recency": 1.0 / (1.0 + hours_since_study / 24),
            "recent_accuracy": recent_accuracy,
            "study_time": min(total_study_minutes / 60, 1.0),  # Normalize to 0-1
            "attempt_count": min(len(context.recent_attempts) / 20, 1.0),
        }

    def _suggest_counterfactuals_from_shap(
        self,
        shap_explanation: Optional[SHAPExplanation],
        context: ExplanationContext,
    ) -> List[Dict[str, Any]]:
        """Generate suggested counterfactual queries from SHAP explanation."""
        if not shap_explanation:
            return []

        suggestions = []
        for fc in shap_explanation.feature_contributions:
            if fc.contribution < -0.05:  # Negative contribution
                if fc.feature == "study_time":
                    suggestions.append({
                        "intervention": {"study_time": fc.value * 1.5},
                        "question": "What if you had studied 50% longer?",
                    })
                elif fc.feature == "recency":
                    suggestions.append({
                        "intervention": {"hours_since_study": 12},
                        "question": "What if you had reviewed more recently?",
                    })
                elif fc.feature == "recent_accuracy":
                    suggestions.append({
                        "intervention": {"practice_sessions": 3},
                        "question": "What if you had done more practice problems?",
                    })

        return suggestions[:3]

    def _generate_natural_language_explanation(
        self,
        cf_result: CounterfactualResult,
        shap_explanation: Optional[SHAPExplanation],
        intervention: Dict[str, Any],
    ) -> str:
        """Generate natural language explanation from counterfactual result."""
        intervention_desc = ", ".join([f"{k} to {v}" for k, v in intervention.items()])

        if cf_result.probability_change > 0.1:
            return (
                f"If you had changed {intervention_desc}, your success probability "
                f"would have improved by {cf_result.probability_change:.0%}. "
                f"This suggests this factor significantly affects your performance."
            )
        elif cf_result.probability_change > 0:
            return (
                f"Changing {intervention_desc} would have slightly improved your "
                f"chances ({cf_result.probability_change:+.1%}), but other factors "
                f"may be more impactful."
            )
        else:
            return (
                f"Interestingly, changing {intervention_desc} wouldn't have "
                f"significantly improved the outcome. Other factors may be "
                f"more important to focus on."
            )

    def _generate_recommendation_explanation(
        self,
        shap_explanation: Optional[SHAPExplanation],
        arm_values: Dict[str, float],
        context: ExplanationContext,
    ) -> str:
        """Generate explanation for why a recommendation was made."""
        if not shap_explanation:
            return "This content was selected based on your current learning progress."

        # Find top contributing factors
        positive = [fc for fc in shap_explanation.feature_contributions if fc.contribution > 0]
        negative = [fc for fc in shap_explanation.feature_contributions if fc.contribution < 0]

        parts = []
        if positive:
            top_positive = max(positive, key=lambda fc: fc.contribution)
            parts.append(
                f"Your {top_positive.feature} ({top_positive.value:.2f}) "
                f"contributed positively to this recommendation."
            )

        if negative:
            top_negative = min(negative, key=lambda fc: fc.contribution)
            parts.append(
                f"Your {top_negative.feature} ({top_negative.value:.2f}) "
                f"suggested you might benefit from this content."
            )

        return " ".join(parts) if parts else "This content aligns with your current learning needs."

    def _interpret_feature_contribution(self, fc) -> str:
        """Generate human-readable interpretation of a feature contribution."""
        if fc.contribution > 0.1:
            return f"Strong positive factor"
        elif fc.contribution > 0:
            return f"Mild positive factor"
        elif fc.contribution > -0.1:
            return f"Mild negative factor"
        else:
            return f"Strong negative factor - consider addressing"

    def _recommend_path(self, comparison) -> str:
        """Generate recommendation based on path comparison."""
        alt_better = (
            hasattr(comparison, 'alternative_success_probability') and
            hasattr(comparison, 'original_success_probability') and
            comparison.alternative_success_probability > comparison.original_success_probability + 0.05
        )

        if alt_better:
            delta = comparison.alternative_success_probability - comparison.original_success_probability
            return f"Path B is recommended (+{delta:.1%} success probability)"
        else:
            return "Path A is slightly better or equivalent to Path B"


# Factory function for dependency injection
def get_counterfactual_service(
    td_bkt: Optional[TemporalDifferenceBKT] = None,
) -> CounterfactualService:
    """
    Factory function to create CounterfactualService with dependencies.

    Args:
        td_bkt: Optional TD-BKT instance

    Returns:
        Configured CounterfactualService
    """
    return CounterfactualService(td_bkt=td_bkt)
