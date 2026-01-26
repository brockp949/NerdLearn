"""
Socratic Prompt Templates for Counterfactual Explanations

This module provides structured prompt templates that convert causal
data (SCM states, counterfactual results, SHAP explanations, recourse plans)
into pedagogically-sound Socratic dialogue prompts.

The Socratic method asks questions to guide discovery rather than stating facts.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class DialogueMode(str, Enum):
    """Types of Socratic dialogue modes."""
    RETROSPECTIVE = "retrospective"  # What went wrong?
    PROSPECTIVE = "prospective"      # What to do next?
    EXPLORATORY = "exploratory"      # General understanding
    CHALLENGE = "challenge"          # Student disagrees


@dataclass
class SocraticContext:
    """
    Context for Socratic dialogue generation.

    Aggregates all causal information needed to generate
    meaningful pedagogical dialogue.
    """
    # Student state
    student_id: str
    concept_id: str
    concept_name: str
    current_mastery: float
    target_mastery: float = 0.85

    # Recent history
    recent_attempts: List[Dict[str, Any]] = None
    failures: List[Dict[str, Any]] = None
    study_sessions: List[Dict[str, Any]] = None

    # Causal analysis
    shap_explanation: Optional[Dict[str, Any]] = None
    counterfactual_result: Optional[Dict[str, Any]] = None
    recourse_plan: Optional[Dict[str, Any]] = None
    critical_decisions: Optional[List[Dict[str, Any]]] = None

    # Conversation state
    dialogue_history: List[Dict[str, str]] = None
    student_challenge: Optional[str] = None

    def __post_init__(self):
        self.recent_attempts = self.recent_attempts or []
        self.failures = self.failures or []
        self.study_sessions = self.study_sessions or []
        self.dialogue_history = self.dialogue_history or []


class SocraticPrompts:
    """
    Prompt templates for Socratic dialogue generation.

    Converts structured causal data into prompts that guide
    an LLM to generate pedagogically appropriate dialogue.
    """

    @staticmethod
    def system_prompt() -> str:
        """Base system prompt establishing Socratic tutor persona."""
        return """You are a Socratic tutor helping a student understand their learning journey.

Your role is to:
1. Ask questions that guide the student to discover insights themselves
2. Never state facts directly - always phrase as questions
3. Use the causal data provided to identify root causes and opportunities
4. Be encouraging while honest about areas needing improvement
5. Connect observations to actionable learning strategies
6. Respect student autonomy - acknowledge when they have valid points

Socratic principles:
- "What do you think might have contributed to...?"
- "Have you noticed any patterns in...?"
- "What if you had...?"
- "How might X relate to Y?"

Never say things like:
- "You failed because..."
- "The data shows that..."
- "You should do X, Y, Z"

Always respond in a conversational, warm tone appropriate for education."""

    @staticmethod
    def retrospective_prompt(context: SocraticContext) -> str:
        """
        Generate prompt for retrospective analysis: "What went wrong?"

        Uses counterfactual results and critical decision points to
        guide the student to discover root causes.
        """
        # Serialize failure history
        failure_text = ""
        if context.failures:
            failure_text = "\n".join([
                f"- {f.get('timestamp', 'Unknown time')}: {f.get('question_type', 'Question')} "
                f"on {f.get('concept', context.concept_name)} (difficulty: {f.get('difficulty', 'medium')})"
                for f in context.failures[:5]  # Last 5 failures
            ])

        # Serialize SHAP explanation
        shap_text = ""
        if context.shap_explanation:
            contributions = context.shap_explanation.get('feature_contributions', [])
            shap_text = "\n".join([
                f"- {c.get('feature')}: {c.get('contribution'):+.3f} "
                f"({'helped' if c.get('contribution', 0) > 0 else 'hindered'})"
                for c in contributions[:5]
            ])

        # Serialize counterfactual
        cf_text = ""
        if context.counterfactual_result:
            cf = context.counterfactual_result
            cf_text = f"""
Counterfactual scenario analyzed:
- Intervention: {cf.get('intervention', 'N/A')}
- Original outcome probability: {cf.get('original_probability', 0):.1%}
- Counterfactual probability: {cf.get('counterfactual_probability', 0):.1%}
- Probability change: {cf.get('probability_change', 0):+.1%}
"""

        # Serialize critical decisions
        critical_text = ""
        if context.critical_decisions:
            critical_text = "\nCritical decision points identified:\n" + "\n".join([
                f"- {d.get('timestamp', 'Unknown')}: {d.get('description', 'Decision')} "
                f"(impact: {d.get('impact_score', 0):.2f})"
                for d in context.critical_decisions[:3]
            ])

        return f"""RETROSPECTIVE ANALYSIS CONTEXT

Student is reviewing their learning journey for: {context.concept_name}
Current mastery: {context.current_mastery:.1%}
Target mastery: {context.target_mastery:.1%}
Gap to close: {context.target_mastery - context.current_mastery:.1%}

Recent failures:
{failure_text if failure_text else "No recent failures recorded."}

Factors influencing outcomes (SHAP attribution):
{shap_text if shap_text else "No attribution data available."}
{cf_text}
{critical_text}

INSTRUCTIONS:
Generate Socratic questions that help the student discover:
1. What patterns exist in their recent attempts?
2. What factors (from SHAP) might have contributed to difficulty?
3. What the counterfactual analysis suggests about alternative approaches?
4. What they might do differently next time?

Remember: Ask questions, don't state conclusions. Let the student discover insights.
Start with a warm, non-judgmental opening question about their experience."""

    @staticmethod
    def prospective_prompt(context: SocraticContext) -> str:
        """
        Generate prompt for prospective planning: "What to do next?"

        Uses recourse plan and SHAP to guide student toward
        actionable next steps.
        """
        # Serialize recourse plan
        recourse_text = ""
        if context.recourse_plan:
            plan = context.recourse_plan
            actions = plan.get('actions', [])
            recourse_text = f"""
Recommended actions (by impact/effort ratio):
{chr(10).join([
    f"- {a.get('action')}: effort={a.get('effort', 0):.1f}, "
    f"expected impact={a.get('expected_impact', 0):+.1%}"
    for a in actions[:4]
])}

Total estimated effort: {plan.get('total_effort', 0):.1f} units
Time estimate: {plan.get('time_estimate_minutes', 0):.0f} minutes
Expected probability after completion: {plan.get('expected_probability', 0):.1%}
"""

        # Study session analysis
        study_text = ""
        if context.study_sessions:
            total_time = sum(s.get('duration_minutes', 0) for s in context.study_sessions)
            avg_gap = sum(s.get('gap_hours', 24) for s in context.study_sessions) / max(len(context.study_sessions), 1)
            study_text = f"""
Recent study patterns:
- Total study time: {total_time:.0f} minutes
- Average gap between sessions: {avg_gap:.1f} hours
- Number of sessions: {len(context.study_sessions)}
"""

        # SHAP for actionable features
        actionable_text = ""
        if context.shap_explanation:
            contributions = context.shap_explanation.get('feature_contributions', [])
            # Filter to actionable features (not past performance)
            actionable = [c for c in contributions
                         if c.get('feature') in ['study_time', 'practice_frequency',
                                                   'spacing_interval', 'content_difficulty']]
            if actionable:
                actionable_text = "\nActionable factors from analysis:\n" + "\n".join([
                    f"- {c.get('feature')}: current impact {c.get('contribution'):+.3f}"
                    for c in actionable
                ])

        return f"""PROSPECTIVE PLANNING CONTEXT

Student is planning their next steps for: {context.concept_name}
Current mastery: {context.current_mastery:.1%}
Target mastery: {context.target_mastery:.1%}
Gap to close: {context.target_mastery - context.current_mastery:.1%}
{study_text}
{recourse_text}
{actionable_text}

INSTRUCTIONS:
Generate Socratic questions that help the student:
1. Reflect on what study strategies have worked for them before
2. Consider how they might adjust their approach based on the patterns
3. Set realistic, specific goals for their next study session
4. Think about spacing and timing of practice

Remember: Guide toward the recommended actions without prescribing them directly.
Ask questions like "What if you tried..." rather than "You should..."
Start by acknowledging their current progress and asking about their goals."""

    @staticmethod
    def exploratory_prompt(context: SocraticContext) -> str:
        """
        Generate prompt for general exploration.

        Used when student wants to understand their learning
        without specific failure or planning focus.
        """
        return f"""EXPLORATORY LEARNING CONTEXT

Student wants to understand their learning journey for: {context.concept_name}
Current mastery: {context.current_mastery:.1%}
Target mastery: {context.target_mastery:.1%}

Recent activity:
- Attempts: {len(context.recent_attempts)}
- Study sessions: {len(context.study_sessions)}

INSTRUCTIONS:
Generate open-ended Socratic questions that help the student:
1. Reflect on their overall experience with this concept
2. Identify what aspects they find challenging or interesting
3. Connect this concept to other things they've learned
4. Think about how they learn best

Be curious and open. Let the conversation flow naturally.
Start with a broad question about their experience with {context.concept_name}."""

    @staticmethod
    def challenge_response_prompt(context: SocraticContext) -> str:
        """
        Generate prompt for handling student challenges/disagreements.

        Used when student provides information that contradicts
        the causal model (e.g., "But I did study for hours!").
        """
        challenge = context.student_challenge or "The student has raised a concern."

        return f"""STUDENT CHALLENGE CONTEXT

Student is discussing: {context.concept_name}
Current mastery: {context.current_mastery:.1%}

Student's challenge: "{challenge}"

Previous dialogue:
{chr(10).join([f"- {d.get('role', 'unknown')}: {d.get('content', '')}"
               for d in (context.dialogue_history or [])[-4:]])}

INSTRUCTIONS:
The student has raised a valid point or disagreement. You must:

1. ACKNOWLEDGE their perspective genuinely - they may have information the model doesn't
2. ASK clarifying questions to understand their experience better
3. EXPLORE what this might mean for updating our understanding
4. AVOID being defensive about the analysis

Possible responses to consider:
- "That's really helpful to know. Can you tell me more about...?"
- "You're right that might change things. What kind of studying were you doing?"
- "I appreciate you sharing that. How did those study sessions feel to you?"

Never dismiss the student's experience. The causal model is an estimate,
and the student's lived experience provides important information.

Respond with curiosity and openness to revising the analysis."""

    @staticmethod
    def serialize_for_llm(context: SocraticContext, mode: DialogueMode) -> Dict[str, str]:
        """
        Serialize context into system and user prompts for LLM.

        Args:
            context: The SocraticContext with all relevant data
            mode: The type of dialogue to generate

        Returns:
            Dict with 'system' and 'user' prompt strings
        """
        system = SocraticPrompts.system_prompt()

        if mode == DialogueMode.RETROSPECTIVE:
            user = SocraticPrompts.retrospective_prompt(context)
        elif mode == DialogueMode.PROSPECTIVE:
            user = SocraticPrompts.prospective_prompt(context)
        elif mode == DialogueMode.CHALLENGE:
            user = SocraticPrompts.challenge_response_prompt(context)
        else:
            user = SocraticPrompts.exploratory_prompt(context)

        # Add dialogue history if continuing conversation
        if context.dialogue_history:
            history_text = "\n\nPREVIOUS DIALOGUE:\n" + "\n".join([
                f"{d.get('role', 'unknown').upper()}: {d.get('content', '')}"
                for d in context.dialogue_history[-6:]  # Last 6 turns
            ])
            user += history_text
            user += "\n\nContinue the conversation with your next Socratic question."

        return {
            "system": system,
            "user": user
        }


def format_shap_for_dialogue(shap_explanation: Dict[str, Any]) -> str:
    """
    Convert SHAP explanation into natural language for dialogue context.

    Args:
        shap_explanation: SHAP explanation dictionary

    Returns:
        Human-readable summary of feature contributions
    """
    contributions = shap_explanation.get('feature_contributions', [])
    if not contributions:
        return "No specific factors identified."

    # Sort by absolute contribution
    sorted_contrib = sorted(contributions,
                           key=lambda x: abs(x.get('contribution', 0)),
                           reverse=True)

    positive = [c for c in sorted_contrib if c.get('contribution', 0) > 0]
    negative = [c for c in sorted_contrib if c.get('contribution', 0) < 0]

    parts = []
    if positive:
        factors = ", ".join([c.get('feature', 'unknown') for c in positive[:2]])
        parts.append(f"Helping factors: {factors}")
    if negative:
        factors = ", ".join([c.get('feature', 'unknown') for c in negative[:2]])
        parts.append(f"Challenging factors: {factors}")

    return "; ".join(parts)


def format_counterfactual_for_dialogue(cf_result: Dict[str, Any]) -> str:
    """
    Convert counterfactual result into natural language question.

    Args:
        cf_result: Counterfactual result dictionary

    Returns:
        Socratic question based on the counterfactual
    """
    intervention = cf_result.get('intervention', {})
    prob_change = cf_result.get('probability_change', 0)

    # Extract intervention details
    if isinstance(intervention, dict):
        intervention_desc = ", ".join([
            f"{k}={v}" for k, v in intervention.items()
        ])
    else:
        intervention_desc = str(intervention)

    direction = "improved" if prob_change > 0 else "changed"

    return (
        f"What if {intervention_desc}? "
        f"The analysis suggests this would have {direction} outcomes by {abs(prob_change):.1%}."
    )


def format_recourse_for_dialogue(recourse_plan: Dict[str, Any]) -> str:
    """
    Convert recourse plan into actionable Socratic framing.

    Args:
        recourse_plan: Recourse plan dictionary

    Returns:
        Socratic framing of the recommended actions
    """
    actions = recourse_plan.get('actions', [])
    if not actions:
        return "Let's explore what strategies might help."

    # Pick top action by impact/effort
    top_action = max(actions,
                     key=lambda a: a.get('expected_impact', 0) / max(a.get('effort', 1), 0.1))

    action_desc = top_action.get('action', 'adjust your approach')
    impact = top_action.get('expected_impact', 0)

    return (
        f"What do you think might happen if you {action_desc}? "
        f"This could potentially improve your success rate by about {impact:.1%}."
    )
