"""
Socratic Agent for Counterfactual Dialogue Generation

This module implements a Socratic conversational agent that converts
causal insights (SCM states, counterfactual results, SHAP explanations,
recourse plans) into pedagogically appropriate dialogue.

The agent uses LLM to generate questions that guide students to
discover insights themselves, following the Socratic method.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import json
import os

from app.adaptive.counterfactual.socratic_prompts import (
    SocraticContext,
    SocraticPrompts,
    DialogueMode,
    format_shap_for_dialogue,
    format_counterfactual_for_dialogue,
    format_recourse_for_dialogue,
)

logger = logging.getLogger(__name__)


@dataclass
class DialogueTurn:
    """A single turn in the Socratic dialogue."""
    role: str  # "tutor" or "student"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SocraticDialogue:
    """Complete Socratic dialogue session."""
    session_id: str
    student_id: str
    concept_id: str
    mode: DialogueMode
    turns: List[DialogueTurn] = field(default_factory=list)
    context: Optional[SocraticContext] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def add_turn(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a turn to the dialogue."""
        self.turns.append(DialogueTurn(
            role=role,
            content=content,
            metadata=metadata or {}
        ))
        self.updated_at = datetime.utcnow()

    def to_history(self) -> List[Dict[str, str]]:
        """Convert turns to dialogue history format."""
        return [{"role": t.role, "content": t.content} for t in self.turns]


class LLMClient:
    """
    Abstract LLM client interface.

    Supports multiple backends (OpenAI, Anthropic, local models).
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        """Lazy initialization of LLM client."""
        if self._client is None:
            if self.provider == "openai":
                try:
                    from openai import OpenAI
                    self._client = OpenAI(
                        api_key=self.api_key,
                        base_url=self.base_url
                    )
                except ImportError:
                    logger.warning("OpenAI package not installed, using mock client")
                    self._client = MockLLMClient()
            elif self.provider == "anthropic":
                try:
                    from anthropic import Anthropic
                    self._client = Anthropic(api_key=self.api_key)
                except ImportError:
                    logger.warning("Anthropic package not installed, using mock client")
                    self._client = MockLLMClient()
            else:
                self._client = MockLLMClient()

        return self._client

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        dialogue_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            system_prompt: System-level instructions
            user_prompt: User message / context
            dialogue_history: Previous conversation turns

        Returns:
            Generated response text
        """
        client = self._get_client()

        if isinstance(client, MockLLMClient):
            return client.generate_mock_response(user_prompt)

        messages = [{"role": "system", "content": system_prompt}]

        # Add dialogue history
        if dialogue_history:
            for turn in dialogue_history:
                role = "assistant" if turn.get("role") == "tutor" else "user"
                messages.append({"role": role, "content": turn.get("content", "")})

        messages.append({"role": "user", "content": user_prompt})

        try:
            if self.provider == "openai":
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content

            elif self.provider == "anthropic":
                response = client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system_prompt,
                    messages=messages[1:],  # Skip system message
                )
                return response.content[0].text

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._fallback_response()

    def _fallback_response(self) -> str:
        """Fallback response when LLM fails."""
        return (
            "I'd like to understand your experience better. "
            "What aspects of this topic have you found most challenging?"
        )


class MockLLMClient:
    """Mock LLM client for testing without API access."""

    def generate_mock_response(self, prompt: str) -> str:
        """Generate a mock Socratic response based on prompt keywords."""
        prompt_lower = prompt.lower()

        if "retrospective" in prompt_lower:
            return (
                "Looking back at your recent work, I'm curious about something. "
                "When you were working through these problems, what felt most "
                "challenging? Was it the concepts themselves, or perhaps something "
                "about the timing or approach?"
            )
        elif "prospective" in prompt_lower:
            return (
                "As you think about moving forward, what strategies have worked "
                "well for you in the past? I'm wondering if there might be "
                "opportunities to build on those strengths here."
            )
        elif "challenge" in prompt_lower:
            return (
                "That's a really important point you're raising. Tell me more "
                "about that experience - what was that study session like for you? "
                "Understanding your perspective will help us figure out what's "
                "going on."
            )
        else:
            return (
                "What has your experience been like learning this concept so far? "
                "I'm curious to hear what aspects have resonated with you and "
                "which parts have felt more difficult."
            )


class SocraticAgent:
    """
    Socratic conversational agent for learning path explanations.

    Converts causal insights from the counterfactual framework into
    pedagogically appropriate Socratic dialogue.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_dialogue_turns: int = 20,
    ):
        """
        Initialize the Socratic agent.

        Args:
            llm_client: LLM client for generation (uses default if None)
            max_dialogue_turns: Maximum turns before suggesting break
        """
        self.llm_client = llm_client or LLMClient()
        self.max_dialogue_turns = max_dialogue_turns
        self.active_dialogues: Dict[str, SocraticDialogue] = {}

    async def generate_dialogue(
        self,
        context: SocraticContext,
        mode: DialogueMode = DialogueMode.EXPLORATORY,
        session_id: Optional[str] = None,
    ) -> SocraticDialogue:
        """
        Generate a new Socratic dialogue or continue an existing one.

        Args:
            context: The SocraticContext with all causal data
            mode: Type of dialogue (retrospective, prospective, etc.)
            session_id: Optional session ID to continue existing dialogue

        Returns:
            SocraticDialogue with generated response
        """
        # Resume existing dialogue or create new
        if session_id and session_id in self.active_dialogues:
            dialogue = self.active_dialogues[session_id]
            context.dialogue_history = dialogue.to_history()
        else:
            import uuid
            session_id = session_id or str(uuid.uuid4())
            dialogue = SocraticDialogue(
                session_id=session_id,
                student_id=context.student_id,
                concept_id=context.concept_id,
                mode=mode,
                context=context,
            )
            self.active_dialogues[session_id] = dialogue

        # Check if we should suggest a break
        if len(dialogue.turns) >= self.max_dialogue_turns:
            dialogue.add_turn(
                role="tutor",
                content=(
                    "We've covered a lot of ground together. "
                    "Would you like to take a break and reflect on what we've discussed, "
                    "or is there something specific you'd still like to explore?"
                ),
                metadata={"suggested_break": True}
            )
            return dialogue

        # Generate prompts
        prompts = SocraticPrompts.serialize_for_llm(context, mode)

        # Generate response
        response = await self.llm_client.generate(
            system_prompt=prompts["system"],
            user_prompt=prompts["user"],
            dialogue_history=dialogue.to_history(),
        )

        # Add tutor response
        dialogue.add_turn(
            role="tutor",
            content=response,
            metadata={
                "mode": mode.value,
                "context_summary": self._summarize_context(context),
            }
        )

        return dialogue

    async def respond_to_challenge(
        self,
        session_id: str,
        student_message: str,
    ) -> SocraticDialogue:
        """
        Handle a student response or challenge.

        This is called when the student provides additional information
        that might contradict or supplement the causal model.

        Args:
            session_id: The dialogue session ID
            student_message: The student's message

        Returns:
            Updated SocraticDialogue with response
        """
        if session_id not in self.active_dialogues:
            raise ValueError(f"No active dialogue for session: {session_id}")

        dialogue = self.active_dialogues[session_id]

        # Add student message
        dialogue.add_turn(
            role="student",
            content=student_message,
        )

        # Update context with the challenge
        context = dialogue.context
        if context:
            context.student_challenge = student_message
            context.dialogue_history = dialogue.to_history()

            # Detect if this is a challenge/correction
            is_challenge = self._detect_challenge(student_message)

            if is_challenge:
                mode = DialogueMode.CHALLENGE
            else:
                mode = dialogue.mode

            # Generate response
            prompts = SocraticPrompts.serialize_for_llm(context, mode)

            response = await self.llm_client.generate(
                system_prompt=prompts["system"],
                user_prompt=prompts["user"],
                dialogue_history=dialogue.to_history()[:-1],  # Exclude the just-added student message
            )

            dialogue.add_turn(
                role="tutor",
                content=response,
                metadata={
                    "mode": mode.value,
                    "detected_challenge": is_challenge,
                }
            )

        return dialogue

    async def generate_summary(self, session_id: str) -> str:
        """
        Generate a summary of the dialogue session.

        Args:
            session_id: The dialogue session ID

        Returns:
            Summary text of key insights and next steps
        """
        if session_id not in self.active_dialogues:
            raise ValueError(f"No active dialogue for session: {session_id}")

        dialogue = self.active_dialogues[session_id]

        if len(dialogue.turns) < 2:
            return "Not enough dialogue to summarize yet."

        summary_prompt = f"""Summarize this Socratic dialogue about learning {dialogue.context.concept_name if dialogue.context else 'a concept'}.

Dialogue:
{self._format_dialogue_for_summary(dialogue)}

Provide a brief summary (2-3 sentences) covering:
1. Key insights the student discovered
2. Agreed-upon next steps (if any)
3. Any outstanding questions to explore later

Keep it encouraging and action-oriented."""

        response = await self.llm_client.generate(
            system_prompt="You are a learning assistant summarizing a tutoring session.",
            user_prompt=summary_prompt,
        )

        return response

    def close_dialogue(self, session_id: str) -> Optional[SocraticDialogue]:
        """
        Close and return a dialogue session.

        Args:
            session_id: The dialogue session ID

        Returns:
            The closed dialogue or None if not found
        """
        return self.active_dialogues.pop(session_id, None)

    def get_dialogue(self, session_id: str) -> Optional[SocraticDialogue]:
        """Get an active dialogue by session ID."""
        return self.active_dialogues.get(session_id)

    def _detect_challenge(self, message: str) -> bool:
        """
        Detect if a student message is a challenge/correction.

        Uses keyword matching; could be enhanced with NLI model.
        """
        challenge_indicators = [
            "but i did",
            "that's not right",
            "actually",
            "i already",
            "i disagree",
            "that's not what happened",
            "you're wrong",
            "the data is wrong",
            "i don't think",
            "that doesn't match",
            "i spent",
            "i studied",
        ]

        message_lower = message.lower()
        return any(indicator in message_lower for indicator in challenge_indicators)

    def _summarize_context(self, context: SocraticContext) -> Dict[str, Any]:
        """Create a brief summary of the context for metadata."""
        return {
            "concept": context.concept_name,
            "mastery": context.current_mastery,
            "gap": context.target_mastery - context.current_mastery,
            "has_shap": context.shap_explanation is not None,
            "has_counterfactual": context.counterfactual_result is not None,
            "has_recourse": context.recourse_plan is not None,
        }

    def _format_dialogue_for_summary(self, dialogue: SocraticDialogue) -> str:
        """Format dialogue turns for summarization."""
        return "\n".join([
            f"{t.role.upper()}: {t.content}"
            for t in dialogue.turns
        ])


class SocraticDialogueBuilder:
    """
    Builder for constructing SocraticContext from various sources.

    Provides convenient methods to aggregate data from SCM,
    counterfactual engine, SHAP, and recourse into a context.
    """

    def __init__(self, student_id: str, concept_id: str, concept_name: str):
        self.context = SocraticContext(
            student_id=student_id,
            concept_id=concept_id,
            concept_name=concept_name,
            current_mastery=0.0,
        )

    def with_mastery(self, current: float, target: float = 0.85) -> "SocraticDialogueBuilder":
        """Set mastery levels."""
        self.context.current_mastery = current
        self.context.target_mastery = target
        return self

    def with_attempts(self, attempts: List[Dict[str, Any]]) -> "SocraticDialogueBuilder":
        """Add recent attempt history."""
        self.context.recent_attempts = attempts
        self.context.failures = [a for a in attempts if not a.get('correct', True)]
        return self

    def with_study_sessions(self, sessions: List[Dict[str, Any]]) -> "SocraticDialogueBuilder":
        """Add study session history."""
        self.context.study_sessions = sessions
        return self

    def with_shap_explanation(self, explanation: Dict[str, Any]) -> "SocraticDialogueBuilder":
        """Add SHAP explanation data."""
        self.context.shap_explanation = explanation
        return self

    def with_counterfactual(self, result: Dict[str, Any]) -> "SocraticDialogueBuilder":
        """Add counterfactual analysis result."""
        self.context.counterfactual_result = result
        return self

    def with_recourse_plan(self, plan: Dict[str, Any]) -> "SocraticDialogueBuilder":
        """Add recourse plan."""
        self.context.recourse_plan = plan
        return self

    def with_critical_decisions(self, decisions: List[Dict[str, Any]]) -> "SocraticDialogueBuilder":
        """Add critical decision points."""
        self.context.critical_decisions = decisions
        return self

    def build(self) -> SocraticContext:
        """Build and return the SocraticContext."""
        return self.context


# Convenience functions for quick dialogue generation

async def explain_failure_socratically(
    student_id: str,
    concept_id: str,
    concept_name: str,
    current_mastery: float,
    shap_explanation: Optional[Dict[str, Any]] = None,
    counterfactual_result: Optional[Dict[str, Any]] = None,
    failures: Optional[List[Dict[str, Any]]] = None,
    agent: Optional[SocraticAgent] = None,
) -> SocraticDialogue:
    """
    Quick function to generate retrospective Socratic dialogue for a failure.

    Args:
        student_id: Student identifier
        concept_id: Concept identifier
        concept_name: Human-readable concept name
        current_mastery: Current mastery level (0-1)
        shap_explanation: Optional SHAP explanation
        counterfactual_result: Optional counterfactual result
        failures: Optional list of failure events
        agent: Optional pre-configured SocraticAgent

    Returns:
        SocraticDialogue with initial tutor question
    """
    context = (
        SocraticDialogueBuilder(student_id, concept_id, concept_name)
        .with_mastery(current_mastery)
        .with_shap_explanation(shap_explanation or {})
        .with_counterfactual(counterfactual_result or {})
        .with_attempts(failures or [])
        .build()
    )

    agent = agent or SocraticAgent()
    return await agent.generate_dialogue(context, DialogueMode.RETROSPECTIVE)


async def plan_next_steps_socratically(
    student_id: str,
    concept_id: str,
    concept_name: str,
    current_mastery: float,
    recourse_plan: Optional[Dict[str, Any]] = None,
    study_sessions: Optional[List[Dict[str, Any]]] = None,
    agent: Optional[SocraticAgent] = None,
) -> SocraticDialogue:
    """
    Quick function to generate prospective Socratic dialogue for planning.

    Args:
        student_id: Student identifier
        concept_id: Concept identifier
        concept_name: Human-readable concept name
        current_mastery: Current mastery level (0-1)
        recourse_plan: Optional recourse plan
        study_sessions: Optional list of study sessions
        agent: Optional pre-configured SocraticAgent

    Returns:
        SocraticDialogue with initial tutor question
    """
    context = (
        SocraticDialogueBuilder(student_id, concept_id, concept_name)
        .with_mastery(current_mastery)
        .with_recourse_plan(recourse_plan or {})
        .with_study_sessions(study_sessions or [])
        .build()
    )

    agent = agent or SocraticAgent()
    return await agent.generate_dialogue(context, DialogueMode.PROSPECTIVE)
