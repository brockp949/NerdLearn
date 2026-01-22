"""
Teachable Agent - Feynman Protocol Implementation

Research alignment:
- Feynman Technique: "If you can't explain it simply, you don't understand it well enough"
- Learning by Teaching: Teaching deepens understanding more than passive review
- Self-Explanation Effect: Generating explanations improves retention
- Desirable Difficulties: Productive struggle during teaching enhances learning

The learner teaches a simulated "student" agent who:
1. Asks clarifying questions when confused
2. Requests concrete examples
3. Challenges assumptions and gaps in reasoning
4. Shows increasing understanding as explanations improve
"""
from typing import Dict, List, Any, Optional, TypedDict
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
import logging
import json

logger = logging.getLogger(__name__)


class StudentPersona(str, Enum):
    """Different simulated student personas for varied interactions"""
    CURIOUS = "curious"          # Asks lots of "why" questions
    CONFUSED = "confused"        # Needs multiple explanations
    CHALLENGER = "challenger"    # Questions assumptions
    VISUAL = "visual"            # Asks for diagrams/examples
    PRACTICAL = "practical"      # Wants real-world applications


class ComprehensionLevel(str, Enum):
    """Student's current understanding level"""
    LOST = "lost"               # < 20% - completely confused
    STRUGGLING = "struggling"   # 20-40% - getting fragments
    EMERGING = "emerging"       # 40-60% - grasping basics
    DEVELOPING = "developing"   # 60-80% - understanding well
    MASTERING = "mastering"     # 80%+ - near full understanding


class QuestionType(str, Enum):
    """Types of questions the student can ask"""
    CLARIFICATION = "clarification"     # "What do you mean by...?"
    EXAMPLE = "example"                 # "Can you give me an example?"
    CONNECTION = "connection"           # "How does this relate to...?"
    APPLICATION = "application"         # "When would I use this?"
    CHALLENGE = "challenge"             # "But what about...?"
    ELABORATION = "elaboration"         # "Can you explain more about...?"
    CONFIRMATION = "confirmation"       # "So you're saying that...?"


@dataclass
class TeachingExchange:
    """Single exchange in a teaching session"""
    timestamp: datetime
    learner_explanation: str
    student_response: str
    question_type: Optional[QuestionType]
    comprehension_before: float  # 0-1
    comprehension_after: float   # 0-1
    identified_gaps: List[str]
    key_concepts_covered: List[str]


@dataclass
class TeachingSession:
    """Complete teaching session state"""
    session_id: str
    user_id: str
    concept_id: str
    concept_name: str
    persona: StudentPersona

    # Progress tracking
    exchanges: List[TeachingExchange] = field(default_factory=list)
    current_comprehension: float = 0.0  # 0-1 scale
    comprehension_level: ComprehensionLevel = ComprehensionLevel.LOST

    # Learning analytics
    explanation_quality_scores: List[float] = field(default_factory=list)
    identified_misconceptions: List[str] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)
    strong_points: List[str] = field(default_factory=list)

    # Session metadata
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    completed: bool = False


class StudentResponse(BaseModel):
    """Structured response from the student agent"""
    response: str = Field(description="The student's verbal response")
    question_type: Optional[QuestionType] = Field(default=None, description="Type of question if asking one")
    comprehension_delta: float = Field(default=0.0, description="Change in comprehension (-0.1 to 0.3)")
    identified_gaps: List[str] = Field(default_factory=list, description="Knowledge gaps detected in explanation")
    concepts_understood: List[str] = Field(default_factory=list, description="Concepts the student now understands")
    confusion_points: List[str] = Field(default_factory=list, description="Points of confusion")
    thinking: str = Field(default="", description="Internal reasoning about the explanation")


class TeachableAgent:
    """
    Simulates a student learning from the user's explanations

    Implements the Feynman Protocol: teaching to learn
    """

    # Persona-specific behaviors
    PERSONA_PROMPTS = {
        StudentPersona.CURIOUS: """You are a curious learner who loves asking "why" and "how" questions.
You're genuinely interested in understanding the deeper reasons behind concepts.
You frequently ask follow-up questions like:
- "Why does that happen?"
- "How does that work exactly?"
- "What makes it work that way?"
- "I'm curious - what would happen if...?"
""",
        StudentPersona.CONFUSED: """You are a student who struggles to understand new concepts quickly.
You need things explained multiple ways before they click.
You often:
- Ask for things to be repeated or rephrased
- Admit when you're lost: "I'm sorry, I don't follow..."
- Need concepts broken down into smaller pieces
- Eventually have "aha!" moments when things finally click
""",
        StudentPersona.CHALLENGER: """You are a critical thinker who questions assumptions.
You push back on explanations to ensure they're complete and accurate.
You frequently:
- Ask "But what about...?" to probe edge cases
- Point out potential contradictions
- Request evidence or reasoning for claims
- Play devil's advocate to strengthen understanding
""",
        StudentPersona.VISUAL: """You are a visual learner who needs concrete examples and mental images.
Abstract explanations confuse you without examples.
You often ask:
- "Can you give me an example?"
- "What would that look like in practice?"
- "Can you draw that out or describe it visually?"
- "Is it like [analogy]?"
""",
        StudentPersona.PRACTICAL: """You are a practical learner focused on real-world applications.
You want to know how concepts apply to actual situations.
You frequently ask:
- "When would I actually use this?"
- "What's a real-world example?"
- "How would this help me solve [problem]?"
- "Why does this matter practically?"
"""
    }

    # Comprehension level thresholds
    COMPREHENSION_THRESHOLDS = {
        ComprehensionLevel.LOST: 0.2,
        ComprehensionLevel.STRUGGLING: 0.4,
        ComprehensionLevel.EMERGING: 0.6,
        ComprehensionLevel.DEVELOPING: 0.8,
        ComprehensionLevel.MASTERING: 1.0
    }

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.8,  # Higher for natural conversation
        llm: Optional[ChatOpenAI] = None
    ):
        """Initialize teachable agent"""
        self.llm = llm or ChatOpenAI(model=model, temperature=temperature)
        self.sessions: Dict[str, TeachingSession] = {}

    def start_session(
        self,
        user_id: str,
        concept_id: str,
        concept_name: str,
        persona: StudentPersona = StudentPersona.CURIOUS,
        concept_prerequisites: Optional[List[str]] = None,
        concept_description: Optional[str] = None
    ) -> TeachingSession:
        """
        Start a new teaching session

        Args:
            user_id: ID of the learner who is teaching
            concept_id: ID of the concept being taught
            concept_name: Human-readable concept name
            persona: Which student persona to use
            concept_prerequisites: What the student "already knows"
            concept_description: Brief description for the agent to evaluate against

        Returns:
            New teaching session
        """
        session_id = f"{user_id}_{concept_id}_{datetime.utcnow().timestamp()}"

        session = TeachingSession(
            session_id=session_id,
            user_id=user_id,
            concept_id=concept_id,
            concept_name=concept_name,
            persona=persona
        )

        self.sessions[session_id] = session
        logger.info(f"Started teaching session {session_id} for concept '{concept_name}'")

        return session

    async def get_opening_question(
        self,
        session: TeachingSession,
        concept_description: Optional[str] = None
    ) -> str:
        """
        Get the student's opening question to begin the teaching session

        Returns:
            Opening question from the simulated student
        """
        persona_prompt = self.PERSONA_PROMPTS[session.persona]

        system_prompt = f"""You are a student named Alex who wants to learn about "{session.concept_name}".

{persona_prompt}

You're about to have someone teach you this concept. Generate an opening question or statement
that shows you're ready to learn but don't understand the topic yet.

Keep your response to 1-2 sentences. Be natural and conversational.
Do NOT include any JSON or structured formatting.
"""

        context = f"Concept: {session.concept_name}"
        if concept_description:
            context += f"\nBrief description (for your understanding): {concept_description}"

        response = await self.llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Please ask your opening question about: {context}")
        ])

        return response.content.strip()

    async def process_explanation(
        self,
        session_id: str,
        explanation: str,
        concept_description: Optional[str] = None
    ) -> StudentResponse:
        """
        Process a learner's explanation and generate student response

        Args:
            session_id: Teaching session ID
            explanation: The learner's explanation attempt
            concept_description: Ground truth for evaluation (optional)

        Returns:
            Structured student response
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Build conversation history
        history = []
        for exchange in session.exchanges[-5:]:  # Last 5 exchanges for context
            history.append(HumanMessage(content=exchange.learner_explanation))
            history.append(AIMessage(content=exchange.student_response))

        # Create evaluation prompt
        persona_prompt = self.PERSONA_PROMPTS[session.persona]
        comprehension_context = self._get_comprehension_context(session)

        system_prompt = f"""You are Alex, a student learning about "{session.concept_name}".

{persona_prompt}

CURRENT UNDERSTANDING LEVEL: {session.comprehension_level.value} ({session.current_comprehension:.0%})
{comprehension_context}

Your task:
1. React naturally to the teacher's explanation
2. Evaluate how well it helps you understand
3. Ask follow-up questions based on your persona
4. Show progress when explanations are clear, confusion when they're not

IMPORTANT GUIDELINES:
- If the explanation is clear and complete, show understanding and ask a deeper question
- If the explanation is vague or incomplete, express confusion and ask for clarification
- If the explanation has errors, politely point out what doesn't make sense
- Be encouraging when the teacher explains well
- Your comprehension should increase with good explanations (0.05-0.2 per exchange)
- Bad or confusing explanations might decrease comprehension slightly

Respond with a JSON object containing:
{{
    "response": "Your natural conversational response to the teacher",
    "question_type": "clarification|example|connection|application|challenge|elaboration|confirmation" or null,
    "comprehension_delta": <number from -0.1 to 0.2>,
    "identified_gaps": ["list of knowledge gaps in the explanation"],
    "concepts_understood": ["concepts you now understand better"],
    "confusion_points": ["things that still confuse you"],
    "thinking": "Your internal assessment of the explanation quality"
}}
"""

        messages = [SystemMessage(content=system_prompt)] + history + [
            HumanMessage(content=f"Teacher's explanation:\n{explanation}")
        ]

        if concept_description:
            messages.append(SystemMessage(
                content=f"[Hidden concept definition for evaluation: {concept_description}]"
            ))

        response = await self.llm.ainvoke(messages)

        # Parse response
        try:
            # Try to extract JSON from response
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            result = json.loads(content)
            student_response = StudentResponse(**result)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse structured response: {e}")
            # Fallback to unstructured response
            student_response = StudentResponse(
                response=response.content,
                comprehension_delta=0.1,  # Assume slight progress
            )

        # Update session state
        new_comprehension = min(1.0, max(0.0,
            session.current_comprehension + student_response.comprehension_delta
        ))

        # Record exchange
        exchange = TeachingExchange(
            timestamp=datetime.utcnow(),
            learner_explanation=explanation,
            student_response=student_response.response,
            question_type=student_response.question_type,
            comprehension_before=session.current_comprehension,
            comprehension_after=new_comprehension,
            identified_gaps=student_response.identified_gaps,
            key_concepts_covered=student_response.concepts_understood
        )
        session.exchanges.append(exchange)

        # Update comprehension
        session.current_comprehension = new_comprehension
        session.comprehension_level = self._calculate_level(new_comprehension)

        # Track gaps and strengths
        session.knowledge_gaps.extend(
            g for g in student_response.identified_gaps
            if g not in session.knowledge_gaps
        )
        session.strong_points.extend(
            c for c in student_response.concepts_understood
            if c not in session.strong_points
        )

        logger.info(
            f"Session {session_id}: Comprehension {session.current_comprehension:.0%} "
            f"({session.comprehension_level.value})"
        )

        return student_response

    def _get_comprehension_context(self, session: TeachingSession) -> str:
        """Generate context description based on comprehension level"""
        level = session.comprehension_level

        contexts = {
            ComprehensionLevel.LOST: "You're completely lost and need the basics explained from scratch.",
            ComprehensionLevel.STRUGGLING: "You're getting fragments but missing the big picture. Need clearer explanations.",
            ComprehensionLevel.EMERGING: "You're starting to understand the basics but need more depth.",
            ComprehensionLevel.DEVELOPING: "You understand the core concepts well. Ready for advanced questions.",
            ComprehensionLevel.MASTERING: "You have a strong grasp. Looking for edge cases or deeper insights."
        }

        return contexts.get(level, "")

    def _calculate_level(self, comprehension: float) -> ComprehensionLevel:
        """Calculate comprehension level from score"""
        for level, threshold in self.COMPREHENSION_THRESHOLDS.items():
            if comprehension <= threshold:
                return level
        return ComprehensionLevel.MASTERING

    def end_session(self, session_id: str) -> Dict[str, Any]:
        """
        End a teaching session and generate summary

        Returns:
            Session summary with learning analytics
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        session.end_time = datetime.utcnow()
        session.completed = True

        # Calculate statistics
        total_exchanges = len(session.exchanges)
        comprehension_progress = [
            ex.comprehension_after for ex in session.exchanges
        ]

        # Identify teaching effectiveness
        improvement_per_exchange = (
            (session.current_comprehension - 0) / max(1, total_exchanges)
        )

        summary = {
            "session_id": session_id,
            "concept": session.concept_name,
            "persona_used": session.persona.value,
            "duration_minutes": (
                (session.end_time - session.start_time).total_seconds() / 60
            ),
            "total_exchanges": total_exchanges,
            "final_comprehension": session.current_comprehension,
            "comprehension_level": session.comprehension_level.value,
            "comprehension_progress": comprehension_progress,
            "improvement_per_exchange": improvement_per_exchange,
            "knowledge_gaps_identified": session.knowledge_gaps,
            "strong_explanations": session.strong_points,
            "teaching_effectiveness": self._calculate_teaching_score(session),
            "recommendations": self._generate_recommendations(session)
        }

        logger.info(f"Ended session {session_id}: {summary['teaching_effectiveness']:.0%} effective")

        return summary

    def _calculate_teaching_score(self, session: TeachingSession) -> float:
        """
        Calculate overall teaching effectiveness score

        Factors:
        - Final comprehension achieved
        - Efficiency (fewer exchanges = better)
        - Consistency of improvement
        """
        if not session.exchanges:
            return 0.0

        # Final comprehension (50% weight)
        comprehension_score = session.current_comprehension * 0.5

        # Efficiency (25% weight) - reaching good comprehension quickly
        exchanges = len(session.exchanges)
        efficiency_score = max(0, 1 - (exchanges - 3) * 0.1) * 0.25  # Ideal: 3-5 exchanges

        # Consistency (25% weight) - steady improvement vs. volatility
        if len(session.exchanges) >= 2:
            deltas = [
                ex.comprehension_after - ex.comprehension_before
                for ex in session.exchanges
            ]
            positive_deltas = sum(1 for d in deltas if d > 0) / len(deltas)
            consistency_score = positive_deltas * 0.25
        else:
            consistency_score = 0.125  # Neutral if not enough data

        return comprehension_score + efficiency_score + consistency_score

    def _generate_recommendations(self, session: TeachingSession) -> List[str]:
        """Generate recommendations for improving teaching ability"""
        recommendations = []

        # Based on comprehension level
        if session.current_comprehension < 0.4:
            recommendations.append(
                "Consider reviewing the fundamentals before teaching this concept."
            )

        # Based on gaps
        if session.knowledge_gaps:
            recommendations.append(
                f"Focus on clarifying: {', '.join(session.knowledge_gaps[:3])}"
            )

        # Based on exchange count
        if len(session.exchanges) > 8:
            recommendations.append(
                "Try to explain more concisely - use analogies and examples."
            )
        elif len(session.exchanges) < 3 and session.current_comprehension < 0.6:
            recommendations.append(
                "Provide more thorough explanations with multiple examples."
            )

        # Based on persona
        persona_tips = {
            StudentPersona.CURIOUS: "Great curiosity! Channel this into your own learning.",
            StudentPersona.CONFUSED: "Practice breaking down complex ideas into simpler parts.",
            StudentPersona.CHALLENGER: "Your explanations handle edge cases well!",
            StudentPersona.VISUAL: "Incorporate more visual analogies and concrete examples.",
            StudentPersona.PRACTICAL: "Strong real-world connections strengthen understanding."
        }
        recommendations.append(persona_tips[session.persona])

        return recommendations

    def get_session(self, session_id: str) -> Optional[TeachingSession]:
        """Get session by ID"""
        return self.sessions.get(session_id)


# Singleton instance
_teachable_agent: Optional[TeachableAgent] = None


def get_teachable_agent() -> TeachableAgent:
    """Get or create the teachable agent singleton"""
    global _teachable_agent
    if _teachable_agent is None:
        _teachable_agent = TeachableAgent()
    return _teachable_agent
