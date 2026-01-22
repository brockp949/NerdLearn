"""
SimClass Multi-Agent Debate System

Research alignment:
- Socratic Dialogue: Learning through structured argumentation
- Cognitive Conflict: Exposure to contrasting viewpoints deepens understanding
- Elaborative Interrogation: Defending positions strengthens knowledge
- Perspective-Taking: Understanding multiple views enhances critical thinking
- Active Learning: Participation in debates increases engagement

SimClass simulates a classroom debate where multiple AI agents with
distinct perspectives argue different sides of a topic, allowing
learners to observe and optionally participate.
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
import asyncio

logger = logging.getLogger(__name__)


class DebateRole(str, Enum):
    """Roles agents can take in a debate"""
    ADVOCATE = "advocate"           # Argues in favor
    SKEPTIC = "skeptic"             # Questions and challenges
    SYNTHESIZER = "synthesizer"     # Finds common ground
    HISTORIAN = "historian"         # Provides historical context
    FUTURIST = "futurist"           # Speculates on implications
    PRACTITIONER = "practitioner"   # Focuses on practical applications
    THEORIST = "theorist"           # Emphasizes underlying principles
    CONTRARIAN = "contrarian"       # Deliberately takes opposite view


class DebateFormat(str, Enum):
    """Different debate formats"""
    OXFORD = "oxford"               # Formal pro/con structure
    SOCRATIC = "socratic"           # Question-based exploration
    ROUNDTABLE = "roundtable"       # Open discussion among equals
    DEVILS_ADVOCATE = "devils_advocate"  # Challenge the consensus
    SYNTHESIS = "synthesis"         # Find common ground


class ArgumentStrength(str, Enum):
    """How strong an argument is rated"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    COMPELLING = "compelling"


@dataclass
class DebateArgument:
    """Single argument in the debate"""
    speaker_id: str
    speaker_role: DebateRole
    content: str
    timestamp: datetime
    responding_to: Optional[str] = None  # ID of argument being responded to
    argument_type: str = "statement"  # statement, rebuttal, question, evidence
    key_points: List[str] = field(default_factory=list)
    evidence_cited: List[str] = field(default_factory=list)
    strength: Optional[ArgumentStrength] = None


@dataclass
class DebateAgent:
    """Configuration for a debate participant agent"""
    agent_id: str
    name: str
    role: DebateRole
    personality: str  # Brief personality description
    expertise: List[str]
    stance: str  # Their position on the topic

    def __hash__(self):
        return hash(self.agent_id)


@dataclass
class DebateSession:
    """Complete debate session state"""
    session_id: str
    topic: str
    format: DebateFormat
    agents: List[DebateAgent]
    arguments: List[DebateArgument] = field(default_factory=list)

    # Learner participation
    learner_id: Optional[str] = None
    learner_arguments: List[DebateArgument] = field(default_factory=list)

    # Session state
    current_round: int = 1
    max_rounds: int = 5
    current_speaker_idx: int = 0
    paused: bool = False
    completed: bool = False

    # Analytics
    key_insights: List[str] = field(default_factory=list)
    consensus_points: List[str] = field(default_factory=list)
    disagreement_points: List[str] = field(default_factory=list)

    # Metadata
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None


class DebateContribution(BaseModel):
    """Structured debate contribution from an agent"""
    content: str = Field(description="The argument or response")
    argument_type: str = Field(description="statement, rebuttal, question, evidence, concession")
    key_points: List[str] = Field(default_factory=list, description="Main points made")
    evidence: List[str] = Field(default_factory=list, description="Evidence or examples cited")
    acknowledges: List[str] = Field(default_factory=list, description="Points acknowledged from others")
    challenges: List[str] = Field(default_factory=list, description="Points being challenged")


class SimClassDebate:
    """
    Orchestrates multi-agent debates on educational topics

    Creates a simulated classroom discussion with multiple AI agents
    taking different perspectives to explore concepts deeply.
    """

    # Role-specific system prompts
    ROLE_PROMPTS = {
        DebateRole.ADVOCATE: """You are an advocate who argues IN FAVOR of the topic or position.
Find the strongest arguments supporting this view.
Use evidence, examples, and logical reasoning to make your case.
Be persuasive but intellectually honest - acknowledge limitations when appropriate.""",

        DebateRole.SKEPTIC: """You are a skeptic who QUESTIONS claims and challenges assumptions.
Don't just disagree - ask probing questions that reveal weaknesses or gaps.
Look for:
- Unsubstantiated claims
- Logical fallacies
- Missing evidence
- Alternative explanations
Be constructive - your goal is truth, not winning.""",

        DebateRole.SYNTHESIZER: """You are a synthesizer who finds COMMON GROUND between positions.
Look for:
- Points of agreement that others might miss
- Ways to integrate seemingly opposing views
- Underlying shared values or principles
- Compromises that preserve what's valuable from each side
Help move the discussion toward productive resolution.""",

        DebateRole.HISTORIAN: """You are a historian who provides HISTORICAL CONTEXT.
Contribute:
- How this topic has evolved over time
- Historical precedents and parallels
- What past thinkers have said
- How previous attempts worked out
Ground the discussion in concrete historical evidence.""",

        DebateRole.FUTURIST: """You are a futurist who considers FUTURE IMPLICATIONS.
Explore:
- Where current trends might lead
- Potential consequences of different positions
- Emerging technologies or developments that might change things
- Long-term vs short-term considerations
Help others think beyond the immediate.""",

        DebateRole.PRACTITIONER: """You are a practitioner focused on PRACTICAL APPLICATION.
Emphasize:
- How theories translate to practice
- Real-world constraints and considerations
- Implementation challenges
- What actually works vs. what sounds good
Keep the discussion grounded in reality.""",

        DebateRole.THEORIST: """You are a theorist who emphasizes UNDERLYING PRINCIPLES.
Focus on:
- First principles and foundational concepts
- Logical consistency
- Theoretical frameworks
- Definitional clarity
Help build rigorous understanding.""",

        DebateRole.CONTRARIAN: """You are a contrarian who deliberately takes the OPPOSITE VIEW.
Your role is to:
- Challenge the majority opinion
- Find weaknesses in popular positions
- Raise unpopular but valid points
- Prevent groupthink
Be intellectually honest - argue positions that have genuine merit, even if unconventional."""
    }

    # Format-specific rules
    FORMAT_RULES = {
        DebateFormat.OXFORD: {
            "description": "Formal debate with clear pro/con sides",
            "turns_per_round": 2,
            "rules": "Each side presents arguments in turn. Rebuttals follow opening statements."
        },
        DebateFormat.SOCRATIC: {
            "description": "Question-driven exploration",
            "turns_per_round": 3,
            "rules": "Participants ask probing questions to explore ideas. Focus on questions, not statements."
        },
        DebateFormat.ROUNDTABLE: {
            "description": "Open discussion among equals",
            "turns_per_round": 4,
            "rules": "All participants contribute equally. Build on each other's ideas."
        },
        DebateFormat.DEVILS_ADVOCATE: {
            "description": "Challenge the consensus position",
            "turns_per_round": 2,
            "rules": "One participant challenges the accepted view while others defend it."
        },
        DebateFormat.SYNTHESIS: {
            "description": "Find common ground and resolution",
            "turns_per_round": 4,
            "rules": "Goal is to find points of agreement and build toward synthesis."
        }
    }

    # Pre-configured debate panels for common topics
    PANEL_PRESETS = {
        "technical_pros_cons": [
            DebateAgent("a1", "Alex", DebateRole.ADVOCATE,
                       "Enthusiastic technologist", ["software", "innovation"], "pro"),
            DebateAgent("a2", "Morgan", DebateRole.SKEPTIC,
                       "Cautious analyst", ["risk assessment", "security"], "con"),
            DebateAgent("a3", "Sam", DebateRole.SYNTHESIZER,
                       "Balanced moderator", ["technology policy", "ethics"], "neutral")
        ],
        "philosophical": [
            DebateAgent("a1", "Sophia", DebateRole.THEORIST,
                       "Deep thinker", ["philosophy", "logic"], "analytical"),
            DebateAgent("a2", "Marcus", DebateRole.HISTORIAN,
                       "History buff", ["history", "classics"], "contextual"),
            DebateAgent("a3", "Nova", DebateRole.FUTURIST,
                       "Forward thinker", ["futures studies", "technology"], "speculative"),
            DebateAgent("a4", "Quinn", DebateRole.CONTRARIAN,
                       "Devil's advocate", ["rhetoric", "critical thinking"], "challenging")
        ],
        "practical_application": [
            DebateAgent("a1", "Jordan", DebateRole.PRACTITIONER,
                       "Hands-on expert", ["implementation", "operations"], "practical"),
            DebateAgent("a2", "Taylor", DebateRole.THEORIST,
                       "Conceptual thinker", ["theory", "research"], "theoretical"),
            DebateAgent("a3", "Casey", DebateRole.ADVOCATE,
                       "Solution-focused", ["problem-solving", "optimization"], "pro")
        ]
    }

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.8,
        llm: Optional[ChatOpenAI] = None
    ):
        """Initialize SimClass debate system"""
        self.llm = llm or ChatOpenAI(model=model, temperature=temperature)
        self.sessions: Dict[str, DebateSession] = {}

    def create_panel(
        self,
        roles: List[DebateRole],
        topic: str
    ) -> List[DebateAgent]:
        """
        Create a custom debate panel with specified roles

        Args:
            roles: List of roles to include
            topic: Topic for context in agent creation

        Returns:
            List of configured debate agents
        """
        names = ["Alex", "Morgan", "Sam", "Jordan", "Taylor", "Casey", "Quinn", "Nova"]
        agents = []

        for i, role in enumerate(roles):
            agent = DebateAgent(
                agent_id=f"agent_{i}",
                name=names[i % len(names)],
                role=role,
                personality=f"Expert debater specializing in {role.value} perspective",
                expertise=[topic, role.value],
                stance=role.value
            )
            agents.append(agent)

        return agents

    def start_debate(
        self,
        topic: str,
        format: DebateFormat = DebateFormat.ROUNDTABLE,
        panel_preset: Optional[str] = None,
        custom_agents: Optional[List[DebateAgent]] = None,
        learner_id: Optional[str] = None,
        max_rounds: int = 5
    ) -> DebateSession:
        """
        Start a new debate session

        Args:
            topic: The topic to debate
            format: Debate format to use
            panel_preset: Name of preset panel (technical_pros_cons, philosophical, practical_application)
            custom_agents: Custom list of debate agents (overrides preset)
            learner_id: If provided, learner can participate
            max_rounds: Maximum debate rounds

        Returns:
            New debate session
        """
        # Determine agents
        if custom_agents:
            agents = custom_agents
        elif panel_preset and panel_preset in self.PANEL_PRESETS:
            agents = self.PANEL_PRESETS[panel_preset]
        else:
            # Default panel
            agents = self.create_panel(
                [DebateRole.ADVOCATE, DebateRole.SKEPTIC, DebateRole.SYNTHESIZER],
                topic
            )

        session_id = f"debate_{datetime.utcnow().timestamp()}"

        session = DebateSession(
            session_id=session_id,
            topic=topic,
            format=format,
            agents=agents,
            learner_id=learner_id,
            max_rounds=max_rounds
        )

        self.sessions[session_id] = session
        logger.info(f"Started debate '{topic}' with {len(agents)} agents")

        return session

    async def get_opening_statements(
        self,
        session_id: str
    ) -> List[DebateArgument]:
        """
        Get opening statements from all debate agents

        Returns:
            List of opening arguments
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        opening_args = []

        # Get opening statement from each agent
        for agent in session.agents:
            argument = await self._generate_argument(
                session, agent, is_opening=True
            )
            session.arguments.append(argument)
            opening_args.append(argument)

        return opening_args

    async def advance_debate(
        self,
        session_id: str,
        learner_contribution: Optional[str] = None
    ) -> List[DebateArgument]:
        """
        Advance the debate by one round

        Args:
            session_id: Debate session ID
            learner_contribution: Optional contribution from observing learner

        Returns:
            List of new arguments from this round
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        if session.completed:
            raise ValueError("Debate is already completed")

        round_arguments = []

        # Add learner contribution if provided
        if learner_contribution and session.learner_id:
            learner_arg = DebateArgument(
                speaker_id=session.learner_id,
                speaker_role=DebateRole.ADVOCATE,  # Learner as participant
                content=learner_contribution,
                timestamp=datetime.utcnow(),
                argument_type="contribution"
            )
            session.learner_arguments.append(learner_arg)
            session.arguments.append(learner_arg)
            round_arguments.append(learner_arg)

        # Get responses from each agent
        for agent in session.agents:
            argument = await self._generate_argument(
                session, agent, is_opening=False
            )
            session.arguments.append(argument)
            round_arguments.append(argument)

        # Update round counter
        session.current_round += 1

        # Check if debate should end
        if session.current_round > session.max_rounds:
            session.completed = True
            session.end_time = datetime.utcnow()

        return round_arguments

    async def _generate_argument(
        self,
        session: DebateSession,
        agent: DebateAgent,
        is_opening: bool = False
    ) -> DebateArgument:
        """Generate an argument from a specific agent"""

        # Build context from previous arguments
        context = self._build_debate_context(session, agent)
        role_prompt = self.ROLE_PROMPTS[agent.role]
        format_rules = self.FORMAT_RULES[session.format]

        system_prompt = f"""You are {agent.name}, participating in a {session.format.value} debate.

TOPIC: {session.topic}

YOUR ROLE: {agent.role.value}
{role_prompt}

YOUR PERSONALITY: {agent.personality}
YOUR EXPERTISE: {', '.join(agent.expertise)}
YOUR STANCE: {agent.stance}

DEBATE FORMAT: {format_rules['description']}
RULES: {format_rules['rules']}

{"This is your OPENING STATEMENT. Introduce your position clearly and compellingly." if is_opening else "Respond to the discussion, engaging with others' points."}

Keep your contribution focused and under 200 words.
Be intellectually rigorous but conversational.
Reference specific points others have made when relevant.

Respond with a JSON object:
{{
    "content": "Your argument or response",
    "argument_type": "statement|rebuttal|question|evidence|concession",
    "key_points": ["main point 1", "main point 2"],
    "evidence": ["evidence or examples cited"],
    "acknowledges": ["points from others you agree with"],
    "challenges": ["points you're challenging"]
}}
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=context)
        ]

        response = await self.llm.ainvoke(messages)

        # Parse response
        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            result = json.loads(content)
            contribution = DebateContribution(**result)
        except Exception as e:
            logger.warning(f"Failed to parse debate response: {e}")
            contribution = DebateContribution(
                content=response.content,
                argument_type="statement"
            )

        # Create argument
        argument = DebateArgument(
            speaker_id=agent.agent_id,
            speaker_role=agent.role,
            content=contribution.content,
            timestamp=datetime.utcnow(),
            argument_type=contribution.argument_type,
            key_points=contribution.key_points,
            evidence_cited=contribution.evidence
        )

        return argument

    def _build_debate_context(
        self,
        session: DebateSession,
        current_agent: DebateAgent
    ) -> str:
        """Build context string from debate history"""

        if not session.arguments:
            return f"This is the opening of the debate on: {session.topic}"

        context_parts = [f"Debate on: {session.topic}\n\nPrevious contributions:\n"]

        # Include last N arguments for context
        recent_args = session.arguments[-10:]

        for arg in recent_args:
            speaker = next(
                (a.name for a in session.agents if a.agent_id == arg.speaker_id),
                "Participant"
            )
            context_parts.append(
                f"[{speaker} ({arg.speaker_role.value})]:\n{arg.content}\n"
            )

        context_parts.append(f"\nNow it's your turn to contribute as {current_agent.name}.")

        return "\n".join(context_parts)

    async def generate_summary(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive debate summary

        Returns:
            Summary with key insights, consensus points, and learning takeaways
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Build full debate transcript
        transcript = self._build_debate_context(session, session.agents[0])

        system_prompt = """You are an expert debate analyst. Analyze this debate and provide a comprehensive summary.

Identify:
1. KEY INSIGHTS: The most important ideas that emerged
2. POINTS OF CONSENSUS: Where participants agreed
3. POINTS OF DISAGREEMENT: Unresolved tensions
4. STRONGEST ARGUMENTS: The most compelling points made
5. LEARNING TAKEAWAYS: What someone observing could learn from this debate

Respond with JSON:
{
    "executive_summary": "2-3 sentence overview",
    "key_insights": ["insight 1", "insight 2", ...],
    "consensus_points": ["point 1", "point 2", ...],
    "disagreement_points": ["point 1", "point 2", ...],
    "strongest_arguments": [
        {"speaker": "name", "argument": "summary", "strength": "compelling|strong|moderate"}
    ],
    "learning_takeaways": ["takeaway 1", "takeaway 2", ...],
    "recommended_further_reading": ["topic 1", "topic 2", ...]
}
"""

        response = await self.llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=transcript)
        ])

        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            summary = json.loads(content)
        except Exception as e:
            logger.warning(f"Failed to parse summary: {e}")
            summary = {
                "executive_summary": "Debate analysis unavailable",
                "key_insights": [],
                "learning_takeaways": []
            }

        # Update session with insights
        session.key_insights = summary.get("key_insights", [])
        session.consensus_points = summary.get("consensus_points", [])
        session.disagreement_points = summary.get("disagreement_points", [])

        # Add metadata
        summary["session_id"] = session_id
        summary["topic"] = session.topic
        summary["format"] = session.format.value
        summary["total_rounds"] = session.current_round
        summary["total_arguments"] = len(session.arguments)
        summary["participants"] = [
            {"name": a.name, "role": a.role.value} for a in session.agents
        ]

        if session.learner_arguments:
            summary["learner_contributions"] = len(session.learner_arguments)

        return summary

    def get_session(self, session_id: str) -> Optional[DebateSession]:
        """Get session by ID"""
        return self.sessions.get(session_id)


# Singleton instance
_simclass_debate: Optional[SimClassDebate] = None


def get_simclass_debate() -> SimClassDebate:
    """Get or create the SimClass debate singleton"""
    global _simclass_debate
    if _simclass_debate is None:
        _simclass_debate = SimClassDebate()
    return _simclass_debate
