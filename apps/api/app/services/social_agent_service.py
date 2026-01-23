import uuid
import random
from datetime import datetime
from typing import Dict, List, Optional
from app.schemas.social_agent import (
    CodingChallenge, CodingDifficultyLevel, EvaluationResult, DimensionScore,
    DebateSessionResponse, DebateArgument, DebateFormat, PanelPreset,
    DebateRoundResponse, DebateSummary, TeachingSessionResponse,
    TeachingResponse, TeachingSessionSummary, FeedbackItem, TestCase
)
from app.services.social.teachable_agent import TeachableAgent
from app.services.social.coding_agent import CodingAgent
from app.services.social.debate_agent import DebateAgent

# Mock Data for Coding Challenges (Fallback/Standard Library)
MOCK_CHALLENGES = [
    CodingChallenge(
        challenge_id="py_fib_01",
        title="Fibonacci Generator",
        description="Implement a function that yields the first n Fibonacci numbers.",
        difficulty="beginner", # Upcasted string lit to CodingDifficultyLevel
        category="Algorithms",
        concepts=["Generators", "Loops", "Math"],
        function_name="fibonacci_gen",
        parameters=[{"name": "n", "type": "int"}],
        return_type="Iterator[int]",
        test_cases=[
            TestCase(input=[5], expected=[0, 1, 1, 2, 3], description="First 5 numbers"),
        ],
        estimated_minutes=15,
        language="python"
    ),
    CodingChallenge(
        challenge_id="py_pal_02",
        title="Palindrome Checker",
        description="Check if a given string is a palindrome (reads same forwards and backwards), ignoring case and spaces.",
        difficulty="beginner",
        category="Strings",
        concepts=["String Manipulation", "Conditionals"],
        function_name="is_palindrome",
        parameters=[{"name": "text", "type": "str"}],
        return_type="bool",
        test_cases=[
            TestCase(input=["Race Car"], expected=True, description="Case insensitive with spaces"),
            TestCase(input=["hello"], expected=False, description="Not a palindrome"),
        ],
        estimated_minutes=10,
        language="python"
    )
]

# In-memory session blocks (replace with Redis/DB in production)
debate_sessions: Dict[str, Dict] = {}
teaching_sessions: Dict[str, Dict] = {}

class SocialAgentService:
    
    def __init__(self):
        self.teachable_agent = TeachableAgent()
        self.coding_agent = CodingAgent()
        self.debate_agent = DebateAgent()

    # ==================== Coding Challenges ====================
    
    async def get_challenges(self) -> List[CodingChallenge]:
        # Return mocks for stability + maybe generate 1 fresh one?
        # For this phase, just return mocks to ensure UI works, as generation is slow/costly on every load
        return MOCK_CHALLENGES

    async def get_challenge(self, challenge_id: str) -> Optional[CodingChallenge]:
        for ch in MOCK_CHALLENGES:
            if ch.challenge_id == challenge_id:
                return ch
        return None

    async def evaluate_code(self, challenge_id: str, code: str) -> EvaluationResult:
        challenge = await self.get_challenge(challenge_id)
        if not challenge:
            raise ValueError("Challenge not found")
            
        # Call Agent
        result_data = await self.coding_agent.evaluate_submission(challenge, code)
        
        # Map JSON dict to Pydantic Model
        dim_scores = {}
        if "dimension_scores" in result_data:
            for k, v in result_data["dimension_scores"].items():
                if k in ["correctness", "quality", "efficiency", "security", "completeness", "documentation"]:
                    dim_scores[k] = DimensionScore(
                        score=v.get("score", 0),
                        feedback=v.get("feedback", ""),
                        strengths=v.get("strengths", []),
                        improvements=v.get("improvements", [])
                    )

        feedback_items = []
        if "feedback" in result_data:
            for f in result_data["feedback"]:
                 feedback_items.append(FeedbackItem(
                     type=f.get("type", "issue"),
                     dimension="correctness", # Default
                     message=f.get("message", ""),
                     line_number=f.get("line_number"),
                     suggestion=f.get("suggestion")
                 ))
        
        return EvaluationResult(
            submission_id=str(uuid.uuid4()),
            passed=result_data.get("passed", False),
            overall_score=result_data.get("overall_score", 0),
            tests_passed=result_data.get("tests_passed", 0),
            tests_total=len(challenge.test_cases), # Approximation
            dimension_scores=dim_scores,
            feedback=feedback_items,
            concepts_demonstrated=result_data.get("concepts_demonstrated", []),
            concepts_to_review=result_data.get("concepts_to_review", []),
            execution_time_ms=0, # No real exec yet
            runtime_errors=[]
        )

    async def get_hint(self, challenge_id: str, code: str, hint_level: str) -> str:
        # Simple dynamic hint (could be moved to agent)
        return f"Consider checking the requirements for {hint_level} level guidance."

    # ==================== Debates ====================

    async def start_debate(self, topic: str, format: str, preset: str, max_rounds: int) -> DebateSessionResponse:
        session_id = str(uuid.uuid4())
        
        # Define participants based on preset
        participants = []
        if preset == "technical_pros_cons":
            participants = [
                {"name": "Alice", "role": "advocate"}, 
                {"name": "Bob", "role": "skeptic"}
            ]
        else:
             participants = [
                {"name": "Alice", "role": "advocate"}, 
                {"name": "Bob", "role": "skeptic"}, 
                {"name": "Charlie", "role": "synthesizer"}
            ]
        
        # Generate opening statement from first participant
        p1 = participants[0]
        opening_arg_data = await self.debate_agent.generate_argument(topic, p1["role"], p1["name"], [], 1)
        
        opening_statement = DebateArgument(
            speaker=p1["name"],
            role=p1["role"],
            content=opening_arg_data.get("content", "Let's begin."),
            argument_type="opening",
            key_points=opening_arg_data.get("key_points", []),
            timestamp=datetime.utcnow()
        )
        
        debate_sessions[session_id] = {
            "topic": topic,
            "format": format,
            "rounds": 1,
            "max_rounds": max_rounds,
            "participants": participants,
            "history": [opening_statement],
            "next_turn_index": 1
        }
        
        return DebateSessionResponse(
            session_id=session_id,
            topic=topic,
            format=format,
            participants=participants,
            current_round=1,
            max_rounds=max_rounds,
            opening_statements=[opening_statement]
        )

    async def advance_debate(self, session_id: str, learner_contribution: Optional[str]) -> DebateRoundResponse:
        session = debate_sessions.get(session_id)
        if not session:
            raise ValueError("Session not found")
            
        # Add learner contribution if exists
        if learner_contribution:
            session["history"].append(DebateArgument(
                speaker="Learner",
                role="practitioner",
                content=learner_contribution,
                argument_type="contribution",
                key_points=[],
                timestamp=datetime.utcnow()
            ))

        # Determine next speaker
        participants = session["participants"]
        next_idx = session.get("next_turn_index", 0)
        speaker = participants[next_idx % len(participants)]
        
        # Generate Argument
        arg_data = await self.debate_agent.generate_argument(
            session["topic"], 
            speaker["role"], 
            speaker["name"], 
            [{"speaker": h.speaker, "role": h.role, "content": h.content} for h in session["history"]],
            session["rounds"]
        )
        
        new_arg = DebateArgument(
            speaker=speaker["name"],
            role=speaker["role"],
            content=arg_data.get("content", "..."),
            argument_type=arg_data.get("argument_type", "rebuttal"),
            key_points=arg_data.get("key_points", []),
            timestamp=datetime.utcnow()
        )
        
        session["history"].append(new_arg)
        session["next_turn_index"] = next_idx + 1
        
        # Check round completion (all participants spoke once)
        if (next_idx + 1) % len(participants) == 0:
            session["rounds"] += 1
            
        completed = session["rounds"] > session["max_rounds"]
        
        return DebateRoundResponse(
            session_id=session_id,
            current_round=session["rounds"],
            arguments=[new_arg],
            completed=completed
        )

    async def get_debate_summary(self, session_id: str) -> DebateSummary:
        session = debate_sessions.get(session_id)
        if not session:
             return DebateSummary(
                session_id=session_id,
                topic="Unknown", format="roundtable",
                executive_summary="Session not found",
                total_rounds=0, total_arguments=0,
                participants=[], key_insights=[], consensus_points=[],
                disagreement_points=[], strongest_arguments=[], learning_takeaways=[], recommended_further_reading=[]
            )
             
        data = await self.debate_agent.generate_summary(
            session["topic"],
            [{"speaker": h.speaker, "content": h.content} for h in session["history"]]
        )
        
        return DebateSummary(
            session_id=session_id,
            topic=session["topic"],
            format=session["format"],
            total_rounds=session["rounds"],
            total_arguments=len(session["history"]),
            participants=[p for p in session["participants"]],
            executive_summary=data.get("executive_summary", ""),
            key_insights=data.get("key_insights", []),
            consensus_points=data.get("consensus_points", []),
            disagreement_points=data.get("disagreement_points", []),
            strongest_arguments=data.get("strongest_arguments", []),
            learning_takeaways=[],
            recommended_further_reading=[]
        )

    # ==================== Teaching ====================

    async def start_teaching_session(self, user_id: str, concept_name: str, persona: str) -> TeachingSessionResponse:
        session_id = str(uuid.uuid4())
        
        # Generate opening
        init_data = await self.teachable_agent.start_session(concept_name, persona)
        
        teaching_sessions[session_id] = {
            "user_id": user_id,
            "concept": concept_name,
            "persona": persona,
            "comprehension": 0.0,
            "start_time": datetime.utcnow(),
            "exchanges": 0,
            "history": [] # Store message history
        }
        
        # Add agent's opening to history
        teaching_sessions[session_id]["history"].append({"role": "assistant", "content": init_data.get("opening_question", "")})
        
        return TeachingSessionResponse(
            session_id=session_id,
            persona=persona,
            concept_name=concept_name,
            student_name="Alex",
            opening_question=init_data.get("opening_question", "Ready?"),
            comprehension=0.0,
            comprehension_level="lost",
            message=init_data.get("greeting", "Hello!")
        )

    async def submit_explanation(self, session_id: str, explanation: str) -> TeachingResponse:
        session = teaching_sessions.get(session_id)
        if not session:
             raise ValueError("Session not found")
        
        # Add user's explanation to history
        session["history"].append({"role": "user", "content": explanation})
        
        # Process explanation
        response_data = await self.teachable_agent.process_explanation(
            session["concept"],
            session["persona"],
            session["comprehension"],
            session["history"],
            explanation
        )
        
        # Update state
        delta = response_data.get("comprehension_delta", 0)
        session["comprehension"] = max(0.0, min(1.0, session["comprehension"] + delta))
        session["exchanges"] += 1
        
        # Add agent response to history
        agent_resp = response_data.get("response", "")
        session["history"].append({"role": "assistant", "content": agent_resp})
        
        level = "mastering" if session["comprehension"] > 0.8 else \
                "developing" if session["comprehension"] > 0.5 else \
                "emerging" if session["comprehension"] > 0.3 else \
                "struggling" if session["comprehension"] > 0.1 else "lost"
        
        return TeachingResponse(
            response=agent_resp,
            question_type=response_data.get("question_type"),
            comprehension=session["comprehension"],
            comprehension_level=level,
            knowledge_gaps=response_data.get("knowledge_gaps", []),
            concepts_understood=response_data.get("concepts_understood", [])
        )

    async def end_teaching_session(self, session_id: str) -> TeachingSessionSummary:
        session = teaching_sessions.get(session_id)
        duration = 0
        if session:
             duration = (datetime.utcnow() - session["start_time"]).total_seconds() / 60
             
        return TeachingSessionSummary(
            session_id=session_id,
            persona_used=session["persona"] if session else "curious",
            concept=session["concept"] if session else "Unknown",
            teaching_effectiveness=0.85, # Logic to calc this?
            final_comprehension=session["comprehension"] if session else 0,
            comprehension_level="developing", # derived
            comprehension_progress=[], # track this?
            improvement_per_exchange=0.1,
            total_exchanges=session["exchanges"] if session else 0,
            duration_minutes=duration,
            recommendations=["Keep using analogies"],
            knowledge_gaps_identified=[],
            strong_explanations=[]
        )
