"""
Agentic Social Layer - Phase 4

Research alignment:
- Learning by Teaching (Feynman Technique): Teaching deepens understanding
- Social Constructivism: Knowledge constructed through social interaction
- Cognitive Conflict: Exposure to different perspectives enhances learning
- Formative Assessment: Continuous feedback for improvement
- Scaffolding: Progressive support tailored to learner needs

Components:
1. Teachable Agent (Feynman Protocol): Learners teach an AI student
2. SimClass Debates: Multi-agent debates for perspective exploration
3. Code Evaluator: Agentic code review and evaluation
"""

from .teachable_agent import (
    TeachableAgent,
    StudentPersona,
    ComprehensionLevel,
    QuestionType,
    TeachingExchange,
    TeachingSession,
    StudentResponse,
    get_teachable_agent,
)

from .simclass_debate import (
    SimClassDebate,
    DebateRole,
    DebateFormat,
    ArgumentStrength,
    DebateArgument,
    DebateAgent,
    DebateSession,
    DebateContribution,
    get_simclass_debate,
)

from .code_evaluator import (
    CodeEvaluator,
    DifficultyLevel,
    EvaluationDimension,
    FeedbackType,
    HintLevel,
    TestCase,
    CodingChallenge,
    FeedbackItem,
    DimensionScore,
    EvaluationResult,
    get_code_evaluator,
    register_sample_challenges,
)

__all__ = [
    # Teachable Agent (Feynman Protocol)
    "TeachableAgent",
    "StudentPersona",
    "ComprehensionLevel",
    "QuestionType",
    "TeachingExchange",
    "TeachingSession",
    "StudentResponse",
    "get_teachable_agent",
    # SimClass Debates
    "SimClassDebate",
    "DebateRole",
    "DebateFormat",
    "ArgumentStrength",
    "DebateArgument",
    "DebateAgent",
    "DebateSession",
    "DebateContribution",
    "get_simclass_debate",
    # Code Evaluator
    "CodeEvaluator",
    "DifficultyLevel",
    "EvaluationDimension",
    "FeedbackType",
    "HintLevel",
    "TestCase",
    "CodingChallenge",
    "FeedbackItem",
    "DimensionScore",
    "EvaluationResult",
    "get_code_evaluator",
    "register_sample_challenges",
]
