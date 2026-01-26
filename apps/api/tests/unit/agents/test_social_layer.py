"""
Phase 4: Agentic Social Layer Tests

Tests for:
1. Teachable Agent (Feynman Protocol)
2. SimClass Debates (Multi-agent perspective exploration)
3. Code Evaluator (TDD challenge generation & evaluation)
4. Watchtower Agent (Living Syllabus)
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# Teachable Agent imports
from app.agents.social.teachable_agent import (
    TeachableAgent,
    StudentPersona,
    ComprehensionLevel,
    QuestionType,
    TeachingSession,
    TeachingExchange,
    StudentResponse,
    get_teachable_agent,
)

# SimClass imports
from app.agents.social.simclass_debate import (
    SimClassDebate,
    DebateRole,
    DebateFormat,
    DebateAgent,
    DebateSession,
    DebateArgument,
    DebateContribution,
    get_simclass_debate,
)

# Code Evaluator imports
from app.agents.social.code_evaluator import (
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

# Watchtower Agent imports
from app.agents.watchtower_agent import (
    WatchtowerAgent,
    WatchtowerConfig,
    DomainUpdate,
    SyllabusUpdateModule,
    NewsSource,
    UpdatePriority,
)


# =============================================================================
# TEACHABLE AGENT TESTS
# =============================================================================

class TestTeachableAgent:
    """Tests for the Feynman Protocol teachable agent"""

    def test_student_personas_exist(self):
        """Verify all expected student personas are defined"""
        assert StudentPersona.CURIOUS
        assert StudentPersona.CONFUSED
        assert StudentPersona.CHALLENGER
        assert StudentPersona.VISUAL
        assert StudentPersona.PRACTICAL

    def test_comprehension_levels_exist(self):
        """Verify all comprehension levels are defined"""
        assert ComprehensionLevel.LOST
        assert ComprehensionLevel.STRUGGLING
        assert ComprehensionLevel.EMERGING
        assert ComprehensionLevel.DEVELOPING
        assert ComprehensionLevel.MASTERING

    def test_question_types_exist(self):
        """Verify all question types are defined"""
        assert QuestionType.CLARIFICATION
        assert QuestionType.EXAMPLE
        assert QuestionType.CONNECTION
        assert QuestionType.APPLICATION
        assert QuestionType.CHALLENGE
        assert QuestionType.ELABORATION
        assert QuestionType.CONFIRMATION

    def test_teachable_agent_initialization(self):
        """Test agent initializes with default settings"""
        agent = TeachableAgent()
        assert agent.llm is not None
        assert agent.sessions == {}

    def test_start_session(self):
        """Test starting a teaching session"""
        agent = TeachableAgent()
        session = agent.start_session(
            user_id="user123",
            concept_id="concept456",
            concept_name="Binary Search",
            persona=StudentPersona.CURIOUS
        )

        assert session.user_id == "user123"
        assert session.concept_id == "concept456"
        assert session.concept_name == "Binary Search"
        assert session.persona == StudentPersona.CURIOUS
        assert session.current_comprehension == 0.0
        assert session.comprehension_level == ComprehensionLevel.LOST
        assert session.completed is False

    def test_session_stored_in_agent(self):
        """Test that sessions are stored in the agent"""
        agent = TeachableAgent()
        session = agent.start_session(
            user_id="user123",
            concept_id="concept456",
            concept_name="Binary Search"
        )

        assert session.session_id in agent.sessions
        assert agent.get_session(session.session_id) == session

    def test_calculate_comprehension_level(self):
        """Test comprehension level calculation"""
        agent = TeachableAgent()

        assert agent._calculate_level(0.1) == ComprehensionLevel.LOST
        assert agent._calculate_level(0.3) == ComprehensionLevel.STRUGGLING
        assert agent._calculate_level(0.5) == ComprehensionLevel.EMERGING
        assert agent._calculate_level(0.7) == ComprehensionLevel.DEVELOPING
        assert agent._calculate_level(0.9) == ComprehensionLevel.MASTERING

    def test_end_session_generates_summary(self):
        """Test ending a session generates proper summary"""
        agent = TeachableAgent()
        session = agent.start_session(
            user_id="user123",
            concept_id="concept456",
            concept_name="Binary Search"
        )

        # Simulate some exchanges
        session.exchanges.append(TeachingExchange(
            timestamp=datetime.utcnow(),
            learner_explanation="Binary search divides the array in half",
            student_response="That makes sense!",
            question_type=QuestionType.CONFIRMATION,
            comprehension_before=0.0,
            comprehension_after=0.2,
            identified_gaps=[],
            key_concepts_covered=["divide and conquer"]
        ))
        session.current_comprehension = 0.6

        summary = agent.end_session(session.session_id)

        assert summary["session_id"] == session.session_id
        assert summary["concept"] == "Binary Search"
        assert summary["total_exchanges"] == 1
        assert summary["final_comprehension"] == 0.6
        assert "teaching_effectiveness" in summary
        assert "recommendations" in summary

    def test_singleton_getter(self):
        """Test singleton pattern for teachable agent"""
        agent1 = get_teachable_agent()
        agent2 = get_teachable_agent()
        assert agent1 is agent2


# =============================================================================
# SIMCLASS DEBATE TESTS
# =============================================================================

class TestSimClassDebate:
    """Tests for the multi-agent debate system"""

    def test_debate_roles_exist(self):
        """Verify all debate roles are defined"""
        assert DebateRole.ADVOCATE
        assert DebateRole.SKEPTIC
        assert DebateRole.SYNTHESIZER
        assert DebateRole.HISTORIAN
        assert DebateRole.FUTURIST
        assert DebateRole.PRACTITIONER
        assert DebateRole.THEORIST
        assert DebateRole.CONTRARIAN

    def test_debate_formats_exist(self):
        """Verify all debate formats are defined"""
        assert DebateFormat.OXFORD
        assert DebateFormat.SOCRATIC
        assert DebateFormat.ROUNDTABLE
        assert DebateFormat.DEVILS_ADVOCATE
        assert DebateFormat.SYNTHESIS

    def test_simclass_initialization(self):
        """Test SimClass debate system initializes"""
        debate = SimClassDebate()
        assert debate.llm is not None
        assert debate.sessions == {}

    def test_create_panel(self):
        """Test creating a custom debate panel"""
        debate = SimClassDebate()
        roles = [DebateRole.ADVOCATE, DebateRole.SKEPTIC, DebateRole.SYNTHESIZER]
        panel = debate.create_panel(roles, "AI Ethics")

        assert len(panel) == 3
        assert panel[0].role == DebateRole.ADVOCATE
        assert panel[1].role == DebateRole.SKEPTIC
        assert panel[2].role == DebateRole.SYNTHESIZER

    def test_start_debate(self):
        """Test starting a debate session"""
        debate = SimClassDebate()
        session = debate.start_debate(
            topic="Is AI dangerous?",
            format=DebateFormat.ROUNDTABLE,
            max_rounds=3
        )

        assert session.topic == "Is AI dangerous?"
        assert session.format == DebateFormat.ROUNDTABLE
        assert session.max_rounds == 3
        assert len(session.agents) >= 2
        assert session.current_round == 1
        assert session.completed is False

    def test_start_debate_with_preset(self):
        """Test starting a debate with panel preset"""
        debate = SimClassDebate()
        session = debate.start_debate(
            topic="Cloud vs On-Prem",
            panel_preset="technical_pros_cons"
        )

        assert session.topic == "Cloud vs On-Prem"
        assert len(session.agents) == 3  # technical_pros_cons has 3 agents

    def test_debate_agent_structure(self):
        """Test DebateAgent dataclass structure"""
        agent = DebateAgent(
            agent_id="test_agent",
            name="Alex",
            role=DebateRole.ADVOCATE,
            personality="Enthusiastic",
            expertise=["technology"],
            stance="pro"
        )

        assert agent.agent_id == "test_agent"
        assert agent.name == "Alex"
        assert agent.role == DebateRole.ADVOCATE
        assert hash(agent)  # Verify hashable

    def test_format_rules_configured(self):
        """Test all formats have rules configured"""
        debate = SimClassDebate()

        for format_type in DebateFormat:
            assert format_type in debate.FORMAT_RULES
            rules = debate.FORMAT_RULES[format_type]
            assert "description" in rules
            assert "turns_per_round" in rules
            assert "rules" in rules

    def test_role_prompts_configured(self):
        """Test all roles have prompts configured"""
        debate = SimClassDebate()

        for role in DebateRole:
            assert role in debate.ROLE_PROMPTS
            assert len(debate.ROLE_PROMPTS[role]) > 50  # Substantial prompt

    def test_singleton_getter(self):
        """Test singleton pattern for debate system"""
        debate1 = get_simclass_debate()
        debate2 = get_simclass_debate()
        assert debate1 is debate2


# =============================================================================
# CODE EVALUATOR TESTS
# =============================================================================

class TestCodeEvaluator:
    """Tests for the agentic code evaluation system"""

    def test_difficulty_levels_exist(self):
        """Verify all difficulty levels are defined"""
        assert DifficultyLevel.BEGINNER
        assert DifficultyLevel.INTERMEDIATE
        assert DifficultyLevel.ADVANCED
        assert DifficultyLevel.EXPERT

    def test_evaluation_dimensions_exist(self):
        """Verify all evaluation dimensions are defined"""
        assert EvaluationDimension.CORRECTNESS
        assert EvaluationDimension.QUALITY
        assert EvaluationDimension.EFFICIENCY
        assert EvaluationDimension.SECURITY
        assert EvaluationDimension.COMPLETENESS
        assert EvaluationDimension.DOCUMENTATION

    def test_hint_levels_exist(self):
        """Verify progressive hint levels are defined"""
        assert HintLevel.NUDGE
        assert HintLevel.GUIDANCE
        assert HintLevel.EXPLANATION
        assert HintLevel.PARTIAL
        assert HintLevel.SOLUTION

    def test_code_evaluator_initialization(self):
        """Test code evaluator initializes"""
        evaluator = CodeEvaluator()
        assert evaluator.llm is not None
        assert evaluator.challenges == {}

    def test_create_challenge(self):
        """Test creating a coding challenge"""
        evaluator = CodeEvaluator()
        challenge = evaluator.create_challenge(
            challenge_id="test_challenge",
            title="Test Challenge",
            description="A test coding challenge",
            difficulty=DifficultyLevel.BEGINNER,
            function_name="test_func",
            parameters=[{"name": "x", "type": "int"}],
            return_type="int",
            test_cases=[
                {"input": 5, "expected": 10, "description": "Basic test"}
            ],
            concepts_tested=["loops"],
            hints=["Think about iteration"],
            reference_solution="def test_func(x): return x * 2"
        )

        assert challenge.challenge_id == "test_challenge"
        assert challenge.title == "Test Challenge"
        assert challenge.difficulty == DifficultyLevel.BEGINNER
        assert len(challenge.test_cases) == 1

    def test_register_challenge(self):
        """Test registering a challenge stores it"""
        evaluator = CodeEvaluator()
        challenge = evaluator.create_challenge(
            challenge_id="test_reg",
            title="Registration Test",
            description="Test",
            difficulty=DifficultyLevel.BEGINNER,
            function_name="test",
            parameters=[],
            return_type="int",
            test_cases=[{"input": None, "expected": 42}],
            concepts_tested=[],
            hints=[],
            reference_solution="def test(): return 42"
        )

        assert "test_reg" in evaluator.challenges
        assert evaluator.get_challenge("test_reg") == challenge

    def test_register_sample_challenges(self):
        """Test sample challenges registration"""
        evaluator = CodeEvaluator()
        register_sample_challenges(evaluator)

        assert "two_sum" in evaluator.challenges
        assert "is_palindrome" in evaluator.challenges

        two_sum = evaluator.get_challenge("two_sum")
        assert two_sum.title == "Two Sum"
        assert two_sum.difficulty == DifficultyLevel.BEGINNER
        assert len(two_sum.test_cases) >= 3

    def test_agent_prompts_configured(self):
        """Test all dimensions have evaluation prompts"""
        evaluator = CodeEvaluator()

        for dimension in [
            EvaluationDimension.CORRECTNESS,
            EvaluationDimension.QUALITY,
            EvaluationDimension.EFFICIENCY,
            EvaluationDimension.SECURITY,
            EvaluationDimension.DOCUMENTATION
        ]:
            assert dimension in evaluator.AGENT_PROMPTS
            assert len(evaluator.AGENT_PROMPTS[dimension]) > 100

    def test_format_input_string(self):
        """Test input formatting for strings"""
        evaluator = CodeEvaluator()
        assert evaluator._format_input("hello") == '"hello"'

    def test_format_input_list(self):
        """Test input formatting for lists"""
        evaluator = CodeEvaluator()
        result = evaluator._format_input([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_format_input_tuple(self):
        """Test input formatting for tuples (unpacked as args)"""
        evaluator = CodeEvaluator()
        result = evaluator._format_input((5, 10))
        assert result == "5, 10"

    def test_singleton_getter(self):
        """Test singleton pattern for code evaluator"""
        eval1 = get_code_evaluator()
        eval2 = get_code_evaluator()
        assert eval1 is eval2


# =============================================================================
# WATCHTOWER AGENT TESTS
# =============================================================================

class TestWatchtowerAgent:
    """Tests for the Living Syllabus watchtower agent"""

    def test_news_sources_exist(self):
        """Verify all news sources are defined"""
        assert NewsSource.ARXIV
        assert NewsSource.GITHUB_TRENDING
        assert NewsSource.HACKER_NEWS
        assert NewsSource.TECH_NEWS
        assert NewsSource.DOCUMENTATION

    def test_update_priorities_exist(self):
        """Verify update priority levels are defined"""
        assert UpdatePriority.CRITICAL
        assert UpdatePriority.HIGH
        assert UpdatePriority.MEDIUM
        assert UpdatePriority.LOW

    def test_watchtower_initialization(self):
        """Test watchtower agent initializes with defaults"""
        agent = WatchtowerAgent()
        assert agent.llm is not None
        assert agent.config is not None
        assert agent._update_cache == {}

    def test_watchtower_config_defaults(self):
        """Test watchtower config has sensible defaults"""
        config = WatchtowerConfig()
        assert config.poll_interval_minutes == 60
        assert config.min_relevance_score == 0.6
        assert config.max_updates_per_day == 5
        assert NewsSource.ARXIV in config.enabled_sources

    def test_custom_config(self):
        """Test watchtower with custom config"""
        config = WatchtowerConfig(
            poll_interval_minutes=30,
            min_relevance_score=0.8,
            max_updates_per_day=3,
            domains=["machine learning", "deep learning"]
        )
        agent = WatchtowerAgent(config=config)

        assert agent.config.poll_interval_minutes == 30
        assert agent.config.min_relevance_score == 0.8
        assert "machine learning" in agent.config.domains

    def test_generate_update_id(self):
        """Test update ID generation is deterministic"""
        agent = WatchtowerAgent()

        id1 = agent._generate_update_id("Test Title", "http://example.com")
        id2 = agent._generate_update_id("Test Title", "http://example.com")
        id3 = agent._generate_update_id("Different", "http://other.com")

        assert id1 == id2  # Same input = same ID
        assert id1 != id3  # Different input = different ID
        assert len(id1) == 12  # Expected length

    def test_determine_priority(self):
        """Test priority determination from relevance scores"""
        agent = WatchtowerAgent()

        assert agent._determine_priority(0.95) == UpdatePriority.CRITICAL
        assert agent._determine_priority(0.80) == UpdatePriority.HIGH
        assert agent._determine_priority(0.65) == UpdatePriority.MEDIUM
        assert agent._determine_priority(0.40) == UpdatePriority.LOW

    def test_domain_update_structure(self):
        """Test DomainUpdate dataclass structure"""
        update = DomainUpdate(
            id="test123",
            source=NewsSource.ARXIV,
            title="Test Paper",
            summary="A test paper about testing",
            url="http://arxiv.org/test",
            priority=UpdatePriority.HIGH,
            related_concepts=["testing", "automation"],
            relevance_score=0.85,
            detected_at=datetime.utcnow()
        )

        assert update.id == "test123"
        assert update.source == NewsSource.ARXIV
        assert update.priority == UpdatePriority.HIGH
        assert len(update.related_concepts) == 2

    def test_syllabus_update_module_structure(self):
        """Test SyllabusUpdateModule dataclass structure"""
        update = DomainUpdate(
            id="src123",
            source=NewsSource.ARXIV,
            title="Source Update",
            summary="",
            url="",
            priority=UpdatePriority.MEDIUM,
            related_concepts=[],
            relevance_score=0.7,
            detected_at=datetime.utcnow()
        )

        module = SyllabusUpdateModule(
            id="mod456",
            title="Update Module",
            description="A brief update on new developments",
            concepts_covered=["new concept"],
            estimated_minutes=10,
            parent_module_id="week3",
            prerequisite_concepts=["basics"],
            content_type="briefing",
            priority=UpdatePriority.MEDIUM,
            source_update=update
        )

        assert module.id == "mod456"
        assert module.estimated_minutes == 10
        assert module.content_type == "briefing"
        assert module.source_update.id == "src123"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestPhase4Integration:
    """Integration tests for Phase 4 components"""

    def test_all_components_importable(self):
        """Test all Phase 4 components can be imported"""
        from app.agents.social import (
            TeachableAgent,
            SimClassDebate,
            CodeEvaluator,
            get_teachable_agent,
            get_simclass_debate,
            get_code_evaluator,
        )

        assert TeachableAgent is not None
        assert SimClassDebate is not None
        assert CodeEvaluator is not None

    def test_main_agents_package_exports(self):
        """Test main agents package exports Phase 4 components"""
        from app.agents import (
            TeachableAgent,
            StudentPersona,
            SimClassDebate,
            DebateRole,
            CodeEvaluator,
            DifficultyLevel,
        )

        assert TeachableAgent is not None
        assert StudentPersona.CURIOUS is not None
        assert SimClassDebate is not None
        assert DebateRole.ADVOCATE is not None
        assert CodeEvaluator is not None
        assert DifficultyLevel.BEGINNER is not None


# =============================================================================
# ASYNC TESTS (require pytest-asyncio)
# =============================================================================

@pytest.mark.asyncio
class TestAsyncOperations:
    """Async tests for Phase 4 components"""

    @pytest.mark.skip(reason="Requires LLM API key")
    async def test_teachable_agent_opening_question(self):
        """Test generating opening question (requires API key)"""
        agent = TeachableAgent()
        session = agent.start_session(
            user_id="test",
            concept_id="test",
            concept_name="Recursion"
        )
        question = await agent.get_opening_question(session)
        assert len(question) > 10

    @pytest.mark.skip(reason="Requires LLM API key")
    async def test_simclass_opening_statements(self):
        """Test generating debate opening statements (requires API key)"""
        debate = SimClassDebate()
        session = debate.start_debate(topic="AI Ethics")
        openings = await debate.get_opening_statements(session.session_id)
        assert len(openings) >= 2

    @pytest.mark.skip(reason="Requires subprocess execution")
    async def test_code_evaluator_runs_tests(self):
        """Test code evaluation with actual test execution"""
        evaluator = CodeEvaluator()
        register_sample_challenges(evaluator)

        result = await evaluator.evaluate_submission(
            challenge_id="two_sum",
            user_id="test_user",
            code="""
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
"""
        )

        assert result.tests_passed == result.tests_total
        assert result.passed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
