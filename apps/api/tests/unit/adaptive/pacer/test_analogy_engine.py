"""
Analogy Critique Engine Counter-Tests

Tests designed to validate the analogy critique evaluation system.
These counter-tests ensure:
1. Correct identification of breakdown points is properly scored
2. False positives are appropriately penalized
3. Fuzzy matching handles paraphrasing
4. Edge cases don't break evaluation
"""
import pytest
from app.adaptive.pacer.analogy_engine import (
    AnalogyCritiqueEngine,
    Analogy,
    StructuralMapping,
    BreakdownPoint,
    CritiqueEvaluation,
)


class TestAnalogyCritiqueEngineBasic:
    """Basic functionality tests for the Analogy Critique Engine"""

    @pytest.fixture
    def engine(self):
        return AnalogyCritiqueEngine()

    @pytest.fixture
    def sample_analogy(self):
        """Sample analogy for testing: Electricity as Water Flow"""
        return Analogy(
            id=1,
            source_domain="Water Flow",
            target_domain="Electricity",
            summary="Electricity flows through wires like water flows through pipes.",
            mappings=[
                StructuralMapping("pressure", "voltage", "corresponds_to"),
                StructuralMapping("flow rate", "current", "corresponds_to"),
                StructuralMapping("pipe diameter", "wire thickness", "corresponds_to"),
            ],
            valid_aspects=["pressure", "flow rate", "resistance"],
            breakdown_points=[
                BreakdownPoint(
                    aspect="Water is visible",
                    reason="Electricity is invisible and cannot be directly observed",
                    severity="minor",
                    educational_note="This is why electrical safety requires special instruments"
                ),
                BreakdownPoint(
                    aspect="Water leaks out of cut pipes",
                    reason="Electrons don't leak out of cut wires - the circuit simply breaks",
                    severity="moderate",
                    educational_note="Understanding this prevents misconceptions about open circuits"
                ),
                BreakdownPoint(
                    aspect="Water can be stored in tanks",
                    reason="Electricity cannot be stored directly in wires",
                    severity="major",
                    educational_note="Capacitors and batteries use different mechanisms"
                ),
            ],
            critique_prompt="Where does this analogy break down?"
        )

    def test_create_analogy(self, engine):
        """Test analogy creation from raw data"""
        analogy = engine.create_analogy(
            source_domain="Solar System",
            target_domain="Atom",
            content="An atom is like a solar system",
            mappings=[
                {"source_element": "sun", "target_element": "nucleus"},
                {"source_element": "planets", "target_element": "electrons"},
            ],
            breakdowns=[
                {"aspect": "Scale", "reason": "Atoms are far smaller than solar systems"},
            ]
        )

        assert analogy.source_domain == "Solar System"
        assert analogy.target_domain == "Atom"
        assert len(analogy.mappings) == 2
        assert len(analogy.breakdown_points) == 1


class TestCritiqueEvaluationCounterTests:
    """Counter-tests for critique evaluation accuracy"""

    @pytest.fixture
    def engine(self):
        return AnalogyCritiqueEngine()

    @pytest.fixture
    def sample_analogy(self):
        """Sample analogy with 3 breakdown points"""
        return Analogy(
            id=1,
            source_domain="Water Flow",
            target_domain="Electricity",
            summary="Electricity flows like water",
            mappings=[
                StructuralMapping("pressure", "voltage", "corresponds_to"),
            ],
            valid_aspects=["pressure"],
            breakdown_points=[
                BreakdownPoint("Water is visible", "Electricity is not", "minor", "Safety note"),
                BreakdownPoint("Water leaks out", "Circuit breaks", "moderate", "Open circuit concept"),
                BreakdownPoint("Water stores in tanks", "Electricity needs batteries", "major", "Storage concept"),
            ],
            critique_prompt="Find the breakdowns"
        )

    def test_perfect_critique(self, engine, sample_analogy):
        """
        Counter-test: User identifies all breakdown points correctly.
        Should get perfect score (F1 = 1.0).
        """
        user_breakdowns = [
            "Water is visible",
            "Water leaks out",
            "Water stores in tanks"
        ]

        result = engine.evaluate_critique(sample_analogy, user_breakdowns)

        assert result.score == 1.0, f"Perfect match should get F1=1.0, got {result.score}"
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert len(result.missed_breakdowns) == 0
        assert len(result.false_positives) == 0

    def test_critique_with_false_positives(self, engine, sample_analogy):
        """
        Counter-test: User identifies correct breakdowns AND non-existent ones.
        False positives should penalize precision.
        """
        user_breakdowns = [
            "Water is visible",  # Correct
            "Water leaks out",   # Correct
            "Pipes can rust",    # False positive - not in actual breakdowns
        ]

        result = engine.evaluate_critique(sample_analogy, user_breakdowns)

        assert result.precision < 1.0, "False positives should reduce precision"
        assert result.recall < 1.0, "Missing one breakdown reduces recall"
        assert "pipes can rust" in result.false_positives
        assert len(result.false_positives) == 1

    def test_critique_with_missed_breakdowns(self, engine, sample_analogy):
        """
        Counter-test: User misses important breakdown points.
        Should penalize recall.
        """
        user_breakdowns = [
            "Water is visible",  # Only identified 1 of 3
        ]

        result = engine.evaluate_critique(sample_analogy, user_breakdowns)

        assert result.recall < 0.5, "Missing 2 of 3 breakdowns should heavily penalize recall"
        assert result.precision == 1.0, "Identified breakdown is correct, precision should be 1.0"
        assert len(result.missed_breakdowns) == 2

    def test_empty_critique(self, engine, sample_analogy):
        """
        Counter-test: User submits empty critique.
        Should get zero score.
        """
        user_breakdowns = []

        result = engine.evaluate_critique(sample_analogy, user_breakdowns)

        assert result.score == 0.0, "Empty critique should score 0"
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert len(result.missed_breakdowns) == 3

    def test_all_false_positives(self, engine, sample_analogy):
        """
        Counter-test: User identifies only incorrect breakdowns.
        Should score 0 (no true positives).
        """
        user_breakdowns = [
            "Pipes can rust",
            "Water is blue",
            "Pipes are cylindrical"
        ]

        result = engine.evaluate_critique(sample_analogy, user_breakdowns)

        assert result.score == 0.0, "All false positives should score 0"
        assert result.precision == 0.0
        assert len(result.false_positives) == 3
        assert len(result.missed_breakdowns) == 3


class TestFuzzyMatchingCounterTests:
    """Counter-tests for fuzzy matching of breakdown points"""

    @pytest.fixture
    def engine(self):
        return AnalogyCritiqueEngine()

    @pytest.fixture
    def sample_analogy(self):
        """Sample analogy for fuzzy matching tests"""
        return Analogy(
            id=2,
            source_domain="Factory",
            target_domain="Cell",
            summary="A cell is like a factory",
            mappings=[StructuralMapping("workers", "enzymes", "functions_as")],
            valid_aspects=["workers"],
            breakdown_points=[
                BreakdownPoint(
                    aspect="Factories close at night",
                    reason="Cells operate continuously 24/7",
                    severity="moderate",
                    educational_note="Cellular processes don't stop"
                ),
                BreakdownPoint(
                    aspect="Factories have managers",
                    reason="Cells don't have hierarchical management",
                    severity="minor",
                    educational_note="Cellular regulation is distributed"
                ),
            ],
            critique_prompt=""
        )

    def test_fuzzy_match_paraphrased_breakdown(self, engine, sample_analogy):
        """
        Counter-test: User uses different wording for same concept.
        Fuzzy matching should recognize paraphrased breakdowns.
        """
        # User says "Factories stop at night" instead of "Factories close at night"
        user_breakdowns = [
            "Factories stop working at night",
            "Factories have bosses and supervisors"  # Paraphrase of "have managers"
        ]

        result = engine.evaluate_critique_fuzzy(
            sample_analogy,
            user_breakdowns,
            similarity_threshold=0.4  # Lower threshold for more lenient matching
        )

        # Should recognize at least partial matches with fuzzy matching
        # The exact result depends on similarity algorithm
        assert result.recall > 0, "Fuzzy matching should find some paraphrased matches"

    def test_fuzzy_match_strict_threshold(self, engine, sample_analogy):
        """
        Counter-test: High similarity threshold should require close matches.
        """
        user_breakdowns = [
            "Manufacturing plants shut down overnight"  # Very different wording
        ]

        result = engine.evaluate_critique_fuzzy(
            sample_analogy,
            user_breakdowns,
            similarity_threshold=0.8  # Very strict threshold
        )

        # Strict threshold should not match paraphrased content
        assert len(result.correctly_identified) == 0, \
            "Strict threshold should not match loosely paraphrased content"

    def test_fuzzy_match_exact_wording(self, engine, sample_analogy):
        """
        Counter-test: Exact wording should always match regardless of threshold.
        """
        user_breakdowns = [
            "Factories close at night"  # Exact match
        ]

        result = engine.evaluate_critique_fuzzy(
            sample_analogy,
            user_breakdowns,
            similarity_threshold=0.9
        )

        assert "factories close at night" in [b.lower() for b in result.correctly_identified], \
            "Exact matches should always be identified"


class TestFeedbackGenerationCounterTests:
    """Counter-tests for feedback quality"""

    @pytest.fixture
    def engine(self):
        return AnalogyCritiqueEngine()

    def test_excellent_feedback_for_high_score(self, engine):
        """High scores should receive encouraging feedback"""
        feedback = engine._generate_feedback(0.9, missed_count=0, false_positive_count=0)

        assert "excellent" in feedback.lower() or "great" in feedback.lower() or "key" in feedback.lower()

    def test_constructive_feedback_for_medium_score(self, engine):
        """Medium scores should receive constructive feedback"""
        feedback = engine._generate_feedback(0.5, missed_count=2, false_positive_count=1)

        assert "missed" in feedback.lower() or "review" in feedback.lower()

    def test_guidance_feedback_for_low_score(self, engine):
        """Low scores should receive guiding feedback"""
        feedback = engine._generate_feedback(0.2, missed_count=4, false_positive_count=2)

        assert "review" in feedback.lower() or "nuance" in feedback.lower()


class TestBreakdownPointSeverityCounterTests:
    """Counter-tests for breakdown point severity handling"""

    @pytest.fixture
    def engine(self):
        return AnalogyCritiqueEngine()

    def test_missed_breakdowns_include_severity(self, engine):
        """Missed breakdowns should include severity information"""
        analogy = Analogy(
            id=3,
            source_domain="Test",
            target_domain="Target",
            summary="Test analogy",
            mappings=[],
            valid_aspects=[],
            breakdown_points=[
                BreakdownPoint("Minor issue", "Reason", "minor", "Note"),
                BreakdownPoint("Major issue", "Reason", "major", "Note"),
            ],
            critique_prompt=""
        )

        result = engine.evaluate_critique(analogy, [])

        assert all("severity" in mb for mb in result.missed_breakdowns), \
            "Missed breakdowns should include severity"

        severities = [mb["severity"] for mb in result.missed_breakdowns]
        assert "minor" in severities
        assert "major" in severities


class TestAnalogyCreationCounterTests:
    """Counter-tests for analogy creation edge cases"""

    @pytest.fixture
    def engine(self):
        return AnalogyCritiqueEngine()

    def test_create_analogy_minimal_data(self, engine):
        """Test analogy creation with minimal required data"""
        analogy = engine.create_analogy(
            source_domain="A",
            target_domain="B",
            content="A is like B",
            mappings=[{"source_element": "x", "target_element": "y"}],
            breakdowns=[{"aspect": "z", "reason": "w"}]
        )

        assert analogy is not None
        assert analogy.source_domain == "A"
        assert len(analogy.breakdown_points) == 1

    def test_create_analogy_default_severity(self, engine):
        """Test that breakdowns get default severity when not specified"""
        analogy = engine.create_analogy(
            source_domain="A",
            target_domain="B",
            content="Content",
            mappings=[],
            breakdowns=[{"aspect": "Test", "reason": "Reason"}]  # No severity specified
        )

        assert analogy.breakdown_points[0].severity == "moderate", \
            "Default severity should be 'moderate'"

    def test_create_analogy_default_relationship(self, engine):
        """Test that mappings get default relationship when not specified"""
        analogy = engine.create_analogy(
            source_domain="A",
            target_domain="B",
            content="Content",
            mappings=[{"source_element": "x", "target_element": "y"}],  # No relationship
            breakdowns=[]
        )

        assert analogy.mappings[0].relationship == "corresponds_to", \
            "Default relationship should be 'corresponds_to'"

    def test_critique_prompt_generation(self, engine):
        """Test that critique prompt is properly generated"""
        analogy = engine.create_analogy(
            source_domain="Water",
            target_domain="Electricity",
            content="Test",
            mappings=[],
            breakdowns=[]
        )

        assert "Water" in analogy.critique_prompt
        assert "Electricity" in analogy.critique_prompt
        assert "break" in analogy.critique_prompt.lower()


class TestCaseSensitivityCounterTests:
    """Counter-tests for case sensitivity in matching"""

    @pytest.fixture
    def engine(self):
        return AnalogyCritiqueEngine()

    def test_case_insensitive_matching(self, engine):
        """Matching should be case-insensitive"""
        analogy = Analogy(
            id=4,
            source_domain="Test",
            target_domain="Target",
            summary="Test",
            mappings=[],
            valid_aspects=[],
            breakdown_points=[
                BreakdownPoint("Water Is Visible", "Reason", "minor", "Note"),
            ],
            critique_prompt=""
        )

        # User types in different case
        user_breakdowns = ["water is visible"]

        result = engine.evaluate_critique(analogy, user_breakdowns)

        assert result.recall == 1.0, "Matching should be case-insensitive"

    def test_whitespace_handling(self, engine):
        """Matching should handle extra whitespace"""
        analogy = Analogy(
            id=5,
            source_domain="Test",
            target_domain="Target",
            summary="Test",
            mappings=[],
            valid_aspects=[],
            breakdown_points=[
                BreakdownPoint("Water is visible", "Reason", "minor", "Note"),
            ],
            critique_prompt=""
        )

        # User types with extra whitespace
        user_breakdowns = ["  Water is visible  "]

        result = engine.evaluate_critique(analogy, user_breakdowns)

        assert result.recall == 1.0, "Matching should handle whitespace"
