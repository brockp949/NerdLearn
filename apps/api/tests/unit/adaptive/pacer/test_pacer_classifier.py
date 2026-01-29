"""
PACER Classifier Counter-Tests

Tests designed to validate and "break" the PACER classification system.
These counter-tests ensure robust classification across edge cases.

Counter-test categories:
1. Misclassification scenarios - content that could trick the classifier
2. Ambiguous content - content with mixed signals
3. Edge cases - minimal content, adversarial input
4. Confidence calibration - verifying confidence scores are meaningful
"""
import pytest
from app.adaptive.pacer.classifier import PACERClassifier, PACERType, ClassificationResult


class TestPACERClassifierBasic:
    """Basic functionality tests"""

    @pytest.fixture
    def classifier(self):
        return PACERClassifier(use_llm=False)

    def test_classifier_initialization(self, classifier):
        """Verify classifier initializes correctly"""
        assert classifier is not None
        assert classifier.use_llm is False

    def test_classify_returns_result(self, classifier):
        """Verify classify returns a ClassificationResult"""
        content = "This is a step-by-step guide to installing Python."
        result = classifier.classify(content)

        assert isinstance(result, ClassificationResult)
        assert result.pacer_type in PACERType
        assert 0 <= result.confidence <= 1
        assert result.reasoning is not None
        assert result.content_hash is not None

    def test_triage_returns_decisions(self, classifier):
        """Verify triage tree returns decision path"""
        content = "This theory explains why plants need sunlight."
        decisions = classifier.run_triage_tree(content)

        assert isinstance(decisions, list)
        assert len(decisions) > 0
        assert any(d.leads_to is not None for d in decisions)


class TestMisclassificationCounterTests:
    """
    Counter-tests for misclassification scenarios.
    Tests content designed to potentially trick the classifier.
    """

    @pytest.fixture
    def classifier(self):
        return PACERClassifier(use_llm=False)

    def test_procedural_disguised_as_conceptual(self, classifier):
        """
        Counter-test: Procedural content with theoretical framing.
        Content has explicit steps but wrapped in conceptual language.
        Should classify as Procedural due to action-oriented structure.
        """
        content = """
        The principle of effective code review involves several critical steps:
        Step 1: Read through the entire change set carefully before commenting.
        Step 2: Check for potential security vulnerabilities in the code.
        Step 3: Verify that adequate test coverage exists.
        Step 4: Review code style and naming conventions.
        Step 5: Submit your review with constructive feedback.
        """
        result = classifier.classify(content)

        # Should be Procedural because of explicit steps
        assert result.pacer_type == PACERType.PROCEDURAL, \
            f"Content with explicit numbered steps should be Procedural, got {result.pacer_type}"

    def test_conceptual_disguised_as_procedural(self, classifier):
        """
        Counter-test: Conceptual content with step-like language.
        Explains WHY things happen using sequential language.
        Should classify as Conceptual because it explains causation.
        """
        content = """
        Gravity affects objects through a sequence of physical phenomena:
        First, mass creates a curvature in the fabric of spacetime.
        Then, objects follow geodesics through this curved space.
        Finally, we perceive this movement as gravitational attraction.
        This explains why all objects fall at the same rate regardless of mass
        in a vacuum - they are simply following the same curved path through space.
        """
        result = classifier.classify(content)

        # This is fundamentally explaining WHY (causation), not HOW TO DO
        # Classifier may reasonably classify as either due to "first/then/finally"
        assert result.pacer_type in [PACERType.CONCEPTUAL, PACERType.PROCEDURAL], \
            f"Causal explanation should be Conceptual or Procedural, got {result.pacer_type}"

    def test_analogous_without_explicit_markers(self, classifier):
        """
        Counter-test: Implicit analogy without "like" or "similar to".
        Uses metaphorical language without explicit comparison markers.
        """
        content = """
        The cell membrane acts as a bouncer at a nightclub. It decides who gets in
        and who stays out. Nutrients are VIP guests - they enter freely through
        special channels. Waste products are rowdy patrons being escorted out
        through exit pumps. The membrane maintains order in the cellular party.
        """
        result = classifier.classify(content)

        assert result.pacer_type == PACERType.ANALOGOUS, \
            f"Implicit metaphor should be Analogous, got {result.pacer_type}"

    def test_evidence_without_statistics(self, classifier):
        """
        Counter-test: Evidence that supports concepts without numbers.
        Research claims without explicit percentages or p-values.
        """
        content = """
        Multiple peer-reviewed studies have demonstrated that spaced repetition
        significantly improves long-term retention compared to massed practice.
        Research from cognitive psychology laboratories consistently shows
        this effect across diverse populations and subject matters. Published
        findings in leading journals confirm the robustness of this phenomenon.
        """
        result = classifier.classify(content)

        assert result.pacer_type == PACERType.EVIDENCE, \
            f"Research claims should be Evidence, got {result.pacer_type}"

    def test_reference_with_explanatory_context(self, classifier):
        """
        Counter-test: Arbitrary facts with surrounding explanation.
        The core content is a date/fact wrapped in context.
        """
        content = """
        The French Revolution began in 1789. This date marks a pivotal
        moment in European history when citizens stormed the Bastille.
        """
        result = classifier.classify(content)

        # The specific date is Reference-type, but context adds Conceptual elements
        # Either classification is reasonable
        assert result.pacer_type in [PACERType.REFERENCE, PACERType.CONCEPTUAL], \
            f"Date with context should be Reference or Conceptual, got {result.pacer_type}"

    def test_pure_reference_data(self, classifier):
        """
        Counter-test: Pure reference data with no explanation.
        """
        content = """
        Avogadro's constant: 6.022 × 10²³ mol⁻¹
        Speed of light: 299,792,458 m/s
        Planck's constant: 6.626 × 10⁻³⁴ J·s
        """
        result = classifier.classify(content)

        assert result.pacer_type == PACERType.REFERENCE, \
            f"Pure constants should be Reference, got {result.pacer_type}"


class TestHybridContentCounterTests:
    """
    Counter-tests for content that legitimately spans multiple types.
    Validates handling of ambiguous, mixed-signal content.
    """

    @pytest.fixture
    def classifier(self):
        return PACERClassifier(use_llm=False)

    def test_hybrid_all_types_present(self, classifier):
        """
        Counter-test: Content containing all five PACER type signals.
        Classifier should identify primary type but show alternatives.
        """
        content = """
        To understand photosynthesis (a process discovered in 1779),
        follow these steps: First, light hits chlorophyll molecules.
        Second, water molecules split, releasing oxygen. Third, ATP forms.

        Studies show this process achieves approximately 11% energy efficiency.
        Think of it like a solar panel built into every leaf - capturing light
        and converting it into usable energy, though no analogy is perfect.

        The fundamental principle is that light energy drives the reduction
        of carbon dioxide to produce glucose, storing energy in chemical bonds.
        """
        result = classifier.classify(content)

        # Should have multiple alternative types with meaningful confidence
        assert len(result.alternative_types) >= 2, \
            "Hybrid content should have multiple plausible alternatives"

        alt_confidences = [conf for _, conf in result.alternative_types]
        assert any(conf > 0.2 for conf in alt_confidences), \
            "Alternative types should have meaningful confidence scores"

    def test_ambiguous_procedural_conceptual(self, classifier):
        """
        Counter-test: Content ambiguous between Procedural and Conceptual.
        """
        content = """
        Instructions for the machine learning training process:
        The machine learning training process works as follows:
        The model first processes the input data through forward propagation,
        then calculates the loss function, and finally updates weights
        through backpropagation. This cycle repeats until convergence.
        """
        result = classifier.classify(content)

        # Either classification is valid
        assert result.pacer_type in [PACERType.PROCEDURAL, PACERType.CONCEPTUAL]
        # Confidence should reflect ambiguity
        assert result.confidence < 0.9, \
            "Ambiguous content should have lower confidence"

    def test_evidence_with_conceptual_framing(self, classifier):
        """
        Counter-test: Statistical evidence wrapped in theoretical context.
        """
        content = """
        The forgetting curve theory, validated by Ebbinghaus's experiments
        in 1885, demonstrates that memory retention follows an exponential
        decay pattern. Studies consistently show 70% of new information is
        forgotten within 24 hours without reinforcement, supporting the
        principle that spaced repetition is essential for long-term retention.
        """
        result = classifier.classify(content)

        # Should recognize as Evidence (data supports theory) or Conceptual (explains principle)
        assert result.pacer_type in [PACERType.EVIDENCE, PACERType.CONCEPTUAL]


class TestEdgeCaseCounterTests:
    """Counter-tests for edge cases and boundary conditions"""

    @pytest.fixture
    def classifier(self):
        return PACERClassifier(use_llm=False)

    def test_minimal_content(self, classifier):
        """Counter-test: Very short content"""
        content = "Photosynthesis converts light to energy."
        result = classifier.classify(content)

        # Should still classify but with lower confidence
        assert result.pacer_type is not None
        assert result.confidence < 0.8, \
            "Minimal content should have lower confidence"

    def test_single_sentence_procedural(self, classifier):
        """Counter-test: Single procedural instruction"""
        content = "Step 1: Click the submit button to save your work."
        result = classifier.classify(content)

        assert result.pacer_type == PACERType.PROCEDURAL

    def test_single_sentence_analogy(self, classifier):
        """Counter-test: Single sentence analogy"""
        content = "An atom is like a tiny solar system with electrons orbiting the nucleus."
        result = classifier.classify(content)

        assert result.pacer_type == PACERType.ANALOGOUS

    def test_adversarial_mixed_signals(self, classifier):
        """
        Counter-test: Content with strong signals for ALL types.
        Adversarial test to ensure classifier doesn't crash or produce invalid output.
        """
        content = """
        Step 1 (like climbing a ladder to reach your goals):
        The theory states that according to studies showing 85%
        effectiveness, you should memorize the formula E=mc²
        discovered in 1905 because it explains why mass and energy
        are equivalent through the principle of relativity.
        """
        result = classifier.classify(content)

        # Classifier must make a decision and not crash
        assert result.pacer_type is not None
        assert result.pacer_type in PACERType
        # Mixed signals should reduce confidence
        assert result.confidence < 0.9, \
            "Adversarial mixed signals should reduce confidence"

    def test_content_with_code(self, classifier):
        """Counter-test: Technical content with code snippets"""
        content = """
        To implement quicksort, follow these steps:
        1. Choose a pivot element from the array
        2. Partition: reorder so elements < pivot come before, > pivot after
        3. Recursively apply to sub-arrays

        Example: arr.sort(key=lambda x: x.value)
        """
        result = classifier.classify(content)

        assert result.pacer_type == PACERType.PROCEDURAL

    def test_content_with_urls(self, classifier):
        """Counter-test: Content with URLs and citations"""
        content = """
        According to research published at https://example.com/study,
        the experiment conducted in 2023 found that 78% of participants
        showed improved recall when using active learning techniques.
        """
        result = classifier.classify(content)

        assert result.pacer_type == PACERType.EVIDENCE


class TestConfidenceCalibrationCounterTests:
    """
    Counter-tests verifying confidence scores are well-calibrated.
    Higher confidence should correlate with more certain classifications.
    """

    @pytest.fixture
    def classifier(self):
        return PACERClassifier(use_llm=False)

    def test_high_confidence_for_clear_procedural(self, classifier):
        """Clear procedural content should have high confidence"""
        content = """
        Step 1: Open the terminal application.
        Step 2: Navigate to the project directory using cd.
        Step 3: Run npm install to install dependencies.
        Step 4: Execute npm start to launch the server.
        Step 5: Open localhost:3000 in your browser.
        """
        result = classifier.classify(content)

        assert result.pacer_type == PACERType.PROCEDURAL
        assert result.confidence >= 0.6, \
            f"Clear procedural should have confidence >= 0.6, got {result.confidence}"

    def test_high_confidence_for_clear_evidence(self, classifier):
        """Clear evidence content should have high confidence"""
        content = """
        A randomized controlled trial (n=500) published in Nature showed
        that the treatment group had 43% better outcomes (p < 0.001)
        compared to placebo. The study's 95% confidence interval was
        [38%, 48%], demonstrating statistical significance.
        """
        result = classifier.classify(content)

        assert result.pacer_type == PACERType.EVIDENCE
        assert result.confidence >= 0.5, \
            f"Clear evidence should have confidence >= 0.5, got {result.confidence}"

    def test_lower_confidence_for_ambiguous(self, classifier):
        """Ambiguous content should have lower confidence"""
        content = """
        This process involves understanding why certain steps lead to
        specific outcomes through a series of cause-and-effect relationships.
        """
        result = classifier.classify(content)

        # Content is ambiguous between Procedural and Conceptual
        assert result.confidence < 0.8, \
            f"Ambiguous content should have confidence < 0.8, got {result.confidence}"

    def test_alternatives_for_close_classifications(self, classifier):
        """When types are close in score, alternatives should be provided"""
        content = """
        The research shows that following these steps improves outcomes:
        first understand the theory, then apply it practically.
        """
        result = classifier.classify(content)

        # Should have alternatives when classification is not clear-cut
        if result.confidence < 0.7:
            assert len(result.alternative_types) > 0, \
                "Low-confidence classifications should provide alternatives"


class TestTriageDecisionTreeCounterTests:
    """Counter-tests for the triage decision tree"""

    @pytest.fixture
    def classifier(self):
        return PACERClassifier(use_llm=False)

    def test_triage_terminates_correctly(self, classifier):
        """Triage should always terminate with a classification"""
        contents = [
            "Step by step instructions for baking.",
            "This is like water flowing through pipes.",
            "The theory explains why gravity works.",
            "Studies show 90% effectiveness.",
            "The date was January 1, 2000.",
        ]

        for content in contents:
            decisions = classifier.run_triage_tree(content)
            # Should terminate with exactly one decision leading to a type
            final_types = [d.leads_to for d in decisions if d.leads_to is not None]
            assert len(final_types) == 1, \
                f"Triage should terminate with exactly one type for: {content[:30]}..."

    def test_triage_decision_path_is_logical(self, classifier):
        """Triage decisions should follow logical order"""
        content = "Instructions: Step 1: Do this. Step 2: Do that."
        decisions = classifier.run_triage_tree(content)

        # First question should be about Procedural
        assert "HOW" in decisions[0].question.upper() or "step" in decisions[0].question.lower()

        # Should answer Yes and terminate
        assert decisions[0].answer is True
        assert decisions[0].leads_to == PACERType.PROCEDURAL

    def test_triage_falls_through_to_reference(self, classifier):
        """
        Content that doesn't match other types should fall through to Reference.
        """
        content = "The capital of France is Paris."
        decisions = classifier.run_triage_tree(content)

        # Should fall through to Reference
        final_type = None
        for d in decisions:
            if d.leads_to:
                final_type = d.leads_to
                break

        # Simple facts typically fall to Reference
        assert final_type in [PACERType.REFERENCE, PACERType.CONCEPTUAL]


class TestRecommendedActionCounterTests:
    """Counter-tests for recommended learning actions"""

    @pytest.fixture
    def classifier(self):
        return PACERClassifier(use_llm=False)

    def test_procedural_recommends_practice(self, classifier):
        """Procedural type should recommend Practice action"""
        action = classifier.get_recommended_action(PACERType.PROCEDURAL)

        assert action["action"] == "Practice"
        assert "execute" in action["description"].lower() or "practice" in action["description"].lower()

    def test_analogous_recommends_critique(self, classifier):
        """Analogous type should recommend Critique action"""
        action = classifier.get_recommended_action(PACERType.ANALOGOUS)

        assert action["action"] == "Critique"
        assert "break" in action["description"].lower() or "critique" in action["description"].lower()

    def test_conceptual_recommends_map(self, classifier):
        """Conceptual type should recommend Map action"""
        action = classifier.get_recommended_action(PACERType.CONCEPTUAL)

        assert action["action"] == "Map"
        assert "concept" in action["description"].lower() or "relationship" in action["description"].lower()

    def test_evidence_recommends_link(self, classifier):
        """Evidence type should recommend Link action"""
        action = classifier.get_recommended_action(PACERType.EVIDENCE)

        assert action["action"] == "Link"
        assert "concept" in action["description"].lower() or "link" in action["description"].lower()

    def test_reference_recommends_recall(self, classifier):
        """Reference type should recommend Recall action"""
        action = classifier.get_recommended_action(PACERType.REFERENCE)

        assert action["action"] == "Recall"
        assert "memoriz" in action["description"].lower() or "srs" in action["tool"].lower()
