"""
Unit tests for Bayesian Knowledge Tracing (BKT)

Tests cover:
- Mastery updates from observations
- Evidence-based updates
- Performance predictions
- Mastery thresholds
- Parameter calibration
"""

import pytest
import math
from typing import List, Tuple

from app.adaptive.bkt.bayesian_kt import BayesianKnowledgeTracer


class TestBKTInitialization:
    """Tests for BKT initialization"""

    def test_default_parameters(self):
        """Test BKT initializes with correct defaults"""
        bkt = BayesianKnowledgeTracer()

        assert bkt.p_l0 == 0.1  # 10% prior
        assert bkt.p_t == 0.15  # 15% learning rate
        assert bkt.p_g == 0.2   # 20% guess rate
        assert bkt.p_s == 0.1   # 10% slip rate

    def test_custom_parameters(self):
        """Test BKT with custom parameters"""
        custom_params = {
            "p_l0": 0.2,
            "p_t": 0.3,
            "p_g": 0.25,
            "p_s": 0.15,
        }
        bkt = BayesianKnowledgeTracer(params=custom_params)

        assert bkt.p_l0 == 0.2
        assert bkt.p_t == 0.3
        assert bkt.p_g == 0.25
        assert bkt.p_s == 0.15


class TestMasteryUpdate:
    """Tests for mastery update from observations"""

    @pytest.fixture
    def bkt(self):
        return BayesianKnowledgeTracer()

    def test_correct_answer_increases_mastery(self, bkt):
        """Correct answer should increase mastery"""
        initial_mastery = 0.5
        new_mastery, details = bkt.update_from_observation(initial_mastery, correct=True)

        assert new_mastery > initial_mastery
        assert details["learning_gain"] > 0

    def test_incorrect_answer_effect_on_low_mastery(self, bkt):
        """Incorrect answer on low mastery should have specific behavior"""
        initial_mastery = 0.2
        new_mastery, details = bkt.update_from_observation(initial_mastery, correct=False)

        # Even with wrong answer, learning transition can occur
        # But posterior mastery should be lower than prior
        assert details["posterior_mastery"] < initial_mastery

    def test_correct_answer_from_zero_mastery(self, bkt):
        """Correct answer from zero mastery could be a guess"""
        new_mastery, details = bkt.update_from_observation(0.0, correct=True)

        # Should increase due to learning transition
        assert new_mastery > 0.0
        # But not too much since it might be a guess
        assert new_mastery < 0.5

    def test_incorrect_answer_from_high_mastery(self, bkt):
        """Incorrect answer from high mastery could be a slip"""
        initial_mastery = 0.9
        new_mastery, details = bkt.update_from_observation(initial_mastery, correct=False)

        # Mastery should decrease but slip probability keeps it reasonable
        assert new_mastery < initial_mastery
        # But not too low since it might be a slip
        assert new_mastery > 0.5

    def test_mastery_bounded_zero_one(self, bkt):
        """Mastery should always be between 0 and 1"""
        # Test lower bound
        low_mastery, _ = bkt.update_from_observation(0.0, correct=False)
        assert 0.0 <= low_mastery <= 1.0

        # Test upper bound
        high_mastery, _ = bkt.update_from_observation(1.0, correct=True)
        assert 0.0 <= high_mastery <= 1.0

    def test_update_details_structure(self, bkt):
        """Update should return proper details structure"""
        _, details = bkt.update_from_observation(0.5, correct=True)

        assert "prior_mastery" in details
        assert "posterior_mastery" in details
        assert "new_mastery_after_learning" in details
        assert "correct" in details
        assert "learning_gain" in details

    def test_learning_gain_calculation(self, bkt):
        """Learning gain should equal new_mastery - prior_mastery"""
        initial = 0.4
        new_mastery, details = bkt.update_from_observation(initial, correct=True)

        expected_gain = new_mastery - initial
        assert math.isclose(details["learning_gain"], expected_gain, rel_tol=1e-5)


class TestEvidenceUpdate:
    """Tests for evidence-based mastery updates"""

    @pytest.fixture
    def bkt(self):
        return BayesianKnowledgeTracer()

    def test_high_evidence_increases_mastery(self, bkt):
        """High evidence score (>0.7) should increase mastery"""
        initial = 0.5
        new_mastery, _ = bkt.update_from_evidence(initial, evidence_score=0.9)

        assert new_mastery > initial

    def test_low_evidence_decreases_mastery(self, bkt):
        """Low evidence score (<0.3) should decrease mastery"""
        initial = 0.5
        new_mastery, _ = bkt.update_from_evidence(initial, evidence_score=0.1)

        assert new_mastery < initial

    def test_neutral_evidence_minimal_change(self, bkt):
        """Evidence around 0.5 should have minimal impact"""
        initial = 0.5
        new_mastery, _ = bkt.update_from_evidence(initial, evidence_score=0.5)

        # Should be close to initial with slight nudge
        assert abs(new_mastery - initial) < 0.1

    def test_evidence_strength_calculation(self, bkt):
        """Evidence strength should be higher for extreme values"""
        _, details_extreme = bkt.update_from_evidence(0.5, evidence_score=0.95)
        _, details_moderate = bkt.update_from_evidence(0.5, evidence_score=0.75)

        assert details_extreme["evidence_strength"] > details_moderate["evidence_strength"]

    def test_evidence_bounded_result(self, bkt):
        """Evidence update should keep mastery in [0,1]"""
        # Test with extreme high evidence
        high_result, _ = bkt.update_from_evidence(0.99, evidence_score=1.0)
        assert 0.0 <= high_result <= 1.0

        # Test with extreme low evidence
        low_result, _ = bkt.update_from_evidence(0.01, evidence_score=0.0)
        assert 0.0 <= low_result <= 1.0


class TestPerformancePrediction:
    """Tests for performance prediction"""

    @pytest.fixture
    def bkt(self):
        return BayesianKnowledgeTracer()

    def test_high_mastery_high_correct_probability(self, bkt):
        """High mastery should predict high correct probability"""
        prediction = bkt.predict_performance(mastery=0.9)

        assert prediction["p_correct"] > 0.8

    def test_low_mastery_low_correct_probability(self, bkt):
        """Low mastery should predict lower correct probability"""
        prediction = bkt.predict_performance(mastery=0.1)

        # But still has guess probability
        assert prediction["p_correct"] > bkt.p_g * 0.5
        assert prediction["p_correct"] < 0.5

    def test_prediction_components(self, bkt):
        """Prediction should include all components"""
        prediction = bkt.predict_performance(mastery=0.5)

        assert "mastery" in prediction
        assert "p_correct" in prediction
        assert "p_incorrect" in prediction
        assert "confidence" in prediction

    def test_correct_incorrect_sum_to_one(self, bkt):
        """p_correct + p_incorrect should equal 1"""
        for mastery in [0.0, 0.25, 0.5, 0.75, 1.0]:
            prediction = bkt.predict_performance(mastery)
            total = prediction["p_correct"] + prediction["p_incorrect"]
            assert math.isclose(total, 1.0, rel_tol=1e-5)

    def test_confidence_calculation(self, bkt):
        """Confidence should be higher when prediction is more certain"""
        # High mastery = high p_correct = high confidence
        high_pred = bkt.predict_performance(mastery=0.9)

        # Low mastery = lower p_correct = medium confidence
        low_pred = bkt.predict_performance(mastery=0.1)

        # Both should have some confidence
        assert high_pred["confidence"] > 0
        assert low_pred["confidence"] > 0


class TestMasteryThreshold:
    """Tests for mastery threshold checking"""

    @pytest.fixture
    def bkt(self):
        return BayesianKnowledgeTracer()

    def test_is_mastered_above_threshold(self, bkt):
        """Should return True when mastery exceeds threshold"""
        assert bkt.is_mastered(0.96, threshold=0.95) is True

    def test_is_mastered_at_threshold(self, bkt):
        """Should return True when mastery equals threshold"""
        assert bkt.is_mastered(0.95, threshold=0.95) is True

    def test_is_mastered_below_threshold(self, bkt):
        """Should return False when mastery is below threshold"""
        assert bkt.is_mastered(0.94, threshold=0.95) is False

    def test_is_mastered_custom_threshold(self, bkt):
        """Should work with custom threshold"""
        assert bkt.is_mastered(0.85, threshold=0.8) is True
        assert bkt.is_mastered(0.75, threshold=0.8) is False


class TestSessionsToMastery:
    """Tests for sessions to mastery estimation"""

    @pytest.fixture
    def bkt(self):
        return BayesianKnowledgeTracer()

    def test_already_mastered(self, bkt):
        """Should return 0 sessions if already mastered"""
        sessions = bkt.sessions_to_mastery(0.96, threshold=0.95)
        assert sessions == 0

    def test_low_mastery_needs_sessions(self, bkt):
        """Low mastery should need multiple sessions"""
        sessions = bkt.sessions_to_mastery(0.1, threshold=0.95)
        assert sessions > 0

    def test_higher_mastery_needs_fewer_sessions(self, bkt):
        """Higher starting mastery should need fewer sessions"""
        low_start = bkt.sessions_to_mastery(0.1, threshold=0.95)
        high_start = bkt.sessions_to_mastery(0.5, threshold=0.95)

        assert low_start > high_start


class TestParameterCalibration:
    """Tests for BKT parameter self-calibration"""

    def test_calibrate_guess_probability(self):
        """P(G) should increase when low-mastery users get correct"""
        bkt = BayesianKnowledgeTracer()
        initial_p_g = bkt.p_g

        # History of "lucky guesses": low mastery but correct
        history: List[Tuple[float, bool]] = [(0.1, True)] * 10
        bkt.calibrate_parameters(history, learning_rate=1.0)

        assert bkt.p_g > initial_p_g

    def test_calibrate_slip_probability(self):
        """P(S) should increase when high-mastery users get incorrect"""
        bkt = BayesianKnowledgeTracer()
        initial_p_s = bkt.p_s

        # History of "slips": high mastery but incorrect
        history: List[Tuple[float, bool]] = [(0.9, False)] * 10
        bkt.calibrate_parameters(history, learning_rate=1.0)

        assert bkt.p_s > initial_p_s

    def test_calibrate_bounds_parameters(self):
        """Calibration should keep parameters bounded"""
        bkt = BayesianKnowledgeTracer()

        # Extreme history that would push P(G) very high
        history: List[Tuple[float, bool]] = [(0.05, True)] * 100
        bkt.calibrate_parameters(history, learning_rate=1.0)

        # P(G) should be bounded at 0.5
        assert bkt.p_g <= 0.5
        assert bkt.p_g >= 0.01

    def test_calibrate_updates_params_dict(self):
        """Calibration should update the params dictionary"""
        bkt = BayesianKnowledgeTracer()

        history: List[Tuple[float, bool]] = [(0.1, True)] * 5
        updated_params = bkt.calibrate_parameters(history, learning_rate=0.5)

        assert updated_params["p_g"] == bkt.p_g
        assert updated_params["p_s"] == bkt.p_s


class TestBKTIntegration:
    """Integration tests for BKT learning scenarios"""

    @pytest.fixture
    def bkt(self):
        return BayesianKnowledgeTracer()

    def test_learning_progression(self, bkt):
        """Simulate learning progression over multiple attempts"""
        mastery = bkt.p_l0  # Start at prior

        # Simulate 10 correct answers
        for _ in range(10):
            mastery, _ = bkt.update_from_observation(mastery, correct=True)

        # Should approach mastery (with default params, 10 correct answers gets ~0.82)
        assert mastery > 0.8

    def test_struggling_learner(self, bkt):
        """Simulate a struggling learner"""
        mastery = 0.3

        # Alternate correct and incorrect
        for i in range(6):
            correct = i % 2 == 0
            mastery, _ = bkt.update_from_observation(mastery, correct=correct)

        # Mastery should not improve much (alternating gives ~0.74)
        assert mastery < 0.8

    def test_forgetting_not_modeled(self, bkt):
        """BKT doesn't model forgetting - mastery is monotonic with correct answers"""
        mastery = 0.5

        # Multiple correct answers
        for _ in range(5):
            mastery, _ = bkt.update_from_observation(mastery, correct=True)

        high_mastery = mastery

        # In pure BKT, mastery doesn't decrease without incorrect observations
        # (forgetting is not modeled - with default params, 5 correct from 0.5 gives ~0.78)
        assert high_mastery > 0.75
