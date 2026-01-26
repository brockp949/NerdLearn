import pytest
from app.adaptive.bkt.bayesian_kt import BayesianKnowledgeTracer


class TestBayesianKnowledgeTracer:
    """Unit tests for Bayesian Knowledge Tracing logic"""

    @pytest.fixture
    def bkt(self):
        return BayesianKnowledgeTracer()

    def test_initialization(self, bkt):
        """Test default parameter initialization"""
        assert bkt.p_l0 == 0.1
        assert bkt.p_t == 0.15
        assert bkt.p_g == 0.2
        assert bkt.p_s == 0.1

    def test_update_correct_answer(self, bkt):
        """Test mastery update after a correct answer"""
        initial_mastery = 0.5
        new_mastery, details = bkt.update_from_observation(initial_mastery, correct=True)
        
        # Expect mastery to increase
        assert new_mastery > initial_mastery
        assert details["correct"] is True
        assert details["learning_gain"] > 0

    def test_update_incorrect_answer(self, bkt):
        """Test mastery update after an incorrect answer"""
        initial_mastery = 0.5
        new_mastery, details = bkt.update_from_observation(initial_mastery, correct=False)
        
        # Expect mastery to decrease (or stay same if purely guessing, but usually decreases)
        # With default params: p_s=0.1, p_g=0.2
        # Incorrect answer is evidence against mastery
        assert new_mastery < initial_mastery
        assert details["correct"] is False
        assert details["learning_gain"] < 0

    def test_update_from_evidence_strong_positive(self, bkt):
        """Test update from strong positive continuous evidence"""
        initial_mastery = 0.3
        evidence_score = 0.95  # High confidence correct
        new_mastery, _ = bkt.update_from_evidence(initial_mastery, evidence_score)
        
        assert new_mastery > initial_mastery
        # Should be a significant jump
        assert new_mastery > 0.4

    def test_update_from_evidence_strong_negative(self, bkt):
        """Test update from strong negative continuous evidence"""
        initial_mastery = 0.7
        evidence_score = 0.1  # High confidence incorrect
        new_mastery, _ = bkt.update_from_evidence(initial_mastery, evidence_score)
        
        assert new_mastery < initial_mastery

    def test_update_from_evidence_uncertain(self, bkt):
        """Test update from uncertain evidence (near 0.5)"""
        initial_mastery = 0.5
        evidence_score = 0.5
        new_mastery, _ = bkt.update_from_evidence(initial_mastery, evidence_score)
        
        # Should change very little
        assert abs(new_mastery - initial_mastery) < 0.1

    def test_sessions_to_mastery(self, bkt):
        """Test prediction of sessions needed to master"""
        # Already mastered
        assert bkt.sessions_to_mastery(0.96) == 0
        
        # Needs practice
        sessions = bkt.sessions_to_mastery(0.1)
        assert sessions > 0
        assert sessions < 100  # Should be reachable

    def test_boundary_conditions(self, bkt):
        """Test updates at probability boundaries"""
        # At 0.0 mastery
        m0, _ = bkt.update_from_observation(0.0, correct=False)
        assert m0 >= 0.0
        
        # At 1.0 mastery
        m1, _ = bkt.update_from_observation(1.0, correct=True)
        assert m1 <= 1.0
