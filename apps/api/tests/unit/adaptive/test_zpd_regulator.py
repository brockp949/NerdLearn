"""
Unit tests for Zone of Proximal Development (ZPD) Regulator

Tests cover:
- ZPD score calculations
- Prerequisite readiness
- Content recommendations
- Difficulty adjustments
- Learning velocity
- Review scheduling
"""

import pytest
from unittest.mock import MagicMock

from app.adaptive.zpd.zpd_regulator import (
    ZPDRegulator,
    ConceptMastery,
    ContentRecommendation,
)


class TestZPDRegulatorInitialization:
    """Tests for ZPD regulator initialization"""

    def test_default_parameters(self):
        """Test default ZPD parameters"""
        zpd = ZPDRegulator()

        assert zpd.zpd_width == 0.3
        assert zpd.optimal_mastery == 0.6
        assert zpd.frustration_threshold == 0.9
        assert zpd.boredom_threshold == 0.3

    def test_custom_parameters(self):
        """Test custom ZPD parameters"""
        zpd = ZPDRegulator(
            zpd_width=0.4,
            optimal_mastery=0.5,
            frustration_threshold=0.8,
            boredom_threshold=0.2,
        )

        assert zpd.zpd_width == 0.4
        assert zpd.optimal_mastery == 0.5


class TestConceptMastery:
    """Tests for ConceptMastery dataclass"""

    def test_concept_mastery_creation(self):
        """Test creating ConceptMastery"""
        concept = ConceptMastery(
            concept_id=1,
            concept_name="Binary Search",
            mastery_level=0.7,
            stability=5.0,
        )

        assert concept.concept_id == 1
        assert concept.concept_name == "Binary Search"
        assert concept.mastery_level == 0.7
        assert concept.stability == 5.0
        assert concept.is_prerequisite is False
        assert concept.difficulty == 5.0


class TestZPDScoreCalculation:
    """Tests for ZPD score calculation"""

    @pytest.fixture
    def zpd(self):
        return ZPDRegulator()

    def test_optimal_zpd_score(self, zpd):
        """Content slightly harder than mastery should have high ZPD score"""
        # User mastery at 0.5, content at 0.65 (in the ZPD)
        score, zone = zpd.calculate_zpd_score(
            user_mastery=0.5,
            content_difficulty=0.65,
            prerequisites_met=True,
        )

        assert score > 0.7
        assert "Optimal" in zone or "Acceptable" in zone

    def test_too_easy_content(self, zpd):
        """Content much easier than mastery should score low"""
        score, zone = zpd.calculate_zpd_score(
            user_mastery=0.8,
            content_difficulty=0.3,
            prerequisites_met=True,
        )

        assert "Easy" in zone or "Boredom" in zone

    def test_too_hard_content(self, zpd):
        """Content much harder than mastery should score low"""
        score, zone = zpd.calculate_zpd_score(
            user_mastery=0.2,
            content_difficulty=0.95,
            prerequisites_met=True,
        )

        assert score < 0.5
        assert "Hard" in zone or "Frustration" in zone or "Suboptimal" in zone

    def test_prerequisites_not_met(self, zpd):
        """Missing prerequisites should give zero score"""
        score, zone = zpd.calculate_zpd_score(
            user_mastery=0.5,
            content_difficulty=0.6,
            prerequisites_met=False,
        )

        assert score == 0.0
        assert "Prerequisites" in zone

    def test_zpd_score_bounded(self, zpd):
        """ZPD score should be between 0 and 1"""
        for mastery in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for difficulty in [0.0, 0.25, 0.5, 0.75, 1.0]:
                score, _ = zpd.calculate_zpd_score(mastery, difficulty, True)
                assert 0.0 <= score <= 1.0


class TestPrerequisiteReadiness:
    """Tests for prerequisite readiness calculation"""

    @pytest.fixture
    def zpd(self):
        return ZPDRegulator()

    def test_no_prerequisites(self, zpd):
        """No prerequisites should return ready"""
        met, avg = zpd.calculate_prerequisite_readiness([])

        assert met is True
        assert avg == 1.0

    def test_all_prerequisites_met(self, zpd):
        """All prerequisites above threshold should return ready"""
        prereqs = [
            ConceptMastery(1, "Concept A", mastery_level=0.8, stability=5.0),
            ConceptMastery(2, "Concept B", mastery_level=0.9, stability=6.0),
            ConceptMastery(3, "Concept C", mastery_level=0.75, stability=4.0),
        ]

        met, avg = zpd.calculate_prerequisite_readiness(prereqs, threshold=0.7)

        assert met is True
        assert avg > 0.7

    def test_some_prerequisites_not_met(self, zpd):
        """Some prerequisites below threshold should return not ready"""
        prereqs = [
            ConceptMastery(1, "Concept A", mastery_level=0.9, stability=5.0),
            ConceptMastery(2, "Concept B", mastery_level=0.5, stability=3.0),
        ]

        met, avg = zpd.calculate_prerequisite_readiness(prereqs, threshold=0.7)

        assert met is False

    def test_average_mastery_calculation(self, zpd):
        """Should correctly calculate average mastery"""
        prereqs = [
            ConceptMastery(1, "A", mastery_level=0.6, stability=1.0),
            ConceptMastery(2, "B", mastery_level=0.8, stability=1.0),
        ]

        _, avg = zpd.calculate_prerequisite_readiness(prereqs)

        assert avg == 0.7


class TestContentRecommendation:
    """Tests for content recommendation"""

    @pytest.fixture
    def zpd(self):
        return ZPDRegulator()

    @pytest.fixture
    def sample_modules(self):
        return [
            {"id": 1, "concepts": [101, 102], "difficulty": 3.0},
            {"id": 2, "concepts": [103, 104], "difficulty": 5.0},
            {"id": 3, "concepts": [105, 106], "difficulty": 7.0},
            {"id": 4, "concepts": [107], "difficulty": 9.0},
        ]

    def test_recommend_content_returns_sorted_list(self, zpd, sample_modules):
        """Recommendations should be sorted by ZPD score"""
        masteries = {
            101: 0.5, 102: 0.6,  # Module 1
            103: 0.4, 104: 0.5,  # Module 2
            105: 0.3, 106: 0.4,  # Module 3
            107: 0.2,           # Module 4
        }

        recommendations = zpd.recommend_content(
            user_concept_masteries=masteries,
            available_modules=sample_modules,
            concept_prerequisites={},
            top_n=4,
        )

        # Should be sorted by ZPD score (descending)
        scores = [r.zpd_score for r in recommendations]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_content_respects_top_n(self, zpd, sample_modules):
        """Should return at most top_n recommendations"""
        masteries = {c: 0.5 for c in range(101, 108)}

        recommendations = zpd.recommend_content(
            user_concept_masteries=masteries,
            available_modules=sample_modules,
            concept_prerequisites={},
            top_n=2,
        )

        assert len(recommendations) <= 2

    def test_recommend_content_skips_unmet_prerequisites(self, zpd):
        """Should penalize content with unmet prerequisites"""
        modules = [
            {"id": 1, "concepts": [101], "difficulty": 5.0},
            {"id": 2, "concepts": [102], "difficulty": 5.0},
        ]

        masteries = {101: 0.5, 102: 0.5, 100: 0.3}
        prerequisites = {102: [100]}  # 102 requires 100 (mastery 0.3)

        recommendations = zpd.recommend_content(
            user_concept_masteries=masteries,
            available_modules=modules,
            concept_prerequisites=prerequisites,
            top_n=2,
        )

        # Module 1 (no prereqs) should rank higher than module 2 (unmet prereqs)
        module_ids = [r.module_id for r in recommendations]
        assert module_ids[0] == 1

    def test_recommendation_has_required_fields(self, zpd, sample_modules):
        """ContentRecommendation should have all required fields"""
        masteries = {c: 0.5 for c in range(101, 108)}

        recommendations = zpd.recommend_content(
            user_concept_masteries=masteries,
            available_modules=sample_modules,
            concept_prerequisites={},
        )

        for rec in recommendations:
            assert hasattr(rec, "module_id")
            assert hasattr(rec, "concept_ids")
            assert hasattr(rec, "zpd_score")
            assert hasattr(rec, "difficulty")
            assert hasattr(rec, "rationale")
            assert hasattr(rec, "estimated_success_rate")


class TestDifficultyAdjustment:
    """Tests for difficulty adjustment"""

    @pytest.fixture
    def zpd(self):
        return ZPDRegulator()

    def test_high_performance_increases_difficulty(self, zpd):
        """High performance should increase difficulty"""
        new_diff = zpd.adjust_difficulty(
            current_difficulty=0.5,
            performance=0.9,
            adaptation_rate=0.1,
        )

        assert new_diff > 0.5

    def test_low_performance_decreases_difficulty(self, zpd):
        """Low performance should decrease difficulty"""
        new_diff = zpd.adjust_difficulty(
            current_difficulty=0.5,
            performance=0.4,
            adaptation_rate=0.1,
        )

        assert new_diff < 0.5

    def test_target_performance_no_change(self, zpd):
        """Performance at target should minimize change"""
        new_diff = zpd.adjust_difficulty(
            current_difficulty=0.5,
            performance=0.75,  # Target is 0.75
            adaptation_rate=0.1,
        )

        assert abs(new_diff - 0.5) < 0.01

    def test_difficulty_bounded(self, zpd):
        """Difficulty should stay bounded"""
        # Test lower bound
        low_diff = zpd.adjust_difficulty(0.1, performance=0.2, adaptation_rate=0.5)
        assert low_diff >= 0.1

        # Test upper bound
        high_diff = zpd.adjust_difficulty(0.9, performance=1.0, adaptation_rate=0.5)
        assert high_diff <= 0.9


class TestLearningVelocity:
    """Tests for learning velocity calculation"""

    @pytest.fixture
    def zpd(self):
        return ZPDRegulator()

    def test_positive_velocity(self, zpd):
        """Increasing mastery should give positive velocity"""
        history = [(1, 0.1), (2, 0.2), (3, 0.3), (4, 0.4)]
        velocity = zpd.get_learning_velocity(history)

        assert velocity > 0

    def test_negative_velocity(self, zpd):
        """Decreasing mastery should give negative velocity"""
        history = [(1, 0.4), (2, 0.3), (3, 0.2), (4, 0.1)]
        velocity = zpd.get_learning_velocity(history)

        assert velocity < 0

    def test_zero_velocity_constant(self, zpd):
        """Constant mastery should give near-zero velocity"""
        history = [(1, 0.5), (2, 0.5), (3, 0.5), (4, 0.5)]
        velocity = zpd.get_learning_velocity(history)

        assert abs(velocity) < 0.01

    def test_insufficient_history(self, zpd):
        """Single point should return zero velocity"""
        velocity = zpd.get_learning_velocity([(1, 0.5)])
        assert velocity == 0.0

    def test_empty_history(self, zpd):
        """Empty history should return zero velocity"""
        velocity = zpd.get_learning_velocity([])
        assert velocity == 0.0


class TestReviewScheduling:
    """Tests for review scheduling decisions"""

    @pytest.fixture
    def zpd(self):
        return ZPDRegulator()

    def test_mastered_concept_no_review(self, zpd):
        """Mastered concept with high stability shouldn't need review"""
        should_review, reason = zpd.should_review_concept(
            mastery=0.9,
            stability=40,
            days_since_review=5,
        )

        assert should_review is False
        assert "mastered" in reason.lower()

    def test_low_mastery_needs_review(self, zpd):
        """Low mastery concept should need review"""
        should_review, reason = zpd.should_review_concept(
            mastery=0.5,
            stability=5,
            days_since_review=3,
        )

        assert should_review is True
        assert "mastery" in reason.lower()

    def test_low_retrievability_needs_review(self, zpd):
        """Low retrievability should trigger review"""
        should_review, reason = zpd.should_review_concept(
            mastery=0.6,
            stability=2,
            days_since_review=20,  # Long time since review
        )

        assert should_review is True
        assert "retrievability" in reason.lower()

    def test_recent_review_no_need(self, zpd):
        """Recent review with good mastery shouldn't need another"""
        should_review, reason = zpd.should_review_concept(
            mastery=0.75,
            stability=10,
            days_since_review=1,
        )

        assert should_review is False


class TestZPDIntegration:
    """Integration tests for ZPD scenarios"""

    @pytest.fixture
    def zpd(self):
        return ZPDRegulator()

    def test_learning_path_difficulty_progression(self, zpd):
        """Content should progress in difficulty as user learns"""
        modules = [
            {"id": 1, "concepts": [1], "difficulty": 3.0},
            {"id": 2, "concepts": [2], "difficulty": 5.0},
            {"id": 3, "concepts": [3], "difficulty": 7.0},
        ]

        # Beginner user
        beginner_masteries = {1: 0.2, 2: 0.1, 3: 0.0}
        beginner_recs = zpd.recommend_content(
            beginner_masteries, modules, {}, top_n=3
        )

        # Intermediate user
        intermediate_masteries = {1: 0.8, 2: 0.5, 3: 0.2}
        intermediate_recs = zpd.recommend_content(
            intermediate_masteries, modules, {}, top_n=3
        )

        # Beginner should prefer easier content
        # Intermediate should prefer harder content
        beginner_top_difficulty = beginner_recs[0].difficulty
        intermediate_top_difficulty = intermediate_recs[0].difficulty

        assert beginner_top_difficulty < intermediate_top_difficulty

    def test_estimated_success_rate_reasonable(self, zpd):
        """Estimated success rates should be reasonable"""
        modules = [{"id": 1, "concepts": [1], "difficulty": 5.0}]
        masteries = {1: 0.5}

        recs = zpd.recommend_content(masteries, modules, {})

        for rec in recs:
            assert 0.0 <= rec.estimated_success_rate <= 1.0
