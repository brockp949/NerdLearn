"""
Unit tests for FSRS (Free Spaced Repetition Scheduler) Algorithm

Tests cover:
- Retrievability calculations
- Stability updates
- Difficulty adjustments
- Interval calculations
- Card state management
- Rating scenarios
"""

import pytest
import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from app.adaptive.fsrs.fsrs_algorithm import (
    FSRSAlgorithm,
    FSRSCard,
    Rating,
)


class TestRating:
    """Tests for Rating enum"""

    def test_rating_values(self):
        """Verify rating enum values"""
        assert Rating.AGAIN.value == 1
        assert Rating.HARD.value == 2
        assert Rating.GOOD.value == 3
        assert Rating.EASY.value == 4

    def test_rating_order(self):
        """Verify ratings are ordered by difficulty"""
        ratings = list(Rating)
        assert ratings[0] == Rating.AGAIN
        assert ratings[-1] == Rating.EASY


class TestFSRSCard:
    """Tests for FSRSCard dataclass"""

    def test_card_initialization_defaults(self):
        """Test card initializes with correct defaults"""
        card = FSRSCard(concept_id=1, user_id=1)

        assert card.concept_id == 1
        assert card.user_id == 1
        assert card.stability == 0.0
        assert card.difficulty == 0.0
        assert card.elapsed_days == 0
        assert card.scheduled_days == 0
        assert card.reps == 0
        assert card.lapses == 0
        assert card.state == "new"
        assert card.last_review is None
        assert card.due is not None

    def test_card_initialization_with_values(self):
        """Test card initializes with provided values"""
        now = datetime.now()
        card = FSRSCard(
            concept_id=123,
            user_id=456,
            stability=5.0,
            difficulty=3.0,
            elapsed_days=7,
            scheduled_days=14,
            reps=5,
            lapses=1,
            state="review",
            last_review=now,
            due=now + timedelta(days=14),
        )

        assert card.stability == 5.0
        assert card.difficulty == 3.0
        assert card.reps == 5
        assert card.state == "review"

    def test_card_to_dict(self):
        """Test card serialization"""
        card = FSRSCard(concept_id=1, user_id=1)
        card_dict = card.to_dict()

        assert "concept_id" in card_dict
        assert "stability" in card_dict
        assert "difficulty" in card_dict
        assert "state" in card_dict
        assert card_dict["concept_id"] == 1


class TestFSRSAlgorithm:
    """Tests for FSRSAlgorithm"""

    @pytest.fixture
    def fsrs(self):
        """Create FSRS algorithm instance"""
        return FSRSAlgorithm()

    @pytest.fixture
    def custom_fsrs(self):
        """Create FSRS with custom retention target"""
        params = FSRSAlgorithm.DEFAULT_PARAMS.copy()
        params["request_retention"] = 0.85
        return FSRSAlgorithm(params=params)

    # === Retrievability Tests ===

    def test_retrievability_at_zero_elapsed(self, fsrs):
        """Retrievability should be 1.0 when no time has elapsed"""
        retrievability = fsrs.calculate_retrievability(elapsed_days=0, stability=10.0)
        assert math.isclose(retrievability, 1.0, rel_tol=1e-5)

    def test_retrievability_at_stability(self, fsrs):
        """Retrievability should be 0.9 when elapsed equals stability"""
        # R = 0.9^(t/S) = 0.9^(10/10) = 0.9
        stability = 10.0
        elapsed = 10.0
        retrievability = fsrs.calculate_retrievability(elapsed, stability)
        assert math.isclose(retrievability, 0.9, rel_tol=1e-5)

    def test_retrievability_decreases_over_time(self, fsrs):
        """Retrievability should decrease as time passes"""
        stability = 10.0
        r1 = fsrs.calculate_retrievability(5, stability)
        r2 = fsrs.calculate_retrievability(10, stability)
        r3 = fsrs.calculate_retrievability(20, stability)

        assert r1 > r2 > r3
        assert r1 > 0.9  # Less than stability
        assert r3 < 0.9  # More than stability

    def test_retrievability_zero_stability(self, fsrs):
        """Retrievability should be 0 when stability is 0"""
        retrievability = fsrs.calculate_retrievability(elapsed_days=5, stability=0.0)
        assert retrievability == 0.0

    def test_retrievability_exponential_decay(self, fsrs):
        """Verify exponential decay formula"""
        stability = 5.0
        for elapsed in [1, 5, 10, 20]:
            expected = math.pow(0.9, elapsed / stability)
            actual = fsrs.calculate_retrievability(elapsed, stability)
            assert math.isclose(actual, expected, rel_tol=1e-5)

    # === Initial Stability Tests ===

    def test_init_stability_again(self, fsrs):
        """Initial stability for AGAIN rating"""
        stability = fsrs.init_stability(Rating.AGAIN)
        assert stability == fsrs.w[0]  # 0.4

    def test_init_stability_hard(self, fsrs):
        """Initial stability for HARD rating"""
        stability = fsrs.init_stability(Rating.HARD)
        assert stability == fsrs.w[1]  # 0.6

    def test_init_stability_good(self, fsrs):
        """Initial stability for GOOD rating"""
        stability = fsrs.init_stability(Rating.GOOD)
        assert stability == fsrs.w[2]  # 2.4

    def test_init_stability_easy(self, fsrs):
        """Initial stability for EASY rating"""
        stability = fsrs.init_stability(Rating.EASY)
        assert stability == fsrs.w[3]  # 5.8

    def test_init_stability_increases_with_rating(self, fsrs):
        """Initial stability should increase with better ratings"""
        s_again = fsrs.init_stability(Rating.AGAIN)
        s_hard = fsrs.init_stability(Rating.HARD)
        s_good = fsrs.init_stability(Rating.GOOD)
        s_easy = fsrs.init_stability(Rating.EASY)

        assert s_again < s_hard < s_good < s_easy

    # === Initial Difficulty Tests ===

    def test_init_difficulty_bounded(self, fsrs):
        """Initial difficulty should be bounded 1-10"""
        for rating in Rating:
            difficulty = fsrs.init_difficulty(rating)
            assert 1 <= difficulty <= 10

    def test_init_difficulty_again_highest(self, fsrs):
        """AGAIN rating should give highest difficulty"""
        d_again = fsrs.init_difficulty(Rating.AGAIN)
        d_easy = fsrs.init_difficulty(Rating.EASY)
        assert d_again > d_easy

    # === Next Stability Tests ===

    def test_next_stability_again_resets(self, fsrs):
        """AGAIN rating should reduce stability"""
        current_stability = 10.0
        difficulty = 5.0
        retrievability = 0.9

        new_stability = fsrs.next_stability(
            current_stability, difficulty, retrievability, Rating.AGAIN
        )

        assert new_stability < current_stability

    def test_next_stability_good_increases(self, fsrs):
        """GOOD rating should increase stability"""
        current_stability = 10.0
        difficulty = 5.0
        retrievability = 0.9

        new_stability = fsrs.next_stability(
            current_stability, difficulty, retrievability, Rating.GOOD
        )

        assert new_stability > current_stability

    def test_next_stability_easy_bonus(self, fsrs):
        """EASY rating should give larger stability increase than GOOD"""
        current_stability = 10.0
        difficulty = 5.0
        retrievability = 0.9

        s_good = fsrs.next_stability(
            current_stability, difficulty, retrievability, Rating.GOOD
        )
        s_easy = fsrs.next_stability(
            current_stability, difficulty, retrievability, Rating.EASY
        )

        assert s_easy > s_good

    def test_next_stability_minimum(self, fsrs):
        """Stability should never go below minimum (0.1)"""
        new_stability = fsrs.next_stability(
            current_stability=0.1,
            difficulty=10.0,
            retrievability=0.1,
            rating=Rating.AGAIN,
        )

        assert new_stability >= 0.1

    # === Next Difficulty Tests ===

    def test_next_difficulty_again_increases(self, fsrs):
        """AGAIN rating should increase difficulty"""
        current_difficulty = 5.0
        new_difficulty = fsrs.next_difficulty(current_difficulty, Rating.AGAIN)
        assert new_difficulty > current_difficulty

    def test_next_difficulty_easy_decreases(self, fsrs):
        """EASY rating should decrease difficulty"""
        current_difficulty = 5.0
        new_difficulty = fsrs.next_difficulty(current_difficulty, Rating.EASY)
        assert new_difficulty < current_difficulty

    def test_next_difficulty_bounded(self, fsrs):
        """Difficulty should stay bounded 1-10"""
        # Test lower bound
        low_diff = fsrs.next_difficulty(1.0, Rating.EASY)
        assert 1 <= low_diff <= 10

        # Test upper bound
        high_diff = fsrs.next_difficulty(10.0, Rating.AGAIN)
        assert 1 <= high_diff <= 10

    def test_next_difficulty_mean_reversion(self, fsrs):
        """Difficulty should have mean reversion"""
        # Very high difficulty should trend down even with GOOD
        high_diff = 9.5
        new_diff = fsrs.next_difficulty(high_diff, Rating.GOOD)
        # After many GOOD ratings, should trend toward mean
        assert new_diff < high_diff

    # === Next Interval Tests ===

    def test_next_interval_basic(self, fsrs):
        """Test basic interval calculation"""
        # I = S * ln(R_req) / ln(0.9)
        # With R_req = 0.9 and S = 10, I should be 10
        stability = 10.0
        interval = fsrs.next_interval(stability)
        assert interval == 10

    def test_next_interval_custom_retention(self, custom_fsrs):
        """Test interval with custom retention target"""
        stability = 10.0
        interval = custom_fsrs.next_interval(stability)
        # With R = 0.85, interval should be longer
        assert interval > 10

    def test_next_interval_minimum_one_day(self, fsrs):
        """Interval should be at least 1 day"""
        interval = fsrs.next_interval(stability=0.1)
        assert interval >= 1

    def test_next_interval_maximum_constrained(self, fsrs):
        """Interval should not exceed maximum"""
        interval = fsrs.next_interval(stability=100000)
        assert interval <= fsrs.params["maximum_interval"]

    # === Review Card Tests ===

    def test_review_new_card_good(self, fsrs):
        """Test reviewing a new card with GOOD rating"""
        card = FSRSCard(concept_id=1, user_id=1)
        assert card.state == "new"

        updated_card, log = fsrs.review_card(card, Rating.GOOD)

        assert updated_card.state == "review"
        assert updated_card.stability > 0
        assert updated_card.difficulty > 0
        assert updated_card.reps == 1
        assert updated_card.scheduled_days > 0

    def test_review_new_card_again(self, fsrs):
        """Test reviewing a new card with AGAIN rating"""
        card = FSRSCard(concept_id=1, user_id=1)

        updated_card, log = fsrs.review_card(card, Rating.AGAIN)

        assert updated_card.state == "learning"
        assert updated_card.stability > 0
        assert updated_card.reps == 1

    def test_review_card_increments_reps(self, fsrs):
        """Each review should increment reps"""
        card = FSRSCard(concept_id=1, user_id=1, reps=5, state="review", stability=5.0)

        updated_card, _ = fsrs.review_card(card, Rating.GOOD)

        assert updated_card.reps == 6

    def test_review_card_again_increments_lapses(self, fsrs):
        """AGAIN on review card should increment lapses"""
        card = FSRSCard(
            concept_id=1, user_id=1, reps=5, lapses=0, state="review", stability=5.0, difficulty=5.0
        )

        updated_card, _ = fsrs.review_card(card, Rating.AGAIN)

        assert updated_card.lapses == 1
        assert updated_card.state == "relearning"

    def test_review_card_updates_due_date(self, fsrs):
        """Review should update due date"""
        card = FSRSCard(concept_id=1, user_id=1, state="review", stability=5.0)
        review_time = datetime.now()

        updated_card, _ = fsrs.review_card(card, Rating.GOOD, review_time=review_time)

        assert updated_card.due > review_time
        assert updated_card.last_review == review_time

    def test_review_log_contains_required_fields(self, fsrs):
        """Review log should contain all required fields"""
        card = FSRSCard(concept_id=1, user_id=1)

        _, log = fsrs.review_card(card, Rating.GOOD)

        assert "card_id" in log
        assert "rating" in log
        assert "review_time" in log
        assert "elapsed_days" in log
        assert "scheduled_days" in log
        assert "stability" in log
        assert "difficulty" in log
        assert "state" in log
        assert "reps" in log

    # === Get Next States Tests ===

    def test_get_next_states_returns_all_ratings(self, fsrs):
        """get_next_states should return predictions for all ratings"""
        card = FSRSCard(concept_id=1, user_id=1, state="review", stability=5.0, difficulty=5.0)

        predictions = fsrs.get_next_states(card)

        assert Rating.AGAIN in predictions
        assert Rating.HARD in predictions
        assert Rating.GOOD in predictions
        assert Rating.EASY in predictions

    def test_get_next_states_intervals_increase(self, fsrs):
        """Higher ratings should give longer intervals"""
        card = FSRSCard(
            concept_id=1, user_id=1, state="review", stability=5.0, difficulty=5.0
        )

        predictions = fsrs.get_next_states(card)

        assert predictions[Rating.AGAIN]["interval"] < predictions[Rating.GOOD]["interval"]
        assert predictions[Rating.GOOD]["interval"] <= predictions[Rating.EASY]["interval"]

    def test_get_next_states_doesnt_modify_card(self, fsrs):
        """get_next_states should not modify the original card"""
        card = FSRSCard(
            concept_id=1, user_id=1, state="review", stability=5.0, difficulty=5.0, reps=3
        )
        original_stability = card.stability
        original_reps = card.reps

        _ = fsrs.get_next_states(card)

        assert card.stability == original_stability
        assert card.reps == original_reps


class TestFSRSIntegration:
    """Integration tests for FSRS learning scenarios"""

    @pytest.fixture
    def fsrs(self):
        return FSRSAlgorithm()

    def test_learning_session_simulation(self, fsrs):
        """Simulate a realistic learning session"""
        card = FSRSCard(concept_id=1, user_id=1)

        # First review - GOOD
        card, _ = fsrs.review_card(card, Rating.GOOD)
        first_interval = card.scheduled_days
        assert card.state == "review"

        # Second review after scheduled interval - GOOD
        review_time = card.due
        card, _ = fsrs.review_card(card, Rating.GOOD, review_time)
        second_interval = card.scheduled_days

        # Intervals should increase with successful reviews
        assert second_interval > first_interval

    def test_forgetting_and_relearning(self, fsrs):
        """Test forgetting a card and relearning"""
        # Initial learning
        card = FSRSCard(concept_id=1, user_id=1)
        card, _ = fsrs.review_card(card, Rating.GOOD)
        card, _ = fsrs.review_card(card, Rating.GOOD)
        stability_before_forget = card.stability

        # Forget the card
        card, _ = fsrs.review_card(card, Rating.AGAIN)

        assert card.state == "relearning"
        assert card.stability < stability_before_forget
        assert card.lapses == 1

        # Relearn
        card, _ = fsrs.review_card(card, Rating.GOOD)
        assert card.state == "review"

    def test_difficulty_adjustment_over_time(self, fsrs):
        """Test that difficulty adjusts based on performance"""
        card = FSRSCard(concept_id=1, user_id=1)

        # Start with GOOD
        card, _ = fsrs.review_card(card, Rating.GOOD)
        initial_difficulty = card.difficulty

        # Multiple EASY ratings should decrease difficulty
        for _ in range(3):
            card, _ = fsrs.review_card(card, Rating.EASY)

        assert card.difficulty < initial_difficulty
