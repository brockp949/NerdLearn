"""
Unit tests for State Encoder.

Tests the initial state computation and cold-start functionality.
"""

import pytest
import torch
from app.adaptive.neural_ode.encoder import (
    StateEncoder,
    HierarchicalStateEncoder,
    PHENOTYPE_MAP,
    PHENOTYPE_CHARACTERISTICS,
)


class TestStateEncoder:
    """Tests for the basic StateEncoder."""

    def test_encoder_shapes(self):
        """Verify encoder produces correct output dimensions."""
        card_feat_dim = 64
        user_feat_dim = 16
        state_dim = 32
        batch_size = 4

        encoder = StateEncoder(
            card_feat_dim=card_feat_dim,
            user_feat_dim=user_feat_dim,
            state_dim=state_dim,
        )

        card_features = torch.randn(batch_size, card_feat_dim)
        user_features = torch.randn(batch_size, user_feat_dim)

        h0 = encoder(card_features, user_features)

        assert h0.shape == (batch_size, state_dim)
        assert h0.dtype == card_features.dtype

    def test_encoder_with_phenotype_only(self):
        """Verify encoder works with phenotype ID instead of user features."""
        encoder = StateEncoder(card_feat_dim=64, state_dim=32)

        card_features = torch.randn(2, 64)
        phenotype_id = torch.tensor([1, 3])  # fast_forgetter, cramper

        h0 = encoder(card_features, phenotype_id=phenotype_id)

        assert h0.shape == (2, 32)

    def test_encoder_with_neither_defaults_to_unknown(self):
        """Encoder should use unknown phenotype when no user info provided."""
        encoder = StateEncoder(card_feat_dim=64, state_dim=32)

        card_features = torch.randn(2, 64)

        # Neither user_features nor phenotype_id
        h0 = encoder(card_features)

        assert h0.shape == (2, 32)

    def test_cold_start_init_by_name(self):
        """Verify cold_start_init accepts phenotype names."""
        encoder = StateEncoder(card_feat_dim=64, state_dim=32)
        card_features = torch.randn(1, 64)

        # Test each phenotype
        for name in PHENOTYPE_MAP.keys():
            h0 = encoder.cold_start_init(card_features, phenotype=name)
            assert h0.shape == (1, 32)
            assert not torch.isnan(h0).any()

    def test_cold_start_init_by_id(self):
        """Verify cold_start_init accepts phenotype IDs."""
        encoder = StateEncoder(card_feat_dim=64, state_dim=32)
        card_features = torch.randn(1, 64)

        for pid in range(8):
            h0 = encoder.cold_start_init(card_features, phenotype=pid)
            assert h0.shape == (1, 32)

    def test_different_phenotypes_different_states(self):
        """Different phenotypes should produce distinguishable initial states."""
        encoder = StateEncoder(card_feat_dim=64, state_dim=32)

        # Use same card features
        card_features = torch.randn(1, 64)

        states = {}
        for name, pid in PHENOTYPE_MAP.items():
            phenotype_id = torch.tensor([pid])
            h0 = encoder(card_features, phenotype_id=phenotype_id)
            states[name] = h0

        # Check that different phenotypes produce different states
        phenotype_names = list(PHENOTYPE_MAP.keys())
        for i, name1 in enumerate(phenotype_names[:-1]):
            for name2 in phenotype_names[i+1:]:
                if name1 != 'unknown' and name2 != 'unknown':
                    # Different phenotypes should give different states
                    assert not torch.allclose(states[name1], states[name2], atol=1e-4), \
                        f"Phenotypes {name1} and {name2} produced identical states"

    def test_difficulty_scaling(self):
        """Verify difficulty parameter affects initial state."""
        encoder = StateEncoder(card_feat_dim=64, state_dim=32)
        card_features = torch.randn(1, 64)

        # Easy card
        h0_easy = encoder(card_features, difficulty=torch.tensor([[0.2]]))

        # Hard card
        h0_hard = encoder(card_features, difficulty=torch.tensor([[0.9]]))

        # Should produce different states
        assert not torch.allclose(h0_easy, h0_hard)

    def test_same_input_same_output(self):
        """Deterministic: same inputs should produce same outputs."""
        encoder = StateEncoder(card_feat_dim=64, state_dim=32)
        encoder.eval()

        card_features = torch.randn(1, 64)
        phenotype_id = torch.tensor([2])

        h0_1 = encoder(card_features, phenotype_id=phenotype_id)
        h0_2 = encoder(card_features, phenotype_id=phenotype_id)

        assert torch.allclose(h0_1, h0_2)

    def test_different_cards_different_states(self):
        """Different card features should produce different states."""
        encoder = StateEncoder(card_feat_dim=64, state_dim=32)

        card1 = torch.randn(1, 64)
        card2 = torch.randn(1, 64)  # Different card

        phenotype_id = torch.tensor([2])

        h0_1 = encoder(card1, phenotype_id=phenotype_id)
        h0_2 = encoder(card2, phenotype_id=phenotype_id)

        assert not torch.allclose(h0_1, h0_2)

    def test_batch_consistency(self):
        """Batched processing should match individual processing."""
        encoder = StateEncoder(card_feat_dim=64, state_dim=32)
        encoder.eval()

        batch_size = 4
        card_features = torch.randn(batch_size, 64)
        phenotype_ids = torch.tensor([1, 2, 3, 4])

        # Batched
        h0_batch = encoder(card_features, phenotype_id=phenotype_ids)

        # Individual
        for i in range(batch_size):
            h0_single = encoder(
                card_features[i:i+1],
                phenotype_id=phenotype_ids[i:i+1]
            )
            assert torch.allclose(h0_batch[i:i+1], h0_single, atol=1e-6)

    def test_gradient_flow(self):
        """Verify gradients flow through the encoder."""
        encoder = StateEncoder(card_feat_dim=64, state_dim=32)

        card_features = torch.randn(1, 64, requires_grad=True)
        user_features = torch.randn(1, 16, requires_grad=True)

        h0 = encoder(card_features, user_features)
        loss = h0.sum()
        loss.backward()

        assert card_features.grad is not None
        assert user_features.grad is not None

    def test_get_phenotype_embedding(self):
        """Verify phenotype embedding retrieval."""
        encoder = StateEncoder(card_feat_dim=64, state_dim=32)

        # By name
        emb_name = encoder.get_phenotype_embedding('fast_forgetter')
        assert emb_name.dim() == 1

        # By ID
        emb_id = encoder.get_phenotype_embedding(1)
        assert torch.allclose(emb_name, emb_id)


class TestHierarchicalStateEncoder:
    """Tests for the hierarchical encoder variant."""

    def test_hierarchical_encoder_shapes(self):
        """Verify hierarchical encoder produces correct dimensions."""
        low_dim = 8
        mid_dim = 48
        high_dim = 8
        state_dim = 32

        encoder = HierarchicalStateEncoder(
            low_level_dim=low_dim,
            mid_level_dim=mid_dim,
            high_level_dim=high_dim,
            state_dim=state_dim,
        )

        # Total card features = low + mid + high = 64
        card_features = torch.randn(2, 64)
        phenotype_id = torch.tensor([1, 2])

        h0 = encoder(card_features, phenotype_id=phenotype_id)

        assert h0.shape == (2, state_dim)

    def test_hierarchical_vs_basic_different_outputs(self):
        """Hierarchical and basic encoders should behave differently."""
        torch.manual_seed(42)

        basic = StateEncoder(card_feat_dim=64, state_dim=32)
        hierarchical = HierarchicalStateEncoder(
            low_level_dim=8,
            mid_level_dim=48,
            high_level_dim=8,
            state_dim=32,
        )

        card_features = torch.randn(1, 64)
        phenotype_id = torch.tensor([2])

        h0_basic = basic(card_features, phenotype_id=phenotype_id)
        h0_hier = hierarchical(card_features, phenotype_id=phenotype_id)

        # Different architectures should produce different outputs
        assert not torch.allclose(h0_basic, h0_hier)


class TestPhenotypeInitialization:
    """Tests for phenotype embedding initialization."""

    def test_phenotype_map_completeness(self):
        """Verify all expected phenotypes are defined."""
        expected = {
            'unknown', 'fast_forgetter', 'steady_learner', 'cramper',
            'deep_processor', 'night_owl', 'morning_lark', 'variable'
        }
        assert set(PHENOTYPE_MAP.keys()) == expected

    def test_phenotype_characteristics_match_map(self):
        """Verify characteristics exist for all non-unknown phenotypes."""
        for name in PHENOTYPE_MAP:
            if name != 'unknown':
                assert name in PHENOTYPE_CHARACTERISTICS

    def test_phenotype_ids_are_contiguous(self):
        """Phenotype IDs should be 0-7."""
        ids = sorted(PHENOTYPE_MAP.values())
        assert ids == list(range(8))

    def test_phenotype_embeddings_initialized(self):
        """Verify phenotype embeddings are initialized based on characteristics."""
        encoder = StateEncoder(card_feat_dim=64, state_dim=32)

        for name, pid in PHENOTYPE_MAP.items():
            emb = encoder.phenotype_embed.weight[pid]
            if name == 'unknown':
                # Unknown should be zeros
                assert torch.allclose(emb, torch.zeros_like(emb))
            elif name == 'variable':
                # Variable is intentionally neutral (all 0.5 centered to 0)
                assert torch.allclose(emb, torch.zeros_like(emb))
            else:
                # Others should be non-zero (initialized based on characteristics)
                assert not torch.allclose(emb, torch.zeros_like(emb))


class TestEncoderEdgeCases:
    """Edge case tests for State Encoder."""

    def test_zero_card_features(self):
        """Encoder should handle zero card features."""
        encoder = StateEncoder(card_feat_dim=64, state_dim=32)
        card_features = torch.zeros(1, 64)
        phenotype_id = torch.tensor([2])

        h0 = encoder(card_features, phenotype_id=phenotype_id)

        assert h0.shape == (1, 32)
        assert not torch.isnan(h0).any()

    def test_large_card_features(self):
        """Encoder should handle large feature values."""
        encoder = StateEncoder(card_feat_dim=64, state_dim=32)
        card_features = torch.randn(1, 64) * 100
        phenotype_id = torch.tensor([2])

        h0 = encoder(card_features, phenotype_id=phenotype_id)

        assert not torch.isnan(h0).any()
        assert not torch.isinf(h0).any()

    def test_unknown_phenotype_string(self):
        """Unknown phenotype names should map to ID 0."""
        encoder = StateEncoder(card_feat_dim=64, state_dim=32)
        card_features = torch.randn(1, 64)

        # Invalid phenotype name
        h0 = encoder.cold_start_init(card_features, phenotype='invalid_name')

        # Should use unknown (0) and still work
        assert h0.shape == (1, 32)
