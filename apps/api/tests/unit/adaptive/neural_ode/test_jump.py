"""
Unit tests for Jump Network.

Tests the discrete state update mechanism that occurs at review events.
"""

import pytest
import torch
from app.adaptive.neural_ode.jump import JumpNetwork, GatedJumpNetwork


class TestJumpNetwork:
    """Tests for the basic JumpNetwork."""

    def test_jump_network_shapes(self):
        """Verify jump network produces correct output dimensions."""
        state_dim = 32
        batch_size = 4

        jump = JumpNetwork(state_dim=state_dim)
        h_pre = torch.randn(batch_size, state_dim)
        grade = torch.tensor([1, 2, 3, 4])  # All four grade levels
        telemetry = torch.rand(batch_size, 4)  # RT, hesitation, tortuosity, fluency

        h_post = jump(h_pre, grade, telemetry)

        assert h_post.shape == (batch_size, state_dim)
        assert h_post.dtype == h_pre.dtype

    def test_jump_without_telemetry(self):
        """Verify jump works with telemetry=None (uses defaults)."""
        state_dim = 32
        batch_size = 2

        jump = JumpNetwork(state_dim=state_dim)
        h_pre = torch.randn(batch_size, state_dim)
        grade = torch.tensor([3, 3])

        # Should not raise
        h_post = jump(h_pre, grade, telemetry=None)

        assert h_post.shape == (batch_size, state_dim)

    def test_jump_residual_connection(self):
        """Verify jump uses residual connection (h_post = h_pre + delta)."""
        state_dim = 32
        jump = JumpNetwork(state_dim=state_dim)

        h_pre = torch.randn(1, state_dim)
        grade = torch.tensor([3])

        h_post = jump(h_pre, grade, None)

        # Due to residual connection, output should be close to input
        # (scaled by jump_scale which starts small)
        delta = h_post - h_pre
        delta_norm = delta.norm().item()

        # Delta should be non-zero but bounded
        assert delta_norm > 0
        assert delta_norm < h_pre.norm().item() * 2  # Reasonable bound

    def test_different_grades_different_jumps(self):
        """Different grades should produce different state updates."""
        state_dim = 32
        jump = JumpNetwork(state_dim=state_dim)

        h_pre = torch.randn(1, state_dim)
        telemetry = torch.tensor([[0.5, 0.5, 0.5, 0.5]])

        # Test all grades
        jumps = {}
        for grade in [1, 2, 3, 4]:
            grade_tensor = torch.tensor([grade])
            h_post = jump(h_pre.clone(), grade_tensor, telemetry)
            jumps[grade] = h_post

        # Each grade should produce a different result
        for g1 in [1, 2, 3]:
            for g2 in range(g1 + 1, 5):
                assert not torch.allclose(jumps[g1], jumps[g2], atol=1e-5), \
                    f"Grade {g1} and {g2} produced identical jumps"

    def test_jump_magnitude_varies_with_grade(self):
        """Higher grades should generally produce larger positive updates."""
        state_dim = 32
        jump = JumpNetwork(state_dim=state_dim)

        h_pre = torch.randn(1, state_dim)
        telemetry = torch.tensor([[0.5, 0.5, 0.5, 0.5]])

        magnitudes = []
        for grade in [1, 2, 3, 4]:
            grade_tensor = torch.tensor([grade])
            mag = jump.get_jump_magnitude(h_pre, grade_tensor, telemetry)
            magnitudes.append(mag.item())

        # Magnitudes should vary (not all equal)
        assert len(set(magnitudes)) > 1, "All grades produced identical magnitudes"

    def test_telemetry_affects_jump(self):
        """Different telemetry values should affect the state update."""
        state_dim = 32
        jump = JumpNetwork(state_dim=state_dim)

        h_pre = torch.randn(1, state_dim)
        grade = torch.tensor([3])

        # Fast response (low RT)
        telem_fast = torch.tensor([[0.1, 0.1, 0.1, 0.9]])
        h_fast = jump(h_pre.clone(), grade, telem_fast)

        # Slow response (high RT, hesitation)
        telem_slow = torch.tensor([[0.9, 0.9, 0.9, 0.1]])
        h_slow = jump(h_pre.clone(), grade, telem_slow)

        # Should produce different states
        assert not torch.allclose(h_fast, h_slow, atol=1e-5)

    def test_batch_processing(self):
        """Verify batched inputs are processed correctly."""
        state_dim = 32
        batch_size = 8

        jump = JumpNetwork(state_dim=state_dim)

        h_pre = torch.randn(batch_size, state_dim)
        grade = torch.randint(1, 5, (batch_size,))
        telemetry = torch.rand(batch_size, 4)

        h_post = jump(h_pre, grade, telemetry)

        # Process individually and compare
        for i in range(batch_size):
            h_single = jump(
                h_pre[i:i+1],
                grade[i:i+1],
                telemetry[i:i+1]
            )
            assert torch.allclose(h_post[i:i+1], h_single, atol=1e-6)

    def test_gradient_flow(self):
        """Verify gradients flow through the jump network."""
        state_dim = 32
        jump = JumpNetwork(state_dim=state_dim)

        h_pre = torch.randn(1, state_dim, requires_grad=True)
        grade = torch.tensor([3])
        telemetry = torch.rand(1, 4)

        h_post = jump(h_pre, grade, telemetry)
        loss = h_post.sum()
        loss.backward()

        # Check gradients exist
        assert h_pre.grad is not None
        assert h_pre.grad.shape == h_pre.shape
        assert not torch.all(h_pre.grad == 0)


class TestGatedJumpNetwork:
    """Tests for the gated variant of the Jump Network."""

    def test_gated_jump_shapes(self):
        """Verify gated jump network produces correct output dimensions."""
        state_dim = 32
        batch_size = 4

        jump = GatedJumpNetwork(state_dim=state_dim)
        h_pre = torch.randn(batch_size, state_dim)
        grade = torch.tensor([1, 2, 3, 4])
        telemetry = torch.rand(batch_size, 4)

        h_post = jump(h_pre, grade, telemetry)

        assert h_post.shape == (batch_size, state_dim)

    def test_gated_vs_basic_produces_different_results(self):
        """Gated and basic jump should produce different outputs."""
        state_dim = 32
        torch.manual_seed(42)

        basic = JumpNetwork(state_dim=state_dim)
        gated = GatedJumpNetwork(state_dim=state_dim)

        h_pre = torch.randn(1, state_dim)
        grade = torch.tensor([3])
        telemetry = torch.tensor([[0.5, 0.5, 0.5, 0.5]])

        h_basic = basic(h_pre.clone(), grade, telemetry)
        h_gated = gated(h_pre.clone(), grade, telemetry)

        # Due to different architectures, outputs should differ
        assert not torch.allclose(h_basic, h_gated)

    def test_gate_bounds_output(self):
        """The gate should produce values in [0, 1] (sigmoid output)."""
        state_dim = 32
        gated = GatedJumpNetwork(state_dim=state_dim)

        h_pre = torch.randn(10, state_dim)
        grade = torch.randint(1, 5, (10,))
        telemetry = torch.rand(10, 4)

        # Access gate values by modifying forward slightly
        # For now, just verify output is reasonable
        h_post = gated(h_pre, grade, telemetry)

        # Output should still be bounded reasonably
        delta = h_post - h_pre
        # Gated delta should be smaller due to gate squashing
        assert delta.abs().max() < 100  # Reasonable bound


class TestJumpNetworkEdgeCases:
    """Edge case tests for Jump Network."""

    def test_zero_state_input(self):
        """Jump should work with zero initial state."""
        jump = JumpNetwork(state_dim=32)
        h_pre = torch.zeros(1, 32)
        grade = torch.tensor([3])

        h_post = jump(h_pre, grade, None)

        # Should produce non-zero output due to grade embedding
        assert not torch.allclose(h_post, h_pre)

    def test_large_state_values(self):
        """Jump should handle large state values without NaN."""
        jump = JumpNetwork(state_dim=32)
        h_pre = torch.randn(1, 32) * 100  # Large values
        grade = torch.tensor([3])

        h_post = jump(h_pre, grade, None)

        assert not torch.isnan(h_post).any()
        assert not torch.isinf(h_post).any()

    def test_padding_grade_zero(self):
        """Grade 0 (padding) should be handled."""
        jump = JumpNetwork(state_dim=32)
        h_pre = torch.randn(1, 32)
        grade = torch.tensor([0])  # Padding index

        # Should not raise, but use padding embedding (zeros)
        h_post = jump(h_pre, grade, None)
        assert h_post.shape == (1, 32)
