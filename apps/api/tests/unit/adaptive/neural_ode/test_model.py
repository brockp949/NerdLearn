
import pytest
import torch
import numpy as np
from app.adaptive.neural_ode.model import NeuralMemoryODE
from app.adaptive.neural_ode.scheduler import NeuralODEScheduler

def test_memory_ode_shapes():
    """Verify input/output shapes of the model."""
    model = NeuralMemoryODE(state_dim=4)
    h0 = torch.randn(2, 4) # Batch size 2
    t = torch.linspace(0, 10, 11) # 11 time steps
    
    # Predict
    out = model.predict_retention(h0, t)
    
    # Expected: (num_steps, batch_size, 1)
    assert out.shape == (11, 2, 1)
    assert (out >= 0).all() and (out <= 1).all()

def test_sleep_mode_activation():
    """Verify that specific sleep schedule function alters dynamics."""
    model = NeuralMemoryODE(state_dim=4)
    h0 = torch.randn(1, 4)
    t = torch.tensor([0.0, 1.0])
    
    # 1. Capture dynamics in Wake
    model.ode_func.set_context(sleep_schedule_fn=lambda t: False)
    dh_dt_wake = model.ode_func(0.0, h0)
    
    # 2. Capture dynamics in Sleep
    model.ode_func.set_context(sleep_schedule_fn=lambda t: True)
    dh_dt_sleep = model.ode_func(0.0, h0)
    
    # 3. Ensure they utilize different networks/params
    # By default random init should produce different gradients
    assert not torch.allclose(dh_dt_wake, dh_dt_sleep)

def test_stress_modulation():
    """Verify that higher stress levels alter the derivative."""
    model = NeuralMemoryODE(state_dim=4)
    h0 = torch.randn(1, 4)
    
    # Wake mode required for stress to apply
    model.ode_func.set_context(sleep_schedule_fn=lambda t: False, stress_level=0.0)
    dh_dt_calm = model.ode_func(0.0, h0)
    
    model.ode_func.set_context(sleep_schedule_fn=lambda t: False, stress_level=1.0)
    dh_dt_stressed = model.ode_func(0.0, h0)
    
    # Stress modulates the magnitude of the derivative
    # dh_dt_stressed should be dh_dt_calm * (1 + stress_sensitivity)
    ratio = dh_dt_stressed / (dh_dt_calm + 1e-8)
    
    # Since it's element-wise multiplication by scalar, ratio should be constant approx (1 + 0.5) = 1.5
    # (assuming dh_dt_calm is not zero)
    expected_ratio = 1.0 + model.ode_func.stress_sensitivity.item() * 1.0
    
    assert torch.allclose(dh_dt_stressed, dh_dt_calm * expected_ratio, atol=1e-5)

def test_scheduler_output():
    """Verify scheduler computes intervals correctly."""
    scheduler = NeuralODEScheduler(state_dim=4)
    h0 = torch.randn(3, 4) # batch of 3
    
    intervals = scheduler.schedule_review(h0, current_time=0.0, horizon_days=5)
    
    assert len(intervals) == 3
    for interval in intervals:
        assert isinstance(interval, float)
        assert 0 <= interval <= 5 * 24

def test_circadian_modulation():
    """Verify that derivative oscillates over 24h period."""
    model = NeuralMemoryODE(state_dim=4)
    h0 = torch.ones(1, 4) # Use ones to clearly see modulation
    
    # We check the modulation factor directly or its effect
    # M(t) = 1 + A * cos(omega*t + phi)
    
    t0 = 0.0
    t_half = 12.0 # Half day
    
    # Force wake
    model.ode_func.set_context(sleep_schedule_fn=lambda t: False)
    
    # Get base derivative without modulation at t=0 (cos(0)=1 -> max boost)
    # vs t=12 (cos(pi)=-1 -> max damp) assuming phase=0
    
    # Set phase to 0 for predictability
    model.ode_func.circadian_phase.data.fill_(0.0)
    
    dy_0 = model.ode_func(torch.tensor(t0), h0)
    dy_12 = model.ode_func(torch.tensor(t_half), h0)
    
    # If parameters base net output is roughly constant for small h change constraint, 
    # we expect dy_0 != dy_12 due to modulation
    assert not torch.allclose(dy_0, dy_12)
