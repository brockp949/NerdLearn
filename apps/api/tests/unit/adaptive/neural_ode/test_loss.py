
import pytest
import torch
from app.adaptive.neural_ode.model import NeuralMemoryODE
from app.adaptive.neural_ode.loss import ImplicitTelemetryLoss

def test_telemetry_predictions():
    """Verify model outputs telemetry signals."""
    model = NeuralMemoryODE(state_dim=4)
    h0 = torch.randn(2, 4)
    t = torch.linspace(0, 10, 11)
    
    out = model.predict_all(h0, t)
    
    assert "probs" in out
    assert "latency" in out
    assert "hesitation" in out
    
    # Latency should be positive (Softplus)
    assert (out["latency"] >= 0).all()
    
    # Hesitation should be in [0, 1] (Sigmoid)
    assert (out["hesitation"] >= 0).all() and (out["hesitation"] <= 1).all()
    
    # Shapes
    assert out["probs"].shape == (11, 2, 1)

def test_loss_computation():
    """Verify loss function computes gradients."""
    loss_fn = ImplicitTelemetryLoss()
    
    # Fake predictions (requires gradients)
    preds = {
        "probs": torch.rand(5, 1, 1, requires_grad=True),
        "latency": torch.rand(5, 1, 1, requires_grad=True),
        "hesitation": torch.rand(5, 1, 1, requires_grad=True)
    }
    
    # Fake targets
    targets = {
        "recall": torch.ones(5, 1, 1),
        "latency": torch.ones(5, 1, 1) * 0.5,
        "hesitation": torch.zeros(5, 1, 1)
    }
    
    loss, components = loss_fn(preds, targets)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0
    
    # Check backprop
    loss.backward()
    assert preds["probs"].grad is not None
    assert preds["latency"].grad is not None
