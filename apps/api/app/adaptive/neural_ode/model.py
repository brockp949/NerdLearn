import torch
import torch.nn as nn
import numpy as np
from .solver import odeint

class MemoryDerivative(nn.Module):
    def __init__(self, state_dim, hidden_dim, chronotype_phase=0.0):
        super().__init__()
        # Base decay network: h -> dh/dt (Negative to represent decay naturally, but net can learn sign)
        # We use a simple MLP
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Consolidation network (Sleep dynamics)
        self.sleep_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Circadian parameters
        # A: amplitude, phi: phase
        self.circadian_amp = nn.Parameter(torch.tensor(0.1))
        self.circadian_phase = nn.Parameter(torch.tensor(chronotype_phase))
        self.circadian_omega = 2 * np.pi / 24.0 # Period 24h
        
        # Stress parameter
        self.stress_sensitivity = nn.Parameter(torch.tensor(0.5))
        
        # Context (set before forward pass)
        self.sleep_schedule_fn = None
        self.stress_level = 0.0

    def set_context(self, sleep_schedule_fn, stress_level=0.0):
        self.sleep_schedule_fn = sleep_schedule_fn
        self.stress_level = stress_level

    def forward(self, t, h):
        """
        Computes dh/dt given time t and state h.
        """
        # 1. Determine Sleep State
        is_sleeping = False
        if self.sleep_schedule_fn is not None:
            # Assuming t is a tensor scalar
            is_sleeping = self.sleep_schedule_fn(float(t))
        
        if is_sleeping:
            # Sleep Regime: Consolidation
            # Dynamics driven by sleep_net (could essentially be 'anti-decay' or restructuring)
            dh_dt = self.sleep_net(h)
        else:
            # Wake Regime: Decay modulated by Circadian Rhythm and Stress
            dh_dt = self.net(h)
            
            # Circadian Modulation: M(t) = 1 + A * cos(omega * t + phi)
            circadian_mod = 1.0 + self.circadian_amp * torch.cos(self.circadian_omega * t + self.circadian_phase)
            dh_dt = dh_dt * circadian_mod
            
            # Stress Modulation: Increases decay rate (assuming negative dh_dt)
            # If dh_dt is negative (decay), multipying by (1 + stress) makes it more negative (faster decay)
            stress_mod = 1.0 + self.stress_sensitivity * self.stress_level
            dh_dt = dh_dt * stress_mod
            
        return dh_dt

class NeuralMemoryODE(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=16):
        super().__init__()
        self.ode_func = MemoryDerivative(state_dim, hidden_dim)
        

        # Readout head: Projects state h to log-odds of recall (logit)
        self.readout = nn.Linear(state_dim, 1)

        # Implicit Telemetry Heads (Section 4.1)
        # Latency (RT) = alpha / |h_retrieval| + beta
        # We model this by predicting the parameters or directly mapping h to RT.
        # Here we map h -> RT using a positive activation (e.g. Softplus)
        self.latency_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus() # RT must be positive
        )
        
        # Hesitation Head (e.g. Tortuosity, Uncertainty)
        self.hesitation_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Normalized [0,1] or Softplus if unbounded
        )
        
    def forward(self, h0, t_eval, sleep_schedule_fn=None, stress_level=0.0):
        """
        Simulates memory trajectory.
        
        Args:
            h0: Initial state (batch_size, state_dim)
            t_eval: Time points to evaluate (num_steps,)
            sleep_schedule_fn: Function taking time t and returning bool (True if sleeping)
            stress_level: Scalar float representing stress (0.0 to 1.0+)
            
        Returns:
            trajectory: (num_steps, batch_size, state_dim)
        """
        self.ode_func.set_context(sleep_schedule_fn, stress_level)
        
        # Integrate
        # Note: odeint returns states at t_eval points
        trajectory = odeint(self.ode_func, h0, t_eval)
        
        return trajectory
    
    def predict_all(self, h0, t_eval, sleep_schedule_fn=None, stress_level=0.0):
        """
        Predicts recall probability AND telemetry signals at t_eval points.
        Returns dictionary of tensors.
        """
        trajectory = self.forward(h0, t_eval, sleep_schedule_fn, stress_level) # (T, B, D)
        
        # Flatten time and batch dims for head processing if needed, but Linear works on last dim
        logits = self.readout(trajectory)
        probs = torch.sigmoid(logits)
        
        latency = self.latency_head(trajectory)
        hesitation = self.hesitation_head(trajectory)
        
        return {
            "probs": probs,      # (T, B, 1)
            "latency": latency,  # (T, B, 1)
            "hesitation": hesitation # (T, B, 1)
        }

    def predict_retention(self, h0, t_eval, sleep_schedule_fn=None, stress_level=0.0):
        """
        Predicts probability of recall at t_eval points.
        Returns: (num_steps, batch_size, 1) values in [0, 1]
        """
        trajectory = self.forward(h0, t_eval, sleep_schedule_fn, stress_level)
        return torch.sigmoid(self.readout(trajectory))
