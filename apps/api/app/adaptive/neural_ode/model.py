import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from .solver import odeint
from .jump import JumpNetwork
from .encoder import StateEncoder

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
    """
    Complete Neural ODE model for memory dynamics.

    Combines:
    - StateEncoder: Maps card/user features to initial state h₀
    - MemoryDerivative: Continuous-time drift dynamics (decay)
    - JumpNetwork: Discrete state updates at review events
    - Readout heads: Predict recall probability and telemetry

    The model processes a sequence of reviews as:
    1. h₀ = Encoder(card_features, user_features)
    2. For each review at time t_i:
       a. Evolve: h(t_i⁻) = ODE_integrate(h(t_{i-1}⁺), t_{i-1}, t_i)
       b. Jump: h(t_i⁺) = h(t_i⁻) + Jump(h(t_i⁻), grade, telemetry)
    3. Readout: P(recall) = σ(Readout(h))
    """

    def __init__(
        self,
        state_dim: int = 32,
        hidden_dim: int = 64,
        card_feat_dim: int = 64,
        user_feat_dim: int = 16,
        use_gated_jump: bool = False,
    ):
        """
        Initialize the Neural Memory ODE model.

        Args:
            state_dim: Dimension of the latent memory state
            hidden_dim: Hidden layer dimension for networks
            card_feat_dim: Dimension of card feature vector
            user_feat_dim: Dimension of user feature vector
            use_gated_jump: Whether to use gated jump network
        """
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # State Encoder: card/user features → h₀
        self.encoder = StateEncoder(
            card_feat_dim=card_feat_dim,
            user_feat_dim=user_feat_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
        )

        # Drift Network: continuous decay dynamics
        self.ode_func = MemoryDerivative(state_dim, hidden_dim)

        # Jump Network: discrete updates at reviews
        if use_gated_jump:
            from .jump import GatedJumpNetwork
            self.jump_net = GatedJumpNetwork(state_dim=state_dim)
        else:
            self.jump_net = JumpNetwork(state_dim=state_dim)

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

    def encode_initial_state(
        self,
        card_features: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
        phenotype_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode card/user features into initial memory state h₀.

        Args:
            card_features: Card feature vector [batch, card_feat_dim]
            user_features: User statistics [batch, user_feat_dim] or None
            phenotype_id: Learner phenotype ID [batch] for cold-start

        Returns:
            h0: Initial memory state [batch, state_dim]
        """
        return self.encoder(card_features, user_features, phenotype_id)

    def apply_jump(
        self,
        h: torch.Tensor,
        grade: torch.Tensor,
        telemetry: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply discrete state update at a review event.

        Args:
            h: Pre-review state [batch, state_dim]
            grade: Review grade 1-4 [batch]
            telemetry: Behavioral signals [batch, 4] or None

        Returns:
            h_post: Post-review state [batch, state_dim]
        """
        return self.jump_net(h, grade, telemetry)

    def _integrate(
        self,
        h: torch.Tensor,
        t_start: float,
        t_end: float,
        n_steps: int = 10,
        sleep_schedule_fn: Optional[callable] = None,
        stress_level: float = 0.0,
    ) -> torch.Tensor:
        """
        Integrate ODE from t_start to t_end.

        Args:
            h: Starting state [batch, state_dim]
            t_start: Start time (hours since first review)
            t_end: End time
            n_steps: Number of integration steps
            sleep_schedule_fn: Function returning True if sleeping
            stress_level: Stress level [0, 1]

        Returns:
            h_end: State at t_end [batch, state_dim]
        """
        if t_end <= t_start:
            return h

        # Create time points for integration
        t_eval = torch.linspace(t_start, t_end, n_steps, device=h.device, dtype=h.dtype)

        # Set context for ODE function
        self.ode_func.set_context(sleep_schedule_fn, stress_level)

        # Integrate
        trajectory = odeint(self.ode_func, h, t_eval)

        # Return final state
        return trajectory[-1]

    def process_review_sequence(
        self,
        card_features: torch.Tensor,
        review_events: List[Tuple[float, int, Optional[torch.Tensor]]],
        user_features: Optional[torch.Tensor] = None,
        phenotype_id: Optional[torch.Tensor] = None,
        sleep_schedule_fn: Optional[callable] = None,
        stress_level: float = 0.0,
        return_trajectory: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a complete sequence of reviews for a user-card pair.

        This is the main entry point for training and inference.

        Args:
            card_features: Static card features [batch, card_feat_dim]
            review_events: List of (time, grade, telemetry) tuples
                - time: Hours since card creation
                - grade: Review grade 1-4 (int)
                - telemetry: [RT_norm, hesitation, tortuosity, fluency] or None
            user_features: User statistics [batch, user_feat_dim] or None
            phenotype_id: Learner phenotype ID [batch] for cold-start
            sleep_schedule_fn: Function(t) -> bool indicating sleep
            stress_level: Current stress level [0, 1]
            return_trajectory: Whether to return full trajectory

        Returns:
            Dictionary containing:
                - 'final_state': Final memory state [batch, state_dim]
                - 'final_prob': Final recall probability [batch, 1]
                - 'trajectory': List of (time, state, prob) if return_trajectory
        """
        device = card_features.device
        batch_size = card_features.size(0)

        # 1. Initialize state from features
        h = self.encoder(card_features, user_features, phenotype_id)
        initial_prob = torch.sigmoid(self.readout(h))

        trajectory = [(0.0, h.clone(), initial_prob.clone())] if return_trajectory else None

        # 2. Process each review event
        prev_time = 0.0

        for i, event in enumerate(review_events):
            t_review, grade, telemetry = event

            # Convert grade to tensor if needed
            if isinstance(grade, int):
                grade = torch.tensor([grade], device=device).expand(batch_size)

            # Evolve state continuously to review time
            if t_review > prev_time:
                h = self._integrate(
                    h, prev_time, t_review,
                    sleep_schedule_fn=sleep_schedule_fn,
                    stress_level=stress_level
                )

            # Apply discrete jump at review
            h = self.jump_net(h, grade, telemetry)

            # Record trajectory point
            if return_trajectory:
                prob = torch.sigmoid(self.readout(h))
                trajectory.append((t_review, h.clone(), prob.clone()))

            prev_time = t_review

        # 3. Compute final outputs
        final_prob = torch.sigmoid(self.readout(h))
        final_latency = self.latency_head(h)
        final_hesitation = self.hesitation_head(h)

        result = {
            'final_state': h,
            'final_prob': final_prob,
            'final_latency': final_latency,
            'final_hesitation': final_hesitation,
        }

        if return_trajectory:
            result['trajectory'] = trajectory

        return result

    def predict_next_review(
        self,
        h: torch.Tensor,
        target_retention: float = 0.9,
        max_interval: float = 365 * 24,  # 1 year in hours
        sleep_schedule_fn: Optional[callable] = None,
        stress_level: float = 0.0,
    ) -> torch.Tensor:
        """
        Predict optimal time for next review to maintain target retention.

        Uses binary search to find when retention drops to target.

        Args:
            h: Current memory state [batch, state_dim]
            target_retention: Desired retention probability
            max_interval: Maximum interval to search (hours)
            sleep_schedule_fn: Sleep schedule function
            stress_level: Current stress level

        Returns:
            interval: Optimal interval in hours [batch, 1]
        """
        batch_size = h.size(0)
        device = h.device

        # Binary search bounds
        low = torch.zeros(batch_size, device=device)
        high = torch.full((batch_size,), max_interval, device=device)

        # Binary search for each sample
        for _ in range(20):  # ~2^20 precision
            mid = (low + high) / 2

            # Predict retention at mid point
            probs = []
            for i in range(batch_size):
                h_i = h[i:i+1]
                t_mid = mid[i].item()
                h_future = self._integrate(
                    h_i, 0.0, t_mid,
                    sleep_schedule_fn=sleep_schedule_fn,
                    stress_level=stress_level
                )
                prob = torch.sigmoid(self.readout(h_future))
                probs.append(prob.squeeze())

            probs = torch.stack(probs)

            # Update bounds
            below_target = probs < target_retention
            high = torch.where(below_target, mid, high)
            low = torch.where(below_target, low, mid)

        return ((low + high) / 2).unsqueeze(-1)
