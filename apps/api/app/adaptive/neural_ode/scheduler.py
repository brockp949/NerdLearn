import torch
import numpy as np
from .model import NeuralMemoryODE

class NeuralODEScheduler:
    def __init__(self, model_path=None, state_dim=4):
        self.model = NeuralMemoryODE(state_dim=state_dim)
        if model_path:
            # map_location='cpu' ensures it loads even without CUDA
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
    def get_latent_state_dim(self):
        return self.model.ode_func.net[0].in_features

    def schedule_review(self, h0, current_time=0.0, retention_threshold=0.9, horizon_days=60, stress_level=0.0):
        """
        Predicts next review time for a single item or batch.
        
        Args:
            h0: Current latent state. Tensor shape (batch_size, state_dim)
            current_time: Current absolute time in hours (scalar)
            retention_threshold: Desired retention probability (e.g. 0.9)
            horizon_days: How far ahead to simulate (max interval)
            stress_level: Current user stress level (0.0 to 1.0)
            
        Returns:
            recommended_intervals: List of floats (hours until next review) for each item in batch
        """
        # Create evaluation grid (resolution: 1 hour)
        # Using coarser resolution for far future could optimize, but 1h is fine for inference
        num_hours = int(horizon_days * 24)
        hours = np.linspace(current_time, current_time + num_hours, num_hours + 1)
        t_eval = torch.tensor(hours, dtype=torch.float32)
        
        # Define default sleep schedule (11pm - 7am)
        # TODO: Inject user-specific schedule
        def default_sleep_schedule(t):
            hour_of_day = t % 24
            return hour_of_day >= 23 or hour_of_day < 7
            
        with torch.no_grad():
            # shape: (num_steps, batch_size, 1)
            probs = self.model.predict_retention(h0, t_eval, sleep_schedule_fn=default_sleep_schedule, stress_level=stress_level)
        
        probs = probs.squeeze(-1).permute(1, 0).numpy() # (batch_size, num_steps)
        batch_size = probs.shape[0]
        
        results = []
        for i in range(batch_size):
            # Find first index where prob < threshold
            # We skip index 0 (current time)
            p_curve = probs[i]
            below_thresh = np.where(p_curve < retention_threshold)[0]
            
            if len(below_thresh) > 0:
                # First point below threshold
                idx = below_thresh[0]
                if idx == 0: idx = 1 # Immediate review if already below
                target_time = hours[idx]
                interval = target_time - current_time
            else:
                # Never drops below threshold in horizon
                interval = horizon_days * 24
                
            results.append(float(interval))
            
        return results

    def simulate_trajectory(self, h0, hours=48, stress_level=0.0):
        """
        Helper to visualize or debug the memory curve.
        """
        t_eval = torch.linspace(0, hours, int(hours)+1)
        def default_sleep_schedule(t):
            hour_of_day = t % 24
            return hour_of_day >= 23 or hour_of_day < 7
            
        with torch.no_grad():
            probs = self.model.predict_retention(h0, t_eval, sleep_schedule_fn=default_sleep_schedule, stress_level=stress_level)
        
        return t_eval.numpy(), probs.squeeze().numpy()
