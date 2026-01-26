
import torch
import matplotlib.pyplot as plt
import numpy as np
from apps.api.app.adaptive.neural_ode.scheduler import NeuralODEScheduler

def simulate():
    scheduler = NeuralODEScheduler(state_dim=4)
    h0 = torch.randn(1, 4)
    
    # Simulate 48 hours
    hours = 48
    stress_levels = [0.0, 0.5, 1.0]
    
    print(f"Simulating Memory Decay over {hours} hours...")
    
    for stress in stress_levels:
        t, probs = scheduler.simulate_trajectory(h0, hours=hours, stress_level=stress)
        print(f"\nStress Level: {stress}")
        print(f"Start Retention: {probs[0]:.4f}")
        print(f"End Retention:   {probs[-1]:.4f}")
        
        # Check sleep impact (hours 23-31 roughly corresponds to 11pm-7am in simulation time starting at 0:00)
        # Assuming t starts at 0 = midnight for simplicity in this check
        # But default sleep is t%24 >= 23 or t%24 < 7.
        # So t=0 is 00:00 (sleeping), t=7 wakes up.
        
        print(f"Retention at t=10 (Wake): {probs[10]:.4f}")
        print(f"Retention at t=30 (Wake): {probs[30]:.4f}")

if __name__ == "__main__":
    simulate()
