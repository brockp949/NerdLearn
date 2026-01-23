
import json
import random
import numpy as np
import os

def generate_synthetic_data(num_samples=1000, output_file="affect_data.json"):
    """
    Generate synthetic mouse movement data linked to affective states.
    
    Features:
    - velocity_mean: Average speed of mouse in pixels/ms
    - velocity_variance: Variability of speed (jitter)
    - click_rate: Clicks per minute
    - idle_time_ratio: Percentage of time spent idle
    - scroll_depth_rate: Rate of vertical scrolling
    
    Labels:
    0: Flow (Focused, productive)
    1: Frustrated (Rage clicks, erratic movement)
    2: Bored/Confused (Low activity, aimless wandering)
    """
    
    data = []
    
    for _ in range(num_samples):
        label = random.choice([0, 1, 2])
        
        if label == 0: # Flow
            # Consistent movement, moderate speed, low idle
            velocity_mean = np.random.normal(0.5, 0.1)
            velocity_variance = np.random.normal(0.05, 0.02)
            click_rate = np.random.normal(5, 2)
            idle_time_ratio = np.random.normal(0.1, 0.05)
            scroll_depth_rate = np.random.normal(0.3, 0.1)
            
        elif label == 1: # Frustrated
            # Fast, erratic movement, rage clicks
            velocity_mean = np.random.normal(0.8, 0.2)
            velocity_variance = np.random.normal(0.3, 0.1)
            click_rate = np.random.normal(20, 5) # Rage clicks
            idle_time_ratio = np.random.normal(0.05, 0.02)
            scroll_depth_rate = np.random.normal(0.5, 0.2) # Frantic scrolling
            
        else: # Bored
            # Slow, high idle
            velocity_mean = np.random.normal(0.1, 0.05)
            velocity_variance = np.random.normal(0.02, 0.01)
            click_rate = np.random.normal(1, 1)
            idle_time_ratio = np.random.normal(0.6, 0.15)
            scroll_depth_rate = np.random.normal(0.05, 0.05)
            
        # Clamp values
        features = {
            "velocity_mean": max(0.0, velocity_mean),
            "velocity_variance": max(0.0, velocity_variance),
            "click_rate": max(0.0, click_rate),
            "idle_time_ratio": min(1.0, max(0.0, idle_time_ratio)),
            "scroll_depth_rate": max(0.0, scroll_depth_rate),
            "label": label
        }
        data.append(features)
        
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Generated {num_samples} samples to {output_file}")
    return output_file

if __name__ == "__main__":
    generate_synthetic_data(output_file="apps/api/data/synthetic_affect_data.json")
