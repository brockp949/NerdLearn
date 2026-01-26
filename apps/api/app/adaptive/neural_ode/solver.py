import torch

def rk4_step(func, t, y, dt):
    """
    Performs a single step of the Runge-Kutta 4th order method.
    
    Args:
        func: The function computing the derivative dy/dt. Signature: func(t, y) -> dy/dt
        t: Current time (scalar or tensor)
        y: Current state tensor
        dt: Time step size
        
    Returns:
        New state tensor after time dt
    """
    k1 = func(t, y)
    k2 = func(t + dt / 2, y + dt * k1 / 2)
    k3 = func(t + dt / 2, y + dt * k2 / 2)
    k4 = func(t + dt, y + dt * k3)
    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

def odeint(func, y0, t):
    """
    Solves ODE using Runge-Kutta 4th order method.
    
    Args:
        func: The function computing the derivative dy/dt. 
              Signature: func(t, y) -> dy/dt
        y0: Initial state tensor. Shape: (batch_size, state_dim)
        t: Tensor of time points to evaluate at, including t0 as the first element. Shape: (num_steps,)
        
    Returns:
        Tensor of states at each time point. Shape: (num_steps, batch_size, state_dim)
    """
    states = [y0]
    current_state = y0
    
    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        current_t = t[i]
        current_state = rk4_step(func, current_t, current_state, dt)
        states.append(current_state)
        
    return torch.stack(states)
