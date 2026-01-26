"""
Neural ODE Memory Model (CT-MCN).

Continuous-time memory calibration network for personalized forgetting curves.
Implements the architecture from "Neural ODEs for Personalized Memory Decay".

Components:
- NeuralMemoryODE: Main model combining encoder, drift, and jump networks
- MemoryDerivative: Continuous drift dynamics with circadian/sleep modulation
- JumpNetwork: Discrete state updates at review events
- StateEncoder: Maps card/user features to initial latent state
- ImplicitTelemetryLoss: Multi-task loss with TD-BKT telemetry
- NeuralODEScheduler: Scheduling service for production use
"""

from .model import NeuralMemoryODE, MemoryDerivative
from .jump import JumpNetwork, GatedJumpNetwork
from .encoder import StateEncoder, HierarchicalStateEncoder, PHENOTYPE_MAP
from .solver import odeint, rk4_step
from .loss import ImplicitTelemetryLoss
from .scheduler import NeuralODEScheduler

__all__ = [
    # Main model
    'NeuralMemoryODE',
    'MemoryDerivative',
    # Jump networks
    'JumpNetwork',
    'GatedJumpNetwork',
    # Encoders
    'StateEncoder',
    'HierarchicalStateEncoder',
    'PHENOTYPE_MAP',
    # ODE solver
    'odeint',
    'rk4_step',
    # Loss function
    'ImplicitTelemetryLoss',
    # Scheduler
    'NeuralODEScheduler',
]
