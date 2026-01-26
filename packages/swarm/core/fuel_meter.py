from enum import Enum
from typing import Optional
import logging

class FuelType(Enum):
    TOKEN = "TOKEN"
    STEP = "STEP"
    API_CALL = "API_CALL"

class FuelLimit:
    """Standardized fuel limits for different agent tasks."""
    LOW = 1000  # Quick checks
    MEDIUM = 5000  # Standard validations
    HIGH = 20000  # Deep research/complex architectures
    INFINITE = float('inf')

class FuelMeter:
    def __init__(self, limit: float = FuelLimit.MEDIUM, name: str = "Agent"):
        self.limit = limit
        self.current_usage = 0.0
        self.name = name
        self.logger = logging.getLogger(f"FuelMeter.{name}")

    def spend(self, amount: float, fuel_type: FuelType = FuelType.TOKEN):
        """
        Record usage of resources.
        Raises RuntimeError if fuel is depleted.
        """
        self.current_usage += amount
        self.logger.debug(f"{self.name} spent {amount} {fuel_type.value}. Total: {self.current_usage}/{self.limit}")
        
        if self.current_usage > self.limit:
            raise RuntimeError(f"FUEL DEPLETED: {self.name} exceeded limit of {self.limit} units.")

    def check_remaining(self) -> float:
        return self.limit - self.current_usage

    def is_empty(self) -> bool:
        return self.current_usage >= self.limit

    def reset(self):
        self.current_usage = 0.0
        self.logger.info(f"{self.name} fuel meter reset.")
