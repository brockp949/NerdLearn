"""
Fuel Meter - Token and Step Cost Controls

Implements cost control from "Building Code Testing Agents" PDF.

From PDF:
"Deploy the 'Fuel Meter': Implement strict token/step limits on testing
agents to prevent runaway costs while allowing for 'Deep Think' capabilities."

This prevents:
- Runaway API costs during testing
- Infinite loops in agent reasoning
- Resource exhaustion

While still allowing:
- Deep reasoning when needed
- Extended token windows for complex analysis
"""

from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class FuelBudget:
    """Budget allocation for an agent"""
    max_tokens: int = 100000         # Maximum tokens per session
    max_steps: int = 50              # Maximum reasoning steps
    max_duration_seconds: int = 300  # Maximum wall-clock time
    deep_think_reserve: float = 0.2  # Reserve 20% for deep analysis
    
    # Current consumption
    tokens_used: int = 0
    steps_taken: int = 0
    start_time: Optional[datetime] = None
    
    def remaining_tokens(self) -> int:
        """Get remaining token budget"""
        return max(0, self.max_tokens - self.tokens_used)
    
    def remaining_steps(self) -> int:
        """Get remaining step budget"""
        return max(0, self.max_steps - self.steps_taken)
    
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds()
    
    def is_exhausted(self) -> bool:
        """Check if any budget is exhausted"""
        return (
            self.tokens_used >= self.max_tokens or
            self.steps_taken >= self.max_steps or
            self.elapsed_seconds() >= self.max_duration_seconds
        )
    
    def deep_think_available(self) -> bool:
        """Check if deep think reserve is available"""
        reserve_tokens = int(self.max_tokens * self.deep_think_reserve)
        return self.remaining_tokens() >= reserve_tokens


@dataclass
class FuelUsageReport:
    """Report of fuel consumption"""
    agent_name: str
    tokens_used: int
    tokens_budget: int
    steps_taken: int
    steps_budget: int
    duration_seconds: float
    duration_budget: int
    exhaustion_reason: Optional[str] = None
    deep_think_triggered: bool = False
    estimated_cost_usd: float = 0.0
    
    def utilization_pct(self) -> float:
        """Overall budget utilization"""
        token_pct = self.tokens_used / max(1, self.tokens_budget)
        step_pct = self.steps_taken / max(1, self.steps_budget)
        time_pct = self.duration_seconds / max(1, self.duration_budget)
        return max(token_pct, step_pct, time_pct)


class FuelMeter:
    """
    Monitors and controls resource usage for testing agents.
    
    Features:
    - Token counting and limits
    - Step counting and limits
    - Time-based limits
    - Deep Think reserve for complex analysis
    - Cost estimation
    
    Usage:
        meter = FuelMeter(budget)
        with meter.track():
            result = await agent.run()
        report = meter.get_report()
    """
    
    # Approximate cost per 1K tokens (adjust based on model)
    COST_PER_1K_TOKENS = 0.01  # $0.01 per 1K tokens (example)
    
    def __init__(
        self,
        budget: Optional[FuelBudget] = None,
        agent_name: str = "unknown",
        on_exhaustion: Optional[Callable] = None
    ):
        """
        Initialize Fuel Meter.
        
        Args:
            budget: Resource budget (uses defaults if None)
            agent_name: Name of agent being monitored
            on_exhaustion: Callback when budget exhausted
        """
        self.budget = budget or FuelBudget()
        self.agent_name = agent_name
        self.on_exhaustion = on_exhaustion
        self._deep_think_mode = False
    
    def start(self):
        """Start tracking usage"""
        self.budget.start_time = datetime.now()
        logger.info(
            f"Fuel meter started for {self.agent_name}: "
            f"{self.budget.max_tokens} tokens, {self.budget.max_steps} steps"
        )
    
    def stop(self):
        """Stop tracking and generate report"""
        return self.get_report()
    
    def record_tokens(self, count: int):
        """Record token usage"""
        self.budget.tokens_used += count
        
        if self.budget.is_exhausted():
            self._handle_exhaustion("tokens")
    
    def record_step(self):
        """Record a reasoning step"""
        self.budget.steps_taken += 1
        
        if self.budget.is_exhausted():
            self._handle_exhaustion("steps")
    
    def enter_deep_think(self) -> bool:
        """
        Attempt to enter deep think mode.
        
        Uses the reserved budget for complex analysis.
        Returns True if deep think is available.
        """
        if self.budget.deep_think_available():
            self._deep_think_mode = True
            logger.info(f"{self.agent_name} entering Deep Think mode")
            return True
        
        logger.warning(f"{self.agent_name} cannot enter Deep Think - insufficient reserve")
        return False
    
    def exit_deep_think(self):
        """Exit deep think mode"""
        self._deep_think_mode = False
    
    def check_budget(self) -> bool:
        """
        Check if budget allows continuation.
        
        Returns True if agent can continue.
        """
        if self.budget.is_exhausted():
            return False
        
        # Also check time
        if self.budget.elapsed_seconds() >= self.budget.max_duration_seconds:
            self._handle_exhaustion("time")
            return False
        
        return True
    
    def _handle_exhaustion(self, reason: str):
        """Handle budget exhaustion"""
        logger.warning(f"{self.agent_name} fuel exhausted: {reason}")
        
        if self.on_exhaustion:
            self.on_exhaustion(reason, self.get_report())
    
    def get_report(self) -> FuelUsageReport:
        """Generate usage report"""
        tokens_used = self.budget.tokens_used
        estimated_cost = (tokens_used / 1000) * self.COST_PER_1K_TOKENS
        
        exhaustion_reason = None
        if self.budget.tokens_used >= self.budget.max_tokens:
            exhaustion_reason = "token_limit"
        elif self.budget.steps_taken >= self.budget.max_steps:
            exhaustion_reason = "step_limit"
        elif self.budget.elapsed_seconds() >= self.budget.max_duration_seconds:
            exhaustion_reason = "time_limit"
        
        return FuelUsageReport(
            agent_name=self.agent_name,
            tokens_used=self.budget.tokens_used,
            tokens_budget=self.budget.max_tokens,
            steps_taken=self.budget.steps_taken,
            steps_budget=self.budget.max_steps,
            duration_seconds=self.budget.elapsed_seconds(),
            duration_budget=self.budget.max_duration_seconds,
            exhaustion_reason=exhaustion_reason,
            deep_think_triggered=self._deep_think_mode,
            estimated_cost_usd=estimated_cost
        )
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
        return False


class FuelMeterWrapper:
    """
    Decorator/wrapper for adding fuel metering to agent methods.
    
    Usage:
        @FuelMeterWrapper(budget)
        async def my_agent_method(self, input):
            ...
    """
    
    def __init__(self, budget: Optional[FuelBudget] = None):
        self.budget = budget or FuelBudget()
    
    def __call__(self, func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            meter = FuelMeter(self.budget, agent_name=func.__name__)
            meter.start()
            
            try:
                # Inject meter into kwargs if function accepts it
                if 'fuel_meter' in func.__code__.co_varnames:
                    kwargs['fuel_meter'] = meter
                
                result = await func(*args, **kwargs)
                return result
                
            finally:
                report = meter.stop()
                logger.info(
                    f"{func.__name__} fuel usage: "
                    f"{report.tokens_used}/{report.tokens_budget} tokens, "
                    f"{report.steps_taken}/{report.steps_budget} steps, "
                    f"${report.estimated_cost_usd:.4f}"
                )
        
        return wrapper
