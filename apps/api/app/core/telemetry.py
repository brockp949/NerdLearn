from datetime import datetime
from typing import Dict, Any, Optional, List
from threading import Lock
import logging

logger = logging.getLogger(__name__)


DEFAULT_BUDGET_LIMIT_USD = 10.0

class BudgetExceededError(Exception):
    pass

class TelemetryService:

    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TelemetryService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self._metrics: Dict[str, Any] = {
            "llm_calls_total": 0,
            "llm_tokens_total": 0,
            "llm_cost_est_usd": 0.0,
            "agent_actions_total": 0,
            "errors_total": 0,
            "start_time": datetime.utcnow().isoformat()
        }
        self._action_log: List[Dict[str, Any]] = []
        self._max_log_size = 1000
        self._initialized = True
        logger.info("TelemetryService initialized")

    def track_llm_call(self, model: str, tokens: int, cost: float, success: bool = True):
        """Track an LLM API call."""
        with self._lock:
            # Check budget first
            if self._metrics["llm_cost_est_usd"] + cost > DEFAULT_BUDGET_LIMIT_USD:
                logger.error(f"Budget exceeded! Current: ${self._metrics['llm_cost_est_usd']:.4f}, Request: ${cost:.4f}, Limit: ${DEFAULT_BUDGET_LIMIT_USD}")
                raise BudgetExceededError("Daily LLM Budget Exceeded")

            self._metrics["llm_calls_total"] += 1
            self._metrics["llm_tokens_total"] += tokens
            self._metrics["llm_cost_est_usd"] += cost
            if not success:
                self._metrics["errors_total"] += 1
        
        logger.debug(f"LLM Call tracked: model={model}, tokens={tokens}, cost=${cost:.6f}")

    def log_agent_action(self, agent_name: str, action: str, metadata: Optional[Dict[str, Any]] = None):
        """Log a significant agent action."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": agent_name,
            "action": action,
            "metadata": metadata or {}
        }
        
        with self._lock:
            self._metrics["agent_actions_total"] += 1
            self._action_log.append(entry)
            if len(self._action_log) > self._max_log_size:
                self._action_log.pop(0)  # Keep rolling window
                
        logger.info(f"Agent Action: [{agent_name}] {action}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        return self._metrics.copy()

    def get_recent_actions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent agent actions."""
        return self._action_log[-limit:]

    def reset_metrics(self):
        """Reset metrics (useful for testing)."""
        with self._lock:
             self._metrics = {
                "llm_calls_total": 0,
                "llm_tokens_total": 0,
                "llm_cost_est_usd": 0.0,
                "agent_actions_total": 0,
                "errors_total": 0,
                "start_time": datetime.utcnow().isoformat()
            }
             self._action_log = []
