from typing import Any, Dict, List, Optional
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from app.core.telemetry import TelemetryService

# Estimated costs per 1k tokens (input/output blended or separated if possible)
# As of early 2025, these are rough estimates for tracking
MODEL_COSTS = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "default": {"input": 0.001, "output": 0.002} # Fallback
}

class CostTrackingCallback(BaseCallbackHandler):
    """Callback handler to track LLM costs and usage via TelemetryService."""

    def __init__(self):
        super().__init__()
        self.telemetry = TelemetryService()

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM ends running."""
        try:
            if not response.llm_output:
                return

            token_usage = response.llm_output.get("token_usage", {})
            if not token_usage:
                return

            model_name = response.llm_output.get("model_name", "unknown")
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            total_tokens = token_usage.get("total_tokens", 0)

            # Calculate cost
            pricing = MODEL_COSTS.get(model_name, MODEL_COSTS["default"])
            cost = (prompt_tokens / 1000 * pricing["input"]) + \
                   (completion_tokens / 1000 * pricing["output"])

            self.telemetry.track_llm_call(
                model=model_name, 
                tokens=total_tokens, 
                cost=cost,
                success=True
            )
            
        except Exception as e:
            # Don't fail the agent just because tracking failed
            print(f"Error in CostTrackingCallback: {e}")

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors."""
        self.telemetry.track_llm_call(model="unknown", tokens=0, cost=0, success=False)
